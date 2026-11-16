"""Build multi-resolution min/max pyramids for electrophysiology Zarr stores.

This module provides two entry points:

- :func:`build_mipmap` writes raw data and all pyramid levels in a single
  pass (finest level computed alongside raw, coarser levels cascaded).
- :func:`add_pyramid` reads an existing ``/raw`` array and writes
  pyramid-only output with a back-reference to the original store.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ecephys_mipmap_zarr.read import MipmapStore

import numpy as np
import zarr

from ecephys_mipmap_zarr._engine import (  # noqa: F401 — re-exported for tests and cli.py
    ProgressCallback,
    _cascade_minmax,
    _compute_minmax_block,
    _resolve_encoding,
    _StoredRawSource,
    _write_encoded,
)
from ecephys_mipmap_zarr._store import (
    _apply_adc_pyramid_quantization,
    _apply_adc_quantization,
    _copy_source_arrays,
    _extract_source_metadata,
    _extract_t_start,
    _find_raw_array,
    _open_store,
    _write_source_metadata_dict,
)
from ecephys_mipmap_zarr.metadata import (
    DEFAULT_BUILD_CONFIG,
    DEFAULT_ENCODING_CONFIG,
    BuildConfig,
    EncodingConfig,
    MipmapAttrs,
    compute_strides,
)

__all__ = [
    "ProgressCallback",
    "add_pyramid",
    "build_mipmap",
    "ensure_pyramid",
]

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


# NOTE: _compute_minmax_block, _cascade_minmax, ProgressCallback, and other
# engine symbols are re-exported from _engine via the import above for
# backward compatibility (tests and cli.py import them from build).


def build_mipmap(
    source: Any,
    output: str | Path,
    *,
    sampling_frequency: float,
    units: str,
    dtype: str | None = None,
    config: BuildConfig = DEFAULT_BUILD_CONFIG,
    encoding: EncodingConfig = DEFAULT_ENCODING_CONFIG,
    metadata_from: str | Path | None = None,
    source_metadata: dict[str, Any] | None = None,
    t_start_offset: float = 0.0,
    progress: ProgressCallback | None = None,
) -> None:
    """Build a complete mipmap Zarr store from an array-like source.

    Writes ``/raw`` and all ``/minmax/sN`` pyramid levels to *output*.
    The finest pyramid level is computed as a byproduct of writing raw
    chunks (single-pass optimisation).  Coarser levels cascade from the
    previous level.

    Parameters
    ----------
    source : array-like
        Input data with shape ``(n_samples, n_channels)`` and a ``dtype``
        attribute.  NumPy, Dask, and Zarr arrays all work.
    output : str or Path
        Filesystem path for the new Zarr v3 store.
    sampling_frequency : float
        Sampling rate in Hz.
    units : str
        Physical units of the data (e.g. ``"uV"``).
    dtype : str or None, optional
        Dtype of *source*'s values, by default ``source.dtype``. The stored
        dtype follows from *encoding*.
    config : BuildConfig, optional
        Pyramid geometry, chunking, compression and read settings.
    encoding : EncodingConfig, optional
        How raw and pyramids are stored; by default a noise-chosen shift for
        integer sources and a range fit for float ones.
    metadata_from : str, Path, or None, optional
        Path or URI to a zarr store from which to copy source metadata
        (e.g. SpikeInterface probe geometry, channel IDs).  Useful
        when building from ``.dat``/``.cbin`` files that lack metadata.
        Mutually exclusive with *source_metadata*.
    source_metadata : dict or None, optional
        In-memory metadata dict (e.g. from a SpikeInterface recording
        that slices channels).  Schema: ``channel_ids`` (required),
        and optionally ``location``, ``gain_to_uV``, ``offset_to_uV``,
        ``t_start``, ``probe``, ``annotations``.  Used when the source
        is derived from a processing chain whose channel set no longer
        matches the underlying zarr (e.g. after bad-channel removal).
        Mutually exclusive with *metadata_from*.
    t_start_offset : float, optional
        Seconds added to the derived ``t_start`` before it's written
        to root attrs.  Intended for windowed builds (e.g. when the
        caller is only mipmapping seconds 60-120 of a longer
        recording): the mipmap's absolute time axis should start at
        ``raw_t_start + t_start_offset``.  When no ``t_start`` is
        derived and *t_start_offset* > 0, the offset alone is written
        as ``t_start``.  Default 0.0.

    Raises
    ------
    ValueError
        If the recording is too short for any pyramid level, or *encoding*
        does not suit the source (see :class:`EncodingConfig`).
    """
    if metadata_from is not None and source_metadata is not None:
        msg = "metadata_from and source_metadata are mutually exclusive"
        raise ValueError(msg)

    source_dtype = str(np.dtype(dtype if dtype is not None else source.dtype))
    n_samples, n_channels = source.shape
    strides = compute_strides(
        n_samples, base=config.base_stride, factor=config.stride_factor, min_top_blocks=config.min_top_blocks
    )
    if not strides:
        msg = f"Recording too short ({n_samples} samples) for any pyramid level"
        raise ValueError(msg)

    plan = _resolve_encoding(encoding, source=source, source_dtype=source_dtype, sampling_frequency=sampling_frequency)
    root = zarr.open_group(str(output), mode="w", zarr_format=3)
    _write_encoded(
        source,
        root,
        plan=plan,
        encoding=encoding,
        writes_raw=True,
        source_dtype=source_dtype,
        strides=strides,
        cfg=config.chunk_config.for_channels(n_channels),
        stride_factor=config.stride_factor,
        compressor=config.compressor,
        progress=progress,
        prefetch_chunks=config.prefetch_chunks,
        source_retries=config.source_retries,
    )

    attrs_source_metadata: dict[str, Any] | None = None
    t_start: float | None = None
    if metadata_from is not None:
        meta_root = _open_store(metadata_from, mode="r")
        attrs_source_metadata = _extract_source_metadata(meta_root)
        t_start = _extract_t_start(meta_root)
        _copy_source_arrays(meta_root, root)

        _apply_adc_quantization(meta_root, root)
        _apply_adc_pyramid_quantization(meta_root, root, strides=strides)
    elif source_metadata is not None:
        _write_source_metadata_dict(root, source_metadata, strides=strides)
        t_start = source_metadata.get("t_start")
        passthrough = {k: source_metadata[k] for k in ("probe", "annotations", "si_chain") if k in source_metadata}
        attrs_source_metadata = passthrough if passthrough else None

    if t_start_offset:
        t_start = (t_start or 0.0) + float(t_start_offset)

    # Set root attributes
    attrs = MipmapAttrs(
        sampling_frequency=sampling_frequency,
        units=units,
        strides=strides,
        n_samples=n_samples,
        n_channels=n_channels,
        dtype=source_dtype,
        source_metadata=attrs_source_metadata,
        t_start=t_start,
    )
    root.attrs.update(attrs.to_dict())

    if config.consolidate_metadata:
        zarr.consolidate_metadata(zarr.storage.LocalStore(str(output)))


def add_pyramid(
    raw_store: str | Path,
    output: str | Path,
    *,
    sampling_frequency: float | None = None,
    units: str = "uV",
    raw_path: str = "raw",
    config: BuildConfig = DEFAULT_BUILD_CONFIG,
    encoding: EncodingConfig = DEFAULT_ENCODING_CONFIG,
    progress: ProgressCallback | None = None,
) -> None:
    """Build pyramid levels for an existing raw Zarr store.

    Reads the raw array from *raw_store*, builds all ``/minmax/sN`` levels,
    and writes them to *output*.  Does **not** copy raw data; instead sets
    ``raw_source`` in root attributes to point back to *raw_store*.

    Parameters
    ----------
    raw_store : str or Path
        Path to existing Zarr store containing the raw data array.
    output : str or Path
        Filesystem path for the pyramid-only output store.
    sampling_frequency : float or None, optional
        Sampling rate in Hz.  If ``None``, auto-detected from store
        attributes.
    units : str, optional
        Physical units of the data (e.g. ``"uV"``), by default ``"uV"``.
    raw_path : str, optional
        Path to the raw array within *raw_store*, by default ``"raw"``.
        Override for stores with non-standard layouts.
    config : BuildConfig, optional
        Pyramid geometry, chunking, compression and read settings.
    encoding : EncodingConfig, optional
        How the pyramids are stored; raw settings are an error, since no raw
        array is written.

    Raises
    ------
    ValueError
        If the recording is too short for any pyramid level, or *encoding*
        does not suit the source.
    """
    src_root = _open_store(raw_store, mode="r")
    raw_array = _find_raw_array(src_root, raw_path)
    source = _StoredRawSource(raw_array)
    n_samples, n_channels = source.shape
    raw_dtype = str(source.dtype)

    if sampling_frequency is None:
        sampling_frequency = float(dict(src_root.attrs).get("sampling_frequency", 0))  # type: ignore[arg-type]
        if sampling_frequency <= 0:
            msg = "sampling_frequency not provided and could not be inferred from store attrs"
            raise ValueError(msg)

    strides = compute_strides(
        n_samples, base=config.base_stride, factor=config.stride_factor, min_top_blocks=config.min_top_blocks
    )
    if not strides:
        msg = f"Recording too short ({n_samples} samples) for any pyramid level"
        raise ValueError(msg)

    plan = _resolve_encoding(
        encoding, source=source, source_dtype=raw_dtype, sampling_frequency=sampling_frequency, writes_raw=False
    )
    out_root = zarr.open_group(str(output), mode="w", zarr_format=3)
    _write_encoded(
        source,
        out_root,
        plan=plan,
        encoding=encoding,
        writes_raw=False,
        source_dtype=raw_dtype,
        strides=strides,
        cfg=config.chunk_config.for_channels(n_channels),
        stride_factor=config.stride_factor,
        compressor=config.compressor,
        progress=progress,
        prefetch_chunks=config.prefetch_chunks,
        source_retries=config.source_retries,
    )

    # Extract source metadata (probe, annotations) if available
    source_metadata = _extract_source_metadata(src_root)
    t_start = _extract_t_start(src_root)

    # Copy channel_ids and properties/location arrays if present
    _copy_source_arrays(src_root, out_root)

    # Set root attributes with raw_source
    attrs = MipmapAttrs(
        sampling_frequency=sampling_frequency,
        units=units,
        strides=strides,
        n_samples=n_samples,
        n_channels=n_channels,
        dtype=raw_dtype,
        raw_source=str(raw_store),
        raw_path=raw_array.path,
        source_metadata=source_metadata,
        t_start=t_start,
    )
    out_root.attrs.update(attrs.to_dict())

    if config.consolidate_metadata:
        zarr.consolidate_metadata(zarr.storage.LocalStore(str(output)))


def ensure_pyramid(
    raw_store: str | Path,
    output: str | Path,
    *,
    sampling_frequency: float | None = None,
    units: str = "uV",
    raw_path: str = "raw",
    config: BuildConfig = DEFAULT_BUILD_CONFIG,
    encoding: EncodingConfig = DEFAULT_ENCODING_CONFIG,
) -> MipmapStore:
    """Ensure a pyramid store exists for the given raw data, building if needed.

    Idempotent convenience wrapper around :func:`add_pyramid`.  If *output*
    already contains a consistent pyramid (matching *n_samples* and expected
    strides), it is reused.  Otherwise it is (re)built.

    Parameters
    ----------
    raw_store : str or Path
        Path to existing Zarr store containing the raw data array.
    output : str or Path
        Filesystem path for the pyramid-only output store.
    sampling_frequency : float or None, optional
        Sampling rate in Hz.  If ``None``, auto-detected from store
        attributes.
    units : str, optional
        Physical units of the data (e.g. ``"uV"``), by default ``"uV"``.
    raw_path : str, optional
        Path to the raw array within *raw_store*, by default ``"raw"``.
    config : BuildConfig, optional
        Pyramid geometry, chunking, compression and read settings.
    encoding : EncodingConfig, optional
        How the pyramids are stored, as for :func:`add_pyramid`.

    Returns
    -------
    MipmapStore
        A ready-to-use store handle for reading.

    Raises
    ------
    ValueError
        If the recording is too short for any pyramid level.
    """
    from ecephys_mipmap_zarr.read import open_mipmap

    # Unpack geometry params for stride validation
    base_stride = config.base_stride
    stride_factor = config.stride_factor
    min_top_blocks = config.min_top_blocks

    output_path = Path(output)
    needs_build = True

    if (output_path / "zarr.json").exists():
        # Output exists — check if it's still valid
        try:
            existing = zarr.open_group(str(output_path), mode="r", use_consolidated=None)
            existing_attrs = MipmapAttrs.from_dict(dict(existing.attrs))

            # Read raw store to get current n_samples
            src_root = _open_store(raw_store, mode="r")
            raw_arr = _find_raw_array(src_root, raw_path)
            cur_n_samples = raw_arr.shape[0]

            expected_strides = compute_strides(
                cur_n_samples, base=base_stride, factor=stride_factor, min_top_blocks=min_top_blocks
            )

            if existing_attrs.n_samples == cur_n_samples and existing_attrs.strides == expected_strides:
                needs_build = False
        except (KeyError, FileNotFoundError):
            pass  # Corrupted or incomplete — rebuild

    if needs_build:
        # Remove stale output if it exists
        if output_path.exists():
            shutil.rmtree(output_path)

        add_pyramid(
            raw_store,
            output,
            sampling_frequency=sampling_frequency,
            units=units,
            raw_path=raw_path,
            config=config,
            encoding=encoding,
        )

    return open_mipmap(output)
