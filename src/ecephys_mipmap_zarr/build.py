"""Build multi-resolution min/max pyramids for electrophysiology Zarr stores.

This module provides two entry points:

- :func:`build_mipmap` writes raw data and all pyramid levels in a single
  pass (finest level computed alongside raw, coarser levels cascaded).
- :func:`add_pyramid` reads an existing ``/raw`` array and writes
  pyramid-only output with a back-reference to the original store.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ecephys_mipmap_zarr.read import MipmapStore

import numpy as np
import zarr
from zarr.abc.codec import BytesBytesCodec

from ecephys_mipmap_zarr._engine import (  # noqa: F401 — re-exported for backward compat
    ProgressCallback,
    _AccumulatedStats,
    _build_cascade_levels,
    _build_finest_from_raw,
    _build_non_quantized,
    _cascade_minmax,
    _compute_minmax_block,
    _convert_array_to_int8,
    _create_zarr_array,
    _quantize_raw_in_place,
    _resolve_compressor,
    _resolve_lsb_shift,
    _write_raw_and_compute_finest,
)
from ecephys_mipmap_zarr._store import (
    _apply_adc_quantization,
    _copy_source_arrays,
    _extract_source_metadata,
    _find_raw_array,
    _open_store,
)
from ecephys_mipmap_zarr.metadata import (
    BASE_STRIDE,
    DEFAULT_CHUNK_CONFIG,
    DEFAULT_DTYPE,
    MIN_TOP_BLOCKS,
    STRIDE_FACTOR,
    ChunkConfig,
    LevelQuantization,
    MipmapAttrs,
    RawQuantization,
    compute_entropy,
    compute_quantization_params,
    compute_strides,
    minmax_path,
    n_blocks,
    quantize_array,
)

__all__ = [
    "ProgressCallback",
    "add_pyramid",
    "build_mipmap",
    "ensure_pyramid",
]


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
    dtype: str = DEFAULT_DTYPE,
    chunk_config: ChunkConfig = DEFAULT_CHUNK_CONFIG,
    base_stride: int = BASE_STRIDE,
    stride_factor: int = STRIDE_FACTOR,
    min_top_blocks: int = MIN_TOP_BLOCKS,
    compressor: str | BytesBytesCodec = "blosc",
    consolidate_metadata: bool = True,
    quantize: bool = False,
    quantize_raw: bool | None = None,
    quantize_pyramids: bool | None = None,
    raw_quantized_dtype: str = "uint16",
    pyramid_quantized_dtype: str = "uint8",
    raw_quantization: RawQuantization | None = None,
    level_quantization: LevelQuantization | None = None,
    metadata_from: str | Path | None = None,
    progress: ProgressCallback | None = None,
    lsb_shift_raw: int | str = "auto",
    lsb_shift_pyramid: int | str = "auto",
    noise_floor: float | None = None,
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
    dtype : str, optional
        Dtype for raw and pyramid arrays, by default ``"int16"``.
    chunk_config : ChunkConfig, optional
        Chunk/shard configuration, by default :data:`DEFAULT_CHUNK_CONFIG`.
    base_stride : int, optional
        Base stride for the finest pyramid level, by default 32.
    stride_factor : int, optional
        Multiplicative factor between levels, by default 8.
    min_top_blocks : int, optional
        Minimum blocks at the coarsest level, by default 1024.
    compressor : str or Codec, optional
        Compressor preset (``"blosc"``, ``"gzip"``, ``"zstd"``) or a
        zarr ``Codec`` instance, by default ``"blosc"``.
    consolidate_metadata : bool, optional
        Write consolidated metadata for faster store opens, by default
        ``True``.
    quantize : bool, optional
        Quantize raw to *raw_quantized_dtype* and pyramids to
        *pyramid_quantized_dtype* for better compression, by default
        ``False``.
    raw_quantized_dtype : str, optional
        Target dtype for quantized raw data, by default ``"uint16"``.
    pyramid_quantized_dtype : str, optional
        Target dtype for quantized pyramid data, by default ``"uint8"``.
    raw_quantization : RawQuantization or None, optional
        User-provided quantization params for raw data.  If ``None`` and
        *quantize* is ``True``, params are auto-computed from data
        statistics (requires two passes over the source).
    level_quantization : LevelQuantization or None, optional
        User-provided quantization params for pyramid levels (shared
        across all levels).  If ``None`` and *quantize* is ``True``,
        auto-computed from finest level statistics.
    metadata_from : str, Path, or None, optional
        Path or URI to a zarr store from which to copy source metadata
        (e.g. SpikeInterface probe geometry, channel IDs).  Useful
        when building from ``.dat``/``.cbin`` files that lack metadata.

    Raises
    ------
    ValueError
        If *source* has fewer than two dimensions, or the recording is
        too short for any pyramid level.
    """
    n_samples, n_channels = source.shape

    # Compute strides
    strides = compute_strides(n_samples, base=base_stride, factor=stride_factor, min_top_blocks=min_top_blocks)
    if not strides:
        msg = f"Recording too short ({n_samples} samples) for any pyramid level"
        raise ValueError(msg)

    # Adjust chunk config for actual channel count
    cfg = chunk_config.for_channels(n_channels)

    # Open output store
    root = zarr.open_group(str(output), mode="w", zarr_format=3)

    finest_stride = strides[0]
    finest_n_blocks = n_blocks(n_samples, finest_stride)

    # Resolve quantization flags with smart detection
    is_integer_source = np.issubdtype(np.dtype(dtype), np.integer)
    do_quantize_raw = quantize_raw if quantize_raw is not None else (quantize and not is_integer_source)
    do_quantize_pyramids = quantize_pyramids if quantize_pyramids is not None else quantize

    raw_shift = 0  # set in non-quantized path; stays 0 for quantized path

    if do_quantize_raw or do_quantize_pyramids:
        # --- Quantized path ---
        # Pass 1: write raw + collect s32 blocks + accumulate stats
        stats = _AccumulatedStats.create(n_channels, dtype)

        raw_array = _create_zarr_array(
            root,
            "raw",
            shape=(n_channels, n_samples),
            dtype=dtype,
            chunks=cfg.raw_chunk,
            shards=cfg.raw_shard,
            compressor=compressor,
        )

        s32_blocks = _write_raw_and_compute_finest(
            source,
            raw_array,
            None,
            shard_samples=cfg.raw_shard[1],
            base_stride=finest_stride,
            n_samples=n_samples,
            n_channels=n_channels,
            stats=stats,
            progress=progress,
        )

        all_s32 = np.concatenate(s32_blocks, axis=2)

        if do_quantize_pyramids:
            # Compute level quantization from s32 stats
            if level_quantization is None:
                min_offset, min_scale = compute_quantization_params(
                    stats.mm_min_min,
                    stats.mm_min_max,
                    pyramid_quantized_dtype,
                )
                max_offset, max_scale = compute_quantization_params(
                    stats.mm_max_min,
                    stats.mm_max_max,
                    pyramid_quantized_dtype,
                )
                combined_scale = np.maximum(min_scale, max_scale)
                level_quantization = LevelQuantization(
                    min_offset=min_offset.tolist(),
                    max_offset=max_offset.tolist(),
                    scale=combined_scale.tolist(),
                    source_dtype=dtype,
                    quantized_dtype=pyramid_quantized_dtype,
                )

            # Quantize s32 blocks and write
            lq_min_offset = np.array(level_quantization.min_offset)
            lq_max_offset = np.array(level_quantization.max_offset)
            lq_scale = np.array(level_quantization.scale)

            q_mins = quantize_array(all_s32[:, 0, :], lq_min_offset, lq_scale, pyramid_quantized_dtype)  # (C, B)
            q_maxs = quantize_array(all_s32[:, 1, :], lq_max_offset, lq_scale, pyramid_quantized_dtype)  # (C, B)
            q_s32 = np.stack([q_mins, q_maxs], axis=1)

            mm_dtype = pyramid_quantized_dtype
            finest_array = _create_zarr_array(
                root,
                minmax_path(finest_stride).lstrip("/"),
                shape=(n_channels, 2, finest_n_blocks),
                dtype=mm_dtype,
                chunks=cfg.minmax_chunk,
                shards=cfg.minmax_shard,
                compressor=compressor,
                quantized=True,
            )
            finest_array[:] = q_s32
            finest_array.attrs.update({"quantization": level_quantization.to_dict()})
        else:
            # Write s32 in source dtype (no pyramid quantization)
            mm_dtype = dtype
            finest_array = _create_zarr_array(
                root,
                minmax_path(finest_stride).lstrip("/"),
                shape=(n_channels, 2, finest_n_blocks),
                dtype=mm_dtype,
                chunks=cfg.minmax_chunk,
                shards=cfg.minmax_shard,
                compressor=compressor,
            )
            finest_array[:] = all_s32

        # Cascade
        _build_cascade_levels(
            root,
            finest_array,
            strides,
            n_samples=n_samples,
            n_channels=n_channels,
            dtype=mm_dtype,
            chunk_config=cfg,
            stride_factor=stride_factor,
            compressor=compressor,
            quantized=do_quantize_pyramids,
        )

        # Store quantization attrs on cascade levels
        if do_quantize_pyramids and level_quantization is not None:
            for stride in strides[1:]:
                path = minmax_path(stride).lstrip("/")
                level_arr: zarr.Array = root[path]  # type: ignore[assignment]
                level_arr.attrs.update({"quantization": level_quantization.to_dict()})

        # Quantize raw (pass 2) — skip if source is already integer
        if do_quantize_raw:
            if raw_quantization is None:
                rq_offset, rq_scale = compute_quantization_params(
                    stats.raw_min,
                    stats.raw_max,
                    raw_quantized_dtype,
                )
                raw_quantization = RawQuantization(
                    offset=rq_offset.tolist(),
                    scale=rq_scale.tolist(),
                    source_dtype=dtype,
                    quantized_dtype=raw_quantized_dtype,
                )
            _quantize_raw_in_place(
                root,
                raw_quantization,
                cfg=cfg,
                n_samples=n_samples,
                n_channels=n_channels,
                compressor=compressor,
            )
    else:
        # --- Non-quantized path with optional LSB shift ---
        raw_shift = _build_non_quantized(
            source=source,
            root=root,
            strides=strides,
            finest_stride=finest_stride,
            finest_n_blocks=finest_n_blocks,
            n_samples=n_samples,
            n_channels=n_channels,
            dtype=dtype,
            cfg=cfg,
            stride_factor=stride_factor,
            compressor=compressor,
            lsb_shift_raw=lsb_shift_raw,
            lsb_shift_pyramid=lsb_shift_pyramid,
            noise_floor=noise_floor,
            progress=progress,
        )

    # Copy source metadata if requested
    source_metadata: dict[str, Any] | None = None
    if metadata_from is not None:
        meta_root = _open_store(metadata_from, mode="r")
        source_metadata = _extract_source_metadata(meta_root)
        _copy_source_arrays(meta_root, root)

        # Store ADC gain/offset as RawQuantization if available
        # When LSB shift is active, combine shift into gain scale
        _apply_adc_quantization(meta_root, root, lsb_shift=raw_shift)

    # Set root attributes
    attrs = MipmapAttrs(
        sampling_frequency=sampling_frequency,
        units=units,
        strides=strides,
        n_samples=n_samples,
        n_channels=n_channels,
        dtype=dtype,
        source_metadata=source_metadata,
    )
    root.attrs.update(attrs.to_dict())

    if consolidate_metadata:
        zarr.consolidate_metadata(zarr.storage.LocalStore(str(output)))


def add_pyramid(
    raw_store: str | Path,
    output: str | Path,
    *,
    sampling_frequency: float | None = None,
    units: str = "uV",
    raw_path: str = "raw",
    chunk_config: ChunkConfig = DEFAULT_CHUNK_CONFIG,
    base_stride: int = BASE_STRIDE,
    stride_factor: int = STRIDE_FACTOR,
    min_top_blocks: int = MIN_TOP_BLOCKS,
    compressor: str | BytesBytesCodec = "blosc",
    consolidate_metadata: bool = True,
    quantize: bool = False,
    pyramid_quantized_dtype: str = "uint8",
    level_quantization: LevelQuantization | None = None,
    progress: ProgressCallback | None = None,
    lsb_shift_raw: int | str = "auto",
    lsb_shift_pyramid: int | str = "auto",
    noise_floor: float | None = None,
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
    sampling_frequency : float
        Sampling rate in Hz.
    units : str
        Physical units of the data (e.g. ``"uV"``).
    raw_path : str, optional
        Path to the raw array within *raw_store*, by default ``"raw"``.
        Override for stores with non-standard layouts.
    chunk_config : ChunkConfig, optional
        Chunk/shard configuration, by default :data:`DEFAULT_CHUNK_CONFIG`.
    base_stride : int, optional
        Base stride for the finest pyramid level, by default 32.
    stride_factor : int, optional
        Multiplicative factor between levels, by default 8.
    min_top_blocks : int, optional
        Minimum blocks at the coarsest level, by default 1024.
    compressor : str or Codec, optional
        Compressor preset (``"blosc"``, ``"gzip"``, ``"zstd"``) or a
        zarr ``Codec`` instance, by default ``"blosc"``.
    consolidate_metadata : bool, optional
        Write consolidated metadata for faster store opens, by default
        ``True``.
    quantize : bool, optional
        Quantize pyramid levels to *pyramid_quantized_dtype*, by default
        ``False``.
    pyramid_quantized_dtype : str, optional
        Target dtype for quantized pyramid data, by default ``"uint8"``.
    level_quantization : LevelQuantization or None, optional
        User-provided quantization params.  If ``None`` and *quantize*
        is ``True``, auto-computed from finest level statistics.

    Raises
    ------
    ValueError
        If the recording is too short for any pyramid level.
    """
    # Open source store and read raw array metadata
    src_root = _open_store(raw_store, mode="r")
    raw_array: zarr.Array = _find_raw_array(src_root, raw_path)
    # Detect (T, C) vs (C, T) layout
    if raw_array.shape[0] > raw_array.shape[1]:
        n_samples, n_channels = raw_array.shape  # SI convention (T, C)
    else:
        n_channels, n_samples = raw_array.shape  # our convention (C, T)
    raw_dtype = str(raw_array.dtype)

    # Auto-detect sampling rate from store attrs if not provided
    if sampling_frequency is None:
        src_attrs = dict(src_root.attrs)
        sampling_frequency = float(src_attrs.get("sampling_frequency", 0))  # type: ignore[arg-type]
        if sampling_frequency <= 0:
            msg = "sampling_frequency not provided and could not be inferred from store attrs"
            raise ValueError(msg)

    # Compute strides
    strides = compute_strides(n_samples, base=base_stride, factor=stride_factor, min_top_blocks=min_top_blocks)
    if not strides:
        msg = f"Recording too short ({n_samples} samples) for any pyramid level"
        raise ValueError(msg)

    # Adjust chunk config
    cfg = chunk_config.for_channels(n_channels)

    # Open output store (pyramid-only, no /raw)
    out_root = zarr.open_group(str(output), mode="w", zarr_format=3)

    finest_stride = strides[0]
    finest_n_blocks = n_blocks(n_samples, finest_stride)

    if quantize:
        # Collect s32 blocks in memory + accumulate stats
        stats = _AccumulatedStats.create(n_channels, raw_dtype)

        s32_blocks = _build_finest_from_raw(
            raw_array,
            None,
            shard_samples=cfg.raw_shard[1],
            base_stride=finest_stride,
            n_samples=n_samples,
            stats=stats,
            progress=progress,
        )

        # Compute level quantization
        if level_quantization is None:
            min_offset, min_scale = compute_quantization_params(
                stats.mm_min_min,
                stats.mm_min_max,
                pyramid_quantized_dtype,
            )
            max_offset, max_scale = compute_quantization_params(
                stats.mm_max_min,
                stats.mm_max_max,
                pyramid_quantized_dtype,
            )
            combined_scale = np.maximum(min_scale, max_scale)
            level_quantization = LevelQuantization(
                min_offset=min_offset.tolist(),
                max_offset=max_offset.tolist(),
                scale=combined_scale.tolist(),
                source_dtype=raw_dtype,
                quantized_dtype=pyramid_quantized_dtype,
            )

        # Quantize and write s32
        lq_min_offset = np.array(level_quantization.min_offset)
        lq_max_offset = np.array(level_quantization.max_offset)
        lq_scale = np.array(level_quantization.scale)

        all_s32 = np.concatenate(s32_blocks, axis=2)
        q_mins = quantize_array(all_s32[:, 0, :], lq_min_offset, lq_scale, pyramid_quantized_dtype)
        q_maxs = quantize_array(all_s32[:, 1, :], lq_max_offset, lq_scale, pyramid_quantized_dtype)
        q_s32 = np.stack([q_mins, q_maxs], axis=1)

        finest_array = _create_zarr_array(
            out_root,
            minmax_path(finest_stride).lstrip("/"),
            shape=(n_channels, 2, finest_n_blocks),
            dtype=pyramid_quantized_dtype,
            chunks=cfg.minmax_chunk,
            shards=cfg.minmax_shard,
            compressor=compressor,
            quantized=True,
        )
        finest_array[:] = q_s32
        finest_array.attrs.update({"quantization": level_quantization.to_dict()})

        # Cascade on uint8
        _build_cascade_levels(
            out_root,
            finest_array,
            strides,
            n_samples=n_samples,
            n_channels=n_channels,
            dtype=pyramid_quantized_dtype,
            chunk_config=cfg,
            stride_factor=stride_factor,
            compressor=compressor,
            quantized=True,
        )

        # Store quantization attrs on cascade levels
        for stride in strides[1:]:
            path = minmax_path(stride).lstrip("/")
            level_arr: zarr.Array = out_root[path]  # type: ignore[assignment]
            level_arr.attrs.update({"quantization": level_quantization.to_dict()})
    else:
        # Non-quantized path with optional LSB shift
        pyr_shift = _resolve_lsb_shift(
            lsb_shift_pyramid,
            raw_array,
            n_samples,
            noise_floor,
            extra_shift=2,
        )

        # Compute entropy for shuffle auto-selection
        pyr_entropy: float | None = None
        if pyr_shift > 0:
            n_sample_ent = min(30_000, n_samples)
            is_tc = raw_array.shape[0] > raw_array.shape[1]
            if is_tc:
                sample_ct = np.ascontiguousarray(np.asarray(raw_array[0:n_sample_ent, :]).T)
            else:
                sample_ct = np.asarray(raw_array[:, 0:n_sample_ent])
            mm_sample = _compute_minmax_block(sample_ct, finest_stride)
            pyr_entropy = compute_entropy(mm_sample >> pyr_shift)

        if pyr_shift > 0:
            # Collect blocks, shift, then write
            s32_blocks = _build_finest_from_raw(
                raw_array,
                None,
                shard_samples=cfg.raw_shard[1],
                base_stride=finest_stride,
                n_samples=n_samples,
                progress=progress,
            )
            all_s32 = np.concatenate(s32_blocks, axis=2)
            shifted_s32 = (all_s32 >> pyr_shift).astype(np.int16)
            pyr_s_min = int(shifted_s32.min())
            pyr_s_max = int(shifted_s32.max())

            finest_array = _create_zarr_array(
                out_root,
                minmax_path(finest_stride).lstrip("/"),
                shape=(n_channels, 2, finest_n_blocks),
                dtype=raw_dtype,
                chunks=cfg.minmax_chunk,
                shards=cfg.minmax_shard,
                compressor=compressor,
                entropy=pyr_entropy,
                is_pyramid=True,
            )
            finest_array[:] = shifted_s32

            _build_cascade_levels(
                out_root,
                finest_array,
                strides,
                n_samples=n_samples,
                n_channels=n_channels,
                dtype=raw_dtype,
                chunk_config=cfg,
                stride_factor=stride_factor,
                compressor=compressor,
                entropy=pyr_entropy,
            )

            # Store lsb_shift attrs
            for s in strides:
                pyr_arr: zarr.Array = out_root[minmax_path(s).lstrip("/")]  # type: ignore[assignment]
                pyr_arr.attrs.update({"lsb_shift": pyr_shift, "original_dtype": raw_dtype})

            # Convert to int8 if range fits
            if pyr_s_min >= -128 and pyr_s_max <= 127:
                for s in strides:
                    _convert_array_to_int8(
                        out_root,
                        minmax_path(s).lstrip("/"),
                        cfg.minmax_chunk,
                        cfg.minmax_shard,
                        compressor,
                        entropy=pyr_entropy,
                        is_pyramid=True,
                    )
        else:
            # No shift — direct write path
            finest_array = _create_zarr_array(
                out_root,
                minmax_path(finest_stride).lstrip("/"),
                shape=(n_channels, 2, finest_n_blocks),
                dtype=raw_dtype,
                chunks=cfg.minmax_chunk,
                shards=cfg.minmax_shard,
                compressor=compressor,
                is_pyramid=True,
            )

            _build_finest_from_raw(
                raw_array,
                finest_array,
                shard_samples=cfg.raw_shard[1],
                base_stride=finest_stride,
                n_samples=n_samples,
                progress=progress,
            )

            _build_cascade_levels(
                out_root,
                finest_array,
                strides,
                n_samples=n_samples,
                n_channels=n_channels,
                dtype=raw_dtype,
                chunk_config=cfg,
                stride_factor=stride_factor,
                compressor=compressor,
            )

    # Extract source metadata (probe, annotations) if available
    source_metadata = _extract_source_metadata(src_root)

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
        source_metadata=source_metadata,
    )
    out_root.attrs.update(attrs.to_dict())

    if consolidate_metadata:
        zarr.consolidate_metadata(zarr.storage.LocalStore(str(output)))


def ensure_pyramid(
    raw_store: str | Path,
    output: str | Path,
    *,
    sampling_frequency: float | None = None,
    units: str = "uV",
    raw_path: str = "raw",
    chunk_config: ChunkConfig = DEFAULT_CHUNK_CONFIG,
    base_stride: int = BASE_STRIDE,
    stride_factor: int = STRIDE_FACTOR,
    min_top_blocks: int = MIN_TOP_BLOCKS,
    compressor: str | BytesBytesCodec = "blosc",
    consolidate_metadata: bool = True,
    quantize: bool = False,
    pyramid_quantized_dtype: str = "uint8",
    level_quantization: LevelQuantization | None = None,
    lsb_shift_raw: int | str = "auto",
    lsb_shift_pyramid: int | str = "auto",
    noise_floor: float | None = None,
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
    sampling_frequency : float
        Sampling rate in Hz.
    units : str
        Physical units of the data (e.g. ``"uV"``).
    raw_path : str, optional
        Path to the raw array within *raw_store*, by default ``"raw"``.
    chunk_config : ChunkConfig, optional
        Chunk/shard configuration, by default :data:`DEFAULT_CHUNK_CONFIG`.
    base_stride : int, optional
        Base stride for the finest pyramid level, by default 32.
    stride_factor : int, optional
        Multiplicative factor between levels, by default 8.
    min_top_blocks : int, optional
        Minimum blocks at the coarsest level, by default 1024.
    compressor : str or Codec, optional
        Compressor preset (``"blosc"``, ``"gzip"``, ``"zstd"``) or a
        zarr ``Codec`` instance, by default ``"blosc"``.
    consolidate_metadata : bool, optional
        Write consolidated metadata for faster store opens, by default
        ``True``.
    quantize : bool, optional
        Quantize pyramid levels, by default ``False``.
    pyramid_quantized_dtype : str, optional
        Target dtype for quantized pyramid data, by default ``"uint8"``.
    level_quantization : LevelQuantization or None, optional
        User-provided quantization params for pyramid levels.

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
            chunk_config=chunk_config,
            base_stride=base_stride,
            stride_factor=stride_factor,
            min_top_blocks=min_top_blocks,
            compressor=compressor,
            consolidate_metadata=consolidate_metadata,
            quantize=quantize,
            pyramid_quantized_dtype=pyramid_quantized_dtype,
            level_quantization=level_quantization,
            lsb_shift_raw=lsb_shift_raw,
            lsb_shift_pyramid=lsb_shift_pyramid,
            noise_floor=noise_floor,
        )

    return open_mipmap(output)
