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
    _ShiftStats,
    _write_raw_and_compute_finest,
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
    DEFAULT_DTYPE,
    DEFAULT_QUANTIZATION_CONFIG,
    BuildConfig,
    LevelQuantization,
    MipmapAttrs,
    QuantizationConfig,
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
    dtype: str = DEFAULT_DTYPE,
    config: BuildConfig = DEFAULT_BUILD_CONFIG,
    quantization: QuantizationConfig = DEFAULT_QUANTIZATION_CONFIG,
    metadata_from: str | Path | None = None,
    source_metadata: dict[str, Any] | None = None,
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
    dtype : str, optional
        Dtype for raw and pyramid arrays, by default ``"int16"``.
    config : BuildConfig, optional
        Build configuration for pyramid geometry, chunking, compression,
        and LSB-shift settings, by default :data:`DEFAULT_BUILD_CONFIG`.
    quantization : QuantizationConfig, optional
        Quantization configuration for raw and pyramid data,
        by default :data:`DEFAULT_QUANTIZATION_CONFIG`.
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

    Raises
    ------
    ValueError
        If *source* has fewer than two dimensions, or the recording is
        too short for any pyramid level.
    """
    # Unpack config
    chunk_config = config.chunk_config
    base_stride = config.base_stride
    stride_factor = config.stride_factor
    min_top_blocks = config.min_top_blocks
    compressor = config.compressor
    consolidate_metadata = config.consolidate_metadata
    lsb_shift_raw = config.lsb_shift_raw
    lsb_shift_pyramid = config.lsb_shift_pyramid
    noise_floor = config.noise_floor

    # Unpack quantization
    quantize = quantization.quantize
    quantize_raw = quantization.quantize_raw
    quantize_pyramids = quantization.quantize_pyramids
    raw_quantized_dtype = quantization.raw_quantized_dtype
    pyramid_quantized_dtype = quantization.pyramid_quantized_dtype
    raw_quantization = quantization.raw_quantization
    level_quantization = quantization.level_quantization

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
    pyr_shift = 0

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

        # Write s32 directly to zarr during the pass (no in-memory collection)
        finest_array = _create_zarr_array(
            root,
            minmax_path(finest_stride).lstrip("/"),
            shape=(n_channels, 2, finest_n_blocks),
            dtype=dtype,
            chunks=cfg.minmax_chunk,
            shards=cfg.minmax_shard,
            compressor=compressor,
        )

        _write_raw_and_compute_finest(
            source,
            raw_array,
            finest_array,
            shard_samples=cfg.raw_shard[1],
            base_stride=finest_stride,
            n_samples=n_samples,
            n_channels=n_channels,
            stats=stats,
            progress=progress,
            prefetch_chunks=config.prefetch_chunks,
        )

        if do_quantize_pyramids:
            # Compute level quantization from accumulated stats
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

            # Requantize s32 in-place: read chunks from zarr, quantize, write back
            lq_min_offset = np.array(level_quantization.min_offset)
            lq_max_offset = np.array(level_quantization.max_offset)
            lq_scale = np.array(level_quantization.scale)

            # Replace finest array with quantized dtype
            q_finest = _create_zarr_array(
                root,
                minmax_path(finest_stride).lstrip("/") + "__q_tmp",
                shape=(n_channels, 2, finest_n_blocks),
                dtype=pyramid_quantized_dtype,
                chunks=cfg.minmax_chunk,
                shards=cfg.minmax_shard,
                compressor=compressor,
                quantized=True,
            )

            shard_blocks = cfg.minmax_shard[2]
            for b0 in range(0, finest_n_blocks, shard_blocks):
                b1 = min(b0 + shard_blocks, finest_n_blocks)
                s32_chunk = np.asarray(finest_array[:, :, b0:b1])
                q_mins = quantize_array(s32_chunk[:, 0, :], lq_min_offset, lq_scale, pyramid_quantized_dtype)
                q_maxs = quantize_array(s32_chunk[:, 1, :], lq_max_offset, lq_scale, pyramid_quantized_dtype)
                q_finest[:, :, b0:b1] = np.stack([q_mins, q_maxs], axis=1)

            # Swap tmp → final
            mm_path = minmax_path(finest_stride).lstrip("/")
            del root[mm_path]
            store = root.store
            if hasattr(store, "root"):
                import shutil

                shutil.move(str(store.root / (mm_path + "__q_tmp")), str(store.root / mm_path))
                finest_array = root[mm_path]  # type: ignore[assignment]
            else:
                finest_array = q_finest

            mm_dtype = pyramid_quantized_dtype
            finest_array.attrs.update({"quantization": level_quantization.to_dict()})
        else:
            mm_dtype = dtype

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
        raw_shift, pyr_shift = _build_non_quantized(
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
            clip_outliers=config.clip_outliers,
            prefetch_chunks=config.prefetch_chunks,
        )

    # Copy source metadata if requested
    if metadata_from is not None and source_metadata is not None:
        msg = "metadata_from and source_metadata are mutually exclusive"
        raise ValueError(msg)

    attrs_source_metadata: dict[str, Any] | None = None
    t_start: float | None = None
    if metadata_from is not None:
        meta_root = _open_store(metadata_from, mode="r")
        attrs_source_metadata = _extract_source_metadata(meta_root)
        t_start = _extract_t_start(meta_root)
        _copy_source_arrays(meta_root, root)

        # Store ADC gain/offset as quantization metadata if available.
        # When LSB shift is active, combine shift into gain scale.
        _apply_adc_quantization(meta_root, root, lsb_shift=raw_shift)
        _apply_adc_pyramid_quantization(meta_root, root, strides=strides, lsb_shift=pyr_shift)
    elif source_metadata is not None:
        _write_source_metadata_dict(
            root,
            source_metadata,
            strides=strides,
            raw_shift=raw_shift,
            pyr_shift=pyr_shift,
        )
        t_start = source_metadata.get("t_start")
        passthrough = {k: source_metadata[k] for k in ("probe", "annotations") if k in source_metadata}
        attrs_source_metadata = passthrough if passthrough else None

    # Set root attributes
    attrs = MipmapAttrs(
        sampling_frequency=sampling_frequency,
        units=units,
        strides=strides,
        n_samples=n_samples,
        n_channels=n_channels,
        dtype=dtype,
        source_metadata=attrs_source_metadata,
        t_start=t_start,
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
    config: BuildConfig = DEFAULT_BUILD_CONFIG,
    quantization: QuantizationConfig = DEFAULT_QUANTIZATION_CONFIG,
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
        Build configuration for pyramid geometry, chunking, compression,
        and LSB-shift settings, by default :data:`DEFAULT_BUILD_CONFIG`.
    quantization : QuantizationConfig, optional
        Quantization configuration for pyramid data,
        by default :data:`DEFAULT_QUANTIZATION_CONFIG`.

    Raises
    ------
    ValueError
        If the recording is too short for any pyramid level.
    """
    # Unpack config
    chunk_config = config.chunk_config
    base_stride = config.base_stride
    stride_factor = config.stride_factor
    min_top_blocks = config.min_top_blocks
    compressor = config.compressor
    consolidate_metadata = config.consolidate_metadata
    lsb_shift_pyramid = config.lsb_shift_pyramid
    noise_floor = config.noise_floor

    # Unpack quantization
    quantize = quantization.quantize
    pyramid_quantized_dtype = quantization.pyramid_quantized_dtype
    level_quantization = quantization.level_quantization

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
        # Write s32 directly to zarr during the pass + accumulate stats
        stats = _AccumulatedStats.create(n_channels, raw_dtype)

        finest_array = _create_zarr_array(
            out_root,
            minmax_path(finest_stride).lstrip("/"),
            shape=(n_channels, 2, finest_n_blocks),
            dtype=raw_dtype,
            chunks=cfg.minmax_chunk,
            shards=cfg.minmax_shard,
            compressor=compressor,
        )

        _build_finest_from_raw(
            raw_array,
            finest_array,
            shard_samples=cfg.raw_shard[1],
            base_stride=finest_stride,
            n_samples=n_samples,
            stats=stats,
            progress=progress,
            prefetch_chunks=config.prefetch_chunks,
        )

        # Compute level quantization from accumulated stats
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

        # Requantize s32 in-place: read chunks from zarr, quantize, write to new array
        lq_min_offset = np.array(level_quantization.min_offset)
        lq_max_offset = np.array(level_quantization.max_offset)
        lq_scale = np.array(level_quantization.scale)

        mm_path = minmax_path(finest_stride).lstrip("/")
        q_finest = _create_zarr_array(
            out_root,
            mm_path + "__q_tmp",
            shape=(n_channels, 2, finest_n_blocks),
            dtype=pyramid_quantized_dtype,
            chunks=cfg.minmax_chunk,
            shards=cfg.minmax_shard,
            compressor=compressor,
            quantized=True,
        )

        shard_blocks = cfg.minmax_shard[2]
        for b0 in range(0, finest_n_blocks, shard_blocks):
            b1 = min(b0 + shard_blocks, finest_n_blocks)
            s32_chunk = np.asarray(finest_array[:, :, b0:b1])
            q_mins = quantize_array(s32_chunk[:, 0, :], lq_min_offset, lq_scale, pyramid_quantized_dtype)
            q_maxs = quantize_array(s32_chunk[:, 1, :], lq_max_offset, lq_scale, pyramid_quantized_dtype)
            q_finest[:, :, b0:b1] = np.stack([q_mins, q_maxs], axis=1)

        # Swap tmp → final
        del out_root[mm_path]
        store = out_root.store
        if hasattr(store, "root"):
            import shutil

            shutil.move(str(store.root / (mm_path + "__q_tmp")), str(store.root / mm_path))
            finest_array = out_root[mm_path]  # type: ignore[assignment]
        else:
            finest_array = q_finest
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
            pyr_entropy = compute_entropy((mm_sample + (1 << (pyr_shift - 1))) >> pyr_shift)

        if pyr_shift > 0:
            # Collect blocks, shift, then write
            s32_blocks = _build_finest_from_raw(
                raw_array,
                None,
                shard_samples=cfg.raw_shard[1],
                base_stride=finest_stride,
                n_samples=n_samples,
                progress=progress,
                prefetch_chunks=config.prefetch_chunks,
            )
            all_s32 = np.concatenate(s32_blocks, axis=2)
            shifted_s32 = ((all_s32 + (1 << (pyr_shift - 1))) >> pyr_shift).astype(np.int16)
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

            # Convert to int8 if range fits, or clip rare outliers
            clip_outliers = config.clip_outliers
            CLIP_THRESHOLD = 0.0001  # 0.01%
            fits_int8 = pyr_s_min >= -128 and pyr_s_max <= 127
            pyr_outliers = int(((shifted_s32 < -128) | (shifted_s32 > 127)).sum())
            pyr_total = shifted_s32.size

            if fits_int8:
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
            elif clip_outliers and pyr_total > 0 and pyr_outliers / pyr_total < CLIP_THRESHOLD:
                _log.info(
                    "Clipping %d pyramid outlier samples (%.6f%%) to fit int8",
                    pyr_outliers,
                    100 * pyr_outliers / pyr_total,
                )
                for s in strides:
                    _convert_array_to_int8(
                        out_root,
                        minmax_path(s).lstrip("/"),
                        cfg.minmax_chunk,
                        cfg.minmax_shard,
                        compressor,
                        entropy=pyr_entropy,
                        is_pyramid=True,
                        clip=True,
                    )
            else:
                _log.info(
                    "Keeping int16 for pyramids: %d samples (%.4f%%) exceed int8 range",
                    pyr_outliers,
                    100 * pyr_outliers / max(pyr_total, 1),
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
                prefetch_chunks=config.prefetch_chunks,
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
        source_metadata=source_metadata,
        t_start=t_start,
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
    config: BuildConfig = DEFAULT_BUILD_CONFIG,
    quantization: QuantizationConfig = DEFAULT_QUANTIZATION_CONFIG,
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
        Build configuration for pyramid geometry, chunking, compression,
        and LSB-shift settings, by default :data:`DEFAULT_BUILD_CONFIG`.
    quantization : QuantizationConfig, optional
        Quantization configuration for pyramid data,
        by default :data:`DEFAULT_QUANTIZATION_CONFIG`.

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
            quantization=quantization,
        )

    return open_mipmap(output)
