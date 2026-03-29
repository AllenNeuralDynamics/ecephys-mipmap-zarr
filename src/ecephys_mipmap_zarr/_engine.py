"""Write engine for building mipmap zarr stores.

Internal module — contains the chunked I/O, min/max computation,
cascading, quantization, and LSB-shift logic used by the public
build API.  Not part of the public API.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import zarr
from zarr.abc.codec import BytesBytesCodec
from zarr.codecs import BloscCodec, GzipCodec, ZstdCodec
from zarr.codecs.blosc import BloscCname, BloscShuffle

from ecephys_mipmap_zarr.metadata import (
    ChunkConfig,
    RawQuantization,
    compute_entropy,
    compute_optimal_lsb_shift,
    minmax_path,
    n_blocks,
    quantize_array,
)

__all__: list[str] = []  # private module


def _resolve_lsb_shift(
    shift: int | str,
    source: Any,
    n_samples: int,
    noise_floor: float | None,
    extra_shift: int = 0,
) -> int:
    """Resolve an LSB shift parameter to a concrete integer.

    Parameters
    ----------
    shift : int or str
        ``0`` for lossless, a positive ``int`` for manual shift, or
        ``"auto"`` for noise-based auto-detection.
    source : array-like
        Source data (T, C) for noise estimation.
    n_samples : int
        Total samples in the source.
    noise_floor : float or None
        Explicit noise floor override.
    extra_shift : int, optional
        Additional shift for display-quality pyramids.

    Returns
    -------
    int
        Resolved shift value (0 = no shift).
    """
    if isinstance(shift, int):
        return shift
    if shift != "auto":
        msg = f"lsb_shift must be int, 'auto', or 0. Got {shift!r}"
        raise ValueError(msg)

    # Auto-detect: read first 30K samples
    n_sample = min(30_000, n_samples)
    sample = np.ascontiguousarray(np.asarray(source[0:n_sample, :]).T)  # (C, T)
    auto_shift = compute_optimal_lsb_shift(sample, noise_floor=noise_floor)
    return auto_shift + extra_shift


def _convert_array_to_int8(
    root: zarr.Group,
    path: str,
    chunks: tuple[int, ...],
    shards: tuple[int, ...],
    compressor: str | BytesBytesCodec,
    entropy: float | None = None,
    is_pyramid: bool = False,
) -> None:
    """Convert an int16 zarr array to int8 in-place.

    Reads all data, deletes the int16 array, creates int8 replacement,
    writes the data back.

    Parameters
    ----------
    root : zarr.Group
        Store root group.
    path : str
        Array path within root.
    chunks : tuple
        Chunk shape for the new array.
    shards : tuple
        Shard shape for the new array.
    compressor : str or Codec
        Compressor for the new array.
    entropy : float or None, optional
        Shannon entropy for shuffle auto-selection.
    is_pyramid : bool, optional
        Whether this is a pyramid array.
    """
    old_arr: zarr.Array = root[path]  # type: ignore[assignment]
    data = np.asarray(old_arr[:]).astype(np.int8)
    shape = old_arr.shape
    old_attrs = dict(old_arr.attrs)

    del root[path]
    new_arr = _create_zarr_array(
        root,
        path,
        shape=shape,
        dtype="int8",
        chunks=chunks,
        shards=shards,
        compressor=compressor,
        entropy=entropy,
        is_pyramid=is_pyramid,
    )
    new_arr[:] = data
    if old_attrs:
        new_arr.attrs.update(old_attrs)


def _resolve_compressor(
    compressor: str | BytesBytesCodec,
    quantized: bool = False,
    entropy: float | None = None,
    is_pyramid: bool = False,
    stored_dtype: str = "int16",
) -> BytesBytesCodec:
    """Resolve a compressor string preset or pass through a Codec object.

    Parameters
    ----------
    compressor : str or Codec
        A preset name (``"blosc"``, ``"gzip"``, ``"zstd"``) or a zarr
        ``Codec`` instance.
    quantized : bool, optional
        Whether the target array stores quantized unsigned int data.
    entropy : float or None, optional
        Shannon entropy of the data (bits/sample).  When provided and
        compressor is ``"blosc"``, selects the shuffle mode:
        < 6 → noshuffle, ≥ 6 → bitshuffle.
    is_pyramid : bool, optional
        Whether this is a pyramid array with ``(C, 2, B)`` layout.
        int16 pyramids use byte-shuffle (temporal correlation on last axis).
    stored_dtype : str, optional
        The dtype of the stored array. Default ``"int16"``.

    Returns
    -------
    Codec

    Raises
    ------
    ValueError
        If *compressor* is an unknown string preset.
    """
    if not isinstance(compressor, str):
        return compressor
    if compressor == "blosc":
        # Select shuffle based on entropy and array type
        if is_pyramid and stored_dtype == "int16":
            shuffle = BloscShuffle.shuffle  # byte-shuffle for int16 (C,2,B) pyramids
        elif entropy is not None and entropy < 6.0:
            shuffle = BloscShuffle.noshuffle  # low entropy: byte-level LZ is more efficient
        else:
            shuffle = BloscShuffle.bitshuffle  # default: bitshuffle
        return BloscCodec(cname=BloscCname.zstd, clevel=1, shuffle=shuffle)
    if compressor == "gzip":
        return GzipCodec(level=5)
    if compressor == "zstd":
        return ZstdCodec(level=5)
    msg = f"Unknown compressor preset: {compressor!r}. Use 'blosc', 'gzip', 'zstd', or a Codec object."
    raise ValueError(msg)


def _compute_minmax_block(
    data: npt.NDArray[Any],
    stride: int,
) -> npt.NDArray[Any]:
    """Compute min/max for contiguous blocks of samples.

    Parameters
    ----------
    data : np.ndarray
        Input chunk with shape ``(C, T)``.  ``T`` need not be a multiple
        of *stride*; a shorter tail block is handled automatically.
    stride : int
        Block size in samples.

    Returns
    -------
    np.ndarray
        Array of shape ``(C, 2, n_blocks)`` where ``[:, 0, :]`` is min
        and ``[:, 1, :]`` is max.
    """
    n_channels, n_samples = data.shape
    n_full = n_samples // stride
    remainder = n_samples % stride

    parts: list[npt.NDArray[Any]] = []

    if n_full > 0:
        full_data = data[:, : n_full * stride].reshape(n_channels, n_full, stride)
        mins = full_data.min(axis=2)  # (C, n_full)
        maxs = full_data.max(axis=2)  # (C, n_full)
        parts.append(np.stack([mins, maxs], axis=1))  # (C, 2, n_full)

    if remainder > 0:
        tail = data[:, n_full * stride :]  # (C, remainder)
        tail_min = tail.min(axis=1, keepdims=True)  # (C, 1)
        tail_max = tail.max(axis=1, keepdims=True)  # (C, 1)
        parts.append(np.stack([tail_min, tail_max], axis=1))  # (C, 2, 1)

    return np.concatenate(parts, axis=2)


def _cascade_minmax(
    prev_level: npt.NDArray[Any],
    factor: int,
) -> npt.NDArray[Any]:
    """Build a coarser pyramid level from the previous level.

    Takes the previous level's ``(C, 2, B)`` array and reduces every
    *factor* consecutive blocks into one by taking min-of-mins and
    max-of-maxes.

    Parameters
    ----------
    prev_level : np.ndarray
        Previous level data with shape ``(C, 2, B_prev)``.
    factor : int
        Number of previous-level blocks per new block.

    Returns
    -------
    np.ndarray
        Array of shape ``(C, 2, ceil(B_prev / factor))``.
    """
    n_channels = prev_level.shape[0]
    n_blocks_prev = prev_level.shape[2]
    n_full = n_blocks_prev // factor
    remainder = n_blocks_prev % factor

    parts: list[npt.NDArray[Any]] = []

    if n_full > 0:
        full = prev_level[:, :, : n_full * factor].reshape(n_channels, 2, n_full, factor)
        mins = full[:, 0, :, :].min(axis=2)  # (C, n_full)
        maxs = full[:, 1, :, :].max(axis=2)  # (C, n_full)
        parts.append(np.stack([mins, maxs], axis=1))  # (C, 2, n_full)

    if remainder > 0:
        tail = prev_level[:, :, n_full * factor :]  # (C, 2, remainder)
        tail_min = tail[:, 0, :].min(axis=1, keepdims=True)  # (C, 1)
        tail_max = tail[:, 1, :].max(axis=1, keepdims=True)  # (C, 1)
        parts.append(np.stack([tail_min, tail_max], axis=1))  # (C, 2, 1)

    return np.concatenate(parts, axis=2)


def _create_zarr_array(
    group: zarr.Group,
    name: str,
    *,
    shape: tuple[int, ...],
    dtype: str,
    chunks: tuple[int, ...],
    shards: tuple[int, ...],
    compressor: str | BytesBytesCodec = "blosc",
    quantized: bool = False,
    entropy: float | None = None,
    is_pyramid: bool = False,
) -> zarr.Array:
    """Create a Zarr v3 array with sharding codec and compression.

    Parameters
    ----------
    group : zarr.Group
        Parent group.
    name : str
        Array name (path relative to group).
    shape : tuple[int, ...]
        Array shape.
    dtype : str
        NumPy dtype string.
    chunks : tuple[int, ...]
        Inner chunk shape (within shards).
    shards : tuple[int, ...]
        Shard shape.
    compressor : str or Codec, optional
        Compressor preset (``"blosc"``, ``"gzip"``, ``"zstd"``) or a
        zarr ``Codec`` instance.  Default ``"blosc"``.
    quantized : bool, optional
        Whether this array stores quantized data.  Default ``False``.
    entropy : float or None, optional
        Shannon entropy for shuffle auto-selection.
    is_pyramid : bool, optional
        Whether this is a pyramid ``(C, 2, B)`` array.

    Returns
    -------
    zarr.Array
    """
    resolved = _resolve_compressor(
        compressor,
        quantized,
        entropy=entropy,
        is_pyramid=is_pyramid,
        stored_dtype=dtype,
    )
    return group.create_array(
        name=name,
        shape=shape,
        dtype=dtype,
        chunks=chunks,
        shards=shards,
        compressors=resolved,
        fill_value=0,
        overwrite=True,
    )


@dataclass
class _AccumulatedStats:
    """Mutable accumulator for per-channel min/max statistics."""

    raw_min: npt.NDArray[Any]
    raw_max: npt.NDArray[Any]
    mm_min_min: npt.NDArray[Any]  # min of the min-axis values
    mm_min_max: npt.NDArray[Any]  # max of the min-axis values
    mm_max_min: npt.NDArray[Any]  # min of the max-axis values
    mm_max_max: npt.NDArray[Any]  # max of the max-axis values

    @classmethod
    def create(cls, n_channels: int, source_dtype: str) -> _AccumulatedStats:
        """Create an accumulator initialized to extreme values."""
        dt = np.dtype(source_dtype)
        if np.issubdtype(dt, np.floating):
            big = np.finfo(dt).max
        else:
            big = float(np.iinfo(dt).max)
        return cls(
            raw_min=np.full(n_channels, big, dtype=np.float64),
            raw_max=np.full(n_channels, -big, dtype=np.float64),
            mm_min_min=np.full(n_channels, big, dtype=np.float64),
            mm_min_max=np.full(n_channels, -big, dtype=np.float64),
            mm_max_min=np.full(n_channels, big, dtype=np.float64),
            mm_max_max=np.full(n_channels, -big, dtype=np.float64),
        )

    def update_raw(self, chunk: npt.NDArray[Any]) -> None:
        """Update raw stats from a (C, T) chunk."""
        np.minimum(self.raw_min, chunk.min(axis=1), out=self.raw_min)
        np.maximum(self.raw_max, chunk.max(axis=1), out=self.raw_max)

    def update_minmax(self, mm: npt.NDArray[Any]) -> None:
        """Update minmax stats from a (C, 2, B) block."""
        mins = mm[:, 0, :]  # (C, B)
        maxs = mm[:, 1, :]  # (C, B)
        np.minimum(self.mm_min_min, mins.min(axis=1), out=self.mm_min_min)
        np.maximum(self.mm_min_max, mins.max(axis=1), out=self.mm_min_max)
        np.minimum(self.mm_max_min, maxs.min(axis=1), out=self.mm_max_min)
        np.maximum(self.mm_max_max, maxs.max(axis=1), out=self.mm_max_max)


ProgressCallback = Callable[[int, int], None]
"""Type for progress callbacks: ``(samples_processed, total_samples)``."""


def _write_raw_and_compute_finest(
    source: Any,
    raw_array: zarr.Array,
    finest_array: zarr.Array | None,
    *,
    shard_samples: int,
    base_stride: int,
    n_samples: int,
    n_channels: int,
    stats: _AccumulatedStats | None = None,
    progress: ProgressCallback | None = None,
    lsb_shift_raw: int = 0,
    lsb_shift_pyramid: int = 0,
) -> list[npt.NDArray[Any]] | tuple[int, int, int, int]:
    """Write raw data and simultaneously compute the finest pyramid level.

    Iterates over *source* in shard-aligned chunks.  For each chunk,
    writes to *raw_array* and computes the min/max blocks.  If
    *finest_array* is provided, writes them there; otherwise collects
    them in a list (for quantization path where we write uint8 later).

    When *lsb_shift_raw* or *lsb_shift_pyramid* > 0, right-shifts
    values before writing and returns ``(raw_min, raw_max, pyr_min,
    pyr_max)`` of shifted values instead of collected blocks.

    Parameters
    ----------
    source : array-like
        Input data with shape ``(n_samples, n_channels)``.
    raw_array : zarr.Array
        Output raw zarr array.
    finest_array : zarr.Array or None
        Output finest min/max zarr array.  If ``None``, min/max blocks
        are collected and returned.
    shard_samples : int
        Number of time samples per shard.
    base_stride : int
        Base stride for the finest pyramid level.
    n_samples : int
        Total number of samples.
    n_channels : int
        Number of channels.
    stats : _AccumulatedStats or None, optional
        If provided, accumulate per-channel stats during the pass.
    progress : callable or None, optional
        Called with ``(samples_processed, total_samples)`` after each chunk.
    lsb_shift_raw : int, optional
        Right-shift for raw data before writing. Default 0 (lossless).
    lsb_shift_pyramid : int, optional
        Right-shift for pyramid data before writing. Default 0 (lossless).

    Returns
    -------
    list[np.ndarray] or tuple[int, int, int, int]
        Without LSB shift: list of min/max block arrays (empty if
        *finest_array* was provided). With LSB shift: ``(raw_shifted_min,
        raw_shifted_max, pyr_shifted_min, pyr_shifted_max)`` globals.
    """
    collected: list[npt.NDArray[Any]] = []
    raw_s_min = np.iinfo(np.int16).max
    raw_s_max = np.iinfo(np.int16).min
    pyr_s_min = np.iinfo(np.int16).max
    pyr_s_max = np.iinfo(np.int16).min

    for t0 in range(0, n_samples, shard_samples):
        t1 = min(t0 + shard_samples, n_samples)
        chunk = np.ascontiguousarray(np.asarray(source[t0:t1, :]).T)  # (T,C) → (C,T)

        # Compute finest min/max from ORIGINAL (unshifted) data
        mm = _compute_minmax_block(chunk, base_stride)

        # Apply shifts
        if lsb_shift_raw > 0:
            raw_chunk = (chunk >> lsb_shift_raw).astype(np.int16)
            raw_s_min = min(raw_s_min, int(raw_chunk.min()))
            raw_s_max = max(raw_s_max, int(raw_chunk.max()))
        else:
            raw_chunk = chunk

        if lsb_shift_pyramid > 0:
            mm_shifted = (mm >> lsb_shift_pyramid).astype(np.int16)
            pyr_s_min = min(pyr_s_min, int(mm_shifted.min()))
            pyr_s_max = max(pyr_s_max, int(mm_shifted.max()))
        else:
            mm_shifted = mm

        # Write raw
        raw_array[:, t0:t1] = raw_chunk

        # Write pyramid
        if finest_array is not None:
            b0 = t0 // base_stride
            b1 = b0 + mm_shifted.shape[2]
            finest_array[:, :, b0:b1] = mm_shifted
        else:
            collected.append(mm)

        # Accumulate stats
        if stats is not None:
            stats.update_raw(chunk)
            stats.update_minmax(mm)

        if progress is not None:
            progress(t1, n_samples)

    if lsb_shift_raw > 0 or lsb_shift_pyramid > 0:
        return (raw_s_min, raw_s_max, pyr_s_min, pyr_s_max)
    return collected


def _build_finest_from_raw(
    raw_array: zarr.Array,
    finest_array: zarr.Array | None,
    *,
    shard_samples: int,
    base_stride: int,
    n_samples: int,
    stats: _AccumulatedStats | None = None,
    progress: ProgressCallback | None = None,
) -> list[npt.NDArray[Any]]:
    """Build the finest pyramid level by reading an existing raw array.

    Parameters
    ----------
    raw_array : zarr.Array
        Source raw zarr array.
    finest_array : zarr.Array or None
        Output finest min/max zarr array.  If ``None``, blocks are
        collected and returned.
    shard_samples : int
        Number of time samples per read chunk.
    base_stride : int
        Base stride for the finest pyramid level.
    n_samples : int
        Total number of samples.
    stats : _AccumulatedStats or None, optional
        If provided, accumulate per-channel stats during the pass.
    progress : callable or None, optional
        Called with ``(samples_processed, total_samples)`` after each chunk.

    Returns
    -------
    list[np.ndarray]
        List of min/max block arrays.  Empty if *finest_array* was
        provided.
    """
    collected: list[npt.NDArray[Any]] = []

    # Detect if source is (T, C) or (C, T)
    is_tc = raw_array.shape[0] > raw_array.shape[1]

    for t0 in range(0, n_samples, shard_samples):
        t1 = min(t0 + shard_samples, n_samples)
        if is_tc:
            chunk = np.ascontiguousarray(np.asarray(raw_array[t0:t1, :]).T)  # (T,C) → (C,T)
        else:
            chunk = np.asarray(raw_array[:, t0:t1])  # already (C,T)

        mm = _compute_minmax_block(chunk, base_stride)

        if finest_array is not None:
            b0 = t0 // base_stride
            b1 = b0 + mm.shape[2]
            finest_array[:, :, b0:b1] = mm
        else:
            collected.append(mm)

        if stats is not None:
            stats.update_minmax(mm)

        if progress is not None:
            progress(t1, n_samples)

    return collected


def _build_cascade_levels(
    group: zarr.Group,
    finest_array: zarr.Array,
    strides: list[int],
    *,
    n_samples: int,
    n_channels: int,
    dtype: str,
    chunk_config: ChunkConfig,
    stride_factor: int,
    compressor: str | BytesBytesCodec,
    quantized: bool = False,
    entropy: float | None = None,
) -> None:
    """Build coarser pyramid levels by cascading from the finest level.

    For each stride beyond the base, reads the previous level in
    shard-aligned chunks and writes the cascaded result.

    Parameters
    ----------
    group : zarr.Group
        Parent zarr group to write arrays into.
    finest_array : zarr.Array
        The finest pyramid level, already written.
    strides : list[int]
        Full stride sequence (ascending).
    n_samples : int
        Total number of raw samples.
    n_channels : int
        Number of channels.
    dtype : str
        Dtype string for pyramid arrays.
    chunk_config : ChunkConfig
        Chunk/shard configuration.
    stride_factor : int
        Factor between consecutive levels.
    compressor : str or Codec
        Compressor preset or codec instance.
    quantized : bool, optional
        Whether arrays store quantized data, by default ``False``.
    entropy : float or None, optional
        Shannon entropy for shuffle auto-selection.
    """
    if len(strides) < 2:
        return

    prev_array: zarr.Array = finest_array

    for i in range(1, len(strides)):
        stride = strides[i]
        n_blk = n_blocks(n_samples, stride)

        arr_path = minmax_path(stride).lstrip("/")
        new_array = _create_zarr_array(
            group,
            arr_path,
            shape=(n_channels, 2, n_blk),
            dtype=dtype,
            chunks=chunk_config.minmax_chunk,
            shards=chunk_config.minmax_shard,
            compressor=compressor,
            quantized=quantized,
            entropy=entropy,
            is_pyramid=True,
        )

        # Read previous level in shard-aligned chunks and cascade
        n_prev = prev_array.shape[2]  # blocks are axis 2 in (C, 2, B)
        read_chunk = chunk_config.minmax_shard[2] * stride_factor

        for b0 in range(0, n_prev, read_chunk):
            b1 = min(b0 + read_chunk, n_prev)
            prev_chunk = np.asarray(prev_array[:, :, b0:b1])
            cascaded = _cascade_minmax(prev_chunk, stride_factor)

            out_b0 = b0 // stride_factor
            out_b1 = out_b0 + cascaded.shape[2]
            new_array[:, :, out_b0:out_b1] = cascaded

        prev_array = new_array


def _quantize_raw_in_place(
    root: zarr.Group,
    raw_quantization: RawQuantization,
    *,
    cfg: ChunkConfig,
    n_samples: int,
    n_channels: int,
    compressor: str | BytesBytesCodec,
) -> None:
    """Replace the float32 raw array with a quantized unsigned int array.

    Reads the existing float32 ``/raw``, quantizes in chunks, writes to
    a new ``/raw_quantized`` array, then deletes ``/raw`` and renames.

    Parameters
    ----------
    root : zarr.Group
        The zarr store root group (opened in write mode).
    raw_quantization : RawQuantization
        Quantization parameters.
    cfg : ChunkConfig
        Chunk/shard configuration.
    n_samples : int
        Total samples.
    n_channels : int
        Number of channels.
    compressor : str or Codec
        Compressor preset or codec instance.
    """
    offset = np.array(raw_quantization.offset)
    scale = np.array(raw_quantization.scale)
    q_dtype = raw_quantization.quantized_dtype

    # Read the float32 raw array, quantize chunks, buffer in memory,
    # then replace the array.  zarr v3 does not support move/rename.
    old_raw: zarr.Array = root["raw"]  # type: ignore[assignment]

    shard_samples = cfg.raw_shard[1]  # time axis is now axis 1
    quantized_chunks: list[tuple[int, int, npt.NDArray[Any]]] = []
    for t0 in range(0, n_samples, shard_samples):
        t1 = min(t0 + shard_samples, n_samples)
        chunk = np.asarray(old_raw[:, t0:t1])  # (C, T_chunk)
        quantized_chunks.append((t0, t1, quantize_array(chunk, offset, scale, q_dtype)))

    # Delete float32 raw and create quantized replacement
    del root["raw"]
    q_array = _create_zarr_array(
        root,
        "raw",
        shape=(n_channels, n_samples),
        dtype=q_dtype,
        chunks=cfg.raw_chunk,
        shards=cfg.raw_shard,
        compressor=compressor,
        quantized=True,
    )

    for t0, t1, q_chunk in quantized_chunks:
        q_array[:, t0:t1] = q_chunk

    # Store quantization params in array attrs
    q_array.attrs.update({"quantization": raw_quantization.to_dict()})


def _build_non_quantized(
    *,
    source: Any,
    root: zarr.Group,
    strides: list[int],
    finest_stride: int,
    finest_n_blocks: int,
    n_samples: int,
    n_channels: int,
    dtype: str,
    cfg: ChunkConfig,
    stride_factor: int,
    compressor: str | BytesBytesCodec,
    lsb_shift_raw: int | str,
    lsb_shift_pyramid: int | str,
    noise_floor: float | None,
    progress: ProgressCallback | None,
) -> int:
    """Build raw + pyramid arrays with optional LSB shift.

    Returns the resolved raw shift value (0 if no shift).
    """
    raw_shift = _resolve_lsb_shift(lsb_shift_raw, source, n_samples, noise_floor)
    pyr_shift = _resolve_lsb_shift(
        lsb_shift_pyramid,
        source,
        n_samples,
        noise_floor,
        extra_shift=2,
    )

    # Compute entropy from first shard for shuffle auto-selection
    n_sample_ent = min(30_000, n_samples)
    sample_ct = np.ascontiguousarray(np.asarray(source[0:n_sample_ent, :]).T)
    raw_entropy: float | None = None
    pyr_entropy: float | None = None
    if raw_shift > 0:
        raw_entropy = compute_entropy(sample_ct >> raw_shift)
    if pyr_shift > 0:
        mm_sample = _compute_minmax_block(sample_ct, finest_stride)
        pyr_entropy = compute_entropy(mm_sample >> pyr_shift)

    # Create arrays
    raw_array = _create_zarr_array(
        root,
        "raw",
        shape=(n_channels, n_samples),
        dtype=dtype,
        chunks=cfg.raw_chunk,
        shards=cfg.raw_shard,
        compressor=compressor,
        entropy=raw_entropy,
    )
    finest_array = _create_zarr_array(
        root,
        minmax_path(finest_stride).lstrip("/"),
        shape=(n_channels, 2, finest_n_blocks),
        dtype=dtype,
        chunks=cfg.minmax_chunk,
        shards=cfg.minmax_shard,
        compressor=compressor,
        entropy=pyr_entropy,
        is_pyramid=True,
    )

    # Write with shift, tracking min/max of shifted values
    result = _write_raw_and_compute_finest(
        source,
        raw_array,
        finest_array,
        shard_samples=cfg.raw_shard[1],
        base_stride=finest_stride,
        n_samples=n_samples,
        n_channels=n_channels,
        progress=progress,
        lsb_shift_raw=raw_shift,
        lsb_shift_pyramid=pyr_shift,
    )

    _build_cascade_levels(
        root,
        finest_array,
        strides,
        n_samples=n_samples,
        n_channels=n_channels,
        dtype=dtype,
        chunk_config=cfg,
        stride_factor=stride_factor,
        compressor=compressor,
        entropy=pyr_entropy,
    )

    # Store lsb_shift in array attrs (before int8 conversion so attrs are preserved)
    if raw_shift > 0:
        raw_arr: zarr.Array = root["raw"]  # type: ignore[assignment]
        raw_arr.attrs.update({"lsb_shift": raw_shift, "original_dtype": dtype})
    if pyr_shift > 0:
        for s in strides:
            pyr_arr: zarr.Array = root[minmax_path(s).lstrip("/")]  # type: ignore[assignment]
            pyr_arr.attrs.update({"lsb_shift": pyr_shift, "original_dtype": dtype})

    # Convert to int8 if range fits
    if isinstance(result, tuple):
        raw_shifted_min, raw_shifted_max, pyr_shifted_min, pyr_shifted_max = result
        if raw_shift > 0 and raw_shifted_min >= -128 and raw_shifted_max <= 127:
            _convert_array_to_int8(
                root,
                "raw",
                cfg.raw_chunk,
                cfg.raw_shard,
                compressor,
                entropy=raw_entropy,
            )
        if pyr_shift > 0 and pyr_shifted_min >= -128 and pyr_shifted_max <= 127:
            for s in strides:
                _convert_array_to_int8(
                    root,
                    minmax_path(s).lstrip("/"),
                    cfg.minmax_chunk,
                    cfg.minmax_shard,
                    compressor,
                    entropy=pyr_entropy,
                    is_pyramid=True,
                )

    return raw_shift
