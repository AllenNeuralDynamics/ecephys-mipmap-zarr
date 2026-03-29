"""Build multi-resolution min/max pyramids for electrophysiology Zarr stores.

This module provides two entry points:

- :func:`build_mipmap` writes raw data and all pyramid levels in a single
  pass (finest level computed alongside raw, coarser levels cascaded).
- :func:`add_pyramid` reads an existing ``/raw`` array and writes
  pyramid-only output with a back-reference to the original store.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ecephys_mipmap_zarr.read import MipmapStore

import numpy as np
import numpy.typing as npt
import zarr
from zarr.abc.codec import BytesBytesCodec
from zarr.codecs import BloscCodec, GzipCodec, ZstdCodec
from zarr.codecs.blosc import BloscCname, BloscShuffle

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
    compute_optimal_lsb_shift,
    compute_quantization_params,
    compute_strides,
    minmax_path,
    n_blocks,
    quantize_array,
)

__all__ = [
    "add_pyramid",
    "build_mipmap",
    "ensure_pyramid",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _open_store(path: str | Path, mode: str = "r") -> zarr.Group:
    """Open a zarr store from a local path or remote URI.

    Supports ``s3://`` URIs via fsspec (requires the ``s3`` extra).

    Parameters
    ----------
    path : str or Path
        Local filesystem path or remote URI (e.g. ``s3://bucket/store.zarr``).
    mode : str, optional
        Open mode, by default ``"r"``.

    Returns
    -------
    zarr.Group
    """
    path_str = str(path)
    if "://" in path_str:
        # Remote URI — use fsspec
        try:
            import fsspec
        except ImportError as e:
            msg = "fsspec is required for remote URIs: pip install ecephys-mipmap-zarr[s3]"
            raise ImportError(msg) from e
        protocol = path_str.split("://")[0]
        fs = fsspec.filesystem(protocol, anon=True)
        store = zarr.storage.FsspecStore(fs=fs, path=path_str.replace(f"{protocol}://", ""))
        return zarr.open_group(store, mode=mode)  # type: ignore[arg-type]
    return zarr.open_group(path_str, mode=mode)  # type: ignore[arg-type]


_RAW_ARRAY_PATHS = ("raw", "traces_seg0")
"""Candidate paths to probe when looking for the raw array."""


def _find_raw_array(group: zarr.Group, raw_path: str) -> zarr.Array:
    """Find the raw array in a zarr group.

    If *raw_path* is explicit (not the default ``"raw"``), use it
    directly.  Otherwise probe a list of known paths.

    Parameters
    ----------
    group : zarr.Group
        The opened zarr store root.
    raw_path : str
        Explicit path, or ``"raw"`` to trigger auto-detection.

    Returns
    -------
    zarr.Array

    Raises
    ------
    KeyError
        If no raw array is found.
    """
    # If user gave an explicit non-default path, use it directly
    if raw_path != "raw":
        return group[raw_path]  # type: ignore[return-value]

    # Probe known paths
    for candidate in _RAW_ARRAY_PATHS:
        try:
            return group[candidate]  # type: ignore[return-value]
        except KeyError:
            continue

    members = list(group)
    msg = (
        f"Could not find raw array. Tried: {list(_RAW_ARRAY_PATHS)}. "
        f"Store contains: {members}. Use --raw-path to specify explicitly."
    )
    raise KeyError(msg)


def _extract_source_metadata(group: zarr.Group) -> dict[str, Any] | None:
    """Extract probe and annotation metadata from a source zarr store.

    Parameters
    ----------
    group : zarr.Group
        Source store root group.

    Returns
    -------
    dict or None
        Dictionary with ``probe`` and/or ``annotations`` keys, or
        ``None`` if no relevant metadata found.
    """
    src_attrs = dict(group.attrs)
    metadata: dict[str, Any] = {}

    for key in ("probe", "annotations"):
        if key in src_attrs:
            metadata[key] = src_attrs[key]

    return metadata if metadata else None


def _copy_source_arrays(src: zarr.Group, dst: zarr.Group) -> None:
    """Copy channel_ids and properties/location arrays from source to destination.

    Parameters
    ----------
    src : zarr.Group
        Source store root group.
    dst : zarr.Group
        Destination store root group.
    """
    # Copy channel_ids
    if "channel_ids" in list(src):
        ch_ids = np.asarray(src["channel_ids"][:])  # type: ignore[index]
        dst.create_array("channel_ids", data=ch_ids, overwrite=True)

    # Copy properties/location
    if "properties" in list(src):
        try:
            props_src = src["properties"]
            if "location" in list(props_src):  # type: ignore[arg-type]
                loc = np.asarray(props_src["location"][:])  # type: ignore[index]
                props_dst = dst.require_group("properties")
                props_dst.create_array("location", data=loc, overwrite=True)
        except (KeyError, AttributeError):
            pass  # properties group doesn't have what we need


def _apply_adc_quantization(src: zarr.Group, dst: zarr.Group, lsb_shift: int = 0) -> None:
    """Store ADC gain/offset as RawQuantization if gain_to_uV is available.

    When the source is an SI zarr with ``properties/gain_to_uV`` and
    ``properties/offset_to_uV``, store these as :class:`RawQuantization`
    in the destination's ``/raw`` array attrs.  This lets the reader
    dequantize int16 ADC values to float32 µV transparently.

    When *lsb_shift* > 0, the gain is multiplied by ``2**lsb_shift``
    so that the reader's dequantization path handles both the bit-shift
    reconstruction and unit conversion in a single step.

    Parameters
    ----------
    src : zarr.Group
        Source SI zarr store.
    dst : zarr.Group
        Destination mipmap store (must already contain ``/raw``).
    lsb_shift : int, optional
        LSB shift applied to raw data, by default 0.
    """
    try:
        props = src["properties"]
        gain = np.asarray(props["gain_to_uV"][:])  # type: ignore[index]
        offset = np.asarray(props["offset_to_uV"][:])  # type: ignore[index]
    except (KeyError, AttributeError):
        return  # No gain/offset data available

    # Combine lsb_shift into gain: effective_scale = gain * 2^shift
    effective_gain = gain * (1 << lsb_shift) if lsb_shift > 0 else gain

    try:
        raw_array: zarr.Array = dst["raw"]  # type: ignore[assignment]
        raw_dtype = str(raw_array.dtype)
    except KeyError:
        return  # No /raw in destination (pyramid-only store)

    rq = RawQuantization(
        offset=offset.tolist(),
        scale=effective_gain.tolist(),
        source_dtype="float32",
        quantized_dtype=raw_dtype,
    )

    raw_array.attrs.update({"quantization": rq.to_dict()})
    # When combined with lsb_shift, remove the standalone lsb_shift attr
    # since dequantization via RawQuantization handles everything
    if lsb_shift > 0:
        existing = dict(raw_array.attrs)
        existing.pop("lsb_shift", None)
        existing.pop("original_dtype", None)
        existing["quantization"] = rq.to_dict()
        raw_array.attrs.clear()
        raw_array.attrs.update(existing)


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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


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
