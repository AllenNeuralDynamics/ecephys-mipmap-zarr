"""Write engine for building mipmap zarr stores.

Internal module — contains the chunked I/O, min/max computation,
cascading, quantization, and LSB-shift logic used by the public
build API.  Not part of the public API.
"""

from __future__ import annotations

import errno
import logging
from collections.abc import Callable, Generator
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import zarr
from zarr.abc.codec import BytesBytesCodec
from zarr.codecs import BloscCodec, GzipCodec, ZstdCodec

from ecephys_mipmap_zarr.metadata import (
    BAND_LIMITED_FRACTION,
    HF_NOISE_BAND_HZ,
    ChunkConfig,
    Encoding,
    EncodingConfig,
    LevelQuantization,
    RawQuantization,
    compute_entropy,
    compute_quantization_params,
    estimate_hf_noise,
    hf_band_fraction,
    lsb_shift_for_noise,
    minmax_path,
    n_blocks,
    quantize_array,
)

__all__: list[str] = []  # private module

log = logging.getLogger(__name__)


# Samples read from the start of the source to estimate noise and entropy.
_SAMPLE_SAMPLES = 30_000
# Largest fraction of an array clip_outliers will clip to fit int8.
_CLIP_THRESHOLD = 0.0001
#: Array attribute counting the codes clipping changed, present only on an array that was clipped.
CLIPPED_ATTR = "clipped_values"
# "auto" pyramid shift over the raw one: pyramids are only drawn.
_PYRAMID_DISPLAY_BITS = 2


def _read_channels_first(source: Any, t0: int, t1: int) -> npt.NDArray[Any]:
    """Read samples ``[t0, t1)`` of a ``(T, C)`` source as ``(C, T)``."""
    read_channels_first = getattr(source, "read_channels_first", None)
    if read_channels_first is not None:
        return np.asarray(read_channels_first(t0, t1))
    return np.ascontiguousarray(np.asarray(source[t0:t1, :]).T)


class _StoredRawSource:
    """A stored raw array, ``(T, C)`` or ``(C, T)``, presented as a ``(T, C)`` source."""

    def __init__(self, array: zarr.Array) -> None:
        self._array = array
        # More samples than channels identifies SpikeInterface's (T, C) layout.
        self._time_first = array.shape[0] > array.shape[1]
        n_samples, n_channels = array.shape if self._time_first else array.shape[::-1]
        self.shape = (n_samples, n_channels)
        self.dtype = array.dtype

    def read_channels_first(self, t0: int, t1: int) -> npt.NDArray[Any]:
        if self._time_first:
            return np.ascontiguousarray(np.asarray(self._array[t0:t1, :]).T)
        return np.asarray(self._array[:, t0:t1])


@dataclass(frozen=True)
class _ResolvedEncoding:
    """Concrete encodings for one build, with shifts resolved to bit counts."""

    raw: Encoding
    pyramids: Encoding
    raw_shift: int = 0
    pyramid_shift: int = 0


def _resolve_encoding(
    encoding: EncodingConfig,
    *,
    source: Any,
    source_dtype: str,
    sampling_frequency: float,
    writes_raw: bool = True,
) -> _ResolvedEncoding:
    """Pick each array's encoding and resolve ``"auto"`` shifts.

    Raises
    ------
    ValueError
        If a setting would go unused, ``"shift"`` is asked of a float source,
        or an ``"auto"`` shift needs a noise estimate the sampling rate can't give.
    """
    is_integer = np.issubdtype(np.dtype(source_dtype), np.integer)
    default: Encoding = "shift" if is_integer else "range"
    defaults = EncodingConfig()

    if not writes_raw:
        raw_settings = ("raw", "raw_shift", "raw_range_dtype", "raw_quantization")
        # raw="none" agrees with writing no raw, so LOSSLESS_ENCODING suits add_pyramid too.
        set_raw = [
            name
            for name in raw_settings
            if getattr(encoding, name) != getattr(defaults, name) and not (name == "raw" and encoding.raw == "none")
        ]
        if set_raw:
            msg = f"no raw array is written, so {', '.join(set_raw)} would be unused"
            raise ValueError(msg)

    raw: Encoding = (encoding.raw or default) if writes_raw else "none"
    pyramids: Encoding = encoding.pyramids or default

    for name, kind, shift, range_dtype, params, default_dtype in (
        ("raw", raw, encoding.raw_shift, encoding.raw_range_dtype, encoding.raw_quantization, "uint16"),
        (
            "pyramids",
            pyramids,
            encoding.pyramid_shift,
            encoding.pyramid_range_dtype,
            encoding.level_quantization,
            "uint8",
        ),
    ):
        if kind not in ("shift", "range", "none"):
            msg = f"{name} encoding must be 'shift', 'range' or 'none', got {kind!r}"
            raise ValueError(msg)
        if kind == "shift" and not is_integer:
            msg = f"{name}='shift' needs an integer source; use 'range' or 'none' for {source_dtype}"
            raise ValueError(msg)
        shift_name = "raw_shift" if name == "raw" else "pyramid_shift"
        if kind != "shift" and shift != "auto":
            msg = f"{shift_name} applies only to {name}='shift', not {kind!r}"
            raise ValueError(msg)
        if shift != "auto" and (not isinstance(shift, int) or shift < 0):
            msg = f"{shift_name} must be 'auto' or a non-negative int, got {shift!r}"
            raise ValueError(msg)
        if kind != "range" and (params is not None or range_dtype != default_dtype):
            msg = f"range settings for {name} apply only to {name}='range', not {kind!r}"
            raise ValueError(msg)

    raw_auto = raw == "shift" and encoding.raw_shift == "auto"
    pyramid_auto = pyramids == "shift" and encoding.pyramid_shift == "auto"
    if encoding.noise_floor is not None and not (raw_auto or pyramid_auto):
        msg = "noise_floor is used only by an 'auto' shift"
        raise ValueError(msg)

    noise_floor = encoding.noise_floor
    if (raw_auto or pyramid_auto) and noise_floor is None:
        band_top = HF_NOISE_BAND_HZ[1]
        if sampling_frequency / 2 <= band_top:
            msg = (
                f"an 'auto' shift measures noise up to {band_top:g} Hz, above Nyquist at "
                f"{sampling_frequency:g} Hz; give the shift in bits or a noise_floor"
            )
            raise ValueError(msg)
        n = min(_SAMPLE_SAMPLES, source.shape[0])
        sample = _read_channels_first(source, 0, n)
        fraction = hf_band_fraction(sample, sampling_frequency)
        if fraction < BAND_LIMITED_FRACTION:
            msg = (
                f"an 'auto' shift measures noise in {HF_NOISE_BAND_HZ[0]:g}-{band_top:g} Hz, where this source "
                f"has {fraction:.2g} of its spike-band power, so it was low-passed; give a noise_floor "
                "measured before the filter (preprocess.chain_noise_floor) or the shift in bits"
            )
            raise ValueError(msg)
        noise_floor = estimate_hf_noise(sample, sampling_frequency)

    def bits(shift: int | str, auto: bool, extra: int) -> int:
        if auto:
            assert noise_floor is not None
            return lsb_shift_for_noise(noise_floor) + extra
        return int(shift)

    return _ResolvedEncoding(
        raw=raw,
        pyramids=pyramids,
        raw_shift=bits(encoding.raw_shift, raw_auto, 0) if raw == "shift" else 0,
        pyramid_shift=bits(encoding.pyramid_shift, pyramid_auto, _PYRAMID_DISPLAY_BITS) if pyramids == "shift" else 0,
    )


def _replace_array(root: zarr.Group, dst_path: str, src_path: str) -> None:
    """Replace *dst_path* with *src_path* in the zarr store.

    Deletes *dst_path*, then renames *src_path* to *dst_path* via
    filesystem move (LocalStore) or falls back to a chunked copy.
    """
    del root[dst_path]
    store = root.store
    if hasattr(store, "root"):
        import shutil

        src_dir = store.root / src_path
        dst_dir = store.root / dst_path
        if src_dir.exists():
            shutil.move(str(src_dir), str(dst_dir))
            return
    # Fallback: array already at src_path, just leave it
    # (non-local stores would need a copy, but we don't write to S3)


def _convert_array_to_int8(
    root: zarr.Group,
    path: str,
    chunks: tuple[int, ...],
    shards: tuple[int, ...],
    compressor: str | BytesBytesCodec,
    entropy: float | None = None,
    is_pyramid: bool = False,
    clip: bool = False,
) -> None:
    """Convert an int16 zarr array to int8 in-place, streaming shard-by-shard.

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
    clip : bool, optional
        When ``True``, clip values to [-128, 127] before casting and record how many were clipped in the
        array's :data:`CLIPPED_ATTR`. When ``False`` (default), values must already fit in int8.
    """
    old_arr: zarr.Array = root[path]  # type: ignore[assignment]
    shape = old_arr.shape
    old_attrs = dict(old_arr.attrs)

    tmp_path = path + "__int8_tmp"
    tmp_arr = _create_zarr_array(
        root,
        tmp_path,
        shape=shape,
        dtype="int8",
        chunks=chunks,
        shards=shards,
        compressor=compressor,
        entropy=entropy,
        is_pyramid=is_pyramid,
    )

    # Stream shard-by-shard to avoid loading entire array into memory.
    last_axis = len(shape) - 1
    shard_len = shards[last_axis]
    total = shape[last_axis]
    clipped = 0
    for i0 in range(0, total, shard_len):
        i1 = min(i0 + shard_len, total)
        sl = (slice(None), slice(i0, i1)) if last_axis == 1 else (slice(None), slice(None), slice(i0, i1))
        data = np.asarray(old_arr[sl])
        if clip:
            clipped += int(((data < -128) | (data > 127)).sum())
            data = np.clip(data, -128, 127)
        tmp_arr[sl] = data.astype(np.int8)

    if clip:
        old_attrs[CLIPPED_ATTR] = clipped
    if old_attrs:
        tmp_arr.attrs.update(old_attrs)

    _replace_array(root, path, tmp_path)


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
        shuffle: Literal["noshuffle", "shuffle", "bitshuffle"]
        if is_pyramid and stored_dtype == "int16":
            shuffle = "shuffle"  # byte-shuffle for int16 (C,2,B) pyramids
        elif entropy is not None and entropy < 6.0:
            shuffle = "noshuffle"  # low entropy: byte-level LZ is more efficient
        else:
            shuffle = "bitshuffle"  # default: bitshuffle
        return BloscCodec(cname="zstd", clevel=1, shuffle=shuffle)
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
    """Mutable accumulator for per-channel min/max statistics.

    The four ``mm_*`` fields track extrema of the finest pyramid's
    min/max axes.  Per-channel ``raw_min`` / ``raw_max`` are *not* stored
    separately: they equal ``mm_min_min`` / ``mm_max_max`` exactly
    (min-of-block-mins = global min, max-of-block-maxes = global max).
    """

    mm_min_min: npt.NDArray[Any]  # min of the min-axis values (= per-channel raw min)
    mm_min_max: npt.NDArray[Any]  # max of the min-axis values
    mm_max_min: npt.NDArray[Any]  # min of the max-axis values
    mm_max_max: npt.NDArray[Any]  # max of the max-axis values (= per-channel raw max)

    @classmethod
    def create(cls, n_channels: int, source_dtype: str) -> _AccumulatedStats:
        """Create an accumulator initialized to extreme values."""
        dt = np.dtype(source_dtype)
        if np.issubdtype(dt, np.floating):
            big = np.finfo(dt).max
        else:
            big = float(np.iinfo(dt).max)
        return cls(
            mm_min_min=np.full(n_channels, big, dtype=np.float64),
            mm_min_max=np.full(n_channels, -big, dtype=np.float64),
            mm_max_min=np.full(n_channels, big, dtype=np.float64),
            mm_max_max=np.full(n_channels, -big, dtype=np.float64),
        )

    @property
    def raw_min(self) -> npt.NDArray[Any]:
        """Per-channel raw minimum (identical to ``mm_min_min``)."""
        return self.mm_min_min

    @property
    def raw_max(self) -> npt.NDArray[Any]:
        """Per-channel raw maximum (identical to ``mm_max_max``)."""
        return self.mm_max_max

    def update_minmax(self, mm: npt.NDArray[Any]) -> None:
        """Update minmax stats from a (C, 2, B) block."""
        mins = mm[:, 0, :]  # (C, B)
        maxs = mm[:, 1, :]  # (C, B)
        np.minimum(self.mm_min_min, mins.min(axis=1), out=self.mm_min_min)
        np.maximum(self.mm_min_max, mins.max(axis=1), out=self.mm_min_max)
        np.minimum(self.mm_max_min, maxs.min(axis=1), out=self.mm_max_min)
        np.maximum(self.mm_max_max, maxs.max(axis=1), out=self.mm_max_max)


@dataclass
class _ShiftStats:
    """Statistics from an LSB-shifted write pass."""

    raw_min: int = 0
    raw_max: int = 0
    pyr_min: int = 0
    pyr_max: int = 0
    raw_outliers: int = 0
    pyr_outliers: int = 0
    raw_total: int = 0
    pyr_total: int = 0

    def add_raw(self, shifted: npt.NDArray[Any]) -> None:
        """Accumulate one shifted raw chunk."""
        self.raw_min = min(self.raw_min, int(shifted.min()))
        self.raw_max = max(self.raw_max, int(shifted.max()))
        self.raw_outliers += int(((shifted < -128) | (shifted > 127)).sum())
        self.raw_total += shifted.size

    def add_pyramid(self, shifted: npt.NDArray[Any]) -> None:
        """Accumulate one shifted block of the finest level."""
        self.pyr_min = min(self.pyr_min, int(shifted.min()))
        self.pyr_max = max(self.pyr_max, int(shifted.max()))
        self.pyr_outliers += int(((shifted < -128) | (shifted > 127)).sum())
        self.pyr_total += shifted.size


ProgressCallback = Callable[[int, int], None]
"""Type for progress callbacks: ``(samples_processed, total_samples)``."""


_TRANSIENT_ERRNOS: frozenset[int] = frozenset(
    {
        errno.EIO,  # 5 — I/O error (FUSE, network-backed filesystem)
        errno.EAGAIN,  # 11 — resource temporarily unavailable / EWOULDBLOCK
        errno.EBUSY,  # 16 — device or resource busy
        errno.ENETDOWN,  # 100 — network is down
        errno.ENETUNREACH,  # 101 — network unreachable
        errno.ECONNRESET,  # 104 — connection reset by peer
        errno.ETIMEDOUT,  # 110 — connection timed out
        errno.EHOSTUNREACH,  # 113 — no route to host
    }
)
"""OSError ``errno`` values that indicate a transient (retryable) failure.

Permanent errors (``ENOENT``, ``EACCES``, ``EINVAL``, etc.) are deliberately
excluded so a config typo fails immediately rather than burning up to 31 s of
exponential-backoff sleep before surfacing.
"""


def _retry_on_oserror(
    read_fn: Callable[[int, int], npt.NDArray[Any]],
    retries: int,
) -> Callable[[int, int], npt.NDArray[Any]]:
    """Wrap *read_fn* to retry *transient* :class:`OSError` with exponential backoff.

    Intended for reading from remote-backed sources (FUSE-mounted S3,
    fsspec, etc.) where EIO / ETIMEDOUT spikes happen under network or
    rate-limit pressure.  Reads are idempotent (same ``[t0, t1)`` range
    returns the same data), so retrying a failed chunk is safe.

    Only errors whose ``errno`` is in :data:`_TRANSIENT_ERRNOS` are retried.
    Permanent errors (missing file, permission denied, bad argument) raise
    immediately so they fail fast instead of hiding behind 31 s of sleep.

    Parameters
    ----------
    read_fn : callable
        The original ``(t0, t1) -> np.ndarray`` reader.
    retries : int
        Maximum number of additional attempts after the first failure.
        ``0`` returns *read_fn* unchanged.  Backoff is ``2**attempt``
        seconds (1, 2, 4, 8, ...).
    """
    if retries <= 0:
        return read_fn

    import time

    def _wrapped(t0: int, t1: int) -> npt.NDArray[Any]:
        attempt = 0
        while True:
            try:
                return read_fn(t0, t1)
            except OSError as exc:
                # Only retry transient errors; permanent errors (typos,
                # missing files, bad permissions) should fail immediately.
                if exc.errno not in _TRANSIENT_ERRNOS or attempt >= retries:
                    raise
                delay = 2**attempt
                log.warning(
                    "source read [%d, %d) failed (%s: %s, errno=%d), retry %d/%d in %ds",
                    t0,
                    t1,
                    type(exc).__name__,
                    exc,
                    exc.errno if exc.errno is not None else -1,
                    attempt + 1,
                    retries,
                    delay,
                )
                time.sleep(delay)
                attempt += 1

    return _wrapped


def _chunk_iter(
    read_fn: Callable[[int, int], npt.NDArray[Any]],
    n_samples: int,
    shard_samples: int,
    prefetch_chunks: int = 0,
) -> Generator[tuple[int, int, npt.NDArray[Any]], None, None]:
    """Yield ``(t0, t1, chunk)`` in shard-aligned order.

    Parameters
    ----------
    read_fn : callable
        ``read_fn(t0, t1) -> np.ndarray`` returning one chunk.
    n_samples : int
        Total number of samples.
    shard_samples : int
        Samples per shard-aligned chunk.
    prefetch_chunks : int, optional
        When > 0, read ahead using a thread pool (sliding window).
        Memory bounded at *prefetch_chunks* × chunk_size.
    """
    boundaries = [(t0, min(t0 + shard_samples, n_samples)) for t0 in range(0, n_samples, shard_samples)]

    if prefetch_chunks <= 0:
        for t0, t1 in boundaries:
            yield t0, t1, read_fn(t0, t1)
        return

    import contextvars
    from collections import deque
    from concurrent.futures import Future, ThreadPoolExecutor

    # Propagate the caller's context (scipy.fft.set_workers, etc.) into
    # each prefetch worker.  ThreadPoolExecutor.submit doesn't propagate
    # contextvars by default — new threads start with an empty context,
    # which silently drops the caller's ``scipy.fft.set_workers(-1)``.
    # NOTE: we copy *per submit* because ``Context.run`` raises RuntimeError
    # if the same Context object is active on two threads concurrently,
    # which happens whenever prefetch_chunks >= 2.
    def _submit_in_context(t0: int, t1: int) -> Future[npt.NDArray[Any]]:
        ctx = contextvars.copy_context()
        return pool.submit(ctx.run, read_fn, t0, t1)

    with ThreadPoolExecutor(max_workers=prefetch_chunks) as pool:
        pending: deque[tuple[int, int, Future[npt.NDArray[Any]]]] = deque()
        idx = 0

        while idx < len(boundaries) and len(pending) < prefetch_chunks:
            t0, t1 = boundaries[idx]
            pending.append((t0, t1, _submit_in_context(t0, t1)))
            idx += 1

        while pending:
            t0, t1, future = pending.popleft()
            yield t0, t1, future.result()
            if idx < len(boundaries):
                t0_next, t1_next = boundaries[idx]
                pending.append((t0_next, t1_next, _submit_in_context(t0_next, t1_next)))
                idx += 1


def _shift(values: npt.NDArray[Any], bits: int) -> npt.NDArray[Any]:
    """Divide by ``2**bits``, rounding to nearest (half-step bias, not truncation)."""
    return ((values + (1 << (bits - 1))) >> bits).astype(np.int16)


def _write_raw_and_compute_finest(
    source: Any,
    raw_array: zarr.Array | None,
    finest_array: zarr.Array,
    *,
    shard_samples: int,
    base_stride: int,
    n_samples: int,
    stats: _AccumulatedStats | None = None,
    progress: ProgressCallback | None = None,
    lsb_shift_raw: int = 0,
    lsb_shift_pyramid: int = 0,
    prefetch_chunks: int = 0,
    source_retries: int = 0,
) -> _ShiftStats:
    """Write raw (if *raw_array*) and the finest pyramid level in one pass over *source*.

    Min/max blocks come from the unshifted values, so *stats* hold source
    units whatever the shifts.

    Parameters
    ----------
    source : array-like
        ``(n_samples, n_channels)`` source; a ``read_channels_first(t0, t1)``
        method, if present, is used instead of slicing and transposing.
    raw_array : zarr.Array or None
        Output raw array; ``None`` writes the pyramid only.
    finest_array : zarr.Array
        Output finest min/max array.
    shard_samples : int
        Samples per read.
    base_stride : int
        Stride of the finest pyramid level.
    n_samples : int
        Total number of samples.
    stats : _AccumulatedStats or None, optional
        Per-channel extremes to accumulate.
    progress : callable or None, optional
        Called with ``(samples_processed, total_samples)`` after each read.
    lsb_shift_raw, lsb_shift_pyramid : int, optional
        Bits to shift off before writing; 0 writes values unchanged.
    prefetch_chunks : int, optional
        Reads to run ahead in background threads.
    source_retries : int, optional
        Retries for a transient ``OSError`` on read.

    Returns
    -------
    _ShiftStats
        Extremes and int8 outlier counts of the shifted values.
    """
    shift_stats = _ShiftStats(
        raw_min=np.iinfo(np.int16).max,
        raw_max=np.iinfo(np.int16).min,
        pyr_min=np.iinfo(np.int16).max,
        pyr_max=np.iinfo(np.int16).min,
    )

    def _read_source(t0: int, t1: int) -> npt.NDArray[Any]:
        return _read_channels_first(source, t0, t1)

    read_fn = _retry_on_oserror(_read_source, source_retries)

    for t0, t1, chunk in _chunk_iter(read_fn, n_samples, shard_samples, prefetch_chunks):
        mm = _compute_minmax_block(chunk, base_stride)

        if raw_array is not None:
            raw_chunk = chunk
            if lsb_shift_raw > 0:
                raw_chunk = _shift(chunk, lsb_shift_raw)
                shift_stats.add_raw(raw_chunk)
            raw_array[:, t0:t1] = raw_chunk

        mm_out = mm
        if lsb_shift_pyramid > 0:
            mm_out = _shift(mm, lsb_shift_pyramid)
            shift_stats.add_pyramid(mm_out)
        b0 = t0 // base_stride
        finest_array[:, :, b0 : b0 + mm_out.shape[2]] = mm_out

        # mm extremes double as raw extremes (see _AccumulatedStats).
        if stats is not None:
            stats.update_minmax(mm)

        if progress is not None:
            progress(t1, n_samples)

    return shift_stats


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


def _range_fit_array(
    root: zarr.Group,
    path: str,
    encode: Callable[[npt.NDArray[Any]], npt.NDArray[Any]],
    *,
    dtype: str,
    chunks: tuple[int, ...],
    shards: tuple[int, ...],
    compressor: str | BytesBytesCodec,
    attrs: dict[str, Any],
) -> zarr.Array:
    """Re-encode the array at *path* with *encode*, one shard at a time, and return it."""
    old: zarr.Array = root[path]  # type: ignore[assignment]
    tmp_path = path + "__range_tmp"
    tmp = _create_zarr_array(
        root,
        tmp_path,
        shape=old.shape,
        dtype=dtype,
        chunks=chunks,
        shards=shards,
        compressor=compressor,
        quantized=True,
    )
    last_axis = len(old.shape) - 1
    lead = (slice(None),) * last_axis
    for i0 in range(0, old.shape[last_axis], shards[last_axis]):
        sl = (*lead, slice(i0, min(i0 + shards[last_axis], old.shape[last_axis])))
        tmp[sl] = encode(np.asarray(old[sl]))
    tmp.attrs.update(attrs)
    _replace_array(root, path, tmp_path)
    return root[path]  # type: ignore[return-value]


def _narrow_to_int8(
    root: zarr.Group,
    paths: list[str],
    *,
    label: str,
    s_min: int,
    s_max: int,
    outliers: int,
    total: int,
    clip_outliers: bool,
    chunks: tuple[int, ...],
    shards: tuple[int, ...],
    compressor: str | BytesBytesCodec,
    entropy: float | None,
    is_pyramid: bool,
) -> None:
    """Store shifted arrays as int8 when they fit, or when clipping a rare few makes them fit."""
    clip = False
    if not (s_min >= -128 and s_max <= 127):
        if not (clip_outliers and total > 0 and outliers / total < _CLIP_THRESHOLD):
            log.info(
                "Keeping int16 for %s: %d values (%.4f%%) exceed int8", label, outliers, 100 * outliers / max(total, 1)
            )
            return
        log.info("Clipping %d %s values (%.6f%%) to fit int8", outliers, label, 100 * outliers / total)
        clip = True
    for path in paths:
        _convert_array_to_int8(
            root, path, chunks, shards, compressor, entropy=entropy, is_pyramid=is_pyramid, clip=clip
        )


def _write_encoded(
    source: Any,
    root: zarr.Group,
    *,
    plan: _ResolvedEncoding,
    encoding: EncodingConfig,
    writes_raw: bool,
    source_dtype: str,
    strides: list[int],
    cfg: ChunkConfig,
    stride_factor: int,
    compressor: str | BytesBytesCodec,
    progress: ProgressCallback | None,
    prefetch_chunks: int,
    source_retries: int,
) -> None:
    """Write raw (if *writes_raw*) and every pyramid level of *source*, encoded per *plan*.

    One pass writes raw and the finest level, shifted where the plan shifts;
    range-fit arrays are then re-encoded from the pass's extremes, and coarser
    levels cascade on the encoded values, which every encoding keeps in order.
    """
    n_samples, n_channels = source.shape
    finest_stride = strides[0]
    raw_entropy: float | None = None
    pyr_entropy: float | None = None
    if plan.raw_shift > 0 or plan.pyramid_shift > 0:
        sample = _read_channels_first(source, 0, min(_SAMPLE_SAMPLES, n_samples))
        if plan.raw_shift > 0:
            raw_entropy = compute_entropy(_shift(sample, plan.raw_shift))
        if plan.pyramid_shift > 0:
            pyr_entropy = compute_entropy(_shift(_compute_minmax_block(sample, finest_stride), plan.pyramid_shift))

    raw_array = None
    if writes_raw:
        raw_array = _create_zarr_array(
            root,
            "raw",
            shape=(n_channels, n_samples),
            dtype=source_dtype,
            chunks=cfg.raw_chunk,
            shards=cfg.raw_shard,
            compressor=compressor,
            entropy=raw_entropy,
        )
    finest_path = minmax_path(finest_stride).lstrip("/")
    finest_array = _create_zarr_array(
        root,
        finest_path,
        shape=(n_channels, 2, n_blocks(n_samples, finest_stride)),
        dtype=source_dtype,
        chunks=cfg.minmax_chunk,
        shards=cfg.minmax_shard,
        compressor=compressor,
        entropy=pyr_entropy,
        is_pyramid=True,
    )

    fit_raw = plan.raw == "range" and encoding.raw_quantization is None
    fit_pyramids = plan.pyramids == "range" and encoding.level_quantization is None
    stats = _AccumulatedStats.create(n_channels, source_dtype) if fit_raw or fit_pyramids else None

    shift_stats = _write_raw_and_compute_finest(
        source,
        raw_array,
        finest_array,
        shard_samples=cfg.raw_shard[1],
        base_stride=finest_stride,
        n_samples=n_samples,
        stats=stats,
        progress=progress,
        lsb_shift_raw=plan.raw_shift,
        lsb_shift_pyramid=plan.pyramid_shift,
        prefetch_chunks=prefetch_chunks,
        source_retries=source_retries,
    )

    level_paths = [minmax_path(s).lstrip("/") for s in strides]
    if plan.pyramids == "range":
        lq = encoding.level_quantization
        if lq is None:
            assert stats is not None
            dtype = encoding.pyramid_range_dtype
            min_offset, min_scale = compute_quantization_params(stats.mm_min_min, stats.mm_min_max, dtype)
            max_offset, max_scale = compute_quantization_params(stats.mm_max_min, stats.mm_max_max, dtype)
            lq = LevelQuantization(
                min_offset=min_offset.tolist(),
                max_offset=max_offset.tolist(),
                scale=np.maximum(min_scale, max_scale).tolist(),
                source_dtype=source_dtype,
                quantized_dtype=dtype,
            )
        lq_min, lq_max, lq_scale = np.array(lq.min_offset), np.array(lq.max_offset), np.array(lq.scale)

        def encode_minmax(mm: npt.NDArray[Any]) -> npt.NDArray[Any]:
            return np.stack(
                [
                    quantize_array(mm[:, 0, :], lq_min, lq_scale, lq.quantized_dtype),
                    quantize_array(mm[:, 1, :], lq_max, lq_scale, lq.quantized_dtype),
                ],
                axis=1,
            )

        finest_array = _range_fit_array(
            root,
            finest_path,
            encode_minmax,
            dtype=lq.quantized_dtype,
            chunks=cfg.minmax_chunk,
            shards=cfg.minmax_shard,
            compressor=compressor,
            attrs={"quantization": lq.to_dict()},
        )
        _build_cascade_levels(
            root,
            finest_array,
            strides,
            n_samples=n_samples,
            n_channels=n_channels,
            dtype=lq.quantized_dtype,
            chunk_config=cfg,
            stride_factor=stride_factor,
            compressor=compressor,
            quantized=True,
        )
        for path in level_paths[1:]:
            root[path].attrs.update({"quantization": lq.to_dict()})
    else:
        _build_cascade_levels(
            root,
            finest_array,
            strides,
            n_samples=n_samples,
            n_channels=n_channels,
            dtype=source_dtype,
            chunk_config=cfg,
            stride_factor=stride_factor,
            compressor=compressor,
            entropy=pyr_entropy,
        )
        if plan.pyramid_shift > 0:
            for path in level_paths:
                root[path].attrs.update({"lsb_shift": plan.pyramid_shift, "original_dtype": source_dtype})
            _narrow_to_int8(
                root,
                level_paths,
                label="pyramid",
                s_min=shift_stats.pyr_min,
                s_max=shift_stats.pyr_max,
                outliers=shift_stats.pyr_outliers,
                total=shift_stats.pyr_total,
                clip_outliers=encoding.clip_outliers,
                chunks=cfg.minmax_chunk,
                shards=cfg.minmax_shard,
                compressor=compressor,
                entropy=pyr_entropy,
                is_pyramid=True,
            )

    if not writes_raw:
        return
    if plan.raw == "range":
        rq = encoding.raw_quantization
        if rq is None:
            assert stats is not None
            rq_offset, rq_scale = compute_quantization_params(stats.raw_min, stats.raw_max, encoding.raw_range_dtype)
            rq = RawQuantization(
                offset=rq_offset.tolist(),
                scale=rq_scale.tolist(),
                source_dtype=source_dtype,
                quantized_dtype=encoding.raw_range_dtype,
            )
        rq_offset_arr, rq_scale_arr = np.array(rq.offset), np.array(rq.scale)
        _range_fit_array(
            root,
            "raw",
            lambda chunk: quantize_array(chunk, rq_offset_arr, rq_scale_arr, rq.quantized_dtype),
            dtype=rq.quantized_dtype,
            chunks=cfg.raw_chunk,
            shards=cfg.raw_shard,
            compressor=compressor,
            attrs={"quantization": rq.to_dict()},
        )
    elif plan.raw_shift > 0:
        root["raw"].attrs.update({"lsb_shift": plan.raw_shift, "original_dtype": source_dtype})
        _narrow_to_int8(
            root,
            ["raw"],
            label="raw",
            s_min=shift_stats.raw_min,
            s_max=shift_stats.raw_max,
            outliers=shift_stats.raw_outliers,
            total=shift_stats.raw_total,
            clip_outliers=encoding.clip_outliers,
            chunks=cfg.raw_chunk,
            shards=cfg.raw_shard,
            compressor=compressor,
            entropy=raw_entropy,
            is_pyramid=False,
        )
