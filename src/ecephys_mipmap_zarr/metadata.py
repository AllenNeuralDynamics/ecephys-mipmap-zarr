"""Zarr layout constants, stride computation, and root-attribute schema.

This module is the foundation for the ecephys-mipmap-zarr package. It defines
the multi-resolution pyramid geometry, chunking/sharding defaults, path
conventions, and typed schemas for Zarr root and per-level attributes.

Everything here is pure Python with no I/O; ``build`` and ``read`` both
depend on this module.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

__all__ = [
    "BASE_STRIDE",
    "BuildConfig",
    "ChunkConfig",
    "DEFAULT_BUILD_CONFIG",
    "DEFAULT_CHUNK_CONFIG",
    "DEFAULT_CHANNEL_CHUNK",
    "DEFAULT_DTYPE",
    "DEFAULT_ENCODING_CONFIG",
    "Encoding",
    "EncodingConfig",
    "LOSSLESS_ENCODING",
    "HF_NOISE_BAND_HZ",
    "LevelQuantization",
    "MIN_TOP_BLOCKS",
    "MINMAX_PREFIX",
    "MipmapAttrs",
    "RAW_PATH",
    "RawQuantization",
    "STRIDE_FACTOR",
    "compute_quantization_params",
    "compute_strides",
    "compute_optimal_lsb_shift",
    "lsb_shift_for_noise",
    "dequantize_array",
    "estimate_hf_noise",
    "minmax_path",
    "n_blocks",
    "quantize_array",
    "select_level",
]

# ---------------------------------------------------------------------------
# Pyramid geometry defaults
# ---------------------------------------------------------------------------

BASE_STRIDE: int = 32
"""Base stride (S0) in samples for the finest mipmap level."""

STRIDE_FACTOR: int = 8
"""Multiplicative factor between consecutive pyramid levels."""

MIN_TOP_BLOCKS: int = 1024
"""Minimum number of blocks at the coarsest pyramid level.

The pyramid builder stops adding levels once the *next* level would produce
fewer than this many blocks.  The goal is that the coarsest level fits
comfortably on one screen (~1K--2K points per channel).
"""

# ---------------------------------------------------------------------------
# Chunking / sharding defaults (Zarr v3 sharding codec)
# ---------------------------------------------------------------------------

DEFAULT_CHANNEL_CHUNK: int = 64
"""Default channel-axis chunk width (for both raw and minmax arrays)."""

DEFAULT_DTYPE: str = "int16"
"""Default data type for raw and minmax arrays."""

FORMAT_VERSION: int = 1
"""Store layout version; a reader refuses stores with a higher one."""

COUNTS_UNITS: str = "counts"
"""Units of integer values decoded without an ADC gain."""

# ---------------------------------------------------------------------------
# Zarr path conventions
# ---------------------------------------------------------------------------

RAW_PATH: str = "/raw"
"""Zarr array path for the raw time-series data."""

MINMAX_PREFIX: str = "/minmax/s"
"""Zarr group prefix for minmax pyramid arrays (append stride value)."""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ChunkConfig:
    """Chunk and shard shape configuration for raw and minmax arrays.

    Parameters
    ----------
    raw_chunk : tuple[int, int]
        Inner-chunk shape for the raw ``(C, T)`` array.
    raw_shard : tuple[int, int]
        Shard shape for the raw ``(C, T)`` array.
    minmax_chunk : tuple[int, int, int]
        Inner-chunk shape for minmax ``(C, 2, B)`` arrays.
    minmax_shard : tuple[int, int, int]
        Shard shape for minmax ``(C, 2, B)`` arrays.
    """

    raw_chunk: tuple[int, int] = (64, 16384)
    raw_shard: tuple[int, int] = (64, 1_048_576)
    minmax_chunk: tuple[int, int, int] = (64, 2, 4096)
    minmax_shard: tuple[int, int, int] = (64, 2, 32768)

    def for_channels(self, n_channels: int) -> ChunkConfig:
        """Return a new config with channel dims clamped to *n_channels*.

        When the actual channel count is smaller than the default chunk width,
        the channel dimension is reduced to ``min(current, n_channels)``.
        When larger, the defaults are kept (Zarr handles partial edge chunks).

        Parameters
        ----------
        n_channels : int
            Actual number of channels in the data.

        Returns
        -------
        ChunkConfig
            A new instance with adjusted channel dimensions.
        """
        c_raw = min(self.raw_chunk[0], n_channels)
        c_shard = min(self.raw_shard[0], n_channels)
        c_mm = min(self.minmax_chunk[0], n_channels)
        c_mm_shard = min(self.minmax_shard[0], n_channels)
        return replace(
            self,
            raw_chunk=(c_raw, self.raw_chunk[1]),
            raw_shard=(c_shard, self.raw_shard[1]),
            minmax_chunk=(c_mm, self.minmax_chunk[1], self.minmax_chunk[2]),
            minmax_shard=(c_mm_shard, self.minmax_shard[1], self.minmax_shard[2]),
        )


DEFAULT_CHUNK_CONFIG: ChunkConfig = ChunkConfig()
"""Module-level default chunk/shard configuration."""


@dataclass(frozen=True, slots=True)
class BuildConfig:
    """Configuration for pyramid geometry, chunking, and compression.

    Groups the build parameters that are shared identically across
    :func:`~ecephys_mipmap_zarr.build.build_mipmap`,
    :func:`~ecephys_mipmap_zarr.build.add_pyramid`, and
    :func:`~ecephys_mipmap_zarr.build.ensure_pyramid`.

    Parameters
    ----------
    chunk_config : ChunkConfig
        Chunk/shard shape configuration.
    base_stride : int
        Base stride for the finest pyramid level.
    stride_factor : int
        Multiplicative factor between levels.
    min_top_blocks : int
        Minimum blocks at the coarsest level.
    compressor : str or BytesBytesCodec
        Compressor preset or codec instance.
    consolidate_metadata : bool
        Write consolidated metadata.
    prefetch_chunks : int
        Number of source chunks to prefetch in background threads.
        ``0`` (default) disables prefetch. Set to 2-3 for S3 or WavPack
        sources where read latency is high.
    source_retries : int
        Number of times to retry a source chunk read on ``OSError`` with
        exponential backoff (1s, 2s, 4s, ...).  Remote-backed sources
        (FUSE-mounted S3, fsspec) emit transient EIO errors under
        network/rate-limit pressure; retries turn those into logged
        warnings instead of a dead build.  Default 5.  Set to 0 to
        disable (tests, debugging).
    """

    chunk_config: ChunkConfig = DEFAULT_CHUNK_CONFIG
    base_stride: int = BASE_STRIDE
    stride_factor: int = STRIDE_FACTOR
    min_top_blocks: int = MIN_TOP_BLOCKS
    compressor: str | Any = "blosc"  # str | BytesBytesCodec at runtime
    consolidate_metadata: bool = True
    prefetch_chunks: int = 0
    source_retries: int = 5


DEFAULT_BUILD_CONFIG: BuildConfig = BuildConfig()
"""Module-level default build configuration."""


Encoding = Literal["shift", "range", "none"]
"""How an array's values are stored.

- ``"shift"``: round to a step of ``2**bits`` and store signed, int8 when the
  range fits. Integer sources only.
- ``"range"``: map each channel's min..max linearly onto an unsigned dtype.
- ``"none"``: store the source values unchanged.
"""


@dataclass(frozen=True, slots=True)
class EncodingConfig:
    """How :func:`~ecephys_mipmap_zarr.build.build_mipmap` encodes raw and pyramids.

    Each array picks its encoding independently. A setting that the chosen
    encodings would not use is an error rather than ignored.

    Parameters
    ----------
    raw, pyramids : {"shift", "range", "none"} or None
        Encoding per array. ``None`` picks ``"shift"`` for integer sources and
        ``"range"`` for float ones.
    raw_shift, pyramid_shift : int or "auto"
        Bits dropped by ``"shift"``. ``"auto"`` takes the largest step whose
        rounding noise stays below half the noise floor; pyramids, which are
        only drawn, get 2 bits more than that.
    noise_floor : float or None
        Noise floor in source units RMS for ``"auto"``. ``None`` measures it in
        the 10-14 kHz band, which needs a sampling rate above 28 kHz.
    clip_outliers : bool
        After shifting, clip values outside int8 when they are under 0.01% of
        the array, rather than keeping int16.
    raw_range_dtype, pyramid_range_dtype : str
        Unsigned dtype for ``"range"``.
    raw_quantization : RawQuantization or None
        Fixed ``"range"`` parameters for raw instead of the source's extremes.
    level_quantization : LevelQuantization or None
        Fixed ``"range"`` parameters for the pyramids.
    """

    raw: Encoding | None = None
    pyramids: Encoding | None = None
    raw_shift: int | Literal["auto"] = "auto"
    pyramid_shift: int | Literal["auto"] = "auto"
    noise_floor: float | None = None
    clip_outliers: bool = True
    raw_range_dtype: str = "uint16"
    pyramid_range_dtype: str = "uint8"
    # RawQuantization / LevelQuantization, defined below
    raw_quantization: Any = None
    level_quantization: Any = None


DEFAULT_ENCODING_CONFIG: EncodingConfig = EncodingConfig()
"""Module-level default encoding: shift for integer sources, range for float."""

LOSSLESS_ENCODING: EncodingConfig = EncodingConfig(raw="none", pyramids="none")
"""Store raw and pyramids with the source's values unchanged."""


@dataclass(frozen=True, slots=True)
class MipmapAttrs:
    """Typed schema for Zarr root-group attributes.

    Parameters
    ----------
    sampling_frequency : float
        Sampling rate in Hz.
    units : str
        Units the stored values decode to: ``"uV"`` when an ADC gain is composed into the
        encoding, ``"counts"`` for integer values without one, or what the builder was told.
    strides : list[int]
        Computed stride sequence, ascending (e.g. ``[32, 256, 2048]``).
    n_samples : int
        Total number of time samples in the raw array.
    n_channels : int
        Number of channels.
    dtype : str
        NumPy dtype string for the raw data (e.g. ``"int16"``, ``"float32"``).
        Pyramid levels default to this dtype unless quantization is active.
    raw_source : str or None
        Path or URI to the external store containing ``/raw``, or ``None``
        when raw data is co-located in the same store as the pyramids.
    raw_path : str or None
        Path of the raw array inside ``raw_source``.  ``None`` means probe the
        usual names (``raw``, ``traces_seg0``).
    source_metadata : dict or None
        Optional metadata copied from the source store (e.g. SpikeInterface
        probe geometry, annotations).  ``None`` if no source metadata.
    t_start : float or None
        Start time of the recording in seconds (from the acquisition
        clock or ``times_seg0[0]`` in an SI zarr).  ``None`` if not
        available.  Used by the viewer to offset the time axis.
    format_version : int
        Store layout version (:data:`FORMAT_VERSION`); stores written before
        it was recorded read as version 1.
    """

    sampling_frequency: float
    units: str
    strides: list[int]
    n_samples: int
    n_channels: int
    dtype: str = DEFAULT_DTYPE
    raw_source: str | None = None
    raw_path: str | None = None
    source_metadata: dict[str, Any] | None = None
    t_start: float | None = None
    format_version: int = FORMAT_VERSION

    def __post_init__(self) -> None:
        """Coerce numpy scalars / dtype objects to plain Python types.

        Why: callers frequently hand us numpy scalars (e.g. SpikeInterface's
        ``get_sampling_frequency()`` returns ``np.float64``, ``rec.get_dtype()``
        returns a ``np.dtype``).  Stored as-is they survive into
        :meth:`to_dict` → ``zarr.Group.attrs`` where the JSON encoder may
        warn or refuse.  Coercing at construction keeps the write path clean
        and matches what :meth:`from_dict` already does on read.
        """
        object.__setattr__(self, "sampling_frequency", float(self.sampling_frequency))
        object.__setattr__(self, "units", str(self.units))
        object.__setattr__(self, "strides", [int(s) for s in self.strides])
        object.__setattr__(self, "n_samples", int(self.n_samples))
        object.__setattr__(self, "n_channels", int(self.n_channels))
        object.__setattr__(self, "dtype", str(self.dtype))
        if self.raw_source is not None:
            object.__setattr__(self, "raw_source", str(self.raw_source))
        if self.raw_path is not None:
            object.__setattr__(self, "raw_path", str(self.raw_path))
        if self.t_start is not None:
            object.__setattr__(self, "t_start", float(self.t_start))
        object.__setattr__(self, "format_version", int(self.format_version))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict suitable for ``zarr.Group.attrs``.

        Returns
        -------
        dict[str, Any]
            JSON-serializable dictionary.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MipmapAttrs:
        """Construct from a dict (e.g. read back from Zarr root attrs).

        Parameters
        ----------
        d : dict[str, Any]
            Dictionary with keys matching the dataclass fields.

        Returns
        -------
        MipmapAttrs

        Raises
        ------
        KeyError
            If required keys are missing.
        """
        raw_source = d.get("raw_source")
        raw_path = d.get("raw_path")
        source_metadata = d.get("source_metadata")
        t_start = d.get("t_start")
        return cls(
            sampling_frequency=float(d["sampling_frequency"]),
            units=str(d["units"]),
            strides=list(d["strides"]),
            n_samples=int(d["n_samples"]),
            n_channels=int(d["n_channels"]),
            dtype=str(d.get("dtype", DEFAULT_DTYPE)),
            raw_source=str(raw_source) if raw_source is not None else None,
            raw_path=str(raw_path) if raw_path is not None else None,
            source_metadata=dict(source_metadata) if source_metadata is not None else None,
            t_start=float(t_start) if t_start is not None else None,
            format_version=int(d.get("format_version", 1)),
        )


@dataclass(frozen=True, slots=True)
class RawQuantization:
    """Quantization parameters for the raw array.

    Per-channel offset and scale allow quantizing float32 data to a smaller
    unsigned integer type while keeping quantization noise below the signal
    noise floor.

    To dequantize::

        value = quantized_value * scale + offset

    Parameters
    ----------
    offset : list[float]
        Per-channel additive offsets (length *n_channels*).
    scale : list[float]
        Per-channel step sizes (length *n_channels*).
    source_dtype : str
        Original dtype before quantization (e.g. ``"float32"``).
    quantized_dtype : str
        Stored dtype after quantization (e.g. ``"uint16"``).
    """

    offset: list[float]
    scale: list[float]
    source_dtype: str
    quantized_dtype: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict suitable for Zarr array attrs.

        Returns
        -------
        dict[str, Any]
            JSON-serializable dictionary.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RawQuantization:
        """Construct from a dict (e.g. read back from Zarr array attrs).

        Parameters
        ----------
        d : dict[str, Any]
            Dictionary with keys matching the dataclass fields.

        Returns
        -------
        RawQuantization

        Raises
        ------
        KeyError
            If required keys are missing.
        """
        return cls(
            offset=[float(x) for x in d["offset"]],
            scale=[float(x) for x in d["scale"]],
            source_dtype=str(d["source_dtype"]),
            quantized_dtype=str(d["quantized_dtype"]),
        )


@dataclass(frozen=True, slots=True)
class LevelQuantization:
    """Quantization parameters for pyramid minmax arrays.

    Per-channel offsets for min and max axes (they cluster in different
    ranges), with per-channel scale.  Shared across all pyramid levels —
    computed from the finest level (s32) and used unchanged for coarser
    levels since min/max cascading is order-preserving.

    To dequantize::

        value = quantized_value * scale + offset

    Parameters
    ----------
    min_offset : list[float]
        Per-channel additive offsets for the min axis (length *n_channels*).
    max_offset : list[float]
        Per-channel additive offsets for the max axis (length *n_channels*).
    scale : list[float]
        Per-channel step sizes (length *n_channels*).
    source_dtype : str
        Original dtype before quantization (e.g. ``"float32"``).
    quantized_dtype : str
        Stored dtype after quantization (e.g. ``"uint8"``).
    """

    min_offset: list[float]
    max_offset: list[float]
    scale: list[float]
    source_dtype: str
    quantized_dtype: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict suitable for Zarr array attrs.

        Returns
        -------
        dict[str, Any]
            JSON-serializable dictionary.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LevelQuantization:
        """Construct from a dict (e.g. read back from Zarr level attrs).

        Parameters
        ----------
        d : dict[str, Any]
            Dictionary with keys matching the dataclass fields.

        Returns
        -------
        LevelQuantization

        Raises
        ------
        KeyError
            If required keys are missing.
        """
        return cls(
            min_offset=[float(x) for x in d["min_offset"]],
            max_offset=[float(x) for x in d["max_offset"]],
            scale=[float(x) for x in d["scale"]],
            source_dtype=str(d["source_dtype"]),
            quantized_dtype=str(d["quantized_dtype"]),
        )


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def compute_strides(
    n_samples: int,
    base: int = BASE_STRIDE,
    factor: int = STRIDE_FACTOR,
    min_top_blocks: int = MIN_TOP_BLOCKS,
) -> list[int]:
    """Compute the pyramid stride sequence for a given recording length.

    Starting from *base*, each subsequent stride is multiplied by *factor*.
    Levels are added as long as the number of blocks at that stride is at
    least *min_top_blocks*.

    Parameters
    ----------
    n_samples : int
        Total number of time samples.
    base : int, optional
        Base stride (finest level), by default :data:`BASE_STRIDE` (32).
    factor : int, optional
        Multiplicative factor between levels, by default
        :data:`STRIDE_FACTOR` (8).
    min_top_blocks : int, optional
        Stop adding levels when the number of blocks would fall below this
        threshold, by default :data:`MIN_TOP_BLOCKS` (1024).

    Returns
    -------
    list[int]
        Ascending stride sequence.  May be empty if *n_samples* is too small
        even for a single level.

    Raises
    ------
    ValueError
        If *n_samples*, *base*, or *min_top_blocks* is not positive, or
        *factor* is not greater than 1.

    Examples
    --------
    >>> compute_strides(162_000_000)
    [32, 256, 2048, 16384, 131072]
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if base <= 0:
        raise ValueError(f"base must be positive, got {base}")
    if factor <= 1:
        raise ValueError(f"factor must be > 1, got {factor}")
    if min_top_blocks <= 0:
        raise ValueError(f"min_top_blocks must be positive, got {min_top_blocks}")

    strides: list[int] = []
    stride = base
    while n_blocks(n_samples, stride) >= min_top_blocks:
        strides.append(stride)
        stride *= factor
    return strides


def n_blocks(n_samples: int, stride: int) -> int:
    """Compute the number of blocks for a given stride.

    Parameters
    ----------
    n_samples : int
        Total number of time samples.
    stride : int
        Block stride in samples.

    Returns
    -------
    int
        ``ceil(n_samples / stride)``.

    Raises
    ------
    ValueError
        If *stride* is not positive.
    """
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    return math.ceil(n_samples / stride)


def minmax_path(stride: int) -> str:
    """Return the Zarr array path for a minmax pyramid level.

    Parameters
    ----------
    stride : int
        The stride value for this level.

    Returns
    -------
    str
        Path string, e.g. ``"/minmax/s256"``.
    """
    return f"{MINMAX_PREFIX}{stride}"


def select_level(
    strides: list[int],
    samples_per_pixel: float,
    oversample: float = 2.0,
) -> int | None:
    """Choose the best pyramid level for a given zoom.

    Picks the coarsest stride that still oversamples by at least *oversample*
    relative to the display resolution.  Returns ``None`` when raw data should
    be used (zoomed in past the finest pyramid level).

    Parameters
    ----------
    strides : list[int]
        The stride sequence (ascending), as stored in
        :attr:`MipmapAttrs.strides`.
    samples_per_pixel : float
        Number of raw samples represented by one screen pixel.
    oversample : float, optional
        Minimum oversampling factor.  A stride is acceptable if
        ``samples_per_pixel / stride >= oversample``, meaning each pixel
        has at least *oversample* mipmap blocks contributing.  Default 4.0.

    Returns
    -------
    int or None
        The chosen stride, or ``None`` if raw data is the best option.

    Examples
    --------
    >>> select_level([32, 256, 2048], samples_per_pixel=10000.0)
    2048
    >>> select_level([32, 256, 2048], samples_per_pixel=10.0)
    """
    chosen: int | None = None
    for stride in strides:
        if samples_per_pixel / stride >= oversample:
            chosen = stride
        else:
            break
    return chosen


def compute_quantization_params(
    channel_mins: npt.NDArray[Any],
    channel_maxs: npt.NDArray[Any],
    quantized_dtype: str,
    headroom: float = 0.01,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Compute per-channel offset and scale for quantization.

    Parameters
    ----------
    channel_mins : np.ndarray
        Per-channel minimum values, shape ``(C,)``.
    channel_maxs : np.ndarray
        Per-channel maximum values, shape ``(C,)``.
    quantized_dtype : str
        Target unsigned integer dtype (e.g. ``"uint16"``, ``"uint8"``).
    headroom : float, optional
        Fractional headroom added to each side of the range to prevent
        clipping from rounding, by default 0.01 (1%).

    Returns
    -------
    offset : np.ndarray
        Per-channel offsets, shape ``(C,)``.
    scale : np.ndarray
        Per-channel step sizes, shape ``(C,)``.
    """
    dtype_info = np.iinfo(np.dtype(quantized_dtype))
    dtype_range = float(dtype_info.max - dtype_info.min)

    data_range = channel_maxs - channel_mins
    headroom_amount = data_range * headroom

    offset = channel_mins - headroom_amount
    scale = (data_range + 2.0 * headroom_amount) / dtype_range

    # Prevent zero scale for constant channels
    scale = np.where(scale == 0.0, 1.0, scale)

    return offset.astype(np.float64), scale.astype(np.float64)


def quantize_array(
    data: npt.NDArray[Any],
    offset: npt.NDArray[Any],
    scale: npt.NDArray[Any],
    quantized_dtype: str,
) -> npt.NDArray[Any]:
    """Quantize a float array to an unsigned integer type.

    Applies ``stored = clip(round((data - offset) / scale), 0, max_val)``.

    Parameters
    ----------
    data : np.ndarray
        Input array.  The first dimension must match the length of
        *offset* and *scale* (channel-first layout).
    offset : np.ndarray
        Per-channel offsets, shape ``(C,)``.
    scale : np.ndarray
        Per-channel step sizes, shape ``(C,)``.
    quantized_dtype : str
        Target unsigned integer dtype (e.g. ``"uint16"``, ``"uint8"``).

    Returns
    -------
    np.ndarray
        Quantized array with dtype *quantized_dtype*.
    """
    dtype_info = np.iinfo(np.dtype(quantized_dtype))
    # Reshape offset/scale for broadcasting: (C,) -> (C, 1) or (C, 1, 1)
    extra = data.ndim - 1
    off = offset.reshape(offset.shape + (1,) * extra)
    sc = scale.reshape(scale.shape + (1,) * extra)
    result = np.round((data - off) / sc)
    np.clip(result, dtype_info.min, dtype_info.max, out=result)
    return np.asarray(result, dtype=quantized_dtype)


def dequantize_array(
    data: npt.NDArray[Any],
    offset: npt.NDArray[Any],
    scale: npt.NDArray[Any],
) -> npt.NDArray[Any]:
    """Dequantize an unsigned integer array back to float32.

    Applies ``value = data * scale + offset``.

    Parameters
    ----------
    data : np.ndarray
        Quantized input array.  The first dimension must match the length
        of *offset* and *scale* (channel-first layout).
    offset : np.ndarray
        Per-channel offsets, shape ``(C,)``.
    scale : np.ndarray
        Per-channel step sizes, shape ``(C,)``.

    Returns
    -------
    np.ndarray
        Dequantized array with dtype float32.
    """
    # Reshape offset/scale for broadcasting: (C,) -> (C, 1) or (C, 1, 1)
    extra = data.ndim - 1
    sc = np.asarray(scale, dtype=np.float32).reshape(scale.shape + (1,) * extra)
    off = np.asarray(offset, dtype=np.float32).reshape(offset.shape + (1,) * extra)
    result: npt.NDArray[Any] = np.asarray(
        np.asarray(data, dtype=np.float32) * sc + off,
        dtype=np.float32,
    )
    return result


HF_NOISE_BAND_HZ: tuple[float, float] = (10_000.0, 14_000.0)
"""Band where :func:`estimate_hf_noise` measures noise: above spike power, below 30 kHz Nyquist."""


def estimate_hf_noise(data: npt.NDArray[Any], fs: float = 30000.0) -> float:
    """Estimate the analog noise floor from the high-frequency spectrum.

    Computes the power spectral density in the 10--14 kHz band (where
    neural spike content is negligible) and extrapolates to the full
    bandwidth.  Returns the **median** across channels, which is robust
    to dead or broken channels.

    Parameters
    ----------
    data : np.ndarray
        Input data with shape ``(C, T)`` (channel-first).  At least
        10 000 samples recommended for a stable estimate.
    fs : float, optional
        Sampling rate in Hz, by default 30000.

    Returns
    -------
    float
        Estimated noise floor in ADC RMS units.

    Notes
    -----
    This is a **conservative lower bound** on the true noise.  Real
    recording noise includes 1/f components that make low-frequency
    noise higher than the HF estimate.  The estimate is more accurate
    for high-pass filtered data.
    """
    n_channels, n_samples = data.shape
    freqs = np.fft.rfftfreq(n_samples, 1.0 / fs)
    hf_mask = (freqs >= HF_NOISE_BAND_HZ[0]) & (freqs < HF_NOISE_BAND_HZ[1])

    if not np.any(hf_mask):
        return 0.0  # Not enough frequency resolution

    full_bw = min(fs / 2, 15000) - 300  # approximate usable bandwidth
    per_channel_noise = np.empty(n_channels, dtype=np.float64)

    for ch in range(n_channels):
        psd = np.abs(np.fft.rfft(data[ch, :].astype(np.float64))) ** 2 / n_samples
        hf_psd = float(np.mean(psd[hf_mask]))
        per_channel_noise[ch] = math.sqrt(hf_psd * full_bw * 2.0 / fs)

    return float(np.median(per_channel_noise))


def compute_optimal_lsb_shift(
    data: npt.NDArray[Any],
    fs: float = 30000.0,
    safety_factor: float = 2.0,
    noise_floor: float | None = None,
) -> int:
    """Compute the maximum safe LSB right-shift for lossy compression.

    The shift is chosen so that the quantization noise introduced by
    dropping the lowest *N* bits remains below the analog noise floor
    divided by *safety_factor*.

    Parameters
    ----------
    data : np.ndarray
        Input data with shape ``(C, T)`` (channel-first).
    fs : float, optional
        Sampling rate in Hz, by default 30000.
    safety_factor : float, optional
        Quantization noise must be below ``noise / safety_factor``.
        Default 2.0 (< 0.5 dB added noise).
    noise_floor : float or None, optional
        Explicit noise floor in ADC RMS.  If ``None``, estimated from
        the data via :func:`estimate_hf_noise`.

    Returns
    -------
    int
        Optimal right-shift (0 means no shift / lossless).
    """
    if noise_floor is None:
        noise_floor = estimate_hf_noise(data, fs)
    return lsb_shift_for_noise(noise_floor, safety_factor)


def lsb_shift_for_noise(noise_floor: float, safety_factor: float = 2.0) -> int:
    """Largest right-shift whose rounding noise stays below ``noise_floor / safety_factor``.

    Parameters
    ----------
    noise_floor : float
        Noise RMS, in the units being shifted.
    safety_factor : float, optional
        Default 2.0 (< 0.5 dB added noise).

    Returns
    -------
    int
        Bits to shift off; 0 when even one bit would be too coarse.
    """
    if noise_floor <= 0:
        return 0

    # quant_noise = 2^shift / sqrt(12)
    # Require: quant_noise < noise_floor / safety_factor
    # 2^shift < noise_floor * sqrt(12) / safety_factor
    max_step = noise_floor * math.sqrt(12) / safety_factor
    if max_step < 1:
        return 0
    return int(math.floor(math.log2(max_step)))


def compute_entropy(data: npt.NDArray[Any]) -> float:
    """Compute the median per-channel Shannon entropy in bits per sample.

    Parameters
    ----------
    data : np.ndarray
        Input data with shape ``(C, T)`` or ``(C, 2, B)`` (channel-first).

    Returns
    -------
    float
        Median entropy across channels.
    """
    n_channels = data.shape[0]
    # Flatten non-channel axes
    flat = data.reshape(n_channels, -1)

    entropies = np.empty(n_channels, dtype=np.float64)
    for ch in range(n_channels):
        _, counts = np.unique(flat[ch], return_counts=True)
        probs = counts / counts.sum()
        entropies[ch] = float(-np.sum(probs * np.log2(probs)))

    return float(np.median(entropies))
