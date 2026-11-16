"""Read time-windowed data from mipmap Zarr stores.

This module provides :class:`MipmapStore` for reading raw or min/max
pyramid data at the appropriate resolution, and :func:`open_mipmap`
as a convenience for opening a store by path.

Two construction patterns are supported:

**High-level** (convenience)::

    store = open_mipmap("path/to/store.zarr")
    result = store.get_window(t0, t1, num_points=1920)

**Low-level** (user manages zarr groups)::

    root = zarr.open_group("path/to/store.zarr", mode="r")
    attrs = MipmapAttrs.from_dict(dict(root.attrs))
    store = MipmapStore(root, attrs)
    result = store.get_window(t0, t1, num_points=1920)

**Plan/execute** (for caching and deferred I/O)::

    store = open_mipmap("path/to/store.zarr")
    plan = store.plan_window(t0, t1, num_points=1920)
    # plan is frozen and hashable -- usable as a cache key
    result = store.execute_plan(plan, dequantize=False)
    # ... later, dequantize on demand:
    result = store.dequantize(result, plan)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import zarr

from ecephys_mipmap_zarr._store import _default_cache_dir, _find_raw_array, _open_store
from ecephys_mipmap_zarr.metadata import (
    LevelQuantization,
    MipmapAttrs,
    RawQuantization,
    dequantize_array,
    minmax_path,
    select_level,
)

__all__ = [
    "Channels",
    "MipmapStore",
    "ReadPlan",
    "WindowResult",
    "open_mipmap",
]

Channels = slice | list[int] | npt.NDArray[np.intp] | None
"""Type alias for channel selection: slice, list of ints, int array, or None (all)."""


@dataclass(frozen=True, slots=True)
class ReadPlan:
    """Description of a windowed read, without executing I/O.

    Frozen and hashable -- usable as a cache key.

    Parameters
    ----------
    array_path : str
        Zarr array path to read from (``"raw"`` or e.g. ``"minmax/s256"``).
    stride : int or None
        The pyramid stride, or ``None`` when reading raw data.
    t0_actual : int
        First sample index covered by the read (block-aligned for minmax).
    t1_actual : int
        One-past-the-end sample index covered.
    block_start : int or None
        First minmax block index, or ``None`` for raw reads.
    block_end : int or None
        One-past-the-end minmax block index, or ``None`` for raw reads.
    channels : tuple[int, ...] or None
        Normalized channel selection as a hashable tuple, or ``None``
        to read all channels.
    """

    array_path: str
    stride: int | None
    t0_actual: int
    t1_actual: int
    block_start: int | None = None
    block_end: int | None = None
    channels: tuple[int, ...] | None = None


@dataclass(frozen=True, slots=True)
class WindowResult:
    """Result of a windowed read from a mipmap store.

    Parameters
    ----------
    data : np.ndarray
        The read data.  Shape is ``(C, T)`` for raw reads or
        ``(C, 2, B)`` for minmax reads, where axis 1 is ``[min, max]``.
    stride : int or None
        The pyramid stride used, or ``None`` when raw data was returned.
    t0_actual : int
        First sample index covered by the returned data.  For raw reads
        this equals the requested *t0* (after clamping).  For minmax
        reads this is block-aligned: ``block_start * stride``.
    t1_actual : int
        One-past-the-end sample index covered.  For raw reads this
        equals the requested *t1* (after clamping).  For minmax reads
        this is ``block_end * stride``, clamped to *n_samples*.
    dequantized : bool
        Whether the data has been dequantized.  ``True`` (default) when
        the reader applied quantization reversal or LSB-shift
        reconstruction; ``False`` when raw stored values are returned.
    """

    data: npt.NDArray[Any]
    stride: int | None
    t0_actual: int
    t1_actual: int
    dequantized: bool = True

    @property
    def is_raw(self) -> bool:
        """Return ``True`` if this result contains raw sample data."""
        return self.stride is None


def _normalize_channels(channels: Channels, n_channels: int) -> tuple[int, ...] | None:
    """Convert a channel selection to a hashable form.

    Parameters
    ----------
    channels : Channels
        Channel selection: ``None`` (all), a ``slice``, ``list[int]``,
        or ``np.ndarray`` of integer indices.
    n_channels : int
        Total number of channels (used for slice expansion).

    Returns
    -------
    tuple[int, ...] or None
        ``None`` when all channels are selected, otherwise a tuple of
        integer channel indices.
    """
    if channels is None:
        return None
    if isinstance(channels, slice):
        return tuple(range(*channels.indices(n_channels)))
    if isinstance(channels, np.ndarray):
        return tuple(int(c) for c in channels)
    return tuple(channels)


class MipmapStore:
    """Read interface for a mipmap Zarr store.

    Holds open Zarr group handles for the pyramid store and (optionally)
    an external raw store.  Use :func:`open_mipmap` for convenient
    construction, or instantiate directly for low-level control.

    Parameters
    ----------
    root : zarr.Group
        The pyramid store root group (opened in read mode).
    attrs : MipmapAttrs
        Parsed root attributes.
    raw_group : zarr.Group or None, optional
        If the store is pyramid-only (``attrs.raw_source`` is set), this
        is the separately opened raw store.  For co-located stores this
        is ``None`` and the raw array is looked up in *root*.
    """

    def __init__(
        self,
        root: zarr.Group,
        attrs: MipmapAttrs,
        raw_group: zarr.Group | None = None,
    ) -> None:
        self._root = root
        self._attrs = attrs

        # Locate raw array
        if raw_group is not None:
            self._raw_array: zarr.Array | None = _find_raw_array(raw_group, attrs.raw_path or "raw")
        else:
            try:
                self._raw_array = root["raw"]  # type: ignore[assignment]
            except KeyError:
                self._raw_array = None

        # Eagerly look up all minmax arrays
        self._minmax_arrays: dict[int, zarr.Array] = {}
        for stride in attrs.strides:
            path = minmax_path(stride).lstrip("/")
            self._minmax_arrays[stride] = root[path]  # type: ignore[assignment]

        # Detect quantization attrs
        self._raw_quantization: RawQuantization | None = None
        if self._raw_array is not None:
            q_dict = dict(self._raw_array.attrs).get("quantization")
            if q_dict is not None:
                self._raw_quantization = RawQuantization.from_dict(dict(q_dict))  # type: ignore[arg-type]

        self._level_quantization: LevelQuantization | None = None
        self._raw_lsb_shift: int = 0
        self._pyr_lsb_shift: int = 0

        if self._raw_array is not None:
            self._raw_lsb_shift = int(dict(self._raw_array.attrs).get("lsb_shift", 0))  # type: ignore[arg-type]

        if attrs.strides:
            first_mm = self._minmax_arrays[attrs.strides[0]]
            q_dict = dict(first_mm.attrs).get("quantization")
            if q_dict is not None:
                self._level_quantization = LevelQuantization.from_dict(dict(q_dict))  # type: ignore[arg-type]
            self._pyr_lsb_shift = int(dict(first_mm.attrs).get("lsb_shift", 0))  # type: ignore[arg-type]

    @property
    def sampling_frequency(self) -> float:
        """Sampling rate in Hz."""
        return self._attrs.sampling_frequency

    @property
    def n_samples(self) -> int:
        """Total number of time samples."""
        return self._attrs.n_samples

    @property
    def n_channels(self) -> int:
        """Number of channels."""
        return self._attrs.n_channels

    @property
    def t_start(self) -> float:
        """Recording start time in seconds (0 if not available)."""
        return self._attrs.t_start if self._attrs.t_start is not None else 0.0

    @property
    def strides(self) -> list[int]:
        """Pyramid stride sequence (ascending)."""
        return self._attrs.strides

    @property
    def channel_ids(self) -> npt.NDArray[Any] | None:
        """Channel ID labels, or ``None`` if not available."""
        try:
            return np.asarray(self._root["channel_ids"][:])  # type: ignore[index]
        except KeyError:
            return None

    @property
    def channel_locations(self) -> npt.NDArray[Any] | None:
        """Per-channel (x, y) positions, or ``None`` if not available."""
        try:
            return np.asarray(self._root["properties"]["location"][:])  # type: ignore[index]
        except KeyError:
            return None

    @property
    def channel_properties(self) -> dict[str, npt.NDArray[Any]]:
        """Per-channel properties from the source (``location``, ``group``, ...), by name; empty if none."""
        try:
            props = self._root["properties"]
        except KeyError:
            return {}
        if not isinstance(props, zarr.Group):
            return {}
        return {name: np.asarray(node[:]) for name, node in props.arrays()}

    @property
    def source_metadata(self) -> dict[str, Any] | None:
        """Source metadata (probe, annotations), or ``None``."""
        return self._attrs.source_metadata

    @property
    def is_remote(self) -> bool:
        """Whether the underlying store is backed by a remote filesystem."""
        store = self._root.store
        return hasattr(store, "fs")  # FsspecStore has an .fs attribute

    def plan_window(
        self,
        t0: int,
        t1: int,
        channels: Channels = None,
        num_points: int = 1920,
    ) -> ReadPlan:
        """Plan a windowed read without executing I/O.

        Performs the same level-selection logic as :meth:`get_window` but
        returns a frozen :class:`ReadPlan` instead of reading data.  The
        plan is hashable and can be used as a cache key.

        Parameters
        ----------
        t0 : int
            Start sample index (inclusive).
        t1 : int
            End sample index (exclusive).
        channels : slice, list[int], np.ndarray, or None, optional
            Channel selection.  ``None`` (default) selects all channels.
        num_points : int, optional
            Desired number of data points per channel.  Defaults to 1920.

        Returns
        -------
        ReadPlan
            A frozen description of the read to execute.

        Raises
        ------
        ValueError
            If the window is empty after clamping, or *num_points* <= 0.
        RuntimeError
            If raw data is needed but not available.
        """
        if num_points <= 0:
            msg = f"num_points must be positive, got {num_points}"
            raise ValueError(msg)

        # Clamp sample range and coerce to int (callers sometimes pass
        # numpy scalars or floats, which zarr's slice indexer rejects).
        t0 = max(0, int(t0))
        t1 = min(int(t1), int(self._attrs.n_samples))
        if t0 >= t1:
            msg = f"Empty window after clamping: t0={t0}, t1={t1}"
            raise ValueError(msg)

        # Normalize channels
        norm_channels = _normalize_channels(channels, self._attrs.n_channels)

        # Select level
        spp = (t1 - t0) / num_points
        level = select_level(self._attrs.strides, spp)

        if level is None:
            if self._raw_array is None:
                msg = "Raw data not available: store is pyramid-only and raw_source is not reachable"
                raise RuntimeError(msg)
            return ReadPlan(
                array_path="raw",
                stride=None,
                t0_actual=t0,
                t1_actual=t1,
                channels=norm_channels,
            )

        mm_array = self._minmax_arrays[level]
        total_blocks = mm_array.shape[2]  # blocks are axis 2 in (C, 2, B)
        block_start = t0 // level
        block_end = min(math.ceil(t1 / level), total_blocks)
        t0_actual = block_start * level
        t1_actual = min(block_end * level, self._attrs.n_samples)

        return ReadPlan(
            array_path=minmax_path(level).lstrip("/"),
            stride=level,
            t0_actual=t0_actual,
            t1_actual=t1_actual,
            block_start=block_start,
            block_end=block_end,
            channels=norm_channels,
        )

    def execute_plan(self, plan: ReadPlan, dequantize: bool = True) -> WindowResult:
        """Execute a :class:`ReadPlan` and return the data.

        Parameters
        ----------
        plan : ReadPlan
            A plan previously obtained from :meth:`plan_window`.
        dequantize : bool, optional
            If ``True`` (default), apply dequantization or LSB-shift
            reconstruction.  If ``False``, return data in its stored
            dtype with ``dequantized=False`` on the result.

        Returns
        -------
        WindowResult
            The read data.
        """
        # Convert plan.channels back to a channel selector for array indexing
        if plan.channels is None:
            chan_sel: slice | list[int] = slice(None)
        else:
            chan_sel = list(plan.channels)

        if plan.stride is None:
            return self._read_raw(plan.t0_actual, plan.t1_actual, chan_sel, dequantize=dequantize)
        return self._read_minmax(plan.stride, plan.t0_actual, plan.t1_actual, chan_sel, dequantize=dequantize)

    def get_window(
        self,
        t0: int,
        t1: int,
        channels: Channels = None,
        num_points: int = 1920,
        dequantize: bool = True,
    ) -> WindowResult:
        """Read a time window at the appropriate resolution level.

        Estimates samples-per-pixel from the time range and *num_points*,
        picks the coarsest pyramid level that still oversamples, and falls
        back to raw data when zoomed in past the finest pyramid level.

        Parameters
        ----------
        t0 : int
            Start sample index (inclusive).
        t1 : int
            End sample index (exclusive).
        channels : slice, list[int], np.ndarray, or None, optional
            Channel selection.  ``None`` (default) selects all channels.
        num_points : int, optional
            Desired number of data points per channel for this window.
            The returned data will have at least this many points
            (oversampled by ~4x at the chosen pyramid level).  Defaults
            to 1920.
        dequantize : bool, optional
            If ``True`` (default), apply dequantization or LSB-shift
            reconstruction so that the returned data is in the original
            dtype/range.  If ``False``, return data in its stored dtype
            with ``dequantized=False`` on the result.

        Returns
        -------
        WindowResult
            The read data, the stride used (or ``None`` for raw), and the
            actual sample range covered.

        Raises
        ------
        ValueError
            If the window is empty after clamping, or *num_points* <= 0.
        RuntimeError
            If raw data is needed but not available (pyramid-only store
            without a reachable raw source).
        """
        plan = self.plan_window(t0, t1, channels, num_points)
        return self.execute_plan(plan, dequantize=dequantize)

    def dequantize(self, result: WindowResult, plan: ReadPlan) -> WindowResult:
        """Dequantize a :class:`WindowResult` obtained with ``dequantize=False``.

        If the result is already dequantized this is a no-op.

        Parameters
        ----------
        result : WindowResult
            A result previously obtained from :meth:`execute_plan` or
            :meth:`get_window` with ``dequantize=False``.
        plan : ReadPlan
            The plan that produced *result* (needed to select the correct
            per-channel quantization parameters).

        Returns
        -------
        WindowResult
            A new result with dequantized data and ``dequantized=True``.
        """
        if result.dequantized:
            return result

        data = result.data

        # Determine channel selector for subsetting quantization params
        if plan.channels is None:
            chan_sel: slice | list[int] = slice(None)
        else:
            chan_sel = list(plan.channels)

        if result.is_raw:
            if self._raw_quantization is not None:
                offset = np.array(self._raw_quantization.offset)[chan_sel]
                scale = np.array(self._raw_quantization.scale)[chan_sel]
                data = dequantize_array(data, offset, scale)
            elif self._raw_lsb_shift > 0:
                data = data.astype(np.int16) << self._raw_lsb_shift
        else:
            if self._level_quantization is not None:
                min_offset = np.array(self._level_quantization.min_offset)[chan_sel]
                max_offset = np.array(self._level_quantization.max_offset)[chan_sel]
                lq_scale = np.array(self._level_quantization.scale)[chan_sel]
                mins = dequantize_array(data[:, 0, :], min_offset, lq_scale)
                maxs = dequantize_array(data[:, 1, :], max_offset, lq_scale)
                data = np.stack([mins, maxs], axis=1)
            elif self._pyr_lsb_shift > 0:
                data = data.astype(np.int16) << self._pyr_lsb_shift

        return WindowResult(
            data=data,
            stride=result.stride,
            t0_actual=result.t0_actual,
            t1_actual=result.t1_actual,
            dequantized=True,
        )

    def _read_raw(
        self,
        t0: int,
        t1: int,
        chan_sel: slice | list[int] | npt.NDArray[np.intp],
        dequantize: bool = True,
    ) -> WindowResult:
        """Read raw data for the given sample range.

        Parameters
        ----------
        t0 : int
            Start sample index (inclusive).
        t1 : int
            End sample index (exclusive).
        chan_sel : slice, list[int], or np.ndarray
            Channel selector.
        dequantize : bool, optional
            Apply dequantization / LSB-shift reconstruction (default ``True``).
        """
        if self._raw_array is None:
            msg = "Raw data not available: store is pyramid-only and raw_source is not reachable"
            raise RuntimeError(msg)

        # Detect (T, C) vs (C, T) layout
        if self._raw_array.shape[0] > self._raw_array.shape[1]:
            # SI convention (T, C) — transpose to (C, T)
            data = np.ascontiguousarray(
                np.asarray(self._raw_array[t0:t1, chan_sel]).T  # type: ignore[index]
            )
        else:
            # Our convention (C, T)
            data = np.asarray(self._raw_array[chan_sel, t0:t1])  # type: ignore[index]

        if not dequantize:
            return WindowResult(data=data, stride=None, t0_actual=t0, t1_actual=t1, dequantized=False)

        if self._raw_quantization is not None:
            offset = np.array(self._raw_quantization.offset)
            scale = np.array(self._raw_quantization.scale)
            # Select channels from offset/scale to match the channel selection
            if isinstance(chan_sel, (slice, list, np.ndarray)):
                offset = offset[chan_sel]
                scale = scale[chan_sel]
            data = dequantize_array(data, offset, scale)
        elif self._raw_lsb_shift > 0:
            data = data.astype(np.int16) << self._raw_lsb_shift

        return WindowResult(data=data, stride=None, t0_actual=t0, t1_actual=t1)

    def _read_minmax(
        self,
        stride: int,
        t0: int,
        t1: int,
        chan_sel: slice | list[int] | npt.NDArray[np.intp],
        dequantize: bool = True,
    ) -> WindowResult:
        """Read minmax data for the given sample range at a pyramid level.

        Parameters
        ----------
        stride : int
            Pyramid stride.
        t0 : int
            Start sample index (inclusive).
        t1 : int
            End sample index (exclusive).
        chan_sel : slice, list[int], or np.ndarray
            Channel selector.
        dequantize : bool, optional
            Apply dequantization / LSB-shift reconstruction (default ``True``).
        """
        mm_array = self._minmax_arrays[stride]
        total_blocks = mm_array.shape[2]  # blocks are axis 2 in (C, 2, B)

        block_start = t0 // stride
        block_end = min(math.ceil(t1 / stride), total_blocks)

        t0_actual = block_start * stride
        t1_actual = min(block_end * stride, self._attrs.n_samples)

        data = np.asarray(mm_array[chan_sel, :, block_start:block_end])  # type: ignore[index]

        if not dequantize:
            return WindowResult(data=data, stride=stride, t0_actual=t0_actual, t1_actual=t1_actual, dequantized=False)

        if self._level_quantization is not None:
            min_offset = np.array(self._level_quantization.min_offset)
            max_offset = np.array(self._level_quantization.max_offset)
            lq_scale = np.array(self._level_quantization.scale)
            # Select channels
            if isinstance(chan_sel, (list, np.ndarray)):
                min_offset = min_offset[chan_sel]
                max_offset = max_offset[chan_sel]
                lq_scale = lq_scale[chan_sel]
            elif isinstance(chan_sel, slice):
                min_offset = min_offset[chan_sel]
                max_offset = max_offset[chan_sel]
                lq_scale = lq_scale[chan_sel]
            # Dequantize min and max axes separately
            mins = dequantize_array(data[:, 0, :], min_offset, lq_scale)
            maxs = dequantize_array(data[:, 1, :], max_offset, lq_scale)
            data = np.stack([mins, maxs], axis=1)
        elif self._pyr_lsb_shift > 0:
            data = data.astype(np.int16) << self._pyr_lsb_shift

        return WindowResult(data=data, stride=stride, t0_actual=t0_actual, t1_actual=t1_actual)


def open_mipmap(
    store_path: str | Path,
    cache_dir: str | Path | None = None,
    cache_max_gb: float = 10,
) -> MipmapStore:
    """Open a mipmap Zarr store for reading.

    Handles both co-located stores (raw + pyramids together) and
    pyramid-only stores (opens ``raw_source`` separately for raw
    fallback).

    For remote URIs (``s3://``, etc.), an fsspec ``filecache`` layer
    is automatically enabled using the OS-standard cache directory.
    Override with *cache_dir*, or set ``cache_dir=""`` to disable.

    Parameters
    ----------
    store_path : str or Path
        Path to the mipmap Zarr store.
    cache_dir : str, Path, or None, optional
        Local directory for cached remote data.  ``None`` (default)
        auto-detects: uses the OS cache directory for remote URIs,
        disabled for local paths.  Pass ``""`` to explicitly disable.
    cache_max_gb : float, optional
        Maximum cache size in GB.  Oldest files are pruned on open.
        Default 10 GB.

    Returns
    -------
    MipmapStore
        Ready-to-use store handle.

    Raises
    ------
    KeyError
        If required root attributes are missing.
    """
    # Auto-default cache_dir for remote URIs
    if cache_dir is None and "://" in str(store_path):
        cache_dir = _default_cache_dir()
    elif cache_dir == "":
        cache_dir = None  # explicitly disabled

    root = _open_store(store_path, mode="r", cache_dir=cache_dir, cache_max_gb=cache_max_gb)
    attrs = MipmapAttrs.from_dict(dict(root.attrs))

    raw_group: zarr.Group | None = None
    if attrs.raw_source is not None:
        raw_group = _open_store(attrs.raw_source, mode="r", cache_dir=cache_dir, cache_max_gb=cache_max_gb)

    return MipmapStore(root, attrs, raw_group=raw_group)
