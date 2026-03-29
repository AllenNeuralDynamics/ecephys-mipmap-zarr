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
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import zarr

from ecephys_mipmap_zarr._store import _open_store
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
    "WindowResult",
    "open_mipmap",
]

Channels = slice | list[int] | npt.NDArray[np.intp] | None
"""Type alias for channel selection: slice, list of ints, int array, or None (all)."""


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
    """

    data: npt.NDArray[Any]
    stride: int | None
    t0_actual: int
    t1_actual: int

    @property
    def is_raw(self) -> bool:
        """Return ``True`` if this result contains raw sample data."""
        return self.stride is None


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
            self._raw_array: zarr.Array | None = raw_group["raw"]  # type: ignore[assignment]
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
    def source_metadata(self) -> dict[str, Any] | None:
        """Source metadata (probe, annotations), or ``None``."""
        return self._attrs.source_metadata

    def get_window(
        self,
        t0: int,
        t1: int,
        channels: Channels = None,
        num_points: int = 1920,
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
        if num_points <= 0:
            msg = f"num_points must be positive, got {num_points}"
            raise ValueError(msg)

        # Clamp sample range
        t0 = max(0, t0)
        t1 = min(t1, self._attrs.n_samples)
        if t0 >= t1:
            msg = f"Empty window after clamping: t0={t0}, t1={t1}"
            raise ValueError(msg)

        # Normalize channels
        chan_sel: slice | list[int] | npt.NDArray[np.intp] = slice(None) if channels is None else channels

        # Select level
        spp = (t1 - t0) / num_points
        level = select_level(self._attrs.strides, spp)

        if level is None:
            return self._read_raw(t0, t1, chan_sel)
        return self._read_minmax(level, t0, t1, chan_sel)

    def _read_raw(
        self,
        t0: int,
        t1: int,
        chan_sel: slice | list[int] | npt.NDArray[np.intp],
    ) -> WindowResult:
        """Read raw data for the given sample range."""
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
    ) -> WindowResult:
        """Read minmax data for the given sample range at a pyramid level."""
        mm_array = self._minmax_arrays[stride]
        total_blocks = mm_array.shape[2]  # blocks are axis 2 in (C, 2, B)

        block_start = t0 // stride
        block_end = min(math.ceil(t1 / stride), total_blocks)

        t0_actual = block_start * stride
        t1_actual = min(block_end * stride, self._attrs.n_samples)

        data = np.asarray(mm_array[chan_sel, :, block_start:block_end])  # type: ignore[index]

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


def open_mipmap(store_path: str | Path) -> MipmapStore:
    """Open a mipmap Zarr store for reading.

    Handles both co-located stores (raw + pyramids together) and
    pyramid-only stores (opens ``raw_source`` separately for raw
    fallback).

    Parameters
    ----------
    store_path : str or Path
        Path to the mipmap Zarr store.

    Returns
    -------
    MipmapStore
        Ready-to-use store handle.

    Raises
    ------
    KeyError
        If required root attributes are missing.
    """
    root = _open_store(store_path, mode="r")
    attrs = MipmapAttrs.from_dict(dict(root.attrs))

    raw_group: zarr.Group | None = None
    if attrs.raw_source is not None:
        raw_group = _open_store(attrs.raw_source, mode="r")

    return MipmapStore(root, attrs, raw_group=raw_group)
