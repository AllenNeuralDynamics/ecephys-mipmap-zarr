"""Per-channel display statistics, stored in a store's root attrs so a viewer can scale traces before reading samples.

A viewer measuring them itself must read the whole coarsest level and a raw chunk for every channel before its first
draw. The rules here are the web viewer's (``viewer-web/src/render/renderer.ts``), so a stored value matches what it
would measure.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import zarr

from .read import MipmapStore, open_mipmap

__all__ = [
    "EXCURSION_QUANTILE",
    "STATS_ATTR",
    "ChannelStats",
    "clipped_stats",
    "compute_channel_stats",
    "write_channel_stats",
]

#: Root attribute holding :class:`ChannelStats`.
STATS_ATTR = "channel_stats"
#: Quantile of a channel's excursion from its centre, across the coarsest level's blocks.
EXCURSION_QUANTILE = 0.95
_CLIP_SD = 3.0
_CLIP_PASSES = 5


@dataclass(frozen=True, slots=True)
class ChannelStats:
    """Per-channel statistics in decoded units, one entry per channel.

    Parameters
    ----------
    centre : list[float]
        Value traces are centred on: the clipped mean of a raw chunk, else the median block midpoint of the coarsest
        level, else 0.
    noise_sd : list[float]
        Clipped standard deviation of the same raw chunk; 0 when there is no raw array.
    excursion : list[float or None]
        :data:`EXCURSION_QUANTILE` quantile of the coarsest level's distance from `centre`; None when no block is
        finite.
    samples : tuple[int, int]
        Raw sample range `noise_sd` was measured over, half-open; ``(0, 0)`` without raw.
    """

    centre: list[float]
    noise_sd: list[float]
    excursion: list[float | None]
    samples: tuple[int, int]

    def to_dict(self) -> dict[str, Any]:
        """Serialize for root attrs; JSON has no NaN, so an unknown excursion is null."""
        return {
            "centre": self.centre,
            "noise_sd": self.noise_sd,
            "excursion": self.excursion,
            "samples": list(self.samples),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ChannelStats:
        """Parse the root attr written by :meth:`to_dict`."""
        start, stop = d["samples"]
        return cls(
            centre=[float(v) for v in d["centre"]],
            noise_sd=[float(v) for v in d["noise_sd"]],
            excursion=[None if v is None else float(v) for v in d["excursion"]],
            samples=(int(start), int(stop)),
        )


def clipped_stats(values: npt.NDArray[np.floating[Any]]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Mean and sd of each row after repeatedly dropping values beyond 3 sd, so spikes do not count.

    Unlike a median absolute deviation, the sd does not snap to the step of coarsely quantized values. A row with
    fewer than two values left is NaN.
    """
    x = np.asarray(values, dtype=np.float64)
    lo = np.full(x.shape[0], -np.inf)
    hi = np.full(x.shape[0], np.inf)
    mean = np.full(x.shape[0], np.nan)
    sd = np.full(x.shape[0], np.nan)
    for _ in range(_CLIP_PASSES):
        keep = (x >= lo[:, None]) & (x <= hi[:, None])
        n = keep.sum(axis=1)
        kept = np.where(keep, x, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = kept.sum(axis=1) / n
            sd = np.sqrt(np.maximum(0.0, (kept * kept).sum(axis=1) / n - mean * mean))
        few = n < 2
        mean[few] = np.nan
        sd[few] = np.nan
        lo = mean - _CLIP_SD * sd
        hi = mean + _CLIP_SD * sd
    return mean, sd


def compute_channel_stats(store: MipmapStore) -> ChannelStats:
    """Measure :class:`ChannelStats` from the raw chunk holding the middle sample and the whole coarsest level."""
    n_channels = store.n_channels
    baseline = np.full(n_channels, np.nan)
    noise_sd = np.zeros(n_channels)
    samples = (0, 0)
    raw = store._raw_array
    if raw is not None:
        # A pyramid-only store's raw may be SpikeInterface's (T, C), told apart by shape as the reader does.
        chunk = int(raw.chunks[0] if raw.shape[0] > raw.shape[1] else raw.chunks[1])
        start = (store.n_samples // 2) // chunk * chunk
        samples = (start, min(start + chunk, store.n_samples))
        values = store._read_raw(*samples, slice(None)).data
        baseline, sd = clipped_stats(values)
        noise_sd = np.where(np.isfinite(sd), sd, 0.0)

    coarsest = max(store.strides)
    level = store._read_minmax(coarsest, 0, store.n_samples, slice(None)).data.astype(np.float64)
    lows, highs = level[:, 0, :], level[:, 1, :]
    with np.errstate(invalid="ignore"):
        midpoint = np.nanmedian(0.5 * (lows + highs), axis=1) if lows.shape[1] else np.full(n_channels, np.nan)
    centre = np.where(np.isfinite(baseline), baseline, midpoint)
    distance = np.maximum(highs - centre[:, None], centre[:, None] - lows)
    excursion: list[float | None] = []
    for row in distance:
        finite = row[np.isfinite(row)]
        excursion.append(float(np.quantile(finite, EXCURSION_QUANTILE)) if finite.size else None)
    centre = np.where(np.isfinite(centre) & (centre != 0), centre, 0.0)
    return ChannelStats(
        centre=[float(v) for v in centre],
        noise_sd=[float(v) for v in noise_sd],
        excursion=excursion,
        samples=samples,
    )


def write_channel_stats(path: str | Path, *, consolidate: bool = True) -> ChannelStats:
    """Measure a local store's :class:`ChannelStats` and write them to its root attrs.

    Parameters
    ----------
    path : str or Path
        A local mipmap store; a pyramid-only store reads raw from its ``raw_source``.
    consolidate : bool, optional
        Rewrite consolidated metadata afterwards, so readers using it see the new attr.

    Returns
    -------
    ChannelStats
        What was written.
    """
    stats = compute_channel_stats(open_mipmap(str(path)))
    root = zarr.open_group(str(path), mode="r+", use_consolidated=False)
    root.attrs[STATS_ATTR] = stats.to_dict()
    if consolidate:
        zarr.consolidate_metadata(zarr.storage.LocalStore(str(path)))
    return stats
