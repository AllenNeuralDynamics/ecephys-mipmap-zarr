"""Tests for stored per-channel display statistics."""

from pathlib import Path

import numpy as np
import pytest
import zarr

from ecephys_mipmap_zarr.build import add_pyramid, build_mipmap
from ecephys_mipmap_zarr.cli import AddStatsSettings
from ecephys_mipmap_zarr.metadata import LOSSLESS_ENCODING, BuildConfig, ChunkConfig
from ecephys_mipmap_zarr.read import open_mipmap
from ecephys_mipmap_zarr.stats import STATS_ATTR, ChannelStats, clipped_stats, write_channel_stats

N_SAMPLES, N_CHANNELS = 100_000, 4
CHUNKS = ChunkConfig(raw_chunk=(4, 1024), raw_shard=(4, 4096), minmax_chunk=(4, 2, 256), minmax_shard=(4, 2, 1024))


def _source() -> np.ndarray:
    """(T, C) noise of sd 10, 20, 30, 40 on baselines 0, 100, 200, 300, with spikes the clipping must drop."""
    rng = np.random.default_rng(0)
    data = rng.normal(0, 1, (N_SAMPLES, N_CHANNELS)) * [10, 20, 30, 40] + [0, 100, 200, 300]
    data[::500] -= 2000
    return np.round(data).astype(np.int16)


@pytest.fixture
def store(tmp_path: Path) -> Path:
    out = tmp_path / "rec.zarr"
    build_mipmap(
        _source(),
        out,
        sampling_frequency=30000.0,
        units="uV",
        config=BuildConfig(chunk_config=CHUNKS),
        encoding=LOSSLESS_ENCODING,
    )
    return out


def test_clipped_stats_drop_outliers_and_report_nan_for_empty_rows() -> None:
    rng = np.random.default_rng(1)
    row = rng.normal(5, 2, 10_000)
    row[:50] = 1e4
    mean, sd = clipped_stats(np.stack([row, np.full(10_000, np.nan)]))
    assert mean[0] == pytest.approx(5, abs=0.1)
    assert sd[0] == pytest.approx(2, rel=0.05)
    assert np.isnan(mean[1]) and np.isnan(sd[1])


def test_a_build_stores_stats_measured_from_the_middle_raw_chunk(store: Path) -> None:
    root = zarr.open_group(str(store), mode="r", use_consolidated=True)
    stats = ChannelStats.from_dict(dict(root.attrs[STATS_ATTR]))  # type: ignore[arg-type]
    start = (N_SAMPLES // 2) // 1024 * 1024
    assert stats.samples == (start, start + 1024)
    middle = _source()[start : start + 1024].T
    mean, sd = clipped_stats(middle.astype(np.float64))
    np.testing.assert_allclose(stats.centre, mean)
    np.testing.assert_allclose(stats.noise_sd, sd)
    np.testing.assert_allclose(stats.noise_sd, [10, 20, 30, 40], rtol=0.15)
    assert all(e is not None and e > 0 for e in stats.excursion)


def test_excursion_is_the_coarsest_levels_distance_quantile(store: Path) -> None:
    mip = open_mipmap(store)
    stats = ChannelStats.from_dict(dict(zarr.open_group(str(store), mode="r").attrs[STATS_ATTR]))  # type: ignore[arg-type]
    level = mip._read_minmax(max(mip.strides), 0, N_SAMPLES, slice(None)).data.astype(np.float64)
    centre = np.asarray(stats.centre)[:, None]
    expected = np.quantile(np.maximum(level[:, 1] - centre, centre - level[:, 0]), 0.95, axis=1)
    np.testing.assert_allclose(stats.excursion, expected)  # type: ignore[arg-type]


def test_a_pyramid_only_store_measures_noise_from_its_raw_source(tmp_path: Path) -> None:
    raw = tmp_path / "raw.zarr"
    # SpikeInterface's (T, C) layout, so the measured chunk runs along the first axis.
    root = zarr.open_group(str(raw), mode="w", zarr_format=3)
    root.create_array("raw", data=_source(), chunks=(2048, N_CHANNELS))
    out = tmp_path / "pyr.zarr"
    add_pyramid(
        raw,
        out,
        sampling_frequency=30000.0,
        units="uV",
        config=BuildConfig(chunk_config=CHUNKS),
        encoding=LOSSLESS_ENCODING,
    )
    stats = ChannelStats.from_dict(dict(zarr.open_group(str(out), mode="r").attrs[STATS_ATTR]))  # type: ignore[arg-type]
    start = (N_SAMPLES // 2) // 2048 * 2048
    assert stats.samples == (start, start + 2048)
    np.testing.assert_allclose(stats.noise_sd, [10, 20, 30, 40], rtol=0.15)


def test_add_stats_backfills_a_store_and_its_consolidated_metadata(store: Path) -> None:
    root = zarr.open_group(str(store), mode="r+", use_consolidated=False)
    del root.attrs[STATS_ATTR]
    zarr.consolidate_metadata(zarr.storage.LocalStore(str(store)))
    assert STATS_ATTR not in zarr.open_group(str(store), mode="r", use_consolidated=True).attrs
    AddStatsSettings(str(store)).cli_cmd()
    assert STATS_ATTR in zarr.open_group(str(store), mode="r", use_consolidated=True).attrs
    assert write_channel_stats(store).samples[1] > 0
