"""Tests for ecephys_mipmap_zarr.read."""

import dataclasses

import numpy as np
import pytest
import zarr

from ecephys_mipmap_zarr.build import add_pyramid, build_mipmap
from ecephys_mipmap_zarr.metadata import (
    LOSSLESS_ENCODING,
    BuildConfig,
    ChunkConfig,
    EncodingConfig,
    MipmapAttrs,
    compute_strides,
)
from ecephys_mipmap_zarr.read import MipmapStore, ReadPlan, WindowResult, open_mipmap

SMALL_CFG = ChunkConfig(
    raw_chunk=(4, 1024),
    raw_shard=(4, 4096),
    minmax_chunk=(4, 2, 256),
    minmax_shard=(4, 2, 1024),
)
N_SAMPLES = 100_000
N_CHANNELS = 4


@pytest.fixture()
def source_data():
    rng = np.random.default_rng(42)
    return rng.integers(-1000, 1000, size=(N_SAMPLES, N_CHANNELS), dtype=np.int16)


@pytest.fixture()
def colocated_store(tmp_path, source_data):
    out = tmp_path / "colocated.zarr"
    build_mipmap(
        source_data,
        out,
        sampling_frequency=30000.0,
        units="uV",
        config=BuildConfig(chunk_config=SMALL_CFG),
        encoding=LOSSLESS_ENCODING,
    )
    return out


@pytest.fixture()
def pyramid_only_store(tmp_path, source_data):
    raw_path = tmp_path / "raw.zarr"
    root = zarr.open_group(str(raw_path), mode="w", zarr_format=3)
    raw = root.create_array("raw", shape=source_data.shape, dtype="int16", chunks=(4096, 4))
    raw[:] = source_data

    pyr_path = tmp_path / "pyramid.zarr"
    add_pyramid(
        raw_path,
        pyr_path,
        sampling_frequency=30000.0,
        units="uV",
        config=BuildConfig(chunk_config=SMALL_CFG),
        encoding=LOSSLESS_ENCODING,
    )
    return pyr_path, raw_path


# ---------------------------------------------------------------------------
# WindowResult
# ---------------------------------------------------------------------------


class TestWindowResult:
    def test_is_raw_true(self):
        wr = WindowResult(data=np.zeros((4, 10)), stride=None, t0_actual=0, t1_actual=10)
        assert wr.is_raw is True

    def test_is_raw_false(self):
        wr = WindowResult(data=np.zeros((4, 2, 5)), stride=32, t0_actual=0, t1_actual=160)
        assert wr.is_raw is False

    def test_frozen(self):
        wr = WindowResult(data=np.zeros((4, 10)), stride=None, t0_actual=0, t1_actual=10)
        with pytest.raises(dataclasses.FrozenInstanceError):
            wr.stride = 32  # type: ignore[misc]


# ---------------------------------------------------------------------------
# open_mipmap
# ---------------------------------------------------------------------------


class TestOpenMipmap:
    def test_colocated(self, colocated_store):
        store = open_mipmap(colocated_store)
        assert store.sampling_frequency == 30000.0
        assert store.n_samples == N_SAMPLES
        assert store.n_channels == N_CHANNELS
        assert store.strides == compute_strides(N_SAMPLES)

    def test_pyramid_only(self, pyramid_only_store):
        pyr_path, _ = pyramid_only_store
        store = open_mipmap(pyr_path)
        assert store.sampling_frequency == 30000.0
        assert store.n_samples == N_SAMPLES

    def test_direct_construction(self, colocated_store):
        root = zarr.open_group(str(colocated_store), mode="r", use_consolidated=None)
        attrs = MipmapAttrs.from_dict(dict(root.attrs))
        store = MipmapStore(root, attrs)
        assert store.n_samples == N_SAMPLES


# ---------------------------------------------------------------------------
# get_window — raw fallback
# ---------------------------------------------------------------------------


class TestGetWindowRaw:
    def test_zoomed_in_returns_raw(self, colocated_store, source_data):
        store = open_mipmap(colocated_store)
        # Small window, many points -> spp is small -> raw fallback
        result = store.get_window(1000, 1100, num_points=1000)
        assert result.is_raw
        assert result.stride is None
        assert result.t0_actual == 1000
        assert result.t1_actual == 1100
        np.testing.assert_array_equal(result.data, source_data[1000:1100, :].T)

    def test_channels_none(self, colocated_store, source_data):
        store = open_mipmap(colocated_store)
        result = store.get_window(0, 100, channels=None, num_points=1000)
        assert result.data.shape == (N_CHANNELS, 100)

    def test_channels_slice(self, colocated_store, source_data):
        store = open_mipmap(colocated_store)
        result = store.get_window(0, 100, channels=slice(0, 2), num_points=1000)
        assert result.data.shape == (2, 100)
        np.testing.assert_array_equal(result.data, source_data[0:100, 0:2].T)

    def test_channels_list(self, colocated_store, source_data):
        store = open_mipmap(colocated_store)
        result = store.get_window(0, 100, channels=[0, 3], num_points=1000)
        assert result.data.shape == (2, 100)
        np.testing.assert_array_equal(result.data, source_data[0:100, [0, 3]].T)

    def test_channels_ndarray(self, colocated_store, source_data):
        store = open_mipmap(colocated_store)
        ch = np.array([1, 2])
        result = store.get_window(0, 100, channels=ch, num_points=1000)
        assert result.data.shape == (2, 100)


# ---------------------------------------------------------------------------
# get_window — minmax
# ---------------------------------------------------------------------------


class TestGetWindowMinmax:
    def test_zoomed_out_returns_minmax(self, colocated_store):
        store = open_mipmap(colocated_store)
        # Full recording, few points -> large spp -> pyramid level
        result = store.get_window(0, N_SAMPLES, num_points=100)
        assert not result.is_raw
        assert result.stride is not None
        assert result.data.ndim == 3
        assert result.data.shape[1] == 2  # [min, max] axis

    def test_stride_32_selected(self, colocated_store):
        store = open_mipmap(colocated_store)
        # 100K samples / 200 points = 500 spp; 500/32 = 15.6 >= 4 -> stride 32 passes
        # Only stride 32 exists for 100K samples
        result = store.get_window(0, N_SAMPLES, num_points=200)
        assert result.stride == 32

    def test_block_alignment(self, colocated_store):
        store = open_mipmap(colocated_store)
        # Request a non-block-aligned window
        t0, t1 = 50, 5000
        result = store.get_window(t0, t1, num_points=10)
        if not result.is_raw:
            assert result.t0_actual <= t0
            assert result.t1_actual >= t1

    def test_minmax_values_correct(self, colocated_store, source_data):
        store = open_mipmap(colocated_store)
        result = store.get_window(0, N_SAMPLES, num_points=100)
        assert result.stride is not None

        # Spot-check block 0
        stride = result.stride
        block_data = source_data[:stride, :]
        np.testing.assert_array_equal(result.data[:, 0, 0], block_data.min(axis=0))
        np.testing.assert_array_equal(result.data[:, 1, 0], block_data.max(axis=0))

    def test_minmax_channel_selection(self, colocated_store):
        store = open_mipmap(colocated_store)
        result = store.get_window(0, N_SAMPLES, channels=slice(0, 2), num_points=100)
        if not result.is_raw:
            assert result.data.shape[0] == 2


# ---------------------------------------------------------------------------
# get_window — edge cases
# ---------------------------------------------------------------------------


class TestGetWindowEdgeCases:
    def test_clamp_negative_t0(self, colocated_store, source_data):
        store = open_mipmap(colocated_store)
        result = store.get_window(-100, 100, num_points=1000)
        assert result.t0_actual == 0
        np.testing.assert_array_equal(result.data, source_data[0:100, :].T)

    def test_clamp_t1_past_end(self, colocated_store, source_data):
        store = open_mipmap(colocated_store)
        result = store.get_window(N_SAMPLES - 50, N_SAMPLES + 100, num_points=1000)
        assert result.t1_actual == N_SAMPLES

    def test_empty_window_raises(self, colocated_store):
        store = open_mipmap(colocated_store)
        with pytest.raises(ValueError, match="Empty window"):
            store.get_window(100, 50, num_points=100)

    def test_zero_num_points_raises(self, colocated_store):
        store = open_mipmap(colocated_store)
        with pytest.raises(ValueError, match="num_points must be positive"):
            store.get_window(0, 1000, num_points=0)

    def test_negative_num_points_raises(self, colocated_store):
        store = open_mipmap(colocated_store)
        with pytest.raises(ValueError, match="num_points must be positive"):
            store.get_window(0, 1000, num_points=-10)

    def test_single_sample_window(self, colocated_store, source_data):
        store = open_mipmap(colocated_store)
        result = store.get_window(50, 51, num_points=100)
        assert result.is_raw
        assert result.data.shape[1] == 1
        np.testing.assert_array_equal(result.data[:, 0], source_data[50, :])

    def test_full_recording_coarsest_level(self, colocated_store):
        store = open_mipmap(colocated_store)
        result = store.get_window(0, N_SAMPLES, num_points=10)
        # Should pick coarsest available level
        assert not result.is_raw


# ---------------------------------------------------------------------------
# get_window — pyramid-only stores
# ---------------------------------------------------------------------------


class TestGetWindowPyramidOnly:
    def test_raw_fallback_works(self, pyramid_only_store, source_data):
        pyr_path, _ = pyramid_only_store
        store = open_mipmap(pyr_path)
        # Zoomed in -> raw fallback via raw_source
        result = store.get_window(0, 100, num_points=1000)
        assert result.is_raw
        np.testing.assert_array_equal(result.data, source_data[0:100, :].T)

    def test_minmax_works(self, pyramid_only_store):
        pyr_path, _ = pyramid_only_store
        store = open_mipmap(pyr_path)
        result = store.get_window(0, N_SAMPLES, num_points=100)
        assert not result.is_raw

    def test_raw_unavailable_raises(self, tmp_path, source_data):
        """Pyramid-only store with unreachable raw_source raises RuntimeError."""
        raw_path = tmp_path / "raw.zarr"
        root = zarr.open_group(str(raw_path), mode="w", zarr_format=3)
        raw = root.create_array("raw", shape=source_data.shape, dtype="int16", chunks=(4096, 4))
        raw[:] = source_data

        pyr_path = tmp_path / "pyramid.zarr"
        add_pyramid(
            raw_path,
            pyr_path,
            sampling_frequency=30000.0,
            units="uV",
            config=BuildConfig(chunk_config=SMALL_CFG),
            encoding=LOSSLESS_ENCODING,
        )

        # Open pyramid store but construct MipmapStore without raw_group
        pyr_root = zarr.open_group(str(pyr_path), mode="r", use_consolidated=None)
        attrs = MipmapAttrs.from_dict(dict(pyr_root.attrs))
        store = MipmapStore(pyr_root, attrs, raw_group=None)

        # Minmax still works
        result = store.get_window(0, N_SAMPLES, num_points=100)
        assert not result.is_raw

        # Raw fallback raises
        with pytest.raises(RuntimeError, match="Raw data not available"):
            store.get_window(0, 100, num_points=1000)


# ---------------------------------------------------------------------------
# get_window — quantized stores
# ---------------------------------------------------------------------------


class TestGetWindowQuantized:
    def test_unquantized_unchanged(self, colocated_store, source_data):
        """Existing unquantized stores still work without dequantization."""
        store = open_mipmap(colocated_store)
        result = store.get_window(0, 100, num_points=1000)
        assert result.is_raw
        assert result.data.dtype == np.int16
        np.testing.assert_array_equal(result.data, source_data[0:100, :].T)

    def test_dequantizes_raw(self, tmp_path):
        rng = np.random.default_rng(42)
        source = rng.standard_normal((N_SAMPLES, N_CHANNELS)).astype(np.float32) * 100
        out = tmp_path / "quantized.zarr"
        cfg = SMALL_CFG

        build_mipmap(
            source,
            out,
            sampling_frequency=30000.0,
            units="uV",
            dtype="float32",
            config=BuildConfig(chunk_config=cfg),
            encoding=EncodingConfig(raw="range", pyramids="range"),
        )

        store = open_mipmap(out)
        result = store.get_window(0, 100, num_points=1000)
        assert result.is_raw
        assert result.data.dtype == np.float32
        np.testing.assert_allclose(result.data, source[0:100, :].T, atol=0.1)

    def test_dequantizes_minmax(self, tmp_path):
        rng = np.random.default_rng(42)
        source = rng.standard_normal((N_SAMPLES, N_CHANNELS)).astype(np.float32) * 100
        out = tmp_path / "quantized.zarr"
        cfg = SMALL_CFG

        build_mipmap(
            source,
            out,
            sampling_frequency=30000.0,
            units="uV",
            dtype="float32",
            config=BuildConfig(chunk_config=cfg),
            encoding=EncodingConfig(raw="range", pyramids="range"),
        )

        store = open_mipmap(out)
        result = store.get_window(0, N_SAMPLES, num_points=100)
        assert not result.is_raw
        assert result.data.dtype == np.float32
        # Should be approximately correct
        block0 = source[:32, :]
        np.testing.assert_allclose(result.data[:, 0, 0], block0.min(axis=0), atol=5.0)
        np.testing.assert_allclose(result.data[:, 1, 0], block0.max(axis=0), atol=5.0)


# ---------------------------------------------------------------------------
# ReadPlan
# ---------------------------------------------------------------------------


class TestReadPlan:
    def test_hashable(self):
        """ReadPlan instances are hashable and can be used as dict keys."""
        plan_a = ReadPlan(array_path="raw", stride=None, t0_actual=0, t1_actual=100, channels=(0, 1))
        plan_b = ReadPlan(array_path="raw", stride=None, t0_actual=0, t1_actual=100, channels=(0, 1))
        assert hash(plan_a) == hash(plan_b)
        d = {plan_a: "value"}
        assert d[plan_b] == "value"

    def test_is_raw(self):
        """stride=None means raw."""
        plan = ReadPlan(array_path="raw", stride=None, t0_actual=0, t1_actual=100)
        assert plan.stride is None
        plan_mm = ReadPlan(
            array_path="minmax/s32", stride=32, t0_actual=0, t1_actual=3200, block_start=0, block_end=100
        )
        assert plan_mm.stride is not None

    def test_equality(self):
        """Frozen dataclass equality works correctly."""
        plan_a = ReadPlan(
            array_path="minmax/s32",
            stride=32,
            t0_actual=0,
            t1_actual=3200,
            block_start=0,
            block_end=100,
            channels=(0, 1, 2),
        )
        plan_b = ReadPlan(
            array_path="minmax/s32",
            stride=32,
            t0_actual=0,
            t1_actual=3200,
            block_start=0,
            block_end=100,
            channels=(0, 1, 2),
        )
        assert plan_a == plan_b

    def test_inequality(self):
        """Different plans are not equal."""
        plan_a = ReadPlan(array_path="raw", stride=None, t0_actual=0, t1_actual=100)
        plan_b = ReadPlan(array_path="raw", stride=None, t0_actual=0, t1_actual=200)
        assert plan_a != plan_b

    def test_frozen(self):
        """ReadPlan is frozen (immutable)."""
        plan = ReadPlan(array_path="raw", stride=None, t0_actual=0, t1_actual=100)
        with pytest.raises(dataclasses.FrozenInstanceError):
            plan.t0_actual = 10  # type: ignore[misc]


# ---------------------------------------------------------------------------
# plan_window
# ---------------------------------------------------------------------------


class TestPlanWindow:
    def test_raw_plan(self, colocated_store):
        """Zoomed in produces a raw plan."""
        store = open_mipmap(colocated_store)
        plan = store.plan_window(1000, 1100, num_points=1000)
        assert plan.stride is None
        assert plan.array_path == "raw"
        assert plan.t0_actual == 1000
        assert plan.t1_actual == 1100
        assert plan.block_start is None
        assert plan.block_end is None

    def test_minmax_plan(self, colocated_store):
        """Zoomed out produces a minmax plan with block alignment."""
        store = open_mipmap(colocated_store)
        plan = store.plan_window(0, N_SAMPLES, num_points=100)
        assert plan.stride is not None
        assert plan.array_path.startswith("minmax/s")
        assert plan.block_start is not None
        assert plan.block_end is not None
        assert plan.t0_actual == plan.block_start * plan.stride
        assert plan.t1_actual <= N_SAMPLES

    def test_channel_normalization_none(self, colocated_store):
        """None channels produce None in the plan."""
        store = open_mipmap(colocated_store)
        plan = store.plan_window(0, 100, channels=None, num_points=1000)
        assert plan.channels is None

    def test_channel_normalization_slice(self, colocated_store):
        """Slice channels are expanded to tuple."""
        store = open_mipmap(colocated_store)
        plan = store.plan_window(0, 100, channels=slice(0, 2), num_points=1000)
        assert plan.channels == (0, 1)

    def test_channel_normalization_list(self, colocated_store):
        """List channels become tuple."""
        store = open_mipmap(colocated_store)
        plan = store.plan_window(0, 100, channels=[0, 3], num_points=1000)
        assert plan.channels == (0, 3)

    def test_channel_normalization_ndarray(self, colocated_store):
        """ndarray channels become tuple."""
        store = open_mipmap(colocated_store)
        plan = store.plan_window(0, 100, channels=np.array([1, 2]), num_points=1000)
        assert plan.channels == (1, 2)

    def test_clamping_negative_t0(self, colocated_store):
        """t0 < 0 is clamped to 0."""
        store = open_mipmap(colocated_store)
        plan = store.plan_window(-100, 100, num_points=1000)
        assert plan.t0_actual == 0

    def test_clamping_t1_past_end(self, colocated_store):
        """t1 > n_samples is clamped."""
        store = open_mipmap(colocated_store)
        plan = store.plan_window(N_SAMPLES - 50, N_SAMPLES + 100, num_points=1000)
        assert plan.t1_actual == N_SAMPLES

    def test_empty_raises(self, colocated_store):
        """t0 >= t1 after clamping raises ValueError."""
        store = open_mipmap(colocated_store)
        with pytest.raises(ValueError, match="Empty window"):
            store.plan_window(100, 50, num_points=100)

    def test_zero_num_points_raises(self, colocated_store):
        """num_points <= 0 raises ValueError."""
        store = open_mipmap(colocated_store)
        with pytest.raises(ValueError, match="num_points must be positive"):
            store.plan_window(0, 1000, num_points=0)

    def test_raw_unavailable_raises(self, tmp_path, source_data):
        """plan_window raises RuntimeError if raw needed but unavailable."""
        raw_path = tmp_path / "raw.zarr"
        root = zarr.open_group(str(raw_path), mode="w", zarr_format=3)
        raw_arr = root.create_array("raw", shape=source_data.shape, dtype="int16", chunks=(4096, 4))
        raw_arr[:] = source_data

        pyr_path = tmp_path / "pyramid.zarr"
        add_pyramid(
            raw_path,
            pyr_path,
            sampling_frequency=30000.0,
            units="uV",
            config=BuildConfig(chunk_config=SMALL_CFG),
            encoding=LOSSLESS_ENCODING,
        )

        pyr_root = zarr.open_group(str(pyr_path), mode="r", use_consolidated=None)
        attrs = MipmapAttrs.from_dict(dict(pyr_root.attrs))
        store = MipmapStore(pyr_root, attrs, raw_group=None)

        # Minmax plan works fine
        plan = store.plan_window(0, N_SAMPLES, num_points=100)
        assert plan.stride is not None

        # Raw plan raises
        with pytest.raises(RuntimeError, match="Raw data not available"):
            store.plan_window(0, 100, num_points=1000)


# ---------------------------------------------------------------------------
# execute_plan
# ---------------------------------------------------------------------------


class TestExecutePlan:
    def test_matches_get_window_raw(self, colocated_store, source_data):
        """plan + execute matches get_window for raw reads."""
        store = open_mipmap(colocated_store)
        plan = store.plan_window(1000, 1100, num_points=1000)
        result_pe = store.execute_plan(plan)
        result_gw = store.get_window(1000, 1100, num_points=1000)
        np.testing.assert_array_equal(result_pe.data, result_gw.data)
        assert result_pe.stride == result_gw.stride
        assert result_pe.t0_actual == result_gw.t0_actual
        assert result_pe.t1_actual == result_gw.t1_actual

    def test_matches_get_window_minmax(self, colocated_store):
        """plan + execute matches get_window for minmax reads."""
        store = open_mipmap(colocated_store)
        plan = store.plan_window(0, N_SAMPLES, num_points=100)
        result_pe = store.execute_plan(plan)
        result_gw = store.get_window(0, N_SAMPLES, num_points=100)
        np.testing.assert_array_equal(result_pe.data, result_gw.data)
        assert result_pe.stride == result_gw.stride
        assert result_pe.t0_actual == result_gw.t0_actual
        assert result_pe.t1_actual == result_gw.t1_actual

    def test_matches_get_window_with_channels(self, colocated_store, source_data):
        """plan + execute matches get_window with channel selection."""
        store = open_mipmap(colocated_store)
        channels = [0, 3]
        plan = store.plan_window(0, 100, channels=channels, num_points=1000)
        result_pe = store.execute_plan(plan)
        result_gw = store.get_window(0, 100, channels=channels, num_points=1000)
        np.testing.assert_array_equal(result_pe.data, result_gw.data)

    def test_dequantize_false_raw(self, colocated_store):
        """dequantize=False on raw returns stored dtype and dequantized=False."""
        store = open_mipmap(colocated_store)
        plan = store.plan_window(0, 100, num_points=1000)
        result = store.execute_plan(plan, dequantize=False)
        assert result.dequantized is False
        assert result.is_raw

    def test_dequantize_false_minmax(self, colocated_store):
        """dequantize=False on minmax returns stored dtype and dequantized=False."""
        store = open_mipmap(colocated_store)
        plan = store.plan_window(0, N_SAMPLES, num_points=100)
        result = store.execute_plan(plan, dequantize=False)
        assert result.dequantized is False
        assert not result.is_raw


# ---------------------------------------------------------------------------
# dequantize method
# ---------------------------------------------------------------------------


class TestDequantize:
    def test_roundtrip_unquantized(self, colocated_store, source_data):
        """execute(dequantize=False) -> dequantize() matches execute(dequantize=True) for raw."""
        store = open_mipmap(colocated_store)
        plan = store.plan_window(0, 100, num_points=1000)
        result_raw_stored = store.execute_plan(plan, dequantize=False)
        result_dequantized = store.dequantize(result_raw_stored, plan)
        result_direct = store.execute_plan(plan, dequantize=True)
        np.testing.assert_array_equal(result_dequantized.data, result_direct.data)
        assert result_dequantized.dequantized is True

    def test_roundtrip_unquantized_minmax(self, colocated_store):
        """execute(dequantize=False) -> dequantize() matches execute(dequantize=True) for minmax."""
        store = open_mipmap(colocated_store)
        plan = store.plan_window(0, N_SAMPLES, num_points=100)
        result_stored = store.execute_plan(plan, dequantize=False)
        result_dequantized = store.dequantize(result_stored, plan)
        result_direct = store.execute_plan(plan, dequantize=True)
        np.testing.assert_array_equal(result_dequantized.data, result_direct.data)
        assert result_dequantized.dequantized is True

    def test_already_dequantized(self, colocated_store):
        """dequantize() on already-dequantized result is a no-op."""
        store = open_mipmap(colocated_store)
        plan = store.plan_window(0, 100, num_points=1000)
        result = store.execute_plan(plan, dequantize=True)
        assert result.dequantized is True
        result2 = store.dequantize(result, plan)
        assert result2 is result  # exact same object

    def test_lsb_shift_raw(self, tmp_path):
        """dequantize() correctly reconstructs LSB-shifted raw data."""
        rng = np.random.default_rng(42)
        n_samples, n_channels = 100_000, 4
        source = rng.integers(-500, 500, size=(n_samples, n_channels), dtype=np.int16)
        out = tmp_path / "lsb.zarr"
        cfg = SMALL_CFG

        build_mipmap(
            source,
            out,
            sampling_frequency=30000.0,
            units="uV",
            config=BuildConfig(chunk_config=cfg),
            encoding=EncodingConfig(raw_shift=2, pyramid_shift=4),
        )

        store = open_mipmap(out)

        # Raw roundtrip
        plan = store.plan_window(0, 1000, num_points=10000)
        result_stored = store.execute_plan(plan, dequantize=False)
        assert result_stored.dequantized is False
        result_dequantized = store.dequantize(result_stored, plan)
        result_direct = store.execute_plan(plan, dequantize=True)
        np.testing.assert_array_equal(result_dequantized.data, result_direct.data)
        assert result_dequantized.data.dtype == np.int16

    def test_lsb_shift_minmax(self, tmp_path):
        """dequantize() correctly reconstructs LSB-shifted minmax data."""
        rng = np.random.default_rng(42)
        n_samples, n_channels = 100_000, 4
        source = rng.integers(-500, 500, size=(n_samples, n_channels), dtype=np.int16)
        out = tmp_path / "lsb.zarr"
        cfg = SMALL_CFG

        build_mipmap(
            source,
            out,
            sampling_frequency=30000.0,
            units="uV",
            config=BuildConfig(chunk_config=cfg),
            encoding=EncodingConfig(raw_shift=2, pyramid_shift=4),
        )

        store = open_mipmap(out)

        plan = store.plan_window(0, n_samples, num_points=100)
        result_stored = store.execute_plan(plan, dequantize=False)
        assert result_stored.dequantized is False
        result_dequantized = store.dequantize(result_stored, plan)
        result_direct = store.execute_plan(plan, dequantize=True)
        np.testing.assert_array_equal(result_dequantized.data, result_direct.data)
        assert result_dequantized.data.dtype == np.int16

    def test_quantized_roundtrip(self, tmp_path):
        """dequantize() correctly reconstructs quantized float32 data."""
        rng = np.random.default_rng(42)
        source = rng.standard_normal((N_SAMPLES, N_CHANNELS)).astype(np.float32) * 100
        out = tmp_path / "quantized.zarr"
        cfg = SMALL_CFG

        build_mipmap(
            source,
            out,
            sampling_frequency=30000.0,
            units="uV",
            dtype="float32",
            config=BuildConfig(chunk_config=cfg),
            encoding=EncodingConfig(raw="range", pyramids="range"),
        )

        store = open_mipmap(out)

        # Raw roundtrip
        plan_raw = store.plan_window(0, 100, num_points=1000)
        result_stored = store.execute_plan(plan_raw, dequantize=False)
        result_dequantized = store.dequantize(result_stored, plan_raw)
        result_direct = store.execute_plan(plan_raw, dequantize=True)
        np.testing.assert_allclose(result_dequantized.data, result_direct.data)

        # Minmax roundtrip
        plan_mm = store.plan_window(0, N_SAMPLES, num_points=100)
        result_stored_mm = store.execute_plan(plan_mm, dequantize=False)
        result_deq_mm = store.dequantize(result_stored_mm, plan_mm)
        result_direct_mm = store.execute_plan(plan_mm, dequantize=True)
        np.testing.assert_allclose(result_deq_mm.data, result_direct_mm.data)

    def test_channel_subset_quantized(self, tmp_path):
        """dequantize() works correctly with channel subsets on quantized data."""
        rng = np.random.default_rng(42)
        source = rng.standard_normal((N_SAMPLES, N_CHANNELS)).astype(np.float32) * 100
        out = tmp_path / "quantized.zarr"
        cfg = SMALL_CFG

        build_mipmap(
            source,
            out,
            sampling_frequency=30000.0,
            units="uV",
            dtype="float32",
            config=BuildConfig(chunk_config=cfg),
            encoding=EncodingConfig(raw="range", pyramids="range"),
        )

        store = open_mipmap(out)

        plan = store.plan_window(0, 100, channels=[0, 2], num_points=1000)
        result_stored = store.execute_plan(plan, dequantize=False)
        result_dequantized = store.dequantize(result_stored, plan)
        result_direct = store.execute_plan(plan, dequantize=True)
        np.testing.assert_allclose(result_dequantized.data, result_direct.data)
        assert result_dequantized.data.shape[0] == 2


# ---------------------------------------------------------------------------
# get_window — dequantize parameter
# ---------------------------------------------------------------------------


class TestGetWindowDequantize:
    def test_dequantize_false_raw(self, colocated_store, source_data):
        """get_window(dequantize=False) returns stored dtype for raw."""
        store = open_mipmap(colocated_store)
        result = store.get_window(0, 100, num_points=1000, dequantize=False)
        assert result.dequantized is False
        assert result.is_raw

    def test_dequantize_false_minmax(self, colocated_store):
        """get_window(dequantize=False) returns stored dtype for minmax."""
        store = open_mipmap(colocated_store)
        result = store.get_window(0, N_SAMPLES, num_points=100, dequantize=False)
        assert result.dequantized is False
        assert not result.is_raw

    def test_backward_compat_default_true(self, colocated_store, source_data):
        """get_window without dequantize param works as before (default=True)."""
        store = open_mipmap(colocated_store)
        result = store.get_window(0, 100, num_points=1000)
        assert result.dequantized is True
        np.testing.assert_array_equal(result.data, source_data[0:100, :].T)

    def test_dequantize_false_lsb_shifted_raw(self, tmp_path):
        """dequantize=False on LSB-shifted store returns stored int8 data."""
        rng = np.random.default_rng(42)
        n_samples, n_channels = 100_000, 4
        source = rng.integers(-500, 500, size=(n_samples, n_channels), dtype=np.int16)
        out = tmp_path / "lsb.zarr"
        cfg = SMALL_CFG

        build_mipmap(
            source,
            out,
            sampling_frequency=30000.0,
            units="uV",
            config=BuildConfig(chunk_config=cfg),
            encoding=EncodingConfig(raw_shift=2, pyramid_shift=4),
        )

        store = open_mipmap(out)

        # With dequantize=True, we get int16 (reconstructed)
        result_deq = store.get_window(0, 100, num_points=1000, dequantize=True)
        assert result_deq.data.dtype == np.int16
        assert result_deq.dequantized is True

        # With dequantize=False, we get the stored dtype (int8)
        result_raw = store.get_window(0, 100, num_points=1000, dequantize=False)
        assert result_raw.data.dtype == np.int8
        assert result_raw.dequantized is False

    def test_dequantize_false_quantized_raw(self, tmp_path):
        """dequantize=False on quantized store returns stored uint16 data."""
        rng = np.random.default_rng(42)
        source = rng.standard_normal((N_SAMPLES, N_CHANNELS)).astype(np.float32) * 100
        out = tmp_path / "quantized.zarr"
        cfg = SMALL_CFG

        build_mipmap(
            source,
            out,
            sampling_frequency=30000.0,
            units="uV",
            dtype="float32",
            config=BuildConfig(chunk_config=cfg),
            encoding=EncodingConfig(raw="range", pyramids="range"),
        )

        store = open_mipmap(out)

        # With dequantize=True, we get float32
        result_deq = store.get_window(0, 100, num_points=1000, dequantize=True)
        assert result_deq.data.dtype == np.float32
        assert result_deq.dequantized is True

        # With dequantize=False, we get the stored quantized dtype
        result_raw = store.get_window(0, 100, num_points=1000, dequantize=False)
        assert result_raw.data.dtype == np.uint16
        assert result_raw.dequantized is False


# ---------------------------------------------------------------------------
# WindowResult.dequantized field
# ---------------------------------------------------------------------------


class TestWindowResultDequantized:
    def test_default_true(self):
        """WindowResult.dequantized defaults to True."""
        wr = WindowResult(data=np.zeros((4, 10)), stride=None, t0_actual=0, t1_actual=10)
        assert wr.dequantized is True

    def test_explicit_false(self):
        """WindowResult.dequantized can be set to False."""
        wr = WindowResult(data=np.zeros((4, 10)), stride=None, t0_actual=0, t1_actual=10, dequantized=False)
        assert wr.dequantized is False


class TestPruneCache:
    """Tests for the _prune_cache function."""

    def test_deletes_oldest_when_over_budget(self, tmp_path):
        """Oldest files are deleted when cache exceeds budget."""
        import time

        from ecephys_mipmap_zarr._store import _prune_cache

        # Create files with different mtimes
        for i in range(5):
            f = tmp_path / f"file_{i}.dat"
            f.write_bytes(b"x" * 1000)
            time.sleep(0.05)  # ensure different mtimes

        # Budget allows 3 files (3000 bytes)
        _prune_cache(tmp_path, 3000)
        remaining = sorted(f.name for f in tmp_path.iterdir())
        assert len(remaining) == 3
        # Oldest files (file_0, file_1) should be deleted
        assert "file_0.dat" not in remaining
        assert "file_1.dat" not in remaining

    def test_no_deletion_when_under_budget(self, tmp_path):
        """No files deleted when cache is under budget."""
        from ecephys_mipmap_zarr._store import _prune_cache

        for i in range(3):
            (tmp_path / f"file_{i}.dat").write_bytes(b"x" * 100)

        _prune_cache(tmp_path, 10000)
        assert len(list(tmp_path.iterdir())) == 3

    def test_nonexistent_dir(self, tmp_path):
        """No error when cache dir doesn't exist."""
        from ecephys_mipmap_zarr._store import _prune_cache

        _prune_cache(tmp_path / "nonexistent", 1000)  # should not raise


class TestDefaultCacheDir:
    """Tests for _default_cache_dir."""

    def test_returns_path(self):
        """Returns a Path containing 'ecephys-mipmap'."""
        from pathlib import Path

        from ecephys_mipmap_zarr._store import _default_cache_dir

        result = _default_cache_dir()
        assert isinstance(result, Path)
        assert "ecephys-mipmap" in str(result)

    def test_xdg_override(self, monkeypatch, tmp_path):
        """Respects XDG_CACHE_HOME environment variable."""
        from ecephys_mipmap_zarr._store import _default_cache_dir

        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        result = _default_cache_dir()
        assert result == tmp_path / "ecephys-mipmap"


class TestIsRemote:
    """Tests for MipmapStore.is_remote property."""

    def test_local_store_not_remote(self, tmp_path):
        """Local store is not remote."""
        source = np.random.default_rng(42).integers(-500, 500, size=(N_SAMPLES, N_CHANNELS), dtype=np.int16)
        out = tmp_path / "test.zarr"
        build_mipmap(
            source,
            out,
            sampling_frequency=30000.0,
            units="uV",
            config=BuildConfig(chunk_config=SMALL_CFG),
            encoding=LOSSLESS_ENCODING,
        )
        store = open_mipmap(out)
        assert store.is_remote is False
