"""Tests for ecephys_mipmap_zarr.read."""

import dataclasses

import numpy as np
import pytest
import zarr

from ecephys_mipmap_zarr.build import add_pyramid, build_mipmap
from ecephys_mipmap_zarr.metadata import BuildConfig, ChunkConfig, MipmapAttrs, QuantizationConfig, compute_strides
from ecephys_mipmap_zarr.read import MipmapStore, WindowResult, open_mipmap

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
        config=BuildConfig(chunk_config=SMALL_CFG, lsb_shift_raw=0, lsb_shift_pyramid=0),
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
        config=BuildConfig(chunk_config=SMALL_CFG, lsb_shift_raw=0, lsb_shift_pyramid=0),
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
            config=BuildConfig(chunk_config=SMALL_CFG, lsb_shift_raw=0, lsb_shift_pyramid=0),
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
            config=BuildConfig(chunk_config=cfg, lsb_shift_raw=0, lsb_shift_pyramid=0),
            quantization=QuantizationConfig(quantize=True),
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
            config=BuildConfig(chunk_config=cfg, lsb_shift_raw=0, lsb_shift_pyramid=0),
            quantization=QuantizationConfig(quantize=True),
        )

        store = open_mipmap(out)
        result = store.get_window(0, N_SAMPLES, num_points=100)
        assert not result.is_raw
        assert result.data.dtype == np.float32
        # Should be approximately correct
        block0 = source[:32, :]
        np.testing.assert_allclose(result.data[:, 0, 0], block0.min(axis=0), atol=5.0)
        np.testing.assert_allclose(result.data[:, 1, 0], block0.max(axis=0), atol=5.0)
