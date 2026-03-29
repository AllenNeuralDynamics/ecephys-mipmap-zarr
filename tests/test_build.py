"""Tests for ecephys_mipmap_zarr.build."""

from pathlib import Path

import numpy as np
import pytest
import zarr

from ecephys_mipmap_zarr.build import (
    _cascade_minmax,
    _compute_minmax_block,
    add_pyramid,
    build_mipmap,
    ensure_pyramid,
)
from ecephys_mipmap_zarr.metadata import (
    BuildConfig,
    ChunkConfig,
    MipmapAttrs,
    QuantizationConfig,
    compute_strides,
    minmax_path,
    n_blocks,
)

# ---------------------------------------------------------------------------
# _compute_minmax_block
# ---------------------------------------------------------------------------


class TestComputeMinmaxBlock:
    def test_exact_division(self):
        rng = np.random.default_rng(42)
        data = rng.integers(-1000, 1000, size=(4, 1024), dtype=np.int16)
        result = _compute_minmax_block(data, stride=32)

        assert result.shape == (4, 2, 32)
        # Spot-check first block
        block0 = data[:, :32]
        np.testing.assert_array_equal(result[:, 0, 0], block0.min(axis=1))
        np.testing.assert_array_equal(result[:, 1, 0], block0.max(axis=1))

    def test_with_remainder(self):
        rng = np.random.default_rng(42)
        data = rng.integers(-1000, 1000, size=(2, 1025), dtype=np.int16)
        result = _compute_minmax_block(data, stride=32)

        # 1025 // 32 = 32 full blocks, + 1 tail = 33 blocks
        assert result.shape == (2, 2, 33)
        # Check tail block covers the last sample
        tail = data[:, 1024:]
        np.testing.assert_array_equal(result[:, 0, 32], tail.min(axis=1))
        np.testing.assert_array_equal(result[:, 1, 32], tail.max(axis=1))

    def test_fewer_than_stride(self):
        data = np.array([[1, 3, 0], [2, 4, 5]], dtype=np.int16)
        result = _compute_minmax_block(data, stride=32)

        assert result.shape == (2, 2, 1)
        np.testing.assert_array_equal(result[:, 0, 0], [0, 2])  # mins
        np.testing.assert_array_equal(result[:, 1, 0], [3, 5])  # maxes

    def test_constant_data(self):
        data = np.full((3, 64), 42, dtype=np.int16)
        result = _compute_minmax_block(data, stride=32)

        assert result.shape == (3, 2, 2)
        np.testing.assert_array_equal(result[:, 0, :], 42)  # mins
        np.testing.assert_array_equal(result[:, 1, :], 42)  # maxes

    def test_single_channel(self):
        rng = np.random.default_rng(42)
        data = rng.integers(-100, 100, size=(1, 128), dtype=np.int16)
        result = _compute_minmax_block(data, stride=32)

        assert result.shape == (1, 2, 4)
        for i in range(4):
            block = data[:, i * 32 : (i + 1) * 32]
            np.testing.assert_array_equal(result[:, 0, i], block.min(axis=1))
            np.testing.assert_array_equal(result[:, 1, i], block.max(axis=1))


# ---------------------------------------------------------------------------
# _cascade_minmax
# ---------------------------------------------------------------------------


class TestCascadeMinmax:
    def test_exact_division(self):
        rng = np.random.default_rng(42)
        prev = rng.integers(-1000, 1000, size=(4, 2, 64), dtype=np.int16)
        result = _cascade_minmax(prev, factor=8)

        assert result.shape == (4, 2, 8)
        # Check first output block
        group = prev[:, :, :8]
        np.testing.assert_array_equal(result[:, 0, 0], group[:, 0, :].min(axis=1))
        np.testing.assert_array_equal(result[:, 1, 0], group[:, 1, :].max(axis=1))

    def test_with_remainder(self):
        rng = np.random.default_rng(42)
        prev = rng.integers(-1000, 1000, size=(2, 2, 65), dtype=np.int16)
        result = _cascade_minmax(prev, factor=8)

        # 65 // 8 = 8 full groups + 1 tail = 9 blocks
        assert result.shape == (2, 2, 9)
        # Check tail block
        tail = prev[:, :, 64:]
        np.testing.assert_array_equal(result[:, 0, 8], tail[:, 0, :].min(axis=1))
        np.testing.assert_array_equal(result[:, 1, 8], tail[:, 1, :].max(axis=1))

    def test_preserves_extremes(self):
        """Min-of-mins should be global min, max-of-maxes should be global max."""
        rng = np.random.default_rng(42)
        prev = rng.integers(-1000, 1000, size=(3, 2, 16), dtype=np.int16)
        result = _cascade_minmax(prev, factor=16)

        assert result.shape == (3, 2, 1)
        np.testing.assert_array_equal(result[:, 0, 0], prev[:, 0, :].min(axis=1))
        np.testing.assert_array_equal(result[:, 1, 0], prev[:, 1, :].max(axis=1))

    def test_single_block(self):
        prev = np.array([[[10], [30]], [[20], [40]]], dtype=np.int16)  # (2, 2, 1)
        result = _cascade_minmax(prev, factor=8)

        assert result.shape == (2, 2, 1)
        np.testing.assert_array_equal(result, prev)


# ---------------------------------------------------------------------------
# build_mipmap integration tests
# ---------------------------------------------------------------------------


class TestBuildMipmap:
    def _make_source(self, n_samples, n_channels, dtype="int16", seed=42):
        rng = np.random.default_rng(seed)
        if np.issubdtype(np.dtype(dtype), np.integer):
            return rng.integers(-1000, 1000, size=(n_samples, n_channels), dtype=dtype)
        return rng.standard_normal((n_samples, n_channels)).astype(dtype)

    def _small_chunk_config(self):
        """Small chunk/shard config for fast tests."""
        return ChunkConfig(
            raw_chunk=(4, 1024),
            raw_shard=(4, 4096),
            minmax_chunk=(4, 2, 256),
            minmax_shard=(4, 2, 1024),
        )

    def test_basic(self, tmp_path):
        """Build mipmap from 100K samples × 4 channels."""
        n_samples, n_channels = 100_000, 4
        source = self._make_source(n_samples, n_channels)
        out = tmp_path / "test.zarr"
        cfg = self._small_chunk_config()

        build_mipmap(
            source,
            out,
            sampling_frequency=30000.0,
            units="uV",
            config=BuildConfig(chunk_config=cfg, lsb_shift_raw=0, lsb_shift_pyramid=0),
        )

        root = zarr.open_group(str(out), mode="r", use_consolidated=None)

        # Check raw array
        raw = root["raw"]
        assert raw.shape == (n_channels, n_samples)
        np.testing.assert_array_equal(np.asarray(raw[:]), source.T)

        # Check strides
        strides = compute_strides(n_samples)
        for stride in strides:
            path = minmax_path(stride).lstrip("/")
            arr = root[path]
            expected_blocks = n_blocks(n_samples, stride)
            assert arr.shape == (n_channels, 2, expected_blocks)

        # Check root attrs
        attrs = MipmapAttrs.from_dict(dict(root.attrs))
        assert attrs.sampling_frequency == 30000.0
        assert attrs.units == "uV"
        assert attrs.strides == strides
        assert attrs.n_samples == n_samples
        assert attrs.n_channels == n_channels
        assert attrs.raw_source is None

    def test_s32_values_correct(self, tmp_path):
        """Spot-check that s32 min/max values are correct."""
        n_samples, n_channels = 100_000, 4
        source = self._make_source(n_samples, n_channels)
        out = tmp_path / "test.zarr"
        cfg = self._small_chunk_config()

        build_mipmap(
            source,
            out,
            sampling_frequency=30000.0,
            units="uV",
            config=BuildConfig(chunk_config=cfg, lsb_shift_raw=0, lsb_shift_pyramid=0),
        )

        root = zarr.open_group(str(out), mode="r", use_consolidated=None)
        s32 = np.asarray(root["minmax/s32"][:])

        # Check a few blocks
        for block_idx in [0, 10, 100]:
            t0 = block_idx * 32
            t1 = t0 + 32
            block_data = source[t0:t1, :]
            np.testing.assert_array_equal(s32[:, 0, block_idx], block_data.min(axis=0))
            np.testing.assert_array_equal(s32[:, 1, block_idx], block_data.max(axis=0))

    def test_cascade_values_correct(self, tmp_path):
        """Verify s256 = cascaded min-of-mins / max-of-maxes from s32."""
        # Need >= 256 * 1024 = 262144 samples for s256 level
        n_samples, n_channels = 300_000, 4
        source = self._make_source(n_samples, n_channels)
        out = tmp_path / "test.zarr"
        cfg = self._small_chunk_config()

        build_mipmap(
            source,
            out,
            sampling_frequency=30000.0,
            units="uV",
            config=BuildConfig(chunk_config=cfg, lsb_shift_raw=0, lsb_shift_pyramid=0),
        )

        root = zarr.open_group(str(out), mode="r", use_consolidated=None)
        s32 = np.asarray(root["minmax/s32"][:])
        s256 = np.asarray(root["minmax/s256"][:])

        # Block 0 of s256 covers blocks 0-7 of s32
        np.testing.assert_array_equal(s256[:, 0, 0], s32[:, 0, :8].min(axis=1))
        np.testing.assert_array_equal(s256[:, 1, 0], s32[:, 1, :8].max(axis=1))

    def test_single_level(self, tmp_path):
        """32768 samples → exactly 1024 blocks at stride 32, one level."""
        n_samples, n_channels = 32768, 2
        source = self._make_source(n_samples, n_channels)
        out = tmp_path / "test.zarr"
        cfg = self._small_chunk_config().for_channels(n_channels)

        build_mipmap(
            source,
            out,
            sampling_frequency=30000.0,
            units="uV",
            config=BuildConfig(chunk_config=cfg, lsb_shift_raw=0, lsb_shift_pyramid=0),
        )

        root = zarr.open_group(str(out), mode="r", use_consolidated=None)
        attrs = MipmapAttrs.from_dict(dict(root.attrs))
        assert attrs.strides == [32]

    def test_too_short_raises(self, tmp_path):
        source = self._make_source(100, 2)
        out = tmp_path / "test.zarr"

        with pytest.raises(ValueError, match="too short"):
            build_mipmap(
                source,
                out,
                sampling_frequency=30000.0,
                units="uV",
                config=BuildConfig(lsb_shift_raw=0, lsb_shift_pyramid=0),
            )

    def test_float32_dtype(self, tmp_path):
        n_samples, n_channels = 100_000, 4
        source = self._make_source(n_samples, n_channels, dtype="float32")
        out = tmp_path / "test.zarr"
        cfg = self._small_chunk_config()

        build_mipmap(
            source,
            out,
            sampling_frequency=30000.0,
            units="uV",
            dtype="float32",
            config=BuildConfig(chunk_config=cfg, lsb_shift_raw=0, lsb_shift_pyramid=0),
        )

        root = zarr.open_group(str(out), mode="r", use_consolidated=None)
        assert root["raw"].dtype == np.float32
        assert root["minmax/s32"].dtype == np.float32

    def test_non_multiple_of_stride(self, tmp_path):
        """n_samples not a multiple of 32 → tail block."""
        n_samples, n_channels = 32769, 2
        source = self._make_source(n_samples, n_channels)
        out = tmp_path / "test.zarr"
        cfg = self._small_chunk_config().for_channels(n_channels)

        build_mipmap(
            source,
            out,
            sampling_frequency=30000.0,
            units="uV",
            config=BuildConfig(chunk_config=cfg, lsb_shift_raw=0, lsb_shift_pyramid=0),
        )

        root = zarr.open_group(str(out), mode="r", use_consolidated=None)
        s32 = root["minmax/s32"]
        expected_blocks = n_blocks(n_samples, 32)
        assert s32.shape[2] == expected_blocks

        # Check tail block (last 1 sample)
        s32_data = np.asarray(s32[:])
        np.testing.assert_array_equal(s32_data[:, 0, -1], source[-1, :])
        np.testing.assert_array_equal(s32_data[:, 1, -1], source[-1, :])

    def test_dask_source(self, tmp_path):
        import dask.array as da

        n_samples, n_channels = 100_000, 4
        np_source = self._make_source(n_samples, n_channels)
        dask_source = da.from_array(np_source, chunks=(4096, 4))
        out = tmp_path / "test.zarr"
        cfg = self._small_chunk_config()

        build_mipmap(
            dask_source,
            out,
            sampling_frequency=30000.0,
            units="uV",
            config=BuildConfig(chunk_config=cfg, lsb_shift_raw=0, lsb_shift_pyramid=0),
        )

        root = zarr.open_group(str(out), mode="r", use_consolidated=None)
        np.testing.assert_array_equal(np.asarray(root["raw"][:]), np_source.T)


# ---------------------------------------------------------------------------
# add_pyramid integration tests
# ---------------------------------------------------------------------------


class TestAddPyramid:
    def _build_raw_zarr(self, path: Path, n_samples: int, n_channels: int, seed: int = 42):
        """Create a zarr store with just /raw for testing add_pyramid."""
        rng = np.random.default_rng(seed)
        data = rng.integers(-1000, 1000, size=(n_samples, n_channels), dtype=np.int16)

        root = zarr.open_group(str(path), mode="w", zarr_format=3)
        raw = root.create_array(
            "raw",
            shape=(n_samples, n_channels),
            dtype="int16",
            chunks=(4096, n_channels),
        )
        raw[:] = data
        return data

    def _small_chunk_config(self):
        return ChunkConfig(
            raw_chunk=(4, 1024),
            raw_shard=(4, 4096),
            minmax_chunk=(4, 2, 256),
            minmax_shard=(4, 2, 1024),
        )

    def test_creates_pyramid_levels(self, tmp_path):
        n_samples, n_channels = 100_000, 4
        raw_path = tmp_path / "raw.zarr"
        self._build_raw_zarr(raw_path, n_samples, n_channels)
        out = tmp_path / "pyramid.zarr"
        cfg = self._small_chunk_config()

        add_pyramid(
            raw_path,
            out,
            sampling_frequency=30000.0,
            units="uV",
            config=BuildConfig(chunk_config=cfg, lsb_shift_raw=0, lsb_shift_pyramid=0),
        )

        root = zarr.open_group(str(out), mode="r", use_consolidated=None)
        strides = compute_strides(n_samples)
        for stride in strides:
            path = minmax_path(stride).lstrip("/")
            arr = root[path]
            assert arr.shape == (n_channels, 2, n_blocks(n_samples, stride))

    def test_no_raw_in_output(self, tmp_path):
        n_samples, n_channels = 100_000, 4
        raw_path = tmp_path / "raw.zarr"
        self._build_raw_zarr(raw_path, n_samples, n_channels)
        out = tmp_path / "pyramid.zarr"
        cfg = self._small_chunk_config()

        add_pyramid(
            raw_path,
            out,
            sampling_frequency=30000.0,
            units="uV",
            config=BuildConfig(chunk_config=cfg, lsb_shift_raw=0, lsb_shift_pyramid=0),
        )

        root = zarr.open_group(str(out), mode="r", use_consolidated=None)
        assert "raw" not in dict(root)

    def test_raw_source_attr(self, tmp_path):
        n_samples, n_channels = 100_000, 4
        raw_path = tmp_path / "raw.zarr"
        self._build_raw_zarr(raw_path, n_samples, n_channels)
        out = tmp_path / "pyramid.zarr"
        cfg = self._small_chunk_config()

        add_pyramid(
            raw_path,
            out,
            sampling_frequency=30000.0,
            units="uV",
            config=BuildConfig(chunk_config=cfg, lsb_shift_raw=0, lsb_shift_pyramid=0),
        )

        root = zarr.open_group(str(out), mode="r", use_consolidated=None)
        attrs = MipmapAttrs.from_dict(dict(root.attrs))
        assert attrs.raw_source == str(raw_path)

    def test_values_match_build_mipmap(self, tmp_path):
        """Pyramid values from add_pyramid should match those from build_mipmap."""
        n_samples, n_channels = 100_000, 4
        rng = np.random.default_rng(42)
        source = rng.integers(-1000, 1000, size=(n_samples, n_channels), dtype=np.int16)
        cfg = self._small_chunk_config()

        # Build full mipmap
        full_path = tmp_path / "full.zarr"
        build_mipmap(
            source,
            full_path,
            sampling_frequency=30000.0,
            units="uV",
            config=BuildConfig(chunk_config=cfg, lsb_shift_raw=0, lsb_shift_pyramid=0),
        )

        # Build raw-only zarr, then add_pyramid
        raw_path = tmp_path / "raw.zarr"
        root = zarr.open_group(str(raw_path), mode="w", zarr_format=3)
        raw = root.create_array("raw", shape=source.shape, dtype="int16", chunks=(4096, 4))
        raw[:] = source

        pyr_path = tmp_path / "pyramid.zarr"
        add_pyramid(
            raw_path,
            pyr_path,
            sampling_frequency=30000.0,
            units="uV",
            config=BuildConfig(chunk_config=cfg, lsb_shift_raw=0, lsb_shift_pyramid=0),
        )

        # Compare pyramid values
        full_root = zarr.open_group(str(full_path), mode="r", use_consolidated=None)
        pyr_root = zarr.open_group(str(pyr_path), mode="r", use_consolidated=None)

        strides = compute_strides(n_samples)
        for stride in strides:
            path = minmax_path(stride).lstrip("/")
            np.testing.assert_array_equal(
                np.asarray(full_root[path][:]),
                np.asarray(pyr_root[path][:]),
            )

    def test_custom_raw_path(self, tmp_path):
        """Test add_pyramid with a non-default raw array path."""
        n_samples, n_channels = 100_000, 4
        rng = np.random.default_rng(42)
        data = rng.integers(-1000, 1000, size=(n_samples, n_channels), dtype=np.int16)

        raw_store_path = tmp_path / "source.zarr"
        root = zarr.open_group(str(raw_store_path), mode="w", zarr_format=3)
        arr = root.create_array("recording", shape=data.shape, dtype="int16", chunks=(4096, 4))
        arr[:] = data

        out = tmp_path / "pyramid.zarr"
        cfg = self._small_chunk_config()

        add_pyramid(
            raw_store_path,
            out,
            sampling_frequency=30000.0,
            units="uV",
            raw_path="recording",
            config=BuildConfig(chunk_config=cfg, lsb_shift_raw=0, lsb_shift_pyramid=0),
        )

        pyr_root = zarr.open_group(str(out), mode="r", use_consolidated=None)
        strides = compute_strides(n_samples)
        # Access nested path directly
        s32 = pyr_root[minmax_path(strides[0]).lstrip("/")]
        assert s32.shape == (n_channels, 2, n_blocks(n_samples, strides[0]))


# ---------------------------------------------------------------------------
# ensure_pyramid
# ---------------------------------------------------------------------------


class TestEnsurePyramid:
    def _small_chunk_config(self):
        return ChunkConfig(
            raw_chunk=(4, 1024),
            raw_shard=(4, 4096),
            minmax_chunk=(4, 2, 256),
            minmax_shard=(4, 2, 1024),
        )

    def _build_raw_zarr(self, path, n_samples, n_channels, seed=42):
        rng = np.random.default_rng(seed)
        data = rng.integers(-1000, 1000, size=(n_samples, n_channels), dtype=np.int16)
        root = zarr.open_group(str(path), mode="w", zarr_format=3)
        raw = root.create_array("raw", shape=data.shape, dtype="int16", chunks=(4096, n_channels))
        raw[:] = data
        return data

    def test_fresh_build(self, tmp_path):
        n_samples, n_channels = 100_000, 4
        raw_path = tmp_path / "raw.zarr"
        self._build_raw_zarr(raw_path, n_samples, n_channels)
        out = tmp_path / "pyramid.zarr"

        store = ensure_pyramid(
            raw_path,
            out,
            sampling_frequency=30000.0,
            units="uV",
            config=BuildConfig(chunk_config=self._small_chunk_config(), lsb_shift_raw=0, lsb_shift_pyramid=0),
        )

        assert store.n_samples == n_samples
        assert store.n_channels == n_channels
        assert store.sampling_frequency == 30000.0
        assert store.strides == compute_strides(n_samples)

    def test_skip_existing(self, tmp_path):
        n_samples, n_channels = 100_000, 4
        raw_path = tmp_path / "raw.zarr"
        self._build_raw_zarr(raw_path, n_samples, n_channels)
        out = tmp_path / "pyramid.zarr"
        cfg = self._small_chunk_config()

        store1 = ensure_pyramid(
            raw_path,
            out,
            sampling_frequency=30000.0,
            units="uV",
            config=BuildConfig(chunk_config=cfg, lsb_shift_raw=0, lsb_shift_pyramid=0),
        )
        # Get modification time of output zarr.json
        mtime = (out / "zarr.json").stat().st_mtime

        store2 = ensure_pyramid(
            raw_path,
            out,
            sampling_frequency=30000.0,
            units="uV",
            config=BuildConfig(chunk_config=cfg, lsb_shift_raw=0, lsb_shift_pyramid=0),
        )
        # zarr.json should not have been rewritten
        assert (out / "zarr.json").stat().st_mtime == mtime
        assert store2.n_samples == store1.n_samples

    def test_rebuild_on_stale_n_samples(self, tmp_path):
        n_channels = 4
        raw_path = tmp_path / "raw.zarr"
        self._build_raw_zarr(raw_path, 100_000, n_channels)
        out = tmp_path / "pyramid.zarr"
        cfg = self._small_chunk_config()

        store1 = ensure_pyramid(
            raw_path,
            out,
            sampling_frequency=30000.0,
            units="uV",
            config=BuildConfig(chunk_config=cfg, lsb_shift_raw=0, lsb_shift_pyramid=0),
        )
        assert store1.n_samples == 100_000

        # Replace raw with different n_samples
        self._build_raw_zarr(raw_path, 200_000, n_channels, seed=99)

        store2 = ensure_pyramid(
            raw_path,
            out,
            sampling_frequency=30000.0,
            units="uV",
            config=BuildConfig(chunk_config=cfg, lsb_shift_raw=0, lsb_shift_pyramid=0),
        )
        assert store2.n_samples == 200_000

    def test_rebuild_on_stale_strides(self, tmp_path):
        n_samples, n_channels = 100_000, 4
        raw_path = tmp_path / "raw.zarr"
        self._build_raw_zarr(raw_path, n_samples, n_channels)
        out = tmp_path / "pyramid.zarr"
        cfg = self._small_chunk_config()

        store1 = ensure_pyramid(
            raw_path,
            out,
            sampling_frequency=30000.0,
            units="uV",
            config=BuildConfig(chunk_config=cfg, lsb_shift_raw=0, lsb_shift_pyramid=0),
        )
        strides1 = store1.strides

        # Call again with different min_top_blocks → different strides
        store2 = ensure_pyramid(
            raw_path,
            out,
            sampling_frequency=30000.0,
            units="uV",
            config=BuildConfig(chunk_config=cfg, min_top_blocks=100, lsb_shift_raw=0, lsb_shift_pyramid=0),
        )
        strides2 = store2.strides
        assert strides1 != strides2

    def test_returns_mipmap_store(self, tmp_path):
        from ecephys_mipmap_zarr.read import MipmapStore

        n_samples, n_channels = 100_000, 4
        raw_path = tmp_path / "raw.zarr"
        self._build_raw_zarr(raw_path, n_samples, n_channels)
        out = tmp_path / "pyramid.zarr"

        store = ensure_pyramid(
            raw_path,
            out,
            sampling_frequency=30000.0,
            units="uV",
            config=BuildConfig(chunk_config=self._small_chunk_config(), lsb_shift_raw=0, lsb_shift_pyramid=0),
        )
        assert isinstance(store, MipmapStore)

    def test_raw_fallback_works(self, tmp_path):
        n_samples, n_channels = 100_000, 4
        raw_path = tmp_path / "raw.zarr"
        source = self._build_raw_zarr(raw_path, n_samples, n_channels)
        out = tmp_path / "pyramid.zarr"

        store = ensure_pyramid(
            raw_path,
            out,
            sampling_frequency=30000.0,
            units="uV",
            config=BuildConfig(chunk_config=self._small_chunk_config(), lsb_shift_raw=0, lsb_shift_pyramid=0),
        )
        # Zoomed in → raw fallback via raw_source
        result = store.get_window(0, 100, num_points=1000)
        assert result.is_raw
        np.testing.assert_array_equal(result.data, source[0:100, :].T)


# ---------------------------------------------------------------------------
# Quantized build tests
# ---------------------------------------------------------------------------


class TestBuildMipmapQuantized:
    def _small_chunk_config(self):
        return ChunkConfig(
            raw_chunk=(4, 1024),
            raw_shard=(4, 4096),
            minmax_chunk=(4, 2, 256),
            minmax_shard=(4, 2, 1024),
        )

    def test_quantized_dtypes(self, tmp_path):
        """Float32 source → uint16 raw + uint8 pyramids."""
        rng = np.random.default_rng(42)
        source = rng.standard_normal((100_000, 4)).astype(np.float32) * 100
        out = tmp_path / "test.zarr"
        cfg = self._small_chunk_config()

        build_mipmap(
            source,
            out,
            sampling_frequency=30000.0,
            units="uV",
            dtype="float32",
            config=BuildConfig(chunk_config=cfg, lsb_shift_raw=0, lsb_shift_pyramid=0),
            quantization=QuantizationConfig(quantize=True),
        )

        root = zarr.open_group(str(out), mode="r", use_consolidated=None)
        assert root["raw"].dtype == np.uint16
        assert root["minmax/s32"].dtype == np.uint8

    def test_quantized_attrs_present(self, tmp_path):
        rng = np.random.default_rng(42)
        source = rng.standard_normal((100_000, 4)).astype(np.float32) * 100
        out = tmp_path / "test.zarr"
        cfg = self._small_chunk_config()

        build_mipmap(
            source,
            out,
            sampling_frequency=30000.0,
            units="uV",
            dtype="float32",
            config=BuildConfig(chunk_config=cfg, lsb_shift_raw=0, lsb_shift_pyramid=0),
            quantization=QuantizationConfig(quantize=True),
        )

        root = zarr.open_group(str(out), mode="r", use_consolidated=None)
        raw_q = dict(root["raw"].attrs).get("quantization")
        assert raw_q is not None
        assert raw_q["quantized_dtype"] == "uint16"

        s32_q = dict(root["minmax/s32"].attrs).get("quantization")
        assert s32_q is not None
        assert s32_q["quantized_dtype"] == "uint8"

    def test_quantized_values_approximate(self, tmp_path):
        """Dequantized raw should approximately match original."""
        rng = np.random.default_rng(42)
        source = rng.standard_normal((100_000, 4)).astype(np.float32) * 100
        out = tmp_path / "test.zarr"
        cfg = self._small_chunk_config()

        build_mipmap(
            source,
            out,
            sampling_frequency=30000.0,
            units="uV",
            dtype="float32",
            config=BuildConfig(chunk_config=cfg, lsb_shift_raw=0, lsb_shift_pyramid=0),
            quantization=QuantizationConfig(quantize=True),
        )

        from ecephys_mipmap_zarr.read import open_mipmap

        store = open_mipmap(out)
        result = store.get_window(0, 100, num_points=1000)
        assert result.is_raw
        assert result.data.dtype == np.float32
        # uint16 quantization error should be small relative to the data range
        np.testing.assert_allclose(result.data, source[0:100, :].T, atol=0.1)

    def test_quantized_minmax_approximate(self, tmp_path):
        """Dequantized s32 min/max should approximately match float32 computation."""
        rng = np.random.default_rng(42)
        source = rng.standard_normal((100_000, 4)).astype(np.float32) * 100
        out = tmp_path / "test.zarr"
        cfg = self._small_chunk_config()

        build_mipmap(
            source,
            out,
            sampling_frequency=30000.0,
            units="uV",
            dtype="float32",
            config=BuildConfig(chunk_config=cfg, lsb_shift_raw=0, lsb_shift_pyramid=0),
            quantization=QuantizationConfig(quantize=True),
        )

        from ecephys_mipmap_zarr.read import open_mipmap

        store = open_mipmap(out)
        result = store.get_window(0, 100_000, num_points=100)
        assert not result.is_raw
        # Spot-check first block
        block0 = source[:32, :]
        # uint8 pyramid has coarser quantization but should be in the right ballpark
        np.testing.assert_allclose(result.data[:, 0, 0], block0.min(axis=0), atol=5.0)
        np.testing.assert_allclose(result.data[:, 1, 0], block0.max(axis=0), atol=5.0)

    def test_user_provided_quantization(self, tmp_path):
        """User-provided quantization params are used."""
        from ecephys_mipmap_zarr.metadata import RawQuantization

        rng = np.random.default_rng(42)
        source = rng.standard_normal((100_000, 4)).astype(np.float32) * 100
        out = tmp_path / "test.zarr"
        cfg = self._small_chunk_config()

        rq = RawQuantization(
            offset=[-500.0] * 4,
            scale=[0.02] * 4,
            source_dtype="float32",
            quantized_dtype="uint16",
        )

        build_mipmap(
            source,
            out,
            sampling_frequency=30000.0,
            units="uV",
            dtype="float32",
            config=BuildConfig(chunk_config=cfg, lsb_shift_raw=0, lsb_shift_pyramid=0),
            quantization=QuantizationConfig(quantize=True, raw_quantization=rq),
        )

        root = zarr.open_group(str(out), mode="r", use_consolidated=None)
        raw_q = dict(root["raw"].attrs)["quantization"]
        assert raw_q["offset"] == [-500.0] * 4
        assert raw_q["scale"] == [0.02] * 4


class TestAddPyramidQuantized:
    def _small_chunk_config(self):
        return ChunkConfig(
            raw_chunk=(4, 1024),
            raw_shard=(4, 4096),
            minmax_chunk=(4, 2, 256),
            minmax_shard=(4, 2, 1024),
        )

    def test_quantized_pyramid_levels(self, tmp_path):
        rng = np.random.default_rng(42)
        source = rng.standard_normal((100_000, 4)).astype(np.float32) * 100

        raw_path = tmp_path / "raw.zarr"
        root = zarr.open_group(str(raw_path), mode="w", zarr_format=3)
        raw = root.create_array("raw", shape=source.shape, dtype="float32", chunks=(4096, 4))
        raw[:] = source

        out = tmp_path / "pyramid.zarr"
        add_pyramid(
            raw_path,
            out,
            sampling_frequency=30000.0,
            units="uV",
            config=BuildConfig(chunk_config=self._small_chunk_config(), lsb_shift_raw=0, lsb_shift_pyramid=0),
            quantization=QuantizationConfig(quantize=True),
        )

        pyr_root = zarr.open_group(str(out), mode="r", use_consolidated=None)
        assert pyr_root["minmax/s32"].dtype == np.uint8
        assert dict(pyr_root["minmax/s32"].attrs).get("quantization") is not None


class TestBuildMipmapLsbShift:
    """Tests for LSB-shift compression."""

    def _small_chunk_config(self):
        return ChunkConfig(
            raw_chunk=(4, 1024),
            raw_shard=(4, 4096),
            minmax_chunk=(4, 2, 256),
            minmax_shard=(4, 2, 1024),
        )

    def test_explicit_shift(self, tmp_path):
        """Build with explicit lsb_shift values and verify stored data."""
        rng = np.random.default_rng(42)
        n_samples, n_channels = 100_000, 4
        # Values in range [-500, 500] — shift >>2 gives [-125, 125], fits int8
        source = rng.integers(-500, 500, size=(n_samples, n_channels), dtype=np.int16)
        out = tmp_path / "test.zarr"
        cfg = self._small_chunk_config()

        build_mipmap(
            source,
            out,
            sampling_frequency=30000.0,
            units="uV",
            config=BuildConfig(chunk_config=cfg, lsb_shift_raw=2, lsb_shift_pyramid=4),
        )

        root = zarr.open_group(str(out), mode="r", use_consolidated=None)

        # Raw should be int8 (shifted values fit in [-128, 127])
        raw = root["raw"]
        assert raw.dtype == np.int8
        assert raw.shape == (n_channels, n_samples)

        # Check lsb_shift attr
        assert dict(raw.attrs)["lsb_shift"] == 2
        assert dict(raw.attrs)["original_dtype"] == "int16"

        # Verify reconstruction: stored << shift ≈ original (within quantization error)
        stored_raw = np.asarray(raw[:])
        reconstructed = stored_raw.astype(np.int16) << 2
        original_ct = source.T  # (C, T)
        max_error = np.abs(reconstructed - original_ct).max()
        assert max_error <= (1 << 2)  # max error is one step size

        # Pyramid should have lsb_shift attr
        s32 = root["minmax/s32"]
        assert dict(s32.attrs)["lsb_shift"] == 4

    def test_shift_with_reader_roundtrip(self, tmp_path):
        """Reader correctly reconstructs shifted data."""
        from ecephys_mipmap_zarr.read import open_mipmap

        rng = np.random.default_rng(42)
        n_samples, n_channels = 100_000, 4
        source = rng.integers(-500, 500, size=(n_samples, n_channels), dtype=np.int16)
        out = tmp_path / "test.zarr"
        cfg = self._small_chunk_config()

        build_mipmap(
            source,
            out,
            sampling_frequency=30000.0,
            units="uV",
            config=BuildConfig(chunk_config=cfg, lsb_shift_raw=2, lsb_shift_pyramid=4),
        )

        store = open_mipmap(out)

        # Read raw window
        result = store.get_window(0, 1000, num_points=10000)
        assert result.is_raw
        assert result.data.dtype == np.int16
        # Reconstruction error bounded by step size
        original_ct = source[:1000, :].T
        max_error = np.abs(result.data - original_ct).max()
        assert max_error <= (1 << 2)

        # Read minmax window (zoomed out)
        result_mm = store.get_window(0, n_samples, num_points=100)
        assert not result_mm.is_raw
        assert result_mm.data.dtype == np.int16

    def test_no_shift_when_zero(self, tmp_path):
        """lsb_shift=0 produces lossless output (same as no shift)."""
        rng = np.random.default_rng(42)
        n_samples, n_channels = 100_000, 4
        source = rng.integers(-1000, 1000, size=(n_samples, n_channels), dtype=np.int16)
        out = tmp_path / "test.zarr"
        cfg = self._small_chunk_config()

        build_mipmap(
            source,
            out,
            sampling_frequency=30000.0,
            units="uV",
            config=BuildConfig(chunk_config=cfg, lsb_shift_raw=0, lsb_shift_pyramid=0),
        )

        root = zarr.open_group(str(out), mode="r", use_consolidated=None)
        raw = root["raw"]
        assert raw.dtype == np.int16  # no conversion
        assert "lsb_shift" not in dict(raw.attrs)
        np.testing.assert_array_equal(np.asarray(raw[:]), source.T)

    def test_entropy_based_shuffle(self, tmp_path):
        """Verify that entropy-based shuffle selection produces valid output."""
        rng = np.random.default_rng(42)
        n_samples, n_channels = 100_000, 4
        # Low-entropy data: only ~20 unique values after >>2
        source = rng.integers(-40, 40, size=(n_samples, n_channels), dtype=np.int16)
        out = tmp_path / "test.zarr"
        cfg = self._small_chunk_config()

        build_mipmap(
            source,
            out,
            sampling_frequency=30000.0,
            units="uV",
            config=BuildConfig(chunk_config=cfg, lsb_shift_raw=2, lsb_shift_pyramid=4),
        )

        # Just verify it builds and reads without error
        from ecephys_mipmap_zarr.read import open_mipmap

        store = open_mipmap(out)
        result = store.get_window(0, 1000, num_points=10000)
        assert result.data.shape[0] == n_channels

    def test_add_pyramid_with_shift(self, tmp_path):
        """add_pyramid applies LSB shift to pyramid levels."""
        rng = np.random.default_rng(42)
        n_samples, n_channels = 100_000, 4
        source = rng.integers(-500, 500, size=(n_samples, n_channels), dtype=np.int16)

        # Write raw store
        raw_path = tmp_path / "raw.zarr"
        root = zarr.open_group(str(raw_path), mode="w", zarr_format=3)
        raw = root.create_array("raw", shape=source.shape, dtype="int16", chunks=(4096, 4))
        raw[:] = source

        out = tmp_path / "pyramid.zarr"
        add_pyramid(
            raw_path,
            out,
            sampling_frequency=30000.0,
            units="uV",
            config=BuildConfig(chunk_config=self._small_chunk_config(), lsb_shift_pyramid=4, lsb_shift_raw=0),
        )

        pyr_root = zarr.open_group(str(out), mode="r", use_consolidated=None)
        s32 = pyr_root["minmax/s32"]
        assert dict(s32.attrs)["lsb_shift"] == 4
