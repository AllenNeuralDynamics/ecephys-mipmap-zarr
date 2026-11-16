"""Tests for ecephys_mipmap_zarr.metadata."""

import dataclasses

import numpy as np
import pytest

from ecephys_mipmap_zarr.metadata import (
    DEFAULT_CHUNK_CONFIG,
    ChunkConfig,
    LevelQuantization,
    MipmapAttrs,
    RawQuantization,
    compute_quantization_params,
    compute_strides,
    dequantize_array,
    minmax_path,
    n_blocks,
    quantize_array,
    select_level,
)

# ---------------------------------------------------------------------------
# compute_strides
# ---------------------------------------------------------------------------


class TestComputeStrides:
    def test_162m_samples(self):
        """Standard ~1.5h Neuropixels at 30 kHz (162M samples)."""
        result = compute_strides(162_000_000)
        assert result == [32, 256, 2048, 16384, 131072]

    def test_30m_samples(self):
        """~17 min recording.  131072 would give 229 blocks < 1024."""
        result = compute_strides(30_000_000)
        assert result == [32, 256, 2048, 16384]

    def test_exact_threshold(self):
        """32768 / 32 = 1024 exactly — should include stride 32."""
        result = compute_strides(32768)
        assert result == [32]

    def test_too_short(self):
        """100 samples — too few even for base level."""
        result = compute_strides(100)
        assert result == []

    def test_very_long(self):
        """1 billion samples — includes 131072 (7630 blocks)."""
        result = compute_strides(1_000_000_000)
        assert result == [32, 256, 2048, 16384, 131072]

    def test_negative_n_samples(self):
        with pytest.raises(ValueError, match="n_samples must be positive"):
            compute_strides(-1)

    def test_zero_n_samples(self):
        with pytest.raises(ValueError, match="n_samples must be positive"):
            compute_strides(0)

    def test_bad_base(self):
        with pytest.raises(ValueError, match="base must be positive"):
            compute_strides(1_000_000, base=0)

    def test_bad_factor(self):
        with pytest.raises(ValueError, match="factor must be > 1"):
            compute_strides(1_000_000, factor=1)

    def test_bad_min_top_blocks(self):
        with pytest.raises(ValueError, match="min_top_blocks must be positive"):
            compute_strides(1_000_000, min_top_blocks=0)

    def test_custom_params(self):
        """Custom base=16, factor=4."""
        result = compute_strides(1_000_000, base=16, factor=4)
        # 16, 64, 256, 1024, 4096, 16384, ...
        # 1M/16=62500, 1M/64=15625, 1M/256=3907, 1M/1024=977 < 1024
        assert result == [16, 64, 256]


# ---------------------------------------------------------------------------
# n_blocks
# ---------------------------------------------------------------------------


class TestNBlocks:
    def test_exact_division(self):
        assert n_blocks(1024, 32) == 32

    def test_remainder(self):
        assert n_blocks(1025, 32) == 33

    def test_one_sample(self):
        assert n_blocks(1, 32) == 1

    def test_zero_stride(self):
        with pytest.raises(ValueError, match="stride must be positive"):
            n_blocks(100, 0)

    def test_negative_stride(self):
        with pytest.raises(ValueError, match="stride must be positive"):
            n_blocks(100, -1)


# ---------------------------------------------------------------------------
# minmax_path
# ---------------------------------------------------------------------------


class TestMinmaxPath:
    def test_base_stride(self):
        assert minmax_path(32) == "/minmax/s32"

    def test_higher_stride(self):
        assert minmax_path(256) == "/minmax/s256"

    def test_large_stride(self):
        assert minmax_path(131072) == "/minmax/s131072"


# ---------------------------------------------------------------------------
# select_level
# ---------------------------------------------------------------------------


class TestSelectLevel:
    STRIDES = [32, 256, 2048]

    def test_picks_coarsest_valid(self):
        """spp=10000: all levels pass, picks coarsest (2048)."""
        assert select_level(self.STRIDES, 10000.0) == 2048

    def test_mid_zoom(self):
        """spp=1000: 1000/32=31.25 passes, 1000/256=3.9 passes, 1000/2048=0.49 fails → stride 256."""
        assert select_level(self.STRIDES, 1000.0) == 256

    def test_raw_fallback(self):
        """spp=10: 10/32=0.3 fails → None (use raw)."""
        assert select_level(self.STRIDES, 10.0) is None

    def test_empty_strides(self):
        assert select_level([], 1000.0) is None

    def test_all_pass(self):
        """Very high spp — takes coarsest available."""
        assert select_level([32, 256], 100000.0) == 256

    def test_custom_oversample(self):
        """spp=1000, oversample=2: 1000/256=3.9 >= 2 → picks 256."""
        assert select_level(self.STRIDES, 1000.0, oversample=2.0) == 256

    def test_exact_threshold(self):
        """spp=128, oversample=4: 128/32=4.0 exactly → passes."""
        assert select_level(self.STRIDES, 128.0) == 32


# ---------------------------------------------------------------------------
# ChunkConfig
# ---------------------------------------------------------------------------


class TestChunkConfig:
    def test_defaults(self):
        cfg = DEFAULT_CHUNK_CONFIG
        assert cfg.raw_chunk == (64, 16384)
        assert cfg.raw_shard == (64, 1_048_576)
        assert cfg.minmax_chunk == (64, 2, 4096)
        assert cfg.minmax_shard == (64, 2, 32768)

    def test_for_channels_fewer(self):
        cfg = DEFAULT_CHUNK_CONFIG.for_channels(16)
        assert cfg.raw_chunk == (16, 16384)
        assert cfg.raw_shard == (16, 1_048_576)
        assert cfg.minmax_chunk == (16, 2, 4096)
        assert cfg.minmax_shard == (16, 2, 32768)

    def test_for_channels_more(self):
        cfg = DEFAULT_CHUNK_CONFIG.for_channels(384)
        assert cfg.raw_chunk == (64, 16384)
        assert cfg.raw_shard == (64, 1_048_576)
        assert cfg.minmax_chunk == (64, 2, 4096)
        assert cfg.minmax_shard == (64, 2, 32768)

    def test_for_channels_exact(self):
        cfg = DEFAULT_CHUNK_CONFIG.for_channels(64)
        assert cfg == DEFAULT_CHUNK_CONFIG

    def test_for_channels_one(self):
        cfg = DEFAULT_CHUNK_CONFIG.for_channels(1)
        assert cfg.raw_chunk[0] == 1
        assert cfg.minmax_chunk[0] == 1

    def test_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            DEFAULT_CHUNK_CONFIG.raw_chunk = (1, 1)  # type: ignore[misc]

    def test_custom_config(self):
        cfg = ChunkConfig(raw_chunk=(32, 8192), raw_shard=(32, 524288))
        assert cfg.raw_chunk == (32, 8192)
        assert cfg.raw_shard == (32, 524288)
        # minmax fields still default
        assert cfg.minmax_chunk == (64, 2, 4096)


# ---------------------------------------------------------------------------
# MipmapAttrs
# ---------------------------------------------------------------------------


class TestMipmapAttrs:
    def test_roundtrip(self):
        attrs = MipmapAttrs(
            sampling_frequency=30000.0,
            units="uV",
            strides=[32, 256, 2048],
            n_samples=162_000_000,
            n_channels=384,
        )
        d = attrs.to_dict()
        restored = MipmapAttrs.from_dict(d)
        assert restored == attrs

    def test_default_dtype(self):
        attrs = MipmapAttrs(
            sampling_frequency=30000.0,
            units="uV",
            strides=[32],
            n_samples=100000,
            n_channels=64,
        )
        assert attrs.dtype == "int16"
        assert attrs.raw_source is None

    def test_explicit_dtype(self):
        attrs = MipmapAttrs(
            sampling_frequency=30000.0,
            units="uV",
            strides=[32],
            n_samples=100000,
            n_channels=64,
            dtype="float32",
        )
        assert attrs.dtype == "float32"

    def test_frozen(self):
        attrs = MipmapAttrs(
            sampling_frequency=30000.0,
            units="uV",
            strides=[32],
            n_samples=100000,
            n_channels=64,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            attrs.n_samples = 0  # type: ignore[misc]

    def test_from_dict_missing_key(self):
        with pytest.raises(KeyError):
            MipmapAttrs.from_dict({"sampling_frequency": 30000.0})

    def test_from_dict_default_dtype(self):
        """dtype defaults to int16 when not in dict."""
        d = {
            "sampling_frequency": 30000.0,
            "units": "uV",
            "strides": [32],
            "n_samples": 100000,
            "n_channels": 64,
        }
        attrs = MipmapAttrs.from_dict(d)
        assert attrs.dtype == "int16"

    def test_to_dict_keys(self):
        attrs = MipmapAttrs(
            sampling_frequency=30000.0,
            units="uV",
            strides=[32, 256],
            n_samples=162_000_000,
            n_channels=384,
        )
        d = attrs.to_dict()
        assert set(d.keys()) == {
            "sampling_frequency",
            "units",
            "strides",
            "n_samples",
            "n_channels",
            "dtype",
            "source_metadata",
            "raw_source",
            "raw_path",
            "t_start",
        }

    def test_raw_source(self):
        attrs = MipmapAttrs(
            sampling_frequency=30000.0,
            units="uV",
            strides=[32, 256],
            n_samples=162_000_000,
            n_channels=384,
            raw_source="s3://bucket/path/to/raw.zarr",
        )
        d = attrs.to_dict()
        restored = MipmapAttrs.from_dict(d)
        assert restored.raw_source == "s3://bucket/path/to/raw.zarr"
        assert restored == attrs

    def test_raw_source_none_roundtrip(self):
        attrs = MipmapAttrs(
            sampling_frequency=30000.0,
            units="uV",
            strides=[32],
            n_samples=100000,
            n_channels=64,
        )
        d = attrs.to_dict()
        restored = MipmapAttrs.from_dict(d)
        assert restored.raw_source is None
        assert restored == attrs


# ---------------------------------------------------------------------------
# LevelQuantization
# ---------------------------------------------------------------------------


class TestRawQuantization:
    def test_roundtrip(self):
        rq = RawQuantization(
            offset=[-100.0, -200.0, -150.0],
            scale=[0.01, 0.012, 0.011],
            source_dtype="float32",
            quantized_dtype="uint16",
        )
        d = rq.to_dict()
        restored = RawQuantization.from_dict(d)
        assert restored == rq

    def test_frozen(self):
        rq = RawQuantization(offset=[0.0], scale=[1.0], source_dtype="float32", quantized_dtype="uint16")
        with pytest.raises(dataclasses.FrozenInstanceError):
            rq.scale = [2.0]  # type: ignore[misc]

    def test_from_dict_missing_key(self):
        with pytest.raises(KeyError):
            RawQuantization.from_dict({"offset": [0.0]})

    def test_to_dict_keys(self):
        rq = RawQuantization(offset=[0.0], scale=[1.0], source_dtype="float32", quantized_dtype="uint16")
        d = rq.to_dict()
        assert set(d.keys()) == {"offset", "scale", "source_dtype", "quantized_dtype"}


class TestLevelQuantization:
    def test_roundtrip(self):
        lq = LevelQuantization(
            min_offset=[-300.0, -250.0],
            max_offset=[200.0, 180.0],
            scale=[0.005, 0.004],
            source_dtype="float32",
            quantized_dtype="uint8",
        )
        d = lq.to_dict()
        restored = LevelQuantization.from_dict(d)
        assert restored == lq

    def test_frozen(self):
        lq = LevelQuantization(
            min_offset=[0.0],
            max_offset=[0.0],
            scale=[1.0],
            source_dtype="float32",
            quantized_dtype="uint8",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            lq.scale = [2.0]  # type: ignore[misc]

    def test_from_dict_missing_key(self):
        with pytest.raises(KeyError):
            LevelQuantization.from_dict({"min_offset": [0.0]})

    def test_to_dict_keys(self):
        lq = LevelQuantization(
            min_offset=[-100.0],
            max_offset=[100.0],
            scale=[1.0],
            source_dtype="float32",
            quantized_dtype="uint8",
        )
        d = lq.to_dict()
        assert set(d.keys()) == {
            "min_offset",
            "max_offset",
            "scale",
            "source_dtype",
            "quantized_dtype",
        }


# ---------------------------------------------------------------------------
# Quantization helpers
# ---------------------------------------------------------------------------


class TestComputeQuantizationParams:
    def test_basic(self):
        mins = np.array([-500.0, -400.0])
        maxs = np.array([500.0, 400.0])
        offset, scale = compute_quantization_params(mins, maxs, "uint16")
        # offset should be slightly below mins
        assert offset[0] < -500.0
        assert offset[1] < -400.0
        # scale should map the range to 0..65535
        assert scale[0] > 0
        assert scale[1] > 0
        # Reconstructed range should cover the original
        assert offset[0] + scale[0] * 65535 > 500.0

    def test_constant_channel(self):
        """Constant channel should not produce zero scale."""
        mins = np.array([100.0, 0.0])
        maxs = np.array([100.0, 200.0])
        offset, scale = compute_quantization_params(mins, maxs, "uint8")
        assert scale[0] > 0  # constant channel gets scale=1.0, not 0

    def test_uint8_range(self):
        mins = np.array([0.0])
        maxs = np.array([255.0])
        offset, scale = compute_quantization_params(mins, maxs, "uint8", headroom=0.0)
        assert scale[0] == pytest.approx(1.0)
        assert offset[0] == pytest.approx(0.0)


class TestQuantizeArray:
    def test_basic_roundtrip(self):
        """Quantize then dequantize should approximately recover original."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((4, 100)).astype(np.float32) * 100
        mins = data.min(axis=1)
        maxs = data.max(axis=1)
        offset, scale = compute_quantization_params(mins, maxs, "uint16")

        quantized = quantize_array(data, offset, scale, "uint16")
        assert quantized.dtype == np.uint16

        recovered = dequantize_array(quantized, offset, scale)
        assert recovered.dtype == np.float32
        np.testing.assert_allclose(recovered, data, atol=scale.max())

    def test_uint8_roundtrip(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((2, 50)).astype(np.float32) * 500
        mins = data.min(axis=1)
        maxs = data.max(axis=1)
        offset, scale = compute_quantization_params(mins, maxs, "uint8")

        quantized = quantize_array(data, offset, scale, "uint8")
        assert quantized.dtype == np.uint8

        recovered = dequantize_array(quantized, offset, scale)
        np.testing.assert_allclose(recovered, data, atol=scale.max() * 1.5)

    def test_clipping(self):
        """Values outside the quantization range should be clipped."""
        offset = np.array([0.0])
        scale = np.array([1.0])
        data = np.array([[-10.0, 300.0]])
        quantized = quantize_array(data, offset, scale, "uint8")
        assert quantized[0, 0] == 0
        assert quantized[0, 1] == 255

    def test_3d_minmax(self):
        """Quantize/dequantize works on (C, 2, B) arrays."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((4, 2, 10)).astype(np.float32) * 100
        offset = np.array([0.0, 0.0, 0.0, 0.0])
        scale = np.array([1.0, 1.0, 1.0, 1.0])
        quantized = quantize_array(data, offset, scale, "uint8")
        assert quantized.shape == (4, 2, 10)
        assert quantized.dtype == np.uint8


# ---------------------------------------------------------------------------
# Noise estimation and LSB shift
# ---------------------------------------------------------------------------


class TestEstimateHfNoise:
    def test_known_noise(self):
        from ecephys_mipmap_zarr.metadata import estimate_hf_noise

        # Create signal with known noise level
        rng = np.random.default_rng(42)
        noise_std = 5.0
        data = (rng.standard_normal((4, 100_000)) * noise_std).astype(np.float64)
        # White noise: HF estimate should be close to the true RMS
        est = estimate_hf_noise(data, fs=30000.0)
        assert est > noise_std * 0.5
        assert est < noise_std * 2.0

    def test_returns_positive(self):
        from ecephys_mipmap_zarr.metadata import estimate_hf_noise

        rng = np.random.default_rng(42)
        data = rng.integers(-100, 100, size=(4, 30000), dtype=np.int16)
        est = estimate_hf_noise(data.astype(np.float64), fs=30000.0)
        assert est > 0


class TestComputeOptimalLsbShift:
    def test_known_noise_floor(self):
        from ecephys_mipmap_zarr.metadata import compute_optimal_lsb_shift

        rng = np.random.default_rng(42)
        data = rng.integers(-100, 100, size=(4, 30000), dtype=np.int16)
        # With noise_floor=4: max_step = 4 * sqrt(12) / 2 = 6.93, shift = floor(log2(6.93)) = 2
        shift = compute_optimal_lsb_shift(data, noise_floor=4.0)
        assert shift == 2

    def test_high_noise(self):
        from ecephys_mipmap_zarr.metadata import compute_optimal_lsb_shift

        rng = np.random.default_rng(42)
        data = rng.integers(-100, 100, size=(4, 30000), dtype=np.int16)
        # noise=10: max_step = 10 * sqrt(12) / 2 = 17.3, floor(log2(17.3)) = 4
        shift = compute_optimal_lsb_shift(data, noise_floor=10.0)
        assert shift == 4

    def test_zero_noise(self):
        from ecephys_mipmap_zarr.metadata import compute_optimal_lsb_shift

        rng = np.random.default_rng(42)
        data = rng.integers(-100, 100, size=(4, 30000), dtype=np.int16)
        shift = compute_optimal_lsb_shift(data, noise_floor=0.0)
        assert shift == 0

    def test_lossless(self):
        from ecephys_mipmap_zarr.metadata import compute_optimal_lsb_shift

        rng = np.random.default_rng(42)
        data = rng.integers(-100, 100, size=(4, 30000), dtype=np.int16)
        shift = compute_optimal_lsb_shift(data, noise_floor=0.1)
        assert shift == 0


class TestComputeEntropy:
    def test_constant_data(self):
        from ecephys_mipmap_zarr.metadata import compute_entropy

        data = np.full((4, 1000), 42, dtype=np.int16)
        ent = compute_entropy(data)
        assert ent == 0.0

    def test_binary_data(self):
        from ecephys_mipmap_zarr.metadata import compute_entropy

        rng = np.random.default_rng(42)
        data = rng.integers(0, 2, size=(4, 10000), dtype=np.int16)
        ent = compute_entropy(data)
        assert 0.9 < ent < 1.1  # ~1 bit for binary

    def test_int16_ephys_like(self):
        from ecephys_mipmap_zarr.metadata import compute_entropy

        rng = np.random.default_rng(42)
        data = rng.integers(-50, 50, size=(4, 100000), dtype=np.int16)
        ent = compute_entropy(data)
        assert 5 < ent < 8  # ~6.6 bits for uniform [-50, 50]
