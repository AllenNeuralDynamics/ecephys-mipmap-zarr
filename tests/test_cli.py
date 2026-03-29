"""Tests for ecephys_mipmap_zarr.cli."""

import numpy as np
import zarr

from ecephys_mipmap_zarr.build import build_mipmap
from ecephys_mipmap_zarr.cli import AddPyramidSettings, BuildSettings, InfoSettings
from ecephys_mipmap_zarr.metadata import ChunkConfig, MipmapAttrs, compute_strides

SMALL_CFG = ChunkConfig(
    raw_chunk=(4, 1024),
    raw_shard=(4, 4096),
    minmax_chunk=(4, 2, 256),
    minmax_shard=(4, 2, 1024),
)


class TestBuildFromBinary:
    def test_basic(self, tmp_path):
        rng = np.random.default_rng(42)
        data = rng.integers(-1000, 1000, size=(100_000, 4), dtype=np.int16)
        dat_path = tmp_path / "recording.dat"
        data.tofile(dat_path)

        out = tmp_path / "mipmap.zarr"
        settings = BuildSettings(
            input=str(dat_path),
            output=str(out),
            sampling_frequency=30000.0,
            n_samples=100_000,
            n_channels=4,
            dtype="int16",
        )
        settings.cli_cmd()

        root = zarr.open_group(str(out), mode="r", use_consolidated=None)
        assert root["raw"].shape == (4, 100_000)
        attrs = MipmapAttrs.from_dict(dict(root.attrs))
        assert attrs.sampling_frequency == 30000.0
        assert attrs.strides == compute_strides(100_000)

    def test_float32_quantized(self, tmp_path):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((100_000, 4)).astype(np.float32) * 100
        dat_path = tmp_path / "recording.dat"
        data.tofile(dat_path)

        out = tmp_path / "mipmap.zarr"
        settings = BuildSettings(
            input=str(dat_path),
            output=str(out),
            sampling_frequency=30000.0,
            n_samples=100_000,
            n_channels=4,
            dtype="float32",
            quantize=True,
        )
        settings.cli_cmd()

        root = zarr.open_group(str(out), mode="r", use_consolidated=None)
        assert root["raw"].dtype == np.uint16
        assert root["minmax/s32"].dtype == np.uint8

    def test_missing_params_raises(self, tmp_path):
        dat_path = tmp_path / "recording.dat"
        np.zeros(100, dtype=np.int16).tofile(dat_path)

        import pytest

        settings = BuildSettings(
            input=str(dat_path),
            output=str(tmp_path / "out.zarr"),
            sampling_frequency=30000.0,
        )
        with pytest.raises(SystemExit):
            settings.cli_cmd()


class TestBuildFromZarr:
    def test_basic(self, tmp_path):
        rng = np.random.default_rng(42)
        data = rng.integers(-1000, 1000, size=(100_000, 4), dtype=np.int16)

        raw_path = tmp_path / "raw.zarr"
        root = zarr.open_group(str(raw_path), mode="w", zarr_format=3)
        raw = root.create_array("raw", shape=data.shape, dtype="int16", chunks=(4096, 4))
        raw[:] = data

        out = tmp_path / "mipmap.zarr"
        settings = BuildSettings(
            input=str(raw_path),
            output=str(out),
            sampling_frequency=30000.0,
        )
        settings.cli_cmd()

        out_root = zarr.open_group(str(out), mode="r", use_consolidated=None)
        assert out_root["raw"].shape == (4, 100_000)


class TestAddPyramidCLI:
    def test_basic(self, tmp_path):
        rng = np.random.default_rng(42)
        data = rng.integers(-1000, 1000, size=(100_000, 4), dtype=np.int16)

        raw_path = tmp_path / "raw.zarr"
        root = zarr.open_group(str(raw_path), mode="w", zarr_format=3)
        raw = root.create_array("raw", shape=data.shape, dtype="int16", chunks=(4096, 4))
        raw[:] = data

        out = tmp_path / "pyramid.zarr"
        settings = AddPyramidSettings(
            raw_store=str(raw_path),
            output=str(out),
            sampling_frequency=30000.0,
        )
        settings.cli_cmd()

        pyr_root = zarr.open_group(str(out), mode="r", use_consolidated=None)
        attrs = MipmapAttrs.from_dict(dict(pyr_root.attrs))
        assert attrs.raw_source == str(raw_path)
        assert len(attrs.strides) > 0


class TestInfoCLI:
    def test_basic(self, tmp_path, capsys):
        rng = np.random.default_rng(42)
        source = rng.integers(-1000, 1000, size=(100_000, 4), dtype=np.int16)
        out = tmp_path / "mipmap.zarr"
        cfg = SMALL_CFG

        build_mipmap(
            source, out, sampling_frequency=30000.0, units="uV", chunk_config=cfg, lsb_shift_raw=0, lsb_shift_pyramid=0
        )

        settings = InfoSettings(store_path=str(out))
        settings.cli_cmd()

        # rich prints to its own console, not capsys — just verify no exception

    def test_quantized_store(self, tmp_path):
        rng = np.random.default_rng(42)
        source = rng.standard_normal((100_000, 4)).astype(np.float32) * 100
        out = tmp_path / "mipmap.zarr"
        cfg = SMALL_CFG

        build_mipmap(
            source,
            out,
            sampling_frequency=30000.0,
            units="uV",
            dtype="float32",
            chunk_config=cfg,
            quantize=True,
            lsb_shift_raw=0,
            lsb_shift_pyramid=0,
        )

        settings = InfoSettings(store_path=str(out))
        settings.cli_cmd()  # Should show quantization info without errors
