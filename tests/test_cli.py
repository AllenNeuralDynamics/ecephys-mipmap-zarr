"""Tests for ecephys_mipmap_zarr.cli."""

import sys

import numpy as np
import pytest
import zarr

from ecephys_mipmap_zarr import cli as cli_module
from ecephys_mipmap_zarr.build import build_mipmap
from ecephys_mipmap_zarr.cli import AddPyramidSettings, BuildSettings, InfoSettings, ViewSettings
from ecephys_mipmap_zarr.metadata import BuildConfig, ChunkConfig, MipmapAttrs, QuantizationConfig, compute_strides

SMALL_CFG = ChunkConfig(
    raw_chunk=(4, 1024),
    raw_shard=(4, 4096),
    minmax_chunk=(4, 2, 256),
    minmax_shard=(4, 2, 1024),
)


def _parse_cli(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> dict[str, object]:
    """Run the real CLI parser and capture the parsed subcommand settings."""
    from dataclasses import asdict

    parsed: dict[str, object] = {}

    def capture(self: BuildSettings | AddPyramidSettings | InfoSettings | ViewSettings) -> None:
        parsed["type"] = type(self).__name__
        parsed["data"] = asdict(self)

    for cls in (BuildSettings, AddPyramidSettings, InfoSettings, ViewSettings):
        monkeypatch.setattr(cls, "cli_cmd", capture)

    monkeypatch.setattr(sys, "argv", argv)
    cli_module.main()
    return parsed


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
            source,
            out,
            sampling_frequency=30000.0,
            units="uV",
            config=BuildConfig(chunk_config=cfg, lsb_shift_raw=0, lsb_shift_pyramid=0),
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
            config=BuildConfig(chunk_config=cfg, lsb_shift_raw=0, lsb_shift_pyramid=0),
            quantization=QuantizationConfig(quantize=True),
        )

        settings = InfoSettings(store_path=str(out))
        settings.cli_cmd()  # Should show quantization info without errors


class TestCLIParsing:
    def test_info_accepts_positional_store_path(self, monkeypatch: pytest.MonkeyPatch):
        parsed = _parse_cli(monkeypatch, ["ecephys-mipmap", "info", "mipmap.zarr"])

        assert parsed == {
            "type": "InfoSettings",
            "data": {"store_path": "mipmap.zarr"},
        }

    def test_info_rejects_named_store_path_flag(self, monkeypatch: pytest.MonkeyPatch):
        with pytest.raises(SystemExit) as exc_info:
            _parse_cli(monkeypatch, ["ecephys-mipmap", "info", "--store-path", "mipmap.zarr"])

        assert exc_info.value.code == 2

    def test_build_accepts_bare_quantize_flag(self, monkeypatch: pytest.MonkeyPatch):
        parsed = _parse_cli(
            monkeypatch,
            [
                "ecephys-mipmap",
                "build",
                "--input",
                "recording.dat",
                "--output",
                "mipmap.zarr",
                "--sampling-frequency",
                "30000",
                "--n-samples",
                "10",
                "--n-channels",
                "2",
                "--quantize",
            ],
        )

        assert parsed["type"] == "BuildSettings"
        assert parsed["data"] == {
            "input": "recording.dat",
            "output": "mipmap.zarr",
            "from_analyzer": False,
            "raw_path_override": None,
            "keep_bad_channels": False,
            "start_seconds": 0.0,
            "duration_seconds": None,
            "sampling_frequency": 30000.0,
            "units": "uV",
            "n_samples": 10,
            "n_channels": 2,
            "dtype": "int16",
            "raw_path": "raw",
            "quantize": True,
            "quantize_pyramids": None,
            "compressor": "blosc",
            "metadata_from": None,
            "lsb_shift_raw": "auto",
            "lsb_shift_pyramid": "auto",
            "noise_floor": None,
            "clip_outliers": True,
            "prefetch_chunks": -1,
            "wavpack_decode_threads": 4,
            "fft_workers": -1,
        }

    def test_build_accepts_no_clip_outliers_flag(self, monkeypatch: pytest.MonkeyPatch):
        parsed = _parse_cli(
            monkeypatch,
            [
                "ecephys-mipmap",
                "build",
                "--input",
                "recording.dat",
                "--output",
                "mipmap.zarr",
                "--sampling-frequency",
                "30000",
                "--n-samples",
                "10",
                "--n-channels",
                "2",
                "--no-clip-outliers",
            ],
        )

        assert parsed["type"] == "BuildSettings"
        assert parsed["data"]["clip_outliers"] is False

    def test_view_accepts_positional_store_path_and_space_separated_channels(self, monkeypatch: pytest.MonkeyPatch):
        parsed = _parse_cli(
            monkeypatch,
            ["ecephys-mipmap", "view", "mipmap.zarr", "--channels", "1", "2", "3"],
        )

        assert parsed == {
            "type": "ViewSettings",
            "data": {
                "store_path": "mipmap.zarr",
                "channels": [1, 2, 3],
                "n_channels": 32,
                "width": 1200,
                "height": 800,
                "cache_dir": None,
                "cache_max_gb": 10.0,
                "channel_order": "depth",
                "spikes_from": None,
                "units": None,
                "remote": False,
                "port": None,
                "url_prefix": None,
                "allow_origin": None,
            },
        }
