"""Tests for ecephys_mipmap_zarr.viewer."""

from unittest.mock import patch

import numpy as np

from ecephys_mipmap_zarr.build import build_mipmap
from ecephys_mipmap_zarr.metadata import LOSSLESS_ENCODING, BuildConfig, ChunkConfig
from ecephys_mipmap_zarr.read import open_mipmap
from ecephys_mipmap_zarr.viewer import TraceViewer, view

SMALL_CFG = ChunkConfig(
    raw_chunk=(4, 1024),
    raw_shard=(4, 4096),
    minmax_chunk=(4, 2, 256),
    minmax_shard=(4, 2, 1024),
)


def _make_store(tmp_path, n_samples=100_000, n_channels=8):
    rng = np.random.default_rng(42)
    source = rng.integers(-1000, 1000, size=(n_samples, n_channels), dtype=np.int16)
    out = tmp_path / "test.zarr"
    build_mipmap(
        source,
        out,
        sampling_frequency=30000.0,
        units="uV",
        config=BuildConfig(chunk_config=SMALL_CFG),
        encoding=LOSSLESS_ENCODING,
    )
    return open_mipmap(out), out


class TestTraceViewerCreation:
    def test_creates(self, tmp_path):
        store, _ = _make_store(tmp_path)
        viewer = TraceViewer(store, n_channels=4)
        assert viewer._fig is not None
        assert len(viewer.channels) == 4
        # Glyphs pre-allocated for max channels (8 in test store)
        assert len(viewer._envelope_glyphs) == viewer._max_glyphs
        assert len(viewer._line_glyphs) == viewer._max_glyphs

    def test_default_channels(self, tmp_path):
        store, _ = _make_store(tmp_path, n_channels=4)
        viewer = TraceViewer(store)
        assert viewer.channels == [0, 1, 2, 3]

    def test_custom_channels(self, tmp_path):
        store, _ = _make_store(tmp_path, n_channels=8)
        viewer = TraceViewer(store, channels=[2, 5, 7])
        assert viewer.channels == [2, 5, 7]
        assert len(viewer._envelope_glyphs) == viewer._max_glyphs

    def test_auto_spacing(self, tmp_path):
        store, _ = _make_store(tmp_path)
        viewer = TraceViewer(store, n_channels=4)
        assert viewer.channel_spacing > 0

    def test_custom_spacing(self, tmp_path):
        store, _ = _make_store(tmp_path)
        viewer = TraceViewer(store, n_channels=4, channel_spacing=500.0)
        assert viewer.channel_spacing == 500.0


class TestTraceViewerRendering:
    def test_initial_render_has_data(self, tmp_path):
        store, _ = _make_store(tmp_path)
        viewer = TraceViewer(store, n_channels=4)
        # After initial render, at least one source should have data
        env_has_data = len(viewer._envelope_source.data.get("x", [])) > 0
        line_has_data = len(viewer._line_source.data.get("x", [])) > 0
        assert env_has_data or line_has_data

    def test_update_data(self, tmp_path):
        store, _ = _make_store(tmp_path)
        viewer = TraceViewer(store, n_channels=4)
        viewer._update_data(0.0, 1.0)
        env_has_data = len(viewer._envelope_source.data.get("x", [])) > 0
        line_has_data = len(viewer._line_source.data.get("x", [])) > 0
        assert env_has_data or line_has_data

    def test_set_gain(self, tmp_path):
        store, _ = _make_store(tmp_path)
        viewer = TraceViewer(store, n_channels=4)
        viewer.set_gain(2.0)
        assert viewer._gain == 2.0
        # Gain affects data scaling, not offsets (offsets stay constant)


class TestViewFunction:
    def test_from_path(self, tmp_path):
        _, path = _make_store(tmp_path)
        with patch.object(TraceViewer, "show"):
            viewer = view(path, n_channels=4)
            assert isinstance(viewer, TraceViewer)

    def test_from_store(self, tmp_path):
        store, _ = _make_store(tmp_path)
        with patch.object(TraceViewer, "show"):
            viewer = view(store, n_channels=4)
            assert isinstance(viewer, TraceViewer)


class TestPanel:
    def test_panel_returns_layout(self, tmp_path):
        import panel as pn

        store, _ = _make_store(tmp_path)
        viewer = TraceViewer(store, n_channels=4)
        layout = viewer.panel()
        assert isinstance(layout, pn.Row)
