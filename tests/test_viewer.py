"""Tests for ecephys_mipmap_zarr.viewer."""

from unittest.mock import patch

import numpy as np

from ecephys_mipmap_zarr.build import build_mipmap
from ecephys_mipmap_zarr.metadata import ChunkConfig
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
        chunk_config=SMALL_CFG,
        lsb_shift_raw=0,
        lsb_shift_pyramid=0,
    )
    return open_mipmap(out), out


class TestTraceViewerCreation:
    def test_creates(self, tmp_path):
        store, _ = _make_store(tmp_path)
        viewer = TraceViewer(store, n_channels=4)
        assert viewer._fig is not None
        assert len(viewer.channels) == 4
        assert len(viewer._envelope_sources) == 4
        assert len(viewer._line_sources) == 4

    def test_default_channels(self, tmp_path):
        store, _ = _make_store(tmp_path, n_channels=4)
        viewer = TraceViewer(store)
        assert viewer.channels == [0, 1, 2, 3]

    def test_custom_channels(self, tmp_path):
        store, _ = _make_store(tmp_path, n_channels=8)
        viewer = TraceViewer(store, channels=[2, 5, 7])
        assert viewer.channels == [2, 5, 7]
        assert len(viewer._envelope_sources) == 3

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
        # After initial render, at least one glyph type should be visible with data
        has_data = False
        for src in viewer._envelope_sources:
            if len(src.data["x"]) > 0:
                has_data = True
                break
        if not has_data:
            for src in viewer._line_sources:
                if len(src.data["x"]) > 0:
                    has_data = True
                    break
        assert has_data

    def test_update_data(self, tmp_path):
        store, _ = _make_store(tmp_path)
        viewer = TraceViewer(store, n_channels=4)
        # Simulate a range change
        viewer._update_data(0.0, 1.0)
        # Should have data in sources
        for src in viewer._envelope_sources + viewer._line_sources:
            if len(src.data.get("x", [])) > 0:
                return
        raise AssertionError("No sources updated after _update_data")

    def test_set_gain(self, tmp_path):
        store, _ = _make_store(tmp_path)
        viewer = TraceViewer(store, n_channels=4)
        old_offsets = viewer._offsets.copy()
        viewer.set_gain(2.0)
        assert viewer._gain == 2.0
        assert not np.array_equal(viewer._offsets, old_offsets)


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
