"""Tests for the parallel-median CMR monkey-patch."""

from __future__ import annotations

import numpy as np
import pytest

from ecephys_mipmap_zarr import _fast_cmr


class TestParallelReduceMatchesNumpy:
    def test_median_axis1_matches_stock(self):
        rng = np.random.default_rng(0)
        T, C = 30_000, 384
        x = rng.standard_normal((T, C)).astype(np.float32) * 100.0
        ref = np.median(x, axis=1)
        out = _fast_cmr._parallel_reduce_axis1(x, np.median)
        # Median is deterministic; output must be bitwise identical.
        np.testing.assert_array_equal(out, ref)

    def test_mean_axis1_matches_stock(self):
        rng = np.random.default_rng(0)
        T, C = 30_000, 384
        x = rng.standard_normal((T, C)).astype(np.float32) * 100.0
        ref = np.mean(x, axis=1)
        out = _fast_cmr._parallel_reduce_axis1(x, np.mean)
        # Mean isn't associative in float — tolerate single-ULP differences
        # from sum order across block boundaries.
        np.testing.assert_allclose(out, ref, rtol=1e-6, atol=1e-4)

    def test_small_time_falls_through_to_serial(self):
        """Below min_block × n_workers, we short-circuit to a direct call."""
        rng = np.random.default_rng(0)
        x = rng.standard_normal((500, 64)).astype(np.float32)
        ref = np.median(x, axis=1)
        out = _fast_cmr._parallel_reduce_axis1(x, np.median)
        np.testing.assert_array_equal(out, ref)


class TestInstallUninstall:
    def test_install_idempotent(self):
        pytest.importorskip("spikeinterface")
        _fast_cmr.install()
        _fast_cmr.install()  # no-op
        _fast_cmr.uninstall()

    def test_install_replaces_and_uninstall_restores(self):
        pytest.importorskip("spikeinterface")
        import sys

        import spikeinterface.preprocessing.common_reference  # noqa: F401

        cr = sys.modules["spikeinterface.preprocessing.common_reference"]
        _fast_cmr.uninstall()
        original = cr.CommonReferenceRecordingSegment.get_traces
        _fast_cmr.install()
        assert cr.CommonReferenceRecordingSegment.get_traces is _fast_cmr._fast_cmr_get_traces
        _fast_cmr.uninstall()
        assert cr.CommonReferenceRecordingSegment.get_traces is original

    def test_cmr_global_median_matches_stock(self):
        """End-to-end on the fast path: global reference, no groups, median operator."""
        pytest.importorskip("spikeinterface")
        import spikeinterface as si
        from spikeinterface.preprocessing import CommonReferenceRecording

        rng = np.random.default_rng(0)
        T, C = 60_000, 64
        traces = (rng.standard_normal((T, C)) * 100).astype(np.float32)
        rec = si.NumpyRecording([traces], sampling_frequency=30_000.0)

        _fast_cmr.uninstall()
        try:
            stock_rec = CommonReferenceRecording(rec, operator="median", reference="global")
            ref = stock_rec.get_traces(start_frame=5_000, end_frame=T - 5_000)
            _fast_cmr.install()
            fast_rec = CommonReferenceRecording(rec, operator="median", reference="global")
            out = fast_rec.get_traces(start_frame=5_000, end_frame=T - 5_000)
        finally:
            _fast_cmr.uninstall()

        # Median-subtracted output is deterministic; expect bitwise equality.
        np.testing.assert_array_equal(out, ref)

    def test_delegates_when_group_indices_set(self):
        """Grouped CMR path must delegate to SI's original get_traces.

        We verify by installing, calling get_traces on a mocked segment
        whose ``group_indices`` is set, and asserting the call reaches the
        stock function (captured through ``_original_get_traces``).
        """
        pytest.importorskip("spikeinterface")

        _fast_cmr.uninstall()
        called = {"stock": False}

        def _fake_stock(self, start_frame, end_frame, channel_indices):
            called["stock"] = True
            return np.zeros((end_frame - start_frame, 1), dtype=np.float32)

        # Plug a fake "original" in place and install.
        _fast_cmr._original_get_traces = _fake_stock

        class _FakeSegment:
            group_indices = [[0, 1], [2, 3]]  # non-None triggers delegation
            reference = "global"
            ref_channel_indices = None

        _fast_cmr._fast_cmr_get_traces(_FakeSegment(), 0, 100, slice(None))
        assert called["stock"], "non-fast-path configuration must delegate to stock get_traces"

        _fast_cmr._original_get_traces = None
