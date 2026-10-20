"""Tests for the parallel-sosfiltfilt monkey-patch."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.signal

from ecephys_mipmap_zarr import _fast_filter


FS = 30_000.0


def _make_sos() -> np.ndarray:
    """5th-order Butterworth bandpass 300–5000 Hz, SOS format."""
    return scipy.signal.butter(5, [300.0, 5000.0], btype="bandpass", fs=FS, output="sos")


class TestParallelSosMatchesScipy:
    def test_sosfiltfilt_bitwise_or_close(self):
        """Output of parallel sosfiltfilt matches scipy stock within floating noise."""
        rng = np.random.default_rng(0)
        T, C = 30_000, 32
        signal = rng.standard_normal((T, C)).astype(np.float32) * 50.0
        sos = _make_sos()

        ref = scipy.signal.sosfiltfilt(sos, signal, axis=0)
        out = _fast_filter._parallel_sos_apply(sos, signal, kind="sosfiltfilt")

        # Should match to within float32 roundoff (each thread independently
        # runs the same scipy call on disjoint columns).
        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)

    def test_sosfilt_bitwise_or_close(self):
        rng = np.random.default_rng(0)
        T, C = 30_000, 48
        signal = rng.standard_normal((T, C)).astype(np.float32) * 50.0
        sos = _make_sos()

        ref = scipy.signal.sosfilt(sos, signal, axis=0)
        out = _fast_filter._parallel_sos_apply(sos, signal, kind="sosfilt")
        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)

    def test_falls_through_for_tiny_channel_counts(self):
        """With fewer than 2*n_workers channels, short-circuit to stock scipy."""
        sos = _make_sos()
        tiny = np.random.default_rng(0).standard_normal((1000, 3)).astype(np.float32)
        # Should not raise and should match scipy stock.
        out = _fast_filter._parallel_sos_apply(sos, tiny, kind="sosfiltfilt")
        ref = scipy.signal.sosfiltfilt(sos, tiny, axis=0)
        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


class TestInstallUninstall:
    def test_install_idempotent(self):
        pytest.importorskip("spikeinterface")
        _fast_filter.install()
        _fast_filter.install()  # no-op
        _fast_filter.uninstall()

    def test_install_replaces_and_uninstall_restores(self):
        pytest.importorskip("spikeinterface")
        import sys

        import spikeinterface.preprocessing.filter  # noqa: F401

        filt = sys.modules["spikeinterface.preprocessing.filter"]
        _fast_filter.uninstall()  # clean slate
        original = filt.FilterRecordingSegment.get_traces
        _fast_filter.install()
        assert filt.FilterRecordingSegment.get_traces is _fast_filter._fast_filter_get_traces
        _fast_filter.uninstall()
        assert filt.FilterRecordingSegment.get_traces is original

    def test_bandpass_recording_produces_same_output_as_stock(self):
        """End-to-end: BandpassFilterRecording with and without the patch."""
        pytest.importorskip("spikeinterface")
        import spikeinterface as si
        from spikeinterface.preprocessing.filter import BandpassFilterRecording

        rng = np.random.default_rng(0)
        T, C = 60_000, 64  # 2 s at 30 kHz, 64 channels
        traces = (rng.standard_normal((T, C)) * 100).astype(np.float32)
        rec = si.NumpyRecording([traces], sampling_frequency=FS)

        _fast_filter.uninstall()
        try:
            stock_rec = BandpassFilterRecording(rec, freq_min=300.0, freq_max=5000.0, dtype="float32")
            ref = stock_rec.get_traces(start_frame=5_000, end_frame=T - 5_000)
            _fast_filter.install()
            fast_rec = BandpassFilterRecording(rec, freq_min=300.0, freq_max=5000.0, dtype="float32")
            out = fast_rec.get_traces(start_frame=5_000, end_frame=T - 5_000)
        finally:
            _fast_filter.uninstall()

        assert out.dtype == np.float32
        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-4)
