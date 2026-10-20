"""Tests for the sinc-FIR monkey-patch of SI's apply_frequency_shift.

The FIR implementation is numerically not identical to SI's FFT-based one,
but it's designed to be functionally equivalent within the spike band
(300 Hz–5 kHz at 30 kHz sampling).  These tests lock in the tolerances
measured against real NP data:

- Max abs error <= 10 ADC units on a realistic band-limited signal.
- Spike-band RMS error <= 1% of spike-band signal RMS.
- install() / uninstall() round-trip correctly.

The test uses a synthetic signal (band-limited noise + spike-like
Gaussians) so the tests stay deterministic and don't need real ephys
data on disk.  See ``/tmp/check_phase_shift_correctness.py`` for the
real-data equivalent used during development.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.fft
import scipy.signal

from ecephys_mipmap_zarr import _fast_phase_shift


@pytest.fixture(autouse=True)
def _patch_regardless_of_si_version(monkeypatch):
    """Exercise the patch even on an SI that has its own FIR phase shift."""
    monkeypatch.setattr(_fast_phase_shift, "si_has_fir_phase_shift", lambda: False)


FS = 30_000.0
SPIKE_LO = 300.0
SPIKE_HI = 5_000.0


def _make_ephys_like_signal(t_samples: int, n_channels: int, seed: int) -> np.ndarray:
    """Band-limited noise in the spike band (300 Hz–5 kHz).

    Real filtered ephys (post-SI highpass + CMR) is ~band-limited to
    this range — FFT phase-shift and sinc-FIR phase-shift agree to very
    high precision on band-limited inputs.  Testing on a broadband
    signal would expose FIR roll-off near Nyquist that doesn't exist
    in practice after SI's filter chain.
    """
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((t_samples, n_channels)).astype(np.float32) * 30.0
    sos = scipy.signal.butter(4, [SPIKE_LO, SPIKE_HI], btype="bandpass", fs=FS, output="sos")
    return scipy.signal.sosfiltfilt(sos, noise, axis=0).astype(np.float32)


def _si_fft_reference(signal_tc: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """Replicates SI's apply_frequency_shift body — float64 FFT, axis=0."""
    sig = signal_tc.astype(np.float64)
    T = sig.shape[0]
    freq = scipy.fft.rfft(sig, n=T, axis=0)
    grid = 2 * np.pi * np.fft.rfftfreq(T)[:, np.newaxis] * shifts[np.newaxis, :]
    return scipy.fft.irfft(freq * np.exp(-1j * grid), n=T, axis=0)


def _bandpass(x: np.ndarray, axis: int) -> np.ndarray:
    sos = scipy.signal.butter(4, [SPIKE_LO, SPIKE_HI], btype="bandpass", fs=FS, output="sos")
    return scipy.signal.sosfiltfilt(sos, x, axis=axis)


class TestFirEquivalence:
    """Numba FIR output must stay close to SI FFT reference in the spike band."""

    @pytest.fixture(scope="class")
    def signal_and_shifts(self):
        t_samples = 65_536  # ~2 s at 30 kHz; small enough for a unit test
        n_channels = 16
        signal = _make_ephys_like_signal(t_samples, n_channels, seed=42)
        rng = np.random.default_rng(0)
        shifts = rng.uniform(0.0, 1.0, size=n_channels).astype(np.float64)
        return signal, shifts

    def test_max_abs_error_within_tolerance(self, signal_and_shifts):
        signal, shifts = signal_and_shifts
        ref = _si_fft_reference(signal, shifts)
        out = _fast_phase_shift.apply_frequency_shift_fir(signal, shifts, axis=0)
        # Drop a 50 ms margin on each side — any fixed-block method diverges
        # from FFT there without a proper margin scheme.
        edge = int(0.05 * FS)
        diff = (out - ref)[edge:-edge, :]
        assert np.abs(diff).max() < 10.0, f"max abs error {np.abs(diff).max():.2f} exceeds 10 ADC-unit tolerance"

    def test_spike_band_rms_under_one_percent(self, signal_and_shifts):
        signal, shifts = signal_and_shifts
        ref = _si_fft_reference(signal, shifts)
        out = _fast_phase_shift.apply_frequency_shift_fir(signal, shifts, axis=0)
        edge = int(0.05 * FS)
        ref_bp = _bandpass(ref, axis=0)[edge:-edge, :]
        out_bp = _bandpass(out, axis=0)[edge:-edge, :]
        signal_rms = float(np.sqrt(np.mean(ref_bp**2)))
        err_rms = float(np.sqrt(np.mean((out_bp - ref_bp) ** 2)))
        assert err_rms / signal_rms < 0.01, f"spike-band RMS error {err_rms / signal_rms:.2%} exceeds 1% tolerance"

    def test_preserves_dtype(self, signal_and_shifts):
        signal, shifts = signal_and_shifts
        for dtype in (np.float32, np.float64):
            out = _fast_phase_shift.apply_frequency_shift_fir(signal.astype(dtype), shifts, axis=0)
            assert out.dtype == dtype

    def test_int16_input_yields_float32_output(self, signal_and_shifts):
        """int16-native fast path: input int16, output float32 (no rounding).

        SI's stock path would round back to int16; our path skips that so the
        companion ``PhaseShiftRecording.__init__`` patch can advertise
        float32 output and avoid the int16 round-trip entirely.
        """
        signal, shifts = signal_and_shifts
        sig_i16 = np.round(signal).astype(np.int16)
        out = _fast_phase_shift.apply_frequency_shift_fir(sig_i16, shifts, axis=0)
        assert out.dtype == np.float32, f"expected float32 output for int16 input, got {out.dtype}"

    def test_int16_spike_band_matches_float_reference(self, signal_and_shifts):
        """int16 FIR output must track the float32 FIR output in the spike band.

        Quantization to int16 adds ~0.29 ADC RMS broadband noise; after the
        300–5000 Hz bandpass (~0.16 of the Nyquist band) this survives as
        ~0.12 ADC RMS.  Against the synthetic 15 ADC spike-band RMS used
        here, that's ~0.8% irreducible floor — so the tolerance is lifted
        to 1.5% relative to 1% for the float path.  On real NP data (~40
        ADC spike-band RMS) the relative error is well under 1%.
        """
        signal, shifts = signal_and_shifts
        sig_i16 = np.round(signal).astype(np.int16)
        ref = _si_fft_reference(signal, shifts)
        out = _fast_phase_shift.apply_frequency_shift_fir(sig_i16, shifts, axis=0)
        edge = int(0.05 * FS)
        ref_bp = _bandpass(ref, axis=0)[edge:-edge, :]
        out_bp = _bandpass(out, axis=0)[edge:-edge, :]
        signal_rms = float(np.sqrt(np.mean(ref_bp**2)))
        err_rms = float(np.sqrt(np.mean((out_bp - ref_bp) ** 2)))
        assert err_rms / signal_rms < 0.015, (
            f"int16 FIR spike-band RMS error {err_rms / signal_rms:.2%} exceeds 1.5% tolerance"
        )


class TestInstallUninstall:
    def test_install_idempotent(self):
        pytest.importorskip("spikeinterface")
        _fast_phase_shift.install()
        _fast_phase_shift.install()  # second call is a no-op
        _fast_phase_shift.uninstall()

    def test_install_replaces_and_uninstall_restores(self):
        pytest.importorskip("spikeinterface")
        import sys

        import spikeinterface.preprocessing.phase_shift  # noqa: F401

        ps = sys.modules["spikeinterface.preprocessing.phase_shift"]
        _fast_phase_shift.uninstall()  # ensure clean state
        original = ps.apply_frequency_shift
        _fast_phase_shift.install()
        assert ps.apply_frequency_shift is _fast_phase_shift.apply_frequency_shift_fir
        _fast_phase_shift.uninstall()
        assert ps.apply_frequency_shift is original

    def test_install_is_noop_when_si_has_fir(self, monkeypatch):
        pytest.importorskip("spikeinterface")
        import sys

        import spikeinterface.preprocessing.phase_shift  # noqa: F401

        ps = sys.modules["spikeinterface.preprocessing.phase_shift"]
        _fast_phase_shift.uninstall()
        original_init = ps.PhaseShiftRecording.__init__
        original_shift = ps.apply_frequency_shift
        monkeypatch.setattr(_fast_phase_shift, "si_has_fir_phase_shift", lambda: True)
        _fast_phase_shift.install()
        assert ps.PhaseShiftRecording.__init__ is original_init
        assert ps.apply_frequency_shift is original_shift
        _fast_phase_shift.uninstall()

    def test_axis_non_zero_raises(self):
        signal = np.zeros((100, 4), dtype=np.float32)
        shifts = np.zeros(4)
        with pytest.raises(NotImplementedError):
            _fast_phase_shift.apply_frequency_shift_fir(signal, shifts, axis=1)


class TestPhaseShiftRecordingPatch:
    """The __init__ patch must advertise float32 output for int-dtype parents."""

    def test_int_parent_yields_float32_phase_shift(self):
        pytest.importorskip("spikeinterface")
        import spikeinterface as si
        from spikeinterface.preprocessing.phase_shift import PhaseShiftRecording

        rng = np.random.default_rng(0)
        n_ch, fs = 8, 30_000.0
        traces = rng.integers(-1000, 1000, size=(int(0.1 * fs), n_ch), dtype=np.int16)
        rec = si.NumpyRecording([traces], sampling_frequency=fs)
        rec.set_property("inter_sample_shift", rng.uniform(0.0, 1.0, size=n_ch))
        _fast_phase_shift.uninstall()
        try:
            _fast_phase_shift.install()
            shifted = PhaseShiftRecording(rec)
            assert shifted.get_dtype() == np.dtype("float32")
            out = shifted.get_traces(start_frame=0, end_frame=500)
            assert out.dtype == np.dtype("float32")
        finally:
            _fast_phase_shift.uninstall()

    def test_float_parent_unchanged(self):
        """Float-dtype parents should follow SI's original __init__ behavior."""
        pytest.importorskip("spikeinterface")
        import spikeinterface as si
        from spikeinterface.preprocessing.phase_shift import PhaseShiftRecording

        rng = np.random.default_rng(0)
        n_ch, fs = 8, 30_000.0
        traces = rng.standard_normal((int(0.1 * fs), n_ch)).astype(np.float32) * 100.0
        rec = si.NumpyRecording([traces], sampling_frequency=fs)
        rec.set_property("inter_sample_shift", rng.uniform(0.0, 1.0, size=n_ch))
        _fast_phase_shift.uninstall()
        try:
            _fast_phase_shift.install()
            shifted = PhaseShiftRecording(rec)
            # Original SI __init__ inherits parent dtype when dtype=None.
            assert shifted.get_dtype() == np.dtype("float32")
        finally:
            _fast_phase_shift.uninstall()

    def test_explicit_int_dtype_respected(self):
        """If the caller explicitly pins an int dtype, don't override it."""
        pytest.importorskip("spikeinterface")
        import spikeinterface as si
        from spikeinterface.preprocessing.phase_shift import PhaseShiftRecording

        rng = np.random.default_rng(0)
        n_ch, fs = 8, 30_000.0
        traces = rng.integers(-1000, 1000, size=(int(0.1 * fs), n_ch), dtype=np.int16)
        rec = si.NumpyRecording([traces], sampling_frequency=fs)
        rec.set_property("inter_sample_shift", rng.uniform(0.0, 1.0, size=n_ch))
        _fast_phase_shift.uninstall()
        try:
            _fast_phase_shift.install()
            shifted = PhaseShiftRecording(rec, dtype="int16")
            assert shifted.get_dtype() == np.dtype("int16")
        finally:
            _fast_phase_shift.uninstall()

    def test_fast_get_traces_matches_si_stock_in_spike_band(self):
        """End-to-end: fast get_traces vs SI's stock path must agree in the spike band.

        Checks the full int16-native path (parent → _fast_get_traces → FIR →
        output) against SI's original FFT-based path (parent → get_chunk_with_margin
        → FFT phase shift → round/cast to int16 → re-cast to float32).  Both should
        produce spike-band-identical output within the same 1.5% tolerance used
        by the int16 direct-kernel test.
        """
        pytest.importorskip("spikeinterface")
        import spikeinterface as si
        from spikeinterface.preprocessing.phase_shift import PhaseShiftRecording

        # Build a realistic-ish int16 recording: band-limited noise scaled to
        # int16 range.
        rng = np.random.default_rng(42)
        n_ch, fs = 8, 30_000.0
        t_samples = int(2.0 * fs)  # 2 s
        signal_f = _make_ephys_like_signal(t_samples, n_ch, seed=42).astype(np.float32) * 3.0
        traces_i16 = np.round(signal_f).astype(np.int16)
        rec = si.NumpyRecording([traces_i16], sampling_frequency=fs)
        shifts = rng.uniform(0.0, 1.0, size=n_ch)
        rec.set_property("inter_sample_shift", shifts)

        _fast_phase_shift.uninstall()
        try:
            # Stock path first (no patch) — capture reference.
            ref_rec = PhaseShiftRecording(rec)
            ref = ref_rec.get_traces(start_frame=5_000, end_frame=t_samples - 5_000).astype(np.float32)
            # Now install and re-create (patched __init__ changes dtype/fast path flag).
            _fast_phase_shift.install()
            fast_rec = PhaseShiftRecording(rec)
            fast = fast_rec.get_traces(start_frame=5_000, end_frame=t_samples - 5_000)
        finally:
            _fast_phase_shift.uninstall()

        assert fast.dtype == np.dtype("float32")
        ref_bp = _bandpass(ref, axis=0)
        fast_bp = _bandpass(fast, axis=0)
        signal_rms = float(np.sqrt(np.mean(ref_bp**2)))
        err_rms = float(np.sqrt(np.mean((fast_bp - ref_bp) ** 2)))
        rel = err_rms / signal_rms
        assert rel < 0.015, f"fast get_traces spike-band RMS error {rel:.2%} exceeds 1.5% tolerance"

    def test_fast_path_also_triggers_for_explicit_float32_dtype(self):
        """User passing dtype='float32' should get the fast path, not stock."""
        pytest.importorskip("spikeinterface")
        import spikeinterface as si
        from spikeinterface.preprocessing.phase_shift import PhaseShiftRecording

        rng = np.random.default_rng(0)
        n_ch, fs = 8, 30_000.0
        traces = rng.integers(-1000, 1000, size=(int(0.1 * fs), n_ch), dtype=np.int16)
        rec = si.NumpyRecording([traces], sampling_frequency=fs)
        rec.set_property("inter_sample_shift", rng.uniform(0.0, 1.0, size=n_ch))
        _fast_phase_shift.uninstall()
        try:
            _fast_phase_shift.install()
            shifted = PhaseShiftRecording(rec, dtype="float32")
            # Check that the fast path was taken (segments have the flag).
            assert all(getattr(seg, "_fast_int16_path", False) for seg in shifted.segments)
            assert shifted.get_dtype() == np.dtype("float32")
        finally:
            _fast_phase_shift.uninstall()

    def test_kernels_are_cached_on_segments(self):
        """After install, each segment should carry its precomputed FIR kernel."""
        pytest.importorskip("spikeinterface")
        import spikeinterface as si
        from spikeinterface.preprocessing.phase_shift import PhaseShiftRecording

        rng = np.random.default_rng(0)
        n_ch, fs = 8, 30_000.0
        traces = rng.integers(-1000, 1000, size=(int(0.1 * fs), n_ch), dtype=np.int16)
        rec = si.NumpyRecording([traces], sampling_frequency=fs)
        rec.set_property("inter_sample_shift", rng.uniform(0.0, 1.0, size=n_ch))
        _fast_phase_shift.uninstall()
        try:
            _fast_phase_shift.install()
            shifted = PhaseShiftRecording(rec)
            for seg in shifted.segments:
                assert hasattr(seg, "_fir_kernels_kc")
                assert seg._fir_kernels_kc.shape == (_fast_phase_shift._DEFAULT_N_TAPS, n_ch)
                assert seg._fir_kernels_kc.dtype == np.float32
        finally:
            _fast_phase_shift.uninstall()

    def test_fast_get_traces_handles_recording_start_edge(self):
        """get_traces(start=0, ...) must zero-pad the left margin correctly."""
        pytest.importorskip("spikeinterface")
        import spikeinterface as si
        from spikeinterface.preprocessing.phase_shift import PhaseShiftRecording

        rng = np.random.default_rng(0)
        n_ch, fs = 4, 30_000.0
        t_samples = int(0.5 * fs)
        traces_i16 = rng.integers(-500, 500, size=(t_samples, n_ch), dtype=np.int16)
        rec = si.NumpyRecording([traces_i16], sampling_frequency=fs)
        rec.set_property("inter_sample_shift", rng.uniform(0.0, 1.0, size=n_ch))

        _fast_phase_shift.uninstall()
        try:
            _fast_phase_shift.install()
            shifted = PhaseShiftRecording(rec)
            # Reading from the very start should just work (no exception),
            # and should return the right shape.
            out = shifted.get_traces(start_frame=0, end_frame=100)
            assert out.shape == (100, n_ch)
            assert out.dtype == np.dtype("float32")
        finally:
            _fast_phase_shift.uninstall()
