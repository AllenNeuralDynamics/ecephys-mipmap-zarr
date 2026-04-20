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

    def test_axis_non_zero_raises(self):
        signal = np.zeros((100, 4), dtype=np.float32)
        shifts = np.zeros(4)
        with pytest.raises(NotImplementedError):
            _fast_phase_shift.apply_frequency_shift_fir(signal, shifts, axis=1)
