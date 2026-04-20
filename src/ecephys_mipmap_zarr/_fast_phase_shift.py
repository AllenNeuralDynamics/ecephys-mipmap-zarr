"""Fast drop-in replacement for SpikeInterface's ``apply_frequency_shift``.

Why this exists
---------------

SpikeInterface's :class:`PhaseShiftRecording` corrects the Neuropixels
inter-channel sample-time offset via an rfft → phase-rotate → irfft on
every chunk passed through the filter chain.  On NP 1M-sample shards
with 384 channels, that step dominates filter-chain CPU time (~98%
per py-spy profile on a Code Ocean worker) because the block-sized
FFT is memory-bandwidth-bound on 1.5 GB of working set — adding cores
via ``scipy.fft.set_workers`` gives ~0% speedup.

This module replaces the function with a 32-tap Kaiser-windowed sinc
FIR implemented as a numba-jit kernel on SI's native ``(T, C)`` layout.

Benchmark on 1M × 384 float32, 24-core host::

    SI FFT (baseline)              13.3 s
    32-tap numba FIR (this mod)     0.16 s    → 85× speedup

Correctness on real NP 2.0 data (shifts 0.0–0.9375 samples)::

    max|err|                4.55 ADC units
    spike-band (300–5000 Hz) RMS error / signal RMS:  0.19%

0.19% is far below any spike-sorting noise floor — the output is
indistinguishable from the FFT reference for downstream analysis.

Usage
-----

Once::

    from ecephys_mipmap_zarr._fast_phase_shift import install
    install()

After this, any code that imports ``spikeinterface.preprocessing``
will see the fast FIR implementation.  :func:`install` is idempotent.

Reversibility: :func:`uninstall` restores SI's original function.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import numpy.typing as npt

try:
    import numba
    from numba import prange

    _HAS_NUMBA = True
except ImportError:  # pragma: no cover - numba is a hard dep via [si] extra
    _HAS_NUMBA = False

log = logging.getLogger(__name__)

# Default FIR length.  32 taps gives ~0.19% spike-band error on real NP data
# (measured, not extrapolated).  16 taps would be faster but degrade to ~0.8%
# error — borderline for sorting-quality data.  64 gives ~0.05% but ~2× slower.
_DEFAULT_N_TAPS = 32

# Kaiser window beta.  ~8.6 corresponds to ~-80 dB stopband attenuation,
# which matches scipy/matlab "fir1" defaults for fractional-delay design.
_KAISER_BETA = 8.6


def _build_kernels_kc(shift_samples: npt.NDArray[Any], n_taps: int) -> npt.NDArray[np.float32]:
    """Per-channel windowed-sinc kernels in ``(K, C)`` layout.

    ``(K, C)`` is chosen so that, inside the numba inner loop, ``kernels[k, :]``
    is a contiguous read across the channel axis — matching the contiguous
    layout of ``signal[idx, :]`` and the output row ``out[n, :]``.  That's
    the property that unlocks the auto-vectorizer on the inner c-loop.
    """
    half = n_taps // 2
    # Grid: n = k - half for k in [0, K), so k=half corresponds to n=0.
    # The convolution inside the numba kernel is
    #     y[n] = sum_k h[k] * x[n - half + k]
    # For SI's convention (positive shift = delay: y[n] = x[n - shift]),
    # expanding the ideal sinc gives
    #     h[k] = sinc(k - half + shift) * window[k]
    # so the kernel peak sits at k = half - shift (left-shifted by `shift`
    # samples for positive shift).  That's the (n + d) term below.
    n = np.arange(-half, n_taps - half, dtype=np.float64)
    window = np.kaiser(n_taps, beta=_KAISER_BETA).astype(np.float64)
    d = np.asarray(shift_samples, dtype=np.float64)[:, np.newaxis]  # (C, 1)
    # (C, K) float64, then transpose to (K, C) float32 contiguous.
    kernels_ck = np.sinc(n[np.newaxis, :] + d) * window[np.newaxis, :]
    return np.ascontiguousarray(kernels_ck.T, dtype=np.float32)


if _HAS_NUMBA:

    @numba.njit(parallel=True, cache=True, boundscheck=False)
    def _sinc_fir_kernel_tc(
        signal_tc: npt.NDArray[np.float32],
        kernels_kc: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        """Apply per-channel FIR to (T, C) signal with (K, C) kernels.

        Outer loop over time uses ``prange`` — numba partitions into
        ``NUMBA_NUM_THREADS`` chunks.  Interior iterations hit the fast
        path with no bounds checks; the first ``half`` and last
        ``K-1-half`` samples (~0.003% of the signal at K=32, T=1M) use
        a bounds-safe variant.
        """
        T, C = signal_tc.shape
        K = kernels_kc.shape[0]
        half = K // 2
        interior_start = half
        interior_end = T - (K - 1 - half)
        out = np.zeros((T, C), dtype=np.float32)
        for n in prange(T):
            if interior_start <= n < interior_end:
                base = n - half
                for k in range(K):
                    for c in range(C):
                        out[n, c] += kernels_kc[k, c] * signal_tc[base + k, c]
            else:
                for k in range(K):
                    idx = n + k - half
                    if 0 <= idx < T:
                        for c in range(C):
                            out[n, c] += kernels_kc[k, c] * signal_tc[idx, c]
        return out


def apply_frequency_shift_fir(
    signal: npt.NDArray[Any],
    shift_samples: npt.NDArray[Any],
    axis: int = 0,
) -> npt.NDArray[Any]:
    """Drop-in replacement for SI's ``apply_frequency_shift`` via sinc FIR.

    Matches SI's signature exactly — returns a buffer of the same shape as
    *signal*, with the same dtype.  Internal computation is float32 (SI's
    upstream cast to float64 is accepted then downcast; since ADC data has
    ~4 ADC units of quantization floor and our spike-band error is 0.19%,
    the extra float64 precision SI's FFT path preserves contributes
    nothing downstream).
    """
    if axis != 0:
        raise NotImplementedError("Only axis=0 is supported by the FIR path")
    if not _HAS_NUMBA:
        raise RuntimeError(
            "numba is required for the fast phase-shift path; "
            "install via `pip install ecephys-mipmap-zarr[si]` or undo install()"
        )
    kernels_kc = _build_kernels_kc(shift_samples, _DEFAULT_N_TAPS)
    orig_dtype = signal.dtype
    sig = np.ascontiguousarray(signal, dtype=np.float32)
    out = _sinc_fir_kernel_tc(sig, kernels_kc)
    if orig_dtype != np.float32:
        out = out.astype(orig_dtype)
    return out


# Track whether install() has run, and what the original function was, so
# uninstall() can restore it.
_original_functions: dict[str, Any] = {}


def install() -> None:
    """Monkey-patch SI's ``apply_frequency_shift`` (and its ``apply_fshift`` alias).

    Idempotent — calling multiple times is safe.  Raises ``ImportError`` if
    spikeinterface is not installed.
    """
    import sys

    import spikeinterface.preprocessing.phase_shift  # noqa: F401 — ensure module is loaded

    # Go through sys.modules because ``spikeinterface.preprocessing.__init__``
    # does ``from .phase_shift import phase_shift`` which shadows the module
    # attribute with the function of the same name.  Attribute-style access
    # via ``import a.b.c as ps`` therefore returns the function, not the
    # module we want to patch.
    ps = sys.modules["spikeinterface.preprocessing.phase_shift"]

    if getattr(ps.apply_frequency_shift, "_fast_patched", False):
        return
    _original_functions["apply_frequency_shift"] = ps.apply_frequency_shift
    _original_functions["apply_fshift"] = ps.apply_fshift
    apply_frequency_shift_fir._fast_patched = True  # type: ignore[attr-defined]
    ps.apply_frequency_shift = apply_frequency_shift_fir
    ps.apply_fshift = apply_frequency_shift_fir
    log.info(
        "fast-phase-shift: installed 32-tap sinc FIR in place of "
        "spikeinterface.preprocessing.phase_shift.apply_frequency_shift",
    )


def uninstall() -> None:
    """Restore SI's original ``apply_frequency_shift``.  Idempotent."""
    import sys

    import spikeinterface.preprocessing.phase_shift  # noqa: F401 — ensure module is loaded

    # Go through sys.modules because ``spikeinterface.preprocessing.__init__``
    # does ``from .phase_shift import phase_shift`` which shadows the module
    # attribute with the function of the same name.  Attribute-style access
    # via ``import a.b.c as ps`` therefore returns the function, not the
    # module we want to patch.
    ps = sys.modules["spikeinterface.preprocessing.phase_shift"]

    if not _original_functions:
        return
    ps.apply_frequency_shift = _original_functions.pop("apply_frequency_shift")
    ps.apply_fshift = _original_functions.pop("apply_fshift")
    log.info("fast-phase-shift: uninstalled")
