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

    @numba.njit(parallel=True, cache=True, boundscheck=False)
    def _sinc_fir_kernel_int16_tc(
        signal_tc: npt.NDArray[np.int16],
        kernels_kc: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        """int16-native variant: reads int16, accumulates in float32, writes float32.

        Benchmarked ~8% faster than the float32 kernel on (1M, 384) on a 24-core
        host — halved signal working set (24 KB vs 48 KB for 32 × 384) leaves more
        L1 headroom, and the ``int16 → float32`` cast vectorizes cleanly (AVX2
        ``VPMOVSXWD + VCVTDQ2PS``).  More importantly, it lets us skip SI's
        ``int16 → float64`` promotion upstream, saving ~2.4 s/shard of cast
        traffic when PhaseShift is first in the filter chain.
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
                        out[n, c] += kernels_kc[k, c] * np.float32(signal_tc[base + k, c])
            else:
                for k in range(K):
                    idx = n + k - half
                    if 0 <= idx < T:
                        for c in range(C):
                            out[n, c] += kernels_kc[k, c] * np.float32(signal_tc[idx, c])
        return out


def apply_frequency_shift_fir(
    signal: npt.NDArray[Any],
    shift_samples: npt.NDArray[Any],
    axis: int = 0,
) -> npt.NDArray[Any]:
    """Drop-in replacement for SI's ``apply_frequency_shift`` via sinc FIR.

    Matches SI's signature exactly — returns a buffer of the same shape as
    *signal*.  Output dtype:

    - ``int16`` input → ``float32`` output (int16-native fast path).  We
      deliberately do *not* round back to int16 here; the companion patch to
      :class:`spikeinterface.preprocessing.phase_shift.PhaseShiftRecording`
      sets the recording's advertised dtype to ``float32`` so the downstream
      bandpass consumes this directly without another cast round-trip.
    - anything else → same dtype as input (float32 compute, cast back on exit).
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
    if orig_dtype == np.int16:
        sig_i16 = np.ascontiguousarray(signal)
        return _sinc_fir_kernel_int16_tc(sig_i16, kernels_kc)
    sig = np.ascontiguousarray(signal, dtype=np.float32)
    out = _sinc_fir_kernel_tc(sig, kernels_kc)
    if orig_dtype != np.float32:
        out = out.astype(orig_dtype)
    return out


# Track whether install() has run, and what the original function was, so
# uninstall() can restore it.
_original_functions: dict[str, Any] = {}


def _patched_phase_shift_init(
    self: Any,
    recording: Any,
    margin_ms: float = 40.0,
    inter_sample_shift: Any = None,
    dtype: Any = None,
) -> None:
    """Replacement ``PhaseShiftRecording.__init__`` with int16-native fast path.

    When the parent recording is integer-dtyped and the caller didn't pin a
    specific output dtype, this flags the segments to use
    :func:`_fast_get_traces` — a minimal get_traces that bypasses SI's
    ``get_chunk_with_margin`` entirely:

    - Margin shrinks from 40 ms (1200 samples at 30 kHz) to ``K/2`` = 16
      samples, tied to the FIR kernel size rather than an FFT-era constant.
    - No raised-cosine taper (it's FFT-specific; zero-padded linear
      convolution at recording edges is exact for a bounded-support FIR).
    - int16 chunk flows straight into the int16-native kernel, returning
      float32.  No int16→float64 promotion, no float64→int16 round-trip,
      no final astype copy.

    For non-integer parents or when the caller explicitly pinned a dtype,
    fall back to SI's original behavior.
    """
    orig_init = _original_functions["PhaseShiftRecording.__init__"]
    parent_dtype = np.dtype(recording.get_dtype())
    # Fast path triggers when parent is int and the caller either didn't pin
    # a dtype or asked for float32.  Any other combination (float parent, or
    # caller explicitly wants int or float64) falls through to SI stock.
    parent_is_int = parent_dtype.kind in ("i", "u")
    user_wants_f32 = dtype is None or np.dtype(dtype) == np.float32
    take_fast_path = parent_is_int and user_wants_f32
    if take_fast_path:
        orig_init(self, recording, margin_ms=margin_ms, inter_sample_shift=inter_sample_shift, dtype=dtype)
        f32 = np.dtype("float32")
        self._dtype = f32
        # Cache kernels once per segment — sample_shifts are fixed for the
        # lifetime of the recording, so rebuilding sinc×kaiser on every
        # get_traces call is pure redundant work.  (K, C_all) float32.
        for seg in self.segments:
            seg.dtype = f32
            seg.tmp_dtype = None
            seg._fast_int16_path = True
            seg._fir_kernels_kc = _build_kernels_kc(seg.sample_shifts, _DEFAULT_N_TAPS)
        return
    orig_init(self, recording, margin_ms=margin_ms, inter_sample_shift=inter_sample_shift, dtype=dtype)
    for seg in self.segments:
        seg._fast_int16_path = False


def _fast_get_traces(self: Any, start_frame: int, end_frame: int, channel_indices: Any) -> np.ndarray:
    """Bypass-``get_chunk_with_margin`` get_traces for the int16-native path.

    Fetches ``K/2`` samples of margin directly from the parent segment, zero-pads
    at recording edges, runs the int16 FIR kernel, slices off the margin, and
    returns float32.  No taper, no unnecessary buffer copies.
    """
    if not getattr(self, "_fast_int16_path", False):
        # Not on the fast path (float parent or caller pinned int dtype) —
        # run SI's original get_traces.
        return _original_functions["PhaseShiftRecordingSegment.get_traces"](
            self, start_frame, end_frame, channel_indices
        )

    if channel_indices is None:
        channel_indices = slice(None)

    half = _DEFAULT_N_TAPS // 2  # kernel reach
    parent = self.parent_recording_segment
    length = int(parent.get_num_samples())

    # Fetch [start-half, end+half], clipped to recording bounds.
    # left_pad / right_pad = number of zero samples to prepend / append
    # when the requested window spills past the recording edges.
    fetch_start = max(0, start_frame - half)
    fetch_end = min(length, end_frame + half)
    left_pad = half - (start_frame - fetch_start)
    right_pad = half - (fetch_end - end_frame)

    traces = parent.get_traces(start_frame=fetch_start, end_frame=fetch_end, channel_indices=channel_indices)

    if left_pad > 0 or right_pad > 0:
        full_len = (end_frame - start_frame) + 2 * half
        padded = np.zeros((full_len, traces.shape[1]), dtype=traces.dtype)
        padded[left_pad : left_pad + traces.shape[0], :] = traces
        traces = padded

    # Channel-slice the cached kernels.  For the common all-channels case
    # (slice(None)) this is a no-op view; for subsets it's a cheap fancy-index
    # over the (K, C_all) kernel buffer (~48 KB for 384 channels).
    if isinstance(channel_indices, slice) and channel_indices == slice(None):
        kernels_kc = self._fir_kernels_kc
    else:
        kernels_kc = np.ascontiguousarray(self._fir_kernels_kc[:, channel_indices])

    # Int16 dispatch: parent is guaranteed int-typed on the fast path, but
    # traces may be int16, int32, uint16, etc. depending on the source.
    # The int16 kernel handles exactly int16; anything else falls back to the
    # float32 kernel via a cast (rare — typical ephys parents are int16).
    if traces.dtype == np.int16:
        traces = np.ascontiguousarray(traces)
        shifted = _sinc_fir_kernel_int16_tc(traces, kernels_kc)
    else:
        sig_f32 = np.ascontiguousarray(traces, dtype=np.float32)
        shifted = _sinc_fir_kernel_tc(sig_f32, kernels_kc)

    return shifted[half : half + (end_frame - start_frame), :]


def install() -> None:
    """Monkey-patch SI's ``apply_frequency_shift`` and ``PhaseShiftRecording.__init__``.

    Two patches:

    1. ``apply_frequency_shift`` → numba sinc FIR, with an int16-native dispatch.
    2. ``PhaseShiftRecording.__init__`` → when parent recording is int-dtype and
       the caller did not pin a dtype, advertise float32 output and bypass the
       ``int16 → float64`` promotion SI would otherwise do.  Saves ~2.4 s/shard
       of cast traffic; the only caller-visible change is that the
       PhaseShiftRecording's ``.get_dtype()`` now returns ``float32`` instead
       of the parent's int dtype.  Downstream filters (bandpass, CMR) want
       float anyway, so this is almost always the right behavior.

    Idempotent — calling multiple times is safe.
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
    _original_functions["PhaseShiftRecording.__init__"] = ps.PhaseShiftRecording.__init__
    _original_functions["PhaseShiftRecordingSegment.get_traces"] = ps.PhaseShiftRecordingSegment.get_traces
    apply_frequency_shift_fir._fast_patched = True  # type: ignore[attr-defined]
    ps.apply_frequency_shift = apply_frequency_shift_fir
    ps.apply_fshift = apply_frequency_shift_fir
    ps.PhaseShiftRecording.__init__ = _patched_phase_shift_init
    ps.PhaseShiftRecordingSegment.get_traces = _fast_get_traces
    log.info(
        "fast-phase-shift: installed 32-tap sinc FIR + int16-native PhaseShiftRecording patch",
    )


def uninstall() -> None:
    """Restore SI's original ``apply_frequency_shift`` and ``PhaseShiftRecording.__init__``.  Idempotent."""
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
    ps.PhaseShiftRecording.__init__ = _original_functions.pop("PhaseShiftRecording.__init__")
    ps.PhaseShiftRecordingSegment.get_traces = _original_functions.pop("PhaseShiftRecordingSegment.get_traces")
    log.info("fast-phase-shift: uninstalled")
