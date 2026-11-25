"""Parallel sosfiltfilt monkey-patch for SpikeInterface's BandpassFilterRecording.

Why this exists
---------------

After the phase-shift FIR fix, py-spy profiles show scipy's
``sosfiltfilt`` dominates 60–70% of per-shard CPU in the preprocessed
filter chain.  scipy's SOS filter is a single-threaded C implementation
that processes channels serially within one call.  For a 1M × 384 chunk
through a 5th-order IIR, that's ~12 billion ops on one core, ~1.5 s
wall time — and with ``--prefetch-chunks 3`` we have 3 cores doing
bandpass while 20+ cores sit idle.

Since the filter is strictly *per-channel* (no cross-channel state),
we can split the channel axis into blocks and run independent
``sosfiltfilt`` calls in threads.  scipy's C implementation releases
the GIL, so Python threads parallelize cleanly.

Strategy
--------

Replace :class:`spikeinterface.preprocessing.filter.FilterRecordingSegment.get_traces`
with a version that:

1. Fetches the chunk with margin as before.
2. Splits the channel axis into ``n_workers`` contiguous blocks.
3. Runs ``sosfiltfilt`` (or ``sosfilt`` for forward/backward-only modes)
   on each block in a shared :class:`~concurrent.futures.ThreadPoolExecutor`.
4. Stitches and returns — same output as SI stock.

The thread pool is module-level and lazy-initialised to avoid per-call
pool startup overhead.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import numpy.typing as npt

log = logging.getLogger(__name__)


# Module-level pool reused across calls.  Built lazily so setting
# ``NUMBA_NUM_THREADS`` / module-level thread count tweaks still work at
# import time.
_POOL: ThreadPoolExecutor | None = None
_POOL_SIZE: int = 0


def _get_pool() -> tuple[ThreadPoolExecutor, int]:
    """Lazy-init a process-wide thread pool for parallel filtering.

    Pool size defaults to ``min(8, os.cpu_count())`` — enough to saturate
    scipy's SOS filter throughput on typical NP chunks, without
    oversubscribing the box when multiple ``prefetch_chunks`` are running.
    Overridable via the ``EPHYS_FILTER_THREADS`` env var.
    """
    global _POOL, _POOL_SIZE
    if _POOL is None:
        default = min(8, os.cpu_count() or 4)
        n = int(os.environ.get("EPHYS_FILTER_THREADS", default))
        n = max(1, n)
        _POOL = ThreadPoolExecutor(max_workers=n, thread_name_prefix="ephys_filter")
        _POOL_SIZE = n
        log.info("fast-filter: created per-process thread pool (%d workers)", n)
    return _POOL, _POOL_SIZE


def _parallel_sos_apply(
    coeff: npt.NDArray[Any],
    traces: npt.NDArray[Any],
    *,
    kind: str,
) -> npt.NDArray[Any]:
    """Apply ``sosfiltfilt`` or ``sosfilt`` along axis=0, parallel over channels.

    Parameters
    ----------
    coeff : ndarray
        SOS coefficients (shape ``(n_sections, 6)``).
    traces : ndarray
        ``(T, C)`` chunk.
    kind : {"sosfiltfilt", "sosfilt"}
        Which scipy function to apply.
    """
    import scipy.signal

    pool, n_workers = _get_pool()
    T, C = traces.shape
    if C < 2 * n_workers or n_workers == 1:
        # Too few channels to benefit from splitting — fall through to scipy.
        fn = scipy.signal.sosfiltfilt if kind == "sosfiltfilt" else scipy.signal.sosfilt
        result: npt.NDArray[Any] = fn(coeff, traces, axis=0)
        return result

    block = (C + n_workers - 1) // n_workers
    bounds = [(c0, min(c0 + block, C)) for c0 in range(0, C, block)]
    fn = scipy.signal.sosfiltfilt if kind == "sosfiltfilt" else scipy.signal.sosfilt

    # sosfiltfilt/sosfilt default return is float64 for int input, matching
    # scipy's promotion rules.  For float input the output dtype matches input.
    # We allocate the output based on the first block's dtype after the call.
    out: npt.NDArray[Any] | None = None

    def _process(c0: int, c1: int) -> tuple[int, int, npt.NDArray[Any]]:
        return c0, c1, fn(coeff, traces[:, c0:c1], axis=0)

    futures = [pool.submit(_process, c0, c1) for c0, c1 in bounds]
    for fut in futures:
        c0, c1, block_out = fut.result()
        if out is None:
            out = np.empty((T, C), dtype=block_out.dtype)
        out[:, c0:c1] = block_out
    assert out is not None
    return out


def _fast_filter_get_traces(self: Any, start_frame: int, end_frame: int, channel_indices: Any) -> npt.NDArray[Any]:
    """Filter like ``FilterRecordingSegment.get_traces``, with channel-parallel SOS.

    Mirrors SI's original logic (`filter.py:175-219` in SI 0.103.0) but
    swaps the two scipy SOS calls for their channel-parallel variants.
    Falls back to SI's stock path for non-SOS filter modes (``ba``).
    """
    import scipy.signal
    from spikeinterface.core import get_chunk_with_margin

    traces_chunk, left_margin, right_margin = get_chunk_with_margin(
        self.parent_recording_segment,
        start_frame,
        end_frame,
        channel_indices,
        self.margin,
        add_reflect_padding=self.add_reflect_padding,
    )

    traces_dtype = traces_chunk.dtype
    if traces_dtype.kind == "u":
        traces_chunk = traces_chunk.astype("float32")

    # Only SOS mode is parallelised here; ``ba`` is rarely used and would
    # need a separate code path for lfilter/filtfilt.  Fall back to stock
    # scipy for it.
    filtered: npt.NDArray[Any]
    if self.filter_mode != "sos":
        if self.direction == "forward-backward":
            b, a = self.coeff
            filtered = scipy.signal.filtfilt(b, a, traces_chunk, axis=0)
        else:
            if self.direction == "backward":
                traces_chunk = np.flip(traces_chunk, axis=0)
            b, a = self.coeff
            filtered = scipy.signal.lfilter(b, a, traces_chunk, axis=0)
            if self.direction == "backward":
                filtered = np.flip(filtered, axis=0)
    else:
        if self.direction == "forward-backward":
            filtered = _parallel_sos_apply(self.coeff, traces_chunk, kind="sosfiltfilt")
        else:
            if self.direction == "backward":
                traces_chunk = np.flip(traces_chunk, axis=0)
            filtered = _parallel_sos_apply(self.coeff, traces_chunk, kind="sosfilt")
            if self.direction == "backward":
                filtered = np.flip(filtered, axis=0)

    if right_margin > 0:
        filtered = filtered[left_margin:-right_margin, :]
    else:
        filtered = filtered[left_margin:, :]

    if np.issubdtype(self.dtype, np.integer):
        filtered = filtered.round()

    return filtered.astype(self.dtype)


# Track original for uninstall.
_original_get_traces: Any = None


def install() -> None:
    """Monkey-patch ``FilterRecordingSegment.get_traces`` for channel-parallel SOS.

    Idempotent.  Only replaces the segment method; the ``FilterRecording``
    class and its subclasses (``BandpassFilterRecording``, ``HighpassFilterRecording``)
    continue to work as before — they inherit the patched segment behavior
    automatically.
    """
    global _original_get_traces
    import sys

    import spikeinterface.preprocessing.filter  # noqa: F401 — ensure module loaded

    # Match the phase-shift patch's pattern for consistency: go through
    # sys.modules because SI's preprocessing ``__init__`` does shadowing
    # imports.
    filt = sys.modules["spikeinterface.preprocessing.filter"]

    if getattr(filt.FilterRecordingSegment.get_traces, "_fast_patched", False):
        return
    _original_get_traces = filt.FilterRecordingSegment.get_traces
    _fast_filter_get_traces._fast_patched = True  # type: ignore[attr-defined]
    filt.FilterRecordingSegment.get_traces = _fast_filter_get_traces
    log.info("fast-filter: installed parallel sosfiltfilt in FilterRecordingSegment.get_traces")


def uninstall() -> None:
    """Restore SI's original ``FilterRecordingSegment.get_traces``.  Idempotent."""
    global _original_get_traces
    import sys

    import spikeinterface.preprocessing.filter  # noqa: F401

    filt = sys.modules["spikeinterface.preprocessing.filter"]
    if _original_get_traces is None:
        return
    filt.FilterRecordingSegment.get_traces = _original_get_traces
    _original_get_traces = None
    log.info("fast-filter: uninstalled")
