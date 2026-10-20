"""Parallel common-reference median monkey-patch for SpikeInterface's CMR.

Why this exists
---------------

After the phase-shift FIR and parallel bandpass land, py-spy profiles
show ``np.median`` (called by :class:`CommonReferenceRecording` with
``operator="median"``) as the remaining single-threaded hot spot —
~20% of per-shard CPU on typical NP chunks.  ``np.median`` partitions
each row of the input and stays on one thread in numpy's C path.

Fortunately, numpy's partition *does* release the GIL during the
per-row work.  So threading ``np.median`` calls across disjoint time
blocks actually parallelises.  Measured ~10× speedup on 16 workers for
a 1M × 384 chunk, with bitwise-identical output (median is
deterministic on a fixed input).

Strategy
--------

Replace :class:`spikeinterface.preprocessing.common_reference.CommonReferenceRecordingSegment.get_traces`
with a version that:

1. Handles the common NP path (``group_indices=None``, ``reference="global"``,
   ``ref_channel_indices=None``) with a threaded ``np.median``/``np.mean``
   across time blocks.
2. Delegates every other path (``local``, ``single``, grouped) to SI's
   original get_traces — those paths are rare and non-hot.

Pattern matches :mod:`_fast_filter`: module-level thread pool, install()
/uninstall() monkey-patching, idempotent.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import numpy.typing as npt

log = logging.getLogger(__name__)


_POOL: ThreadPoolExecutor | None = None
_POOL_SIZE: int = 0


def _get_pool() -> tuple[ThreadPoolExecutor, int]:
    """Lazy-init a process-wide thread pool for parallel median/mean.

    Size defaults to ``min(16, os.cpu_count())``.  Median parallelises
    almost linearly up to the core count (measured on real NP data), so
    a larger default than the filter pool is appropriate.  Overridable
    via ``EPHYS_CMR_THREADS``.
    """
    global _POOL, _POOL_SIZE
    if _POOL is None:
        default = min(16, os.cpu_count() or 8)
        n = int(os.environ.get("EPHYS_CMR_THREADS", default))
        n = max(1, n)
        _POOL = ThreadPoolExecutor(max_workers=n, thread_name_prefix="ephys_cmr")
        _POOL_SIZE = n
        log.info("fast-cmr: created per-process thread pool (%d workers)", n)
    return _POOL, _POOL_SIZE


def _parallel_reduce_axis1(
    traces: npt.NDArray[Any],
    operator_func: Any,
) -> npt.NDArray[Any]:
    """Apply ``operator_func(traces, axis=1)`` in parallel over time blocks.

    Returns a 1-D array of length ``traces.shape[0]``.
    """
    pool, n_workers = _get_pool()
    T = traces.shape[0]
    # Minimum block size per worker — below this the per-call overhead
    # outweighs the parallel benefit.  Tuned empirically: at T/n_workers
    # < 8192, threading hurts.
    min_block = 8192
    effective_workers = max(1, min(n_workers, T // min_block))
    if effective_workers == 1:
        return operator_func(traces, axis=1)

    block = (T + effective_workers - 1) // effective_workers
    bounds = [(t0, min(t0 + block, T)) for t0 in range(0, T, block)]

    # Pre-allocate the output at the dtype the first call returns.
    out: npt.NDArray[Any] | None = None

    def _work(t0: int, t1: int) -> tuple[int, int, npt.NDArray[Any]]:
        return t0, t1, operator_func(traces[t0:t1, :], axis=1)

    futures = [pool.submit(_work, t0, t1) for t0, t1 in bounds]
    for fut in futures:
        t0, t1, block_out = fut.result()
        if out is None:
            out = np.empty(T, dtype=block_out.dtype)
        out[t0:t1] = block_out
    assert out is not None
    return out


_original_get_traces: Any = None


def _fast_cmr_get_traces(self: Any, start_frame: int, end_frame: int, channel_indices: Any) -> npt.NDArray[Any]:
    """Replacement ``CommonReferenceRecordingSegment.get_traces`` with parallel reduction.

    Only patches the common global-reference, no-groups, all-channels path.
    Every other configuration (``local``, ``single``, grouped, explicit
    ref channels) delegates to SI's stock get_traces.
    """
    # Fall back to stock for any path we don't handle.
    is_common_fast_path = self.group_indices is None and self.reference == "global" and self.ref_channel_indices is None
    if not is_common_fast_path:
        return _original_get_traces(self, start_frame, end_frame, channel_indices)

    traces = self.parent_recording_segment.get_traces(start_frame, end_frame, slice(None))
    shift = _parallel_reduce_axis1(traces, self.operator_func)
    # Restore the keepdims semantics of the original: shape (T, 1) for broadcast.
    shift = shift[:, np.newaxis]
    re_referenced = traces[:, channel_indices] - shift
    return re_referenced.astype(self.dtype, copy=False)


def si_has_row_median() -> bool:
    """Return True when SI's common reference has its own fast exact median (``_row_median``)."""
    import importlib.util

    return importlib.util.find_spec("spikeinterface.preprocessing._row_median") is not None


def install() -> None:
    """Monkey-patch ``CommonReferenceRecordingSegment.get_traces`` for parallel median/mean.

    Idempotent.  Only affects the global-reference, no-groups path — the
    common NP configuration.  No-op when SI has its own fast median
    (:func:`si_has_row_median`), so the two never stack.
    """
    global _original_get_traces
    import sys

    import spikeinterface.preprocessing.common_reference  # noqa: F401 — ensure module loaded

    cr = sys.modules["spikeinterface.preprocessing.common_reference"]

    if getattr(cr.CommonReferenceRecordingSegment.get_traces, "_fast_patched", False):
        return
    if si_has_row_median():
        log.info("fast-cmr: SpikeInterface has its own row median; not patching")
        return
    _original_get_traces = cr.CommonReferenceRecordingSegment.get_traces
    _fast_cmr_get_traces._fast_patched = True  # type: ignore[attr-defined]
    cr.CommonReferenceRecordingSegment.get_traces = _fast_cmr_get_traces
    log.info("fast-cmr: installed parallel median/mean in CommonReferenceRecordingSegment.get_traces")


def uninstall() -> None:
    """Restore SI's original ``CommonReferenceRecordingSegment.get_traces``.  Idempotent."""
    global _original_get_traces
    import sys

    import spikeinterface.preprocessing.common_reference  # noqa: F401

    cr = sys.modules["spikeinterface.preprocessing.common_reference"]
    if _original_get_traces is None:
        return
    cr.CommonReferenceRecordingSegment.get_traces = _original_get_traces
    _original_get_traces = None
    log.info("fast-cmr: uninstalled")
