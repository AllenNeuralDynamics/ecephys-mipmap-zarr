"""Build mipmaps from a SpikeInterface-preprocessed recording.

The AIND pipeline stores the full preprocessing chain (phase-shift,
highpass, bad-channel removal, CMR) as a nested JSON-like dict in the
``SortingAnalyzer`` zarr.  By substituting the raw zarr URI at the
leaf and re-instantiating the chain with ``spikeinterface.load``, we
can stream the same filtered data the sorter saw, without any
intermediate ``.dat`` / ``.cbin`` materialization.

Requires Python >= 3.14 and the ``si`` extra::

    pip install ecephys-mipmap-zarr[si]

SpikeInterface supports zarr 3 only on Python >= 3.14, so the extra
installs it only there.

Typical usage (e.g. from :func:`ecephys_mipmap_zarr.cli.BuildSettings.cli_cmd`)::

    rec = load_preprocessed_recording(
        analyzer_path="s3://.../postprocessed/<probe>.zarr",
        raw_path="s3://.../ecephys_compressed/<probe>.zarr",
    )
    source = SIRecordingSource(rec)
    metadata = build_source_metadata(rec, raw_path)
    build_mipmap(source, output, source_metadata=metadata, ...)
"""

from __future__ import annotations

import copy
import logging
import os
import threading
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from spikeinterface.core.baserecording import BaseRecording

__all__ = [
    "SIRecordingSource",
    "build_source_metadata",
    "load_preprocessed_recording",
]

log = logging.getLogger(__name__)


def load_preprocessed_recording(
    analyzer_path: str,
    raw_path: str,
    *,
    keep_bad_channels: bool = False,
    analyzer_storage_options: dict[str, Any] | None = None,
    raw_storage_options: dict[str, Any] | None = None,
    wavpack_decode_threads: int = 4,
    raw_cache_chunks: int = 0,
    fast_phase_shift: bool = True,
    fast_filter: bool = True,
    fast_cmr: bool = True,
    fuse: bool = False,
) -> BaseRecording:
    """Reconstruct the preprocessed recording from a SortingAnalyzer zarr.

    Loads the recording provenance dict stored in the analyzer's
    ``recording/0`` slot, substitutes *raw_path* at the leaf
    ``ZarrRecordingExtractor``, and instantiates the full preprocessing
    chain (PhaseShift → Highpass → ChannelSlice → CMR → …) via
    :func:`spikeinterface.load`.

    Parameters
    ----------
    analyzer_path : str
        Path or URI to the SortingAnalyzer ``.zarr`` directory
        (typically ``<sorted_asset>/postprocessed/<probe>.zarr``).
    raw_path : str
        Path or URI to the raw recording ``.zarr`` (the wavpack-
        compressed SI zarr, typically under ``ecephys_compressed/``).
    keep_bad_channels : bool, optional
        If True and the analyzer's outermost wrapper is a
        :class:`ChannelSliceRecording` (step-6 "remove dead+noise"),
        peel it off so the returned recording keeps the full channel
        set.  The dropped channel IDs are logged at INFO level for
        diagnosis.  Default False.
    analyzer_storage_options : dict or None, optional
        fsspec storage options for *analyzer_path* (e.g. ``{"anon": True}``
        for public S3).
    raw_storage_options : dict or None, optional
        fsspec storage options for *raw_path*.
    wavpack_decode_threads : int, optional
        Number of threads the WavPack codec should use when decoding the
        leaf zarr array.  Set to ``1`` for single-threaded decode.  Only
        applied when the leaf codec exposes ``num_decoding_threads``
        (``wavpack-numcodecs``); silently ignored otherwise.  Default 4.
    raw_cache_chunks : int, optional
        Keep this many decoded raw chunks and serve raw reads from them, so
        the margins that neighbouring reads share are fetched and decoded
        once.  Size it to cover the reads in flight; 0 (default) disables.
    fast_phase_shift : bool, optional
        When True (default), monkey-patch SI's
        ``apply_frequency_shift`` with a 32-tap numba sinc FIR
        (``_fast_phase_shift.install()``).  On NP 1M-sample shards this
        is ~85× faster than the baseline FFT with ~0.19% spike-band
        error — see :mod:`ecephys_mipmap_zarr._fast_phase_shift` for
        benchmark and correctness details.  Set False to keep SI's
        original FFT implementation.  When SI's ``PhaseShiftRecording`` has
        its own ``method="fir"``, that is selected through the provenance
        kwargs instead of patching.
    fuse : bool, optional
        When True, rebuild the chain with SpikeInterface's FIR phase shift
        and FIR highpass or bandpass and run phase shift, filter and CMR as one
        ``FusedFIRReferenceRecording`` pass (see :func:`_fuse_chain`).  The
        FIR filter replaces the chain's Butterworth, so the output is not
        sample-equivalent to the sorter's.  A chain that can't be fused runs
        unfused with the ``fast_*`` patches, which the fused pass never calls.
        Needs a SpikeInterface with
        ``spikeinterface.preprocessing.fused_fir_reference``.  Default False.

    Returns
    -------
    BaseRecording
        SpikeInterface recording that streams the full preprocessing
        chain.  Has ~same dtype/sampling-rate as the raw zarr, with
        ``is_filtered()`` True.
    """
    try:
        import spikeinterface as si
    except ImportError as e:
        msg = (
            "spikeinterface is required for analyzer-based builds: pip install ecephys-mipmap-zarr[si] "
            "(Python >= 3.14; SpikeInterface pins zarr < 3 on older Pythons)"
        )
        raise ImportError(msg) from e

    use_si_fir_phase_shift = False
    if fast_phase_shift:
        from ecephys_mipmap_zarr._fast_phase_shift import install as _install_fast_phase_shift
        from ecephys_mipmap_zarr._fast_phase_shift import si_has_fir_phase_shift

        use_si_fir_phase_shift = si_has_fir_phase_shift()
        _install_fast_phase_shift()

    if fast_filter:
        from ecephys_mipmap_zarr._fast_filter import install as _install_fast_filter

        _install_fast_filter()

    if fast_cmr:
        from ecephys_mipmap_zarr._fast_cmr import install as _install_fast_cmr

        _install_fast_cmr()

    prov = _read_analyzer_recording_dict(analyzer_path, analyzer_storage_options)
    prov = copy.deepcopy(prov)

    if keep_bad_channels:
        prov = _unwrap_outer_channelslice(prov)

    _fix_leaf(prov, raw_path, raw_storage_options)
    _clear_rel(prov)
    if use_si_fir_phase_shift:
        _select_fir_phase_shift(prov)
    rec: BaseRecording = si.load(prov)
    _configure_wavpack_decode_threads(rec, wavpack_decode_threads)
    if raw_cache_chunks > 0:
        _install_chunk_cache(rec, raw_cache_chunks)
    if fuse:
        rec = _fuse_chain(rec)
    return rec


def _configure_wavpack_decode_threads(rec: Any, num_threads: int) -> None:
    """Set ``num_decoding_threads`` on the leaf zarr's WavPack codec.

    Mutates the codec of each leaf segment's own array handle: zarr builds a
    fresh codec for every handle it opens, so reopening the array would leave
    the segment's untouched.  Best-effort: logs and returns silently if the
    chain, array, or codec doesn't look like WavPack.
    """
    if num_threads <= 0:
        return
    arrays = _leaf_segment_arrays(rec)
    if not arrays:
        log.debug("wavpack-threads: no zarr-backed leaf segments in the chain")
        return
    for arr in arrays:
        compressors = getattr(arr, "compressors", ())
        codec = compressors[0] if compressors else None
        if codec is None or "wavpack" not in type(codec).__name__.lower():
            log.debug("wavpack-threads: leaf codec is not WavPack (got %s)", type(codec).__name__)
            return
        if not hasattr(codec, "num_decoding_threads"):
            log.info(
                "wavpack-threads: %s has no num_decoding_threads attribute — upgrade wavpack-numcodecs?",
                type(codec).__name__,
            )
            return
        codec.num_decoding_threads = num_threads
    log.info("wavpack-threads: set num_decoding_threads=%d on leaf codec", num_threads)


def _leaf_segment_arrays(rec: Any) -> list[Any]:
    """Return the arrays the leaf ``ZarrRecordingExtractor``'s segments read from, or ``[]``."""
    leaf: Any = rec
    for _ in range(64):
        parent = getattr(leaf, "_parent", None)
        if parent is None:
            break
        leaf = parent
    else:
        log.debug("recording chain deeper than 64, giving up")
        return []
    arrays = [getattr(seg, "_timeseries", None) for seg in getattr(leaf, "_recording_segments", [])]
    return arrays if arrays and all(a is not None for a in arrays) else []


def _install_chunk_cache(rec: Any, max_chunks: int) -> None:
    """Route each leaf segment's raw reads through a :class:`_DecodedChunkCache`."""
    leaf: Any = rec
    while getattr(leaf, "_parent", None) is not None:
        leaf = leaf._parent
    segments = [seg for seg in getattr(leaf, "_recording_segments", []) if hasattr(seg, "_timeseries")]
    if not segments:
        log.warning("raw_cache_chunks: leaf %s has no zarr-backed segments; not caching", type(leaf).__name__)
        return
    for seg in segments:
        if not isinstance(seg._timeseries, _DecodedChunkCache):
            seg._timeseries = _DecodedChunkCache(seg._timeseries, max_chunks)


class _DecodedChunkCache:
    """Serve time slices of a ``(T, C)`` zarr array from whole decoded chunks, keeping the most recent *max_chunks*.

    Concurrent readers of the same chunk share one read.  Each run of missing chunks is one zarr read, which
    decodes its chunks concurrently.
    """

    def __init__(self, array: Any, max_chunks: int) -> None:
        self._array = array
        self.shape = tuple(array.shape)
        self.dtype = array.dtype
        self._chunk_len = int(array.chunks[0])
        self._max_chunks = max_chunks
        self._chunks: OrderedDict[int, Future[np.ndarray]] = OrderedDict()
        self._lock = threading.Lock()
        self._pid = os.getpid()

    def __getitem__(self, key: Any) -> np.ndarray:
        if not isinstance(key, slice) or key.step not in (None, 1):
            return np.asarray(self._array[key])
        t0, t1, _ = key.indices(self.shape[0])
        if t1 <= t0:
            return np.asarray(self._array[t0:t0])
        n = self._chunk_len
        first, last = t0 // n, (t1 - 1) // n
        wanted, missing = self._claim(range(first, last + 1))
        try:
            for run in _contiguous_runs(missing):
                block = np.asarray(self._array[run[0] * n : min((run[-1] + 1) * n, self.shape[0])])
                for i, c in enumerate(run):
                    wanted[c].set_result(block[i * n : (i + 1) * n])
        except BaseException as e:
            # Other readers may be waiting on these entries.
            self._abandon({c: wanted[c] for c in missing if not wanted[c].done()}, e)
            raise
        out = np.empty((t1 - t0, *self.shape[1:]), dtype=self.dtype)
        for c, future in wanted.items():
            a, b = max(t0, c * n), min(t1, (c + 1) * n)
            out[a - t0 : b - t0] = future.result()[a - c * n : b - c * n]
        return out

    def _claim(self, indices: range) -> tuple[dict[int, Future[np.ndarray]], list[int]]:
        """Look up *indices*, registering a pending entry for each chunk no one is reading yet."""
        wanted: dict[int, Future[np.ndarray]] = {}
        missing = []
        with self._lock:
            if self._pid != os.getpid():
                # Pending entries from a parent process would never be filled here.
                self._chunks.clear()
                self._pid = os.getpid()
            for c in indices:
                future = self._chunks.get(c)
                if future is None:
                    future = self._chunks[c] = Future()
                    missing.append(c)
                else:
                    self._chunks.move_to_end(c)
                wanted[c] = future
            while len(self._chunks) > self._max_chunks:
                self._chunks.popitem(last=False)
        return wanted, missing

    def _abandon(self, pending: dict[int, Future[np.ndarray]], error: BaseException) -> None:
        with self._lock:
            for c, future in pending.items():
                if self._chunks.get(c) is future:
                    del self._chunks[c]
        for future in pending.values():
            future.set_exception(error)


def _contiguous_runs(indices: list[int]) -> list[list[int]]:
    runs: list[list[int]] = []
    for i in indices:
        if runs and runs[-1][-1] == i - 1:
            runs[-1].append(i)
        else:
            runs.append([i])
    return runs


def build_source_metadata(rec: BaseRecording) -> dict[str, Any]:
    """Build the ``source_metadata`` dict for ``build_mipmap``.

    Pulls per-channel properties (``channel_ids``, ``location``,
    ``gain_to_uV``, ``offset_to_uV``), ``t_start`` (from the per-sample
    time vector), and probe/annotation info directly from *rec*.  This
    avoids opening the raw v2 wavpack zarr ourselves — SI already
    loaded and decoded everything.

    Parameters
    ----------
    rec : BaseRecording
        Output of :func:`load_preprocessed_recording`.

    Returns
    -------
    dict
        Schema compatible with ``build_mipmap(source_metadata=...)``.
    """
    metadata: dict[str, Any] = {"channel_ids": np.asarray(rec.channel_ids)}

    for prop in ("location", "gain_to_uV", "offset_to_uV"):
        values = rec.get_property(prop)
        if values is not None:
            metadata[prop] = np.asarray(values)

    if rec.has_time_vector(segment_index=0):
        # get_times() would read the whole per-sample time vector for its first element.
        metadata["t_start"] = float(rec.get_start_time(segment_index=0))

    # Split annotations into probe (probes_info) + everything else
    annotations = dict(getattr(rec, "_annotations", {}))
    probes_info = annotations.pop("probes_info", None)
    if probes_info is not None:
        metadata["probe"] = {"probes": probes_info}
    if annotations:
        metadata["annotations"] = annotations

    return metadata


class SIRecordingSource:
    """Array-like adapter over a SpikeInterface recording.

    Exposes ``shape`` / ``dtype`` / ``__getitem__`` as expected by
    :func:`build_mipmap`.  The engine reads source data as
    ``source[t0:t1, :]`` and expects ``(T, C)`` shape — which matches
    :meth:`BaseRecording.get_traces`.

    The engine asks for a whole raw shard at a time (~35 s at 30 kHz), far
    past the ~1 s at which SpikeInterface's steps run fastest, so
    *read_samples* splits each request into smaller ``get_traces`` calls.

    Parameters
    ----------
    rec : BaseRecording
        Recording to read.
    start_sample, n_samples : int, optional
        Window of *rec* to expose; ``n_samples=None`` runs to the end.
    read_samples : int or None, optional
        Longest single ``get_traces`` call.  None (default) reads each
        request in one call.
    read_workers : int, optional
        ``get_traces`` calls in flight at once, shared by all callers.
        Default 1.
    step_threads : int, optional
        SpikeInterface thread budget for each call, for the steps that
        split their own work across threads.  Default 1.
    """

    def __init__(
        self,
        rec: BaseRecording,
        *,
        start_sample: int = 0,
        n_samples: int | None = None,
        read_samples: int | None = None,
        read_workers: int = 1,
        step_threads: int = 1,
    ) -> None:
        if read_samples is not None and read_samples < 1:
            msg = f"read_samples must be >= 1, got {read_samples}"
            raise ValueError(msg)
        self._rec = rec
        self._read_samples = read_samples
        self._read_workers = max(1, int(read_workers))
        self._step_threads = max(1, int(step_threads))
        self._pool: ThreadPoolExecutor | None = None
        self._pool_pid = 0
        self._pool_lock = threading.Lock()
        full_n = int(rec.get_num_samples())
        if start_sample < 0 or start_sample >= full_n:
            msg = f"start_sample={start_sample} out of range [0, {full_n})"
            raise ValueError(msg)
        self._start_sample = int(start_sample)
        remaining = full_n - self._start_sample
        if n_samples is None:
            self._n_samples = remaining
        else:
            if n_samples < 0 or n_samples > remaining:
                msg = f"n_samples={n_samples} out of range [0, {remaining}] given start_sample={start_sample}"
                raise ValueError(msg)
            self._n_samples = int(n_samples)
        self._n_channels = int(rec.get_num_channels())
        self._dtype: np.dtype[Any] = np.dtype(rec.get_dtype())

    @property
    def shape(self) -> tuple[int, int]:
        """``(n_samples, n_channels)``."""
        return (self._n_samples, self._n_channels)

    @property
    def dtype(self) -> np.dtype[Any]:
        """Recording dtype as a numpy dtype object."""
        return self._dtype

    def __getitem__(self, key: Any) -> np.ndarray:
        """Support ``src[t0:t1, :]`` indexing used by the engine.

        Indices are interpreted relative to the window (``start_sample``);
        they are shifted into absolute frames before delegating to
        :meth:`BaseRecording.get_traces`.
        """
        if not isinstance(key, tuple) or len(key) != 2:
            msg = f"SIRecordingSource only supports 2D slicing, got {type(key).__name__}"
            raise TypeError(msg)
        t_key, ch_key = key
        if not isinstance(t_key, slice):
            msg = "time axis must be a slice"
            raise TypeError(msg)
        start = t_key.start if t_key.start is not None else 0
        stop = t_key.stop if t_key.stop is not None else self._n_samples
        t0 = int(start) + self._start_sample
        t1 = int(stop) + self._start_sample
        read_samples = self._read_samples
        if read_samples is None or t1 - t0 <= read_samples:
            traces = self._get_traces(t0, t1)
        else:
            traces = np.empty((t1 - t0, self._n_channels), dtype=self._dtype)

            def fill(a: int) -> None:
                b = min(a + read_samples, t1)
                traces[a - t0 : b - t0] = self._get_traces(a, b)

            starts = range(t0, t1, read_samples)
            if self._read_workers > 1:
                list(self._get_pool().map(fill, starts))
            else:
                for a in starts:
                    fill(a)
        if ch_key != slice(None):
            traces = traces[:, ch_key]
        return np.asarray(traces)

    def _get_traces(self, t0: int, t1: int) -> np.ndarray:
        if self._step_threads > 1:
            from spikeinterface.core import thread_budget

            with thread_budget(self._step_threads):
                return np.asarray(self._rec.get_traces(start_frame=t0, end_frame=t1))
        return np.asarray(self._rec.get_traces(start_frame=t0, end_frame=t1))

    def _get_pool(self) -> ThreadPoolExecutor:
        # A forked child inherits the pool object but not its threads, so it would hang on it.
        with self._pool_lock:
            if self._pool is None or self._pool_pid != os.getpid():
                self._pool = ThreadPoolExecutor(max_workers=self._read_workers, thread_name_prefix="si-read")
                self._pool_pid = os.getpid()
            return self._pool


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_analyzer_recording_dict(
    analyzer_path: str,
    storage_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract the recording provenance dict from a SortingAnalyzer zarr.

    The analyzer is a zarr v2 store and its ``recording`` array has
    object dtype (``|O``) with a ``json2`` codec.  Zarr 3.x can't open
    object arrays, so we bypass zarr entirely and decode the single
    chunk manually via ``fsspec`` + ``numcodecs``.
    """
    import json

    import fsspec
    import numcodecs

    path = analyzer_path.rstrip("/")
    opener_kwargs: dict[str, Any] = {}
    if "://" in path:
        opener_kwargs.update(storage_options or {})
    zarray_path = f"{path}/recording/.zarray"
    chunk_path = f"{path}/recording/0"

    with fsspec.open(zarray_path, mode="rb", **opener_kwargs) as f:
        zarray = json.loads(f.read())
    if zarray.get("dtype") != "|O":
        msg = f"unexpected dtype for recording array: {zarray.get('dtype')!r}"
        raise ValueError(msg)

    # Codec pipeline: compressor decompresses bytes → filters decode them
    # back to Python objects (the json2 filter returns an ndarray of dicts).
    compressor = numcodecs.get_codec(zarray["compressor"])
    filters = [numcodecs.get_codec(f) for f in (zarray.get("filters") or [])]

    with fsspec.open(chunk_path, mode="rb", **opener_kwargs) as f:
        raw_bytes = f.read()

    data: Any = compressor.decode(raw_bytes)
    for filt in reversed(filters):
        data = filt.decode(data)

    arr = np.asarray(data).reshape(-1)
    if arr.size == 0:
        msg = f"recording array is empty: {analyzer_path}"
        raise ValueError(msg)
    result: dict[str, Any] = arr[0]
    if not isinstance(result, dict):
        msg = f"decoded recording entry is not a dict: type={type(result).__name__}"
        raise ValueError(msg)
    return result


def _unwrap_outer_channelslice(prov: dict[str, Any]) -> dict[str, Any]:
    """Peel off an outer ``ChannelSliceRecording`` wrapper, if present.

    The AIND pipeline's step-6 "remove dead+noise channels" manifests as
    the outermost ``ChannelSliceRecording`` on top of the CMR output.
    Logs the dropped channel IDs so the caller can see which channels
    were flagged.

    If the outermost class is *not* a ChannelSlice, returns *prov*
    unchanged (some recordings may not have had a step-6 slice).
    """
    cls = prov.get("class", "")
    if not cls.endswith("ChannelSliceRecording"):
        log.info("outermost class is %s, not ChannelSlice — nothing to unwrap", cls or "<missing>")
        return prov

    kwargs = prov.get("kwargs", {})
    # ChannelSliceRecording uses parent_recording under SI 0.103+
    inner = kwargs.get("parent_recording") or kwargs.get("recording")
    if not isinstance(inner, dict):
        log.warning("ChannelSlice wrapper has no inner recording — returning as-is")
        return prov

    kept = kwargs.get("channel_ids")
    inner_kwargs = inner.get("kwargs", {})
    inner_channels = inner.get("properties", {}).get("channel_ids") or inner_kwargs.get("channel_ids")
    if kept is not None and inner_channels is not None:
        dropped = [cid for cid in inner_channels if cid not in kept]
        if dropped:
            log.info("unwrap: dropping %d channels from inner view: %s", len(dropped), dropped)
    else:
        log.info("unwrap: could not determine dropped channel IDs")

    return inner


def _fix_leaf(
    d: dict[str, Any],
    new_path: str,
    storage_options: dict[str, Any] | None = None,
) -> None:
    """Replace the leaf ``ZarrRecordingExtractor`` path with *new_path*.

    Recurses through ``kwargs.recording`` / ``kwargs.parent_recording``
    until it finds a dict without a wrapped child (the leaf), then
    overwrites ``folder_path`` and ``storage_options``.
    """
    for key in ("recording", "parent_recording"):
        child = d.get("kwargs", {}).get(key)
        if isinstance(child, dict) and "class" in child:
            _fix_leaf(child, new_path, storage_options)
            return
    d["kwargs"]["folder_path"] = new_path
    d["kwargs"]["storage_options"] = storage_options
    d["relative_paths"] = False


def _clear_rel(d: dict[str, Any]) -> None:
    """Set ``relative_paths=False`` throughout the chain."""
    if "relative_paths" in d:
        d["relative_paths"] = False
    for key in ("recording", "parent_recording"):
        child = d.get("kwargs", {}).get(key)
        if isinstance(child, dict) and "class" in child:
            _clear_rel(child)


def _select_fir_phase_shift(d: dict[str, Any]) -> None:
    """Switch every ``PhaseShiftRecording`` in the chain to SI's FIR method.

    Matches the monkey-patch: 32 taps, and float32 output when the node leaves
    ``dtype`` unset, so an int16 parent isn't rounded back to int16 before filtering.
    """
    from ecephys_mipmap_zarr._fast_phase_shift import _DEFAULT_N_TAPS

    if d.get("class", "").endswith("PhaseShiftRecording"):
        kwargs = d.setdefault("kwargs", {})
        kwargs["method"] = "fir"
        kwargs["n_taps"] = _DEFAULT_N_TAPS
        if kwargs.get("dtype") is None:
            kwargs["dtype"] = "float32"
    for key in ("recording", "parent_recording"):
        child = d.get("kwargs", {}).get(key)
        if isinstance(child, dict) and "class" in child:
            _select_fir_phase_shift(child)


# Filter kwargs the FIR route can carry over; anything else (filter_order, ftype, ...) has no FIR equivalent.
_FUSABLE_FILTER_KWARGS = {
    "freq_min",
    "freq_max",
    "dtype",
    "margin_ms",
    "method",
    "ignore_low_freq_error",
    "transition_low_hz",
    "transition_high_hz",
    "stopband_db",
}


def _fuse_chain(rec: BaseRecording) -> BaseRecording:
    """Rebuild ``[slice] → CMR → [slice] → Highpass|Bandpass → PhaseShift → base`` as one fused FIR pass.

    SI's ``fuse`` matches CMR → FIR filter → FIR phase shift with nothing between them, so the
    inner channel slices move below the phase shift; the phase shift and filter act per channel,
    so slicing first gives the same channels the same output.  Outer slices (bad-channel removal
    after the CMR) stay on top, since the median is taken over the channels below them.  A chain
    of any other shape is returned unchanged, with a warning.
    """
    try:
        from spikeinterface.preprocessing.fused_fir_reference import FusedFIRReferenceRecording, fuse
    except ImportError as e:
        msg = "fuse=True needs a SpikeInterface with spikeinterface.preprocessing.fused_fir_reference"
        raise ImportError(msg) from e
    from spikeinterface.core.channelslice import ChannelSliceRecording
    from spikeinterface.preprocessing import bandpass_filter, common_reference, highpass_filter, phase_shift
    from spikeinterface.preprocessing.common_reference import CommonReferenceRecording
    from spikeinterface.preprocessing.filter import BandpassFilterRecording, HighpassFilterRecording
    from spikeinterface.preprocessing.phase_shift import PhaseShiftRecording

    from ecephys_mipmap_zarr._fast_phase_shift import _DEFAULT_N_TAPS

    def peel_slices(node: Any) -> tuple[list[dict[str, Any]], Any]:
        slices = []
        while isinstance(node, ChannelSliceRecording):
            slices.append(node._kwargs)
            node = node._parent
        return slices, node

    def apply_slices(node: Any, slices: list[dict[str, Any]]) -> Any:
        for kw in reversed(slices):
            node = ChannelSliceRecording(node, kw["channel_ids"], renamed_channel_ids=kw["renamed_channel_ids"])
        return node

    outer_slices, cmr = peel_slices(rec)
    if not isinstance(cmr, CommonReferenceRecording):
        log.warning("fuse: chain has %s where a CommonReferenceRecording was expected; not fusing", type(cmr).__name__)
        return rec
    cmr_kw = cmr._kwargs
    if cmr_kw.get("reference") != "global" or cmr_kw.get("groups") or cmr_kw.get("ref_channel_ids"):
        log.warning("fuse: CMR is not a plain global reference; not fusing")
        return rec
    inner_slices, filt = peel_slices(cmr._parent)
    filt_kw = {k: v for k, v in filt._kwargs.items() if k != "recording"}
    if (
        not isinstance(filt, HighpassFilterRecording | BandpassFilterRecording)
        or not set(filt_kw) <= _FUSABLE_FILTER_KWARGS
    ):
        log.warning(
            "fuse: no plain highpass or bandpass below the CMR (%s, kwargs %s); not fusing",
            type(filt).__name__,
            sorted(filt_kw),
        )
        return rec
    shift = filt._parent
    if not isinstance(shift, PhaseShiftRecording):
        log.warning("fuse: %s below the filter instead of a PhaseShiftRecording; not fusing", type(shift).__name__)
        return rec

    chain = apply_slices(shift._parent, inner_slices)
    chain = phase_shift(
        chain,
        margin_ms=shift._kwargs["margin_ms"],
        inter_sample_shift=shift._kwargs.get("inter_sample_shift"),
        method="fir",
        n_taps=_DEFAULT_N_TAPS,
        dtype="float32",
    )
    fir_design = {k: filt_kw[k] for k in ("transition_low_hz", "transition_high_hz", "stopband_db") if k in filt_kw}
    if isinstance(filt, BandpassFilterRecording):
        chain = bandpass_filter(
            chain,
            freq_min=filt_kw["freq_min"],
            freq_max=filt_kw["freq_max"],
            method="fir",
            dtype=filt_kw.get("dtype"),
            **fir_design,
        )
    else:
        chain = highpass_filter(
            chain, freq_min=filt_kw["freq_min"], method="fir", dtype=filt_kw.get("dtype"), **fir_design
        )
    chain = common_reference(chain, reference="global", operator=cmr_kw["operator"], dtype=cmr_kw.get("dtype"))
    fused = fuse(chain)
    if not isinstance(fused, FusedFIRReferenceRecording):
        log.warning("fuse: SpikeInterface did not fuse the rebuilt chain; using it unfused")
    return apply_slices(fused, outer_slices)
