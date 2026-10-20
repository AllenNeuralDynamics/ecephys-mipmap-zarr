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
    fast_phase_shift: bool = True,
    fast_filter: bool = True,
    fast_cmr: bool = True,
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
    fast_phase_shift : bool, optional
        When True (default), monkey-patch SI's
        ``apply_frequency_shift`` with a 32-tap numba sinc FIR
        (``_fast_phase_shift.install()``).  On NP 1M-sample shards this
        is ~85× faster than the baseline FFT with ~0.19% spike-band
        error — see :mod:`ecephys_mipmap_zarr._fast_phase_shift` for
        benchmark and correctness details.  Set False to keep SI's
        original FFT implementation.

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

    if fast_phase_shift:
        from ecephys_mipmap_zarr._fast_phase_shift import install as _install_fast_phase_shift

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
    rec: BaseRecording = si.load(prov)
    _configure_wavpack_decode_threads(rec, wavpack_decode_threads)
    return rec


def _configure_wavpack_decode_threads(rec: Any, num_threads: int) -> None:
    """Set ``num_decoding_threads`` on the leaf zarr's WavPack codec.

    Walks ``_parent_recording`` / ``parent_recording`` down to the leaf
    ``ZarrRecordingExtractor``, pulls its backing zarr array, and mutates
    the codec instance in place.  Best-effort: logs and returns silently
    if the chain, array, or codec doesn't look like WavPack.
    """
    if num_threads <= 0:
        return
    leaf: Any = rec
    for _ in range(64):
        parent = getattr(leaf, "_parent_recording", None) or getattr(leaf, "parent_recording", None)
        if parent is None:
            break
        leaf = parent
    else:
        log.debug("wavpack-threads: recording chain deeper than 64, giving up")
        return

    root = getattr(leaf, "_root", None)
    if root is None:
        log.debug("wavpack-threads: leaf %s has no _root attribute", type(leaf).__name__)
        return
    try:
        arr = root["traces_seg0"]
    except (KeyError, TypeError):
        log.debug("wavpack-threads: no traces_seg0 in leaf zarr root")
        return
    codec = getattr(arr, "compressor", None)
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
        metadata["t_start"] = float(rec.get_times(segment_index=0)[0])

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
    """

    def __init__(
        self,
        rec: BaseRecording,
        *,
        start_sample: int = 0,
        n_samples: int | None = None,
    ) -> None:
        self._rec = rec
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
        traces = self._rec.get_traces(
            start_frame=int(start) + self._start_sample,
            end_frame=int(stop) + self._start_sample,
        )
        if ch_key != slice(None):
            traces = traces[:, ch_key]
        return np.asarray(traces)


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
