"""Build mipmaps from a SpikeInterface-preprocessed recording.

The AIND pipeline stores the full preprocessing chain (phase-shift,
highpass, bad-channel removal, CMR) as a nested JSON-like dict in the
``SortingAnalyzer`` zarr.  By substituting the raw zarr URI at the
leaf and re-instantiating the chain with ``spikeinterface.load``, we
can stream the same filtered data the sorter saw, without any
intermediate ``.dat`` / ``.cbin`` materialization.

Requires the ``si`` extra::

    pip install ecephys-mipmap-zarr[si]

which pulls in the ``zarrv3`` branch of SpikeInterface and
``wavpack-numcodecs``.

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

from ecephys_mipmap_zarr._store import (
    _extract_source_metadata,
    _extract_t_start,
    _open_store,
)

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
        msg = "spikeinterface is required for analyzer-based builds: pip install ecephys-mipmap-zarr[si]"
        raise ImportError(msg) from e

    prov = _read_analyzer_recording_dict(analyzer_path, analyzer_storage_options)
    prov = copy.deepcopy(prov)

    if keep_bad_channels:
        prov = _unwrap_outer_channelslice(prov)

    _fix_leaf(prov, raw_path, raw_storage_options)
    _clear_rel(prov)
    rec: BaseRecording = si.load(prov)
    return rec


def build_source_metadata(
    rec: BaseRecording,
    raw_path: str,
    *,
    raw_storage_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the ``source_metadata`` dict for ``build_mipmap``.

    Combines per-channel properties from *rec* (already sliced to the
    preprocessed channel set) with probe/annotation metadata read
    from the raw zarr.

    Parameters
    ----------
    rec : BaseRecording
        Output of :func:`load_preprocessed_recording`.
    raw_path : str
        Path or URI to the raw zarr (for probe info, annotations,
        ``t_start``, and to look up raw-channel indices).
    raw_storage_options : dict or None, optional
        fsspec storage options for *raw_path*.

    Returns
    -------
    dict
        Schema compatible with ``build_mipmap(source_metadata=...)``.
    """
    si_channel_ids = np.asarray(rec.channel_ids)

    src = _open_store(raw_path, mode="r")
    raw_channel_ids = np.asarray(src["channel_ids"][:])

    # Map preprocessed channels → indices in the raw zarr
    raw_idx = np.array([int(np.where(raw_channel_ids == cid)[0][0]) for cid in si_channel_ids])

    metadata: dict[str, Any] = {"channel_ids": si_channel_ids}

    source_meta = _extract_source_metadata(src)
    if source_meta is not None:
        metadata.update(source_meta)  # probe, annotations

    t_start = _extract_t_start(src)
    if t_start is not None:
        metadata["t_start"] = t_start

    try:
        props = src["properties"]
        if "location" in list(props):
            metadata["location"] = np.asarray(props["location"][:])[raw_idx]
        if "gain_to_uV" in list(props) and "offset_to_uV" in list(props):
            metadata["gain_to_uV"] = np.asarray(props["gain_to_uV"][:])[raw_idx]
            metadata["offset_to_uV"] = np.asarray(props["offset_to_uV"][:])[raw_idx]
    except (KeyError, AttributeError):
        pass

    del raw_storage_options  # currently unused; reserved for future fsspec plumbing
    return metadata


class SIRecordingSource:
    """Array-like adapter over a SpikeInterface recording.

    Exposes ``shape`` / ``dtype`` / ``__getitem__`` as expected by
    :func:`build_mipmap`.  The engine reads source data as
    ``source[t0:t1, :]`` and expects ``(T, C)`` shape — which matches
    :meth:`BaseRecording.get_traces`.
    """

    def __init__(self, rec: BaseRecording) -> None:
        self._rec = rec
        self._n_samples = int(rec.get_num_samples())
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
        """Support ``src[t0:t1, :]`` indexing used by the engine."""
        if not isinstance(key, tuple) or len(key) != 2:
            msg = f"SIRecordingSource only supports 2D slicing, got {type(key).__name__}"
            raise TypeError(msg)
        t_key, ch_key = key
        if not isinstance(t_key, slice):
            msg = "time axis must be a slice"
            raise TypeError(msg)
        start = t_key.start if t_key.start is not None else 0
        stop = t_key.stop if t_key.stop is not None else self._n_samples
        traces = self._rec.get_traces(start_frame=int(start), end_frame=int(stop))
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

    Uses SI's ``super_zarr_open`` so the analyzer zarr itself (which may
    still be in zarr v2 layout) is read correctly.
    """
    try:
        from spikeinterface.core.zarrextractors import super_zarr_open
    except ImportError as e:
        msg = "spikeinterface is required for analyzer-based builds: pip install ecephys-mipmap-zarr[si]"
        raise ImportError(msg) from e

    root = super_zarr_open(analyzer_path, mode="r", storage_options=storage_options or {})
    rec_field = root.get("recording")
    if rec_field is None:
        msg = f"No recording dict found in analyzer: {analyzer_path}"
        raise ValueError(msg)
    result: dict[str, Any] = rec_field[0]
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
    inner_channels = (
        inner.get("properties", {}).get("channel_ids")
        or inner_kwargs.get("channel_ids")
    )
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
