"""Event layers: sorted spikes or detected events, stored beside a mipmap store for a viewer to draw over traces.

Layout (Zarr v3)::

    /                         attrs: format, format_version, kind, sampling_frequency, t_start,
                                     channel_ids, window_s, n_events
    /sample_index   (N,)      float64, integer-valued and ascending; float so a browser reads it without BigInt
    /group          (N,)      int32, row of /groups
    /amplitude      (N,)      float32, optional
    /channel        (N,)      int32, optional; index into channel_ids, -1 for none
    /depth_range    (N, 2)    float32, optional; µm, the probe span the event covers
    /width_s        (N,)      float32, optional; overrides window_s for that event
    /index/first_sample       float64, sample_index of each chunk's first event, to find a window's chunks
    /groups/id      (G,)      str
    /groups/footprint_channel (G, K) int32, index into channel_ids, -1 padding
    /groups/footprint_weight  (G, K) float32, largest first
    /groups/properties/<name> (G,)

Samples count from the recording's first sample. ``t_start`` is that sample's time; null means the layer starts
where the store it is drawn on starts, as a sorting of that store's recording does. Channels are named by id, so
footprints map onto any store holding those channel ids. Colours are not stored:
a viewer assigns them for the palette it has.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import zarr

from ecephys_mipmap_zarr._store import _write_string_array

__all__ = [
    "EVENTS_FORMAT",
    "EVENTS_FORMAT_VERSION",
    "EventGroups",
    "events_from_detections",
    "events_from_sorting",
    "footprints_from_templates",
    "read_analyzer",
    "read_events",
    "write_events",
]

EVENTS_FORMAT = "ecephys-events"
EVENTS_FORMAT_VERSION = 1
DEFAULT_CHUNK = 262_144
#: A channel is in a unit's footprint at this fraction of the unit's largest template peak-to-peak.
FOOTPRINT_FRACTION = 0.3
#: Footprint channels kept per unit, largest first.
FOOTPRINT_MAX = 16


@dataclass(frozen=True)
class EventGroups:
    """What events belong to: sorted units, or classes of detection.

    Parameters
    ----------
    ids : sequence of str
        One label per group.
    footprint_channel : ndarray, optional
        ``(G, K)`` indices into the layer's ``channel_ids``, ``-1`` padding; ``None`` for no footprints.
    footprint_weight : ndarray, optional
        ``(G, K)`` weight per footprint channel, such as template peak-to-peak.
    properties : mapping, optional
        Per-group arrays of length ``G``.
    """

    ids: Sequence[str]
    footprint_channel: npt.NDArray[np.int32] | None = None
    footprint_weight: npt.NDArray[np.float32] | None = None
    properties: Mapping[str, npt.ArrayLike] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Check the footprint arrays and properties match the groups."""
        n = len(self.ids)
        if (self.footprint_channel is None) != (self.footprint_weight is None):
            raise ValueError("footprint_channel and footprint_weight go together")
        if self.footprint_channel is not None and self.footprint_weight is not None:
            fc, fw = np.asarray(self.footprint_channel), np.asarray(self.footprint_weight)
            if fc.ndim != 2 or fc.shape[0] != n or fc.shape != fw.shape:
                raise ValueError(f"footprints must be ({n}, K) and alike, got {fc.shape} and {fw.shape}")
        for name, values in self.properties.items():
            if len(np.asarray(values)) != n:
                raise ValueError(f"property {name!r} has {len(np.asarray(values))} values for {n} groups")


def write_events(
    path: str | Path,
    sample_index: npt.ArrayLike,
    group: npt.ArrayLike,
    groups: EventGroups,
    *,
    sampling_frequency: float,
    channel_ids: Sequence[str],
    kind: str,
    t_start: float | None = None,
    window_s: tuple[float, float] = (-0.0005, 0.001),
    amplitude: npt.ArrayLike | None = None,
    channel: npt.ArrayLike | None = None,
    depth_range: npt.ArrayLike | None = None,
    width_s: npt.ArrayLike | None = None,
    chunk: int = DEFAULT_CHUNK,
    overwrite: bool = False,
) -> zarr.Group:
    """Write an event layer, sorting events by time.

    Parameters
    ----------
    path : str or Path
        Destination store.
    sample_index : array_like
        ``(N,)`` sample of each event, counted from the recording's first sample.
    group : array_like
        ``(N,)`` row of `groups` each event belongs to.
    groups : EventGroups
        Group labels, footprints and properties.
    sampling_frequency : float
        Samples per second.
    channel_ids : sequence of str
        Channels that footprints and ``channel`` index.
    kind : str
        What the events are, such as ``"spikes"`` or ``"detections"``.
    t_start : float, optional
        Time of sample 0, seconds; omit when the samples share the first sample of the store they are drawn on.
    window_s : tuple of float, optional
        Seconds before (negative) and after an event's sample that it covers.
    amplitude, channel, depth_range, width_s : array_like, optional
        Per-event values; see the module docstring.
    chunk : int, optional
        Events per chunk.
    overwrite : bool, optional
        Replace an existing store.

    Returns
    -------
    zarr.Group
        The written root.
    """
    samples = np.asarray(sample_index)
    if samples.ndim != 1:
        raise ValueError("sample_index must be 1-D")
    if samples.size and not np.array_equal(samples, np.round(samples)):
        raise ValueError("sample_index must be whole samples")
    n = samples.size
    order = np.argsort(samples, kind="stable")
    rows = np.asarray(group)
    if rows.shape != (n,):
        raise ValueError(f"group must be ({n},), got {rows.shape}")
    if n and (rows.min() < 0 or rows.max() >= len(groups.ids)):
        raise ValueError("group must index groups.ids")
    per_event: dict[str, tuple[npt.NDArray[Any], str]] = {}
    for name, values, dtype in (
        ("amplitude", amplitude, "float32"),
        ("channel", channel, "int32"),
        ("depth_range", depth_range, "float32"),
        ("width_s", width_s, "float32"),
    ):
        if values is None:
            continue
        arr = np.asarray(values)
        if arr.shape[:1] != (n,) or arr.shape[1:] != ((2,) if name == "depth_range" else ()):
            raise ValueError(f"{name} has shape {arr.shape} for {n} events")
        per_event[name] = (arr, dtype)
    if channel is not None:
        ch = per_event["channel"][0]
        if n and (ch.min() < -1 or ch.max() >= len(channel_ids)):
            raise ValueError("channel must index channel_ids or be -1")

    root = zarr.open_group(str(path), mode="w" if overwrite else "w-", zarr_format=3)
    root.attrs.update(
        {
            "format": EVENTS_FORMAT,
            "format_version": EVENTS_FORMAT_VERSION,
            "kind": kind,
            "sampling_frequency": float(sampling_frequency),
            "t_start": None if t_start is None else float(t_start),
            "channel_ids": [str(c) for c in channel_ids],
            "window_s": [float(window_s[0]), float(window_s[1])],
            "n_events": int(n),
        }
    )
    size = max(1, min(chunk, n))
    sorted_samples = samples[order].astype(np.float64)
    root.create_array("sample_index", data=sorted_samples, chunks=(size,))
    root.create_array("group", data=rows[order].astype(np.int32), chunks=(size,))
    for name, (arr, dtype) in per_event.items():
        root.create_array(name, data=arr[order].astype(dtype), chunks=(size, *arr.shape[1:]))
    index = root.create_group("index")
    index.create_array("first_sample", data=sorted_samples[::size].copy())

    g = root.create_group("groups")
    _write_string_array(g, "id", np.asarray([str(i) for i in groups.ids]))
    if groups.footprint_channel is not None and groups.footprint_weight is not None:
        g.create_array("footprint_channel", data=np.asarray(groups.footprint_channel, np.int32))
        g.create_array("footprint_weight", data=np.asarray(groups.footprint_weight, np.float32))
    props = g.create_group("properties")
    for name, values in groups.properties.items():
        arr = np.asarray(values)
        if arr.dtype.kind in "USO":
            _write_string_array(props, name, arr.astype(str))
        else:
            props.create_array(name, data=arr)
    zarr.consolidate_metadata(str(path))
    return root


def _read(group: zarr.Group, name: str) -> npt.NDArray[Any]:
    arr = group[name]
    if not isinstance(arr, zarr.Array):
        raise TypeError(f"{name} is not an array")
    return np.asarray(arr[...])


def read_events(path: str | Path) -> dict[str, Any]:
    """Read a whole event layer into memory: its attrs, per-event arrays and ``groups``."""
    root = zarr.open_group(str(path), mode="r")
    attrs = dict(root.attrs)
    if attrs.get("format") != EVENTS_FORMAT:
        raise ValueError(f"{path} is not an event layer")
    version = attrs.get("format_version", 0)
    if not isinstance(version, int) or version > EVENTS_FORMAT_VERSION:
        raise ValueError(f"{path} is format version {version}, which this reader does not know")
    out: dict[str, Any] = {"attrs": attrs}
    for name in ("sample_index", "group", "amplitude", "channel", "depth_range", "width_s"):
        if name in root:
            out[name] = _read(root, name)
    g = root["groups"]
    if not isinstance(g, zarr.Group):
        raise TypeError(f"{path}: groups is not a group")
    props = g["properties"]
    out["groups"] = {
        "id": np.array([str(v) for v in _read(g, "id")]),
        "properties": {k: np.asarray(v[...]) for k, v in props.arrays()} if isinstance(props, zarr.Group) else {},
    }
    for name in ("footprint_channel", "footprint_weight"):
        if name in g:
            out["groups"][name] = _read(g, name)
    return out


def footprints_from_templates(
    templates: npt.ArrayLike,
    fraction: float = FOOTPRINT_FRACTION,
    max_channels: int = FOOTPRINT_MAX,
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.float32]]:
    """Per unit, the channels whose template peak-to-peak reaches `fraction` of the unit's largest, largest first.

    Parameters
    ----------
    templates : array_like
        ``(U, T, C)`` mean waveforms.
    fraction : float, optional
        Threshold relative to each unit's largest channel.
    max_channels : int, optional
        Channels kept per unit.

    Returns
    -------
    tuple of ndarray
        ``(U, K)`` channel indices padded with ``-1``, and their peak-to-peak, padded with 0.
    """
    ptp = np.ptp(np.asarray(templates, np.float32), axis=1)
    return _top_channels(ptp, fraction, max_channels)


def _top_channels(
    weight: npt.NDArray[np.floating[Any]], fraction: float, max_channels: int
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.float32]]:
    order = np.argsort(-weight, axis=1, kind="stable")[:, :max_channels]
    top = np.take_along_axis(weight, order, axis=1)
    keep = top >= fraction * top[:, :1]
    k = max(1, int(keep.sum(1).max(initial=1)))
    channels = np.where(keep, order, -1)[:, :k].astype(np.int32)
    weights = np.where(keep, top, 0)[:, :k].astype(np.float32)
    return channels, weights


def _load_curation(path: Path) -> tuple[set[Any], list[list[Any]]]:
    if not path.exists():
        return set(), []
    curation = json.loads(path.read_text())
    removed = curation.get("removed", [])
    if isinstance(removed, str):
        removed = json.loads(removed)
    merges = [m["unit_ids"] if isinstance(m, Mapping) else list(m) for m in curation.get("merges", [])]
    return set(removed), merges


def read_analyzer(location: str, storage_options: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Read what :func:`events_from_sorting` needs from a SpikeInterface zarr analyzer.

    Parameters
    ----------
    location : str
        Local path or fsspec URI of the analyzer's ``.zarr``.
    storage_options : mapping, optional
        For a remote URI, such as ``{"anon": True}`` for public data.

    Returns
    -------
    dict
        ``templates`` ``(U, T, C)``, ``template_channel_ids``, ``template_unit_ids`` and, when the analyzer has
        them, ``spike_amplitudes``.
    """
    options = dict(storage_options or {}) or None
    base = location.rstrip("/")

    # Analyzers hold pickled object arrays that zarr cannot parse, so nothing opens the root's consolidated metadata.
    def array(path: str) -> npt.NDArray[Any] | None:
        try:
            arr = zarr.open_array(f"{base}/{path}", mode="r", zarr_format=2, storage_options=options)
        except FileNotFoundError:
            return None
        return np.asarray(arr[...])

    info: Any = zarr.open_group(
        f"{base}/recording_info", mode="r", zarr_format=2, use_consolidated=False, storage_options=options
    ).attrs.asdict()
    channel_ids = info.get("recording_attributes", {}).get("channel_ids")
    if not isinstance(channel_ids, list):
        raise ValueError(f"{location}: recording_info names no channel_ids")
    templates = array("extensions/templates/average")
    unit_ids = array("sorting/unit_ids")
    if templates is None or unit_ids is None:
        raise ValueError(f"{location} has no templates/average or sorting/unit_ids")
    out: dict[str, Any] = {
        "templates": templates,
        "template_channel_ids": [str(c) for c in channel_ids],
        "template_unit_ids": unit_ids.tolist(),
    }
    amplitudes = array("extensions/spike_amplitudes/amplitudes")
    if amplitudes is not None:
        out["spike_amplitudes"] = amplitudes
    return out


def events_from_sorting(
    sorting_folder: str | Path,
    templates: npt.ArrayLike,
    template_channel_ids: Sequence[str],
    *,
    properties: Sequence[str] = ("decoder_label", "default_qc", "KSLabel"),
    template_unit_ids: Sequence[Any] | None = None,
    spike_amplitudes: npt.ArrayLike | None = None,
) -> dict[str, Any]:
    """Arguments for :func:`write_events` from a SpikeInterface numpy-folder sorting with its ``curation.json``.

    Removed units are dropped and merge lists become one group, labelled by its members joined with ``+``, whose
    footprint is the members' channel-wise largest peak-to-peak.

    Parameters
    ----------
    sorting_folder : str or Path
        Folder holding ``spikes.npy``, ``numpysorting_info.json``, ``properties/`` and optionally
        ``curation.json`` (the AIND sorted asset's ``curated/``).
    templates : array_like
        ``(U, T, C)`` mean waveforms, one per unit of the sorting before curation.
    template_channel_ids : sequence of str
        The ``C`` channel ids of `templates`.
    properties : sequence of str, optional
        Unit properties to carry where the folder has them; a merged group takes its first member's.
    template_unit_ids : sequence, optional
        Unit ids in `templates` order, when it differs from the sorting's.
    spike_amplitudes : array_like, optional
        One per spike in ``spikes.npy`` order, such as the analyzer's ``spike_amplitudes``; stored as magnitudes.

    Returns
    -------
    dict
        Keyword arguments for :func:`write_events` except ``path``.
    """
    folder = Path(sorting_folder)
    info = json.loads((folder / "numpysorting_info.json").read_text())
    unit_ids = list(info["unit_ids"])
    spikes = np.load(folder / "spikes.npy", mmap_mode="r")
    if "segment_index" in (spikes.dtype.names or ()) and np.any(spikes["segment_index"] != 0):
        raise ValueError("only a single-segment sorting is supported")
    removed, merges = _load_curation(folder / "curation.json")

    template_order = list(template_unit_ids) if template_unit_ids is not None else unit_ids
    ptp = np.ptp(np.asarray(templates, np.float32), axis=1)
    if ptp.shape[0] != len(template_order):
        raise ValueError(f"{ptp.shape[0]} templates for {len(template_order)} units")
    ptp_of = {u: ptp[i] for i, u in enumerate(template_order)}

    merged_into = {u: tuple(m) for m in merges for u in m}
    members: list[tuple[Any, ...]] = []
    seen: set[tuple[Any, ...]] = set()
    for u in unit_ids:
        if u in removed:
            continue
        key = merged_into.get(u, (u,))
        if key not in seen:
            seen.add(key)
            members.append(key)
    row_of_unit = np.full(len(unit_ids), -1, np.int64)
    position = {u: i for i, u in enumerate(unit_ids)}
    for row, key in enumerate(members):
        for u in key:
            row_of_unit[position[u]] = row

    unit_index = np.asarray(spikes["unit_index"])
    rows = row_of_unit[unit_index]
    keep = rows >= 0
    weight = np.stack([np.max([ptp_of[u] for u in key], axis=0) for key in members])
    channels, weights = _top_channels(weight, FOOTPRINT_FRACTION, FOOTPRINT_MAX)

    props: dict[str, npt.NDArray[Any]] = {}
    for name in properties:
        file = folder / "properties" / f"{name}.npy"
        if file.exists():
            values = np.load(file, allow_pickle=True)
            props[name] = np.asarray([values[position[key[0]]] for key in members])
    out: dict[str, Any] = {
        "sample_index": np.asarray(spikes["sample_index"])[keep],
        "group": rows[keep],
        "groups": EventGroups(
            ids=["+".join(str(u) for u in key) for key in members],
            footprint_channel=channels,
            footprint_weight=weights,
            properties=props,
        ),
        "sampling_frequency": float(info["sampling_frequency"]),
        "channel_ids": [str(c) for c in template_channel_ids],
        "kind": "spikes",
    }
    if spike_amplitudes is not None:
        amplitudes = np.asarray(spike_amplitudes)
        if amplitudes.shape != unit_index.shape:
            raise ValueError(f"{amplitudes.size} spike amplitudes for {unit_index.size} spikes")
        out["amplitude"] = np.abs(amplitudes[keep])
    return out


def events_from_detections(
    table: Mapping[str, npt.ArrayLike],
    *,
    sampling_frequency: float,
    channel_ids: Sequence[str],
    group_by: str | None = None,
    t_start: float = 0.0,
) -> dict[str, Any]:
    """Arguments for :func:`write_events` from a table of detected events, one row each.

    Parameters
    ----------
    table : mapping of str to array_like
        Columns: ``time_s`` (seconds from `t_start`) and optionally ``peak_channel`` (index into `channel_ids`),
        ``peak_uv``, ``width_ms``, and ``half_top_um`` with ``half_bottom_um``.
    sampling_frequency : float
        Samples per second of the recording `time_s` refers to.
    channel_ids : sequence of str
        The recording's channel ids.
    group_by : str, optional
        Column whose values become groups, such as ``"counted"``; one group ``"event"`` without.
    t_start : float, optional
        Time of sample 0.

    Returns
    -------
    dict
        Keyword arguments for :func:`write_events` except ``path``.
    """
    time = np.asarray(table["time_s"], np.float64)
    n = time.size
    if group_by is None:
        ids, rows = np.asarray(["event"]), np.zeros(n, np.int64)
    else:
        ids, rows = np.unique(np.asarray(table[group_by]).astype(str), return_inverse=True)
    out: dict[str, Any] = {
        "sample_index": np.round(time * sampling_frequency),
        "group": rows,
        "groups": EventGroups(ids=[str(i) for i in ids]),
        "sampling_frequency": sampling_frequency,
        "channel_ids": list(channel_ids),
        "kind": "detections",
        "t_start": t_start,
    }
    if "peak_channel" in table:
        out["channel"] = np.asarray(table["peak_channel"], np.int64)
    if "peak_uv" in table:
        out["amplitude"] = np.asarray(table["peak_uv"], np.float32)
    if "width_ms" in table:
        out["width_s"] = np.asarray(table["width_ms"], np.float32) / 1000
    if "half_top_um" in table and "half_bottom_um" in table:
        out["depth_range"] = np.stack(
            [np.asarray(table["half_bottom_um"], np.float32), np.asarray(table["half_top_um"], np.float32)], axis=1
        )
    return out
