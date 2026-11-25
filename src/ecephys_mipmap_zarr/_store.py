"""Shared store utilities for opening and managing zarr groups.

Internal module — not part of the public API.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from ecephys_mipmap_zarr.metadata import LevelQuantization, RawQuantization

__all__: list[str] = []  # private module

log = logging.getLogger(__name__)


def _default_cache_dir() -> Path:
    """Return the OS-standard cache directory for this package.

    Uses ``XDG_CACHE_HOME`` on Linux, ``~/Library/Caches`` on macOS,
    and ``%LOCALAPPDATA%`` on Windows.
    """
    if xdg := os.environ.get("XDG_CACHE_HOME"):
        return Path(xdg) / "ecephys-mipmap"
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))
        return Path(base) / "ecephys-mipmap" / "Cache"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "ecephys-mipmap"
    return Path.home() / ".cache" / "ecephys-mipmap"


def _prune_cache(cache_dir: Path, max_bytes: int) -> None:
    """Delete oldest files in *cache_dir* until total size is under *max_bytes*.

    Files are sorted by modification time (oldest first).  Pruning runs
    once per store open, so the cache may briefly exceed the budget
    between opens.
    """
    if not cache_dir.exists():
        return
    files = []
    total = 0
    for f in cache_dir.rglob("*"):
        if f.is_file():
            sz = f.stat().st_size
            files.append((f.stat().st_mtime, sz, f))
            total += sz

    if total <= max_bytes:
        return

    files.sort()  # oldest mtime first
    deleted = 0
    for _mtime, sz, f in files:
        if total <= max_bytes:
            break
        try:
            f.unlink()
            total -= sz
            deleted += 1
        except OSError:
            pass  # file in use or already gone
    if deleted:
        log.info("Pruned %d cached files (%.1f MB freed)", deleted, (total - max_bytes) / 1e6)


def _open_store(
    path: str | Path,
    mode: str = "r",
    cache_dir: str | Path | None = None,
    cache_max_gb: float = 10,
) -> zarr.Group:
    """Open a zarr store from a local path or remote URI.

    Supports ``s3://`` URIs via fsspec (requires the ``s3`` extra).

    Parameters
    ----------
    path : str or Path
        Local filesystem path or remote URI (e.g. ``s3://bucket/store.zarr``).
    mode : str, optional
        Open mode, by default ``"r"``.
    cache_dir : str, Path, or None, optional
        When set and *path* is a remote URI, an fsspec ``filecache``
        layer is added so that fetched data is persisted locally.
        Ignored for local paths.
    cache_max_gb : float, optional
        Maximum cache size in GB.  Oldest files are pruned on open.
        Default 10 GB.

    Returns
    -------
    zarr.Group
    """
    path_str = str(path)
    if "://" in path_str:
        try:
            import fsspec
        except ImportError as e:
            msg = "fsspec is required for remote URIs: pip install ecephys-mipmap-zarr[s3]"
            raise ImportError(msg) from e
        protocol = path_str.split("://")[0]
        if cache_dir is not None:
            cache_path = Path(cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            _prune_cache(cache_path, int(cache_max_gb * 1024**3))
            fs = fsspec.filesystem(
                "filecache",
                target_protocol=protocol,
                cache_storage=str(cache_path),
                same_names=True,
            )
        else:
            fs = fsspec.filesystem(protocol, anon=True)
        store = zarr.storage.FsspecStore(fs=fs, path=path_str.replace(f"{protocol}://", ""))
        return zarr.open_group(store, mode=mode, use_consolidated=None)  # type: ignore[arg-type]
    return zarr.open_group(path_str, mode=mode, use_consolidated=None)  # type: ignore[arg-type]


_RAW_ARRAY_PATHS = ("raw", "traces_seg0")
"""Candidate paths to probe when looking for the raw array."""


def _find_raw_array(group: zarr.Group, raw_path: str) -> zarr.Array:
    """Find the raw array in a zarr group.

    If *raw_path* is explicit (not the default ``"raw"``), use it
    directly.  Otherwise probe a list of known paths.

    Parameters
    ----------
    group : zarr.Group
        The opened zarr store root.
    raw_path : str
        Explicit path, or ``"raw"`` to trigger auto-detection.

    Returns
    -------
    zarr.Array

    Raises
    ------
    KeyError
        If no raw array is found.
    """
    if raw_path != "raw":
        return group[raw_path]  # type: ignore[return-value]

    for candidate in _RAW_ARRAY_PATHS:
        try:
            return group[candidate]  # type: ignore[return-value]
        except KeyError:
            continue

    members = list(group)
    msg = (
        f"Could not find raw array. Tried: {list(_RAW_ARRAY_PATHS)}. "
        f"Store contains: {members}. Use --raw-path to specify explicitly."
    )
    raise KeyError(msg)


def _extract_source_metadata(group: zarr.Group) -> dict[str, Any] | None:
    """Extract probe and annotation metadata from a source zarr store.

    Parameters
    ----------
    group : zarr.Group
        Source store root group.

    Returns
    -------
    dict or None
        Dictionary with ``probe`` and/or ``annotations`` keys, or
        ``None`` if no relevant metadata found.
    """
    src_attrs = dict(group.attrs)
    metadata: dict[str, Any] = {}

    for key in ("probe", "annotations"):
        if key in src_attrs:
            metadata[key] = src_attrs[key]

    return metadata if metadata else None


def _extract_t_start(group: zarr.Group) -> float | None:
    """Extract the recording start time from an SI zarr store.

    Reads the first element of ``times_seg0`` if present.

    Parameters
    ----------
    group : zarr.Group
        Source SI zarr store.

    Returns
    -------
    float or None
        Start time in seconds, or ``None`` if not available.
    """
    try:
        times = group["times_seg0"]
        return float(np.asarray(times[:1])[0])  # type: ignore[index]
    except (KeyError, IndexError):
        return None


def _write_string_array(dst: zarr.Group, name: str, values: np.ndarray) -> None:
    """Write a 1D string array using the Zarr v3 ``vlen-utf8`` dtype.

    Why: passing a numpy ``<U*`` array to ``create_array`` lands on Zarr's
    ``FixedLengthUTF32`` codec, which is an unratified V3 extension and
    emits ``UnstableSpecificationWarning``.  Variable-length UTF-8 is the
    stable spec type and what other tools expect.
    """
    arr = dst.create_array(name, shape=values.shape, dtype="str", overwrite=True)
    arr[:] = [str(v) for v in values]


_CALIBRATION_PROPERTIES = frozenset(
    {"gain_to_uV", "offset_to_uV", "gain_to_physical_unit", "offset_to_physical_unit", "physical_unit"}
)
"""Left out of ``/properties``: the arrays' ``quantization`` attr already decodes to uV, so a copy would invite
applying the gain twice."""


def _write_channel_properties(dst: zarr.Group, properties: dict[str, Any], n_channels: int) -> None:
    """Write per-channel properties under ``/properties``, one array per name.

    Skips calibration (see ``_CALIBRATION_PROPERTIES``), structured arrays such as SpikeInterface's
    ``contact_vector`` (the probe JSON carries the same fields), and anything not indexed by channel.
    """
    for name, values in properties.items():
        if name in _CALIBRATION_PROPERTIES or values is None:
            continue
        arr = np.asarray(values)
        if arr.dtype.names is not None or arr.ndim == 0 or arr.shape[0] != n_channels:
            continue
        props = dst.require_group("properties")
        if arr.dtype.kind in "USO":
            if arr.ndim == 1:
                _write_string_array(props, name, arr)
        elif arr.dtype.kind in "biuf":
            props.create_array(name, data=arr, overwrite=True)


def _copy_source_arrays(src: zarr.Group, dst: zarr.Group) -> None:
    """Copy ``channel_ids`` and the per-channel ``properties`` arrays of an SI zarr to *dst*."""
    members = list(src)
    n_channels = None
    if "channel_ids" in members:
        ch_ids = np.asarray(src["channel_ids"][:])  # type: ignore[index]
        _write_string_array(dst, "channel_ids", ch_ids)
        n_channels = len(ch_ids)

    if "properties" not in members:
        return
    props_src = src["properties"]
    if not isinstance(props_src, zarr.Group):
        return
    properties: dict[str, Any] = {}
    for name in props_src:
        node = props_src[name]
        if isinstance(node, zarr.Array):
            properties[name] = np.asarray(node[:])
    if "location" in properties:
        properties["location"] = np.asarray(properties["location"], dtype=np.float64)
    if n_channels is None:
        n_channels = next((len(v) for v in properties.values() if np.ndim(v) > 0), 0)
    _write_channel_properties(dst, properties, n_channels)


def _read_gain_offset(src: zarr.Group) -> tuple[np.ndarray, np.ndarray] | None:
    """Read ``gain_to_uV`` / ``offset_to_uV`` from an SI zarr.

    Returns
    -------
    tuple or None
        ``(gain, offset)`` as 1D float arrays, or ``None`` if either
        array is missing.
    """
    try:
        props = src["properties"]
        if not isinstance(props, zarr.Group):
            return None
        gain = np.asarray(props["gain_to_uV"])
        offset = np.asarray(props["offset_to_uV"])
    except (KeyError, AttributeError):
        return None
    return gain, offset


def _metadata_gain_offset(metadata: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    """Return ``(gain_to_uV, offset_to_uV)`` from a source-metadata dict, or ``None`` unless both are present."""
    gain = metadata.get("gain_to_uV")
    offset = metadata.get("offset_to_uV")
    if gain is None or offset is None:
        return None
    return np.asarray(gain), np.asarray(offset)


def _pop_encoding(array: zarr.Array) -> tuple[dict[str, Any], int, dict[str, Any] | None]:
    """Return *array*'s attrs without its shift attrs, the shift in bits, and its range fit if any."""
    attrs: dict[str, Any] = dict(array.attrs)
    shift = int(attrs.pop("lsb_shift", 0))
    attrs.pop("original_dtype", None)
    return attrs, shift, attrs.get("quantization")


def _write_raw_quantization(dst: zarr.Group, gain: np.ndarray, offset: np.ndarray) -> None:
    """Store the ADC calibration on ``/raw``, composed with its encoding, so the reader decodes to uV."""
    try:
        raw_array: zarr.Array = dst["raw"]  # type: ignore[assignment]
    except KeyError:
        return  # No /raw in destination (pyramid-only store)

    attrs, shift, range_fit = _pop_encoding(raw_array)
    if range_fit is not None:  # codes -> source units -> uV
        scale = np.asarray(range_fit["scale"]) * gain
        uv_offset = np.asarray(range_fit["offset"]) * gain + offset
    else:
        scale, uv_offset = gain * (1 << shift), offset
    attrs["quantization"] = RawQuantization(
        offset=uv_offset.tolist(),
        scale=scale.tolist(),
        source_dtype="float32",
        quantized_dtype=str(raw_array.dtype),
    ).to_dict()
    raw_array.attrs.clear()
    raw_array.attrs.update(attrs)


def _write_pyramid_quantization(
    dst: zarr.Group,
    gain: np.ndarray,
    offset: np.ndarray,
    strides: list[int],
) -> None:
    """Store the ADC calibration on each pyramid level, composed with its encoding."""
    for stride in strides:
        try:
            mm_array: zarr.Array = dst[f"minmax/s{stride}"]  # type: ignore[assignment]
        except KeyError:
            continue

        attrs, shift, range_fit = _pop_encoding(mm_array)
        if range_fit is not None:
            scale = np.asarray(range_fit["scale"]) * gain
            min_offset = np.asarray(range_fit["min_offset"]) * gain + offset
            max_offset = np.asarray(range_fit["max_offset"]) * gain + offset
        else:
            scale, min_offset, max_offset = gain * (1 << shift), offset, offset
        attrs["quantization"] = LevelQuantization(
            min_offset=min_offset.tolist(),
            max_offset=max_offset.tolist(),
            scale=scale.tolist(),
            source_dtype="float32",
            quantized_dtype=str(mm_array.dtype),
        ).to_dict()
        mm_array.attrs.clear()
        mm_array.attrs.update(attrs)


def _apply_adc_quantization(src: zarr.Group, dst: zarr.Group) -> None:
    """Store ADC gain/offset on ``/raw`` if gain_to_uV is available in *src*."""
    pair = _read_gain_offset(src)
    if pair is not None:
        _write_raw_quantization(dst, *pair)


def _apply_adc_pyramid_quantization(src: zarr.Group, dst: zarr.Group, strides: list[int]) -> None:
    """Store ADC gain/offset on pyramid arrays if gain_to_uV is available in *src*."""
    pair = _read_gain_offset(src)
    if pair is not None:
        _write_pyramid_quantization(dst, *pair, strides=strides)


def _write_source_metadata_dict(
    dst: zarr.Group,
    metadata: dict[str, Any],
    strides: list[int],
) -> None:
    """Write source metadata (channel_ids, per-channel properties, gain/offset) from an in-memory dict.

    Dict schema (all keys optional except ``channel_ids``):

    - ``channel_ids`` : 1D array of channel labels, required
    - ``properties`` : per-channel arrays by name, written under ``/properties``
    - ``location``   : ``(C, 2)`` probe coordinates; overrides ``properties["location"]``
    - ``gain_to_uV`` / ``offset_to_uV`` : 1D float arrays for ADC
      dequantization — both must be present together
    - ``probe`` / ``annotations`` / ``si_chain`` / ``t_start`` : passed through via the
      caller into root attrs; not written here
    """
    ch_ids = np.asarray(metadata["channel_ids"])
    _write_string_array(dst, "channel_ids", ch_ids)

    properties = dict(metadata.get("properties") or {})
    if metadata.get("location") is not None:
        properties["location"] = metadata["location"]
    if "location" in properties:
        properties["location"] = np.asarray(properties["location"], dtype=np.float64)
    _write_channel_properties(dst, properties, len(ch_ids))

    pair = _metadata_gain_offset(metadata)
    if pair is not None:
        _write_raw_quantization(dst, *pair)
        _write_pyramid_quantization(dst, *pair, strides=strides)
