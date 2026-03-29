"""Shared store utilities for opening and managing zarr groups.

Internal module — not part of the public API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import zarr

from ecephys_mipmap_zarr.metadata import RawQuantization

__all__: list[str] = []  # private module


def _open_store(path: str | Path, mode: str = "r", cache_dir: str | Path | None = None) -> zarr.Group:
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
            fs = fsspec.filesystem(
                "filecache",
                target_protocol=protocol,
                cache_storage=str(cache_dir),
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


def _copy_source_arrays(src: zarr.Group, dst: zarr.Group) -> None:
    """Copy channel_ids and properties/location arrays from source to destination.

    Parameters
    ----------
    src : zarr.Group
        Source store root group.
    dst : zarr.Group
        Destination store root group.
    """
    if "channel_ids" in list(src):
        ch_ids = np.asarray(src["channel_ids"][:])  # type: ignore[index]
        dst.create_array("channel_ids", data=ch_ids, overwrite=True)

    if "properties" in list(src):
        try:
            props_src = src["properties"]
            if "location" in list(props_src):  # type: ignore[arg-type]
                loc = np.asarray(props_src["location"][:])  # type: ignore[index]
                props_dst = dst.require_group("properties")
                props_dst.create_array("location", data=loc, overwrite=True)
        except (KeyError, AttributeError):
            pass  # properties group doesn't have what we need


def _apply_adc_quantization(src: zarr.Group, dst: zarr.Group, lsb_shift: int = 0) -> None:
    """Store ADC gain/offset as RawQuantization if gain_to_uV is available.

    When the source is an SI zarr with ``properties/gain_to_uV`` and
    ``properties/offset_to_uV``, store these as :class:`RawQuantization`
    in the destination's ``/raw`` array attrs.  This lets the reader
    dequantize int16 ADC values to float32 µV transparently.

    When *lsb_shift* > 0, the gain is multiplied by ``2**lsb_shift``
    so that the reader's dequantization path handles both the bit-shift
    reconstruction and unit conversion in a single step.

    Parameters
    ----------
    src : zarr.Group
        Source SI zarr store.
    dst : zarr.Group
        Destination mipmap store (must already contain ``/raw``).
    lsb_shift : int, optional
        LSB shift applied to raw data, by default 0.
    """
    try:
        props = src["properties"]
        gain = np.asarray(props["gain_to_uV"][:])  # type: ignore[index]
        offset = np.asarray(props["offset_to_uV"][:])  # type: ignore[index]
    except (KeyError, AttributeError):
        return  # No gain/offset data available

    # Combine lsb_shift into gain: effective_scale = gain * 2^shift
    effective_gain = gain * (1 << lsb_shift) if lsb_shift > 0 else gain

    try:
        raw_array: zarr.Array = dst["raw"]  # type: ignore[assignment]
        raw_dtype = str(raw_array.dtype)
    except KeyError:
        return  # No /raw in destination (pyramid-only store)

    rq = RawQuantization(
        offset=offset.tolist(),
        scale=effective_gain.tolist(),
        source_dtype="float32",
        quantized_dtype=raw_dtype,
    )

    raw_array.attrs.update({"quantization": rq.to_dict()})
    # When combined with lsb_shift, remove the standalone lsb_shift attr
    # since dequantization via RawQuantization handles everything
    if lsb_shift > 0:
        existing = dict(raw_array.attrs)
        existing.pop("lsb_shift", None)
        existing.pop("original_dtype", None)
        existing["quantization"] = rq.to_dict()
        raw_array.attrs.clear()
        raw_array.attrs.update(existing)
