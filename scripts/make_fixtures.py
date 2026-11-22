"""Build the cross-reader fixture stores and the values Python decodes from them.

Usage: ``uv run python -I scripts/make_fixtures.py [OUT_DIR]`` (default ``viewer-web/test/fixtures``).

Each store is small but chunked so reads cross channel-chunk, time-chunk and shard boundaries and end
in a partial chunk. ``index.json`` lists, per store, entries naming an array, a channel range and a
sample or block range, with the codes and decoded values Python read for it as ``.npy`` files; an
entry's ``raw`` is the store's raw samples, wherever ``raw_source`` puts them.
Pyramid-only stores name their raw store by absolute path, so the fixtures are rebuilt wherever they
are used rather than moved.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from ecephys_mipmap_zarr.build import add_pyramid, build_mipmap
from ecephys_mipmap_zarr.events import EventGroups, write_events
from ecephys_mipmap_zarr.metadata import BuildConfig, ChunkConfig, EncodingConfig
from ecephys_mipmap_zarr.read import MipmapStore, open_mipmap

N_CHANNELS = 8
N_SAMPLES = 200_000
FS = 30_000.0
CFG = BuildConfig(
    chunk_config=ChunkConfig(
        raw_chunk=(4, 1024),
        raw_shard=(4, 4096),
        minmax_chunk=(4, 2, 256),
        minmax_shard=(4, 2, 1024),
    ),
    min_top_blocks=64,  # three levels at 200K samples: s32, s256, s2048
    consolidate_metadata=True,
)
GAIN = 2.34375 + 0.125 * np.arange(N_CHANNELS)
OFFSET = 1.5 * np.arange(N_CHANNELS) - 4.0
ENCODINGS = ("shift", "range", "none")
CHANNEL_RANGES = ((0, N_CHANNELS), (2, 6))  # all, and a range straddling the channel-chunk boundary


def make_signal(dtype: str) -> np.ndarray:
    """Return seeded ``(T, C)`` noise with a per-channel ramp and one large spike per channel.

    The ramp makes channels distinguishable and the spikes land at distinct samples, so a channel swap
    or an off-by-one block changes values.
    """
    rng = np.random.default_rng(20261118)
    t = np.arange(N_SAMPLES)[:, None]
    c = np.arange(N_CHANNELS)[None, :]
    signal = rng.normal(0.0, 12.0, size=(N_SAMPLES, N_CHANNELS))
    signal += 40.0 * c - 150.0 + 100.0 * np.sin(2 * np.pi * t / N_SAMPLES * (c + 1))
    for ch in range(N_CHANNELS):
        signal[5_000 + 20_011 * ch, ch] -= 700.0
    if dtype == "int16":
        return np.round(signal).astype(np.int16)
    return (signal * 0.25).astype(np.float32)


def channel_metadata(gain: bool) -> dict[str, Any]:
    """Return source metadata with ids, properties and, when *gain*, an ADC calibration."""
    meta: dict[str, Any] = {
        "channel_ids": np.array([f"CH{c + 1}" for c in range(N_CHANNELS)]),
        "properties": {
            "location": np.stack([np.tile([16.0, 48.0], N_CHANNELS // 2), 20.0 * np.arange(N_CHANNELS)], axis=1),
            "brain_area": np.array(["CA1"] * 4 + ["DG"] * 4),
        },
    }
    if gain:
        meta["gain_to_uV"] = GAIN
        meta["offset_to_uV"] = OFFSET
    return meta


def ranges(total: int, chunk: int, shard: int, width: int) -> list[tuple[int, int]]:
    """Return ranges covering the start, a chunk boundary, a shard boundary and the partial tail."""
    candidates = [
        (0, width),
        (chunk - width // 2, chunk + width // 2),
        (shard - width // 2, shard + width // 2),
        (total - width, total),
    ]
    clamped = {(max(0, a), min(total, b)) for a, b in candidates}
    return sorted(r for r in clamped if r[0] < r[1])


def write_entries(store: MipmapStore, out: Path, name: str) -> list[dict[str, Any]]:
    """Read every fixture range through *store* and save codes and decoded values."""
    entries: list[dict[str, Any]] = []
    raw_ranges = ranges(store.n_samples, 1024, 4096, 200)
    arrays: list[tuple[str, int | None, list[tuple[int, int]]]] = []
    if store._raw_array is not None:
        arrays.append(("raw", None, raw_ranges))
    for level in store.strides:
        n_blocks = store._minmax_arrays[level].shape[2]
        arrays.append((f"minmax/s{level}", level, ranges(n_blocks, 256, 1024, 12)))

    for array, stride, array_ranges in arrays:
        for c0, c1 in CHANNEL_RANGES:
            for i0, i1 in array_ranges:
                stem = f"{name}/{array.replace('/', '_')}_c{c0}-{c1}_i{i0}-{i1}"
                if stride is None:
                    codes = store._read_raw(i0, i1, slice(c0, c1), dequantize=False).data
                    decoded = store._read_raw(i0, i1, slice(c0, c1), dequantize=True).data
                else:
                    t0, t1 = i0 * stride, min(i1 * stride, store.n_samples)
                    codes = store._read_minmax(stride, t0, t1, slice(c0, c1), dequantize=False).data
                    decoded = store._read_minmax(stride, t0, t1, slice(c0, c1), dequantize=True).data
                np.save(out / f"{stem}.codes.npy", np.ascontiguousarray(codes))
                np.save(out / f"{stem}.decoded.npy", np.ascontiguousarray(decoded, dtype=np.float64))
                entries.append(
                    {
                        "array": array,
                        "channels": [c0, c1],
                        "range": [i0, i1],
                        "codes": f"{stem}.codes.npy",
                        "decoded": f"{stem}.decoded.npy",
                    }
                )
    return entries


def write_events_fixture(path: Path) -> None:
    """Write an event layer: a group per channel's spike, plus a group of evenly spaced events on the first channel.

    Its channel ids run backwards and include one no store has, so a reader must map footprints by id.
    """
    ids = [f"CH{c + 1}" for c in reversed(range(N_CHANNELS))] + ["CH99"]
    layer_index = {cid: i for i, cid in enumerate(ids)}
    spikes = [5_000 + 20_011 * ch for ch in range(N_CHANNELS)]
    extra = [1_000 + 9_000 * k for k in range(20)]
    footprint = [
        [layer_index[f"CH{ch + 1}"], layer_index[f"CH{min(ch, N_CHANNELS - 2) + 2}"], layer_index["CH99"]]
        for ch in range(N_CHANNELS)
    ]
    footprint.append([layer_index["CH1"], -1, -1])
    weight = [[700.0, 100.0, 50.0]] * N_CHANNELS + [[300.0, 0.0, 0.0]]
    write_events(
        path,
        sample_index=np.array(spikes + extra),
        group=np.array(list(range(N_CHANNELS)) + [N_CHANNELS] * len(extra)),
        groups=EventGroups(
            ids=[f"unit-{ch + 1}" for ch in range(N_CHANNELS)] + ["extra"],
            footprint_channel=np.array(footprint, np.int32),
            footprint_weight=np.array(weight, np.float32),
            properties={"decoder_label": np.array(["sua"] * N_CHANNELS + ["mua"])},
        ),
        sampling_frequency=FS,
        channel_ids=ids,
        kind="spikes",
        amplitude=np.array([700.0] * N_CHANNELS + [300.0] * len(extra)),
        chunk=5,
    )


def write_si_layout_store(path: Path, source: np.ndarray) -> None:
    """Write a SpikeInterface-style Zarr v2 store: ``traces_seg0`` ``(T, C)`` plus channel arrays."""
    root = zarr.open_group(str(path), mode="w", zarr_format=2)
    root.attrs.update({"sampling_frequency": FS, "num_segments": 1})
    traces = root.create_array("traces_seg0", shape=source.shape, dtype=source.dtype, chunks=(30_000, N_CHANNELS))
    traces[:] = source
    root.create_array("channel_ids", data=np.array([f"CH{c + 1}" for c in range(N_CHANNELS)], dtype="U8"))
    zarr.consolidate_metadata(str(path))


def main(out: Path) -> None:
    """Build every fixture store under *out* and write ``index.json``."""
    if out.exists():
        if not (out / "index.json").exists():
            sys.exit(f"{out} exists and holds no index.json; refusing to replace it")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    stores: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for dtype in ("int16", "float32"):
        source = make_signal(dtype)
        for gain in (False, True):
            for raw in ENCODINGS:
                for pyramids in ENCODINGS:
                    name = f"{dtype}_raw-{raw}_pyr-{pyramids}" + ("_gain" if gain else "")
                    encoding = EncodingConfig(
                        raw=raw,  # type: ignore[arg-type]
                        pyramids=pyramids,  # type: ignore[arg-type]
                        noise_floor=8.0 if "shift" in (raw, pyramids) else None,
                    )
                    try:
                        build_mipmap(
                            source,
                            out / f"{name}.zarr",
                            sampling_frequency=FS,
                            units="uV" if dtype == "float32" and not gain else None,
                            config=CFG,
                            encoding=encoding,
                            source_metadata=channel_metadata(gain),
                        )
                    except ValueError as exc:
                        skipped.append({"name": name, "reason": str(exc)})
                        shutil.rmtree(out / f"{name}.zarr", ignore_errors=True)
                        continue
                    (out / name).mkdir()
                    stores.append(
                        {
                            "name": name,
                            "path": f"{name}.zarr",
                            "source_dtype": dtype,
                            "raw_encoding": raw,
                            "pyramid_encoding": pyramids,
                            "gain": gain,
                            "entries": write_entries(open_mipmap(out / f"{name}.zarr"), out, name),
                        }
                    )

    int16 = make_signal("int16")
    write_si_layout_store(out / "si_layout.zarr", int16)
    for name, raw_store in (("pyramid-only_v3", "int16_raw-none_pyr-none.zarr"), ("pyramid-only_si", "si_layout.zarr")):
        add_pyramid(
            out / raw_store,
            out / f"{name}.zarr",
            config=CFG,
            encoding=EncodingConfig(pyramids="shift", noise_floor=8.0),
        )
        (out / name).mkdir()
        stores.append(
            {
                "name": name,
                "path": f"{name}.zarr",
                "source_dtype": "int16",
                "raw_encoding": None,
                "pyramid_encoding": "shift",
                "gain": False,
                "raw_source": raw_store,
                "entries": write_entries(open_mipmap(out / f"{name}.zarr"), out, name),
            }
        )

    write_events_fixture(out / "events.zarr")
    index = {"n_channels": N_CHANNELS, "n_samples": N_SAMPLES, "stores": stores, "skipped": skipped}
    (out / "index.json").write_text(json.dumps(index, indent=1, sort_keys=True) + "\n")
    print(f"{len(stores)} stores, {sum(len(s['entries']) for s in stores)} entries, {len(skipped)} skipped -> {out}")


if __name__ == "__main__":
    default_out = Path(__file__).resolve().parent.parent / "viewer-web/test/fixtures"
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else default_out)
