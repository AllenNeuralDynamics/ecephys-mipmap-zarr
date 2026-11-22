"""Tests for event layers."""

import json
from pathlib import Path

import numpy as np
import pytest

from ecephys_mipmap_zarr.events import (
    EventGroups,
    events_from_detections,
    events_from_sorting,
    footprints_from_templates,
    read_events,
    write_events,
)

CHANNEL_IDS = [f"CH{i}" for i in range(6)]


def _write(tmp_path: Path, **kwargs: object) -> Path:
    path = tmp_path / "events.zarr"
    args: dict[str, object] = {
        "sample_index": np.array([30, 10, 20, 40]),
        "group": np.array([1, 0, 1, 0]),
        "groups": EventGroups(ids=["a", "b"]),
        "sampling_frequency": 30_000.0,
        "channel_ids": CHANNEL_IDS,
        "kind": "spikes",
    }
    args.update(kwargs)
    write_events(path, **args)  # type: ignore[arg-type]
    return path


def test_round_trip_sorts_by_time_and_keeps_rows_together(tmp_path: Path) -> None:
    path = _write(tmp_path, amplitude=np.array([3.0, 1.0, 2.0, 4.0]), chunk=2)
    out = read_events(path)
    np.testing.assert_array_equal(out["sample_index"], [10, 20, 30, 40])
    np.testing.assert_array_equal(out["group"], [0, 1, 1, 0])
    np.testing.assert_array_equal(out["amplitude"], [1, 2, 3, 4])
    np.testing.assert_array_equal(out["groups"]["id"], ["a", "b"])
    assert out["attrs"]["channel_ids"] == CHANNEL_IDS
    assert out["attrs"]["n_events"] == 4
    assert out["sample_index"].dtype == np.float64


def test_index_names_each_chunks_first_sample(tmp_path: Path) -> None:
    import zarr

    path = _write(tmp_path, sample_index=np.arange(10) * 7, group=np.zeros(10, int), chunk=4)
    root = zarr.open_group(str(path), mode="r")
    np.testing.assert_array_equal(root["index/first_sample"][...], [0, 28, 56])  # type: ignore[index]
    assert root["sample_index"].chunks == (4,)  # type: ignore[union-attr]


def test_refuses_fractional_samples_and_bad_groups(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="whole samples"):
        _write(tmp_path, sample_index=np.array([1.5, 2, 3, 4]))
    with pytest.raises(ValueError, match="index groups"):
        _write(tmp_path, group=np.array([0, 1, 2, 0]))
    with pytest.raises(ValueError, match="channel"):
        _write(tmp_path, channel=np.array([0, 1, 6, -1]))


def test_footprints_keep_channels_above_the_fraction_largest_first() -> None:
    templates = np.zeros((2, 5, 4))
    templates[0, 2] = [-10, -2, -5, 0]
    templates[1, 1] = [0, 0, 8, 0]
    channels, weights = footprints_from_templates(templates, fraction=0.3)
    np.testing.assert_array_equal(channels, [[0, 2], [2, -1]])
    np.testing.assert_array_equal(weights, [[10, 5], [8, 0]])


def _sorting_folder(tmp_path: Path, curation: dict[str, object] | None) -> Path:
    folder = tmp_path / "curated"
    (folder / "properties").mkdir(parents=True)
    spikes = np.zeros(6, dtype=[("sample_index", "<i8"), ("unit_index", "<i8"), ("segment_index", "<i8")])
    spikes["sample_index"] = [5, 9, 12, 20, 21, 30]
    spikes["unit_index"] = [0, 1, 2, 3, 1, 0]
    np.save(folder / "spikes.npy", spikes)
    (folder / "numpysorting_info.json").write_text(
        json.dumps({"sampling_frequency": 30000.0, "unit_ids": [10, 11, 12, 13]})
    )
    np.save(folder / "properties" / "decoder_label.npy", np.array(["sua", "mua", "noise", "sua"]))
    if curation is not None:
        (folder / "curation.json").write_text(json.dumps(curation))
    return folder


def test_sorting_applies_removals_and_merges(tmp_path: Path) -> None:
    folder = _sorting_folder(tmp_path, {"removed": [12], "merges": [{"unit_ids": [11, 13], "new_unit_id": None}]})
    templates = np.zeros((4, 3, 3))
    templates[0, 1] = [-9, 0, 0]
    templates[1, 1] = [0, -6, 0]
    templates[3, 1] = [0, 0, -7]
    args = events_from_sorting(folder, templates, ["CH0", "CH1", "CH2"], spike_amplitudes=-np.arange(6.0))
    groups = args["groups"]
    assert list(groups.ids) == ["10", "11+13"]
    np.testing.assert_array_equal(args["sample_index"], [5, 9, 20, 21, 30])
    np.testing.assert_array_equal(args["group"], [0, 1, 1, 1, 0])
    np.testing.assert_array_equal(args["amplitude"], [0, 1, 3, 4, 5])
    # The merged group's footprint takes each member's channel.
    np.testing.assert_array_equal(groups.footprint_channel, [[0, -1], [2, 1]])
    np.testing.assert_array_equal(groups.properties["decoder_label"], ["sua", "mua"])
    write_events(tmp_path / "out.zarr", **args)
    assert list(read_events(tmp_path / "out.zarr")["groups"]["properties"]["decoder_label"]) == ["sua", "mua"]


def test_sorting_without_curation_keeps_every_unit(tmp_path: Path) -> None:
    folder = _sorting_folder(tmp_path, None)
    args = events_from_sorting(folder, np.ones((4, 3, 3)), ["CH0", "CH1", "CH2"])
    assert list(args["groups"].ids) == ["10", "11", "12", "13"]
    assert args["sample_index"].size == 6
    assert "amplitude" not in args


def test_detections_group_by_a_column(tmp_path: Path) -> None:
    table = {
        "time_s": np.array([1.0, 0.5]),
        "peak_channel": np.array([3, 1]),
        "peak_uv": np.array([900.0, 1200.0]),
        "width_ms": np.array([20.0, 30.0]),
        "half_top_um": np.array([200.0, 300.0]),
        "half_bottom_um": np.array([100.0, 150.0]),
        "counted": np.array([True, False]),
    }
    args = events_from_detections(table, sampling_frequency=2500.0, channel_ids=CHANNEL_IDS, group_by="counted")
    write_events(tmp_path / "d.zarr", **args)
    out = read_events(tmp_path / "d.zarr")
    np.testing.assert_array_equal(out["sample_index"], [1250, 2500])
    np.testing.assert_array_equal(out["groups"]["id"], ["False", "True"])
    np.testing.assert_array_equal(out["group"], [0, 1])
    np.testing.assert_array_equal(out["channel"], [1, 3])
    np.testing.assert_allclose(out["width_s"], [0.03, 0.02])
    np.testing.assert_array_equal(out["depth_range"], [[150, 300], [100, 200]])
