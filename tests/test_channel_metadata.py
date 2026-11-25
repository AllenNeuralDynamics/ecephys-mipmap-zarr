"""Stores carry the source's per-channel properties, probe and SI chain."""

import numpy as np
import pytest
import zarr

from ecephys_mipmap_zarr import build_mipmap, open_mipmap
from ecephys_mipmap_zarr.metadata import LOSSLESS_ENCODING, BuildConfig, ChunkConfig

SMALL_CFG = BuildConfig(
    chunk_config=ChunkConfig(
        raw_chunk=(4, 1024), raw_shard=(4, 4096), minmax_chunk=(4, 2, 256), minmax_shard=(4, 2, 1024)
    )
)
N_CH = 4
GAIN = np.full(N_CH, 2.5)


def _si_zarr(path, traces):
    """A minimal SpikeInterface-layout zarr v2 store."""
    root = zarr.open_group(str(path), mode="w", zarr_format=2)
    root.attrs.update({"sampling_frequency": 30000.0, "probe": {"probes": [{"ndim": 2, "si_units": "um"}]}})
    root.create_array("traces_seg0", data=traces)
    root.create_array("channel_ids", data=np.array([f"CH{i}" for i in range(N_CH)]))
    props = root.require_group("properties")
    props.create_array("location", data=np.column_stack([np.zeros(N_CH), 20.0 * np.arange(N_CH)]))
    props.create_array("group", data=np.array([0, 0, 1, 1]))
    props.create_array("inter_sample_shift", data=np.linspace(0, 0.75, N_CH))
    props.create_array("channel_name", data=np.array([f"name{i}" for i in range(N_CH)]))
    props.create_array("gain_to_uV", data=GAIN)
    props.create_array("offset_to_uV", data=np.zeros(N_CH))
    return path


def test_metadata_from_copies_channel_properties_but_not_calibration(tmp_path):
    traces = np.random.default_rng(0).integers(-100, 100, (100_000, N_CH), dtype=np.int16)
    src = _si_zarr(tmp_path / "si.zarr", traces)
    out = tmp_path / "out.zarr"
    build_mipmap(
        traces,
        out,
        sampling_frequency=30000.0,
        units="uV",
        config=SMALL_CFG,
        encoding=LOSSLESS_ENCODING,
        metadata_from=str(src),
    )

    props = open_mipmap(out).channel_properties
    assert set(props) == {"location", "group", "inter_sample_shift", "channel_name"}
    np.testing.assert_array_equal(props["group"], [0, 0, 1, 1])
    assert list(props["channel_name"]) == [f"name{i}" for i in range(N_CH)]
    # Calibration lives in the quantization attr only, so decoded data is already uV.
    w = open_mipmap(out).get_window(0, 1000, num_points=100_000)
    np.testing.assert_allclose(np.asarray(w.data), traces[:1000].T * GAIN[:, None], rtol=1e-6)


def test_metadata_from_locates_channels_by_the_probe_without_a_location_property(tmp_path):
    traces = np.random.default_rng(0).integers(-100, 100, (100_000, N_CH), dtype=np.int16)
    src = _si_zarr(tmp_path / "si.zarr", traces)
    root = zarr.open_group(str(src), mode="r+")
    del root["properties/location"]
    # Contacts listed out of channel order: channel c is the contact wired to device channel c.
    positions = [[0.0, 30.0], [16.0, 0.0], [0.0, 15.0], [16.0, 45.0]]
    root.attrs["probe"] = {"probes": [{"contact_positions": positions, "device_channel_indices": [2, 0, 1, 3]}]}
    out = tmp_path / "out.zarr"
    build_mipmap(
        traces, out, sampling_frequency=30000.0, config=SMALL_CFG, encoding=LOSSLESS_ENCODING, metadata_from=str(src)
    )

    store = open_mipmap(out)
    np.testing.assert_array_equal(store.channel_properties["location"], [[16, 0], [0, 15], [0, 30], [16, 45]])
    assert zarr.open_group(str(out), mode="r").attrs["source_metadata"]["source"] == str(src)


def test_metadata_from_leaves_location_out_when_a_channel_has_no_contact(tmp_path):
    traces = np.random.default_rng(0).integers(-100, 100, (100_000, N_CH), dtype=np.int16)
    src = _si_zarr(tmp_path / "si.zarr", traces)
    root = zarr.open_group(str(src), mode="r+")
    del root["properties/location"]
    root.attrs["probe"] = {"probes": [{"contact_positions": [[0.0, 0.0]] * 3, "device_channel_indices": [0, 1, 2]}]}
    out = tmp_path / "out.zarr"
    build_mipmap(
        traces, out, sampling_frequency=30000.0, config=SMALL_CFG, encoding=LOSSLESS_ENCODING, metadata_from=str(src)
    )

    assert "location" not in open_mipmap(out).channel_properties


def test_preprocessed_store_carries_probe_properties_and_chain(tmp_path):
    si = pytest.importorskip("spikeinterface.core")
    spre = pytest.importorskip("spikeinterface.preprocessing")
    probeinterface = pytest.importorskip("probeinterface")
    from ecephys_mipmap_zarr.preprocess import SIRecordingSource, build_source_metadata

    rec = si.generate_recording(num_channels=8, durations=[4.0], seed=0)
    rec.set_property("group", np.arange(8) % 2)
    rec.set_property("inter_sample_shift", np.linspace(0, 0.875, 8))
    kept = rec.channel_ids[[5, 1, 6, 2]]
    chain = spre.common_reference(spre.phase_shift(rec)).select_channels(kept)

    meta = build_source_metadata(chain)
    assert meta["si_chain"][-3:] == ["PhaseShiftRecording", "CommonReferenceRecording", "ChannelSliceRecording"]

    out = tmp_path / "out.zarr"
    build_mipmap(
        SIRecordingSource(chain),
        out,
        sampling_frequency=chain.sampling_frequency,
        units="uV",
        config=SMALL_CFG,
        encoding=LOSSLESS_ENCODING,
        source_metadata=meta,
    )
    store = open_mipmap(out)
    props = store.channel_properties
    np.testing.assert_array_equal(props["location"], chain.get_channel_locations())
    np.testing.assert_array_equal(props["group"], [1, 1, 0, 0])
    np.testing.assert_array_equal(props["inter_sample_shift"], chain.get_property("inter_sample_shift"))
    assert store.source_metadata["si_chain"] == meta["si_chain"]

    probe = probeinterface.ProbeGroup.from_dict(store.source_metadata["probe"]).probes[0]
    # Contacts are in store channel order, so contact i is channel i.
    np.testing.assert_array_equal(probe.device_channel_indices, np.arange(len(kept)))
    np.testing.assert_array_equal(probe.contact_positions, props["location"])
