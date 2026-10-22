"""Unit tests for the preprocess (SI adapter) module.

These tests avoid an actual SpikeInterface dependency by exercising
only the parts of :mod:`ecephys_mipmap_zarr.preprocess` that do not
touch SI: the :class:`SIRecordingSource` adapter and the provenance-
manipulation helpers.

The end-to-end S3 round-trip (``si.load(prov) → get_traces``) is
covered manually via ``scripts/verify_analyzer_chain.py`` because it
requires the ``si`` extra and network access.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from ecephys_mipmap_zarr.preprocess import (
    SIRecordingSource,
    _clear_rel,
    _DecodedChunkCache,
    _fix_leaf,
    _select_fir_phase_shift,
    _unwrap_outer_channelslice,
)


class _FakeRecording:
    """Just enough BaseRecording surface for SIRecordingSource."""

    def __init__(self, n_samples, n_channels, dtype="int16", seed=0):
        self._n_samples = n_samples
        self._n_channels = n_channels
        self._dtype = np.dtype(dtype)
        rng = np.random.default_rng(seed)
        self._data = rng.integers(-1000, 1000, size=(n_samples, n_channels)).astype(dtype)

    def get_num_samples(self):
        return self._n_samples

    def get_num_channels(self):
        return self._n_channels

    def get_dtype(self):
        return self._dtype

    def get_traces(self, start_frame=None, end_frame=None):
        s = 0 if start_frame is None else start_frame
        e = self._n_samples if end_frame is None else end_frame
        return self._data[s:e].copy()


class TestSIRecordingSource:
    def test_shape_and_dtype(self):
        rec = _FakeRecording(1000, 4)
        src = SIRecordingSource(rec)
        assert src.shape == (1000, 4)
        assert src.dtype == np.int16

    def test_slice_get_traces_round_trip(self):
        rec = _FakeRecording(1000, 4)
        src = SIRecordingSource(rec)
        window = src[10:30, :]
        assert window.shape == (20, 4)
        np.testing.assert_array_equal(window, rec.get_traces(start_frame=10, end_frame=30))

    def test_channel_subslice(self):
        rec = _FakeRecording(100, 8)
        src = SIRecordingSource(rec)
        sub = src[:, slice(2, 5)]
        assert sub.shape == (100, 3)

    def test_rejects_1d_indexing(self):
        src = SIRecordingSource(_FakeRecording(100, 4))
        with pytest.raises(TypeError):
            _ = src[10:20]

    def test_window_offset_and_duration(self):
        rec = _FakeRecording(1000, 4)
        src = SIRecordingSource(rec, start_sample=200, n_samples=300)
        assert src.shape == (300, 4)
        # Indices into the window map to the absolute frames
        np.testing.assert_array_equal(src[0:10, :], rec.get_traces(start_frame=200, end_frame=210))
        np.testing.assert_array_equal(src[50:75, :], rec.get_traces(start_frame=250, end_frame=275))
        # Full window slice lines up with the underlying recording
        np.testing.assert_array_equal(src[:, :], rec.get_traces(start_frame=200, end_frame=500))

    @pytest.mark.parametrize("read_workers", [1, 3])
    def test_split_reads_match_one_read(self, read_workers):
        rec = _FakeRecording(1000, 4)
        src = SIRecordingSource(rec, start_sample=13, read_samples=64, read_workers=read_workers)
        np.testing.assert_array_equal(src[5:900, :], rec.get_traces(start_frame=18, end_frame=913))
        np.testing.assert_array_equal(src[:, slice(1, 3)], rec.get_traces(start_frame=13)[:, 1:3])

    def test_split_reads_stay_under_read_samples(self):
        rec = _FakeRecording(1000, 4)
        calls = []
        original = rec.get_traces

        def recording_get_traces(start_frame=None, end_frame=None):
            calls.append(end_frame - start_frame)
            return original(start_frame=start_frame, end_frame=end_frame)

        rec.get_traces = recording_get_traces
        SIRecordingSource(rec, read_samples=300)[0:1000, :]
        assert calls == [300, 300, 300, 100]

    def test_rejects_nonpositive_read_samples(self):
        with pytest.raises(ValueError):
            SIRecordingSource(_FakeRecording(100, 4), read_samples=0)

    def test_window_rejects_out_of_range(self):
        rec = _FakeRecording(100, 4)
        with pytest.raises(ValueError):
            SIRecordingSource(rec, start_sample=200)
        with pytest.raises(ValueError):
            SIRecordingSource(rec, start_sample=50, n_samples=60)  # 50+60 > 100


class TestFixLeaf:
    def test_substitutes_leaf_path(self):
        prov = {
            "class": "CommonReference",
            "kwargs": {
                "recording": {
                    "class": "ZarrRecordingExtractor",
                    "kwargs": {"folder_path": "/data/old.zarr", "storage_options": None},
                },
            },
            "relative_paths": True,
        }
        _fix_leaf(prov, "s3://new/path.zarr", storage_options={"anon": True})
        leaf = prov["kwargs"]["recording"]
        assert leaf["kwargs"]["folder_path"] == "s3://new/path.zarr"
        assert leaf["kwargs"]["storage_options"] == {"anon": True}

    def test_handles_parent_recording_key(self):
        prov = {
            "class": "ChannelSlice",
            "kwargs": {
                "parent_recording": {
                    "class": "ZarrRecordingExtractor",
                    "kwargs": {"folder_path": "/old"},
                },
            },
        }
        _fix_leaf(prov, "/new")
        assert prov["kwargs"]["parent_recording"]["kwargs"]["folder_path"] == "/new"


class TestClearRel:
    def test_clears_throughout(self):
        prov = {
            "relative_paths": True,
            "kwargs": {
                "recording": {
                    "relative_paths": True,
                    "class": "Inner",
                    "kwargs": {
                        "parent_recording": {
                            "relative_paths": True,
                            "class": "Leaf",
                            "kwargs": {},
                        },
                    },
                },
            },
        }
        _clear_rel(prov)
        assert prov["relative_paths"] is False
        assert prov["kwargs"]["recording"]["relative_paths"] is False
        assert prov["kwargs"]["recording"]["kwargs"]["parent_recording"]["relative_paths"] is False


class TestUnwrapOuterChannelslice:
    def test_peels_outer_wrapper(self):
        inner = {"class": "CommonReferenceRecording", "kwargs": {}}
        prov = {
            "class": "spikeinterface.core.channelslice.ChannelSliceRecording",
            "kwargs": {"parent_recording": inner, "channel_ids": ["CH0", "CH1"]},
        }
        result = _unwrap_outer_channelslice(prov)
        assert result is inner

    def test_returns_unchanged_when_no_outer_slice(self):
        prov = {"class": "CommonReferenceRecording", "kwargs": {}}
        result = _unwrap_outer_channelslice(prov)
        assert result is prov

    def test_handles_recording_kwarg_variant(self):
        # Older SI versions used "recording" rather than "parent_recording"
        inner = {"class": "Inner", "kwargs": {}}
        prov = {
            "class": "ChannelSliceRecording",
            "kwargs": {"recording": inner},
        }
        result = _unwrap_outer_channelslice(prov)
        assert result is inner


class TestSelectFirPhaseShift:
    @staticmethod
    def _chain(phase_shift_kwargs):
        leaf = {"class": "spikeinterface.core.zarrextractors.ZarrRecordingExtractor", "kwargs": {}}
        ps = {
            "class": "spikeinterface.preprocessing.phase_shift.PhaseShiftRecording",
            "kwargs": {"recording": leaf, **phase_shift_kwargs},
        }
        hp = {
            "class": "spikeinterface.preprocessing.filter.HighpassFilterRecording",
            "kwargs": {"recording": ps, "dtype": "<i2"},
        }
        return hp, ps, leaf

    def test_selects_fir_with_float32_output_when_dtype_unset(self):
        hp, ps, leaf = self._chain({"margin_ms": 100.0})
        _select_fir_phase_shift(hp)
        assert ps["kwargs"]["method"] == "fir"
        assert ps["kwargs"]["n_taps"] == 32
        assert ps["kwargs"]["dtype"] == "float32"
        assert ps["kwargs"]["margin_ms"] == 100.0
        assert hp["kwargs"] == {"recording": ps, "dtype": "<i2"}
        assert leaf["kwargs"] == {}

    def test_keeps_pinned_dtype(self):
        hp, ps, _ = self._chain({"dtype": "<i2"})
        _select_fir_phase_shift(hp)
        assert ps["kwargs"]["method"] == "fir"
        assert ps["kwargs"]["dtype"] == "<i2"


class _CountingArray:
    """A chunked (T, C) array that records the time slices it is asked for."""

    def __init__(self, data, chunk_len, fail_first=0):
        self._data = data
        self.shape = data.shape
        self.dtype = data.dtype
        self.chunks = (chunk_len, data.shape[1])
        self.reads = []
        self._fail_first = fail_first

    def __getitem__(self, key):
        if self._fail_first:
            self._fail_first -= 1
            raise OSError("transient")
        self.reads.append((key.start, key.stop))
        return self._data[key].copy()


class TestDecodedChunkCache:
    @staticmethod
    def _array(**kwargs):
        data = np.random.default_rng(0).integers(-1000, 1000, size=(1050, 3)).astype("int16")
        return data, _CountingArray(data, 100, **kwargs)

    def test_slices_match_array(self):
        data, arr = self._array()
        cache = _DecodedChunkCache(arr, max_chunks=4)
        for a, b in [(0, 1050), (95, 105), (1000, 1050), (250, 251), (300, 300)]:
            np.testing.assert_array_equal(cache[a:b], data[a:b])
        assert cache.shape == data.shape

    def test_overlapping_reads_read_each_chunk_once(self):
        data, arr = self._array()
        cache = _DecodedChunkCache(arr, max_chunks=4)
        for a in range(0, 500, 50):
            np.testing.assert_array_equal(cache[max(0, a - 20) : a + 70], data[max(0, a - 20) : a + 70])
        assert arr.reads == [(0, 100), (100, 200), (200, 300), (300, 400), (400, 500), (500, 600)]

    def test_one_read_per_run_of_missing_chunks(self):
        _, arr = self._array()
        cache = _DecodedChunkCache(arr, max_chunks=8)
        cache[150:250]
        cache[0:550]
        assert arr.reads == [(100, 300), (0, 100), (300, 600)]

    def test_concurrent_reads_share_chunks(self):
        data, arr = self._array()
        cache = _DecodedChunkCache(arr, max_chunks=16)
        with ThreadPoolExecutor(8) as pool:
            outs = list(pool.map(lambda a: cache[a : a + 130], range(0, 900, 30)))
        for a, out in zip(range(0, 900, 30), outs, strict=True):
            np.testing.assert_array_equal(out, data[a : a + 130])
        covered = sorted(t for start, stop in arr.reads for t in range(start, stop, 100))
        assert covered == list(range(0, 1000, 100))

    def test_read_ahead_fills_following_chunks(self):
        data, arr = self._array()
        cache = _DecodedChunkCache(arr, max_chunks=8, read_ahead=3)
        np.testing.assert_array_equal(cache[0:150], data[0:150])
        np.testing.assert_array_equal(cache[150:500], data[150:500])
        covered = sorted(t for start, stop in arr.reads for t in range(start, stop, 100))
        assert covered[:5] == [0, 100, 200, 300, 400]
        assert len(covered) == len(set(covered))

    def test_read_ahead_stops_at_array_end(self):
        data, arr = self._array()
        cache = _DecodedChunkCache(arr, max_chunks=8, read_ahead=4)
        np.testing.assert_array_equal(cache[900:1050], data[900:1050])
        np.testing.assert_array_equal(cache[0:10], data[0:10])
        assert max(stop for _, stop in arr.reads) == 1050

    def test_read_ahead_capped_by_cache_size(self):
        _, arr = self._array()
        assert _DecodedChunkCache(arr, max_chunks=4, read_ahead=8)._read_ahead == 2

    def test_failed_read_is_retried(self):
        data, arr = self._array(fail_first=1)
        cache = _DecodedChunkCache(arr, max_chunks=4)
        with pytest.raises(OSError):
            cache[0:150]
        np.testing.assert_array_equal(cache[0:150], data[0:150])
