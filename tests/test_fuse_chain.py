"""Tests for rebuilding the AIND preprocessing chain as one fused FIR pass."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("spikeinterface.preprocessing.fused_fir_reference")

from spikeinterface.core import NumpyRecording
from spikeinterface.core.channelslice import ChannelSliceRecording
from spikeinterface.preprocessing import bandpass_filter, common_reference, highpass_filter, phase_shift
from spikeinterface.preprocessing.fused_fir_reference import FusedFIRReferenceRecording

from ecephys_mipmap_zarr._fast_phase_shift import _DEFAULT_N_TAPS
from ecephys_mipmap_zarr.preprocess import SIRecordingSource, _fuse_chain

FS = 30_000.0
N_CHANNELS = 12


def _raw():
    rng = np.random.default_rng(0)
    traces = (rng.standard_normal((60_000, N_CHANNELS)) * 30).astype("int16")
    rec = NumpyRecording([traces], FS)
    rec.set_property("inter_sample_shift", (np.arange(N_CHANNELS) % 6) / 13.0)
    return rec


def _highpass(rec, **kwargs):
    return highpass_filter(rec, freq_min=300.0, dtype="int16", **kwargs)


def _bandpass(rec, **kwargs):
    return bandpass_filter(rec, freq_min=300.0, freq_max=6000.0, dtype="int16", **kwargs)


def _aind_chain(raw, kept, filt=_highpass):
    """The AIND shape: PhaseShift → Highpass or Bandpass → ChannelSlice → CMR."""
    rec = phase_shift(raw, margin_ms=100.0)
    rec = filt(rec, margin_ms=5.0)
    rec = ChannelSliceRecording(rec, kept)
    return common_reference(rec, reference="global", operator="median", dtype="int16")


def _unfused_fir(raw, kept, filt=_highpass):
    """The same steps with the slice first and FIR methods, unfused."""
    rec = ChannelSliceRecording(raw, kept)
    rec = phase_shift(rec, margin_ms=100.0, method="fir", n_taps=_DEFAULT_N_TAPS, dtype="float32")
    rec = filt(rec, method="fir")
    return common_reference(rec, reference="global", operator="median", dtype="int16")


def _assert_rounding_close(actual, expected):
    """Equal up to float32 rounding before the int16 cast.

    A rounding flip of one count moves an even-count median by half a count, which can round to two.
    """
    diff = np.abs(actual.astype(np.int32) - expected)
    assert diff.max() <= 2
    assert (diff > 0).mean() < 1e-3


KEPT = [cid for i, cid in enumerate(_raw().channel_ids) if i not in (2, 7)]


FILTERS = pytest.mark.parametrize("filt", [_highpass, _bandpass], ids=["highpass", "bandpass"])


@FILTERS
def test_fuses_aind_chain(filt):
    raw = _raw()
    fused = _fuse_chain(_aind_chain(raw, KEPT, filt))
    assert isinstance(fused, FusedFIRReferenceRecording)
    assert list(fused.channel_ids) == KEPT
    assert fused.get_dtype() == np.int16


@FILTERS
def test_fused_matches_unfused_fir_chain(filt):
    raw = _raw()
    fused = _fuse_chain(_aind_chain(raw, KEPT, filt)).get_traces(start_frame=10_000, end_frame=40_000)
    expected = _unfused_fir(raw, KEPT, filt).get_traces(start_frame=10_000, end_frame=40_000)
    _assert_rounding_close(fused, expected)


def test_fused_ignores_fast_patches():
    """The fused pass never reaches the segments the ``fast_*`` patches replace."""
    from ecephys_mipmap_zarr import _fast_cmr, _fast_filter

    raw = _raw()
    unpatched = _fuse_chain(_aind_chain(raw, KEPT)).get_traces(start_frame=10_000, end_frame=40_000)
    _fast_filter.install()
    _fast_cmr.install()
    try:
        patched = _fuse_chain(_aind_chain(raw, KEPT)).get_traces(start_frame=10_000, end_frame=40_000)
    finally:
        _fast_filter.uninstall()
        _fast_cmr.uninstall()
    np.testing.assert_array_equal(patched, unpatched)


def test_keeps_outer_slice_on_top():
    raw = _raw()
    outer = KEPT[:5]
    fused = _fuse_chain(ChannelSliceRecording(_aind_chain(raw, KEPT), outer))
    assert list(fused.channel_ids) == outer
    full = _fuse_chain(_aind_chain(raw, KEPT)).get_traces(start_frame=10_000, end_frame=20_000)
    np.testing.assert_array_equal(fused.get_traces(start_frame=10_000, end_frame=20_000), full[:, :5])


def test_leaves_other_chains_unchanged():
    raw = _raw()
    rec = common_reference(highpass_filter(raw, freq_min=300.0, filter_order=3), reference="global")
    assert _fuse_chain(rec) is rec


def test_split_threaded_reads_match_one_read():
    fused = _fuse_chain(_aind_chain(_raw(), KEPT))
    whole = SIRecordingSource(fused)[5_000:50_000, :]
    split = SIRecordingSource(fused, read_samples=7_000, read_workers=3, step_threads=2)[5_000:50_000, :]
    _assert_rounding_close(split, whole)
