"""Unit tests for cli._WindowedSource and the _resolve_window helper."""

from __future__ import annotations

import numpy as np
import pytest

from ecephys_mipmap_zarr.cli import _resolve_window, _WindowedSource


class TestWindowedSource:
    def test_shape_and_dtype_reflect_window(self):
        underlying = np.arange(1000 * 4, dtype=np.int16).reshape(1000, 4)
        w = _WindowedSource(underlying, start_sample=100, end_sample=300)
        assert w.shape == (200, 4)
        assert w.dtype == np.int16

    def test_2d_slice_maps_to_absolute_indices(self):
        underlying = np.arange(1000 * 4, dtype=np.int16).reshape(1000, 4)
        w = _WindowedSource(underlying, start_sample=100, end_sample=300)
        # Window-relative [0:10, :] should equal underlying[100:110, :]
        np.testing.assert_array_equal(w[0:10, :], underlying[100:110, :])
        np.testing.assert_array_equal(w[50:75, :], underlying[150:175, :])
        # Full-window slice lines up with the underlying window
        np.testing.assert_array_equal(w[:, :], underlying[100:300, :])

    def test_channel_subselect(self):
        underlying = np.arange(500 * 8, dtype=np.int16).reshape(500, 8)
        w = _WindowedSource(underlying, start_sample=100, end_sample=200)
        sub = w[:, slice(2, 5)]
        assert sub.shape == (100, 3)
        np.testing.assert_array_equal(sub, underlying[100:200, 2:5])

    def test_1d_slice_supported(self):
        underlying = np.arange(500 * 8, dtype=np.int16).reshape(500, 8)
        w = _WindowedSource(underlying, start_sample=100, end_sample=200)
        sub = w[10:20]
        np.testing.assert_array_equal(sub, underlying[110:120])

    def test_rejects_invalid_bounds(self):
        underlying = np.zeros((100, 4), dtype=np.int16)
        with pytest.raises(ValueError):
            _WindowedSource(underlying, start_sample=200, end_sample=300)
        with pytest.raises(ValueError):
            _WindowedSource(underlying, start_sample=50, end_sample=200)  # end past full length


class TestResolveWindow:
    def test_zero_offset_no_duration_is_full(self):
        start, n = _resolve_window(0.0, None, 30000.0, 1_000_000)
        assert start == 0
        assert n is None

    def test_offset_only(self):
        start, n = _resolve_window(1.0, None, 30000.0, 1_000_000)
        assert start == 30_000
        assert n is None

    def test_duration_clamped_to_end(self):
        # Start at 30s (sample 900k) with 5s asked; only 100k samples remain.
        start, n = _resolve_window(30.0, 5.0, 30_000.0, 1_000_000)
        assert start == 900_000
        assert n == 100_000

    def test_duration_normal(self):
        start, n = _resolve_window(10.0, 2.0, 30_000.0, 1_000_000)
        assert start == 300_000
        assert n == 60_000

    def test_offset_past_end_raises(self):
        with pytest.raises(ValueError):
            _resolve_window(100.0, None, 30_000.0, 1_000_000)  # 100 s × 30k = 3M > 1M
