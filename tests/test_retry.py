"""Tests for ``_retry_on_oserror`` — transient vs permanent OSError handling."""

from __future__ import annotations

import errno
import time

import numpy as np
import pytest

from ecephys_mipmap_zarr._engine import _TRANSIENT_ERRNOS, _retry_on_oserror


class _FlakeyReader:
    """Test double: raises *errors* (in order) before returning *result*."""

    def __init__(self, errors: list[OSError], result: np.ndarray | None = None) -> None:
        self.errors = list(errors)
        self.result = result if result is not None else np.zeros((4, 2), dtype=np.int16)
        self.call_count = 0

    def __call__(self, t0: int, t1: int) -> np.ndarray:
        self.call_count += 1
        if self.errors:
            raise self.errors.pop(0)
        return self.result


@pytest.fixture
def no_sleep(monkeypatch):
    """Skip actual sleep so the test runs in milliseconds."""
    monkeypatch.setattr(time, "sleep", lambda _s: None)


class TestRetryBehavior:
    def test_retries_zero_is_passthrough(self):
        reader = _FlakeyReader([OSError(errno.EIO, "transient")])
        wrapped = _retry_on_oserror(reader, retries=0)
        # With retries=0, the wrapper returns the original function unchanged —
        # the first call raises immediately.
        assert wrapped is reader
        with pytest.raises(OSError):
            wrapped(0, 100)

    def test_transient_eio_retries_and_succeeds(self, no_sleep):
        reader = _FlakeyReader([OSError(errno.EIO, "transient #1"), OSError(errno.EIO, "transient #2")])
        wrapped = _retry_on_oserror(reader, retries=3)
        result = wrapped(0, 100)
        assert reader.call_count == 3  # 2 failures + 1 success
        assert result.shape == (4, 2)

    def test_permanent_enoent_raises_immediately(self, no_sleep):
        """Missing-file error should NOT retry — fail fast."""
        reader = _FlakeyReader([OSError(errno.ENOENT, "no such file")])
        wrapped = _retry_on_oserror(reader, retries=5)
        with pytest.raises(OSError) as ei:
            wrapped(0, 100)
        assert ei.value.errno == errno.ENOENT
        assert reader.call_count == 1  # single attempt, no retries

    def test_permanent_eacces_raises_immediately(self, no_sleep):
        reader = _FlakeyReader([OSError(errno.EACCES, "permission denied")])
        wrapped = _retry_on_oserror(reader, retries=5)
        with pytest.raises(OSError) as ei:
            wrapped(0, 100)
        assert ei.value.errno == errno.EACCES
        assert reader.call_count == 1

    def test_exhausted_retries_raise_last_error(self, no_sleep):
        reader = _FlakeyReader([OSError(errno.EIO, f"failure #{i}") for i in range(10)])
        wrapped = _retry_on_oserror(reader, retries=2)
        with pytest.raises(OSError) as ei:
            wrapped(0, 100)
        # 1 initial attempt + 2 retries = 3 total calls, then raise.
        assert reader.call_count == 3
        assert ei.value.errno == errno.EIO

    def test_errno_none_does_not_retry(self, no_sleep):
        """OSError with errno=None (rare but possible) is treated as permanent."""
        err = OSError("generic OSError with no errno")
        assert err.errno is None
        reader = _FlakeyReader([err])
        wrapped = _retry_on_oserror(reader, retries=5)
        with pytest.raises(OSError):
            wrapped(0, 100)
        assert reader.call_count == 1

    def test_transient_errno_set_includes_expected(self):
        """Spot-check the transient set covers the errors we care about."""
        for code in (errno.EIO, errno.ETIMEDOUT, errno.ECONNRESET, errno.EAGAIN):
            assert code in _TRANSIENT_ERRNOS
        for code in (errno.ENOENT, errno.EACCES, errno.EINVAL, errno.EISDIR):
            assert code not in _TRANSIENT_ERRNOS
