"""Example test module."""

import ecephys_mipmap_zarr


def test_version():
    """Test that version is defined."""
    assert ecephys_mipmap_zarr.__version__ is not None
    assert isinstance(ecephys_mipmap_zarr.__version__, str)
