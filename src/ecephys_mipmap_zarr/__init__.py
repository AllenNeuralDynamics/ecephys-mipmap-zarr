"""Tools to make mipmap zarrs for extracellular ephys"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ecephys-mipmap-zarr")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"
