"""Tools to make mipmap zarrs for extracellular ephys."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ecephys-mipmap-zarr")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

from ecephys_mipmap_zarr.build import add_pyramid, build_mipmap, ensure_pyramid
from ecephys_mipmap_zarr.metadata import (
    DEFAULT_CHUNK_CONFIG,
    ChunkConfig,
    LevelQuantization,
    MipmapAttrs,
    RawQuantization,
    compute_quantization_params,
    compute_strides,
    dequantize_array,
    minmax_path,
    n_blocks,
    quantize_array,
    select_level,
)
from ecephys_mipmap_zarr.read import MipmapStore, WindowResult, open_mipmap

__all__ = [
    "__version__",
    "ChunkConfig",
    "DEFAULT_CHUNK_CONFIG",
    "LevelQuantization",
    "MipmapAttrs",
    "MipmapStore",
    "RawQuantization",
    "WindowResult",
    "add_pyramid",
    "build_mipmap",
    "compute_quantization_params",
    "compute_strides",
    "dequantize_array",
    "ensure_pyramid",
    "minmax_path",
    "n_blocks",
    "open_mipmap",
    "quantize_array",
    "select_level",
]
