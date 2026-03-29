# Ecephys Mipmap Zarr

![CI](https://github.com/AllenNeuralDynamics/ecephys-mipmap-zarr/actions/workflows/ci-call.yml/badge.svg)
[![PyPI - Version](https://img.shields.io/pypi/v/ecephys-mipmap-zarr)](https://pypi.org/project/ecephys-mipmap-zarr/)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)
[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
[![ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

Build and read multi-resolution (mipmap) Zarr v3 files for multichannel extracellular electrophysiology data. Pyramids store (min, max) pairs for efficient waveform envelope rendering at any zoom level.

## Installation

```bash
pip install ecephys-mipmap-zarr
```

Optional extras:

```bash
pip install ecephys-mipmap-zarr[cli]       # CLI (pydantic-settings + rich)
pip install ecephys-mipmap-zarr[viewer]    # Interactive trace viewer (Panel + Bokeh)
pip install ecephys-mipmap-zarr[cbin]      # mtscomp .cbin support
pip install ecephys-mipmap-zarr[s3]        # S3 remote zarr access
pip install ecephys-mipmap-zarr[wavpack]   # WavPack codec for AIND zarrs
```

## Quick start

### Build a mipmap from a numpy array

```python
import numpy as np
from ecephys_mipmap_zarr import build_mipmap

# Simulated 30s recording, 384 channels, 30kHz
rng = np.random.default_rng(42)
data = rng.integers(-1000, 1000, size=(900_000, 384), dtype=np.int16)

build_mipmap(data, "mipmap.zarr", sampling_frequency=30000.0, units="uV")
```

### Read with automatic level selection

```python
from ecephys_mipmap_zarr import open_mipmap

store = open_mipmap("mipmap.zarr")

# Zoomed out: returns min/max envelope from pyramid
result = store.get_window(0, 900_000, num_points=1920)
print(result.stride)  # e.g. 32 — using s32 pyramid level
print(result.data.shape)  # (n_blocks, 2, 384) — axis 1 is [min, max]

# Zoomed in: returns raw samples
result = store.get_window(0, 1000, num_points=1000)
print(result.is_raw)  # True
print(result.data.shape)  # (1000, 384)
```

### Build pyramids for an existing zarr on S3

```python
from ecephys_mipmap_zarr import add_pyramid

# Builds local pyramid store pointing back to S3 for raw data
add_pyramid(
    "s3://aind-open-data/.../experiment1_...zarr",
    "local_pyramid.zarr",
)
# sampling_frequency auto-detected from SpikeInterface zarr attrs
```

### Interactive viewer

```python
from ecephys_mipmap_zarr.viewer import view

view("mipmap.zarr")  # opens browser tab with Pan/Zoom trace viewer
```

With spike overlay:

```python
from ecephys_mipmap_zarr.viewer import view, SpikeData

spikes = SpikeData(
    spike_times={"unit_0": spike_times_array},  # seconds
    unit_channels={"unit_0": 5},                # channel index
)
view("mipmap.zarr", spike_data=spikes)
```

## CLI

```bash
# Build mipmap from a .cbin file (auto-detects sampling rate from params.py)
ecephys-mipmap build --input recording.cbin --output mipmap.zarr

# Build from flat binary
ecephys-mipmap build --input recording.dat --output mipmap.zarr \
    --sampling-frequency 30000 --n-samples 900000 --n-channels 384

# Build with quantization (float32 → uint16 raw + uint8 pyramids)
ecephys-mipmap build --input recording.dat --output mipmap.zarr \
    --sampling-frequency 30000 --n-samples 900000 --n-channels 384 \
    --dtype float32 --quantize

# Copy probe metadata from an S3 zarr
ecephys-mipmap build --input recording.cbin --output mipmap.zarr \
    --metadata-from "s3://aind-open-data/.../experiment1_...zarr"

# Add pyramids to an existing S3 zarr (no --sampling-frequency needed for SI zarrs)
ecephys-mipmap add-pyramid \
    --raw-store "s3://aind-open-data/.../experiment1_...zarr" \
    --output local_pyramid.zarr

# Inspect a mipmap store
ecephys-mipmap info mipmap.zarr

# Launch the viewer
ecephys-mipmap view --store-path mipmap.zarr
```

## Zarr layout

```
/raw                    # shape (C, T), channel-first, int16 or float32
/minmax/s32             # shape (C, 2, ceil(T/32))  — finest pyramid
/minmax/s256            # shape (C, 2, ceil(T/256))
/minmax/s2048           # ...
/minmax/s16384          # ...
/minmax/s131072         # coarsest level
/channel_ids            # (C,) string array (if copied from source)
/properties/location    # (C, 2) float array (if copied from source)
```

- Pyramid levels use base stride 32, factor 8x per level
- Levels added until the coarsest has < 1024 blocks
- Zarr v3 with sharding for S3-friendly access
- Optional quantization: raw → uint16, pyramids → uint8 with per-channel offset/scale
- Blosc(zstd) compression by default, configurable via `compressor` parameter

## Development

```bash
uv sync
./scripts/run_linters_and_checks.sh -c
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
