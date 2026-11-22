# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Library to build and read multi-resolution (mipmap) Zarr v3 files for multichannel extracellular electrophysiology data. Pyramids store (min, max) pairs for efficient waveform envelope rendering at any zoom level. Allen Institute for Neural Dynamics.

Core dependencies: zarr, numpy, dask.

## Zarr Layout

```
/raw                          # shape (C, T), C-order, channel-first
/minmax/s32                   # shape (C, 2, ceil(T/32))
/minmax/s256                  # shape (C, 2, ceil(T/256))
/minmax/s2048                 # ...
/minmax/s16384                # ...
/minmax/s131072               # coarsest level (if needed)
```

- **Raw array**: `(channels, samples)` — channel-first, one channel's time series is contiguous in memory. 4-9% better compression than (T,C), faster writes.
- **Pyramid arrays**: `(channels, 2, n_blocks)` — axis 1 is `[min, max]`, time-like axis last, dtype defaults to raw dtype
- **Source data is (T, C)**: builder transposes internally. SI zarrs and `.cbin` files are (T, C) — the transpose happens at write time.
- **Root attrs** (`MipmapAttrs`): sampling_frequency, units, strides, n_samples, n_channels, dtype, raw_source, raw_path, source_metadata, t_start, format_version. `units` is what values decode to: "uV" with an ADC gain, "counts" for integer sources without one.

### SpikeInterface zarr v2 layout (source data)

AIND raw recordings are stored as SI zarr v2 stores (not v3). The builder reads from these via `metadata_from`. Structure:

```
/                              # root group (zarr v2, use fsspec.get_mapper)
  attrs: sampling_frequency, num_segments, probe, annotations, provenance
/traces_seg0                   # shape (T, C), int16 — raw ADC values
/times_seg0                    # shape (T,), float64 — per-sample timestamps (seconds, acquisition clock)
/channel_ids                   # shape (C,), string — channel ID labels
/properties/
  gain_to_uV                   # shape (C,), float64 — ADC-to-µV gain per channel
  offset_to_uV                 # shape (C,), float64 — ADC-to-µV offset per channel
  location                     # shape (C, 2), float64 — probe (x, y) positions
```

Key details:
- `times_seg0[0]` gives `t_start` (recording start in acquisition clock seconds)
- `gain_to_uV` / `offset_to_uV` → stored as `RawQuantization` in our mipmap store
- `traces_seg0` is (T, C) — our builder transposes to (C, T)
- Open with `zarr.open_group(fsspec.get_mapper("s3://..."), mode="r")` (zarr v2 API)

### Dtype and quantization

- Raw dtype: int16 (typical for acquisition hardware) or float32 (filtered/processed data)
- Pyramid dtype: defaults to raw dtype (lossless — min/max of intN is intN)
- Optional per-level quantization (`LevelQuantization` in level attrs): separate offsets for min/max axes (they cluster in different ranges) with shared scale. Allows e.g. float32 raw → int16 or int8 pyramids for smaller transfer size. Dequantize: `value = quantized * scale + offset`
- **Encoding per array** (`EncodingConfig`, CLI `--raw-encoding` / `--pyramid-encoding`): raw and pyramids each pick `shift`, `range` or `none`, in any combination. Default: `shift` for integer sources, `range` for float. A setting the chosen encodings don't use is an error, not ignored.
- **Shift** (`shift`): right-shifts integer values to drop noise LSBs, stores as int8 when the range fits (`clip_outliers` clips under 0.01% to make it fit). `lsb_shift` stored in array attrs. Reader reconstructs: `stored.astype(int16) << lsb_shift`. `"auto"` picks the shift from the noise floor (10-14 kHz band, median across channels; refuses below 28 kHz sampling, or on a low-passed source with under 5% of its 300-6000 Hz power density left there, unless given `noise_floor`). For a SpikeInterface chain, `preprocess.chain_noise_floor` gives the instrument noise in the chain's output: measured at 10-14 kHz on the unfiltered root, times the chain's RMS gain for white noise (the chain rebuilt on a white-noise root). The CLI's `--from-analyzer` and the capsule use it, since a bandpass both empties the band and lowers the noise, and spike-band MAD would count neural background as noise.. Pyramids get +2 extra shift for display quality, independent of raw's setting.
- **Range fit** (`range`): per-channel offset/scale onto uint16 raw / uint8 pyramids, from the build's extremes or user-provided `RawQuantization` / `LevelQuantization`, stored in array attrs. Pyramid parameters are shared across levels (computed from s32, cascaded on codes — order-preserving).
- **ADC gain/offset**: when `metadata_from` / `source_metadata` carries `gain_to_uV`/`offset_to_uV`, it is composed with each array's encoding into one `quantization` attr (shift: scale `gain * 2**lsb_shift`; range fit: codes → source units → µV), so the reader decodes straight to float32 µV.

### Bit depth analysis (Neuropixels 2.0, 12-bit ADC)

Benchmarked on real Neuropixels 2.0 data (12-bit ADC). Analysis is in ADC counts; the µV figures below are examples at 2.925 µV/count, and the actual gain is per recording, read from `gain_to_uV`. Two datasets: preprocessed (HP 300Hz + CMR, 373 channels, `.cbin`) and raw unfiltered (384 channels, S3 WavPack zarr).

**Noise analysis:**
- **Analog noise floor**: ~4 ADC units RMS (measured from 10-14 kHz spectrum where spike power is negligible — SUA templates have >99% energy below 5 kHz, validated against 105 decoder-labeled SUA units)
- **Quantization noise at 1 LSB**: 0.29 ADC RMS → **14x below analog noise** (massive overkill)
- **Shannon entropy**: filtered ~5.3 bits/sample, raw ~7.0 bits/sample
- **int16 wastes**: ~11 bits (filtered) or ~9 bits (raw) per sample

**Optimal storage: drop 2 LSBs, store as int8**

Right-shift by 2 (`sample >> 2`), cast to int8. Step=4 ADC, quant noise=1.15 ADC, **3.5x below analog noise** (engineering standard: ≥2x). Max error = 2 ADC (half a step, with rounding) ≈ 6 µV, below 1σ of noise. Reconstruction: `stored.astype(int16) << 2`.

| Data | After >>2 | Unique vals/ch | Best codec | Compressed | Ratio vs int16 |
|---|---|---|---|---|---|
| Filtered | [-58, 26] in int8 | ~21 | noshuffle | 14.5 MB | **5.14x** |
| Raw | [-118, 55] in int8 | ~151 | bitshuffle | 19.1 MB | **4.02x** |

Compared to original int16+bitshuffle (current default): **+83% for filtered, +51% for raw**.

No per-channel offset/scale needed — filtered data is already centered near zero (high-pass), raw fits in int8 after >>2.

**Why noshuffle wins for filtered but bitshuffle wins for raw:**
- Filtered int8 has ~21 unique byte values per channel. In F-order, zstd's LZ matching finds long repeated byte patterns trivially. Bitshuffle breaks these patterns into bit planes, hurting net compression.
- Raw int8 has ~151 unique byte values. Too many for efficient byte-level LZ matching. Bitshuffle separates structured upper bit planes (slow-changing) from noisy lower bits, helping zstd.
- Crossover at ~6 bits Shannon entropy per sample.

**Per-channel uint8 quantization is counterproductive** for data already in ADC units — it remaps a small range (83 ADC) to fill 0-255, creating artificial precision below the LSB and destroying the near-zero bit structure that compressors exploit. Just drop LSBs from the original ADC values instead.

**Pyramid compression (s32 level, filtered, 31250 blocks × 2 × 373 ch):**

Pyramid range: [-314, 209], entropy 5.54 bits/sample. Note: >>1 doesn't fit int8 (min -157 < -128), >>2 minimum.

| Purpose | Shift | Step | int8 compressed | int16 compressed |
|---|---|---|---|---|
| Analysis quality | >>2 | 4 ADC | **8.1 MB** (noshuffle) | 8.4 MB (bitshuffle) |
| Display quality | >>4 | 16 ADC | **2.4 MB** (bitshuffle) | 2.5 MB (bitshuffle) |

Display quality (>>4, step=16, max error=8 ADC ≈ 23 µV) is appropriate for pyramids — they exist solely for envelope rendering at screen resolution where sub-pixel differences are invisible.

**Unified approach: `lsb_shift` parameter.** Right-shift original ADC values, cast to int8, store with `lsb_shift` metadata. Reader reconstructs: `stored.astype(int16) << lsb_shift`. No per-channel offset/scale needed.

| Array | Recommended shift | Step | Compressed | Ratio vs int16 original |
|---|---|---|---|---|
| Raw data | >>2 | 4 ADC | 14.5 MB / 100K×373 | **5.14x** |
| Pyramids | >>4 | 16 ADC | 2.4 MB / 31K×2×373 | **~19x** |

### Pyramid rules

- Base stride S0 = 32 samples, factor = 8× per level
- Strides: 32, 256, 2048, 16384, 131072, ...
- **Stop condition**: keep adding levels until the next level would produce fewer than ~1024 blocks. Goal is the coarsest level fits on one screen (~1K–2K points per channel).
- Raw is the base — no stride-1 min/max level

### Chunking and Sharding

Uses Zarr v3 sharding codec for S3-friendly access (fewer objects, range requests within shards).

Grouped in `ChunkConfig` dataclass (`DEFAULT_CHUNK_CONFIG` provides defaults). Use `config.for_channels(n)` to clamp channel dims when actual channel count < 64.

| Array | Chunk shape | Shard shape | Shard size (uncompressed) |
|-------|------------|-------------|---------------------------|
| raw `(C, T)` | `(64, 16384)` | `(64, 1048576)` | ~128 MiB (64 time-chunks/shard) |
| minmax `(C, 2, B)` | `(64, 2, 4096)` | `(64, 2, 32768)` | ~8 MiB (8 block-chunks/shard) |

- Channel chunk width 64 (not 1-wide; batch channel reads)
- Consolidated metadata enabled by default (`consolidate_metadata=True` in builder). Reader auto-detects.

### Codec

- Default: `blosc(zstd, clevel=1)` with **data-dependent shuffle selection**:
  - **int16 pyramids (C,2,B)**: byte-shuffle (temporal correlation on last axis)
  - **Low-entropy data** (<6 bits Shannon entropy): noshuffle (zstd byte-level LZ is more efficient)
  - **Higher-entropy data** (≥6 bits): bitshuffle (bit-plane decomposition helps)
- Entropy measured from first shard's shifted data (median per-channel Shannon entropy)
- Generalized `compressor` parameter: accepts preset strings (`"blosc"`, `"gzip"`, `"zstd"`) or zarr `BytesBytesCodec` objects for full control

### Compression benchmarks (Neuropixels 2.0, int16, 24 cores)

All benchmarks on real Neuropixels 2.0 data (12-bit ADC, 30kHz). Tested on both preprocessed (high-pass 300Hz + common reference, 373 channels from `.cbin`) and raw unfiltered (384 channels from S3 WavPack zarr).

**Best configuration: `blosc(zstd, clevel=1)` with (C,T) layout and LSB-shift to int8.**

With LSB shift enabled (default), compression ratios improve dramatically over raw int16:

| Array | Shift | Codec | Ratio vs int16 |
|---|---|---|---|
| Filtered int8 | >>2 | noshuffle | **5.14x** |
| Raw int8 | >>2 | bitshuffle | **4.02x** |
| Pyramid int8 | >>4 | bitshuffle | **~19x** |

Without LSB shift (int16), baseline benchmarks on (C,T) layout:

| Config | Filtered | Raw |
|---|---|---|
| zstd/bitshuffle/L1 | 2.82x | 2.66x |
| zstd/shuffle/L1 | 2.46x | 2.13x |
| lz4/bitshuffle/L1 | 2.25x | 2.30x |

**Key findings:**
- **(C,T) channel-first layout**: 4-9% better compression than (T,C), 47% faster writes. Channel's time series is contiguous.
- **int16 compresses 17% better than uint16** with bitshuffle (near-zero values have more bit redundancy)
- **Delta encoding never helps** — bitshuffle already captures the temporal redundancy
- **Shuffle selection is data-dependent**: low entropy → noshuffle, high entropy → bitshuffle, int16 pyramids → byte-shuffle

## Phy/Kilosort spike data

The viewer loads spike data from phy/Kilosort output directories via `load_phy_spikes()` and `--spikes-from` CLI flag. Key files:

- `spike_times.npy` — `(N,)` int64, spike times in samples
- `spike_clusters.npy` — `(N,)` int64, cluster ID per spike
- `channel_map.npy` — `(C,)` int32, maps template channel indices to probe channel indices
- `templates.npy` — `(n_templates, n_time, n_active_channels)` float64, mean waveform templates
- `template_ind.npy` — `(n_templates, n_active_channels)` int32, which channel_map indices are active per template (-1 = padding)

### Unit quality labels (AIND sorting pipeline)

Three quality metrics determine a "good" unit:

- **`cluster_KSLabel.tsv`**: Kilosort's own label — `"good"` or `"mua"` (multi-unit activity)
- **`cluster_decoder_label.tsv`**: AIND decoder classification — `"sua"` (single-unit activity), `"mua"`, or `"noise"`. Uses a trained classifier on waveform features.
- **`cluster_default_qc.tsv`**: Default quality control — `"True"` or `"False"`. Composite metric combining ISI violations, presence ratio, amplitude cutoff, etc.

A high-confidence single unit has: KS=good, decoder=sua, default_qc=True.

## Key API

`get_window(t0, t1, channels, num_points)` — the core read/tile function:
- Estimates samples-per-point from the time range and num_points
- Picks the coarsest pyramid level that still oversamples (at least 2 blocks per point)
- Falls back to raw data when zoomed in past the finest pyramid level
- Returns `WindowResult(data, stride, t0_actual, t1_actual)` — data is `(C, T)` for raw or `(C, 2, B)` for minmax

## Module Structure

```
src/ecephys_mipmap_zarr/
    metadata.py   # schema, constants, config dataclasses (BuildConfig, EncodingConfig), pure math
    _store.py     # shared store utilities: _open_store, _find_raw_array, metadata copying
    _engine.py    # write engine: minmax computation, cascade, zarr array factory, LSB shift
    build.py      # public API: build_mipmap, add_pyramid, ensure_pyramid (orchestration only)
    read.py       # get_window, MipmapStore, pyramid-level selection
    events.py     # event layers (sorted spikes, detections) for the web viewer: format, writer, converters
    cli.py        # CLI subcommands (build, add-pyramid, info, view, events-from-sorting, events-from-table)
    viewer.py     # Panel+Bokeh interactive trace viewer with spike overlay
```

## Commands

```bash
# Setup
uv sync

# Format
uv run ruff format

# Lint
uv run ruff check

# Type check (strict mode)
uv run mypy

# Tests with coverage
uv run pytest

# Run a single test
uv run pytest tests/test_example.py::test_version

# Docstring coverage
uv run interrogate -v

# Spell check
uv run codespell --check-filenames

# All checks at once (format + lint + mypy + interrogate + codespell + pytest)
./scripts/run_linters_and_checks.sh -c

# Pass extra args to pytest via the script
./scripts/run_linters_and_checks.sh -c -- -k test_version
```

## Code Standards

- **Build**: hatchling, src layout
- **Python**: >=3.11, CI tests 3.11 and 3.13
- **Typed**: `py.typed` marker, strict mypy on `src/`
- **Line length**: 120
- **Docstrings**: NumPy convention, 30% minimum coverage (interrogate)
- **Ruff rules**: Q, RUF100, C90, I, F, E, W, D, UP, ANN, PYI (ANN skipped in tests)
- **Complexity**: McCabe max 18
- **Versioning**: commitizen with `v$version` tags, semantic-release in CI
