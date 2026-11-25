# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## v0.2.1 (2026-11-25)

### Fix

- type-check against current numpy, zarr, numba, tyro and panel

## v0.2.0 (2026-11-25)

### BREAKING CHANGE

- add_pyramid, ensure_pyramid and the CLI no longer default
units to "uV"; float sources without gain_to_uV need explicit units.
- QuantizationConfig, BuildConfig.lsb_shift_raw,
lsb_shift_pyramid, noise_floor and clip_outliers are removed in favour
of EncodingConfig, and the CLI quantization flags are renamed.

### Feat

- **web**: let a host limit which event groups draw
- record how many values clipping changed in each clipped array
- store per-channel display statistics in mipmap stores
- shard event layers so a view reads small chunks
- **web**: prioritise visible tiles and prefetch around the settled view
- serve store files with validators so reloads revalidate
- serve the web viewer from ecephys-mipmap view
- **web**: add a spike raster, a time axis and event markers
- store per-group event counts in event layers
- **web**: colour units apart by footprint overlap, with selection and colour overrides
- **web**: add controls, noise-gated outlines, themes and spike colouring
- add an event-layer format with writers from sortings and detection tables
- **web**: add the standalone page
- **web**: add the mipmap-viewer element
- **web**: keep a view and redraw on demand
- **web**: draw envelopes and raw traces
- **web**: upload tiles as integer textures
- **web**: fetch tiles in a worker behind a byte-bounded cache
- **web**: plan which blocks a view needs
- **web**: open a store and derive each array's decode
- record a format version and derive the units values decode to
- carry the source's channel properties, probe and SI chain into stores
- choose shift, range fit or none per array with EncodingConfig
- optionally swap the chain's IIR highpass and bandpass for SpikeInterface's FIR
- read raw chunks ahead of the reader in the decoded chunk cache
- cache decoded raw chunks across neighbouring reads
- fuse the AIND chain into one FIR pass and split long reads
- defer phase-shift and CMR patches to SpikeInterface's fast paths
- parallel sosfiltfilt and CMR median monkey-patches
- int16-native fast path for SI PhaseShift via get_traces override
- 32-tap sinc-FIR replacement for SI's FFT phase-shift
- retry source reads on OSError (EIO) with exponential backoff
- --fft-workers parallelizes phase-shift FFT across cores
- auto-default prefetch_chunks=3 for remote sources; dedupe stats
- --start-seconds / --duration-seconds work for all build inputs
- --remote viewer mode for cloud IDE / JupyterHub deployment
- --start-seconds / --duration-seconds for SI-based builds
- build mipmaps from a SortingAnalyzer via SpikeInterface
- viewer spike overlay, scale bar, and initial-view channels
- viewer 2D panning with debounced updates and cached channel_ids
- add --channel-order to CLI, default to depth sorting
- viewer performance overhaul and t_start support
- non-blocking viewer data fetches
- viewer shard prefetch for S3 stores
- auto S3 cache with cross-platform defaults and LRU pruning
- builder prefetch for slow sources (S3, WavPack)
- smart outlier clipping for LSB-shift int8 conversion
- add ReadPlan, I/O separation, dequantize=False, and cache_dir
- complete mipmap zarr library with builder, reader, CLI, and viewer

### Fix

- clear the lint and type errors CI reports
- **cli**: read a zarr source's sampling rate from its attrs
- **web**: draw traces unclipped by their slot
- **web**: keep each channel inside its slot when zoomed out
- measure an auto shift's noise before a low-pass filter
- refuse an encoded raw array in add_pyramid
- record the raw array's path in pyramid-only stores
- fuse bandpass chains and keep fast paths when fusing falls back
- set WavPack decode threads on the segment's own array handle
- reach the WavPack codec and read t_start without the time vector
- narrow retry-on-OSError to transient errnos; refuse install without numba
- copy contextvars per-submit in _chunk_iter
- wrap wildcard websocket origin in list for Bokeh
- read analyzer recording dict without zarr, defensive int coerce
- consistent ADC dequantization across raw and pyramid levels
- dynamic channel changes without figure rebuild
- use rounding instead of truncation for LSB shift
- stream int8 conversion and quantization to avoid OOM
- CLI error handling — use CliApp.run_subcommand for proper dispatch

### Refactor

- migrate CLI from pydantic-settings to tyro
- simplify _engine.py — merge converters, unify iterators
- use BuildConfig/QuantizationConfig in public API
- add BuildConfig and QuantizationConfig dataclasses
- extract _store.py and _engine.py from build.py

### Perf

- **web**: start the first view's fetch and the GPU before the store's statistics finish
- transpose SpikeInterface reads to channel-first inside the read workers
