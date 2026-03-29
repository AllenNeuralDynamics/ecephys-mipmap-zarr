# TODO

## ~~Quantization support~~ DONE

Implemented: `RawQuantization` (uint16) + `LevelQuantization` (uint8, shared across all pyramid levels). Per-channel offset/scale, auto-computed or user-provided. Bitshuffle for quantized arrays. Reader dequantizes transparently.

## ~~`ensure_pyramid` convenience function~~ DONE

Implemented: idempotent, checks n_samples + strides, silently rebuilds on stale.

## ~~SpikeInterface adapter~~ DONE

Handled by `aind-ecephys-s3-to-phy` package (separate env, reconstructs preprocessing chain, writes `.cbin`). Our CLI reads `.cbin` directly via mtscomp.

## ~~CI updates~~ DONE

Updated CI matrix to Python 3.11 and 3.13.

## ~~SpikeInterface metadata pass-through~~ DONE

Implemented: `add_pyramid` auto-copies `probe`, `annotations`, `channel_ids`, `properties/location` from SI zarrs. `MipmapStore` exposes `.channel_ids`, `.channel_locations`, `.source_metadata`. Viewer uses channel IDs for y-axis labels.

## ~~Spike overlay in viewer~~ DONE

Implemented `SpikeData` + `SpikeOverlay` in viewer.py:
- Zoomed out: colored boxes per unit (density-based opacity via `quad` glyph)
- Zoomed in: colored trace segments around each spike time (via `multi_line` glyph)
- Transition automatic with trace display mode
- Configurable: colors, window size, box height, sparse threshold

## ~~`--metadata-from` for binary builds~~ DONE

Implemented: `build_mipmap(metadata_from="s3://...")` and `ecephys-mipmap build --metadata-from ...` copy probe/annotations/channel_ids from a specified zarr into the output mipmap store.

## ~~Documentation~~ DONE

README updated with installation, quick start, CLI usage, zarr layout, and development instructions.

## LSB-shift compression for ADC data

Replace the current per-channel quantization approach with a simpler, more effective LSB-shift scheme for data already in ADC units (int16 from Neuropixels).

### Approach

Right-shift ADC values to drop noise LSBs, cast to int8, store with `lsb_shift` metadata. Reader reconstructs: `stored.astype(int16) << lsb_shift`. When ADC `gain_to_uV` is available, combine into a single scale: `effective_scale = gain_to_uV * (1 << lsb_shift)`, so the reader dequantizes directly to float32 µV.

### Parameters

```python
build_mipmap(...,
    lsb_shift_raw: int = 0,       # 0 = no shift (lossless), 2 = recommended for Neuropixels
    lsb_shift_pyramid: int = 0,   # 0 = no shift, 4 = recommended for display-quality pyramids
)
```

When `lsb_shift > 0`, auto-select dtype from the shifted data range:
- `(data >> shift).min() >= -128` and `(data >> shift).max() <= 127` → **int8** (half the uncompressed size, better for low-entropy data with noshuffle)
- Otherwise → **int16 with zeroed LSBs** (`(data >> shift) << shift`), still benefits from bitshuffle on the zero bit planes

Check is done on the first shard and applied uniformly. No user decision needed.

### Auto shuffle selection

The optimal blosc shuffle depends on how many unique values remain after shifting:
- ≤ ~64 unique values per channel (low entropy, ~6 bits): **noshuffle** — zstd's byte-level LZ matching is more efficient
- \> ~64 unique values (higher entropy): **bitshuffle** — bit-plane decomposition helps

The `"blosc"` preset in `_resolve_compressor` should detect the stored dtype and use noshuffle for int8 when the data is low-entropy. Simplest heuristic: noshuffle for int8, bitshuffle for int16.

### Build flow changes

In `_write_raw_and_compute_finest`:
1. Read chunk from source (int16)
2. Compute min/max blocks (still in int16 for accuracy)
3. Right-shift raw chunk: `(chunk >> lsb_shift_raw).astype(int8)`
4. Write shifted raw to zarr (int8 array)
5. Right-shift min/max: `(mm >> lsb_shift_pyramid).astype(int8)`
6. Write shifted pyramid to zarr (int8 array)

Store metadata in array attrs:
```json
{"lsb_shift": 2, "original_dtype": "int16"}
```

When `metadata_from` provides `gain_to_uV` and `offset_to_uV`, store combined `RawQuantization`:
```python
RawQuantization(
    offset=offset_to_uV,                              # per-channel
    scale=(gain_to_uV * (1 << lsb_shift)).tolist(),   # per-channel, combined
    source_dtype="float32",                            # output is µV
    quantized_dtype="int8",
)
```

### Reader changes

In `_read_raw` and `_read_minmax`:
- If `RawQuantization` present: dequantize to float32 µV (already implemented)
- If only `lsb_shift` present (no gain/offset): `data.astype(int16) << lsb_shift` to reconstruct ADC values

### Auto-detection of optimal shift

The optimal shift depends on the device's noise floor, which varies by probe type and isn't always known in advance. Auto-compute from the data:

```python
def compute_optimal_lsb_shift(data, fs=30000, safety_factor=2.0):
    """Estimate noise floor from HF spectrum, compute max safe shift."""
    # 1. Estimate noise from 10-14 kHz band for each channel
    per_channel_noise = estimate_hf_noise_per_channel(data, fs)

    # 2. Use MEDIAN across channels (robust to dead/broken channels)
    noise_rms = np.median(per_channel_noise)

    # 3. Largest shift where quant_noise < noise / safety_factor
    max_step = noise_rms * sqrt(12) / safety_factor
    shift = max(0, int(floor(log2(max_step))))

    # 4. Verify shifted values fit in int8
    if (data.max() - data.min()) >> shift > 255:
        shift = max(0, shift - 1)

    return shift
```

**Use median, not minimum.** Dead channels (near-zero noise) or broken channels (anomalous noise) would skew min/max. Median is robust to both. For preprocessed data, bad channels are already removed by SI. For raw data, the median handles remaining outliers.

**Caveat: HF noise is a conservative lower bound.** Real recording noise is not white — 1/f (flicker) noise makes low-frequency noise higher than HF noise. The 10-14 kHz band measures only thermal + quantization noise. This means:
- Auto-computed shift is *conservative* (safe) — real noise is ≥ HF estimate, so we keep bits that are likely already in noise
- More accurate for high-pass filtered data (1/f removed) than raw
- User override is essential for edge cases or when the noise floor is known

**Device variation:** NP 2.0 (12-bit ADC) gives shift=2, but other probes (NP 1.0, Cambridge NeuroTech, etc.) have different ADC depths and noise characteristics. The auto-detection handles this without device-specific hardcoding.

### CLI flags

```bash
# Auto-detect optimal shift from data
ecephys-mipmap build --input recording.cbin --output mipmap.zarr --lossy-compress auto

# Manual override (when you know your system)
ecephys-mipmap build --input recording.cbin --output mipmap.zarr \
    --lsb-shift-raw 2 --lsb-shift-pyramid 4

# Explicit noise floor override (ADC units RMS)
ecephys-mipmap build --input recording.cbin --output mipmap.zarr \
    --lossy-compress auto --noise-floor 4.0
```

### Benchmarked results (Neuropixels 2.0, 12-bit ADC, noise floor ~4 ADC RMS)

| Array | Shift | Step | int8 compressed | Ratio vs int16 | Precision |
|---|---|---|---|---|---|
| Filtered raw | >>2 | 4 ADC | 14.5 MB/100K×373 | **5.14x** | quant noise 3.5x below analog noise |
| Raw (unfiltered) | >>2 | 4 ADC | 19.1 MB/100K×384 | **4.02x** | same |
| Pyramid (display) | >>4 | 16 ADC | 2.4 MB/31K×2×373 | **~19x** | max err 35 µV, invisible at screen resolution |

### Implementation tasks

- [x] Add `lsb_shift_raw` and `lsb_shift_pyramid` params to `build_mipmap`, `add_pyramid`, `ensure_pyramid`
- [x] Modify `_write_raw_and_compute_finest` to right-shift + cast to int8 before writing
- [x] Modify `_build_finest_from_raw` similarly for pyramid building (via `add_pyramid`)
- [x] Create int8 zarr arrays when shift is active (change dtype in `_create_zarr_array` calls)
- [x] Store `lsb_shift` in array attrs
- [x] Combine `lsb_shift` with ADC `gain_to_uV` into `RawQuantization.scale` when `metadata_from` is provided
- [x] Update reader to handle `lsb_shift` attrs (reconstruct via `<< shift` if no `RawQuantization`)
- [x] Entropy-based shuffle auto-selection in `_resolve_compressor` (<6 bits → noshuffle, ≥6 → bitshuffle, int16 pyramids → byte-shuffle)
- [x] Add `--lsb-shift-raw`, `--lsb-shift-pyramid`, `--noise-floor` to CLI
- [x] Implement `estimate_hf_noise(data, fs)` — PSD of 10-14 kHz band per channel, return median
- [x] Implement `compute_optimal_lsb_shift(data, fs, safety_factor)` — from median noise floor to shift
- [x] Use 30K samples for noise estimate (~1 shard)
- [x] Auto-detection from first shard, applied to all
- [x] Manual override via `--lsb-shift-raw N` or `--noise-floor X` takes precedence
- [x] Pyramid shift = raw shift + 2 in auto mode
- [x] Tests with synthetic data (explicit shift values, round-trip verification)
- [x] Update CLAUDE.md defaults and caveats

## ~~(C, T) channel-first layout~~ DONE

Switched from `(T, C)` to `(C, T)` for raw and `(B, 2, C)` to `(C, 2, B)` for minmax. Native C-order zarr, no TransposeCodec needed. Builder transposes source (T, C) data on write. Reader detects SI (T, C) zarrs via raw_source and transposes small windows on read.

Benchmarked: 4% compression improvement on filtered, 9% on raw. 47% faster writes.

## Raw (non-dequantized) read mode

Currently `get_window` always dequantizes to float32 when `RawQuantization` is present. Add a way to read the stored values directly without dequantization — useful for caching (avoid repeated float conversion), custom processing, or when the caller wants ADC units.

Options:
- `get_window(..., dequantize=False)` — returns stored dtype (int8/int16/uint16) instead of float32
- `MipmapStore.get_raw_window(t0, t1, channels)` — separate method, always returns stored values
- Both the raw array and pyramid arrays should support this

### Implementation tasks
- [ ] Add `dequantize: bool = True` parameter to `get_window`
- [ ] When `dequantize=False`, skip the `dequantize_array` call in `_read_raw` and `_read_minmax`
- [ ] `WindowResult` should indicate whether data was dequantized (add `dequantized: bool` field?)
- [ ] Document that stored dtype may differ from `MipmapAttrs.dtype` when quantization is active

## Interactive unit selection widget

Currently `SpikeData` is passed at construction and is static. For interactive spike curation, add a widget-driven unit selector to the viewer.

### Design

- Pass all available spike data at construction (all units)
- Add a `pn.widgets.MultiChoice` or `pn.widgets.CheckBoxGroup` listing unit IDs
- User toggles units on/off in the browser — callback filters `SpikeOverlay` to show only selected units
- Unit list shows: unit ID, channel, spike count, color swatch
- Optionally sortable by channel or spike count

### Usage

```python
from ecephys_mipmap_zarr.viewer import view, SpikeData

# Load all units
all_spikes = SpikeData(
    spike_times={"unit_0": t0, "unit_1": t1, "unit_42": t42, ...},
    unit_channels={"unit_0": 5, "unit_1": 12, "unit_42": 15, ...},
)
view("mipmap.zarr", spike_data=all_spikes)
# Browser shows unit selector widget — toggle units on/off
```

### Implementation tasks

- [ ] Add `visible_units: list[str]` state to `SpikeOverlay` (filters which units render)
- [ ] Add `pn.widgets.MultiChoice` to `TraceViewer.panel()` controls
- [ ] Wire widget change callback to `SpikeOverlay.set_visible_units()` → re-render
- [ ] Show unit metadata (channel, count, color) in the widget labels
- [ ] Programmatic API: `viewer.add_spikes()` / `viewer.remove_spikes()` for scripting (uses `pn.state.execute` for thread safety)

### Future

- Cluster quality metrics alongside unit list (like phy's cluster view)
- Click a unit to scroll to its peak channel
- Keyboard shortcuts for unit navigation

## Multi-channel spike display

Currently each unit maps to a single channel (`unit_channels: dict[str, int]`). At zoomed-in levels, spikes are visible across multiple neighboring channels. Add optional multi-channel spread for trace coloring.

### Design

- Add `unit_channel_spread: dict[str, list[int]] | None` to `SpikeData`
- **Boxes** (zoomed out): always use peak channel only — one box per unit per time bin
- **Trace coloring** (zoomed in): if `unit_channel_spread` provided, color the segment on all spread channels; otherwise just the peak channel
- Auto-derivable from SI SortingAnalyzer templates (channels where template amplitude > threshold)

### Implementation tasks

- [ ] Add `unit_channel_spread` field to `SpikeData`
- [ ] Update `SpikeOverlay._update_segments` to iterate over spread channels
- [ ] Helper to compute spread from spike templates (optional, for SI integration)

## Viewer backend refactor

Currently data computation and Bokeh rendering are interleaved in `viewer.py`. Split into:

- **`viewer_core.py`** — pure numpy: channel spacing, offset computation, spike box/segment computation, data dict preparation. No Bokeh imports. Reusable across backends.
- **`viewer_bokeh.py`** (or keep as `viewer.py`) — Bokeh/Panel-specific: figure creation, ColumnDataSource management, glyph lifecycle, widget wiring.

This enables a future `viewer_pyqtgraph.py` (~80-100 lines) for desktop use with no kernel roundtrip. Also makes notebook embedding viable since the Panel layout can be displayed inline (slower but functional).

### Implementation tasks

- [ ] Extract pure computation into `viewer_core.py`: `compute_channel_spacing`, `compute_channel_offsets`, `compute_spike_boxes`, `compute_spike_segments`, `compute_envelope_data`, `compute_raw_data`
- [ ] `viewer.py` imports from `viewer_core.py` and handles only Bokeh rendering
- [ ] No new functionality — just a clean separation

## ~~Progress indication for builds~~ DONE

Implemented: `ProgressCallback` type alias, `_progress_bar()` context manager with rich Progress. Both `build` and `add-pyramid` CLI commands use it. Callback wired through `_write_raw_and_compute_finest` and `_build_finest_from_raw`.

## ~~CLI error handling~~ DONE

Implemented: `main()` wraps `CliApp.run(CLI)` in try/except with `console.print_exception()` and `SystemExit(1)`.

## Overview minimap

A small Bokeh figure above the main viewer showing the full recording duration with a highlighted rectangle for the current viewport. Bokeh's `RangeTool` links the minimap selection to the main figure's x_range.

- [ ] Add minimap figure to `TraceViewer.panel()` layout
- [ ] Use `RangeTool` to link minimap selection → main x_range
- [ ] Show a coarse envelope of the full recording in the minimap (coarsest pyramid level)

## Export current view

Save the current viewport as PNG or SVG.

- [ ] Add export button to Panel controls
- [ ] Use `bokeh.io.export_png` or `export_svgs` (requires selenium or playwright)
- [ ] Or simpler: Bokeh's built-in save tool (already available in toolbar)

## Viewer prefetching/caching

Cache compressed zarr shards for the viewer to reduce latency when panning. The viewer reads small windows frequently — caching previously fetched shards (especially for S3 stores) avoids redundant network requests.

### Design

- Cache at the zarr shard level (compressed bytes), not decoded arrays — saves memory and avoids redundant decompression overhead
- LRU eviction policy, configurable max cache size (default: 256 MB?)
- zarr v3 may support custom store wrappers or caching stores — investigate `zarr.storage.LRUStoreCache` or fsspec caching
- Prefetch adjacent shards when user is panning (predict direction from x_range change)

### Implementation tasks

- [ ] Investigate zarr v3 caching store options (LRUStoreCache, fsspec cache)
- [ ] Implement or configure shard-level caching for MipmapStore
- [ ] Add prefetch logic to viewer: predict pan direction, preload adjacent shards
- [ ] Configurable cache size (CLI flag, viewer param)

## Async/threaded reader

The reader currently blocks on zarr I/O. For the viewer and server-side use, async or threaded reads would prevent UI freezes and enable concurrent requests.

### Design

- zarr v3 has native async support (`zarr.api.asynchronous`)
- Option 1: `async get_window()` using zarr's async API
- Option 2: Thread pool executor wrapping synchronous reads (simpler, works with existing code)
- Viewer could use `pn.state.execute` or `asyncio.to_thread` to avoid blocking the event loop

### Implementation tasks

- [ ] Evaluate zarr v3 async API for reading (array slicing, group access)
- [ ] Add `async get_window()` to MipmapStore or a separate AsyncMipmapStore
- [ ] Wire async reads into viewer update callback
- [ ] Consider thread-safe access patterns for shared MipmapStore instances

## Evaluate codebase structure

Review the current module layout and determine if the codebase would benefit from reorganization as it grows. Consider:

- [ ] Whether `build.py` should be split (e.g. `build_core.py` + `build_lsb.py` or `build_quantize.py`)
- [ ] Whether `viewer.py` split into `viewer_core.py` + `viewer_bokeh.py` is still the right plan
- [ ] Whether metadata/schema types should move to a separate `types.py` or `schema.py`
- [ ] Whether CLI should be split per subcommand
- [ ] Overall import graph and circular dependency risks
