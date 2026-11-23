# The Mipmap Store Format, Version 1

A mipmap store holds a multichannel recording at several resolutions: the samples themselves, and
per-block minimum and maximum envelopes for drawing long stretches at screen resolution. This document
is the contract between the **builder** (`ecephys_mipmap_zarr.build`, which writes stores) and every
**reader** (`ecephys_mipmap_zarr.read.MipmapStore`, the web viewer). A reader that follows it decodes
any version 1 store without knowing how it was built.

Indices here start at 0, as stored. Channel `c` is row `c` of every channel-indexed array.

MUST, MUST NOT, SHOULD and MAY are used in the RFC 2119 sense. A requirement on a store binds the builder
that wrote it.

## The version changes only when an older reader would misread a store

`format_version` is an integer. It increases when a reader of the previous version would decode a store
wrongly: an attribute or array removed, renamed or given a new meaning, or a new decoding rule. A
reader MUST refuse a store whose version is higher than its own.

An addition that an older reader can ignore, such as a new optional attribute or array, does not change
the version, so that readers already deployed keep opening new stores. A reader MUST ignore root
attributes, array attributes and arrays it does not know. **Revisions** at the end lists the additions
made within this version.

## The store is a Zarr v3 group

The root is a Zarr v3 group, usually with consolidated metadata in its `zarr.json`; a reader uses it when
present and reads each array's own `zarr.json` otherwise. Chunking, sharding and the compressor are the
builder's choice and outside this contract: a reader takes them from array metadata.

```
/                     root attributes, below
/raw                  (C, T)        samples; absent in a pyramid-only store
/minmax/s{stride}     (C, 2, B)     one array per stride in `strides`
/channel_ids          (C,)          string; optional
/properties/{name}    (C, ...)      per-channel values; optional
```

## Root attributes describe the recording

| Attribute | Type | Meaning |
|---|---|---|
| `format_version` | int | `1`. A store without it is version 1. A reader refuses a store with a higher version. |
| `sampling_frequency` | float | Samples per second. |
| `n_samples` | int | `T`, samples per channel. |
| `n_channels` | int | `C`. |
| `strides` | list of int | Pyramid strides, ascending; one `/minmax/s{stride}` array each. |
| `units` | string | What decoded values are in: `"uV"` when an ADC gain is composed into the encoding, `"counts"` for integer values without one, otherwise what the builder was told. |
| `dtype` | string | NumPy dtype of the source values before encoding. Descriptive; decoding does not use it. |
| `t_start` | float or null | Time of sample 0 in seconds on the acquisition clock; null means 0. Sample `i` is at `t_start + i / sampling_frequency`. |
| `raw_source` | string or null | Set only in a pyramid-only store: the URI of the store holding raw samples. |
| `raw_path` | string or null | The raw array's path inside `raw_source`; null means try `raw`, then `traces_seg0`. |
| `source_metadata` | object or null | Copied from the source, for interpretation only: `probe` (probeinterface `ProbeGroup.to_dict`), `annotations`, `si_chain` (SpikeInterface preprocessing steps, source first). |
| `channel_stats` | object | Optional. Per-channel display statistics; see below. |

## Channel statistics let a reader scale traces before reading samples

Without them a reader that centres and scales each channel must first read the whole coarsest level
and a raw chunk for every channel. `channel_stats` holds those measurements in decoded units, each list
indexed by channel:

| Member | Type | Meaning |
|---|---|---|
| `centre` | list of float | Value a channel's trace is centred on: the clipped mean over `samples`, else the median block midpoint `(min + max) / 2` of the coarsest level, else 0. |
| `noise_sd` | list of float | Clipped standard deviation over `samples`; 0 when the store has no raw samples. |
| `excursion` | list of float or null | The 0.95 quantile, over the coarsest level's blocks, of `max(max - centre, centre - min)`; null when no block is finite. |
| `samples` | [int, int] | Raw sample range `[start, stop)` measured: the raw chunk holding sample `n_samples // 2`. `[0, 0]` without raw samples. |

The clipped mean and standard deviation take the population mean and standard deviation, drop values
more than 3 standard deviations from the mean, and repeat, five passes in all. Unlike a median absolute
deviation, the result does not snap to the step of coarsely quantized values.

A builder SHOULD write `channel_stats`, and MUST rewrite or remove it when it rewrites `/raw` or the
coarsest level. A reader MAY use it in place of measuring, and MUST NOT require it: stores written
before it was added lack it.

## Arrays hold codes that each array's attributes decode

`/raw[c, i]` is sample `i` of channel `c`. Each channel's samples are contiguous in a chunk.

`/minmax/s{stride}[c, 0, b]` and `[c, 1, b]` are the minimum and maximum of channel `c` over samples
`[b * stride, (b + 1) * stride)`. The last block is truncated at `n_samples`, so `B = ceil(T / stride)`.
Values past `n_samples` in a partial chunk are fill and carry no data.

Every one of these arrays decodes by its own attributes, in this order:

1. **`quantization`** present: `value = code * scale[c] + offset[c]` for `/raw`; for a pyramid level,
   row 0 uses `min_offset[c]` and row 1 uses `max_offset[c]`, with the shared `scale[c]`. The lists are
   indexed by channel. This form covers a range fit, an ADC gain, and both composed.
2. **`lsb_shift`** present: `value = code * 2**lsb_shift`.
3. Neither: `value = code`.

An array whose values were clipped to fit its dtype carries **`clipped_values`**: the number of codes
clipping changed, each now at the dtype's limit. A decoded value at the limit of such an array is a lower bound
on the magnitude it stands for. An array without the attribute was not clipped.

A reader ignores the other array attributes (`source_dtype`, `quantized_dtype`, `original_dtype`). Codes
are stored as `int8`, `uint8`, `int16`, `uint16`, `float32` or the source's dtype. Levels usually share
decode parameters, but a reader decodes each level by its own attributes rather than assuming so.

## Channel arrays label rows

`/channel_ids` holds each channel's id as a string, the label to show beside channel `c`.
`/properties/{name}` holds one SpikeInterface channel property per array, first axis by channel:
numeric arrays as stored, string arrays as strings. `properties/location` is `(C, 2)` or `(C, 3)` probe
coordinates in µm. ADC calibration properties (`gain_to_uV` and the like) are never stored here, since
the encodings already apply them.

## A pyramid-only store reads raw from another store

A pyramid-only store has no `/raw`. Its raw samples are the array at `raw_path` in the store at
`raw_source`, which may be a Zarr v2 store the builder does not control, such as a SpikeInterface zarr
(`traces_seg0`, WavPack-compressed). That array is laid out either way:

- `(C, T)` when `shape[0] <= shape[1]`, as in `/raw`;
- `(T, C)` when `shape[0] > shape[1]`, SpikeInterface's layout.

It decodes by its own attributes as above, which for a SpikeInterface array means identity: its values
are ADC counts, and the store's `units` says so. The builder only builds pyramids over an unencoded raw
array, so the pyramids and the raw they sit beside are on one scale.

## Readers choose levels themselves

Which level to draw for a given view is reader policy, outside the format. The Python reader takes the
coarsest stride with at least 2 blocks per requested point and falls back to raw below the finest; the
web viewer chooses by pixels on screen. Two conforming readers can draw the same view from different
levels.

## Revisions

Additions within version 1, which a version 1 reader may ignore:

| Date | Addition |
|---|---|
| 2026-11-23 | `channel_stats` root attribute. |
| 2026-11-23 | `clipped_values` array attribute. |
