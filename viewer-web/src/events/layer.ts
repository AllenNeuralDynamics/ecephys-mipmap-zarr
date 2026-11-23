import * as zarr from "zarrita";
import type { AsyncReadable } from "@zarrita/storage";

import { FormatError } from "../reader/attrs.js";
import { defaultResolveUrl, readProperty, toStrings, type Property, type ZArray } from "../reader/store.js";

export const EVENTS_FORMAT = "ecephys-events";
export const EVENTS_FORMAT_VERSION = 1;
/** Decoded events kept per layer, as whole chunks: a few raster windows' worth while panning back and forth. */
const CACHED_EVENTS = 1_000_000;

export interface EventGroups {
  ids: string[];
  /** Footprint channels per group, `k` each, as indices into the layer's `channelIds`; -1 pads. */
  footprintChannel: Int32Array | null;
  footprintWeight: Float32Array | null;
  k: number;
  properties: Record<string, Property>;
}

export interface EventLayerInfo {
  kind: string;
  samplingFrequency: number;
  /** Time of sample 0; null when the layer starts where the store it is drawn on starts. */
  tStart: number | null;
  channelIds: string[];
  /** Seconds before (negative) and after an event's sample that it covers. */
  windowS: [number, number];
  nEvents: number;
  groups: EventGroups;
}

/** Events in a sample range, ascending; optional columns are present when the layer stores them. */
export interface EventWindow {
  sampleIndex: Float64Array;
  group: Int32Array;
  amplitude?: Float32Array;
  channel?: Int32Array;
  /** Two per event: the shallow and deep ends of the probe span it covers, µm. */
  depthRange?: Float32Array;
  widthS?: Float32Array;
}

/** Events per group in `nBins` bins of `bin` layer samples from bin `b0`, group-major. */
export interface EventCounts {
  bin: number;
  b0: number;
  nBins: number;
  data: Uint32Array;
}

type Column = Exclude<keyof EventWindow, "sampleIndex" | "group">;
const OPTIONAL: Column[] = ["amplitude", "channel", "depthRange", "widthS"];
const PATHS: Record<keyof EventWindow, string> = {
  sampleIndex: "sample_index",
  group: "group",
  amplitude: "amplitude",
  channel: "channel",
  depthRange: "depth_range",
  widthS: "width_s",
};

function parseAttrs(attrs: Record<string, unknown>): Omit<EventLayerInfo, "groups"> {
  if (attrs.format !== EVENTS_FORMAT) throw new FormatError("not an event layer");
  const version = attrs.format_version;
  if (typeof version !== "number" || version > EVENTS_FORMAT_VERSION) {
    throw new FormatError(`event layer format version ${String(version)} is not one this viewer reads`);
  }
  const window = attrs.window_s;
  const ids = attrs.channel_ids;
  if (!Array.isArray(window) || window.length !== 2 || !Array.isArray(ids)) {
    throw new FormatError("event layer attrs need window_s and channel_ids");
  }
  return {
    kind: typeof attrs.kind === "string" ? attrs.kind : "events",
    samplingFrequency: Number(attrs.sampling_frequency),
    tStart: typeof attrs.t_start === "number" ? attrs.t_start : null,
    channelIds: ids.map(String),
    windowS: [Number(window[0]), Number(window[1])],
    nEvents: Number(attrs.n_events ?? 0),
  };
}

/** An event layer opened for windowed reads: its metadata and groups up front, events by chunk on demand. */
export class EventLayer {
  readonly #arrays: Map<keyof EventWindow, ZArray>;
  readonly #firstSample: Float64Array;
  readonly #chunk: number;
  readonly #cache = new Map<number, Promise<EventWindow>>();
  readonly #counts: Map<number, ZArray>;

  private constructor(
    readonly info: EventLayerInfo,
    arrays: Map<keyof EventWindow, ZArray>,
    firstSample: Float64Array,
    chunk: number,
    counts: Map<number, ZArray>,
  ) {
    this.#arrays = arrays;
    this.#firstSample = firstSample;
    this.#chunk = chunk;
    this.#counts = counts;
  }

  /** Bin widths, layer samples, the layer stores per-group counts at; finest first. */
  get countBins(): number[] {
    return [...this.#counts.keys()];
  }

  /** Open a layer at a URL (`s3://` resolves to HTTPS) or from a store. */
  static async open(location: string | AsyncReadable): Promise<EventLayer> {
    const base = typeof location === "string" ? new zarr.FetchStore(defaultResolveUrl(location)) : location;
    const store = await zarr.withMaybeConsolidatedMetadata(base, { format: "v3" });
    const group = await zarr.open.v3(store, { kind: "group" });
    const attrs = parseAttrs(group.attrs);
    const root = zarr.root(store);
    const openArray = (path: string): Promise<ZArray> => zarr.open.v3(root.resolve(path), { kind: "array" });
    const optional = async (path: string): Promise<ZArray | null> => {
      try {
        return await openArray(path);
      } catch (err) {
        if (err instanceof zarr.NotFoundError) return null;
        throw err;
      }
    };

    const arrays = new Map<keyof EventWindow, ZArray>();
    arrays.set("sampleIndex", await openArray(PATHS.sampleIndex));
    arrays.set("group", await openArray(PATHS.group));
    for (const name of OPTIONAL) {
      const arr = await optional(PATHS[name]);
      if (arr !== null) arrays.set(name, arr);
    }
    const first = await zarr.get(await openArray("index/first_sample"));
    const ids = toStrings((await zarr.get(await openArray("groups/id"))).data);
    const footprintChannel = await optional("groups/footprint_channel");
    const footprintWeight = await optional("groups/footprint_weight");
    const contents = "contents" in store ? store.contents() : [];
    const properties: Record<string, Property> = {};
    await Promise.all(
      contents
        .filter((node) => node.kind === "array" && node.path.startsWith("/groups/properties/"))
        .map(async (node) => {
          properties[node.path.slice("/groups/properties/".length)] = await readProperty(await openArray(node.path));
        }),
    );
    const groups: EventGroups = {
      ids,
      footprintChannel:
        footprintChannel === null ? null : Int32Array.from((await zarr.get(footprintChannel)).data as Int32Array),
      footprintWeight:
        footprintWeight === null ? null : Float32Array.from((await zarr.get(footprintWeight)).data as Float32Array),
      k: footprintChannel?.shape[1] ?? 0,
      properties,
    };
    const counts = new Map<number, ZArray>();
    const bins = Array.isArray(group.attrs.count_bins) ? group.attrs.count_bins.map(Number) : [];
    for (const bin of bins.sort((a, b) => a - b)) {
      const arr = await optional(`counts/b${String(bin)}`);
      if (arr !== null) counts.set(bin, arr);
    }
    const sampleIndex = arrays.get("sampleIndex");
    return new EventLayer(
      { ...attrs, groups },
      arrays,
      Float64Array.from(first.data as Float64Array),
      sampleIndex?.chunks[0] ?? 1,
      counts,
    );
  }

  /** About how many events have `s0 <= sample < s1`, interpolating within chunks; no fetch. */
  estimate(s0: number, s1: number): number {
    return s1 <= s0 ? 0 : Math.max(0, this.#position(s1) - this.#position(s0));
  }

  /** Fractional event index of sample `s`, linear between chunk starts. */
  #position(s: number): number {
    const first = this.#firstSample;
    const c = upperBound(first, s) - 1;
    if (c < 0) return 0;
    const start = first[c] ?? 0;
    const lastChunk = c === first.length - 1;
    const size = lastChunk ? this.info.nEvents - c * this.#chunk : this.#chunk;
    // The last chunk's end is unknown without a fetch; it counts as reached.
    const next = first[c + 1];
    const fraction = next === undefined ? 1 : Math.min(1, (s - start) / Math.max(1, next - start));
    return c * this.#chunk + fraction * size;
  }

  /** Per-group counts at `bin` over bins `[b0, b1)`, clipped to the stored ones. */
  async counts(bin: number, b0: number, b1: number): Promise<EventCounts> {
    const arr = this.#counts.get(bin);
    if (arr === undefined) throw new Error(`no counts at ${String(bin)} samples`);
    const total = arr.shape[1] ?? 0;
    const lo = Math.min(total, Math.max(0, Math.floor(b0)));
    const hi = Math.min(total, Math.max(lo, Math.ceil(b1)));
    if (hi === lo) return { bin, b0: lo, nBins: 0, data: new Uint32Array(0) };
    const out = await zarr.get(arr, [null, zarr.slice(lo, hi)]);
    return { bin, b0: lo, nBins: hi - lo, data: Uint32Array.from(out.data as Uint32Array) };
  }

  /** Events with `s0 <= sample < s1`. */
  async window(s0: number, s1: number): Promise<EventWindow> {
    const n = this.#firstSample.length;
    if (n === 0 || s1 <= s0) return { sampleIndex: new Float64Array(0), group: new Int32Array(0) };
    // The chunk that may hold s0 is the last one starting at or before it.
    let c0 = upperBound(this.#firstSample, s0) - 1;
    c0 = Math.max(0, c0);
    const c1 = Math.max(c0 + 1, lowerBound(this.#firstSample, s1));
    const parts = await Promise.all(Array.from({ length: c1 - c0 }, (_, i) => this.#read(c0 + i)));
    return sliceWindow(concatWindows(parts), s0, s1);
  }

  #read(chunk: number): Promise<EventWindow> {
    let pending = this.#cache.get(chunk);
    if (pending !== undefined) {
      this.#cache.delete(chunk);
      this.#cache.set(chunk, pending);
      return pending;
    }
    pending = this.#fetch(chunk);
    pending.catch(() => this.#cache.delete(chunk));
    this.#cache.set(chunk, pending);
    while (this.#cache.size > Math.max(1, Math.ceil(CACHED_EVENTS / this.#chunk))) {
      const oldest = this.#cache.keys().next().value;
      if (oldest === undefined) break;
      this.#cache.delete(oldest);
    }
    return pending;
  }

  async #fetch(chunk: number): Promise<EventWindow> {
    const start = chunk * this.#chunk;
    const end = Math.min(this.info.nEvents, start + this.#chunk);
    const out: Partial<Record<keyof EventWindow, unknown>> = {};
    await Promise.all(
      [...this.#arrays].map(async ([name, arr]) => {
        const selection = arr.shape.length === 2 ? [zarr.slice(start, end), null] : [zarr.slice(start, end)];
        out[name] = (await zarr.get(arr, selection)).data;
      }),
    );
    return out as unknown as EventWindow;
  }
}

/** First index whose value is >= `x`. */
export function lowerBound(values: ArrayLike<number>, x: number): number {
  let lo = 0;
  let hi = values.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if ((values[mid] ?? Infinity) < x) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/** First index whose value is > `x`. */
export function upperBound(values: ArrayLike<number>, x: number): number {
  let lo = 0;
  let hi = values.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if ((values[mid] ?? Infinity) <= x) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

type Typed = Float64Array | Float32Array | Int32Array;

function concatTyped<T extends Typed>(parts: T[]): T {
  if (parts.length === 1 && parts[0] !== undefined) return parts[0];
  const first = parts[0];
  if (first === undefined) throw new Error("nothing to join");
  const out = new (first.constructor as new (n: number) => T)(parts.reduce((n, p) => n + p.length, 0));
  let at = 0;
  for (const p of parts) {
    out.set(p, at);
    at += p.length;
  }
  return out;
}

function concatWindows(parts: EventWindow[]): EventWindow {
  const out: EventWindow = {
    sampleIndex: concatTyped(parts.map((p) => p.sampleIndex)),
    group: concatTyped(parts.map((p) => p.group)),
  };
  for (const name of OPTIONAL) {
    const columns = parts.map((p) => p[name]).filter((c) => c !== undefined);
    if (columns.length === parts.length) (out as unknown as Record<string, Typed>)[name] = concatTyped(columns);
  }
  return out;
}

/** The events of `window` with `s0 <= sample < s1`. */
export function sliceWindow(window: EventWindow, s0: number, s1: number): EventWindow {
  const a = lowerBound(window.sampleIndex, s0);
  const b = lowerBound(window.sampleIndex, s1);
  const out: EventWindow = { sampleIndex: window.sampleIndex.subarray(a, b), group: window.group.subarray(a, b) };
  for (const name of OPTIONAL) {
    const column = window[name];
    if (column === undefined) continue;
    const width = name === "depthRange" ? 2 : 1;
    (out as unknown as Record<string, Typed>)[name] = column.subarray(a * width, b * width);
  }
  return out;
}
