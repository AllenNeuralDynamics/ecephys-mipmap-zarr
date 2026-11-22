import type { StoreInfo } from "../reader/store.js";
import type { TintRequest } from "../render/renderer.js";
import type { EventLayer, EventWindow } from "./layer.js";
import { assignColours, buildTint, eventRows, rowsOf, sampleMapping, tintStrength, type TintContext } from "./tint.js";

/** Widest view, seconds, that gets group labels. */
export const LABEL_SPAN_S = 0.05;
/** Least gap between labels on one row, device pixels. */
const LABEL_GAP_PX = 28;
/** Events fetched either side of the view, as a fraction of its width, so short pans need no fetch. */
const FETCH_MARGIN = 0.5;

export interface EventLabel {
  /** Device pixels from the canvas's left edge. */
  x: number;
  /** Visible row. */
  row: number;
  text: string;
  /** 0-based palette colour. */
  colour: number;
}

/** Store-sample range of `request`'s columns. */
const requestRange = (request: TintRequest): [number, number] => [
  request.t0,
  request.t0 + request.columns * request.samplesPerPx,
];

/**
 * Colours a renderer's traces from an event layer: keeps a window of events around the view, fetching a wider one
 * when the view leaves it, and asks for a redraw when it arrives.
 */
export class EventTinter {
  readonly #layer: EventLayer;
  readonly #store: StoreInfo;
  readonly #ctx: TintContext;
  readonly #onLoaded: () => void;
  readonly #onError: (err: unknown) => void;
  #selected = new Set<number>();
  #loaded: { range: [number, number]; window: EventWindow } | undefined;
  #pending: { range: [number, number]; promise: Promise<void> } | undefined;
  /** Store samples an event can reach from its own sample. */
  readonly #reach: number;

  constructor(
    layer: EventLayer,
    store: StoreInfo,
    nColours: number,
    callbacks: { onLoaded: () => void; onError: (err: unknown) => void },
  ) {
    this.#layer = layer;
    this.#store = store;
    this.#onLoaded = callbacks.onLoaded;
    this.#onError = callbacks.onError;
    const byId = new Map(store.channelIds.map((id, c) => [id, c]));
    const location = store.properties.location;
    const width = location?.shape[1] ?? 0;
    const depth =
      location !== undefined && width >= 2 && !Array.isArray(location.values)
        ? Float64Array.from({ length: store.nChannels }, (_, c) => Number(location.values[c * width + 1] ?? NaN))
        : null;
    const { scale, offset } = sampleMapping(layer.info, store);
    this.#ctx = {
      info: layer.info,
      storeChannel: Int32Array.from(layer.info.channelIds, (id) => byId.get(id) ?? -1),
      colour: assignColours(layer.info.groups, nColours),
      depth,
      scale,
      offset,
    };
    const [before, after] = layer.info.windowS;
    // Detections state their own widths; a quarter second covers the widest discharges.
    this.#reach = Math.max(Math.abs(before), Math.abs(after), 0.25) * store.samplingFrequency;
  }

  get groupIds(): readonly string[] {
    return this.#layer.info.groups.ids;
  }

  /** Per group, its 0-based palette colour or -1. */
  get colours(): Int32Array {
    return this.#ctx.colour;
  }

  setColours(nColours: number): void {
    this.#ctx.colour = assignColours(this.#layer.info.groups, nColours);
  }

  /** Groups, by row, that alone are coloured, at full strength at any zoom; empty colours every group. */
  setSelected(groups: Iterable<number>): void {
    this.#selected = new Set(groups);
  }

  /** The renderer's `Tinter`. */
  readonly tint = (request: TintRequest): Uint8Array | undefined => {
    const strength = tintStrength(this.#spanS(request));
    if (this.#selected.size === 0 && strength <= 0) return undefined;
    const window = this.#windowFor(request);
    if (window === undefined) return undefined;
    return buildTint(request, window, this.#ctx, this.#store.nChannels, this.#selected, strength);
  };

  /** Group labels where the view is narrow enough: each event's id on its footprint's largest channel in view. */
  labels(request: TintRequest): EventLabel[] {
    if (this.#spanS(request) > LABEL_SPAN_S) return [];
    const window = this.#windowFor(request);
    if (window === undefined) return [];
    const rowOf = rowsOf(request.channels, this.#store.nChannels);
    const last = new Map<number, number>();
    const out: EventLabel[] = [];
    for (let i = 0; i < window.sampleIndex.length; i++) {
      const g = window.group[i] ?? -1;
      if (this.#selected.size > 0 && !this.#selected.has(g)) continue;
      const colour = this.#ctx.colour[g] ?? -1;
      const { rows } = eventRows(window, i, this.#ctx, rowOf);
      const row = rows[0];
      if (colour < 0 || row === undefined) continue;
      const x = ((window.sampleIndex[i] ?? 0) * this.#ctx.scale + this.#ctx.offset - request.t0) / request.samplesPerPx;
      if (x < 0 || x >= request.columns || x - (last.get(row) ?? -Infinity) < LABEL_GAP_PX) continue;
      last.set(row, x);
      out.push({ x, row, text: this.#layer.info.groups.ids[g] ?? "", colour });
    }
    return out;
  }

  /** Resolves once no fetch is outstanding. */
  async idle(): Promise<void> {
    while (this.#pending !== undefined) await this.#pending.promise;
  }

  #spanS(request: TintRequest): number {
    return (request.columns * request.samplesPerPx) / this.#store.samplingFrequency;
  }

  /** Events around `request`, or the last ones fetched while a covering window is on its way. */
  #windowFor(request: TintRequest): EventWindow | undefined {
    const [a, b] = requestRange(request);
    const need: [number, number] = [a - this.#reach, b + this.#reach];
    const covers = (range: [number, number]): boolean => range[0] <= need[0] && range[1] >= need[1];
    if (this.#loaded !== undefined && covers(this.#loaded.range)) return this.#loaded.window;
    if (this.#pending === undefined || !covers(this.#pending.range)) {
      const margin = FETCH_MARGIN * (b - a);
      const range: [number, number] = [need[0] - margin, need[1] + margin];
      const { scale, offset } = this.#ctx;
      const promise = this.#layer
        .window(Math.floor((range[0] - offset) / scale), Math.ceil((range[1] - offset) / scale))
        .then(
          (window) => {
            this.#loaded = { range, window };
            this.#onLoaded();
          },
          (err: unknown) => {
            this.#onError(err);
          },
        )
        .finally(() => {
          if (this.#pending?.promise === promise) this.#pending = undefined;
        });
      this.#pending = { range, promise };
    }
    return this.#loaded?.window;
  }
}
