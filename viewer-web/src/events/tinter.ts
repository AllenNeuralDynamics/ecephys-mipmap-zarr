import type { StoreInfo } from "../reader/store.js";
import type { TintRequest } from "../render/renderer.js";
import type { EventLayer, EventLayerInfo, EventWindow } from "./layer.js";
import {
  assignColours,
  buildTint,
  eventRows,
  groupAt,
  rowsOf,
  sampleMapping,
  tintStrength,
  type Selection,
  type TintContext,
} from "./tint.js";

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
  group: number;
  text: string;
  /** Ramp row, 0-based into the palette's group colours, the label's event tints with. */
  rampRow: number;
  selected: boolean;
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
  /** x, y per layer channel from the store's locations; null when it has none. */
  readonly #positions: Float64Array | null;
  readonly #onLoaded: () => void;
  readonly #onError: (err: unknown) => void;
  #selected = new Set<number>();
  #hideUnselected = false;
  #loaded: { range: [number, number]; window: EventWindow } | undefined;
  #pending: { range: [number, number]; promise: Promise<void> } | undefined;
  /** Store samples an event can reach from its own sample. */
  readonly #reach: number;

  constructor(
    layer: EventLayer,
    store: StoreInfo,
    callbacks: { onLoaded: () => void; onError: (err: unknown) => void },
  ) {
    this.#layer = layer;
    this.#store = store;
    this.#onLoaded = callbacks.onLoaded;
    this.#onError = callbacks.onError;
    const byId = new Map(store.channelIds.map((id, c) => [id, c]));
    const storeChannel = Int32Array.from(layer.info.channelIds, (id) => byId.get(id) ?? -1);
    const location = store.properties.location;
    const width = location?.shape[1] ?? 0;
    const values = location !== undefined && width >= 2 && !Array.isArray(location.values) ? location.values : null;
    const at = (c: number, axis: number): number => (values === null || c < 0 ? NaN : Number(values[c * width + axis]));
    this.#positions =
      values === null
        ? null
        : Float64Array.from({ length: 2 * storeChannel.length }, (_, i) => at(storeChannel[i >> 1] ?? -1, i & 1));
    const { scale, offset } = sampleMapping(layer.info, store);
    const none = new Int32Array(layer.info.groups.ids.length).fill(-1);
    this.#ctx = {
      info: layer.info,
      storeChannel,
      row: none,
      mutedRow: none,
      depth: values === null ? null : Float64Array.from({ length: store.nChannels }, (_, c) => at(c, 1)),
      scale,
      offset,
    };
    const [before, after] = layer.info.windowS;
    // Detections state their own widths; a quarter second covers the widest discharges.
    this.#reach = Math.max(Math.abs(before), Math.abs(after), 0.25) * store.samplingFrequency;
  }

  get info(): EventLayerInfo {
    return this.#layer.info;
  }

  get groupIds(): readonly string[] {
    return this.#layer.info.groups.ids;
  }

  /** Per group, a 0-based colour below `nColours`, apart from groups whose footprints overlap. */
  assignColours(nColours: number): Int32Array {
    return assignColours(this.#layer.info.groups, nColours, this.#positions);
  }

  /** Per group, the ramp rows it tints with normally and while other groups are selected; -1 for none. */
  setRows(row: Int32Array, mutedRow: Int32Array): void {
    this.#ctx.row = row;
    this.#ctx.mutedRow = mutedRow;
  }

  /** Groups, by index, drawn at full strength at any zoom; the rest muted, or hidden with `hideUnselected`. */
  setSelected(groups: Iterable<number>, hideUnselected: boolean): void {
    this.#selected = new Set(groups);
    this.#hideUnselected = hideUnselected;
  }

  /** The renderer's `Tinter`. */
  readonly tint = (request: TintRequest): Uint8Array | undefined => {
    const selection = this.#selection(request);
    if (selection.selected.size === 0 && selection.strength <= 0) return undefined;
    const window = this.#windowFor(request);
    if (window === undefined) return undefined;
    return buildTint(request, window, this.#ctx, this.#store.nChannels, selection);
  };

  /** The group tinted at `column` on visible `row`, allowing `slack` columns either side, or -1. */
  groupAt(request: TintRequest, column: number, row: number, slack: number): number {
    const window = this.#windowFor(request);
    if (window === undefined) return -1;
    return groupAt(request, window, this.#ctx, this.#store.nChannels, this.#selection(request), column, row, slack);
  }

  /** Group labels where the view is narrow enough: each event's id on its footprint's largest channel in view. */
  labels(request: TintRequest): EventLabel[] {
    if (this.#spanS(request) > LABEL_SPAN_S) return [];
    const window = this.#windowFor(request);
    if (window === undefined) return [];
    const rowOf = rowsOf(request.channels, this.#store.nChannels);
    const anySelected = this.#selected.size > 0;
    const last = new Map<number, number>();
    const out: EventLabel[] = [];
    for (let i = 0; i < window.sampleIndex.length; i++) {
      const g = window.group[i] ?? -1;
      const selected = this.#selected.has(g);
      if (anySelected && !selected && this.#hideUnselected) continue;
      const rampRow = (anySelected && !selected ? this.#ctx.mutedRow[g] : this.#ctx.row[g]) ?? -1;
      const { rows } = eventRows(window, i, this.#ctx, rowOf);
      const row = rows[0];
      if (rampRow < 0 || row === undefined) continue;
      const x = ((window.sampleIndex[i] ?? 0) * this.#ctx.scale + this.#ctx.offset - request.t0) / request.samplesPerPx;
      if (x < 0 || x >= request.columns || x - (last.get(row) ?? -Infinity) < LABEL_GAP_PX) continue;
      last.set(row, x);
      out.push({ x, row, group: g, text: this.#layer.info.groups.ids[g] ?? "", rampRow, selected });
    }
    return out;
  }

  /** Resolves once no fetch is outstanding. */
  async idle(): Promise<void> {
    while (this.#pending !== undefined) await this.#pending.promise;
  }

  #selection(request: TintRequest): Selection {
    return {
      selected: this.#selected,
      strength: tintStrength(this.#spanS(request)),
      hideUnselected: this.#hideUnselected,
    };
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
