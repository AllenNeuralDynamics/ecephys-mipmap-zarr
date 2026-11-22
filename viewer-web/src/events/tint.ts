import type { TintRequest } from "../render/renderer.js";
import type { EventGroups, EventLayerInfo, EventWindow } from "./layer.js";

/** View spans, seconds, over which tints of unselected groups fade out: full up to the first, gone by the second. */
export const TINT_FADE_S: [number, number] = [10, 30];

/** Strength of unselected groups' tints for a view `spanS` seconds wide, 0..1. */
export function tintStrength(spanS: number): number {
  const [full, gone] = TINT_FADE_S;
  return Math.min(1, Math.max(0, (gone - spanS) / (gone - full)));
}

/**
 * A colour per group, 0-based and below `nColours`, so groups whose footprints share a channel differ while
 * colours last. Groups go largest footprint first, and each takes the colour its coloured neighbours use least.
 * Groups without footprints cycle through the colours.
 */
export function assignColours(groups: EventGroups, nColours: number): Int32Array {
  const n = groups.ids.length;
  const colours = new Int32Array(n).fill(-1);
  if (nColours <= 0) return colours;
  const { footprintChannel: fc, footprintWeight: fw, k } = groups;
  if (fc === null || fw === null || k === 0) return colours.map((_, g) => g % nColours);
  const largest = (g: number): number => fw[g * k] ?? 0;
  const order = Array.from({ length: n }, (_, g) => g).sort((a, b) => largest(b) - largest(a) || a - b);
  const onChannel = new Map<number, number[]>();
  for (const g of order) {
    const used = new Array<number>(nColours).fill(0);
    for (let j = 0; j < k; j++) {
      const ch = fc[g * k + j] ?? -1;
      if (ch < 0) continue;
      for (const h of onChannel.get(ch) ?? []) {
        const c = colours[h] ?? 0;
        used[c] = (used[c] ?? 0) + 1;
      }
    }
    let best = 0;
    for (let c = 1; c < nColours; c++) if ((used[c] ?? 0) < (used[best] ?? 0)) best = c;
    colours[g] = best;
    for (let j = 0; j < k; j++) {
      const ch = fc[g * k + j] ?? -1;
      if (ch < 0) continue;
      const list = onChannel.get(ch);
      if (list === undefined) onChannel.set(ch, [g]);
      else list.push(g);
    }
  }
  return colours;
}

export interface TintContext {
  info: EventLayerInfo;
  /** Per layer channel, the store's channel index, or -1 where the store has no such id. */
  storeChannel: Int32Array;
  /** Per group, a 0-based palette colour, or -1 for none. */
  colour: Int32Array;
  /** Per store channel, depth along the probe (µm), for events that state a depth range. */
  depth: Float64Array | null;
  /** Store sample = layer sample × `scale` + `offset`. */
  scale: number;
  offset: number;
}

/** The mapping from a layer's samples to a store's, by clock when the layer states its start. */
export function sampleMapping(
  layer: Pick<EventLayerInfo, "samplingFrequency" | "tStart">,
  store: { samplingFrequency: number; tStart: number },
): { scale: number; offset: number } {
  const scale = store.samplingFrequency / layer.samplingFrequency;
  const offset = layer.tStart === null ? 0 : (layer.tStart - store.tStart) * store.samplingFrequency;
  return { scale, offset };
}

/** Store columns `[c0, c1)` event `i` colours, unclipped: its window, and a column either side that outlines join. */
export function eventColumns(
  window: EventWindow,
  i: number,
  ctx: TintContext,
  t0: number,
  samplesPerPx: number,
): [number, number] {
  const fs = ctx.info.samplingFrequency * ctx.scale;
  const width = window.widthS?.[i];
  const [before, after] = width === undefined ? ctx.info.windowS : [-width / 2, width / 2];
  const s = (window.sampleIndex[i] ?? 0) * ctx.scale + ctx.offset;
  const c0 = Math.floor((s + before * fs - t0) / samplesPerPx);
  return [c0 - 1, Math.max(c0 + 1, Math.ceil((s + after * fs - t0) / samplesPerPx)) + 1];
}

/**
 * Rows (visible slots) and weights event `i` colours: its group's footprint, else its own channel and any
 * channels inside its depth range.
 */
export function eventRows(
  window: EventWindow,
  i: number,
  ctx: TintContext,
  rowOf: Int32Array,
): { rows: number[]; weights: number[] } {
  const rows: number[] = [];
  const weights: number[] = [];
  const g = window.group[i] ?? -1;
  const { footprintChannel: fc, footprintWeight: fw, k } = ctx.info.groups;
  if (fc !== null && fw !== null && k > 0) {
    for (let j = 0; j < k; j++) {
      const ch = fc[g * k + j] ?? -1;
      const row = ch < 0 ? -1 : (rowOf[ctx.storeChannel[ch] ?? -1] ?? -1);
      if (row >= 0) {
        rows.push(row);
        weights.push(fw[g * k + j] ?? 0);
      }
    }
    return { rows, weights };
  }
  const weight = Math.abs(window.amplitude?.[i] ?? 1);
  const ch = window.channel?.[i] ?? -1;
  const own = ch < 0 ? -1 : (rowOf[ctx.storeChannel[ch] ?? -1] ?? -1);
  if (own >= 0) {
    rows.push(own);
    weights.push(weight);
  }
  const range = window.depthRange;
  if (range !== undefined && ctx.depth !== null) {
    const lo = Math.min(range[2 * i] ?? NaN, range[2 * i + 1] ?? NaN);
    const hi = Math.max(range[2 * i] ?? NaN, range[2 * i + 1] ?? NaN);
    ctx.depth.forEach((d, storeCh) => {
      const row = rowOf[storeCh] ?? -1;
      if (row >= 0 && row !== own && d >= lo && d <= hi) {
        rows.push(row);
        weights.push(weight / 2);
      }
    });
  }
  return { rows, weights };
}

/** Per store channel, its visible row or -1. */
export function rowsOf(channels: readonly number[], nStoreChannels: number): Int32Array {
  const rowOf = new Int32Array(nStoreChannels).fill(-1);
  channels.forEach((c, row) => {
    if (c >= 0 && c < nStoreChannels) rowOf[c] = row;
  });
  return rowOf;
}

/**
 * The renderer's tint for `window`: per visible row and column, the ramp row (group colour + 1) and strength.
 * Where events overlap on a row, the larger footprint weight there wins. With `selected` non-empty only those
 * groups tint, at full strength; otherwise every group does, at `strength`. Undefined when nothing tints.
 */
export function buildTint(
  request: TintRequest,
  window: EventWindow,
  ctx: TintContext,
  nStoreChannels: number,
  selected: ReadonlySet<number>,
  strength: number,
): Uint8Array | undefined {
  const onlySelected = selected.size > 0;
  const level = Math.round(255 * (onlySelected ? 1 : strength));
  if (level <= 0 || window.sampleIndex.length === 0) return undefined;
  const { columns, channels, t0, samplesPerPx } = request;
  const rowOf = rowsOf(channels, nStoreChannels);
  const out = new Uint8Array(columns * channels.length * 2);
  const priority = new Float32Array(columns * channels.length).fill(-1);
  let any = false;
  for (let i = 0; i < window.sampleIndex.length; i++) {
    const g = window.group[i] ?? -1;
    if (onlySelected && !selected.has(g)) continue;
    const colour = ctx.colour[g] ?? -1;
    if (colour < 0) continue;
    let [c0, c1] = eventColumns(window, i, ctx, t0, samplesPerPx);
    c0 = Math.max(0, c0);
    c1 = Math.min(columns, c1);
    if (c0 >= c1) continue;
    const { rows, weights } = eventRows(window, i, ctx, rowOf);
    for (const [j, row] of rows.entries()) {
      const weight = weights[j] ?? 0;
      for (let c = c0; c < c1; c++) {
        const at = row * columns + c;
        if (weight <= (priority[at] ?? 0)) continue;
        priority[at] = weight;
        out[2 * at] = colour + 1;
        out[2 * at + 1] = level;
        any = true;
      }
    }
  }
  return any ? out : undefined;
}
