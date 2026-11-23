import type { TintRequest } from "../render/renderer.js";
import type { EventGroups, EventLayerInfo, EventWindow } from "./layer.js";

/** View spans, seconds, over which tints of unselected groups fade out: full up to the first, gone by the second. */
export const TINT_FADE_S: [number, number] = [10, 30];
/**
 * Length scale, µm, of the overlap between two footprints' channels: channels this far apart count e^-1 as much
 * as coincident ones, so units on neighbouring sites conflict and units a few sites apart barely do.
 */
export const OVERLAP_UM = 25;
/** Spacing, µm, assumed between channels adjacent in the layer's order when the store has no locations. */
const FALLBACK_PITCH_UM = 15;

/** Strength of unselected groups' tints for a view `spanS` seconds wide, 0..1. */
export function tintStrength(spanS: number): number {
  const [full, gone] = TINT_FADE_S;
  return Math.min(1, Math.max(0, (gone - spanS) / (gone - full)));
}

interface Footprint {
  x: number[];
  y: number[];
  /** Relative to the footprint's largest. */
  w: number[];
  box: [number, number, number, number];
}

function footprints(groups: EventGroups, positions: ArrayLike<number> | null): (Footprint | null)[] {
  const { footprintChannel: fc, footprintWeight: fw, k } = groups;
  return groups.ids.map((_, g) => {
    if (fc === null || fw === null) return null;
    const fp: Footprint = { x: [], y: [], w: [], box: [Infinity, Infinity, -Infinity, -Infinity] };
    let top = 0;
    for (let j = 0; j < k; j++) top = Math.max(top, Math.abs(fw[g * k + j] ?? 0));
    for (let j = 0; j < k; j++) {
      const ch = fc[g * k + j] ?? -1;
      if (ch < 0 || top <= 0) continue;
      const x = positions === null ? 0 : (positions[2 * ch] ?? NaN);
      const y = positions === null ? ch * FALLBACK_PITCH_UM : (positions[2 * ch + 1] ?? NaN);
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      fp.x.push(x);
      fp.y.push(y);
      fp.w.push(Math.abs(fw[g * k + j] ?? 0) / top);
      fp.box = [Math.min(fp.box[0], x), Math.min(fp.box[1], y), Math.max(fp.box[2], x), Math.max(fp.box[3], y)];
    }
    return fp.w.length > 0 ? fp : null;
  });
}

/**
 * Pairwise overlap of groups' footprints, sparse: for each group, its neighbours and the summed product of their
 * channel weights, each pair discounted by `exp(-(d / OVERLAP_UM)²)` for the distance between the channels.
 * `positions` holds x, y per layer channel (NaN where unknown); null spaces channels along a line in layer order.
 */
export function footprintOverlap(groups: EventGroups, positions: ArrayLike<number> | null): Map<number, number>[] {
  const fps = footprints(groups, positions);
  const reach = 3 * OVERLAP_UM;
  const overlap = fps.map(() => new Map<number, number>());
  // Sweep along y so only footprints whose boxes come within reach are compared.
  const present = fps.flatMap((fp, g) => (fp === null ? [] : [{ g, fp }]));
  present.sort((a, b) => a.fp.box[1] - b.fp.box[1]);
  for (const [i, { g: ga, fp: a }] of present.entries()) {
    for (const { g: gb, fp: b } of present.slice(i + 1)) {
      if (b.box[1] - a.box[3] > reach) break;
      if (b.box[3] < a.box[1] - reach || b.box[0] - a.box[2] > reach || a.box[0] - b.box[2] > reach) continue;
      let sum = 0;
      a.w.forEach((wa, p) => {
        b.w.forEach((wb, q) => {
          const d2 = ((a.x[p] ?? 0) - (b.x[q] ?? 0)) ** 2 + ((a.y[p] ?? 0) - (b.y[q] ?? 0)) ** 2;
          sum += wa * wb * Math.exp(-d2 / OVERLAP_UM ** 2);
        });
      });
      if (sum > 1e-3) {
        overlap[ga]?.set(gb, sum);
        overlap[gb]?.set(ga, sum);
      }
    }
  }
  return overlap;
}

/**
 * A colour per group, 0-based and below `nColours`, so groups whose footprints overlap differ while colours last.
 * Greedy, most-overlapped group first: each takes the colour whose groups it overlaps least, ties going to the
 * colour used least so far. Groups without footprints take the least-used colour.
 */
export function assignColours(
  groups: EventGroups,
  nColours: number,
  positions: ArrayLike<number> | null,
  shown?: ReadonlySet<number>,
): Int32Array {
  const overlap = footprintOverlap(groups, positions);
  if (shown === undefined) return colourGraph(overlap, nColours);
  // Colour the shown groups among themselves, so hidden ones neither take colours nor push neighbours apart.
  const kept = [...shown].filter((g) => g >= 0 && g < overlap.length).sort((x, y) => x - y);
  const at = new Map(kept.map((g, i) => [g, i]));
  const sub = kept.map((g) => {
    const m = new Map<number, number>();
    for (const [h, v] of overlap[g] ?? []) {
      const j = at.get(h);
      if (j !== undefined) m.set(j, v);
    }
    return m;
  });
  const colours = new Int32Array(overlap.length).fill(-1);
  colourGraph(sub, nColours).forEach((c, i) => (colours[kept[i] ?? -1] = c));
  return colours;
}

/** Greedy colouring of a weighted conflict graph, as `assignColours` describes. */
export function colourGraph(overlap: readonly Map<number, number>[], nColours: number): Int32Array {
  const n = overlap.length;
  const colours = new Int32Array(n).fill(-1);
  if (nColours <= 0) return colours;
  const total = overlap.map((m) => [...m.values()].reduce((s, v) => s + v, 0));
  const order = Array.from({ length: n }, (_, g) => g).sort((a, b) => (total[b] ?? 0) - (total[a] ?? 0) || a - b);
  const used = new Float64Array(nColours);
  for (const g of order) {
    const cost = new Float64Array(nColours);
    for (const [h, v] of overlap[g] ?? []) {
      const c = colours[h] ?? -1;
      if (c >= 0) cost[c] = (cost[c] ?? 0) + v;
    }
    let best = 0;
    for (let c = 1; c < nColours; c++) {
      const dc = (cost[c] ?? 0) - (cost[best] ?? 0);
      if (dc < -1e-9 || (Math.abs(dc) <= 1e-9 && (used[c] ?? 0) < (used[best] ?? 0))) best = c;
    }
    colours[g] = best;
    used[best] = (used[best] ?? 0) + 1;
  }
  return colours;
}

export interface TintContext {
  info: EventLayerInfo;
  /** Per layer channel, the store's channel index, or -1 where the store has no such id. */
  storeChannel: Int32Array;
  /** Per group, the ramp row (0-based into the palette's group colours) it tints with, or -1 for none. */
  row: Int32Array;
  /** Per group, the row it tints with while other groups are selected. */
  mutedRow: Int32Array;
  /** Per store channel, depth along the probe (µm), for events that state a depth range. */
  depth: Float64Array | null;
  /** Store sample = layer sample × `scale` + `offset`. */
  scale: number;
  offset: number;
}

/** Which groups tint and how: `selected` groups always, at full strength; the rest muted or not at all. */
export interface Selection {
  selected: ReadonlySet<number>;
  /** Strength, 0..1, of groups when none are selected, and of unselected groups shown muted. */
  strength: number;
  hideUnselected: boolean;
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

/** A selected group's spans outrank any unselected one's, whatever their footprint weights. */
const SELECTED_PRIORITY = 1e12;

interface Paint {
  /** Event index in the window. */
  event: number;
  group: number;
  /** Ramp row, 0-based. */
  rampRow: number;
  /** Tint strength, 0..255. */
  level: number;
  row: number;
  c0: number;
  c1: number;
  priority: number;
}

/** Each span of `window`'s events that tints under `selection`, clipped to `request`'s columns. */
function* paints(
  request: TintRequest,
  window: EventWindow,
  ctx: TintContext,
  rowOf: Int32Array,
  selection: Selection,
): Generator<Paint> {
  const { selected, strength, hideUnselected } = selection;
  const anySelected = selected.size > 0;
  const faded = Math.round(255 * strength);
  for (let i = 0; i < window.sampleIndex.length; i++) {
    const g = window.group[i] ?? -1;
    const isSelected = selected.has(g);
    if (!isSelected && (faded <= 0 || (anySelected && hideUnselected))) continue;
    const rampRow = (anySelected && !isSelected ? ctx.mutedRow[g] : ctx.row[g]) ?? -1;
    if (rampRow < 0) continue;
    let [c0, c1] = eventColumns(window, i, ctx, request.t0, request.samplesPerPx);
    c0 = Math.max(0, c0);
    c1 = Math.min(request.columns, c1);
    if (c0 >= c1) continue;
    const { rows, weights } = eventRows(window, i, ctx, rowOf);
    for (const [j, row] of rows.entries()) {
      const weight = weights[j] ?? 0;
      yield {
        event: i,
        group: g,
        rampRow,
        level: isSelected ? 255 : faded,
        row,
        c0,
        c1,
        priority: isSelected ? SELECTED_PRIORITY + weight : weight,
      };
    }
  }
}

/**
 * The renderer's tint for `window`: per visible row and column, the ramp row + 1 and strength. Where events
 * overlap on a row, a selected group's wins, then the larger footprint weight there. Undefined when nothing tints.
 */
export function buildTint(
  request: TintRequest,
  window: EventWindow,
  ctx: TintContext,
  nStoreChannels: number,
  selection: Selection,
): Uint8Array | undefined {
  if (window.sampleIndex.length === 0) return undefined;
  const { columns, channels } = request;
  const out = new Uint8Array(columns * channels.length * 2);
  const priority = new Float64Array(columns * channels.length).fill(-1);
  let any = false;
  for (const p of paints(request, window, ctx, rowsOf(channels, nStoreChannels), selection)) {
    for (let c = p.c0; c < p.c1; c++) {
      const at = p.row * columns + c;
      if (p.priority <= (priority[at] ?? 0)) continue;
      priority[at] = p.priority;
      out[2 * at] = p.rampRow + 1;
      out[2 * at + 1] = p.level;
      any = true;
    }
  }
  return any ? out : undefined;
}

/** The group whose tint covers `column` on visible `row`, within `slack` columns, or -1. */
export function groupAt(
  request: TintRequest,
  window: EventWindow,
  ctx: TintContext,
  nStoreChannels: number,
  selection: Selection,
  column: number,
  row: number,
  slack: number,
): number {
  let best = -1;
  let bestMiss = Infinity;
  let bestPriority = -Infinity;
  for (const p of paints(request, window, ctx, rowsOf(request.channels, nStoreChannels), selection)) {
    if (p.row !== row) continue;
    const miss = Math.max(0, p.c0 - column, column - (p.c1 - 1));
    if (miss > slack) continue;
    // Nearest first, then the one the tint draws on top.
    if (miss < bestMiss || (miss === bestMiss && p.priority > bestPriority)) {
      bestMiss = miss;
      bestPriority = p.priority;
      best = p.group;
    }
  }
  return best;
}
