import type { EventCounts, EventGroups, EventWindow } from "./layer.js";

/** Rate, events per second, that draws a group at about two-thirds opacity in a column. */
export const RASTER_RATE_HZ = 5;
/** Tallest tick, device pixels, however few channels are in view. */
const MAX_TICK_PX = 4;

/** An RGBA image with, per pixel, the group drawn last there or -1. */
export interface RasterImage {
  width: number;
  height: number;
  rgba: Uint8ClampedArray<ArrayBuffer>;
  owner: Int32Array;
}

/** Where the panel's columns and rows fall: columns as the traces' store samples, rows as their visible slots. */
export interface RasterGeometry {
  t0: number;
  samplesPerPx: number;
  firstSlot: number;
  nSlots: number;
  /** Store sample of layer sample `s` is `s * scale + offset`. */
  scale: number;
  offset: number;
  samplingFrequency: number;
}

export interface RasterStyle {
  /** Per group, the displayed slot its footprint centres on, fractional; NaN when none of it is displayed. */
  groupSlot: Float64Array;
  /** Per layer channel, its displayed slot or -1. */
  channelSlot: Int32Array;
  /** Per group, `0xrrggbb`, or -1 to leave it out. */
  colour: Int32Array;
  /** Per group, 1 to draw over the rest, twice as tall. */
  top: Uint8Array;
  background: number;
}

export function newRasterImage(width: number, height: number): RasterImage {
  return {
    width,
    height,
    rgba: new Uint8ClampedArray(width * height * 4),
    owner: new Int32Array(width * height),
  };
}

/** Per group, the weighted mean of its footprint channels' displayed slots, over the ones displayed. */
export function groupSlots(groups: EventGroups, channelSlot: Int32Array): Float64Array {
  const { k, footprintChannel, footprintWeight } = groups;
  return Float64Array.from(groups.ids, (_, g) => {
    let sum = 0;
    let weight = 0;
    for (let j = 0; j < k; j++) {
      const slot = channelSlot[footprintChannel?.[g * k + j] ?? -1] ?? -1;
      const w = footprintWeight?.[g * k + j] ?? 0;
      if (slot < 0 || !(w > 0)) continue;
      sum += w * (slot + 0.5);
      weight += w;
    }
    return weight > 0 ? sum / weight : NaN;
  });
}

/** Fill with `background` and forget every pixel's group. */
export function clearRaster(image: RasterImage, background: number): void {
  const { rgba, owner } = image;
  const r = (background >> 16) & 255;
  const g = (background >> 8) & 255;
  const b = background & 255;
  for (let i = 0; i < rgba.length; i += 4) {
    rgba[i] = r;
    rgba[i + 1] = g;
    rgba[i + 2] = b;
    rgba[i + 3] = 255;
  }
  owner.fill(-1);
}

/** Blend `colour` at `alpha` into a tick at column `x` centred on `y`, `h` rows tall. */
function tick(image: RasterImage, x: number, y: number, h: number, colour: number, alpha: number, group: number): void {
  const { width, height, rgba, owner } = image;
  if (x < 0 || x >= width) return;
  const top = Math.max(0, Math.round(y - h / 2));
  const bottom = Math.min(height, top + h);
  const r = (colour >> 16) & 255;
  const g = (colour >> 8) & 255;
  const b = colour & 255;
  for (let row = top; row < bottom; row++) {
    const p = row * width + x;
    const i = p * 4;
    rgba[i] = (rgba[i] ?? 0) + (r - (rgba[i] ?? 0)) * alpha;
    rgba[i + 1] = (rgba[i + 1] ?? 0) + (g - (rgba[i + 1] ?? 0)) * alpha;
    rgba[i + 2] = (rgba[i + 2] ?? 0) + (b - (rgba[i + 2] ?? 0)) * alpha;
    owner[p] = group;
  }
}

interface Frame {
  tickPx: number;
  /** Device rows per slot. */
  rowPx: number;
  /** Events a column holds at `RASTER_RATE_HZ`, at least one. */
  expected: number;
}

function frame(image: RasterImage, geom: RasterGeometry): Frame {
  const rowPx = image.height / Math.max(1, geom.nSlots);
  const columnS = geom.samplesPerPx / geom.samplingFrequency;
  return {
    rowPx,
    tickPx: Math.max(1, Math.min(MAX_TICK_PX, Math.floor(rowPx))),
    expected: Math.max(1, columnS * RASTER_RATE_HZ),
  };
}

/** Opacity of `n` events in one column: one event alone, or a group at `RASTER_RATE_HZ`, draws at 1 - 1/e. */
const opacity = (n: number, expected: number): number => 1 - Math.exp(-n / expected);

/** Draw each group's per-column totals in `perColumn` (group-major, `image.width` each) at its slot. */
function drawTotals(image: RasterImage, perColumn: Float32Array, geom: RasterGeometry, style: RasterStyle): void {
  const f = frame(image, geom);
  const { width } = image;
  for (const pass of [0, 1]) {
    style.groupSlot.forEach((slot, g) => {
      const colour = style.colour[g] ?? -1;
      if (colour < 0 || Number.isNaN(slot) || (style.top[g] ?? 0) !== pass) return;
      const y = (slot - geom.firstSlot) * f.rowPx;
      const h = f.tickPx * (pass + 1);
      if (y < -h || y > image.height + h) return;
      for (let x = 0; x < width; x++) {
        const n = perColumn[g * width + x] ?? 0;
        if (n > 0) tick(image, x, y, h, colour, opacity(n, f.expected), g);
      }
    });
  }
}

/**
 * Draw events: those with a channel on its slot, one tick each, unless it is not displayed; the rest totalled per
 * column at their group's footprint centre. Groups with `top` draw last.
 */
export function drawEvents(image: RasterImage, window: EventWindow, geom: RasterGeometry, style: RasterStyle): void {
  clearRaster(image, style.background);
  const f = frame(image, geom);
  const { width } = image;
  const perColumn = new Float32Array(style.groupSlot.length * width);
  const { sampleIndex, group, channel } = window;
  const direct: number[] = [];
  for (let i = 0; i < sampleIndex.length; i++) {
    const g = group[i] ?? -1;
    if ((style.colour[g] ?? -1) < 0) continue;
    const x = Math.floor(((sampleIndex[i] ?? 0) * geom.scale + geom.offset - geom.t0) / geom.samplesPerPx);
    if (x < 0 || x >= width) continue;
    const own = channel?.[i] ?? -1;
    if (own < 0) perColumn[g * width + x] = (perColumn[g * width + x] ?? 0) + 1;
    else if ((style.channelSlot[own] ?? -1) >= 0) direct.push(i);
  }
  drawTotals(image, perColumn, geom, style);
  for (const pass of [0, 1]) {
    for (const i of direct) {
      const g = group[i] ?? -1;
      if ((style.top[g] ?? 0) !== pass) continue;
      const slot = style.channelSlot[channel?.[i] ?? -1] ?? -1;
      const x = Math.floor(((sampleIndex[i] ?? 0) * geom.scale + geom.offset - geom.t0) / geom.samplesPerPx);
      const y = (slot + 0.5 - geom.firstSlot) * f.rowPx;
      tick(image, x, y, f.tickPx * (pass + 1), style.colour[g] ?? 0, opacity(1, 1), g);
    }
  }
}

/** Draw per-group counts at each group's footprint centre, a bin's count spread over the columns it covers. */
export function drawCounts(image: RasterImage, counts: EventCounts, geom: RasterGeometry, style: RasterStyle): void {
  clearRaster(image, style.background);
  const { width } = image;
  const nGroups = style.groupSlot.length;
  const perColumn = new Float32Array(nGroups * width);
  const pxPerBin = (counts.bin * geom.scale) / geom.samplesPerPx;
  for (let b = 0; b < counts.nBins; b++) {
    const x0 = ((counts.b0 + b) * counts.bin * geom.scale + geom.offset - geom.t0) / geom.samplesPerPx;
    const x1 = x0 + pxPerBin;
    if (x1 <= 0 || x0 >= width) continue;
    const c0 = Math.max(0, Math.floor(x0));
    const c1 = Math.min(width, Math.max(c0 + 1, Math.ceil(x1)));
    // A bin wider than a column gives each column its share.
    const share = 1 / Math.max(1, pxPerBin);
    for (let g = 0; g < nGroups; g++) {
      const n = counts.data[g * counts.nBins + b] ?? 0;
      if (n === 0) continue;
      for (let x = c0; x < c1; x++) perColumn[g * width + x] = (perColumn[g * width + x] ?? 0) + n * share;
    }
  }
  drawTotals(image, perColumn, geom, style);
}

/** The group drawn nearest pixel `(x, y)` within `slack` pixels, or -1. */
export function rasterGroupAt(image: RasterImage, x: number, y: number, slack: number): number {
  let best = -1;
  let bestD = Infinity;
  const r = Math.ceil(slack);
  for (let dy = -r; dy <= r; dy++) {
    const row = Math.round(y) + dy;
    if (row < 0 || row >= image.height) continue;
    for (let dx = -r; dx <= r; dx++) {
      const col = Math.round(x) + dx;
      if (col < 0 || col >= image.width) continue;
      const g = image.owner[row * image.width + col] ?? -1;
      const d = dx * dx + dy * dy;
      if (g >= 0 && d < bestD) {
        best = g;
        bestD = d;
      }
    }
  }
  return best;
}

/** `#rrggbb` or `rgb[a](r, g, b[, a])` as `0xrrggbb`; -1 for anything else. */
export function packColour(css: string): number {
  const hex = /^#([0-9a-f]{6})$/i.exec(css);
  if (hex?.[1] !== undefined) return parseInt(hex[1], 16);
  const rgb = /^rgba?\(\s*(\d+)[\s,]+(\d+)[\s,]+(\d+)/i.exec(css);
  if (rgb === null) return -1;
  return (Number(rgb[1]) << 16) | (Number(rgb[2]) << 8) | Number(rgb[3]);
}
