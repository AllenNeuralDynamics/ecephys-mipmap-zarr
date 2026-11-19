import { levelKey, type ArrayInfo, type StoreInfo } from "./store.js";

/** Blocks per pixel the chosen level must at least provide; tuned in the renderer. */
export const DEFAULT_OVERSAMPLE = 2;

/** One array and the index range along its time axis (samples for raw, blocks for a level) a view needs. */
export interface Plan {
  key: string;
  /** Samples per index: 1 for raw. */
  stride: number;
  /** Index range, end exclusive, clamped to the array. */
  start: number;
  end: number;
}

/**
 * Choose the array for samples `[t0, t1)` drawn across `widthPx` pixels: the coarsest level with at least
 * `oversample` blocks per pixel, else raw, else (raw unavailable) the finest level.
 */
export function planWindow(
  info: StoreInfo,
  t0: number,
  t1: number,
  widthPx: number,
  oversample: number = DEFAULT_OVERSAMPLE,
): Plan {
  const samplesPerPixel = (t1 - t0) / Math.max(1, widthPx);
  let stride: number | undefined;
  for (const s of info.strides) if (samplesPerPixel / s >= oversample) stride = s;
  if (stride === undefined && !("raw" in info.arrays)) stride = info.strides[0];
  if (stride === undefined) {
    const start = Math.max(0, Math.floor(t0));
    return { key: "raw", stride: 1, start, end: Math.max(start, Math.min(info.nSamples, Math.ceil(t1))) };
  }
  const nBlocks = Math.ceil(info.nSamples / stride);
  const start = Math.max(0, Math.floor(t0 / stride));
  return { key: levelKey(stride), stride, start, end: Math.max(start, Math.min(nBlocks, Math.ceil(t1 / stride))) };
}

/** Axis positions of channel and time in an array's stored shape. */
export function axes(array: ArrayInfo): { channel: number; time: number } {
  if (array.timeMajor) return { channel: 1, time: 0 };
  return { channel: 0, time: array.shape.length - 1 };
}

/** Inner-chunk coordinates covering channels `[c0, c1)` and time indices `[start, end)` of `array`. */
export function tilesFor(
  array: ArrayInfo,
  channels: readonly [number, number],
  start: number,
  end: number,
): number[][] {
  const { channel, time } = axes(array);
  const chunkC = array.chunks[channel] ?? 1;
  const chunkT = array.chunks[time] ?? 1;
  const coords: number[][] = [];
  if (channels[1] <= channels[0] || end <= start) return coords;
  for (let tc = Math.floor(start / chunkT); tc * chunkT < end; tc++) {
    for (let cc = Math.floor(channels[0] / chunkC); cc * chunkC < channels[1]; cc++) {
      const coord = new Array<number>(array.shape.length).fill(0);
      coord[channel] = cc;
      coord[time] = tc;
      coords.push(coord);
    }
  }
  return coords;
}
