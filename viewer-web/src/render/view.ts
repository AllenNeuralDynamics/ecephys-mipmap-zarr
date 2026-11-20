/**
 * What the canvas shows. Times are float64 sample positions (sample `i` spans `[i, i + 1)`); slots are the
 * rows channels are drawn in, top to bottom, and may be fractional while scrolling.
 */
export interface View {
  t0: number;
  samplesPerPx: number;
  firstSlot: number;
  nSlots: number;
  /** Multiplies the store's automatic amplitude scale. */
  gain: number;
}

export interface Bounds {
  nSamples: number;
  /** Channels in the display order. */
  nDisplayed: number;
}

/** Fewest device pixels per sample a view zooms in to. */
export const MAX_PX_PER_SAMPLE = 32;

/** Keep the view inside the recording and its channel list, with the zoom between one sample per 32 px and the whole recording. */
export function clampView(view: View, bounds: Bounds, widthPx: number): View {
  const maxSpp = Math.max(bounds.nSamples / Math.max(1, widthPx), 1 / MAX_PX_PER_SAMPLE);
  const samplesPerPx = Math.min(maxSpp, Math.max(1 / MAX_PX_PER_SAMPLE, view.samplesPerPx));
  const span = samplesPerPx * widthPx;
  const t0 = Math.min(Math.max(0, bounds.nSamples - span), Math.max(0, view.t0));
  const nSlots = Math.max(1, Math.min(bounds.nDisplayed, Math.round(view.nSlots)));
  const firstSlot = Math.min(Math.max(0, bounds.nDisplayed - nSlots), Math.max(0, view.firstSlot));
  return { t0, samplesPerPx, firstSlot, nSlots, gain: view.gain > 0 ? view.gain : 1 };
}

/** The sample range `[t0, t1)` across `widthPx`. */
export function visibleSamples(view: View, widthPx: number): [number, number] {
  return [view.t0, view.t0 + view.samplesPerPx * widthPx];
}

/** Zoom time by `factor` (above 1 zooms out) keeping the sample under `xPx` fixed. */
export function zoomAbout(view: View, factor: number, xPx: number): View {
  const anchor = view.t0 + xPx * view.samplesPerPx;
  const samplesPerPx = view.samplesPerPx * factor;
  return { ...view, samplesPerPx, t0: anchor - xPx * samplesPerPx };
}

/** Move the content right by `dxPx` (the view moves earlier). */
export function panBy(view: View, dxPx: number): View {
  return { ...view, t0: view.t0 - dxPx * view.samplesPerPx };
}

/** View showing samples `[s0, s1)` across `widthPx`. */
export function viewOfRange(view: View, s0: number, s1: number, widthPx: number): View {
  return { ...view, t0: s0, samplesPerPx: Math.max(s1 - s0, 1e-9) / Math.max(1, widthPx) };
}

/** Horizontal device-pixel position of sample position `t`. */
export function pxOf(view: View, t: number): number {
  return (t - view.t0) / view.samplesPerPx;
}
