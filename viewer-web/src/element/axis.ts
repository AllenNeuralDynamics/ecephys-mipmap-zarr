import { lowerBound } from "../events/layer.js";
import type { View } from "../render/view.js";

/** Least spacing, CSS px, between labelled ticks. */
const LABEL_SPACING_PX = 90;
/** Marker tick height, CSS px. */
const MARKER_PX = 7;

/** Markers in store samples, ascending, with their groups and a CSS colour per group. */
export interface Markers {
  sample: Float64Array;
  group: Int32Array;
  ids: readonly string[];
  colours: readonly string[];
}

/** The 1, 2, 5 × 10^k seconds at or above `minStepS`. */
export function niceStep(minStepS: number): number {
  const power = 10 ** Math.floor(Math.log10(minStepS));
  for (const m of [1, 2, 5, 10]) if (m * power >= minStepS) return m * power;
  return 10 * power;
}

/** `seconds` with enough decimals to tell ticks `stepS` apart. */
export function formatSeconds(seconds: number, stepS: number): string {
  const decimals = Math.max(0, Math.min(6, -Math.floor(Math.log10(stepS) + 1e-9)));
  return `${seconds.toFixed(decimals)} s`;
}

/**
 * Index of the marker nearest past `sample` in `direction`, ignoring any within `minGap` samples of it; -1 when
 * there is none.
 */
export function stepMarker(samples: Float64Array, sample: number, direction: 1 | -1, minGap: number): number {
  if (direction > 0) {
    const i = lowerBound(samples, sample + minGap);
    return i < samples.length ? i : -1;
  }
  return lowerBound(samples, sample - minGap) - 1;
}

/** The time axis under the traces, in seconds from the store's first sample, with marker ticks along its top. */
export class TimeAxis {
  readonly canvas: HTMLCanvasElement;
  #markers: Markers | undefined;
  #last: { view: View; width: number; samplingFrequency: number } | undefined;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
  }

  get markers(): Markers | undefined {
    return this.#markers;
  }

  setMarkers(markers: Markers | undefined): void {
    this.#markers = markers;
    this.redraw();
  }

  redraw(): void {
    if (this.#last !== undefined) this.draw(this.#last.view, this.#last.width, this.#last.samplingFrequency);
  }

  /** Draw `view` across `width` device columns, the traces' width. */
  draw(view: View, width: number, samplingFrequency: number): void {
    this.#last = { view, width, samplingFrequency };
    const ratio = globalThis.devicePixelRatio || 1;
    const height = Math.max(1, Math.round(this.canvas.clientHeight * ratio));
    if (this.canvas.clientHeight === 0) return;
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
    }
    const ctx = this.canvas.getContext("2d");
    if (ctx === null) return;
    const style = getComputedStyle(this.canvas);
    ctx.clearRect(0, 0, width, height);
    const x = (sample: number): number => (sample - view.t0) / view.samplesPerPx;

    const markers = this.#markers;
    if (markers !== undefined) {
      const a = lowerBound(markers.sample, view.t0 - view.samplesPerPx);
      const b = lowerBound(markers.sample, view.t0 + (width + 1) * view.samplesPerPx);
      const w = Math.max(1, Math.round(ratio * 1.5));
      for (let i = a; i < b; i++) {
        ctx.fillStyle = markers.colours[markers.group[i] ?? 0] ?? style.color;
        ctx.fillRect(Math.round(x(markers.sample[i] ?? 0) - w / 2), 0, w, MARKER_PX * ratio);
      }
    }

    const fs = samplingFrequency;
    const stepS = niceStep((LABEL_SPACING_PX * ratio * view.samplesPerPx) / fs);
    const first = Math.ceil(view.t0 / fs / stepS);
    const last = Math.floor((view.t0 + width * view.samplesPerPx) / fs / stepS);
    ctx.fillStyle = style.color;
    ctx.font = `${String(10 * ratio)}px system-ui, sans-serif`;
    ctx.textBaseline = "bottom";
    ctx.textAlign = "center";
    for (let k = first; k <= last; k++) {
      const s = k * stepS;
      const px = Math.round(x(s * fs));
      ctx.globalAlpha = 0.5;
      ctx.fillRect(px, MARKER_PX * ratio, Math.max(1, Math.round(ratio)), 3 * ratio);
      ctx.globalAlpha = 1;
      const label = formatSeconds(s, stepS);
      const half = ctx.measureText(label).width / 2;
      ctx.fillText(label, Math.min(width - half, Math.max(half, px)), height - ratio);
    }
  }

  /** The marker nearest device column `x` within `slack` columns, or -1. */
  markerAt(xPx: number, slack: number): number {
    const markers = this.#markers;
    const last = this.#last;
    if (markers === undefined || last === undefined) return -1;
    const sample = last.view.t0 + xPx * last.view.samplesPerPx;
    const reach = slack * last.view.samplesPerPx;
    const a = lowerBound(markers.sample, sample - reach);
    let best = -1;
    for (let i = a; i < markers.sample.length && (markers.sample[i] ?? Infinity) <= sample + reach; i++) {
      if (best < 0 || Math.abs((markers.sample[i] ?? 0) - sample) < Math.abs((markers.sample[best] ?? 0) - sample))
        best = i;
    }
    return best;
  }
}
