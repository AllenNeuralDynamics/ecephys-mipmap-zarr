import type { EventCounts, EventLayer, EventWindow } from "../events/layer.js";
import {
  clearRaster,
  drawCounts,
  drawEvents,
  newRasterImage,
  rasterGroupAt,
  type RasterGeometry,
  type RasterImage,
  type RasterStyle,
} from "../events/raster.js";
import type { View } from "../render/view.js";

/** Most events a view fetches one by one; past it the panel draws the layer's counts. */
export const MAX_RASTER_EVENTS = 300_000;
/** Events or bins fetched either side of the view, as a fraction of its width. */
const FETCH_MARGIN = 0.5;

type Data = { kind: "events"; window: EventWindow } | { kind: "counts"; counts: EventCounts };

/** What a fetch covers, in layer samples, and at which count bin (0 for events). */
interface Coverage {
  range: [number, number];
  bin: number;
}

/**
 * The tick raster under the traces: every event in view at its group's footprint centre, or its own channel,
 * on the traces' visible slots. Wide views draw the layer's per-group counts instead of fetching each event.
 */
export class RasterPanel {
  readonly canvas: HTMLCanvasElement;
  #layer: EventLayer | undefined;
  #mapping = { scale: 1, offset: 0 };
  #samplingFrequency = 1;
  #style: RasterStyle | undefined;
  #image: RasterImage | undefined;
  #loaded: { coverage: Coverage; data: Data } | undefined;
  #pending: { coverage: Coverage; promise: Promise<void> } | undefined;
  /** The view and style last drawn, so frames that change neither skip the redraw. */
  #drawn: { view: View; width: number; style: RasterStyle; data: Data | undefined } | undefined;
  #last: { view: View; width: number } | undefined;
  readonly #onError: (err: unknown) => void;
  /** Called with a message when the view holds too many events to draw and the layer has no counts. */
  readonly #onNotice: (text: string) => void;

  constructor(
    canvas: HTMLCanvasElement,
    callbacks: { onError: (err: unknown) => void; onNotice: (text: string) => void },
  ) {
    this.canvas = canvas;
    this.#onError = callbacks.onError;
    this.#onNotice = callbacks.onNotice;
  }

  setLayer(layer: EventLayer | undefined, mapping: { scale: number; offset: number }, samplingFrequency: number): void {
    // A page's first canvas call starts the GPU process (about 200 ms); a canvas never drawn on needs no clearing.
    const drawnOn = this.#drawn !== undefined;
    this.#layer = layer;
    this.#mapping = mapping;
    this.#samplingFrequency = samplingFrequency;
    this.#loaded = undefined;
    this.#pending = undefined;
    this.#drawn = undefined;
    if (layer === undefined && drawnOn)
      this.canvas.getContext("2d")?.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  setStyle(style: RasterStyle): void {
    this.#style = style;
    this.redraw();
  }

  /** Draw the last view again, as after a style or data change. */
  redraw(): void {
    if (this.#last !== undefined) this.draw(this.#last.view, this.#last.width);
  }

  /** Draw `view` across `width` device columns, the traces' width; fetches what it lacks and draws again on arrival. */
  draw(view: View, width: number): void {
    this.#last = { view, width };
    const layer = this.#layer;
    const style = this.#style;
    const height = Math.max(1, Math.round(this.canvas.clientHeight * (globalThis.devicePixelRatio || 1)));
    if (layer === undefined || style === undefined || this.canvas.clientHeight === 0) return;
    const data = this.#dataFor(layer, view, width);
    const drawn = this.#drawn;
    if (
      drawn?.view === view &&
      drawn.width === width &&
      drawn.style === style &&
      drawn.data === data &&
      this.canvas.height === height
    )
      return;
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
    }
    if (this.#image?.width !== width || this.#image.height !== height) this.#image = newRasterImage(width, height);
    const image = this.#image;
    const geom: RasterGeometry = {
      t0: view.t0,
      samplesPerPx: view.samplesPerPx,
      firstSlot: view.firstSlot,
      nSlots: view.nSlots,
      ...this.#mapping,
      samplingFrequency: this.#samplingFrequency,
    };
    if (data === undefined) clearRaster(image, style.background);
    else if (data.kind === "events") drawEvents(image, data.window, geom, style);
    else drawCounts(image, data.counts, geom, style);
    this.canvas.getContext("2d")?.putImageData(new ImageData(image.rgba, width, height), 0, 0);
    this.#drawn = { view, width, style, data };
  }

  /** The group drawn nearest device pixel `(x, y)`, within `slack` pixels, or -1. */
  groupAt(x: number, y: number, slack: number): number {
    return this.#image === undefined || this.#drawn?.data === undefined ? -1 : rasterGroupAt(this.#image, x, y, slack);
  }

  /** Resolves once no fetch is outstanding. */
  async idle(): Promise<void> {
    while (this.#pending !== undefined) await this.#pending.promise;
  }

  /** What to draw for `view`: loaded data covering it, else the last loaded while a covering fetch runs. */
  #dataFor(layer: EventLayer, view: View, width: number): Data | undefined {
    const { scale, offset } = this.#mapping;
    const a = (view.t0 - offset) / scale;
    const b = (view.t0 + view.samplesPerPx * width - offset) / scale;
    const want = this.#coverage(layer, a, b, view.samplesPerPx / scale);
    if (want === undefined) {
      this.#onNotice("Zoom in to see the raster");
      return undefined;
    }
    this.#onNotice("");
    const covers = (c: Coverage): boolean => c.bin === want.bin && c.range[0] <= a && c.range[1] >= b;
    if (this.#loaded !== undefined && covers(this.#loaded.coverage)) return this.#loaded.data;
    if (this.#pending === undefined || !covers(this.#pending.coverage)) {
      const [lo, hi] = want.range;
      const fetched: Promise<Data> =
        want.bin === 0
          ? layer.window(Math.floor(lo), Math.ceil(hi)).then((window) => ({ kind: "events", window }))
          : layer.counts(want.bin, lo / want.bin, hi / want.bin).then((counts) => ({ kind: "counts", counts }));
      const promise = fetched
        .then(
          (data) => {
            if (this.#layer !== layer) return;
            this.#loaded = { coverage: want, data };
            this.redraw();
          },
          (err: unknown) => {
            this.#onError(err);
          },
        )
        .finally(() => {
          if (this.#pending?.promise === promise) this.#pending = undefined;
        });
      this.#pending = { coverage: want, promise };
    }
    return this.#loaded?.data;
  }

  /**
   * The range and bin to fetch for layer samples `[a, b)`: events when few enough, else the coarsest count bin
   * no wider than a column. Undefined when there are too many events and no counts.
   */
  #coverage(layer: EventLayer, a: number, b: number, samplesPerPx: number): Coverage | undefined {
    const margin = FETCH_MARGIN * (b - a);
    const range: [number, number] = [a - margin, b + margin];
    if (layer.estimate(range[0], range[1]) <= MAX_RASTER_EVENTS) return { range, bin: 0 };
    const bins = layer.countBins;
    if (bins.length === 0) return undefined;
    const bin = bins.filter((w) => w <= samplesPerPx).at(-1) ?? bins[0] ?? 0;
    return { range: [Math.floor(range[0] / bin) * bin, Math.ceil(range[1] / bin) * bin], bin };
  }
}
