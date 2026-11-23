import { ByteLru } from "../reader/cache.js";
import { axes, DEFAULT_OVERSAMPLE, planWindow } from "../reader/plan.js";
import { levelKey, type ArrayInfo, type StoreInfo } from "../reader/store.js";
import { decodeTile, type Tile } from "../reader/tiles.js";
import { Programs } from "./programs.js";
import { buildRamp, FILL_LEVEL, NOISE_LEVEL, RAMP_STEPS, type TracePalette } from "./ramp.js";
import { decodeTerms, floatRow, uploadDecode, uploadTile, type TileTexture } from "./textures.js";
import { clampView, visibleSamples, type View } from "./view.js";
import { centreAndExcursion, clippedStats, quantile, unitRangeOf } from "./stats.js";

/** Anything that serves tiles: the worker client, or a `TileSource` in-process. */
export interface TileProvider {
  getTile(key: string, coords: readonly number[], signal?: AbortSignal): Promise<Tile>;
  /** Fetch `tiles` ahead of need, behind any requested tile; replaces the last prefetch. */
  prefetch?(tiles: readonly { key: string; coords: readonly number[] }[]): void;
}

export interface RendererOptions {
  /** Bytes of tile textures kept on the GPU, apart from the coarsest level, which stays resident. */
  textureBytes?: number;
  oversample?: number;
  palette?: TracePalette;
  onError?: (err: unknown) => void;
  /** Called after `setView`, `setDisplayed` or `resize` changes the view or the channels. */
  onViewChange?: (view: View) => void;
  /** Called after each frame is drawn. */
  onDraw?: () => void;
}

/** What a tinter is asked to colour: canvas columns from sample `t0`, and the channel in each visible row. */
export interface TintRequest {
  t0: number;
  samplesPerPx: number;
  columns: number;
  channels: readonly number[];
}

/**
 * Per visible row and column, two bytes: the palette group (1-based, so 0 means none) and the tint's strength
 * out of 255. `undefined` draws no tint.
 */
export type Tinter = (request: TintRequest) => Uint8Array | undefined;

export const DEFAULT_PALETTE: TracePalette = {
  background: "#fbfbfc",
  fill: "#e8edf5",
  noise: "#a9b6ca",
  ink: "#10141c",
  groups: [],
  lineWidth: [1, 1.7],
};

const DEFAULT_TEXTURE_BYTES = 256 * 1024 ** 2;

interface Needed {
  id: string;
  key: string;
  coords: number[];
  /** Sample range the tile covers, for scissoring a fallback. */
  s0: number;
  s1: number;
  channels: [number, number];
}

const tileId = (key: string, coords: readonly number[]): string => `${key}|${coords.join(",")}`;

/** Sds over which an outline rises from the noise level to full beyond the gate. */
const OUTLINE_RISE = 2;

/**
 * Sds from the centre where an outline starts to brighten: past the expected extreme of a column's samples
 * (about sqrt(2 ln n) sd for n normal samples), so wider columns do not light up on noise alone.
 */
export function gateStart(samplesPerPx: number): number {
  return Math.max(3.5, Math.sqrt(2 * Math.log(Math.max(1, samplesPerPx))) + 1);
}
/** Extent texels no source reached; past any clamped offset and within half-float range. */
const EMPTY_EXTENT = 6e4;

/**
 * The trace canvas's WebGL2 attributes. A context's attributes are fixed by the first `getContext` call, so a host
 * creating it early, to start the GPU while metadata loads, must pass these.
 */
export const CONTEXT_OPTIONS: WebGLContextAttributes = { antialias: false, premultipliedAlpha: true };

/**
 * Draws a store's view on a WebGL2 canvas, fetching the tiles each frame needs and drawing coarser resident
 * tiles in place of any still in flight. Redraws only when the view, the channels or the resident tiles change.
 */
export class Renderer {
  readonly #gl: WebGL2RenderingContext;
  readonly #programs: Programs;
  readonly #textures: ByteLru<string, TileTexture>;
  readonly #pinned = new Map<string, TileTexture>();
  readonly #decode = new Map<string, WebGLTexture>();
  readonly #slots: WebGLTexture;
  readonly #sigmaTexture: WebGLTexture;
  readonly #extent: WebGLTexture;
  readonly #extentBuffer: WebGLFramebuffer;
  #extentSize: [number, number] = [0, 0];
  readonly #level: WebGLTexture;
  readonly #levelBuffer: WebGLFramebuffer;
  #levelSize: [number, number] = [0, 0];
  readonly #ramp: WebGLTexture;
  readonly #tint: WebGLTexture;
  #palette: TracePalette;
  #tinter: Tinter | undefined;
  /** The array whose step the noise texture holds, or undefined when it needs uploading. */
  #noiseKey: string | undefined;
  readonly #inFlight = new Map<string, AbortController>();
  /** Tiles whose fetch failed; not retried, so a missing chunk cannot loop the redraw. */
  readonly #failed = new Set<string>();
  readonly #vao: WebGLVertexArrayObject;
  readonly #coarsest: string;
  #view: View;
  #displayed: number[];
  #centre: Float32Array;
  /** Per channel noise sd and baseline from a raw chunk; 0 and NaN when unknown. */
  #sigma: Float32Array;
  #baseline: Float32Array;
  #unitRange = 1;
  #statsReady = false;
  #frame: number | null = null;
  #idleWaiters: (() => void)[] = [];
  #destroyed = false;
  /** The view last prefetched around. */
  #prefetchedFor: View | undefined;
  /** MAX blending into float32 works, so levels can carry the row that drew them (`PACK` in the shaders). */
  readonly #floatBlend: boolean;

  constructor(
    readonly canvas: HTMLCanvasElement,
    readonly info: StoreInfo,
    readonly tiles: TileProvider,
    readonly options: RendererOptions = {},
  ) {
    const gl = canvas.getContext("webgl2", CONTEXT_OPTIONS);
    if (gl === null) throw new Error("WebGL2 is unavailable");
    if (gl.getExtension("EXT_color_buffer_float") === null)
      throw new Error("WebGL2 float render targets are unavailable");
    this.#gl = gl;
    this.#floatBlend = gl.getExtension("EXT_float_blend") !== null;
    this.#programs = new Programs(gl);
    this.#textures = new ByteLru(options.textureBytes ?? DEFAULT_TEXTURE_BYTES, (_, t) => {
      gl.deleteTexture(t.texture);
    });
    this.#vao = gl.createVertexArray();
    this.#slots = gl.createTexture();
    this.#sigmaTexture = gl.createTexture();
    this.#extent = gl.createTexture();
    this.#extentBuffer = gl.createFramebuffer();
    this.#level = gl.createTexture();
    this.#levelBuffer = gl.createFramebuffer();
    this.#ramp = gl.createTexture();
    this.#tint = gl.createTexture();
    this.#palette = options.palette ?? DEFAULT_PALETTE;
    this.#uploadRamp();
    this.#coarsest = levelKey(info.strides[info.strides.length - 1] ?? 0);
    this.#displayed = Array.from({ length: info.nChannels }, (_, c) => c);
    this.#centre = new Float32Array(info.nChannels);
    this.#sigma = new Float32Array(info.nChannels);
    this.#baseline = new Float32Array(info.nChannels).fill(NaN);
    this.#view = { t0: 0, samplesPerPx: 1, firstSlot: 0, nSlots: Math.min(info.nChannels, 64), gain: 1 };
    this.#uploadSlots();
    this.resize();
    this.#view = clampView(
      { ...this.#view, samplesPerPx: info.nSamples / Math.max(1, canvas.width) },
      this.#bounds(),
      canvas.width,
    );
    this.#loadStats();
  }

  get view(): View {
    return this.#view;
  }

  get displayed(): readonly number[] {
    return this.#displayed;
  }

  /**
   * Value units per slot height at gain 1: twice the median across channels of each one's `EXCURSION_QUANTILE`
   * excursion from its centre at the coarsest level, so slow swings count as well as block widths.
   */
  get unitRange(): number {
    return this.#unitRange;
  }

  /** Where values land: each channel's centre value, slot height and pixels per unit, in device pixels. */
  layout(): { centres: Float32Array; slotPx: number; pxPerUnit: number } {
    const slotPx = this.canvas.height / this.#view.nSlots;
    return { centres: this.#centre, slotPx, pxPerUnit: (slotPx * this.#view.gain) / this.#unitRange };
  }

  setPalette(palette: TracePalette): void {
    this.#palette = palette;
    this.#uploadRamp();
    this.#invalidate();
  }

  /** Colour events on the traces; called on every draw with what is in view. */
  setTinter(tinter: Tinter | undefined): void {
    this.#tinter = tinter;
    this.#invalidate();
  }

  /** Redraw, for a tinter whose data changed. */
  redraw(): void {
    this.#invalidate();
  }

  setView(view: Partial<View>): View {
    this.#view = clampView({ ...this.#view, ...view }, this.#bounds(), this.canvas.width);
    this.#invalidate();
    this.options.onViewChange?.(this.#view);
    return this.#view;
  }

  /** Channels to draw, top to bottom. */
  setDisplayed(channels: readonly number[]): void {
    this.#displayed = channels.filter((c) => Number.isInteger(c) && c >= 0 && c < this.info.nChannels);
    this.#uploadSlots();
    this.setView({});
  }

  /** Match the drawing buffer to the canvas's CSS size at the device pixel ratio. */
  resize(): void {
    const ratio = globalThis.devicePixelRatio || 1;
    const width = Math.max(1, Math.round(this.canvas.clientWidth * ratio));
    const height = Math.max(1, Math.round(this.canvas.clientHeight * ratio));
    if (width === this.canvas.width && height === this.canvas.height) return;
    const span = this.#view.samplesPerPx * this.canvas.width;
    this.canvas.width = width;
    this.canvas.height = height;
    this.#view = clampView({ ...this.#view, samplesPerPx: span / width }, this.#bounds(), width);
    this.#invalidate();
    this.options.onViewChange?.(this.#view);
  }

  /** Resolves once every tile the view needs is drawn and nothing is in flight. */
  idle(): Promise<void> {
    if (this.#isIdle()) return Promise.resolve();
    return new Promise((resolve) => this.#idleWaiters.push(resolve));
  }

  /** Draw now and read the drawing buffer back, bottom row first. */
  readPixels(): { width: number; height: number; data: Uint8Array } {
    this.#draw();
    const gl = this.#gl;
    const data = new Uint8Array(this.canvas.width * this.canvas.height * 4);
    gl.readPixels(0, 0, this.canvas.width, this.canvas.height, gl.RGBA, gl.UNSIGNED_BYTE, data);
    return { width: this.canvas.width, height: this.canvas.height, data };
  }

  destroy(): void {
    this.#destroyed = true;
    if (this.#frame !== null) cancelAnimationFrame(this.#frame);
    for (const controller of this.#inFlight.values()) controller.abort();
    this.#inFlight.clear();
    const gl = this.#gl;
    for (const t of this.#pinned.values()) gl.deleteTexture(t.texture);
    for (const t of this.#decode.values()) gl.deleteTexture(t);
    gl.deleteTexture(this.#slots);
    gl.deleteTexture(this.#sigmaTexture);
    gl.deleteTexture(this.#extent);
    gl.deleteFramebuffer(this.#extentBuffer);
    gl.deleteTexture(this.#level);
    gl.deleteFramebuffer(this.#levelBuffer);
    gl.deleteTexture(this.#ramp);
    gl.deleteTexture(this.#tint);
    gl.getExtension("WEBGL_lose_context")?.loseContext();
  }

  #bounds(): { nSamples: number; nDisplayed: number } {
    return { nSamples: this.info.nSamples, nDisplayed: this.#displayed.length };
  }

  #isIdle(): boolean {
    return this.#statsReady && this.#frame === null && this.#inFlight.size === 0;
  }

  #invalidate(): void {
    if (this.#frame !== null || this.#destroyed) return;
    this.#frame = requestAnimationFrame(() => {
      this.#frame = null;
      this.#draw();
      this.options.onDraw?.();
      if (this.#isIdle()) {
        this.#prefetch();
        for (const resolve of this.#idleWaiters.splice(0)) resolve();
      }
    });
  }

  #array(key: string): ArrayInfo {
    const array = this.info.arrays[key];
    if (array === undefined) throw new Error(`store has no array ${key}`);
    return array;
  }

  #texture(id: string): TileTexture | undefined {
    return this.#pinned.get(id) ?? this.#textures.get(id);
  }

  /**
   * Fetch the whole coarsest level once: it sets the amplitude scale and backs every fallback. Also fetch raw's
   * middle time chunk, for each channel's noise sd and the baseline it is centred on.
   */
  #loadStats(): void {
    const array = this.#array(this.#coarsest);
    const channels = Array.from({ length: this.info.nChannels }, (_, c) => c);
    const all = this.#tilesCovering(array, channels, 0, array.shape[2] ?? 0);
    const stored = this.info.channelStats;
    if (stored !== null) {
      // The coarsest level is still fetched, as the fallback under every tile in flight, but nothing waits on it.
      this.#sigma.set(stored.noiseSd);
      this.#centre = Float32Array.from(stored.centre);
      this.#unitRange = unitRangeOf(Array.from(stored.excursion));
      this.#statsReady = true;
      this.#uploadSlots();
      for (const need of all) this.#request(need, () => undefined);
      return;
    }
    const raw = this.info.arrays.raw;
    const middle = Math.floor(this.info.nSamples / 2);
    const noise = raw === undefined ? [] : this.#tilesCovering(raw, channels, middle, middle + 1);
    const decoded: { tile: Tile; values: Float32Array; shape: number[] }[] = [];
    let remaining = all.length + noise.length;
    const settled = (): void => {
      if (--remaining === 0) this.#finishStats(array, decoded);
    };
    for (const need of all) {
      this.#request(need, (tile) => {
        if (tile !== null) decoded.push({ tile, ...decodeTile(tile, array) });
        settled();
      });
    }
    for (const need of noise) {
      this.#request(need, (tile) => {
        if (tile !== null && raw !== undefined) this.#measureNoise(tile, raw);
        settled();
      });
    }
  }

  #measureNoise(tile: Tile, raw: ArrayInfo): void {
    const { shape, values } = decodeTile(tile, raw);
    const [nC = 0, nT = 0] = shape;
    const c0 = (tile.coords[axes(raw).channel] ?? 0) * (raw.chunks[axes(raw).channel] ?? 0);
    for (let c = 0; c < nC; c++) {
      const { mean, sd } = clippedStats(values.subarray(c * nT, (c + 1) * nT));
      this.#sigma[c0 + c] = Number.isFinite(sd) ? sd : 0;
      this.#baseline[c0 + c] = mean;
    }
  }

  #finishStats(array: ArrayInfo, decoded: { tile: Tile; values: Float32Array; shape: number[] }[]): void {
    const lows: number[][] = Array.from({ length: this.info.nChannels }, () => []);
    const highs: number[][] = Array.from({ length: this.info.nChannels }, () => []);
    for (const { tile, values, shape } of decoded) {
      const [nC = 0, , nB = 0] = shape;
      const c0 = (tile.coords[0] ?? 0) * (array.chunks[0] ?? 0);
      for (let c = 0; c < nC; c++) {
        for (let b = 0; b < nB; b++) {
          lows[c0 + c]?.push(values[c * 2 * nB + b] ?? NaN);
          highs[c0 + c]?.push(values[(c * 2 + 1) * nB + b] ?? NaN);
        }
      }
    }
    const excursions: number[] = [];
    this.#centre = Float32Array.from(lows, (lo, c) => {
      const { centre, excursion } = centreAndExcursion(lo, highs[c] ?? [], this.#baseline[c] ?? NaN);
      excursions.push(excursion);
      return centre;
    });
    this.#unitRange = unitRangeOf(excursions);
    this.#statsReady = true;
    this.#uploadSlots();
    this.#invalidate();
  }

  #uploadSlots(): void {
    const gl = this.#gl;
    const texels = new Float32Array(this.info.nChannels * 2).fill(-1);
    for (let c = 0; c < this.info.nChannels; c++) texels[c * 2 + 1] = this.#centre[c] ?? 0;
    this.#displayed.forEach((c, slot) => (texels[c * 2] = slot));
    floatRow(gl, texels, this.info.nChannels, gl.RG32F, gl.RG, this.#slots);
    this.#noiseKey = undefined;
  }

  /** Per displayed slot: noise sd and `array`'s quantization step, in value units. */
  #uploadNoise(array: ArrayInfo): void {
    const gl = this.#gl;
    // A channel without its own sd (a failed or flat chunk) takes the median of the others.
    const known = [...this.#sigma].filter((v) => v > 0);
    const fallback = known.length > 0 ? quantile(known, 0.5) : 0;
    const texels = new Float32Array(Math.max(1, this.#displayed.length) * 2);
    this.#displayed.forEach((c, slot) => {
      const sd = this.#sigma[c] ?? 0;
      texels[slot * 2] = sd > 0 ? sd : fallback;
      texels[slot * 2 + 1] = Math.abs(decodeTerms(array.decode, c).scale);
    });
    floatRow(gl, texels, texels.length / 2, gl.RG32F, gl.RG, this.#sigmaTexture);
    this.#noiseKey = array.key;
  }

  #uploadRamp(): void {
    const gl = this.#gl;
    const rows = 1 + this.#palette.groups.length;
    gl.bindTexture(gl.TEXTURE_2D, this.#ramp);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, RAMP_STEPS, rows, 0, gl.RGBA, gl.UNSIGNED_BYTE, buildRamp(this.#palette));
    for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, p, gl.NEAREST);
  }

  /** Size the level texture to the canvas and clear it. */
  #prepareLevel(width: number, height: number): void {
    const gl = this.#gl;
    if (this.#levelSize[0] !== width || this.#levelSize[1] !== height) {
      gl.bindTexture(gl.TEXTURE_2D, this.#level);
      if (this.#floatBlend) gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, width, height, 0, gl.RED, gl.FLOAT, null);
      else gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
      for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, p, gl.NEAREST);
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.#levelBuffer);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.#level, 0);
      this.#levelSize = [width, height];
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.#levelBuffer);
    gl.viewport(0, 0, width, height);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
  }

  /** Upload the tinter's colours for the visible rows, or report that there are none. */
  #uploadTint(view: View, columns: number, channels: readonly number[]): boolean {
    const data = this.#tinter?.({ t0: view.t0, samplesPerPx: view.samplesPerPx, columns, channels });
    if (data === undefined || data.length !== columns * channels.length * 2) return false;
    const gl = this.#gl;
    gl.bindTexture(gl.TEXTURE_2D, this.#tint);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RG8, columns, channels.length, 0, gl.RG, gl.UNSIGNED_BYTE, data);
    for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, p, gl.NEAREST);
    return true;
  }

  /** Size the extent texture to `columns` by `rows` and clear it to empty. */
  #prepareExtent(columns: number, rows: number): void {
    const gl = this.#gl;
    if (this.#extentSize[0] !== columns || this.#extentSize[1] !== rows) {
      gl.bindTexture(gl.TEXTURE_2D, this.#extent);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RG16F, columns, rows, 0, gl.RG, gl.HALF_FLOAT, null);
      for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, p, gl.NEAREST);
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.#extentBuffer);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.#extent, 0);
      this.#extentSize = [columns, rows];
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.#extentBuffer);
    gl.viewport(0, 0, columns, rows);
    gl.disable(gl.SCISSOR_TEST);
    gl.clearBufferfv(gl.COLOR, 0, [EMPTY_EXTENT, EMPTY_EXTENT, 0, 0]);
  }

  /** Tiles of `array` covering `channels` and indices `[start, end)`, grouped by channel chunk. */
  /**
   * Once the view's tiles are in, ask for what a pan or zoom-out reaches next: a view's width either side at this
   * level, and three widths of the next coarser level.
   */
  #prefetch(): void {
    const prefetch = this.tiles.prefetch?.bind(this.tiles);
    const view = this.#view;
    if (prefetch === undefined || view === this.#prefetchedFor) return;
    this.#prefetchedFor = view;
    const width = this.canvas.width;
    const [s0, s1] = visibleSamples(view, width);
    const plan = planWindow(this.info, s0, s1, width, this.options.oversample ?? DEFAULT_OVERSAMPLE);
    const visible = this.#displayed.slice(Math.floor(view.firstSlot), Math.ceil(view.firstSlot + view.nSlots) + 1);
    if (visible.length === 0) return;
    const array = this.#array(plan.key);
    const span = plan.end - plan.start;
    const wanted = [
      ...this.#tilesCovering(array, visible, plan.start - span, plan.start),
      ...this.#tilesCovering(array, visible, plan.end, plan.end + span),
    ];
    const coarser = this.info.strides.find((s) => s > array.stride);
    if (coarser !== undefined) {
      const level = this.#array(levelKey(coarser));
      const reach = s1 - s0;
      wanted.push(
        ...this.#tilesCovering(level, visible, Math.floor((s0 - reach) / coarser), Math.ceil((s1 + reach) / coarser)),
      );
    }
    prefetch(wanted.filter((n) => this.#texture(n.id) === undefined).map(({ key, coords }) => ({ key, coords })));
  }

  #tilesCovering(array: ArrayInfo, channels: readonly number[], start: number, end: number): Needed[] {
    const { channel, time } = axes(array);
    const chunkC = array.chunks[channel] ?? 1;
    const chunkT = array.chunks[time] ?? 1;
    const chunksC = [...new Set(channels.map((c) => Math.floor(c / chunkC)))].sort((a, b) => a - b);
    const needed: Needed[] = [];
    start = Math.max(0, start);
    end = Math.min(array.shape[time] ?? 0, end);
    if (end <= start) return needed;
    for (let tc = Math.floor(start / chunkT); tc * chunkT < end; tc++) {
      for (const cc of chunksC) {
        const coords = new Array<number>(array.shape.length).fill(0);
        coords[channel] = cc;
        coords[time] = tc;
        needed.push({
          id: tileId(array.key, coords),
          key: array.key,
          coords,
          s0: tc * chunkT * array.stride,
          s1: (tc + 1) * chunkT * array.stride,
          channels: [cc * chunkC, (cc + 1) * chunkC],
        });
      }
    }
    return needed;
  }

  /** Fetch and upload one tile; `onSettled` gets the tile, or null when the fetch failed or was aborted. */
  #request(need: Needed, onSettled?: (tile: Tile | null) => void): void {
    if (this.#inFlight.has(need.id) || this.#failed.has(need.id) || this.#texture(need.id) !== undefined) return;
    const controller = new AbortController();
    this.#inFlight.set(need.id, controller);
    this.tiles.getTile(need.key, need.coords, controller.signal).then(
      (tile) => {
        if (this.#inFlight.get(need.id) !== controller || this.#destroyed) return;
        this.#inFlight.delete(need.id);
        const texture = uploadTile(this.#gl, tile, this.#array(need.key));
        if (need.key === this.#coarsest) this.#pinned.set(need.id, texture);
        else this.#textures.set(need.id, texture, texture.bytes);
        onSettled?.(tile);
        this.#invalidate();
      },
      (err: unknown) => {
        if (this.#inFlight.get(need.id) === controller) this.#inFlight.delete(need.id);
        if (!(err instanceof Error && err.name === "AbortError")) {
          this.#failed.add(need.id);
          this.options.onError?.(err);
        }
        onSettled?.(null);
        this.#invalidate();
      },
    );
  }

  #decodeTexture(array: ArrayInfo): WebGLTexture {
    let texture = this.#decode.get(array.key);
    if (texture === undefined) {
      texture = uploadDecode(this.#gl, array.decode, this.info.nChannels);
      this.#decode.set(array.key, texture);
    }
    return texture;
  }

  #draw(): void {
    const gl = this.#gl;
    const view = this.#view;
    const width = this.canvas.width;
    const height = this.canvas.height;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, width, height);
    gl.clearColor(0, 0, 0, 0);
    gl.disable(gl.SCISSOR_TEST);
    gl.clear(gl.COLOR_BUFFER_BIT);

    const [s0, s1] = visibleSamples(view, width);
    const plan = planWindow(this.info, s0, s1, width, this.options.oversample ?? DEFAULT_OVERSAMPLE);
    const first = Math.floor(view.firstSlot);
    const visible = this.#displayed.slice(first, Math.ceil(view.firstSlot + view.nSlots) + 1);
    if (visible.length === 0) return;
    const array = this.#array(plan.key);
    const needed = this.#tilesCovering(array, visible, plan.start, plan.end);
    if (!this.#statsReady) {
      // Drawing needs the amplitude scale and centres, but fetching the view's tiles does not: start them now.
      for (const need of needed) this.#request(need);
      return;
    }
    const wanted = new Set(needed.map((n) => n.id));
    for (const [id, controller] of this.#inFlight) {
      if (!wanted.has(id) && !id.startsWith(`${this.#coarsest}|`)) {
        controller.abort();
        this.#inFlight.delete(id);
      }
    }

    const coarser = this.info.strides.filter((s) => s > array.stride).map(levelKey);
    const draws: { array: ArrayInfo; texture: TileTexture; scissor?: [number, number] }[] = [];
    for (const need of needed) {
      const texture = this.#texture(need.id);
      if (texture !== undefined) {
        draws.push({ array, texture });
        continue;
      }
      this.#request(need);
      for (const key of coarser) {
        const level = this.#array(key);
        const channels = visible.filter((c) => c >= need.channels[0] && c < need.channels[1]);
        const fallback = this.#tilesCovering(
          level,
          channels,
          Math.floor(need.s0 / level.stride),
          Math.ceil(need.s1 / level.stride),
        );
        const textures = fallback.map((n) => this.#texture(n.id));
        if (textures.every((t) => t !== undefined)) {
          const x0 = Math.floor((need.s0 - view.t0) / view.samplesPerPx);
          const x1 = Math.ceil((need.s1 - view.t0) / view.samplesPerPx);
          for (const texture of textures) draws.push({ array: level, texture, scissor: [x0, x1] });
          break;
        }
      }
    }

    const slotPx = height / view.nSlots;
    const pxPerUnit = (slotPx * view.gain) / this.#unitRange;
    const rows = visible.length;
    gl.bindVertexArray(this.#vao);
    gl.enable(gl.BLEND);

    // Pass 1: each column's extremes per visible slot. MIN blending keeps the outermost of every source.
    this.#prepareExtent(width, rows);
    gl.blendEquation(gl.MIN);
    for (const { array: drawn, texture, scissor } of draws) {
      const envelope = drawn.key !== "raw";
      const count = envelope ? texture.length * 6 : Math.max(0, texture.length - 1) * 6;
      if (count === 0) continue;
      const { program, uniforms: u } = this.#programs.get(envelope ? "extent-envelope" : "extent-raw", texture.sampler);
      gl.useProgram(program);
      if (scissor === undefined) gl.disable(gl.SCISSOR_TEST);
      else {
        gl.enable(gl.SCISSOR_TEST);
        gl.scissor(scissor[0], 0, Math.max(0, scissor[1] - scissor[0]), rows);
      }
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, texture.texture);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, this.#decodeTexture(drawn));
      gl.activeTexture(gl.TEXTURE2);
      gl.bindTexture(gl.TEXTURE_2D, this.#slots);
      gl.uniform1i(u.u_codes, 0);
      gl.uniform1i(u.u_decode, 1);
      gl.uniform1i(u.u_slots, 2);
      gl.uniform1i(u.u_width, texture.width);
      gl.uniform1i(u.u_rowsPerSeries, texture.rowsPerSeries);
      gl.uniform1i(u.u_c0, texture.c0);
      gl.uniform1f(u.u_originPx, (texture.start * drawn.stride - view.t0) / view.samplesPerPx);
      gl.uniform1f(u.u_pxPerIndex, drawn.stride / view.samplesPerPx);
      gl.uniform2f(u.u_canvasPx, width, height);
      gl.uniform1f(u.u_pxPerUnit, pxPerUnit);
      gl.uniform1i(u.u_row0, first);
      gl.uniform1i(u.u_rows, rows);
      gl.uniform1f(u.u_limit, height + 4);
      gl.drawArraysInstanced(gl.TRIANGLES, 0, count, texture.nChannels);
    }
    gl.disable(gl.SCISSOR_TEST);

    // Pass 2: fill and outlines from the extremes, as levels.
    if (this.#noiseKey !== array.key) this.#uploadNoise(array);
    this.#prepareLevel(width, height);
    gl.blendEquation(gl.MAX);
    const { program, uniforms: u } = this.#programs.get("trace");
    gl.useProgram(program);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.#extent);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.#sigmaTexture);
    gl.uniform1i(u.u_extent, 0);
    gl.uniform1i(u.u_sigma, 1);
    gl.uniform2f(u.u_canvasPx, width, height);
    gl.uniform1f(u.u_firstSlot, view.firstSlot);
    gl.uniform1f(u.u_slotPx, slotPx);
    gl.uniform1f(u.u_pxPerUnit, pxPerUnit);
    gl.uniform1i(u.u_row0, first);
    gl.uniform1i(u.u_columns, width);
    const ratio = Math.max(1, globalThis.devicePixelRatio || 1);
    const [thin, thick] = this.#palette.lineWidth;
    gl.uniform2f(u.u_halfWidth, 0.5 * thin * ratio, 0.5 * thick * ratio);
    gl.uniform2f(u.u_gate, gateStart(view.samplesPerPx), OUTLINE_RISE);
    gl.uniform1f(u.u_dim, NOISE_LEVEL);
    gl.uniform1f(u.u_fill, FILL_LEVEL);
    const packed = this.#floatBlend && rows < 4096;
    gl.uniform1i(u.u_packed, packed ? 1 : 0);
    gl.drawArraysInstanced(gl.TRIANGLES, 0, width * 18, rows);
    gl.disable(gl.BLEND);

    // Pass 3: colour the levels onto the canvas.
    const tinted = this.#uploadTint(view, width, visible);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, width, height);
    const resolve = this.#programs.get("resolve");
    gl.useProgram(resolve.program);
    const r = resolve.uniforms;
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.#level);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.#ramp);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.#tint);
    gl.uniform1i(r.u_level, 0);
    gl.uniform1i(r.u_ramp, 1);
    gl.uniform1i(r.u_tint, 2);
    gl.uniform1i(r.u_tinted, tinted ? 1 : 0);
    gl.uniform2f(r.u_canvasPx, width, height);
    gl.uniform1f(r.u_firstSlot, view.firstSlot);
    gl.uniform1f(r.u_slotPx, slotPx);
    gl.uniform1i(r.u_row0, first);
    gl.uniform1i(r.u_rows, rows);
    gl.uniform1i(r.u_packed, packed ? 1 : 0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }
}
