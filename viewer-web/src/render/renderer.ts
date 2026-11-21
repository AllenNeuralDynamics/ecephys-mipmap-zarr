import { ByteLru } from "../reader/cache.js";
import { axes, DEFAULT_OVERSAMPLE, planWindow } from "../reader/plan.js";
import { levelKey, type ArrayInfo, type StoreInfo } from "../reader/store.js";
import { decodeTile, type Tile } from "../reader/tiles.js";
import { Programs } from "./programs.js";
import { floatRow, uploadDecode, uploadTile, type TileTexture } from "./textures.js";
import { clampView, visibleSamples, type View } from "./view.js";

/** Anything that serves tiles: the worker client, or a `TileSource` in-process. */
export interface TileProvider {
  getTile(key: string, coords: readonly number[], signal?: AbortSignal): Promise<Tile>;
}

export interface RendererOptions {
  /** Bytes of tile textures kept on the GPU, apart from the coarsest level, which stays resident. */
  textureBytes?: number;
  oversample?: number;
  /** Trace colour, RGBA in 0..1. */
  color?: [number, number, number, number];
  onError?: (err: unknown) => void;
}

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

/** The `q` quantile of the finite `values`, interpolating between neighbours; NaN when none are finite. */
function quantile(values: number[], q: number): number {
  const finite = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (finite.length === 0) return NaN;
  const at = q * (finite.length - 1);
  const below = Math.floor(at);
  const lo = finite[below] ?? NaN;
  return lo + (at - below) * ((finite[Math.min(below + 1, finite.length - 1)] ?? lo) - lo);
}

/** Quantile of a channel's excursion from its centre that reaches the slot's edge at gain 1. */
const EXCURSION_QUANTILE = 0.95;

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
  readonly #inFlight = new Map<string, AbortController>();
  /** Tiles whose fetch failed; not retried, so a missing chunk cannot loop the redraw. */
  readonly #failed = new Set<string>();
  readonly #vao: WebGLVertexArrayObject;
  readonly #coarsest: string;
  #view: View;
  #displayed: number[];
  #centre: Float32Array;
  #unitRange = 1;
  #statsReady = false;
  #frame: number | null = null;
  #idleWaiters: (() => void)[] = [];
  #destroyed = false;

  constructor(
    readonly canvas: HTMLCanvasElement,
    readonly info: StoreInfo,
    readonly tiles: TileProvider,
    readonly options: RendererOptions = {},
  ) {
    const gl = canvas.getContext("webgl2", { antialias: false, premultipliedAlpha: true });
    if (gl === null) throw new Error("WebGL2 is unavailable");
    this.#gl = gl;
    this.#programs = new Programs(gl);
    this.#textures = new ByteLru(options.textureBytes ?? DEFAULT_TEXTURE_BYTES, (_, t) => {
      gl.deleteTexture(t.texture);
    });
    this.#vao = gl.createVertexArray();
    this.#slots = gl.createTexture();
    this.#coarsest = levelKey(info.strides[info.strides.length - 1] ?? 0);
    this.#displayed = Array.from({ length: info.nChannels }, (_, c) => c);
    this.#centre = new Float32Array(info.nChannels);
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

  setView(view: Partial<View>): View {
    this.#view = clampView({ ...this.#view, ...view }, this.#bounds(), this.canvas.width);
    this.#invalidate();
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
      if (this.#isIdle()) for (const resolve of this.#idleWaiters.splice(0)) resolve();
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

  /** Fetch the whole coarsest level once: it centres each channel, sets the amplitude scale, and backs every fallback. */
  #loadStats(): void {
    const array = this.#array(this.#coarsest);
    const all = this.#tilesCovering(
      array,
      Array.from({ length: this.info.nChannels }, (_, c) => c),
      0,
      array.shape[2] ?? 0,
    );
    const decoded: { tile: Tile; values: Float32Array; shape: number[] }[] = [];
    let remaining = all.length;
    for (const need of all) {
      this.#request(need, (tile) => {
        if (tile !== null) decoded.push({ tile, ...decodeTile(tile, array) });
        if (--remaining === 0) this.#finishStats(array, decoded);
      });
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
      const hi = highs[c] ?? [];
      const centre = quantile(
        lo.map((v, b) => 0.5 * (v + (hi[b] ?? NaN))),
        0.5,
      );
      excursions.push(
        quantile(
          lo.map((v, b) => Math.max((hi[b] ?? NaN) - centre, centre - v)),
          EXCURSION_QUANTILE,
        ),
      );
      return centre || 0;
    });
    const unitRange = 2 * quantile(excursions, 0.5);
    this.#unitRange = Number.isFinite(unitRange) && unitRange > 0 ? unitRange : 1;
    this.#statsReady = true;
    this.#uploadSlots();
    this.#invalidate();
  }

  #uploadSlots(): void {
    const texels = new Float32Array(this.info.nChannels * 2).fill(-1);
    for (let c = 0; c < this.info.nChannels; c++) texels[c * 2 + 1] = this.#centre[c] ?? 0;
    this.#displayed.forEach((c, slot) => (texels[c * 2] = slot));
    floatRow(this.#gl, texels, this.info.nChannels, this.#gl.RG32F, this.#gl.RG, this.#slots);
  }

  /** Tiles of `array` covering `channels` and indices `[start, end)`, grouped by channel chunk. */
  #tilesCovering(array: ArrayInfo, channels: readonly number[], start: number, end: number): Needed[] {
    const { channel, time } = axes(array);
    const chunkC = array.chunks[channel] ?? 1;
    const chunkT = array.chunks[time] ?? 1;
    const chunksC = [...new Set(channels.map((c) => Math.floor(c / chunkC)))].sort((a, b) => a - b);
    const needed: Needed[] = [];
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
    gl.viewport(0, 0, width, height);
    gl.clearColor(0, 0, 0, 0);
    gl.disable(gl.SCISSOR_TEST);
    gl.clear(gl.COLOR_BUFFER_BIT);
    if (!this.#statsReady) return;

    const [s0, s1] = visibleSamples(view, width);
    const plan = planWindow(this.info, s0, s1, width, this.options.oversample ?? DEFAULT_OVERSAMPLE);
    const first = Math.floor(view.firstSlot);
    const visible = this.#displayed.slice(first, Math.ceil(view.firstSlot + view.nSlots) + 1);
    const array = this.#array(plan.key);
    const needed = this.#tilesCovering(array, visible, plan.start, plan.end);
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
    const color = this.options.color ?? [0.12, 0.14, 0.2, 1];
    gl.bindVertexArray(this.#vao);
    // Fallbacks first, so a resident tile at the chosen level is never drawn under a coarser one.
    draws.sort((a, b) => (a.scissor === undefined ? 1 : 0) - (b.scissor === undefined ? 1 : 0));
    for (const { array: drawn, texture, scissor } of draws) {
      const envelope = drawn.key !== "raw";
      const { program, uniforms: u } = this.#programs.get(envelope ? "envelope" : "raw", texture.sampler);
      gl.useProgram(program);
      if (scissor === undefined) gl.disable(gl.SCISSOR_TEST);
      else {
        gl.enable(gl.SCISSOR_TEST);
        gl.scissor(scissor[0], 0, Math.max(0, scissor[1] - scissor[0]), height);
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
      gl.uniform1f(u.u_firstSlot, view.firstSlot);
      gl.uniform1f(u.u_slotPx, slotPx);
      gl.uniform1f(u.u_pxPerUnit, pxPerUnit);
      gl.uniform4f(u.u_color, ...color);
      const count = envelope ? texture.length * 6 : texture.length;
      gl.drawArraysInstanced(envelope ? gl.TRIANGLES : gl.LINE_STRIP, 0, count, texture.nChannels);
    }
    gl.disable(gl.SCISSOR_TEST);
  }
}
