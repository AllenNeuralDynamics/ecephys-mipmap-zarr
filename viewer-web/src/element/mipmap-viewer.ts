import { ReaderClient } from "../reader/client.js";
import type { StoreInfo } from "../reader/store.js";
import { Renderer } from "../render/renderer.js";
import { viewOfRange, visibleSamples, type View } from "../render/view.js";
import ReaderWorker from "../reader/worker.ts?worker&inline";
import { attachInteraction } from "./interaction.js";

export type ChannelOrder = "index" | "depth";

export interface ViewChangeDetail {
  /** Seconds from `t_start`. */
  timeRange: [number, number];
  gain: number;
  /** Channel indices in view, top to bottom. */
  visibleChannels: number[];
}

export interface ChannelClickDetail {
  channel: number;
  channelId: string;
}

const STYLE = `
:host { display: block; position: relative; min-height: 200px; outline: none; color: #1f2433; }
canvas { position: absolute; inset: 0; width: 100%; height: 100%; touch-action: none; cursor: grab; }
canvas:active { cursor: grabbing; }
.status { position: absolute; left: 8px; top: 8px; font: 12px system-ui, sans-serif; opacity: 0.7; pointer-events: none; }
`;

/** Channels in depth order, shallowest (largest `location` y) first; index order when locations are absent. */
export function depthOrder(info: StoreInfo, channels: readonly number[]): number[] {
  const location = info.properties.location;
  const width = location?.shape[1] ?? 0;
  if (location === undefined || width < 2 || Array.isArray(location.values)) return [...channels];
  const values = location.values;
  const y = (c: number): number => values[c * width + 1] ?? 0;
  const x = (c: number): number => values[c * width] ?? 0;
  return [...channels].sort((a, b) => y(b) - y(a) || x(a) - x(b) || a - b);
}

/**
 * `<mipmap-viewer>`: draws a mipmap store. Set `store` to its URL; `timeRange`, `channels`, `order` and
 * `gain` may be set before or after it opens. Fires `viewchange` when a gesture ends and `channelclick` on a
 * click that is not a drag.
 */
export class MipmapViewer extends HTMLElement {
  static readonly observedAttributes = ["store", "gain", "order", "controls"];

  readonly #canvas: HTMLCanvasElement;
  readonly #status: HTMLDivElement;
  #worker: Worker | undefined;
  #renderer: Renderer | undefined;
  #info: StoreInfo | undefined;
  #detach: (() => void) | undefined;
  #resize: ResizeObserver | undefined;
  #store: string | null = null;
  #timeRange: [number, number] | null = null;
  #channels: number[] | null = null;
  #order: ChannelOrder = "index";
  #gain = 1;
  #opening: Promise<void> = Promise.resolve();

  constructor() {
    super();
    const root = this.attachShadow({ mode: "open" });
    const style = document.createElement("style");
    style.textContent = STYLE;
    this.#canvas = document.createElement("canvas");
    this.#status = document.createElement("div");
    this.#status.className = "status";
    root.append(style, this.#canvas, this.#status);
    if (!this.hasAttribute("tabindex")) this.tabIndex = 0;
  }

  get store(): string | null {
    return this.#store;
  }
  set store(url: string | null) {
    if (url === this.#store) return;
    this.#store = url;
    this.#opening = this.#open(url);
  }

  /** Resolves when the current `store` has opened (or failed to). */
  get ready(): Promise<void> {
    return this.#opening;
  }

  get info(): StoreInfo | undefined {
    return this.#info;
  }

  get timeRange(): [number, number] | null {
    const view = this.#renderer?.view;
    if (view === undefined || this.#info === undefined) return this.#timeRange;
    const [s0, s1] = visibleSamples(view, this.#canvas.width);
    return [s0 / this.#info.samplingFrequency, s1 / this.#info.samplingFrequency];
  }
  set timeRange(range: [number, number] | null) {
    this.#timeRange = range;
    this.#applyTimeRange();
  }

  get channels(): number[] | null {
    return this.#channels;
  }
  set channels(channels: number[] | null) {
    this.#channels = channels === null ? null : [...channels];
    this.#applyChannels();
  }

  get order(): ChannelOrder {
    return this.#order;
  }
  set order(order: ChannelOrder) {
    this.#order = order === "depth" ? "depth" : "index";
    this.#applyChannels();
  }

  get gain(): number {
    return this.#renderer?.view.gain ?? this.#gain;
  }
  set gain(gain: number) {
    if (!(gain > 0)) return;
    this.#gain = gain;
    this.#renderer?.setView({ gain });
  }

  attributeChangedCallback(name: string, _old: string | null, value: string | null): void {
    if (name === "store") this.store = value;
    else if (name === "gain" && value !== null) this.gain = Number(value);
    else if (name === "order") this.order = value as ChannelOrder;
  }

  connectedCallback(): void {
    this.#resize = new ResizeObserver(() => this.#renderer?.resize());
    this.#resize.observe(this);
  }

  disconnectedCallback(): void {
    this.#resize?.disconnect();
    this.#close();
  }

  /** Resolves once the view is fully drawn; for tests and screenshots. */
  async idle(): Promise<void> {
    await this.#opening;
    await this.#renderer?.idle();
  }

  /** Draw and read back the canvas, bottom row first, in device pixels. */
  snapshot(): { width: number; height: number; data: Uint8Array } | undefined {
    return this.#renderer?.readPixels();
  }

  /** The automatic amplitude scale: value units per channel slot at gain 1. */
  get unitRange(): number | undefined {
    return this.#renderer?.unitRange;
  }

  /** The renderer's layout and current plan inputs, for tests. */
  layout(): (ReturnType<Renderer["layout"]> & { view: View; displayed: readonly number[] }) | undefined {
    const renderer = this.#renderer;
    if (renderer === undefined) return undefined;
    return { ...renderer.layout(), view: renderer.view, displayed: renderer.displayed };
  }

  get view(): View | undefined {
    return this.#renderer?.view;
  }

  #close(): void {
    this.#detach?.();
    this.#detach = undefined;
    this.#renderer?.destroy();
    this.#renderer = undefined;
    this.#worker?.terminate();
    this.#worker = undefined;
    this.#info = undefined;
  }

  async #open(url: string | null): Promise<void> {
    this.#close();
    this.#status.textContent = "";
    if (url === null || url === "") return;
    const worker = new ReaderWorker();
    this.#worker = worker;
    const client = new ReaderClient(worker);
    this.#status.textContent = "Opening…";
    try {
      const info = await client.open(new URL(url, document.baseURI).href);
      if (this.#worker !== worker) return;
      this.#info = info;
      this.#renderer = new Renderer(this.#canvas, info, client, {
        onError: (err) => {
          this.#status.textContent = err instanceof Error ? err.message : String(err);
        },
      });
      this.#renderer.setView({ gain: this.#gain });
      this.#applyChannels();
      this.#applyTimeRange();
      this.#detach = attachInteraction({
        element: this,
        canvas: this.#canvas,
        view: () => this.#renderer?.view,
        setView: (view) => this.#renderer?.setView(view),
        settled: () => {
          this.#emitViewChange();
        },
        clicked: (_x, y) => {
          this.#emitChannelClick(y);
        },
      });
      this.#status.textContent = info.rawError === null ? "" : `Raw unavailable: ${info.rawError}`;
    } catch (err) {
      if (this.#worker === worker) this.#status.textContent = err instanceof Error ? err.message : String(err);
    }
  }

  #applyTimeRange(): void {
    const renderer = this.#renderer;
    if (renderer === undefined || this.#info === undefined || this.#timeRange === null) return;
    const fs = this.#info.samplingFrequency;
    renderer.setView(viewOfRange(renderer.view, this.#timeRange[0] * fs, this.#timeRange[1] * fs, this.#canvas.width));
  }

  #applyChannels(): void {
    const renderer = this.#renderer;
    const info = this.#info;
    if (renderer === undefined || info === undefined) return;
    const chosen = this.#channels ?? Array.from({ length: info.nChannels }, (_, c) => c);
    renderer.setDisplayed(this.#order === "depth" ? depthOrder(info, chosen) : chosen);
    if (this.#channels !== null) renderer.setView({ firstSlot: 0, nSlots: renderer.displayed.length });
  }

  #visibleChannels(view: View): number[] {
    const displayed = this.#renderer?.displayed ?? [];
    return displayed.slice(Math.floor(view.firstSlot), Math.ceil(view.firstSlot + view.nSlots));
  }

  #emitViewChange(): void {
    const view = this.#renderer?.view;
    const timeRange = this.timeRange;
    if (view === undefined || timeRange === null) return;
    const detail: ViewChangeDetail = { timeRange, gain: view.gain, visibleChannels: this.#visibleChannels(view) };
    this.dispatchEvent(new CustomEvent("viewchange", { detail, bubbles: true, composed: true }));
  }

  #emitChannelClick(yPx: number): void {
    const view = this.#renderer?.view;
    if (view === undefined || this.#info === undefined) return;
    const slot = Math.floor(view.firstSlot + (yPx / this.#canvas.height) * view.nSlots);
    const channel = this.#renderer?.displayed[slot];
    if (channel === undefined) return;
    const detail: ChannelClickDetail = { channel, channelId: this.#info.channelIds[channel] ?? String(channel + 1) };
    this.dispatchEvent(new CustomEvent("channelclick", { detail, bubbles: true, composed: true }));
  }
}

/** Register `<mipmap-viewer>` unless something already has. */
export function defineMipmapViewer(name = "mipmap-viewer"): void {
  if (customElements.get(name) === undefined) customElements.define(name, MipmapViewer);
}

declare global {
  interface HTMLElementTagNameMap {
    "mipmap-viewer": MipmapViewer;
  }
}
