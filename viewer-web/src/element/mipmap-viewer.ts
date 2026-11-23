import { EventLayer } from "../events/layer.js";
import type { EventLayerInfo } from "../events/layer.js";
import { groupSlots, packColour } from "../events/raster.js";
import { sampleMapping } from "../events/tint.js";
import { EventTinter } from "../events/tinter.js";
import { ReaderClient } from "../reader/client.js";
import type { StoreInfo } from "../reader/store.js";
import { DEFAULT_PALETTE, Renderer, type TintRequest } from "../render/renderer.js";
import { mutedColour, type TracePalette } from "../render/ramp.js";
import { viewOfRange, visibleSamples, type View } from "../render/view.js";
import ReaderWorker from "../reader/worker.ts?worker&inline";
import { stepMarker, TimeAxis } from "./axis.js";
import { Controls, CONTROLS_STYLE } from "./controls.js";
import { attachInteraction } from "./interaction.js";
import { RasterPanel } from "./raster-panel.js";

export type ChannelOrder = "index" | "depth";
/** `auto` follows the browser's colour scheme, light when it states none. */
export type Theme = "auto" | "light" | "dark";

export interface ViewChangeDetail {
  /** Seconds from `t_start`. */
  timeRange: [number, number];
  gain: number;
  order: ChannelOrder;
  /** Channel indices in view, top to bottom. */
  visibleChannels: number[];
}

export interface ChannelClickDetail {
  channel: number;
  channelId: string;
}

/** How groups outside a non-empty selection draw. */
export type UnselectedMode = "muted" | "hidden";

/** The colours an event group draws with, for pages that show the groups elsewhere. */
export interface GroupStyle {
  id: string;
  colour: string;
  /** Its colour while other groups are selected. */
  muted: string;
}

export interface GroupClickDetail {
  groupId: string;
  /** Whether the group was selected before the click. */
  selected: boolean;
}

export interface SelectionChangeDetail {
  selectedGroups: string[];
}

export interface MarkerClickDetail {
  /** Position of the marker in the markers layer, from 0. */
  index: number;
  /** Seconds from `t_start`. */
  time: number;
  groupId: string;
}

/** Ramp rows the tint's one-byte row index can name, after the ink row. */
const MAX_GROUP_ROWS = 254;
/** Columns, CSS px, a click may miss an event's tint by and still pick its group. */
const CLICK_SLACK_PX = 3;
/** Most events a markers layer may hold; all of them are fetched at once. */
const MAX_MARKERS = 100_000;

let colourContext: CanvasRenderingContext2D | null | undefined;
/** Any CSS colour as `#rrggbb` or `rgba()`, which the ramp parses. */
function normaliseColour(text: string): string {
  colourContext ??= document.createElement("canvas").getContext("2d");
  if (colourContext === null) return text;
  // An unparsable colour leaves fillStyle unchanged, so two different starts expose it.
  colourContext.fillStyle = "#000000";
  colourContext.fillStyle = text;
  const fromBlack = colourContext.fillStyle;
  colourContext.fillStyle = "#ffffff";
  colourContext.fillStyle = text;
  if (colourContext.fillStyle !== fromBlack) throw new Error(`unparsable colour: ${text}`);
  return fromBlack;
}

// Traces run fill → noise → ink with how far an outline is past the noise gate; `--mv-units` colours event groups.
const LIGHT = `--mv-bg: #fbfbfc; --mv-fill: #e8edf5; --mv-noise: #a9b6ca; --mv-ink: #10141c;
  --mv-units: #0a5a9c, #b84a00, #007a58, #a63d82, #8a6a00, #5b4bb0, #7a4a3a; --mv-line-width: 1 1.7;
  --mv-text: #3a3d44; --mv-panel: rgb(251 251 252 / 0.85);
  --mv-track: #eceef2; --mv-thumb: #9aa1b0; --mv-thumb-active: #7d8596; color-scheme: light;`;
const DARK = `--mv-bg: #1c1d20; --mv-fill: #484a4d; --mv-noise: #6d6e72; --mv-ink: #e6e8ec;
  --mv-units: #56b4e9, #e69f00, #5dd39e, #e58fc8, #f0e442, #ff8c61, #c49c94; --mv-line-width: 1.1 1.1;
  --mv-text: #b9bcc3; --mv-panel: rgb(28 29 32 / 0.85);
  --mv-track: #2a2c31; --mv-thumb: #5c616d; --mv-thumb-active: #7a808c; color-scheme: dark;`;

/** Colours are custom properties on the host, so a page can override any of them. */
const STYLE = `
:host { ${LIGHT} }
@media (prefers-color-scheme: dark) { :host(:not([theme="light"])) { ${DARK} } }
:host([theme="dark"]) { ${DARK} }
:host { display: block; position: relative; min-height: 200px; outline: none; color: var(--mv-text);
  background: var(--mv-bg); }
.frame { position: absolute; inset: 0; --raster-h: 0px; --axis-h: 18px; }
.frame.with-raster { --raster-h: var(--mv-raster-height, 25%); }
.frame.no-axis { --axis-h: 0px; }
canvas { position: absolute; left: 0; top: 0; width: calc(100% - 14px); touch-action: none; cursor: grab;
  color: var(--mv-ink); }
canvas:active { cursor: grabbing; }
canvas.traces { height: calc(100% - var(--raster-h) - var(--axis-h)); }
canvas.raster { top: calc(100% - var(--raster-h) - var(--axis-h)); height: var(--raster-h);
  border-top: 1px solid var(--mv-track); box-sizing: border-box; }
canvas.axis { top: calc(100% - var(--axis-h)); height: var(--axis-h); color: var(--mv-text); }
.frame:not(.with-raster) canvas.raster, .frame.no-axis canvas.axis { display: none; }
.note { position: absolute; left: 8px; bottom: calc(var(--axis-h) + 4px); font: 11px system-ui, sans-serif;
  opacity: 0.7; pointer-events: none; }
.frame:not(.with-raster) .note { display: none; }
.track { bottom: calc(var(--raster-h) + var(--axis-h)); }
.labels { position: absolute; inset: 0 14px calc(var(--raster-h) + var(--axis-h)) 0; overflow: hidden;
  pointer-events: none; font: 10px system-ui, sans-serif; }
.labels span { position: absolute; white-space: nowrap; line-height: 1; pointer-events: auto; cursor: pointer; }
.labels span.selected { font-weight: 600; }
.status { position: absolute; left: 8px; top: 8px; font: 12px system-ui, sans-serif; opacity: 0.7; pointer-events: none; }
`;

/** The store has `location` columns to sort channels by depth. */
export function hasDepth(info: StoreInfo): boolean {
  const location = info.properties.location;
  return location !== undefined && (location.shape[1] ?? 0) >= 2 && !Array.isArray(location.values);
}

/** Channels in depth order, shallowest (largest `location` y) first; index order when locations are absent. */
export function depthOrder(info: StoreInfo, channels: readonly number[]): number[] {
  const location = info.properties.location;
  if (location === undefined || !hasDepth(info) || Array.isArray(location.values)) return [...channels];
  const width = location.shape[1] ?? 0;
  const values = location.values;
  const y = (c: number): number => values[c * width + 1] ?? 0;
  const x = (c: number): number => values[c * width] ?? 0;
  return [...channels].sort((a, b) => y(b) - y(a) || x(a) - x(b) || a - b);
}

/**
 * `<mipmap-viewer>`: draws a mipmap store. Set `store` to its URL; `timeRange`, `channels`, `order`, `rows`
 * and `gain` may be set before or after it opens. `order` defaults to depth when the store has channel locations.
 * Fires `viewchange` when a gesture or control change ends and `channelclick` on a click that is not a drag.
 * `controls="none"` hides the channel scroll bar and the toolbar. `theme` is `auto` (the default), `light` or
 * `dark`. `events` is the URL of an event layer whose groups colour the traces they cover.
 *
 * Groups whose footprints overlap get different colours from `--mv-units`, each with a muted variant
 * (`--mv-units-muted`, or derived); `groupColours` overrides any group's. `selectedGroups` draw in full colour at
 * any zoom and the rest muted, or not at all with `unselected="hidden"`. A click on a group's tint or label fires
 * a cancelable `groupclick`, then toggles the group and fires `selectionchange`; Escape and the toolbar clear the
 * selection. `selectionchange` and `viewchange` report the viewer's own UI only: changes made through properties
 * fire nothing. `groupschange` fires whenever the groups' colours change, from any cause.
 *
 * With an event layer open, a raster under the traces (`raster="none"` hides it, `--mv-raster-height` sizes it)
 * draws every event in view at its group's footprint centre, or on its own channel, across the traces' visible
 * channels; a click on it toggles the group. Under it a time axis (`axis="none"` hides it) counts seconds from
 * `t_start` and carries a tick per event of the `markers` layer: `[` and `]` or the toolbar step between them,
 * and a click on one fires a cancelable `markerclick`, then centres the view on it.
 */
export class MipmapViewer extends HTMLElement {
  static readonly observedAttributes = [
    "store",
    "gain",
    "order",
    "controls",
    "theme",
    "events",
    "unselected",
    "markers",
    "raster",
    "axis",
  ];
  static readonly #darkQuery = globalThis.matchMedia("(prefers-color-scheme: dark)");

  readonly #canvas: HTMLCanvasElement;
  readonly #frame: HTMLDivElement;
  readonly #status: HTMLDivElement;
  readonly #labels: HTMLDivElement;
  readonly #note: HTMLDivElement;
  readonly #raster: RasterPanel;
  readonly #axis: TimeAxis;
  #events: string | null = null;
  #eventsOpening: Promise<void> = Promise.resolve();
  #layer: EventLayer | undefined;
  #tinter: EventTinter | undefined;
  #markersUrl: string | null = null;
  #markersOpening: Promise<void> = Promise.resolve();
  #detachPanels: (() => void) | undefined;
  #selectedGroups = new Set<string>();
  /** Colours a page set for groups, by id. */
  #groupColours = new Map<string, string>();
  /** Colours from the host's custom properties; `groups` holds `--mv-units`. */
  #basePalette: TracePalette | undefined;
  /** The ramp's group rows as given to the renderer, for labels. */
  #rampColours: string[] = [];
  #groupStyles: GroupStyle[] = [];
  /** Colour indices for the open layer, kept until the layer or the number of colours changes. */
  #colourIndex: { tinter: EventTinter; n: number; index: Int32Array } | undefined;
  readonly #controls: Controls;
  #worker: Worker | undefined;
  #renderer: Renderer | undefined;
  #info: StoreInfo | undefined;
  #detach: (() => void) | undefined;
  #resize: ResizeObserver | undefined;
  #store: string | null = null;
  #timeRange: [number, number] | null = null;
  #channels: number[] | null = null;
  /** Null follows the store: depth when it has locations. */
  #order: ChannelOrder | null = null;
  #gain = 1;
  #rows: number | null = null;
  #opening: Promise<void> = Promise.resolve();
  readonly #onSchemeChange = (): void => {
    this.#applyTheme();
  };

  constructor() {
    super();
    const root = this.attachShadow({ mode: "open" });
    const style = document.createElement("style");
    style.textContent = CONTROLS_STYLE + STYLE;
    this.#canvas = document.createElement("canvas");
    this.#canvas.className = "traces";
    this.#frame = document.createElement("div");
    this.#frame.className = "frame";
    const rasterCanvas = document.createElement("canvas");
    rasterCanvas.className = "raster";
    const axisCanvas = document.createElement("canvas");
    axisCanvas.className = "axis";
    this.#note = document.createElement("div");
    this.#note.className = "note";
    this.#raster = new RasterPanel(rasterCanvas, {
      onError: (err) => {
        this.#status.textContent = `Raster: ${err instanceof Error ? err.message : String(err)}`;
      },
      onNotice: (text) => {
        if (this.#note.textContent !== text) this.#note.textContent = text;
      },
    });
    this.#axis = new TimeAxis(axisCanvas);
    this.#status = document.createElement("div");
    this.#status.className = "status";
    this.#labels = document.createElement("div");
    this.#labels.className = "labels";
    this.#controls = new Controls({
      setFirstSlot: (firstSlot) => this.#renderer?.setView({ firstSlot }),
      setOrder: (order) => {
        this.order = order;
      },
      settled: () => {
        this.#emitViewChange();
      },
      clearSelection: () => {
        this.#userSetSelection([]);
      },
      stepMarker: (direction) => {
        if (this.jumpToMarker(direction) >= 0) this.#emitViewChange();
      },
    });
    this.#frame.append(
      this.#canvas,
      rasterCanvas,
      axisCanvas,
      this.#labels,
      this.#note,
      this.#status,
      this.#controls.toolbar,
      this.#controls.track,
    );
    root.append(style, this.#frame);
    if (!this.hasAttribute("tabindex")) this.tabIndex = 0;
    this.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && this.#selectedGroups.size > 0) {
        event.preventDefault();
        this.#userSetSelection([]);
      } else if ((event.key === "[" || event.key === "]") && this.#axis.markers !== undefined) {
        event.preventDefault();
        if (this.jumpToMarker(event.key === "]" ? 1 : -1) >= 0) this.#emitViewChange();
      }
    });
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

  /** URL of the event layer drawn over the traces, or null for none. */
  get events(): string | null {
    return this.#events;
  }
  set events(url: string | null) {
    if (url === this.#events) return;
    this.#events = url;
    this.#eventsOpening = this.#openEvents();
  }

  /** URL of the event layer whose events tick the time axis, or null for none. */
  get markers(): string | null {
    return this.#markersUrl;
  }
  set markers(url: string | null) {
    if (url === this.#markersUrl) return;
    this.#markersUrl = url;
    this.#markersOpening = this.#openMarkers();
  }

  /**
   * Centre the view on the next marker past its centre, or the previous one with -1, keeping the zoom; returns
   * the marker's position in the layer from 0, or -1 when there is none that way.
   */
  jumpToMarker(direction: 1 | -1): number {
    const markers = this.#axis.markers;
    const view = this.#renderer?.view;
    if (markers === undefined || view === undefined) return -1;
    const centre = view.t0 + (view.samplesPerPx * this.#canvas.width) / 2;
    const i = stepMarker(markers.sample, centre, direction, view.samplesPerPx);
    if (i >= 0) this.#centreOn(markers.sample[i] ?? centre);
    return i;
  }

  /** Ids of the event groups the open layer holds. */
  get eventGroups(): readonly string[] {
    return this.#tinter?.groupIds ?? [];
  }

  /** The open event layer's metadata, with its groups' properties; undefined until it opens. */
  get eventLayer(): EventLayerInfo | undefined {
    return this.#tinter?.info;
  }

  /** Group ids drawn in full colour at any zoom, the rest muted or hidden; empty colours every group alike. */
  get selectedGroups(): string[] {
    return [...this.#selectedGroups];
  }
  set selectedGroups(ids: Iterable<string>) {
    this.#selectedGroups = new Set(ids);
    this.#applySelection();
  }

  /** Select or deselect one group, toggling without `selected`; returns whether it is now selected. */
  toggleGroup(id: string, selected?: boolean): boolean {
    const now = selected ?? !this.#selectedGroups.has(id);
    if (now) this.#selectedGroups.add(id);
    else this.#selectedGroups.delete(id);
    this.#applySelection();
    return now;
  }

  /** How groups outside a non-empty selection draw, from the `unselected` attribute. */
  get unselected(): UnselectedMode {
    return this.getAttribute("unselected") === "hidden" ? "hidden" : "muted";
  }
  set unselected(mode: UnselectedMode) {
    this.setAttribute("unselected", mode);
  }

  /** Colours set for groups by id, any CSS colour; others take one from `--mv-units`. */
  get groupColours(): Record<string, string> {
    return Object.fromEntries(this.#groupColours);
  }
  set groupColours(colours: Record<string, string> | Map<string, string>) {
    const entries = colours instanceof Map ? [...colours] : Object.entries(colours);
    this.#groupColours = new Map(entries.map(([id, c]) => [id, normaliseColour(c)]));
    this.#recolour();
  }

  /** Set one group's colour, or with null return it to the automatic one. */
  setGroupColour(id: string, colour: string | null): void {
    if (colour === null) this.#groupColours.delete(id);
    else this.#groupColours.set(id, normaliseColour(colour));
    this.#recolour();
  }

  /** Each group's colours, in the layer's group order. */
  get groupStyles(): readonly GroupStyle[] {
    return this.#groupStyles;
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
    return this.#order ?? (this.#info !== undefined && hasDepth(this.#info) ? "depth" : "index");
  }
  /** Null returns to the store's default. */
  set order(order: ChannelOrder | null) {
    this.#order = order === null ? null : order === "depth" ? "depth" : "index";
    this.#applyChannels();
  }

  /** Channel rows in view; null shows every chosen channel, or 64 of them when none were chosen. */
  get rows(): number | null {
    return this.#renderer?.view.nSlots ?? this.#rows;
  }
  set rows(rows: number | null) {
    this.#rows = rows !== null && rows >= 1 ? Math.round(rows) : null;
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
    else if (name === "order") this.order = value as ChannelOrder | null;
    else if (name === "theme") this.#applyTheme();
    else if (name === "events") this.events = value;
    else if (name === "unselected") this.#applySelection();
    else if (name === "markers") this.markers = value;
    else if (name === "raster" || name === "axis") this.#applyPanels();
  }

  get theme(): Theme {
    const value = this.getAttribute("theme");
    return value === "light" || value === "dark" ? value : "auto";
  }
  set theme(theme: Theme) {
    if (theme === "auto") this.removeAttribute("theme");
    else this.setAttribute("theme", theme);
  }

  connectedCallback(): void {
    this.#resize = new ResizeObserver(() => this.#renderer?.resize());
    // The canvas, not the host: hiding the controls widens it without resizing the host.
    this.#resize.observe(this.#canvas);
    MipmapViewer.#darkQuery.addEventListener("change", this.#onSchemeChange);
    this.#applyTheme();
  }

  disconnectedCallback(): void {
    this.#resize?.disconnect();
    MipmapViewer.#darkQuery.removeEventListener("change", this.#onSchemeChange);
    this.#close();
  }

  /** Resolves once the view is fully drawn; for tests and screenshots. */
  async idle(): Promise<void> {
    await this.#opening;
    await this.#eventsOpening;
    await this.#markersOpening;
    await this.#renderer?.idle();
    // Events that arrive after a frame redraw it.
    await this.#tinter?.idle();
    await this.#raster.idle();
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
    this.#tinter = undefined;
    this.#layer = undefined;
    this.#raster.setLayer(undefined, { scale: 1, offset: 0 }, 1);
    this.#axis.setMarkers(undefined);
    this.#detachPanels?.();
    this.#detachPanels = undefined;
    this.#labels.replaceChildren();
    this.#detach?.();
    this.#detach = undefined;
    this.#renderer?.destroy();
    this.#renderer = undefined;
    this.#worker?.terminate();
    this.#worker = undefined;
    this.#info = undefined;
    this.#controls.hide();
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
        onViewChange: () => {
          this.#syncControls();
        },
        palette: this.#palette(),
        onDraw: () => {
          this.#drawLabels();
          this.#drawPanels();
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
        clicked: (x, y) => {
          if (!this.#clickGroup(x, y)) this.#emitChannelClick(y);
        },
      });
      this.#detachPanels = this.#attachPanels();
      this.#status.textContent = info.rawError === null ? "" : `Raw unavailable: ${info.rawError}`;
      this.#eventsOpening = this.#openEvents();
      this.#markersOpening = this.#openMarkers();
    } catch (err) {
      if (this.#worker === worker) this.#status.textContent = err instanceof Error ? err.message : String(err);
    }
  }

  /** The trace colours the host's custom properties set, with the light defaults for any left empty. */
  #palette(): TracePalette {
    const style = getComputedStyle(this);
    const get = (name: string, fallback: string): string =>
      normaliseColour(style.getPropertyValue(name).trim() || fallback);
    const groups = this.#colourList("--mv-units");
    const widths = style
      .getPropertyValue("--mv-line-width")
      .split(/\s+/)
      .map(Number)
      .filter((w) => w > 0);
    const [thin = DEFAULT_PALETTE.lineWidth[0], thick = thin] = widths;
    return {
      background: get("--mv-bg", DEFAULT_PALETTE.background),
      fill: get("--mv-fill", DEFAULT_PALETTE.fill),
      noise: get("--mv-noise", DEFAULT_PALETTE.noise),
      ink: get("--mv-ink", DEFAULT_PALETTE.ink),
      groups,
      lineWidth: [thin, thick],
    };
  }

  /** A comma-separated colour list from a custom property; commas inside `rgb()` do not split it. */
  #colourList(name: string): string[] {
    return getComputedStyle(this)
      .getPropertyValue(name)
      .split(/,(?![^(]*\))/)
      .map((c) => c.trim())
      .filter(Boolean)
      .map(normaliseColour);
  }

  #applyTheme(): void {
    try {
      this.#basePalette = this.#palette();
      this.#recolour();
      this.#colourMarkers();
    } catch (err) {
      this.#status.textContent = err instanceof Error ? err.message : String(err);
    }
  }

  /**
   * Build the ramp's group rows and point each group at its pair: the units, then their muted variants, then a
   * colour and muted pair per distinct override.
   */
  #recolour(): void {
    const base = this.#basePalette;
    if (base === undefined) return;
    const units = base.groups.slice(0, MAX_GROUP_ROWS / 2);
    const n = units.length;
    const mutedCss = this.#colourList("--mv-units-muted");
    const rows = [...units, ...units.map((c, i) => mutedCss[i] ?? mutedColour(c, base.noise))];
    const tinter = this.#tinter;
    if (tinter !== undefined) {
      if (this.#colourIndex?.tinter !== tinter || this.#colourIndex.n !== n)
        this.#colourIndex = { tinter, n, index: tinter.assignColours(n) };
      const { index } = this.#colourIndex;
      const ids = tinter.groupIds;
      const row = new Int32Array(ids.length).fill(-1);
      const mutedRow = new Int32Array(ids.length).fill(-1);
      const overrideRow = new Map<string, number>();
      ids.forEach((id, g) => {
        const own = this.#groupColours.get(id);
        if (own !== undefined) {
          let r = overrideRow.get(own);
          if (r === undefined && rows.length + 2 <= MAX_GROUP_ROWS) {
            r = rows.length;
            rows.push(own, mutedColour(own, base.noise));
            overrideRow.set(own, r);
          }
          if (r !== undefined) {
            row[g] = r;
            mutedRow[g] = r + 1;
            return;
          }
        }
        const c = index[g] ?? -1;
        if (c >= 0) {
          row[g] = c;
          mutedRow[g] = n + c;
        }
      });
      tinter.setRows(row, mutedRow);
      this.#groupStyles = ids.map((id, g) => ({
        id,
        colour: rows[row[g] ?? -1] ?? "",
        muted: rows[mutedRow[g] ?? -1] ?? "",
      }));
    } else {
      this.#groupStyles = [];
    }
    this.#rampColours = rows;
    this.#renderer?.setPalette({ ...base, groups: rows });
    this.#styleRaster();
    this.dispatchEvent(new CustomEvent("groupschange", { detail: this.#groupStyles, bubbles: true, composed: true }));
  }

  #applyTimeRange(): void {
    const renderer = this.#renderer;
    if (renderer === undefined || this.#info === undefined || this.#timeRange === null) return;
    const fs = this.#info.samplingFrequency;
    renderer.setView(viewOfRange(renderer.view, this.#timeRange[0] * fs, this.#timeRange[1] * fs, this.#canvas.width));
  }

  /** Open `events` for the current store; a later call supersedes an earlier one still opening. */
  async #openEvents(): Promise<void> {
    const url = this.#events;
    const renderer = this.#renderer;
    const info = this.#info;
    this.#tinter = undefined;
    this.#layer = undefined;
    renderer?.setTinter(undefined);
    this.#raster.setLayer(undefined, { scale: 1, offset: 0 }, 1);
    this.#applyPanels();
    this.#labels.replaceChildren();
    if (this.#groupStyles.length > 0) this.#recolour();
    if (url === null || url === "" || renderer === undefined || info === undefined) return;
    try {
      const layer = await EventLayer.open(new URL(url, document.baseURI).href);
      if (this.#events !== url || this.#renderer !== renderer) return;
      const tinter = new EventTinter(layer, info, {
        onLoaded: () => {
          renderer.redraw();
        },
        onError: (err) => {
          this.#status.textContent = `Events: ${err instanceof Error ? err.message : String(err)}`;
        },
      });
      this.#tinter = tinter;
      this.#layer = layer;
      this.#raster.setLayer(layer, sampleMapping(layer.info, info), info.samplingFrequency);
      this.#recolour();
      this.#applySelection();
      renderer.setTinter(tinter.tint);
    } catch (err) {
      if (this.#events === url)
        this.#status.textContent = `Events: ${err instanceof Error ? err.message : String(err)}`;
    }
  }

  #applySelection(): void {
    this.#controls.setSelection(this.#selectedGroups.size);
    const tinter = this.#tinter;
    if (tinter === undefined) return;
    const wanted = this.#selectedGroups;
    tinter.setSelected(
      tinter.groupIds.flatMap((id, g) => (wanted.has(id) ? [g] : [])),
      this.unselected === "hidden",
    );
    this.#styleRaster();
    this.#renderer?.redraw();
  }

  /** A selection change from the viewer's own UI. */
  #userSetSelection(ids: Iterable<string>): void {
    this.selectedGroups = ids;
    const detail: SelectionChangeDetail = { selectedGroups: this.selectedGroups };
    this.dispatchEvent(new CustomEvent("selectionchange", { detail, bubbles: true, composed: true }));
  }

  /** Fire `groupclick` for group `g` and, unless a listener cancels it, toggle the group. */
  #userClickGroup(g: number): void {
    const id = this.#tinter?.groupIds[g];
    if (id === undefined) return;
    const detail: GroupClickDetail = { groupId: id, selected: this.#selectedGroups.has(id) };
    const event = new CustomEvent("groupclick", { detail, bubbles: true, composed: true, cancelable: true });
    if (!this.dispatchEvent(event)) return;
    const next = new Set(this.#selectedGroups);
    if (detail.selected) next.delete(id);
    else next.add(id);
    this.#userSetSelection(next);
  }

  /** What the tinter sees: the visible rows from the first partly visible slot, and the slot it starts at. */
  #tintRequest(): { request: TintRequest; first: number } | undefined {
    const renderer = this.#renderer;
    if (renderer === undefined) return undefined;
    const view = renderer.view;
    const first = Math.floor(view.firstSlot);
    const channels = renderer.displayed.slice(first, Math.ceil(view.firstSlot + view.nSlots) + 1);
    return {
      request: { t0: view.t0, samplesPerPx: view.samplesPerPx, columns: this.#canvas.width, channels },
      first,
    };
  }

  /** Toggle the group tinted under a click, if any; device pixels. */
  #clickGroup(xPx: number, yPx: number): boolean {
    const tinter = this.#tinter;
    const view = this.#renderer?.view;
    const at = this.#tintRequest();
    if (tinter === undefined || view === undefined || at === undefined) return false;
    const row = Math.floor(view.firstSlot + (yPx / this.#canvas.height) * view.nSlots) - at.first;
    const slack = CLICK_SLACK_PX * (this.#canvas.width / Math.max(1, this.#canvas.clientWidth));
    const g = tinter.groupAt(at.request, Math.floor(xPx), row, slack);
    if (g < 0) return false;
    this.#userClickGroup(g);
    return true;
  }

  /** Group labels under each event, when zoomed in far enough for them to fit. */
  #drawLabels(): void {
    const renderer = this.#renderer;
    const tinter = this.#tinter;
    if (renderer === undefined || tinter === undefined) {
      if (this.#labels.childElementCount > 0) this.#labels.replaceChildren();
      return;
    }
    const view = renderer.view;
    const at = this.#tintRequest();
    if (at === undefined) return;
    const { request, first } = at;
    const labels = tinter.labels(request);
    if (labels.length === 0 && this.#labels.childElementCount === 0) return;
    const ratio = request.columns / Math.max(1, this.#canvas.clientWidth);
    const slotCss = this.#canvas.clientHeight / view.nSlots;
    this.#labels.replaceChildren(
      ...labels.map(({ x, row, group, text, rampRow, selected }) => {
        const span = document.createElement("span");
        span.textContent = text;
        span.className = selected ? "selected" : "";
        span.style.left = `${String(x / ratio + 3)}px`;
        span.style.top = `${String((first + row - view.firstSlot + 1) * slotCss - 11)}px`;
        span.style.color = this.#rampColours[rampRow] ?? "";
        span.addEventListener("click", () => {
          this.#userClickGroup(group);
        });
        return span;
      }),
    );
  }

  /** Show the raster while an event layer is set, and the axis, unless their attributes say `none`. */
  #applyPanels(): void {
    const raster = this.#events !== null && this.#events !== "" && this.getAttribute("raster") !== "none";
    this.#frame.classList.toggle("with-raster", raster);
    this.#frame.classList.toggle("no-axis", this.getAttribute("axis") === "none");
  }

  #drawPanels(): void {
    const view = this.#renderer?.view;
    const info = this.#info;
    if (view === undefined || info === undefined) return;
    this.#raster.draw(view, this.#canvas.width);
    this.#axis.draw(view, this.#canvas.width, info.samplingFrequency);
  }

  /** Pan and zoom from the raster and the axis as from the traces; clicks pick a group or a marker. */
  #attachPanels(): () => void {
    const common = {
      element: this as HTMLElement,
      view: () => this.#renderer?.view,
      setView: (view: Partial<View>) => this.#renderer?.setView(view),
      settled: () => {
        this.#emitViewChange();
      },
      keys: false,
    };
    const slack = (canvas: HTMLCanvasElement): number =>
      CLICK_SLACK_PX * (canvas.width / Math.max(1, canvas.clientWidth));
    const detachRaster = attachInteraction({
      ...common,
      canvas: this.#raster.canvas,
      clicked: (x, y) => {
        const g = this.#raster.groupAt(x, y, slack(this.#raster.canvas));
        if (g >= 0) this.#userClickGroup(g);
      },
    });
    const detachAxis = attachInteraction({
      ...common,
      canvas: this.#axis.canvas,
      clicked: (x) => {
        const i = this.#axis.markerAt(x, slack(this.#axis.canvas));
        if (i >= 0) this.#userClickMarker(i);
      },
    });
    return () => {
      detachRaster();
      detachAxis();
    };
  }

  /** The raster's colours and rows for the current groups, selection and displayed channels. */
  #styleRaster(): void {
    const layer = this.#layer;
    const renderer = this.#renderer;
    const info = this.#info;
    const base = this.#basePalette;
    if (layer === undefined || renderer === undefined || info === undefined || base === undefined) return;
    const slotOf = new Map(renderer.displayed.map((c, slot) => [c, slot]));
    const byId = new Map(info.channelIds.map((id, c) => [id, c]));
    const channelSlot = Int32Array.from(layer.info.channelIds, (id) => slotOf.get(byId.get(id) ?? -1) ?? -1);
    const anySelected = this.#selectedGroups.size > 0;
    const hidden = this.unselected === "hidden";
    const ids = layer.info.groups.ids;
    const colour = new Int32Array(ids.length).fill(-1);
    const top = new Uint8Array(ids.length);
    ids.forEach((id, g) => {
      const style = this.#groupStyles[g];
      if (style === undefined) return;
      const selected = this.#selectedGroups.has(id);
      if (anySelected && !selected && hidden) return;
      colour[g] = packColour(anySelected && !selected ? style.muted : style.colour);
      top[g] = selected ? 1 : 0;
    });
    this.#raster.setStyle({
      groupSlot: groupSlots(layer.info.groups, channelSlot),
      channelSlot,
      colour,
      top,
      background: Math.max(0, packColour(base.background)),
    });
  }

  /** Open `markers` for the current store; a later call supersedes an earlier one still opening. */
  async #openMarkers(): Promise<void> {
    const url = this.#markersUrl;
    const info = this.#info;
    this.#axis.setMarkers(undefined);
    this.#controls.setMarkers(0);
    if (url === null || url === "" || info === undefined) return;
    try {
      const layer = await EventLayer.open(new URL(url, document.baseURI).href);
      if (layer.info.nEvents > MAX_MARKERS)
        throw new Error(`${String(layer.info.nEvents)} events, more than the ${String(MAX_MARKERS)} markers can hold`);
      const window = await layer.window(-Infinity, Infinity);
      if (this.#markersUrl !== url || this.#info !== info) return;
      const { scale, offset } = sampleMapping(layer.info, info);
      this.#axis.setMarkers({
        sample: window.sampleIndex.map((s) => s * scale + offset),
        group: window.group,
        ids: layer.info.groups.ids,
        colours: [],
      });
      this.#colourMarkers();
      this.#controls.setMarkers(window.sampleIndex.length);
    } catch (err) {
      if (this.#markersUrl === url)
        this.#status.textContent = `Markers: ${err instanceof Error ? err.message : String(err)}`;
    }
  }

  /** Marker groups take the units' colours in turn. */
  #colourMarkers(): void {
    const markers = this.#axis.markers;
    const units = this.#basePalette?.groups ?? [];
    if (markers === undefined || units.length === 0) return;
    this.#axis.setMarkers({ ...markers, colours: markers.ids.map((_, g) => units[g % units.length] ?? "") });
  }

  /** Fire `markerclick` for marker `i` and, unless a listener cancels it, centre the view on it. */
  #userClickMarker(i: number): void {
    const markers = this.#axis.markers;
    const info = this.#info;
    const sample = markers?.sample[i];
    if (markers === undefined || info === undefined || sample === undefined) return;
    const detail: MarkerClickDetail = {
      index: i,
      time: sample / info.samplingFrequency,
      groupId: markers.ids[markers.group[i] ?? -1] ?? "",
    };
    const event = new CustomEvent("markerclick", { detail, bubbles: true, composed: true, cancelable: true });
    if (!this.dispatchEvent(event)) return;
    this.#centreOn(sample);
    this.#emitViewChange();
  }

  /** Centre the view on store sample `sample`, keeping the zoom. */
  #centreOn(sample: number): void {
    const view = this.#renderer?.view;
    if (view === undefined) return;
    this.#renderer?.setView({ t0: sample - (view.samplesPerPx * this.#canvas.width) / 2 });
  }

  #applyChannels(): void {
    const renderer = this.#renderer;
    const info = this.#info;
    if (renderer === undefined || info === undefined) return;
    const chosen = this.#channels ?? Array.from({ length: info.nChannels }, (_, c) => c);
    renderer.setDisplayed(this.order === "depth" ? depthOrder(info, chosen) : chosen);
    if (this.#channels !== null || this.#rows !== null)
      renderer.setView({ firstSlot: 0, nSlots: this.#rows ?? renderer.displayed.length });
    this.#styleRaster();
  }

  #syncControls(): void {
    const renderer = this.#renderer;
    const info = this.#info;
    if (renderer === undefined || info === undefined) return;
    const { view, displayed } = renderer;
    const id = (slot: number): string => {
      const channel = displayed[Math.min(displayed.length - 1, Math.max(0, slot))];
      return channel === undefined ? "" : (info.channelIds[channel] ?? String(channel + 1));
    };
    this.#controls.update({
      firstSlot: view.firstSlot,
      nSlots: view.nSlots,
      nDisplayed: displayed.length,
      order: this.order,
      canDepth: hasDepth(info),
      topId: id(Math.floor(view.firstSlot)),
      bottomId: id(Math.ceil(view.firstSlot + view.nSlots) - 1),
    });
  }

  #visibleChannels(view: View): number[] {
    const displayed = this.#renderer?.displayed ?? [];
    return displayed.slice(Math.floor(view.firstSlot), Math.ceil(view.firstSlot + view.nSlots));
  }

  #emitViewChange(): void {
    const view = this.#renderer?.view;
    const timeRange = this.timeRange;
    if (view === undefined || timeRange === null) return;
    const detail: ViewChangeDetail = {
      timeRange,
      gain: view.gain,
      order: this.order,
      visibleChannels: this.#visibleChannels(view),
    };
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
