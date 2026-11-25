import { GAIN_STEP, SETTLE_MS, wheelSlots } from "./interaction.js";
import type { ChannelOrder } from "./mipmap-viewer.js";

export interface ControlsTarget {
  setFirstSlot(firstSlot: number): void;
  setOrder(order: ChannelOrder): void;
  setGain(gain: number): void;
  /** Show this many rows, keeping the top row where it can. */
  setRows(rows: number): void;
  /** A control has finished changing the view. */
  settled(): void;
  clearSelection(): void;
  /** Centre the view on the next marker, or the previous one with -1. */
  stepMarker(direction: 1 | -1): void;
}

export interface ControlsState {
  firstSlot: number;
  nSlots: number;
  nDisplayed: number;
  order: ChannelOrder;
  gain: number;
  /** The store has channel locations to sort by. */
  canDepth: boolean;
  /** Id of the channel in a displayed slot. */
  idAt: (slot: number) => string;
}

export const CONTROLS_STYLE = `
.toolbar { position: absolute; right: calc(22px + var(--ids-w, 0px)); top: 6px; display: flex; gap: 8px; align-items: center;
  font: 12px system-ui, sans-serif; padding: 2px 6px; border-radius: 4px; background: var(--mv-panel); }
.toolbar select, .toolbar button { font: inherit; }
.toolbar button[hidden], .toolbar span[hidden] { display: none; }
.markers, .gain { display: flex; gap: 2px; align-items: center; }
.gain output { min-width: 3.5em; text-align: center; cursor: pointer; font-variant-numeric: tabular-nums; }
.track { position: absolute; right: 0; top: 0; bottom: 0; width: 14px; background: var(--mv-track);
  touch-action: none; }
.thumb { position: absolute; left: 2px; right: 2px; border-radius: 5px; background: var(--mv-thumb); cursor: grab;
  height: var(--thumb); top: calc((100% - var(--thumb)) * var(--at)); }
.thumb:active { cursor: grabbing; background: var(--mv-thumb-active); }
.toolbar[hidden], .track[hidden], :host([controls="none"]) .toolbar, :host([controls="none"]) .track { display: none; }
:host([controls="none"]) canvas { width: 100%; }
.ids { position: absolute; left: 0; right: 14px; top: 0; overflow: hidden; pointer-events: none;
  font: 10px system-ui, sans-serif; }
.ids span { position: absolute; right: 2px; transform: translateY(-50%); line-height: 1; padding: 0 2px;
  border-radius: 2px; background: var(--mv-panel); white-space: nowrap; }
.ids[hidden], :host([controls="none"]) .ids { display: none; }
`;

/** Row counts the rows menu offers, besides every displayed row and the current count. */
const ROW_CHOICES = [8, 16, 32, 64, 128, 256];

/** Thumb height floor, CSS px, so a long channel list keeps a grabbable thumb. */
const MIN_THUMB_PX = 16;
/** Least spacing, CSS px, between channel ids down the traces' right edge; thinner rows label every k-th. */
const MIN_ID_SPACING_PX = 13;

/**
 * The channel scroll bar beside the traces, each row's channel id down their right edge, and the toolbar over
 * them: which channels are in view, and the channel order. Hidden by `controls="none"`.
 */
export class Controls {
  readonly toolbar: HTMLDivElement;
  readonly track: HTMLDivElement;
  readonly ids: HTMLDivElement;
  readonly #thumb: HTMLDivElement;
  readonly #range: HTMLSpanElement;
  readonly #order: HTMLSelectElement;
  readonly #gain: HTMLOutputElement;
  readonly #rows: HTMLSelectElement;
  readonly #selection: HTMLButtonElement;
  readonly #markers: HTMLSpanElement;
  #state: ControlsState | undefined;

  constructor(readonly target: ControlsTarget) {
    this.toolbar = document.createElement("div");
    this.toolbar.className = "toolbar";
    this.#range = document.createElement("span");
    this.#range.className = "range";
    const label = document.createElement("label");
    label.textContent = "Order ";
    this.#order = document.createElement("select");
    this.#order.className = "order";
    for (const [value, text] of [
      ["index", "Store"],
      ["depth", "Depth"],
    ] as const) {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = text;
      this.#order.append(option);
    }
    label.append(this.#order);
    const gain = document.createElement("span");
    gain.className = "gain";
    gain.title = "Display gain (↑ ↓); click the value to reset it";
    gain.append("Gain ");
    this.#gain = document.createElement("output");
    this.#gain.addEventListener("click", () => {
      target.setGain(1);
      target.settled();
    });
    for (const [factor, text] of [
      [1 / GAIN_STEP, "−"],
      [GAIN_STEP, "+"],
    ] as const) {
      const button = document.createElement("button");
      button.textContent = text;
      button.className = factor > 1 ? "gain-up" : "gain-down";
      button.addEventListener("click", () => {
        if (this.#state === undefined) return;
        target.setGain(this.#state.gain * factor);
        target.settled();
      });
      if (factor > 1) gain.append(this.#gain, button);
      else gain.append(button);
    }
    const rowsLabel = document.createElement("label");
    rowsLabel.textContent = "Rows ";
    this.#rows = document.createElement("select");
    this.#rows.className = "rows";
    this.#rows.addEventListener("change", () => {
      target.setRows(Number(this.#rows.value));
      target.settled();
    });
    rowsLabel.append(this.#rows);
    this.#selection = document.createElement("button");
    this.#selection.className = "selection";
    this.#selection.title = "Clear the selected groups (Esc)";
    this.#selection.hidden = true;
    this.#selection.addEventListener("click", () => {
      target.clearSelection();
    });
    this.#markers = document.createElement("span");
    this.#markers.className = "markers";
    this.#markers.hidden = true;
    for (const [direction, text, title] of [
      [-1, "◀", "Previous marker ([)"],
      [1, "▶", "Next marker (])"],
    ] as const) {
      const button = document.createElement("button");
      button.textContent = text;
      button.title = title;
      button.addEventListener("click", () => {
        target.stepMarker(direction);
      });
      this.#markers.append(button);
    }
    this.toolbar.append(this.#range, rowsLabel, gain, label, this.#markers, this.#selection);
    this.toolbar.hidden = true;

    this.track = document.createElement("div");
    this.track.className = "track";
    this.#thumb = document.createElement("div");
    this.#thumb.className = "thumb";
    this.track.append(this.#thumb);
    this.track.hidden = true;
    this.ids = document.createElement("div");
    this.ids.className = "ids";
    this.ids.hidden = true;

    this.#order.addEventListener("change", () => {
      target.setOrder(this.#order.value as ChannelOrder);
      target.settled();
    });
    this.#attachScroll();
  }

  update(state: ControlsState): void {
    this.#state = state;
    this.toolbar.hidden = false;
    this.track.hidden = false;
    this.ids.hidden = false;
    const { firstSlot, nSlots, nDisplayed } = state;
    const first = Math.floor(firstSlot) + 1;
    const last = Math.min(nDisplayed, Math.ceil(firstSlot + nSlots));
    this.#range.textContent = `${state.idAt(first - 1)}–${state.idAt(last - 1)} · rows ${String(first)}–${String(last)} of ${String(nDisplayed)}`;
    this.#range.title =
      "Ids of the channels at the top and bottom of the view, and which rows of the displayed order those are";
    this.#drawIds(state);
    const scrollable = nDisplayed - nSlots;
    this.#thumb.style.setProperty(
      "--thumb",
      `max(${String(MIN_THUMB_PX)}px, ${String((100 * nSlots) / Math.max(1, nDisplayed))}%)`,
    );
    this.#thumb.style.setProperty("--at", String(scrollable > 0 ? firstSlot / scrollable : 0));
    this.#order.value = state.order;
    const depth = this.#order.querySelector<HTMLOptionElement>('option[value="depth"]');
    if (depth !== null) depth.disabled = !state.canDepth;
    this.#gain.value = `×${state.gain.toPrecision(3)}`;
    this.#syncRows(nSlots, nDisplayed);
  }

  /** Offer the preset counts below the displayed total, the total itself, and the current count. */
  #syncRows(nSlots: number, nDisplayed: number): void {
    const counts = [...new Set([...ROW_CHOICES.filter((n) => n < nDisplayed), nSlots, nDisplayed])].sort(
      (a, b) => a - b,
    );
    const key = counts.join(",");
    if (this.#rows.dataset.counts !== key) {
      this.#rows.dataset.counts = key;
      this.#rows.replaceChildren(
        ...counts.map((n) => {
          const option = document.createElement("option");
          option.value = String(n);
          option.textContent = n === nDisplayed ? `All ${String(n)}` : String(n);
          return option;
        }),
      );
    }
    this.#rows.value = String(nSlots);
  }

  /** Show how many event groups are selected, with a button that clears them. */
  setSelection(count: number): void {
    this.#selection.hidden = count === 0;
    this.#selection.textContent = `${String(count)} selected ✕`;
  }

  /** Show the marker buttons while there are markers. */
  setMarkers(count: number): void {
    this.#markers.hidden = count === 0;
  }

  hide(): void {
    this.#state = undefined;
    this.toolbar.hidden = true;
    this.track.hidden = true;
    this.ids.hidden = true;
    this.ids.replaceChildren();
  }

  /** Label every k-th slot, counted from the first displayed, so labels keep their rows while scrolling. */
  #drawIds({ firstSlot, nSlots, nDisplayed, idAt }: ControlsState): void {
    const slotPx = this.ids.clientHeight / Math.max(1, nSlots);
    if (slotPx <= 0) return;
    const every = Math.ceil(MIN_ID_SPACING_PX / slotPx);
    const spans: HTMLSpanElement[] = [];
    const end = Math.min(nDisplayed, Math.ceil(firstSlot + nSlots));
    for (let slot = Math.ceil(Math.floor(firstSlot) / every) * every; slot < end; slot += every) {
      const centre = (slot + 0.5 - firstSlot) * slotPx;
      if (centre < 0) continue;
      const span = document.createElement("span");
      span.textContent = idAt(slot);
      span.style.top = `${String(centre)}px`;
      spans.push(span);
    }
    this.ids.replaceChildren(...spans);
    // The toolbar sits left of the id column, so it hides no row's id.
    const widest = Math.max(0, ...spans.map((span) => span.offsetWidth));
    this.toolbar.style.setProperty("--ids-w", `${String(widest)}px`);
  }

  /** Slots per CSS pixel of thumb travel. */
  #slotsPerPx(state: ControlsState): number {
    const travel = this.track.clientHeight - this.#thumb.offsetHeight;
    return travel > 0 ? (state.nDisplayed - state.nSlots) / travel : 0;
  }

  #attachScroll(): void {
    let drag: { pointer: number; startY: number; startSlot: number } | undefined;
    this.#thumb.addEventListener("pointerdown", (event) => {
      if (event.button !== 0 || this.#state === undefined) return;
      event.stopPropagation();
      this.#thumb.setPointerCapture(event.pointerId);
      drag = { pointer: event.pointerId, startY: event.clientY, startSlot: this.#state.firstSlot };
    });
    this.#thumb.addEventListener("pointermove", (event) => {
      if (drag?.pointer !== event.pointerId || this.#state === undefined) return;
      this.target.setFirstSlot(drag.startSlot + (event.clientY - drag.startY) * this.#slotsPerPx(this.#state));
    });
    const end = (event: PointerEvent): void => {
      if (drag?.pointer !== event.pointerId) return;
      drag = undefined;
      this.target.settled();
    };
    this.#thumb.addEventListener("pointerup", end);
    this.#thumb.addEventListener("pointercancel", end);
    // A press on the track beside the thumb pages toward it.
    this.track.addEventListener("pointerdown", (event) => {
      const state = this.#state;
      if (event.button !== 0 || state === undefined || event.target !== this.track) return;
      const above = event.clientY < this.#thumb.getBoundingClientRect().top;
      this.target.setFirstSlot(state.firstSlot + (above ? -state.nSlots : state.nSlots));
      this.target.settled();
    });
    let settleTimer: ReturnType<typeof setTimeout> | undefined;
    this.track.addEventListener(
      "wheel",
      (event) => {
        const state = this.#state;
        if (state === undefined) return;
        event.preventDefault();
        this.target.setFirstSlot(state.firstSlot + wheelSlots(event, state.nSlots));
        clearTimeout(settleTimer);
        settleTimer = setTimeout(() => {
          this.target.settled();
        }, SETTLE_MS);
      },
      { passive: false },
    );
  }
}
