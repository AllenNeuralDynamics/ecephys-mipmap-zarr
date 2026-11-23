import { SETTLE_MS, wheelSlots } from "./interaction.js";
import type { ChannelOrder } from "./mipmap-viewer.js";

export interface ControlsTarget {
  setFirstSlot(firstSlot: number): void;
  setOrder(order: ChannelOrder): void;
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
  /** The store has channel locations to sort by. */
  canDepth: boolean;
  /** Ids of the channels at the top and bottom of the view. */
  topId: string;
  bottomId: string;
}

export const CONTROLS_STYLE = `
.toolbar { position: absolute; right: 22px; top: 6px; display: flex; gap: 8px; align-items: center;
  font: 12px system-ui, sans-serif; padding: 2px 6px; border-radius: 4px; background: var(--mv-panel); }
.toolbar select, .toolbar button { font: inherit; }
.toolbar button[hidden], .toolbar span[hidden] { display: none; }
.markers { display: flex; gap: 2px; }
.track { position: absolute; right: 0; top: 0; bottom: 0; width: 14px; background: var(--mv-track);
  touch-action: none; }
.thumb { position: absolute; left: 2px; right: 2px; border-radius: 5px; background: var(--mv-thumb); cursor: grab;
  height: var(--thumb); top: calc((100% - var(--thumb)) * var(--at)); }
.thumb:active { cursor: grabbing; background: var(--mv-thumb-active); }
.toolbar[hidden], .track[hidden], :host([controls="none"]) .toolbar, :host([controls="none"]) .track { display: none; }
:host([controls="none"]) canvas { width: 100%; }
`;

/** Thumb height floor, CSS px, so a long channel list keeps a grabbable thumb. */
const MIN_THUMB_PX = 16;

/**
 * The channel scroll bar beside the traces and the toolbar over them: which channels are in view, and
 * the channel order. Hidden by `controls="none"`.
 */
export class Controls {
  readonly toolbar: HTMLDivElement;
  readonly track: HTMLDivElement;
  readonly #thumb: HTMLDivElement;
  readonly #range: HTMLSpanElement;
  readonly #order: HTMLSelectElement;
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
    this.toolbar.append(this.#range, label, this.#markers, this.#selection);
    this.toolbar.hidden = true;

    this.track = document.createElement("div");
    this.track.className = "track";
    this.#thumb = document.createElement("div");
    this.#thumb.className = "thumb";
    this.track.append(this.#thumb);
    this.track.hidden = true;

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
    const { firstSlot, nSlots, nDisplayed } = state;
    const first = Math.floor(firstSlot) + 1;
    const last = Math.min(nDisplayed, Math.ceil(firstSlot + nSlots));
    this.#range.textContent = `${state.topId}–${state.bottomId} · ${String(first)}–${String(last)} of ${String(nDisplayed)}`;
    this.#range.title = "Channel ids at the top and bottom of the view; their positions in the displayed order";
    const scrollable = nDisplayed - nSlots;
    this.#thumb.style.setProperty(
      "--thumb",
      `max(${String(MIN_THUMB_PX)}px, ${String((100 * nSlots) / Math.max(1, nDisplayed))}%)`,
    );
    this.#thumb.style.setProperty("--at", String(scrollable > 0 ? firstSlot / scrollable : 0));
    this.#order.value = state.order;
    const depth = this.#order.querySelector<HTMLOptionElement>('option[value="depth"]');
    if (depth !== null) depth.disabled = !state.canDepth;
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
