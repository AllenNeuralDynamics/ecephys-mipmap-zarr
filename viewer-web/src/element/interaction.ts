import { panBy, zoomAbout, type View } from "../render/view.js";

export interface InteractionTarget {
  readonly element: HTMLElement;
  readonly canvas: HTMLCanvasElement;
  view(): View | undefined;
  setView(view: Partial<View>): void;
  /** A gesture has finished changing the view. */
  settled(): void;
  /** A click that was not a drag, at device pixels from the canvas's top left. */
  clicked(xPx: number, yPx: number): void;
  /** Listen for the gain keys on `element`; one of several canvases sharing an element takes them. */
  keys?: boolean;
}

/** Wheel zoom per 100 px of wheel delta. */
const ZOOM_PER_WHEEL = 1.25;
const GAIN_STEP = 1.25;
const DRAG_THRESHOLD_PX = 4;
/** Quiet time after a wheel or key before a gesture counts as finished. */
export const SETTLE_MS = 200;

/** Wheel delta in CSS pixels, whatever unit the browser reported it in. */
function wheelPx(event: WheelEvent, delta: number): number {
  const unit =
    event.deltaMode === WheelEvent.DOM_DELTA_LINE ? 33 : event.deltaMode === WheelEvent.DOM_DELTA_PAGE ? 400 : 1;
  return delta * unit;
}

/** Slots a channel-scrolling wheel event moves the view by: about an eighth of a screen per 100 px. */
export function wheelSlots(event: WheelEvent, nSlots: number): number {
  // Browsers report shift-wheel as horizontal on some platforms.
  return (wheelPx(event, event.deltaY || event.deltaX) / 100) * Math.max(1, nSlots / 8);
}

/**
 * Drag pans, wheel zooms time about the cursor, shift-wheel scrolls channels, up and down change gain.
 * Returns a function that removes the listeners.
 */
export function attachInteraction(target: InteractionTarget): () => void {
  const { element, canvas } = target;
  const ratio = (): number => globalThis.devicePixelRatio || 1;
  const local = (event: MouseEvent): [number, number] => {
    const rect = canvas.getBoundingClientRect();
    return [(event.clientX - rect.left) * ratio(), (event.clientY - rect.top) * ratio()];
  };
  let settleTimer: ReturnType<typeof setTimeout> | undefined;
  const settleSoon = (): void => {
    clearTimeout(settleTimer);
    settleTimer = setTimeout(() => {
      target.settled();
    }, SETTLE_MS);
  };

  let drag: { pointer: number; lastX: number; startX: number; startY: number; moved: boolean } | undefined;
  const onDown = (event: PointerEvent): void => {
    if (event.button !== 0) return;
    canvas.setPointerCapture(event.pointerId);
    drag = {
      pointer: event.pointerId,
      lastX: event.clientX,
      startX: event.clientX,
      startY: event.clientY,
      moved: false,
    };
  };
  const onMove = (event: PointerEvent): void => {
    if (drag?.pointer !== event.pointerId) return;
    if (Math.hypot(event.clientX - drag.startX, event.clientY - drag.startY) > DRAG_THRESHOLD_PX) drag.moved = true;
    const view = target.view();
    if (drag.moved && view !== undefined) target.setView(panBy(view, (event.clientX - drag.lastX) * ratio()));
    drag.lastX = event.clientX;
  };
  const onUp = (event: PointerEvent): void => {
    if (drag?.pointer !== event.pointerId) return;
    const { moved } = drag;
    drag = undefined;
    if (moved) target.settled();
    else target.clicked(...local(event));
  };
  const onWheel = (event: WheelEvent): void => {
    const view = target.view();
    if (view === undefined) return;
    event.preventDefault();
    if (event.shiftKey) {
      target.setView({ firstSlot: view.firstSlot + wheelSlots(event, view.nSlots) });
    } else {
      const [x] = local(event);
      target.setView(zoomAbout(view, ZOOM_PER_WHEEL ** (wheelPx(event, event.deltaY) / 100), x));
    }
    settleSoon();
  };
  const onKey = (event: KeyboardEvent): void => {
    const view = target.view();
    if (view === undefined || (event.key !== "ArrowUp" && event.key !== "ArrowDown")) return;
    event.preventDefault();
    target.setView({ gain: view.gain * (event.key === "ArrowUp" ? GAIN_STEP : 1 / GAIN_STEP) });
    settleSoon();
  };

  canvas.addEventListener("pointerdown", onDown);
  canvas.addEventListener("pointermove", onMove);
  canvas.addEventListener("pointerup", onUp);
  canvas.addEventListener("pointercancel", onUp);
  canvas.addEventListener("wheel", onWheel, { passive: false });
  const keys = target.keys ?? true;
  if (keys) element.addEventListener("keydown", onKey);
  return () => {
    clearTimeout(settleTimer);
    canvas.removeEventListener("pointerdown", onDown);
    canvas.removeEventListener("pointermove", onMove);
    canvas.removeEventListener("pointerup", onUp);
    canvas.removeEventListener("pointercancel", onUp);
    canvas.removeEventListener("wheel", onWheel);
    if (keys) element.removeEventListener("keydown", onKey);
  };
}
