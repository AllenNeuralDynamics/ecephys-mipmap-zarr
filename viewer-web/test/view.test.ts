import { describe, expect, test } from "vitest";

import { clampView, MAX_PX_PER_SAMPLE, panBy, pxOf, visibleSamples, zoomAbout, type View } from "../src/render/view.js";

const VIEW: View = { t0: 1000, samplesPerPx: 10, firstSlot: 0, nSlots: 8, gain: 1 };
const BOUNDS = { nSamples: 100_000, nDisplayed: 16 };

describe("view", () => {
  test("zooming keeps the sample under the cursor in place", () => {
    const zoomed = zoomAbout(VIEW, 4, 250);
    expect(zoomed.samplesPerPx).toBe(40);
    expect(zoomed.t0 + 250 * zoomed.samplesPerPx).toBe(VIEW.t0 + 250 * VIEW.samplesPerPx);
  });

  test("dragging right shows earlier samples", () => {
    expect(panBy(VIEW, 30).t0).toBe(700);
    expect(pxOf(panBy(VIEW, 30), 1000)).toBe(30);
  });

  test("clamping keeps time and slots inside the recording and channel list", () => {
    const clamped = clampView({ ...VIEW, t0: 99_000, firstSlot: 12, samplesPerPx: 1e-6 }, BOUNDS, 100);
    expect(clamped.samplesPerPx).toBe(1 / MAX_PX_PER_SAMPLE);
    expect(clamped.firstSlot).toBe(8);
    expect(visibleSamples(clamped, 100)[1]).toBeLessThanOrEqual(BOUNDS.nSamples);
    const whole = clampView({ ...VIEW, samplesPerPx: 1e9 }, BOUNDS, 100);
    expect(visibleSamples(whole, 100)).toEqual([0, BOUNDS.nSamples]);
  });
});
