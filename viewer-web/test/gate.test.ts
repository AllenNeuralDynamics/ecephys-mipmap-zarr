import { describe, expect, it } from "vitest";

import { gateStart } from "../src/render/renderer.js";

describe("gateStart", () => {
  it("holds at its floor for narrow columns and clears a wide column's expected noise extreme", () => {
    expect(gateStart(0.1)).toBe(3.5);
    expect(gateStart(8)).toBe(3.5);
    // The largest of 10,000 normal samples is about 3.9 sd.
    expect(gateStart(10_000)).toBeGreaterThan(3.9 + 0.5);
  });
});
