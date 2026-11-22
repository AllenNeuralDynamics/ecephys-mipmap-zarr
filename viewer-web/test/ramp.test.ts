import { describe, expect, it } from "vitest";

import { buildRamp, FILL_LEVEL, NOISE_LEVEL, parseColor, RAMP_STEPS, toLab, fromLab } from "../src/render/ramp.js";

const palette = {
  background: "#fbfbfc",
  fill: "#e8edf5",
  noise: "#a9b6ca",
  ink: "#10141c",
  groups: ["#0a5a9c", "rgb(184 74 0)"],
  lineWidth: [1, 1.7] as [number, number],
};
const texel = (ramp: Uint8Array, row: number, step: number): number[] =>
  Array.from(ramp.slice((row * RAMP_STEPS + step) * 4, (row * RAMP_STEPS + step) * 4 + 4));
const stepOf = (level: number): number => Math.round(level * (RAMP_STEPS - 1));

describe("parseColor", () => {
  it("reads hex and rgb() forms alike", () => {
    expect(parseColor("#fff")).toEqual([1, 1, 1]);
    expect(parseColor("#0a5a9c")).toEqual(parseColor("rgb(10, 90, 156)"));
    expect(parseColor("rgb(10 90 156 / 0.5)")).toEqual(parseColor("#0a5a9c"));
  });
  it("refuses a named colour", () => {
    expect(() => parseColor("red")).toThrow();
  });
});

describe("Lab", () => {
  it("round-trips sRGB", () => {
    for (const hex of ["#10141c", "#e8edf5", "#b84a00"]) {
      const rgb = parseColor(hex);
      fromLab(toLab(rgb)).forEach((c, i) => {
        expect(c).toBeCloseTo(rgb[i] ?? 0, 3);
      });
    }
  });
});

describe("buildRamp", () => {
  const ramp = buildRamp(palette);
  it("has a row for ink and one per group", () => {
    expect(ramp.length).toBe(3 * RAMP_STEPS * 4);
  });
  it("is transparent at zero and the fill colour, premultiplied, below the fill level", () => {
    expect(texel(ramp, 0, 0)).toEqual([0, 0, 0, 0]);
    const half = texel(ramp, 0, stepOf(FILL_LEVEL / 2));
    const alpha = (half[3] ?? 0) / 255;
    expect(alpha).toBeGreaterThan(0.4);
    expect(alpha).toBeLessThan(0.6);
    expect((half[0] ?? 0) / 255 / alpha).toBeCloseTo(parseColor(palette.fill)[0], 1);
  });
  it("passes through noise and ends at each row's colour, opaque", () => {
    const hex = (t: number[]): string =>
      `#${t
        .slice(0, 3)
        .map((c) => c.toString(16).padStart(2, "0"))
        .join("")}`;
    expect(texel(ramp, 0, stepOf(NOISE_LEVEL))[3]).toBe(255);
    expect(hex(texel(ramp, 0, RAMP_STEPS - 1))).toBe(palette.ink);
    expect(hex(texel(ramp, 1, RAMP_STEPS - 1))).toBe("#0a5a9c");
    expect(hex(texel(ramp, 2, RAMP_STEPS - 1))).toBe("#b84a00");
  });
  it("darkens monotonically from the fill level on a light palette", () => {
    let last = Infinity;
    for (let i = stepOf(FILL_LEVEL) + 1; i < RAMP_STEPS; i++) {
      const [r = 0, g = 0, b = 0] = texel(ramp, 0, i);
      const lightness = toLab([r / 255, g / 255, b / 255])[0];
      expect(lightness).toBeLessThanOrEqual(last + 0.5);
      last = lightness;
    }
  });
});
