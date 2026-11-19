import { describe, expect, test } from "vitest";

import { planWindow, tilesFor } from "../src/reader/plan.js";
import type { ArrayInfo, StoreInfo } from "../src/reader/store.js";

const array = (key: string, shape: number[], chunks: number[], timeMajor = false): ArrayInfo => ({
  key,
  shape,
  chunks,
  dtype: "int16",
  timeMajor,
  stride: 1,
  decode: { kind: "none" },
});

const N = 1_000_000;
const info = (withRaw = true): StoreInfo => ({
  formatVersion: 1,
  samplingFrequency: 30000,
  nSamples: N,
  nChannels: 8,
  strides: [32, 256, 2048],
  units: "counts",
  dtype: "int16",
  tStart: 0,
  rawSource: null,
  rawPath: null,
  sourceMetadata: null,
  channelIds: [],
  properties: {},
  rawError: null,
  arrays: {
    ...(withRaw ? { raw: array("raw", [8, N], [4, 1024]) } : {}),
    "minmax/s32": array("minmax/s32", [8, 2, Math.ceil(N / 32)], [4, 2, 256]),
  },
});

describe("planWindow", () => {
  // At oversample 2, stride s serves views with samples per pixel >= 2s.
  test.each([
    [63, "raw"],
    [64, "minmax/s32"],
    [511, "minmax/s32"],
    [512, "minmax/s256"],
    [4095, "minmax/s256"],
    [4096, "minmax/s2048"],
    [1e6, "minmax/s2048"],
  ])("%d samples per pixel draws from %s", (spp, key) => {
    expect(planWindow(info(), 0, spp * 100, 100).key).toBe(key);
  });

  test("block range covers the view and stops at the last block", () => {
    expect(planWindow(info(), 1000, 1000 + 64 * 100, 100)).toEqual({
      key: "minmax/s32",
      stride: 32,
      start: 31,
      end: Math.ceil(7400 / 32),
    });
    const tail = planWindow(info(), N - 1000, N + 5000, 1);
    expect(tail.end).toBe(Math.ceil(N / 2048));
  });

  test("raw clamps to the recording", () => {
    expect(planWindow(info(), -10.5, 20.5, 1000)).toEqual({ key: "raw", stride: 1, start: 0, end: 21 });
  });

  test("without raw, a view finer than the finest level draws from it", () => {
    expect(planWindow(info(false), 0, 100, 100).key).toBe("minmax/s32");
  });
});

describe("tilesFor", () => {
  test("covers a range crossing chunk boundaries in both axes", () => {
    expect(tilesFor(array("raw", [8, N], [4, 1024]), [2, 6], 1000, 1100)).toEqual([
      [0, 0],
      [1, 0],
      [0, 1],
      [1, 1],
    ]);
  });

  test("puts the channel chunk second for a (T, C) array and fixes a level's row axis at 0", () => {
    expect(tilesFor(array("raw", [N, 8], [30000, 8], true), [0, 8], 29999, 30001)).toEqual([
      [0, 0],
      [1, 0],
    ]);
    expect(tilesFor(array("minmax/s32", [8, 2, 100], [4, 2, 256]), [0, 1], 0, 10)).toEqual([[0, 0, 0]]);
  });

  test("an empty range needs no tiles", () => {
    expect(tilesFor(array("raw", [8, N], [4, 1024]), [0, 8], 5, 5)).toEqual([]);
  });
});
