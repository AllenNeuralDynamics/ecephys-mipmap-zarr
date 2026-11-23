import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";

import FileSystemStore from "@zarrita/storage/fs";
import { describe, expect, test } from "vitest";

import type { Codes } from "../src/reader/decode.js";
import { openStore, type ArrayInfo } from "../src/reader/store.js";
import { TileSource } from "../src/reader/tiles.js";
import { centreAndExcursion, clippedStats } from "../src/render/stats.js";
import { readNpy } from "./npy.js";
import { readRegion } from "./region.js";

const FIXTURES = join(import.meta.dirname, "fixtures");

interface Entry {
  array: string;
  channels: [number, number];
  range: [number, number];
  codes: string;
  decoded: string;
}
interface StoreEntry {
  name: string;
  path: string;
  entries: Entry[];
}

const index = existsSync(join(FIXTURES, "index.json"))
  ? (JSON.parse(readFileSync(join(FIXTURES, "index.json"), "utf8")) as { stores: StoreEntry[] })
  : undefined;

/** Largest float32 rounding a decode `code * scale + offset` can carry, per element. */
function tolerance(array: ArrayInfo, channel: number, row: number, code: number): number {
  const d = array.decode;
  if (d.kind === "none" || d.kind === "shift") return 0;
  const scale = d.scale[channel] ?? 0;
  const offset =
    d.kind === "affine" ? (d.offset[channel] ?? 0) : ((row === 0 ? d.minOffset : d.maxOffset)[channel] ?? 0);
  return 2 ** -22 * (Math.abs(code * scale) + Math.abs(offset)) + 1e-12;
}

describe.skipIf(index === undefined)("fixture stores decode as Python decodes them", () => {
  for (const store of index?.stores ?? []) {
    test(store.name, async () => {
      const opened = await openStore(new FileSystemStore(join(FIXTURES, store.path)), {
        storeFor: (location) => new FileSystemStore(location),
      });
      expect(opened.info.rawError).toBeNull();
      const source = new TileSource(opened);
      for (const entry of store.entries) {
        const array = opened.info.arrays[entry.array === "raw" ? "raw" : entry.array];
        expect(array, entry.array).toBeDefined();
        if (array === undefined) continue;
        const got = await readRegion(source, array, entry.channels, entry.range);
        const wantCodes: Codes = readNpy(join(FIXTURES, entry.codes)).data;
        const wantValues: Codes = readNpy(join(FIXTURES, entry.decoded)).data;
        const label = `${entry.array} c${String(entry.channels)} i${String(entry.range)}`;
        expect(got.codes, label).toEqual(Array.from(wantCodes));
        const nRows = array.shape.length === 3 ? 2 : 1;
        const nT = entry.range[1] - entry.range[0];
        for (let i = 0; i < got.values.length; i++) {
          const channel = entry.channels[0] + Math.floor(i / (nRows * nT));
          const row = Math.floor(i / nT) % nRows;
          const want = wantValues[i] ?? NaN;
          const err = Math.abs((got.values[i] ?? NaN) - want);
          if (!(err <= tolerance(array, channel, row, got.codes[i] ?? 0))) {
            expect.fail(`${label} element ${String(i)}: got ${String(got.values[i])}, Python ${String(want)}`);
          }
        }
      }
    });
  }
});

/** Float32 rounding of a stored statistic, plus the decode rounding a sum over a chunk carries. */
const close = (got: number, want: number): boolean => Math.abs(got - want) <= 1e-5 * Math.max(1, Math.abs(want));

describe.skipIf(index === undefined)("stored channel_stats match what the viewer measures", () => {
  for (const store of index?.stores ?? []) {
    test(store.name, async () => {
      const opened = await openStore(new FileSystemStore(join(FIXTURES, store.path)), {
        storeFor: (location) => new FileSystemStore(location),
      });
      const { info } = opened;
      const stored = info.channelStats;
      expect(stored).not.toBeNull();
      if (stored === null) return;
      const source = new TileSource(opened);
      const nC = info.nChannels;
      const baseline = new Array<number>(nC).fill(NaN);
      const raw = info.arrays.raw;
      if (raw !== undefined) {
        const chunk = raw.chunks[raw.timeMajor ? 0 : 1] ?? 1;
        const start = Math.floor(Math.floor(info.nSamples / 2) / chunk) * chunk;
        const stop = Math.min(start + chunk, info.nSamples);
        const { values } = await readRegion(source, raw, [0, nC], [start, stop]);
        for (let c = 0; c < nC; c++) {
          const { mean, sd } = clippedStats(
            Float32Array.from(values.slice(c * (stop - start), (c + 1) * (stop - start))),
          );
          baseline[c] = mean;
          expect(close(stored.noiseSd[c] ?? NaN, sd), `noise_sd ${String(c)}`).toBe(true);
        }
      }
      const coarsest = info.arrays[`minmax/s${String(Math.max(...info.strides))}`];
      expect(coarsest).toBeDefined();
      if (coarsest === undefined) return;
      const nB = coarsest.shape[2] ?? 0;
      const { values } = await readRegion(source, coarsest, [0, nC], [0, nB]);
      for (let c = 0; c < nC; c++) {
        const lows = values.slice(2 * c * nB, (2 * c + 1) * nB);
        const highs = values.slice((2 * c + 1) * nB, (2 * c + 2) * nB);
        const { centre, excursion } = centreAndExcursion(lows, highs, baseline[c] ?? NaN);
        expect(close(stored.centre[c] ?? NaN, centre), `centre ${String(c)}`).toBe(true);
        expect(close(stored.excursion[c] ?? NaN, excursion), `excursion ${String(c)}`).toBe(true);
      }
    });
  }
});
