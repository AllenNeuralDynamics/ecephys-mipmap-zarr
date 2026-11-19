import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";

import FileSystemStore from "@zarrita/storage/fs";
import { describe, expect, test } from "vitest";

import type { Codes } from "../src/reader/decode.js";
import { tilesFor } from "../src/reader/plan.js";
import { openStore, type ArrayInfo } from "../src/reader/store.js";
import { channelFirstCodes, decodeTile, TileSource } from "../src/reader/tiles.js";
import { readNpy } from "./npy.js";

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

/** Assemble channels `[c0, c1)` and indices `[start, end)` of `array` from tiles, channel-first. */
async function readRegion(
  source: TileSource,
  array: ArrayInfo,
  channels: [number, number],
  [start, end]: [number, number],
): Promise<{ codes: number[]; values: number[] }> {
  const nRows = array.shape.length === 3 ? 2 : 1;
  const nC = channels[1] - channels[0];
  const nT = end - start;
  const codes = new Array<number>(nC * nRows * nT).fill(NaN);
  const values = new Array<number>(nC * nRows * nT).fill(NaN);
  const timeAxis = array.timeMajor ? 0 : array.shape.length - 1;
  for (const coords of tilesFor(array, channels, start, end)) {
    const tile = await source.getTile(array.key, coords);
    const cf = channelFirstCodes(tile, array);
    const decoded = decodeTile(tile, array);
    const [tileC = 0] = cf.shape;
    const tileT = cf.shape[cf.shape.length - 1] ?? 0;
    const t0 = (coords[timeAxis] ?? 0) * (array.chunks[timeAxis] ?? 0);
    for (let c = 0; c < tileC; c++) {
      const outC = cf.c0 + c - channels[0];
      if (outC < 0 || outC >= nC) continue;
      for (let r = 0; r < nRows; r++) {
        for (let t = 0; t < tileT; t++) {
          const outT = t0 + t - start;
          if (outT < 0 || outT >= nT) continue;
          const from = (c * nRows + r) * tileT + t;
          const to = (outC * nRows + r) * nT + outT;
          codes[to] = cf.codes[from] ?? NaN;
          values[to] = decoded.values[from] ?? NaN;
        }
      }
    }
  }
  return { codes, values };
}

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
