import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";

import FileSystemStore from "@zarrita/storage/fs";
import { describe, expect, test } from "vitest";

import type { Codes } from "../src/reader/decode.js";
import { openStore, type ArrayInfo } from "../src/reader/store.js";
import { TileSource } from "../src/reader/tiles.js";
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
