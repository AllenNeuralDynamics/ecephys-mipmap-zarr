import { tilesFor } from "../src/reader/plan.js";
import type { ArrayInfo } from "../src/reader/store.js";
import { channelFirstCodes, decodeTile, type TileSource } from "../src/reader/tiles.js";

/** Assemble channels `[c0, c1)` and indices `[start, end)` of `array` from tiles, channel-first. */
export async function readRegion(
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
