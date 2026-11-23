import { join } from "node:path";

import { expect, test, type Page } from "@playwright/test";
import FileSystemStore from "@zarrita/storage/fs";

import { planWindow } from "../../src/reader/plan.js";
import { FILL_LEVEL } from "../../src/render/ramp.js";
import { openStore, type OpenedStore } from "../../src/reader/store.js";
import { TileSource } from "../../src/reader/tiles.js";
import type { View } from "../../src/render/view.js";
import { readRegion } from "../region.js";

const FIXTURES = join(import.meta.dirname, "..", "fixtures");
const FS = 30_000;
const N_SAMPLES = 200_000;
/** Each channel's spike, which make_fixtures.py puts at `5000 + 20011 * channel`. */
const spikeAt = (channel: number): number => 5_000 + 20_011 * channel;
const PX_TOLERANCE = 2;
/** Columns a joining outline reaches either side: at 1.7 px plus antialiasing, a steep one touches the second. */
const REACH_COLUMNS = 2;
/**
 * Alpha a pixel needs to count as drawn: the fill counts, an antialiased line's faint fringe does not. The ramp
 * reaches full alpha at the fill level, so this is a level of 1/8.
 */
const DRAWN_ALPHA = Math.round(((1 / 8) * 255) / FILL_LEVEL);

interface Drawn {
  width: number;
  height: number;
  /** Per slot, per column: topmost and bottommost row drawn at `DRAWN_ALPHA` or more (top = 0), or -1 for none. */
  top: number[][];
  bottom: number[][];
  layout: { centres: number[]; slotPx: number; pxPerUnit: number; view: View; displayed: number[] };
}

/** Open `store` in the test page showing `channels` over `range` samples, and return drawn extents. */
async function render(
  page: Page,
  store: string,
  channels: number[],
  range: [number, number],
  widthPx: number,
  slotsPerExcursion = 1 / 2.1,
): Promise<Drawn> {
  const excursion = await fullRange(store, channels);
  await page.goto("/test/e2e/page.html");
  await page.waitForFunction(() => (window as unknown as { viewerReady?: boolean }).viewerReady === true);
  return page.evaluate(
    async ({ store, channels, range, widthPx, fs, excursion, slotsPerExcursion, drawnAlpha }) => {
      const viewer = document.querySelector("mipmap-viewer");
      if (viewer === null) throw new Error("no viewer");
      viewer.style.width = `${String(widthPx)}px`;
      viewer.channels = channels;
      // Slots follow `channels` as given, not depth.
      viewer.order = "index";
      viewer.setAttribute("controls", "none");
      viewer.store = store;
      await viewer.ready;
      await viewer.idle();
      // Scale a channel's whole range to `slotsPerExcursion` slots; under half a slot, nothing reaches the next.
      viewer.gain = ((viewer.unitRange ?? 1) * slotsPerExcursion) / excursion;
      viewer.timeRange = [range[0] / fs, range[1] / fs];
      await viewer.idle();
      const shot = viewer.snapshot();
      const layout = viewer.layout();
      if (shot === undefined || layout === undefined) throw new Error("viewer did not render");
      const nSlots = layout.view.nSlots;
      const top = Array.from({ length: nSlots }, () => new Array<number>(shot.width).fill(-1));
      const bottom = Array.from({ length: nSlots }, () => new Array<number>(shot.width).fill(-1));
      for (let row = 0; row < shot.height; row++) {
        const y = shot.height - 1 - row;
        const slot = Math.floor(y / layout.slotPx);
        const slotTop = top[slot];
        const slotBottom = bottom[slot];
        if (slotTop === undefined || slotBottom === undefined) continue;
        for (let x = 0; x < shot.width; x++) {
          if ((shot.data[(row * shot.width + x) * 4 + 3] ?? 0) < drawnAlpha) continue;
          if (slotTop[x] === -1 || y < (slotTop[x] ?? 0)) slotTop[x] = y;
          if (y > (slotBottom[x] ?? 0)) slotBottom[x] = y;
        }
      }
      return {
        width: shot.width,
        height: shot.height,
        top,
        bottom,
        layout: { ...layout, centres: Array.from(layout.centres), displayed: [...layout.displayed] },
      };
    },
    {
      store: `/test/fixtures/${store}.zarr`,
      channels,
      range,
      widthPx,
      fs: FS,
      excursion,
      slotsPerExcursion,
      drawnAlpha: DRAWN_ALPHA,
    },
  );
}

/** Largest max minus min of any of `channels` over the whole recording, from the coarsest level. */
async function fullRange(store: string, channels: number[]): Promise<number> {
  const opened = await openStore(new FileSystemStore(join(FIXTURES, `${store}.zarr`)));
  const key = `minmax/s${String(opened.info.strides[opened.info.strides.length - 1])}`;
  const array = opened.info.arrays[key];
  if (array === undefined) throw new Error(`no ${key}`);
  const nBlocks = array.shape[2] ?? 0;
  const source = new TileSource(opened);
  let widest = 0;
  for (const channel of channels) {
    const { values } = await readRegion(source, array, [channel, channel + 1], [0, nBlocks]);
    widest = Math.max(widest, Math.max(...values.slice(nBlocks)) - Math.min(...values.slice(0, nBlocks)));
  }
  return widest;
}

/** Columns whose drawn extent leaves the data's range (widened by the columns and blocks outlines reach) or misses its core. */
async function mismatches(opened: OpenedStore, drawn: Drawn, key: string): Promise<string[]> {
  const array = opened.info.arrays[key];
  if (array === undefined) throw new Error(`no ${key}`);
  const source = new TileSource(opened);
  const { view, slotPx, pxPerUnit, centres, displayed } = drawn.layout;
  const stride = array.stride;
  const margin = 2 * stride;
  const reach = REACH_COLUMNS * view.samplesPerPx;
  const start = Math.max(0, Math.floor((view.t0 - reach - margin) / stride));
  const end = Math.min(
    array.shape.length === 3 ? (array.shape[2] ?? 0) : N_SAMPLES,
    Math.ceil((view.t0 + view.samplesPerPx * drawn.width + reach + margin) / stride),
  );
  const problems: string[] = [];
  for (const [slot, channel] of displayed.entries()) {
    const region = await readRegion(source, array, [channel, channel + 1], [start, end]);
    const n = end - start;
    const lo = (i: number): number => region.values[i] ?? NaN;
    const hi = (i: number): number => region.values[array.shape.length === 3 ? n + i : i] ?? NaN;
    const centre = (slot - view.firstSlot + 0.5) * slotPx;
    const yOf = (v: number): number => centre - (v - (centres[channel] ?? 0)) * pxPerUnit;
    const extent = (s0: number, s1: number): [number, number] | undefined => {
      let min = Infinity;
      let max = -Infinity;
      // Index i is drawn at its centre, sample position (i + 0.5) * stride.
      for (let i = Math.max(start, Math.ceil(s0 / stride - 0.5)); (i + 0.5) * stride < s1 && i < end; i++) {
        min = Math.min(min, lo(i - start));
        max = Math.max(max, hi(i - start));
      }
      return min <= max ? [yOf(max), yOf(min)] : undefined;
    };
    for (let x = 0; x < drawn.width; x++) {
      const c0 = view.t0 + x * view.samplesPerPx;
      const c1 = c0 + view.samplesPerPx;
      const top = drawn.top[slot]?.[x] ?? -1;
      const bottom = drawn.bottom[slot]?.[x] ?? -1;
      const outer = extent(c0 - reach - margin, c1 + reach + margin);
      const inner = extent(c0, c1);
      if (top === -1) {
        if (inner !== undefined) problems.push(`channel ${String(channel + 1)} column ${String(x + 1)}: nothing drawn`);
        continue;
      }
      if (outer === undefined || top < outer[0] - PX_TOLERANCE || bottom > outer[1] + PX_TOLERANCE) {
        problems.push(
          `channel ${String(channel + 1)} column ${String(x + 1)}: drawn ${String(top)}..${String(bottom)} outside ${String(outer)}`,
        );
      } else if (inner !== undefined && (top > inner[0] + PX_TOLERANCE || bottom < inner[1] - PX_TOLERANCE)) {
        problems.push(
          `channel ${String(channel + 1)} column ${String(x + 1)}: drawn ${String(top)}..${String(bottom)} misses ${String(inner)}`,
        );
      }
    }
  }
  return problems;
}

const STORES = [
  "int16_raw-none_pyr-none",
  "int16_raw-shift_pyr-shift",
  "int16_raw-range_pyr-range_gain",
  "float32_raw-none_pyr-range",
];
// Channels 6 then 3 (indices 5, 2): the first is drawn in the top slot.
const CHANNELS = [5, 2];
const ZOOMS: { name: string; key: string; range: [number, number]; widthPx: number }[] = [
  { name: "raw", key: "raw", range: [spikeAt(5) - 200, spikeAt(5) + 200], widthPx: 800 },
  { name: "s32", key: "minmax/s32", range: [spikeAt(2) - 40_000, spikeAt(2) + 40_000], widthPx: 800 },
  { name: "coarsest", key: "minmax/s2048", range: [0, N_SAMPLES], widthPx: 40 },
];

for (const store of STORES) {
  for (const zoom of ZOOMS) {
    test(`${store} at ${zoom.name} matches the decoded data column by column`, async ({ page }) => {
      const drawn = await render(page, store, CHANNELS, zoom.range, zoom.widthPx);
      const opened = await openStore(new FileSystemStore(join(FIXTURES, `${store}.zarr`)));
      const { view } = drawn.layout;
      expect(planWindow(opened.info, view.t0, view.t0 + view.samplesPerPx * drawn.width, drawn.width).key).toBe(
        zoom.key,
      );
      expect(drawn.layout.displayed).toEqual(CHANNELS);
      expect(await mismatches(opened, drawn, zoom.key)).toEqual([]);
    });
  }
}

test("the spike lands in its channel's slot at its sample's column", async ({ page }) => {
  const channel = 5;
  const range: [number, number] = [spikeAt(channel) - 200, spikeAt(channel) + 200];
  const drawn = await render(page, "int16_raw-none_pyr-none", [channel, 2], range, 800);
  const bottoms = drawn.bottom[0] ?? [];
  // A spike clipped at the slot's edge bottoms out flat over a few columns; their middle is the sample's.
  const deepest = Math.max(...bottoms);
  const flat = bottoms.flatMap((b, x) => (b === deepest ? [x] : []));
  const lowest = ((flat[0] ?? 0) + (flat.at(-1) ?? 0)) / 2;
  const { view } = drawn.layout;
  const expected = (spikeAt(channel) + 0.5 - view.t0) / view.samplesPerPx;
  expect(Math.abs(lowest - expected)).toBeLessThanOrEqual(1.5);
  // The other slot has no spike in this window, so it reaches far less deep below its centre.
  const depth = (slot: number): number => Math.max(...(drawn.bottom[slot] ?? [])) - (slot + 0.5) * drawn.layout.slotPx;
  expect(depth(1)).toBeLessThan(depth(0) / 3);
});

for (const [name, halfWidth] of [
  ["raw", 200],
  ["s32", 40_000],
] as const) {
  test(`a spike taller than its slot is clipped there at ${name}`, async ({ page }) => {
    const channel = 5;
    const range: [number, number] = [spikeAt(channel) - halfWidth, spikeAt(channel) + halfWidth];
    const drawn = await render(page, "int16_raw-none_pyr-none", [channel, 2], range, 800, 1.5);
    const { view, slotPx } = drawn.layout;
    const column = Math.floor((spikeAt(channel) + 0.5 - view.t0) / view.samplesPerPx);
    // The spike reaches the bottom of its own slot, and the channel below shows only itself there.
    expect(drawn.bottom[0]?.[column]).toBeGreaterThanOrEqual(Math.ceil(slotPx) - 2);
    expect(drawn.top[1]?.[column]).toBeGreaterThan(slotPx * 1.1);
  });
}
