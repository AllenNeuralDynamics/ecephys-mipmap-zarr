import { join } from "node:path";

import FileSystemStore from "@zarrita/storage/fs";
import { beforeAll, describe, expect, it } from "vitest";

import { formatSeconds, niceStep, stepMarker } from "../src/element/axis.js";
import { EventLayer, type EventGroups, type EventWindow } from "../src/events/layer.js";
import {
  drawCounts,
  drawEvents,
  groupSlots,
  newRasterImage,
  packColour,
  rasterGroupAt,
  type RasterGeometry,
  type RasterImage,
  type RasterStyle,
} from "../src/events/raster.js";

const FIXTURE = join(import.meta.dirname, "fixtures", "events.zarr");
const N_CHANNELS = 8;

let layer: EventLayer;
beforeAll(async () => {
  layer = await EventLayer.open(new FileSystemStore(FIXTURE));
});

describe("EventLayer counts", () => {
  it("bin each group's events", async () => {
    // 0.125 s at 30 kHz rounds to 4096 samples.
    expect(layer.countBins).toEqual([4096]);
    const counts = await layer.counts(4096, 0, 1e9);
    expect(counts.b0).toBe(0);
    const perGroup = (g: number): number =>
      counts.data.subarray(g * counts.nBins, (g + 1) * counts.nBins).reduce((a, b) => a + b, 0);
    expect(perGroup(0)).toBe(1);
    expect(perGroup(N_CHANNELS)).toBe(20);
    // Unit 6's spike at sample 5000 + 20011 * 5.
    expect(counts.data[5 * counts.nBins + Math.floor((5_000 + 20_011 * 5) / 4096)]).toBe(1);
  });

  it("clip a request to the stored bins", async () => {
    const counts = await layer.counts(4096, -10, 3);
    expect([counts.b0, counts.nBins]).toEqual([0, 3]);
  });

  it("estimate a range's events from the chunk index without fetching", async () => {
    expect(layer.estimate(0, Infinity)).toBe(layer.info.nEvents);
    expect(layer.estimate(10, 10)).toBe(0);
    const all = await layer.window(0, 100_000);
    const estimate = layer.estimate(0, 100_000);
    // Chunks of 5 events: the estimate is within one chunk of the truth.
    expect(Math.abs(estimate - all.sampleIndex.length)).toBeLessThanOrEqual(5);
  });
});

const groups = (k: number, channel: number[], weight: number[]): EventGroups => ({
  ids: channel.map((_, g) => `g${String(g)}`).slice(0, channel.length / k),
  footprintChannel: Int32Array.from(channel),
  footprintWeight: Float32Array.from(weight),
  k,
  properties: {},
});

describe("groupSlots", () => {
  it("weights footprint channels by their weight over the displayed ones", () => {
    const slots = groupSlots(groups(2, [0, 1, 2, 3], [3, 1, 1, 1]), Int32Array.from([0, 4, -1, 2]));
    // (3 * 0.5 + 1 * 4.5) / 4; the second group's only displayed channel is slot 2.
    expect(slots[0]).toBeCloseTo(1.5);
    expect(slots[1]).toBeCloseTo(2.5);
  });

  it("is NaN for a group with nothing displayed", () => {
    expect(groupSlots(groups(1, [0], [1]), Int32Array.from([-1]))[0]).toBeNaN();
  });
});

const RED = 0xff0000;
const BLUE = 0x0000ff;
/** 100 columns of 10 samples, 10 slots over 40 rows, one second per 1000 samples. */
const GEOM: RasterGeometry = {
  t0: 0,
  samplesPerPx: 10,
  firstSlot: 0,
  nSlots: 10,
  scale: 1,
  offset: 0,
  samplingFrequency: 1000,
};

function style(partial: Partial<RasterStyle>): RasterStyle {
  return {
    groupSlot: Float64Array.from([2.5, 2.5]),
    channelSlot: Int32Array.from([7]),
    colour: Int32Array.from([RED, BLUE]),
    top: Uint8Array.from([0, 0]),
    background: 0xffffff,
    ...partial,
  };
}

const pixel = (image: RasterImage, x: number, y: number): number[] =>
  Array.from(image.rgba.subarray((y * image.width + x) * 4, (y * image.width + x) * 4 + 3));

const events = (sample: number[], group: number[], channel?: number[]): EventWindow => ({
  sampleIndex: Float64Array.from(sample),
  group: Int32Array.from(group),
  ...(channel === undefined ? {} : { channel: Int32Array.from(channel) }),
});

describe("drawEvents", () => {
  it("ticks each event at its column on its group's slot", () => {
    const image = newRasterImage(100, 40);
    drawEvents(image, events([505], [0]), GEOM, style({}));
    // Slot 2.5 of 10 over 40 rows is row 10; ticks are 4 rows tall.
    expect(rasterGroupAt(image, 50, 10, 0)).toBe(0);
    const [r = 0, g = 0, b = 0] = pixel(image, 50, 10);
    expect(r).toBe(255);
    expect(g).toBeLessThan(255);
    expect(b).toBe(g);
    expect(pixel(image, 51, 10)).toEqual([255, 255, 255]);
    expect(pixel(image, 50, 20)).toEqual([255, 255, 255]);
  });

  it("draws top groups last and leaves out groups without a colour", () => {
    const image = newRasterImage(100, 40);
    drawEvents(image, events([505, 505], [1, 0]), GEOM, style({ top: Uint8Array.from([1, 0]) }));
    expect(rasterGroupAt(image, 50, 10, 0)).toBe(0);
    drawEvents(image, events([505, 505], [1, 0]), GEOM, style({ colour: Int32Array.from([-1, BLUE]) }));
    expect(rasterGroupAt(image, 50, 10, 0)).toBe(1);
  });

  it("puts an event with a channel on that channel's slot, and drops it when the channel is not displayed", () => {
    const image = newRasterImage(100, 40);
    drawEvents(image, events([505, 705], [0, 0], [0, 1]), GEOM, style({ channelSlot: Int32Array.from([7, -1]) }));
    expect(rasterGroupAt(image, 50, 30, 0)).toBe(0);
    expect(rasterGroupAt(image, 50, 10, 0)).toBe(-1);
    expect(rasterGroupAt(image, 70, 30, 2)).toBe(-1);
  });

  it("darkens a column with more events in it", () => {
    const one = newRasterImage(100, 40);
    const many = newRasterImage(100, 40);
    drawEvents(one, events([500], [1]), GEOM, style({}));
    drawEvents(many, events([500, 501, 502, 503], [1, 1, 1, 1]), GEOM, style({}));
    expect(pixel(many, 50, 10)[0]).toBeLessThan(pixel(one, 50, 10)[0] ?? 0);
  });
});

describe("drawCounts", () => {
  it("matches drawing the same events one by one when bins are a column wide", () => {
    const counted = newRasterImage(100, 40);
    const drawn = newRasterImage(100, 40);
    const data = new Uint32Array(2 * 100);
    data[50] = 3;
    drawCounts(counted, { bin: 10, b0: 0, nBins: 100, data }, GEOM, style({}));
    drawEvents(drawn, events([500, 503, 509], [0, 0, 0]), GEOM, style({}));
    expect(pixel(counted, 50, 10)).toEqual(pixel(drawn, 50, 10));
  });

  it("spreads a bin wider than a column over the columns it covers", () => {
    const image = newRasterImage(100, 40);
    const data = new Uint32Array(2 * 10);
    data[10 + 5] = 4;
    drawCounts(image, { bin: 100, b0: 0, nBins: 10, data }, GEOM, style({}));
    expect([49, 50, 59, 60].map((x) => rasterGroupAt(image, x, 10, 0))).toEqual([-1, 1, 1, -1]);
  });
});

describe("rasterGroupAt", () => {
  it("picks the nearest drawn group within the slack", () => {
    const image = newRasterImage(100, 40);
    drawEvents(image, events([505, 805], [0, 1]), GEOM, style({ groupSlot: Float64Array.from([2.5, 7.5]) }));
    expect(rasterGroupAt(image, 52, 11, 3)).toBe(0);
    expect(rasterGroupAt(image, 79, 29, 3)).toBe(1);
    expect(rasterGroupAt(image, 65, 20, 3)).toBe(-1);
  });
});

describe("axis", () => {
  it("steps ticks by 1, 2 or 5 times a power of ten", () => {
    expect([0.003, 0.15, 1, 3, 40, 700].map(niceStep)).toEqual([0.005, 0.2, 1, 5, 50, 1000]);
  });

  it("labels with the decimals the step needs", () => {
    expect(formatSeconds(12, 5)).toBe("12 s");
    expect(formatSeconds(1.25, 0.05)).toBe("1.25 s");
    expect(formatSeconds(0.1, 0.1)).toBe("0.1 s");
  });

  it("steps to the marker past a sample, skipping one under it", () => {
    const samples = Float64Array.from([100, 200, 300]);
    expect(stepMarker(samples, 200, 1, 1)).toBe(2);
    expect(stepMarker(samples, 200, -1, 1)).toBe(0);
    expect(stepMarker(samples, 150, 1, 1)).toBe(1);
    expect(stepMarker(samples, 300, 1, 1)).toBe(-1);
    expect(stepMarker(samples, 100, -1, 1)).toBe(-1);
  });
});

describe("packColour", () => {
  it("reads hex and rgb", () => {
    expect(packColour("#0a5a9c")).toBe(0x0a5a9c);
    expect(packColour("rgba(1, 2, 3, 0.5)")).toBe(0x010203);
    expect(packColour("red")).toBe(-1);
  });
});
