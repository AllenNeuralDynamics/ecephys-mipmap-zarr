import { join } from "node:path";

import FileSystemStore from "@zarrita/storage/fs";
import { beforeAll, describe, expect, it } from "vitest";

import { EventLayer, lowerBound, upperBound } from "../src/events/layer.js";
import { assignColours, buildTint, sampleMapping, tintStrength, type TintContext } from "../src/events/tint.js";
import type { TintRequest } from "../src/render/renderer.js";

const FIXTURE = join(import.meta.dirname, "fixtures", "events.zarr");
const N_CHANNELS = 8;
const spikeAt = (channel: number): number => 5_000 + 20_011 * channel;
/** The fixture's channel ids run CH8..CH1 then CH99, which no store holds. */
const STORE_CHANNEL = Int32Array.from([7, 6, 5, 4, 3, 2, 1, 0, -1]);

let layer: EventLayer;
beforeAll(async () => {
  layer = await EventLayer.open(new FileSystemStore(FIXTURE));
});

describe("EventLayer", () => {
  it("reads metadata and groups", () => {
    const { info } = layer;
    expect(info.kind).toBe("spikes");
    expect(info.tStart).toBeNull();
    expect(info.channelIds.at(-1)).toBe("CH99");
    expect(info.nEvents).toBe(N_CHANNELS + 20);
    expect(info.groups.ids.at(-1)).toBe("extra");
    expect(info.groups.k).toBe(3);
    expect(info.groups.properties.decoder_label?.values).toEqual([...Array<string>(N_CHANNELS).fill("sua"), "mua"]);
  });

  it("returns every event in order across chunks", async () => {
    const all = await layer.window(0, 1e9);
    expect(all.sampleIndex.length).toBe(N_CHANNELS + 20);
    expect([...all.sampleIndex]).toEqual([...all.sampleIndex].sort((a, b) => a - b));
    expect(all.amplitude?.length).toBe(all.sampleIndex.length);
  });

  it("returns only the events inside a window, half-open", async () => {
    const one = await layer.window(spikeAt(2), spikeAt(2) + 1);
    expect([...one.sampleIndex]).toEqual([spikeAt(2)]);
    expect([...one.group]).toEqual([2]);
    expect((await layer.window(spikeAt(2) + 1, spikeAt(3))).group).not.toContain(3);
    expect((await layer.window(5, 5)).sampleIndex.length).toBe(0);
  });
});

describe("search", () => {
  it("bounds sorted values", () => {
    const values = [1, 3, 3, 7];
    expect(lowerBound(values, 3)).toBe(1);
    expect(upperBound(values, 3)).toBe(3);
    expect(lowerBound(values, 8)).toBe(4);
  });
});

describe("colours", () => {
  it("separates groups whose footprints share a channel", () => {
    const colours = assignColours(layer.info.groups, 7);
    // Each unit's footprint includes the next channel's, so neighbours differ.
    for (let g = 0; g + 1 < N_CHANNELS; g++) expect(colours[g]).not.toBe(colours[g + 1]);
    expect(Math.max(...colours)).toBeLessThan(7);
  });

  it("fades between 10 and 30 s", () => {
    expect(tintStrength(1)).toBe(1);
    expect(tintStrength(20)).toBeCloseTo(0.5);
    expect(tintStrength(40)).toBe(0);
  });

  it("maps samples by clock only when the layer states its start", () => {
    expect(sampleMapping({ samplingFrequency: 2500, tStart: null }, { samplingFrequency: 30000, tStart: 5 })).toEqual({
      scale: 12,
      offset: 0,
    });
    expect(sampleMapping({ samplingFrequency: 30000, tStart: 6 }, { samplingFrequency: 30000, tStart: 5 })).toEqual({
      scale: 1,
      offset: 30000,
    });
  });
});

describe("buildTint", () => {
  const context = (): TintContext => ({
    info: layer.info,
    storeChannel: STORE_CHANNEL,
    colour: assignColours(layer.info.groups, 7),
    depth: null,
    scale: 1,
    offset: 0,
  });
  /** 100 columns of 2 samples around channel 3's spike (index 2), showing store channels 2, 3 and 5. */
  const request: TintRequest = { t0: spikeAt(2) - 100, samplesPerPx: 2, columns: 100, channels: [2, 3, 5] };
  const cell = (tint: Uint8Array, row: number, column: number): [number, number] => [
    tint[2 * (row * request.columns + column)] ?? 0,
    tint[2 * (row * request.columns + column) + 1] ?? 0,
  ];

  it("colours the footprint's rows over the event's window and a column either side", async () => {
    const ctx = context();
    const window = await layer.window(request.t0 - 100, request.t0 + 300);
    const tint = buildTint(request, window, ctx, N_CHANNELS, new Set(), 1);
    if (tint === undefined) throw new Error("no tint");
    const colour = (ctx.colour[2] ?? -1) + 1;
    // Window -0.5..+1 ms is samples -15..+30, columns 42..65, widened to 41..65 inclusive.
    expect(cell(tint, 0, 50)).toEqual([colour, 255]);
    expect(cell(tint, 1, 50)).toEqual([colour, 255]);
    expect(cell(tint, 0, 41)[0]).toBe(colour);
    expect(cell(tint, 0, 65)[0]).toBe(colour);
    expect(cell(tint, 0, 40)).toEqual([0, 0]);
    expect(cell(tint, 0, 66)).toEqual([0, 0]);
    expect(cell(tint, 2, 50)).toEqual([0, 0]);
  });

  it("scales strength, and with a selection colours only it at full strength", async () => {
    const window = await layer.window(request.t0 - 100, request.t0 + 300);
    expect(cell(buildTint(request, window, context(), N_CHANNELS, new Set(), 0.5) ?? new Uint8Array(), 0, 50)[1]).toBe(
      128,
    );
    expect(buildTint(request, window, context(), N_CHANNELS, new Set([5]), 0.5)).toBeUndefined();
    expect(cell(buildTint(request, window, context(), N_CHANNELS, new Set([2]), 0) ?? new Uint8Array(), 0, 50)[1]).toBe(
      255,
    );
  });

  it("gives an overlapped pixel to the event with the larger weight there", async () => {
    // Channel 3 (index 2) carries unit 3 at 700 and unit 2 at 100; put both spikes in one window.
    const window = await layer.window(0, 1e9);
    const shifted = Float64Array.from(window.sampleIndex, (s) => (s === spikeAt(1) ? spikeAt(2) : s));
    const ctx = context();
    const tint = buildTint(request, { ...window, sampleIndex: shifted }, ctx, N_CHANNELS, new Set(), 1);
    if (tint === undefined) throw new Error("no tint");
    expect(cell(tint, 0, 50)[0]).toBe((ctx.colour[2] ?? -1) + 1);
  });
});
