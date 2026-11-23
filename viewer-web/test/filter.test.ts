import { describe, expect, it } from "vitest";

import { groupsWhere, parseGroupFilter } from "../src/events/filter.js";
import type { EventGroups } from "../src/events/layer.js";
import { assignColours } from "../src/events/tint.js";

const groups: EventGroups = {
  ids: ["a", "b", "c", "d"],
  footprintChannel: Int32Array.from([0, 0, 1, 1]),
  footprintWeight: Float32Array.from([1, 1, 1, 1]),
  k: 1,
  properties: {
    KSLabel: { shape: [4], values: ["good", "mua", "good", "good"] },
    decoder_label: { shape: [4], values: ["sua", "sua", "mua", "noise"] },
    default_qc: { shape: [4], values: Uint8Array.from([1, 1, 0, 1]) },
  },
};

describe("group filters", () => {
  it("parse properties joined by commas and values joined by bars", () => {
    expect(parseGroupFilter("KSLabel:good,decoder_label:sua|mua,bad,:x,empty:")).toEqual({
      KSLabel: ["good"],
      decoder_label: ["sua", "mua"],
    });
  });

  it("pass a group only when every property holds one of its values", () => {
    expect(groupsWhere(groups, parseGroupFilter("KSLabel:good"))).toEqual(["a", "c", "d"]);
    expect(groupsWhere(groups, parseGroupFilter("KSLabel:good,decoder_label:sua|mua"))).toEqual(["a", "c"]);
    expect(groupsWhere(groups, {})).toEqual(["a", "b", "c", "d"]);
  });

  it("match stored booleans by true and false, and pass nothing on a property the layer lacks", () => {
    expect(groupsWhere(groups, parseGroupFilter("default_qc:True"))).toEqual(["a", "b", "d"]);
    expect(groupsWhere(groups, parseGroupFilter("default_qc:false"))).toEqual(["c"]);
    expect(groupsWhere(groups, parseGroupFilter("KSlabel:good"))).toEqual([]);
  });
});

describe("colours among shown groups", () => {
  it("leave hidden groups uncoloured and out of their neighbours' way", () => {
    // a and b share channel 0, c and d share channel 1; with two colours each pair splits.
    const all = assignColours(groups, 2, null);
    expect(all[0]).not.toBe(all[1]);
    const shown = assignColours(groups, 1, null, new Set([0, 2]));
    expect(Array.from(shown)).toEqual([0, -1, 0, -1]);
  });
});
