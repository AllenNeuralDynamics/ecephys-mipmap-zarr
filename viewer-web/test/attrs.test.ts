import { describe, expect, test } from "vitest";

import { FORMAT_VERSION, FormatError, parseRootAttrs } from "../src/reader/attrs.js";

const ATTRS = {
  sampling_frequency: 30000,
  units: "uV",
  strides: [256, 32],
  n_samples: 1000,
  n_channels: 4,
  dtype: "int16",
  raw_source: null,
  raw_path: null,
  source_metadata: null,
  t_start: 12.5,
  format_version: 1,
};

describe("parseRootAttrs", () => {
  test("reads every field and sorts strides", () => {
    const attrs = parseRootAttrs(ATTRS);
    expect(attrs).toMatchObject({ samplingFrequency: 30000, nSamples: 1000, nChannels: 4, tStart: 12.5 });
    expect(attrs.strides).toEqual([32, 256]);
  });

  test("a store without format_version is version 1 and a null t_start is 0", () => {
    const { format_version: _version, ...old } = ATTRS;
    const attrs = parseRootAttrs({ ...old, t_start: null });
    expect(attrs.formatVersion).toBe(1);
    expect(attrs.tStart).toBe(0);
  });

  test("refuses a newer format version", () => {
    expect(() => parseRootAttrs({ ...ATTRS, format_version: FORMAT_VERSION + 1 })).toThrow(FormatError);
  });

  test("refuses missing or malformed required fields", () => {
    const { n_samples: _nSamples, ...missing } = ATTRS;
    expect(() => parseRootAttrs(missing)).toThrow(/n_samples/);
    expect(() => parseRootAttrs({ ...ATTRS, strides: [32, 0] })).toThrow(/strides/);
    expect(() => parseRootAttrs({ ...ATTRS, n_channels: 2.5 })).toThrow(/n_channels/);
  });
});
