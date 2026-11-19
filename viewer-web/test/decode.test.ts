import { describe, expect, test } from "vitest";

import { decodeCodes, decodeFromAttrs } from "../src/reader/decode.js";

describe("decodeFromAttrs", () => {
  test("quantization wins over lsb_shift", () => {
    const decode = decodeFromAttrs({ quantization: { offset: [1, 2], scale: [0.5, 0.25] }, lsb_shift: 2 }, 2, false);
    expect(decode.kind).toBe("affine");
  });

  test("a level's quantization keeps min and max offsets apart", () => {
    const decode = decodeFromAttrs({ quantization: { min_offset: [1], max_offset: [9], scale: [2] } }, 1, true);
    expect(decode).toEqual({
      kind: "affine-minmax",
      scale: Float32Array.of(2),
      minOffset: Float32Array.of(1),
      maxOffset: Float32Array.of(9),
    });
  });

  test("lsb_shift, then identity; other attrs are ignored", () => {
    expect(decodeFromAttrs({ lsb_shift: 3, source_dtype: "int16" }, 1, false)).toEqual({ kind: "shift", bits: 3 });
    expect(decodeFromAttrs({ lsb_shift: 0 }, 1, false)).toEqual({ kind: "none" });
    expect(decodeFromAttrs({ original_dtype: "float32" }, 1, true)).toEqual({ kind: "none" });
  });

  test("refuses per-channel lists of the wrong length", () => {
    expect(() => decodeFromAttrs({ quantization: { offset: [1], scale: [1] } }, 2, false)).toThrow(
      /quantization.scale/,
    );
  });
});

describe("decodeCodes", () => {
  test("applies each channel's parameters from c0 and each row's offset", () => {
    const decode = decodeFromAttrs(
      { quantization: { min_offset: [0, 10, 20], max_offset: [0, 100, 200], scale: [1, 2, 3] } },
      3,
      true,
    );
    // channels 1 and 2, rows min and max, two blocks each
    const values = decodeCodes(Uint8Array.of(1, 2, 3, 4, 5, 6, 7, 8), decode, 1, 2, 2);
    expect(Array.from(values)).toEqual([12, 14, 106, 108, 35, 38, 221, 224]);
  });

  test("shift multiplies by 2**bits, keeping sign", () => {
    expect(Array.from(decodeCodes(Int8Array.of(-3, 5), { kind: "shift", bits: 2 }, 0, 1, 1))).toEqual([-12, 20]);
  });
});
