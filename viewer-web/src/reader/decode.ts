import { FormatError } from "./attrs.js";

/** How an array's stored codes become values, from its attributes (`docs/store-format.md`). */
export type Decode =
  | { kind: "affine"; scale: Float32Array; offset: Float32Array }
  | { kind: "affine-minmax"; scale: Float32Array; minOffset: Float32Array; maxOffset: Float32Array }
  | { kind: "shift"; bits: number }
  | { kind: "none" };

function perChannel(q: Record<string, unknown>, key: string, nChannels: number): Float32Array {
  const values = q[key];
  if (!Array.isArray(values) || values.length !== nChannels || !values.every((v) => typeof v === "number")) {
    throw new FormatError(`quantization.${key} is not ${String(nChannels)} numbers`);
  }
  return Float32Array.from(values);
}

/**
 * Derive the decode for one array: `quantization` wins, then `lsb_shift`, else identity.
 *
 * `minmax` selects the pyramid form of `quantization` (separate offsets for the min and max rows).
 */
export function decodeFromAttrs(attrs: Record<string, unknown>, nChannels: number, minmax: boolean): Decode {
  const q = attrs.quantization;
  if (q !== undefined && q !== null) {
    if (typeof q !== "object") throw new FormatError("quantization is not an object");
    const qd = q as Record<string, unknown>;
    const scale = perChannel(qd, "scale", nChannels);
    if (minmax) {
      return {
        kind: "affine-minmax",
        scale,
        minOffset: perChannel(qd, "min_offset", nChannels),
        maxOffset: perChannel(qd, "max_offset", nChannels),
      };
    }
    return { kind: "affine", scale, offset: perChannel(qd, "offset", nChannels) };
  }
  const shift = attrs.lsb_shift;
  if (shift !== undefined && shift !== null) {
    if (typeof shift !== "number" || !Number.isInteger(shift) || shift < 0) {
      throw new FormatError("lsb_shift is not a non-negative integer");
    }
    return shift === 0 ? { kind: "none" } : { kind: "shift", bits: shift };
  }
  return { kind: "none" };
}

/** Numeric typed arrays a tile's codes can arrive in. */
export type Codes =
  Int8Array | Uint8Array | Int16Array | Uint16Array | Int32Array | Uint32Array | Float32Array | Float64Array;

/**
 * Decode codes laid out channel-first, `nRows` rows per channel, starting at channel `c0`.
 *
 * A raw tile has one row per channel; a pyramid tile has two, min then max. Arithmetic is float32 like the
 * GPU's; this exists for tests and as a CPU fallback.
 */
export function decodeCodes(codes: Codes, decode: Decode, c0: number, nChannels: number, nRows: number): Float32Array {
  const out = Float32Array.from(codes);
  const rowLength = out.length / (nChannels * nRows);
  if (decode.kind === "none") return out;
  if (decode.kind === "shift") {
    const factor = 2 ** decode.bits;
    for (let i = 0; i < out.length; i++) out[i] = (out[i] ?? 0) * factor;
    return out;
  }
  for (let c = 0; c < nChannels; c++) {
    const scale = decode.scale[c0 + c] ?? NaN;
    for (let r = 0; r < nRows; r++) {
      const offset =
        decode.kind === "affine"
          ? (decode.offset[c0 + c] ?? NaN)
          : ((r === 0 ? decode.minOffset : decode.maxOffset)[c0 + c] ?? NaN);
      const start = (c * nRows + r) * rowLength;
      for (let i = start; i < start + rowLength; i++) out[i] = Math.fround((out[i] ?? 0) * scale) + offset;
    }
  }
  return out;
}
