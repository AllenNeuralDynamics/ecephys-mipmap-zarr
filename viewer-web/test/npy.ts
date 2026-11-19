import { readFileSync } from "node:fs";

import type { Codes } from "../src/reader/decode.js";

const CONSTRUCTORS: Record<string, new (buffer: ArrayBuffer) => Codes> = {
  "|i1": Int8Array,
  "|u1": Uint8Array,
  "<i2": Int16Array,
  "<u2": Uint16Array,
  "<i4": Int32Array,
  "<u4": Uint32Array,
  "<f4": Float32Array,
  "<f8": Float64Array,
};

/** Read a C-order little-endian `.npy` file of a numeric dtype. */
export function readNpy(path: string): { shape: number[]; data: Codes } {
  const bytes = readFileSync(path);
  const major = bytes[6] ?? 1;
  const headerLength = major === 1 ? bytes.readUInt16LE(8) : bytes.readUInt32LE(8);
  const headerStart = major === 1 ? 10 : 12;
  const header = bytes.subarray(headerStart, headerStart + headerLength).toString("latin1");
  const descr = /'descr':\s*'([^']+)'/.exec(header)?.[1] ?? "";
  if (/'fortran_order':\s*True/.test(header)) throw new Error(`${path} is Fortran-ordered`);
  const shape = (/'shape':\s*\(([^)]*)\)/.exec(header)?.[1] ?? "")
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s.length > 0)
    .map(Number);
  const Ctor = CONSTRUCTORS[descr];
  if (Ctor === undefined) throw new Error(`${path}: unsupported dtype ${descr}`);
  const start = headerStart + headerLength;
  const data = new Ctor(bytes.buffer.slice(bytes.byteOffset + start, bytes.byteOffset + bytes.length));
  return { shape, data };
}
