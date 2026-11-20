import type { Codes } from "../reader/decode.js";
import type { Decode } from "../reader/decode.js";
import type { ArrayInfo } from "../reader/store.js";
import { channelFirstCodes, type Tile } from "../reader/tiles.js";

/** Texture width rows wrap at; WebGL2 guarantees 2048, every GPU this targets gives at least 4096. */
export const ROW_WIDTH = 4096;

export type SamplerKind = "int" | "uint" | "float";

/**
 * A tile on the GPU. The tile's series (one per channel for raw, min then max per channel for a level) each
 * span `rowsPerSeries` texture rows of `width` texels, so element `i` of series `s` is at
 * `(i % width, s * rowsPerSeries + floor(i / width))`.
 */
export interface TileTexture {
  texture: WebGLTexture;
  sampler: SamplerKind;
  width: number;
  rowsPerSeries: number;
  /** First channel and count of valid channels. */
  c0: number;
  nChannels: number;
  /** Index of the tile's first element along the array's time axis, and how many are valid. */
  start: number;
  length: number;
  bytes: number;
}

interface Format {
  internal: number;
  format: number;
  type: number;
  sampler: SamplerKind;
}

function formatFor(gl: WebGL2RenderingContext, codes: Codes): { format: Format; data: Codes } {
  if (codes instanceof Int8Array)
    return { format: { internal: gl.R8I, format: gl.RED_INTEGER, type: gl.BYTE, sampler: "int" }, data: codes };
  if (codes instanceof Uint8Array)
    return {
      format: { internal: gl.R8UI, format: gl.RED_INTEGER, type: gl.UNSIGNED_BYTE, sampler: "uint" },
      data: codes,
    };
  if (codes instanceof Int16Array)
    return { format: { internal: gl.R16I, format: gl.RED_INTEGER, type: gl.SHORT, sampler: "int" }, data: codes };
  if (codes instanceof Uint16Array)
    return {
      format: { internal: gl.R16UI, format: gl.RED_INTEGER, type: gl.UNSIGNED_SHORT, sampler: "uint" },
      data: codes,
    };
  if (codes instanceof Int32Array)
    return { format: { internal: gl.R32I, format: gl.RED_INTEGER, type: gl.INT, sampler: "int" }, data: codes };
  if (codes instanceof Uint32Array)
    return {
      format: { internal: gl.R32UI, format: gl.RED_INTEGER, type: gl.UNSIGNED_INT, sampler: "uint" },
      data: codes,
    };
  const floats = codes instanceof Float32Array ? codes : Float32Array.from(codes);
  return { format: { internal: gl.R32F, format: gl.RED, type: gl.FLOAT, sampler: "float" }, data: floats };
}

/** Upload a tile's codes as an integer (or float) texture, reshaping long series into rows of `ROW_WIDTH`. */
export function uploadTile(gl: WebGL2RenderingContext, tile: Tile, array: ArrayInfo): TileTexture {
  const timeAxis = array.timeMajor ? 0 : array.shape.length - 1;
  const channelAxis = array.timeMajor ? 1 : 0;
  const nRows = array.shape.length === 3 ? 2 : 1;
  let codes = tile.codes;
  let seriesLength = tile.shape[timeAxis] ?? 0;
  let nChannelsStored = tile.shape[channelAxis] ?? 0;
  if (array.timeMajor) {
    // A (T, C) tile has no contiguous series; transpose its valid extent to channel-first.
    const cf = channelFirstCodes(tile, array);
    codes = cf.codes;
    seriesLength = cf.shape[cf.shape.length - 1] ?? 0;
    nChannelsStored = cf.shape[0] ?? 0;
  }
  const width = Math.min(seriesLength, ROW_WIDTH);
  const rowsPerSeries = Math.ceil(seriesLength / width);
  const nSeries = nChannelsStored * nRows;
  if (rowsPerSeries * width !== seriesLength) {
    const padded = new (codes.constructor as new (n: number) => Codes)(nSeries * rowsPerSeries * width);
    for (let s = 0; s < nSeries; s++) {
      padded.set(codes.subarray(s * seriesLength, (s + 1) * seriesLength), s * rowsPerSeries * width);
    }
    codes = padded;
  }
  const { format, data } = formatFor(gl, codes);
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.texImage2D(gl.TEXTURE_2D, 0, format.internal, width, nSeries * rowsPerSeries, 0, format.format, format.type, data);
  for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, p, gl.NEAREST);
  for (const p of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T]) gl.texParameteri(gl.TEXTURE_2D, p, gl.CLAMP_TO_EDGE);
  return {
    texture,
    sampler: format.sampler,
    width,
    rowsPerSeries,
    c0: (tile.coords[channelAxis] ?? 0) * (array.chunks[channelAxis] ?? 0),
    nChannels: tile.valid[channelAxis] ?? 0,
    start: (tile.coords[timeAxis] ?? 0) * (array.chunks[timeAxis] ?? 0),
    length: tile.valid[timeAxis] ?? 0,
    bytes: data.byteLength,
  };
}

/** Per-channel decode as one RGBA32F row: scale, offset (or min offset), max offset. */
export function uploadDecode(gl: WebGL2RenderingContext, decode: Decode, nChannels: number): WebGLTexture {
  const texels = new Float32Array(nChannels * 4);
  for (let c = 0; c < nChannels; c++) {
    let scale = 1;
    let minOffset = 0;
    let maxOffset = 0;
    if (decode.kind === "shift") scale = 2 ** decode.bits;
    else if (decode.kind === "affine") {
      scale = decode.scale[c] ?? 1;
      minOffset = maxOffset = decode.offset[c] ?? 0;
    } else if (decode.kind === "affine-minmax") {
      scale = decode.scale[c] ?? 1;
      minOffset = decode.minOffset[c] ?? 0;
      maxOffset = decode.maxOffset[c] ?? 0;
    }
    texels.set([scale, minOffset, maxOffset, 0], c * 4);
  }
  return floatRow(gl, texels, nChannels, gl.RGBA32F, gl.RGBA);
}

/** Upload one row of float texels, `nChannels` wide. */
export function floatRow(
  gl: WebGL2RenderingContext,
  texels: Float32Array,
  nChannels: number,
  internal: number,
  format: number,
  texture: WebGLTexture = gl.createTexture(),
): WebGLTexture {
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.texImage2D(gl.TEXTURE_2D, 0, internal, nChannels, 1, 0, format, gl.FLOAT, texels);
  for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, p, gl.NEAREST);
  return texture;
}
