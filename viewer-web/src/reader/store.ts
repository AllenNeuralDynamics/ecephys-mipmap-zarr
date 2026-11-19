import * as zarr from "zarrita";
import type { AsyncReadable } from "@zarrita/storage";

import { FormatError, parseRootAttrs, type RootAttrs } from "./attrs.js";
import { decodeFromAttrs, type Decode } from "./decode.js";

export type ZArray = zarr.Array<zarr.DataType, AsyncReadable>;

/** One array a reader draws from, keyed `raw` or `minmax/s{stride}` whatever store holds it. */
export interface ArrayInfo {
  key: string;
  /** Stored shape: `(C, T)` or `(T, C)` for raw, `(C, 2, B)` for a level. */
  shape: number[];
  /** Inner chunk shape, the unit one tile request fetches. */
  chunks: number[];
  dtype: string;
  /** Raw stored `(T, C)`, as SpikeInterface writes it. */
  timeMajor: boolean;
  /** Samples per block, or 1 for raw. */
  stride: number;
  decode: Decode;
}

/** A channel property: numbers as a flat typed array of `shape`, or one string per channel. */
export interface Property {
  shape: number[];
  values: Float64Array | Int32Array | Uint8Array | Float32Array | Int16Array | Uint16Array | Int8Array | string[];
}

export interface StoreInfo extends RootAttrs {
  channelIds: string[];
  properties: Record<string, Property>;
  arrays: Record<string, ArrayInfo>;
  /** Why a pyramid-only store's raw samples could not be opened; raw is then absent from `arrays`. */
  rawError: string | null;
}

export interface OpenedStore {
  info: StoreInfo;
  arrays: ReadonlyMap<string, ZArray>;
}

export interface OpenOptions {
  /** Maps a URI the store names (its location or `raw_source`) to a fetchable URL. */
  resolveUrl?: (uri: string) => string;
  /** Builds the store for a resolved location; defaults to HTTP fetch. */
  storeFor?: (location: string) => AsyncReadable;
}

/** Rewrites `s3://bucket/key` to the bucket's virtual-hosted HTTPS URL; leaves other URIs alone. */
export function defaultResolveUrl(uri: string): string {
  const match = /^s3:\/\/([^/]+)\/?(.*)$/.exec(uri);
  if (!match) return uri;
  const [, bucket = "", key = ""] = match;
  return `https://${bucket}.s3.amazonaws.com/${key}`;
}

const RAW_PATHS = ["raw", "traces_seg0"];

export const levelKey = (stride: number): string => `minmax/s${String(stride)}`;

function toStrings(data: unknown): string[] {
  return Array.from(data as Iterable<unknown>, (v) => String(v));
}

async function readProperty(arr: ZArray): Promise<Property> {
  const chunk = await zarr.get(arr);
  const data: unknown = chunk.data;
  let values: Property["values"];
  if (data instanceof BigInt64Array || data instanceof BigUint64Array) {
    values = Float64Array.from(data, Number);
  } else if (
    data instanceof Float64Array ||
    data instanceof Float32Array ||
    data instanceof Int32Array ||
    data instanceof Int16Array ||
    data instanceof Int8Array ||
    data instanceof Uint16Array ||
    data instanceof Uint8Array
  ) {
    values = data;
  } else if (data instanceof zarr.BoolArray) {
    values = Uint8Array.from(data, (v) => (v ? 1 : 0));
  } else {
    values = toStrings(data);
  }
  return { shape: chunk.shape, values };
}

function arrayInfo(arr: ZArray, key: string, nChannels: number, stride: number | null): ArrayInfo {
  const shape = arr.shape;
  const timeMajor = stride === null && shape.length === 2 && (shape[0] ?? 0) > (shape[1] ?? 0);
  return {
    key,
    shape,
    chunks: arr.chunks,
    dtype: arr.dtype,
    timeMajor,
    stride: stride ?? 1,
    decode: decodeFromAttrs(arr.attrs, nChannels, stride !== null),
  };
}

async function openRaw(location: string, rawPath: string | null, opts: Required<OpenOptions>): Promise<ZArray> {
  const store = await zarr.withMaybeConsolidatedMetadata(opts.storeFor(opts.resolveUrl(location)));
  const root = zarr.root(store);
  for (const path of rawPath === null ? RAW_PATHS : [rawPath]) {
    try {
      return await zarr.open(root.resolve(path), { kind: "array" });
    } catch (err) {
      if (!(err instanceof zarr.NotFoundError)) throw err;
    }
  }
  throw new FormatError(`no raw array at ${rawPath ?? RAW_PATHS.join(" or ")} in ${location}`);
}

/**
 * Open a mipmap store and read everything a viewer needs before its first tile: attrs, each array's
 * decode, channel ids and properties. Properties are listed from consolidated metadata, so a store
 * without it opens with none.
 */
export async function openStore(location: string | AsyncReadable, options: OpenOptions = {}): Promise<OpenedStore> {
  const opts: Required<OpenOptions> = {
    resolveUrl: options.resolveUrl ?? defaultResolveUrl,
    storeFor: options.storeFor ?? ((url) => new zarr.FetchStore(url)),
  };
  const base = typeof location === "string" ? opts.storeFor(opts.resolveUrl(location)) : location;
  const store = await zarr.withMaybeConsolidatedMetadata(base, { format: "v3" });
  const group = await zarr.open.v3(store, { kind: "group" });
  const attrs = parseRootAttrs(group.attrs);
  const root = zarr.root(store);

  const arrays = new Map<string, ZArray>();
  const infos: Record<string, ArrayInfo> = {};
  const add = (arr: ZArray, key: string, stride: number | null): void => {
    arrays.set(key, arr);
    infos[key] = arrayInfo(arr, key, attrs.nChannels, stride);
  };

  let rawError: string | null = null;
  if (attrs.rawSource === null) {
    add(await zarr.open.v3(root.resolve("raw"), { kind: "array" }), "raw", null);
  } else {
    try {
      add(await openRaw(attrs.rawSource, attrs.rawPath, opts), "raw", null);
    } catch (err) {
      rawError = err instanceof Error ? err.message : String(err);
    }
  }
  await Promise.all(
    attrs.strides.map(async (stride) => {
      add(await zarr.open.v3(root.resolve(levelKey(stride)), { kind: "array" }), levelKey(stride), stride);
    }),
  );

  const contents = "contents" in store ? store.contents() : [];
  const propertyPaths = contents
    .filter((node) => node.kind === "array" && node.path.startsWith("/properties/"))
    .map((node) => node.path);
  const properties: Record<string, Property> = {};
  await Promise.all(
    propertyPaths.map(async (path) => {
      properties[path.slice("/properties/".length)] = await readProperty(
        await zarr.open.v3(root.resolve(path), { kind: "array" }),
      );
    }),
  );

  let channelIds: string[];
  try {
    const ids = await zarr.get(await zarr.open.v3(root.resolve("channel_ids"), { kind: "array" }));
    channelIds = toStrings(ids.data);
  } catch (err) {
    if (!(err instanceof zarr.NotFoundError)) throw err;
    channelIds = Array.from({ length: attrs.nChannels }, (_, c) => String(c + 1));
  }

  return { info: { ...attrs, channelIds, properties, arrays: infos, rawError }, arrays };
}
