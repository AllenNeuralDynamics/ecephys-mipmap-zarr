import PQueue from "p-queue";

import { ByteLru } from "./cache.js";
import { decodeCodes, type Codes } from "./decode.js";
import { axes } from "./plan.js";
import type { ArrayInfo, OpenedStore, ZArray } from "./store.js";

export const DEFAULT_CACHE_BYTES = 512 * 1024 ** 2;
/**
 * Fetches running at once: a browser's HTTP/1.1 connections per host. Past it, requests wait inside the browser,
 * where a visible tile cannot overtake a prefetch.
 */
const CONCURRENT_FETCHES = 6;
const VISIBLE = 1;
const PREFETCH = 0;

/** One inner chunk's stored codes, C-order over `shape`; past `valid` along any axis is fill. */
export interface Tile {
  key: string;
  coords: number[];
  shape: number[];
  valid: number[];
  codes: Codes;
}

const tileId = (key: string, coords: readonly number[]): string => `${key}|${coords.join(",")}`;

interface InFlight {
  promise: Promise<Tile>;
  controller: AbortController;
  /** Callers awaiting it; a prefetch has none until a caller joins. */
  waiters: number;
  priority: number;
}

function abortError(): DOMException {
  return new DOMException("tile request aborted", "AbortError");
}

/**
 * Fetches tiles behind a byte-bounded LRU, sharing one fetch among concurrent requests for a tile. Requested tiles
 * start before prefetched ones, a few at a time.
 */
export class TileSource {
  readonly #arrays: ReadonlyMap<string, ZArray>;
  readonly #cache: ByteLru<string, Tile>;
  readonly #inFlight = new Map<string, InFlight>();
  readonly #queue = new PQueue({ concurrency: CONCURRENT_FETCHES });
  /** Tiles the last `prefetch` asked for. */
  #prefetched = new Set<string>();

  constructor(store: OpenedStore, maxBytes: number = DEFAULT_CACHE_BYTES) {
    this.#arrays = store.arrays;
    this.#cache = new ByteLru(maxBytes);
  }

  get cachedBytes(): number {
    return this.#cache.bytes;
  }

  /** Resolve the tile at inner-chunk `coords` of array `key`; rejects with an `AbortError` if `signal` fires first. */
  getTile(key: string, coords: readonly number[], signal?: AbortSignal): Promise<Tile> {
    const id = tileId(key, coords);
    const cached = this.#cache.get(id);
    if (cached !== undefined) return Promise.resolve(cached);
    if (signal?.aborted) return Promise.reject(abortError());

    const shared = this.#start(id, key, coords, VISIBLE);
    shared.waiters++;
    if (signal === undefined) return shared.promise;
    return new Promise<Tile>((resolve, reject) => {
      const onAbort = (): void => {
        if (--shared.waiters === 0) shared.controller.abort();
        reject(abortError());
      };
      signal.addEventListener("abort", onAbort, { once: true });
      shared.promise.then(
        (tile) => {
          signal.removeEventListener("abort", onAbort);
          resolve(tile);
        },
        (err: unknown) => {
          signal.removeEventListener("abort", onAbort);
          reject(err instanceof Error ? err : new Error(String(err)));
        },
      );
    });
  }

  /**
   * Fetch `tiles` into the cache behind any requested tile, and drop queued prefetches no longer among them.
   * Nothing is returned; a later `getTile` finds the tile cached or joins its fetch.
   */
  prefetch(tiles: readonly { key: string; coords: readonly number[] }[]): void {
    const wanted = new Set(tiles.map((t) => tileId(t.key, t.coords)));
    for (const id of this.#prefetched) {
      const entry = this.#inFlight.get(id);
      if (!wanted.has(id) && entry?.waiters === 0) entry.controller.abort();
    }
    this.#prefetched = wanted;
    for (const { key, coords } of tiles) {
      const id = tileId(key, coords);
      if (this.#cache.has(id)) continue;
      // Failures surface when a caller asks for the tile.
      this.#start(id, key, coords, PREFETCH).promise.catch(() => undefined);
    }
  }

  /** The in-flight fetch of `id`, started or raised to `priority`. */
  #start(id: string, key: string, coords: readonly number[], priority: number): InFlight {
    const existing = this.#inFlight.get(id);
    if (existing !== undefined) {
      if (priority > existing.priority) {
        existing.priority = priority;
        try {
          this.#queue.setPriority(id, priority);
        } catch {
          // Already running.
        }
      }
      return existing;
    }
    const controller = new AbortController();
    const signal = controller.signal;
    const promise = this.#queue
      .add(({ signal: taskSignal }) => this.#fetch(key, [...coords], taskSignal ?? signal), { priority, id, signal })
      .then(
        (tile) => {
          this.#inFlight.delete(id);
          this.#cache.set(id, tile, tile.codes.byteLength);
          return tile;
        },
        (err: unknown) => {
          this.#inFlight.delete(id);
          throw err instanceof Error ? err : new Error(String(err));
        },
      );
    const entry = { promise, controller, waiters: 0, priority };
    this.#inFlight.set(id, entry);
    return entry;
  }

  async #fetch(key: string, coords: number[], signal: AbortSignal): Promise<Tile> {
    const arr = this.#arrays.get(key);
    if (arr === undefined) throw new Error(`store has no array ${key}`);
    const chunk = await arr.getChunk(coords, { signal });
    const shape = arr.chunks;
    const valid = shape.map((n, axis) => Math.max(0, Math.min(n, (arr.shape[axis] ?? 0) - (coords[axis] ?? 0) * n)));
    return { key, coords, shape, valid, codes: chunk.data as Codes };
  }
}

/** A tile's valid extent as codes, channel-first: `(C, T)` for raw, `(C, 2, B)` for a level, whatever order the array stores. */
export function channelFirstCodes(tile: Tile, array: ArrayInfo): { shape: number[]; c0: number; codes: Codes } {
  const { channel, time } = axes(array);
  const nC = tile.valid[channel] ?? 0;
  const nT = tile.valid[time] ?? 0;
  const nRows = array.shape.length === 3 ? 2 : 1;
  const strides = tile.shape.map((_, axis) => tile.shape.slice(axis + 1).reduce((a, b) => a * b, 1));
  const channelStride = strides[channel] ?? 0;
  const rowStride = nRows === 2 ? (strides[1] ?? 0) : 0;
  const timeStride = strides[time] ?? 0;
  const codes = new (tile.codes.constructor as new (n: number) => Codes)(nC * nRows * nT);
  for (let c = 0; c < nC; c++) {
    for (let r = 0; r < nRows; r++) {
      const base = c * channelStride + r * rowStride;
      const out = (c * nRows + r) * nT;
      for (let t = 0; t < nT; t++) codes[out + t] = tile.codes[base + t * timeStride] ?? 0;
    }
  }
  const c0 = (tile.coords[channel] ?? 0) * (tile.shape[channel] ?? 0);
  return { shape: nRows === 2 ? [nC, 2, nT] : [nC, nT], c0, codes };
}

/** Decode a tile's valid extent to float32 in `channelFirstCodes` layout; for tests and a CPU fallback. */
export function decodeTile(tile: Tile, array: ArrayInfo): { shape: number[]; values: Float32Array } {
  const { shape, c0, codes } = channelFirstCodes(tile, array);
  return { shape, values: decodeCodes(codes, array.decode, c0, shape[0] ?? 0, shape.length === 3 ? 2 : 1) };
}
