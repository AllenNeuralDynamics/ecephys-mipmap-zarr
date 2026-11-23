import { existsSync } from "node:fs";
import { join } from "node:path";
import { MessageChannel } from "node:worker_threads";

import type { AbsolutePath, AsyncReadable, RangeQuery } from "@zarrita/storage";
import FileSystemStore from "@zarrita/storage/fs";
import { describe, expect, test } from "vitest";

import { ReaderClient } from "../src/reader/client.js";
import type { Port } from "../src/reader/protocol.js";
import { serve } from "../src/reader/serve.js";
import { openStore } from "../src/reader/store.js";
import { TileSource } from "../src/reader/tiles.js";

const STORE = join(import.meta.dirname, "fixtures", "int16_raw-shift_pyr-shift.zarr");

/** Counts range reads, optionally holding each until released. */
class CountingStore implements AsyncReadable {
  ranges = 0;
  readonly #inner = new FileSystemStore(STORE);
  gate: Promise<void> | undefined;

  get(key: AbsolutePath): Promise<Uint8Array | undefined> {
    return this.#inner.get(key);
  }

  async getRange(
    key: AbsolutePath,
    range: RangeQuery,
    opts?: { signal?: AbortSignal },
  ): Promise<Uint8Array | undefined> {
    this.ranges++;
    await this.gate;
    opts?.signal?.throwIfAborted();
    return this.#inner.getRange(key, range);
  }
}

describe.skipIf(!existsSync(STORE))("TileSource", () => {
  test("concurrent requests share one fetch and a repeat comes from cache", async () => {
    const store = new CountingStore();
    const source = new TileSource(await openStore(store));
    const [a, b] = await Promise.all([source.getTile("raw", [0, 1]), source.getTile("raw", [0, 1])]);
    const fetched = store.ranges;
    expect(a).toBe(b);
    expect(await source.getTile("raw", [0, 1])).toBe(a);
    expect(store.ranges).toBe(fetched);
    expect(source.cachedBytes).toBe(a.codes.byteLength);
  });

  test("a partial edge tile reports its valid extent", async () => {
    const opened = await openStore(new FileSystemStore(STORE));
    const raw = opened.info.arrays.raw;
    const lastChunk = Math.floor((raw?.shape[1] ?? 0) / 1024);
    const tile = await new TileSource(opened).getTile("raw", [1, lastChunk]);
    expect(tile.shape).toEqual([4, 1024]);
    expect(tile.valid).toEqual([4, 200_000 - lastChunk * 1024]);
  });

  test("aborting one of two waiters leaves the other's fetch running", async () => {
    const store = new CountingStore();
    const source = new TileSource(await openStore(store));
    let release = (): void => undefined;
    store.gate = new Promise((resolve) => (release = resolve));
    const controller = new AbortController();
    const aborted = source.getTile("raw", [0, 2], controller.signal);
    const kept = source.getTile("raw", [0, 2]);
    controller.abort();
    release();
    await expect(aborted).rejects.toThrow(/aborted/);
    expect((await kept).coords).toEqual([0, 2]);
  });

  const rawTiles = (from: number, n: number): { key: string; coords: number[] }[] =>
    Array.from({ length: n }, (_, i) => ({ key: "raw", coords: [0, from + i] }));

  /** Resolves once nothing is fetching: prefetches have no promise to await. */
  const settle = async (store: CountingStore): Promise<void> => {
    for (let last = -1; last !== store.ranges;) {
      last = store.ranges;
      await new Promise((resolve) => setTimeout(resolve, 20));
    }
  };

  test("a prefetched tile is then served from cache", async () => {
    const store = new CountingStore();
    const source = new TileSource(await openStore(store));
    source.prefetch(rawTiles(10, 3));
    await settle(store);
    const fetched = store.ranges;
    await source.getTile("raw", [0, 11]);
    expect(store.ranges).toBe(fetched);
  });

  test("a new prefetch drops queued tiles it no longer names", async () => {
    const store = new CountingStore();
    const source = new TileSource(await openStore(store));
    let release = (): void => undefined;
    store.gate = new Promise((resolve) => (release = resolve));
    source.prefetch(rawTiles(0, 40));
    source.prefetch([]);
    release();
    await settle(store);
    // Only fetches that had started finish: at most the concurrency limit, each a shard index and a chunk read.
    expect(source.cachedBytes).toBeLessThanOrEqual(6 * 4 * 1024 * 2);
  });

  test("a requested tile starts ahead of queued prefetches", async () => {
    const store = new CountingStore();
    const source = new TileSource(await openStore(store));
    let release = (): void => undefined;
    store.gate = new Promise((resolve) => (release = resolve));
    source.prefetch(rawTiles(0, 40));
    const wanted = source.getTile("raw", [0, 100]);
    release();
    const tile = await wanted;
    // The six already running and at most the five beside it can have landed, not all forty.
    expect(source.cachedBytes / tile.codes.byteLength).toBeLessThanOrEqual(12);
  });
});

describe.skipIf(!existsSync(STORE))("worker protocol", () => {
  const connect = (): ReaderClient => {
    const { port1, port2 } = new MessageChannel();
    serve(port2 as unknown as Port, () => openStore(new FileSystemStore(STORE)));
    return new ReaderClient(port1 as unknown as Port);
  };

  test("opens, then returns tiles whose codes match an in-process read", async () => {
    const client = connect();
    const info = await client.open(STORE);
    expect(info.strides).toEqual([32, 256, 2048]);
    const tile = await client.getTile("minmax/s32", [1, 0, 3]);
    const local = await new TileSource(await openStore(new FileSystemStore(STORE))).getTile("minmax/s32", [1, 0, 3]);
    expect(Array.from(tile.codes)).toEqual(Array.from(local.codes));
  });

  test("reports errors under their own name and honours abort", async () => {
    const client = connect();
    await client.open(STORE);
    await expect(client.getTile("minmax/s7", [0, 0, 0])).rejects.toThrow(/no array/);
    const controller = new AbortController();
    const pending = client.getTile("raw", [0, 0], controller.signal);
    controller.abort();
    await expect(pending).rejects.toMatchObject({ name: "AbortError" });
  });
});
