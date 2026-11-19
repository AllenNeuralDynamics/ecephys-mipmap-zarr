import { expect, test } from "vitest";

import { ByteLru } from "../src/reader/cache.js";

test("evicts least recently used entries until under the byte bound", () => {
  const evicted: string[] = [];
  const lru = new ByteLru<string, number>(10, (key) => evicted.push(key));
  lru.set("a", 1, 4);
  lru.set("b", 2, 4);
  lru.get("a");
  lru.set("c", 3, 4);
  expect(evicted).toEqual(["b"]);
  expect(lru.bytes).toBe(8);
  expect(lru.has("a") && lru.has("c")).toBe(true);
});

test("replacing an entry recounts its bytes and an oversized value is not kept", () => {
  const lru = new ByteLru<string, number>(10);
  lru.set("a", 1, 4);
  lru.set("a", 2, 6);
  expect(lru.bytes).toBe(6);
  lru.set("big", 3, 11);
  expect(lru.has("big")).toBe(false);
  expect(lru.get("a")).toBe(2);
});
