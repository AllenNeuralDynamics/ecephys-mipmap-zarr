/** Least-recently-used map bounded by the summed byte size of its values. */
export class ByteLru<K, V> {
  readonly #entries = new Map<K, { value: V; bytes: number }>();
  #bytes = 0;

  constructor(
    readonly maxBytes: number,
    readonly onEvict?: (key: K, value: V) => void,
  ) {}

  get bytes(): number {
    return this.#bytes;
  }

  get size(): number {
    return this.#entries.size;
  }

  get(key: K): V | undefined {
    const entry = this.#entries.get(key);
    if (entry === undefined) return undefined;
    this.#entries.delete(key);
    this.#entries.set(key, entry);
    return entry.value;
  }

  has(key: K): boolean {
    return this.#entries.has(key);
  }

  /** Insert or replace; a value larger than `maxBytes` is not kept. */
  set(key: K, value: V, bytes: number): void {
    this.delete(key);
    if (bytes > this.maxBytes) return;
    this.#entries.set(key, { value, bytes });
    this.#bytes += bytes;
    for (const [oldest, entry] of this.#entries) {
      if (this.#bytes <= this.maxBytes) break;
      this.#entries.delete(oldest);
      this.#bytes -= entry.bytes;
      this.onEvict?.(oldest, entry.value);
    }
  }

  delete(key: K): boolean {
    const entry = this.#entries.get(key);
    if (entry === undefined) return false;
    this.#entries.delete(key);
    this.#bytes -= entry.bytes;
    return true;
  }
}
