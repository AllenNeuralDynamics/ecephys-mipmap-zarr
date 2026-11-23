import type { Port, Request, Response } from "./protocol.js";
import type { StoreInfo } from "./store.js";
import type { Tile } from "./tiles.js";

/** Error a worker reported, carrying the name it had there (`AbortError`, `FormatError`, ...). */
export class ReaderError extends Error {
  constructor(
    override readonly name: string,
    message: string,
  ) {
    super(message);
  }
}

type Waiter = { resolve: (value: Response) => void; reject: (err: Error) => void };

/** Main-thread side of the reader worker: one promise per request, abortable by signal. */
export class ReaderClient {
  readonly #port: Port;
  readonly #waiting = new Map<number, Waiter>();
  #nextId = 1;

  constructor(port: Port) {
    this.#port = port;
    port.addEventListener("message", (event: MessageEvent) => {
      const response = event.data as Response;
      const waiter = this.#waiting.get(response.id);
      if (waiter === undefined) return;
      this.#waiting.delete(response.id);
      if (response.type === "error") waiter.reject(new ReaderError(response.name, response.message));
      else waiter.resolve(response);
    });
    port.start?.();
  }

  #send(request: Exclude<Request, { type: "prefetch" }>, signal?: AbortSignal): Promise<Response> {
    if (signal?.aborted) return Promise.reject(new ReaderError("AbortError", "request aborted"));
    return new Promise((resolve, reject) => {
      this.#waiting.set(request.id, { resolve, reject });
      signal?.addEventListener(
        "abort",
        () => {
          if (!this.#waiting.delete(request.id)) return;
          this.#port.postMessage({ type: "abort", id: request.id } satisfies Request);
          reject(new ReaderError("AbortError", "request aborted"));
        },
        { once: true },
      );
      this.#port.postMessage(request);
    });
  }

  async open(location: string): Promise<StoreInfo> {
    const response = await this.#send({ type: "open", id: this.#nextId++, location });
    if (response.type !== "opened") throw new ReaderError("ProtocolError", `unexpected ${response.type} reply`);
    return response.info;
  }

  /** Warm the worker's cache with `tiles`, behind any requested tile; replaces the last prefetch. */
  prefetch(tiles: readonly { key: string; coords: readonly number[] }[]): void {
    const message: Request = {
      type: "prefetch",
      tiles: tiles.map(({ key, coords }) => ({ key, coords: [...coords] })),
    };
    this.#port.postMessage(message);
  }

  async getTile(key: string, coords: readonly number[], signal?: AbortSignal): Promise<Tile> {
    const response = await this.#send({ type: "tile", id: this.#nextId++, key, coords: [...coords] }, signal);
    if (response.type !== "tile") throw new ReaderError("ProtocolError", `unexpected ${response.type} reply`);
    return response.tile;
  }
}
