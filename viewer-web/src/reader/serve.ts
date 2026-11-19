import type { Port, Request, Response } from "./protocol.js";
import type { OpenedStore } from "./store.js";
import { TileSource } from "./tiles.js";

/**
 * Answer reader requests arriving on `port`. Tiles go out as copies with their buffers transferred, since the
 * cached tile must stay readable.
 */
export function serve(port: Port, open: (location: string) => Promise<OpenedStore>, maxBytes?: number): void {
  let source: TileSource | undefined;
  const pending = new Map<number, AbortController>();
  const reply = (message: Response, transfer: Transferable[] = []): void => {
    port.postMessage(message, transfer);
  };
  const fail = (id: number, err: unknown): void => {
    const error = err instanceof Error ? err : new Error(String(err));
    reply({ type: "error", id, name: error.name, message: error.message });
  };

  port.addEventListener("message", (event: MessageEvent) => {
    const request = event.data as Request;
    switch (request.type) {
      case "open":
        open(request.location).then(
          (store) => {
            source = new TileSource(store, maxBytes);
            reply({ type: "opened", id: request.id, info: store.info });
          },
          (err: unknown) => {
            fail(request.id, err);
          },
        );
        break;
      case "tile": {
        if (source === undefined) {
          fail(request.id, new Error("no store is open"));
          break;
        }
        const controller = new AbortController();
        pending.set(request.id, controller);
        source.getTile(request.key, request.coords, controller.signal).then(
          (tile) => {
            pending.delete(request.id);
            const codes = tile.codes.slice();
            reply({ type: "tile", id: request.id, tile: { ...tile, codes } }, [codes.buffer]);
          },
          (err: unknown) => {
            pending.delete(request.id);
            fail(request.id, err);
          },
        );
        break;
      }
      case "abort":
        pending.get(request.id)?.abort();
        break;
    }
  });
  port.start?.();
}
