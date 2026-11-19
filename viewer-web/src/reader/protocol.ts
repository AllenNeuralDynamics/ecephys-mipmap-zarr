import type { StoreInfo } from "./store.js";
import type { Tile } from "./tiles.js";

/** Messages the main thread sends the reader worker. */
export type Request =
  | { type: "open"; id: number; location: string }
  | { type: "tile"; id: number; key: string; coords: number[] }
  | { type: "abort"; id: number };

/** Messages the worker answers with, each naming the request it answers. */
export type Response =
  | { type: "opened"; id: number; info: StoreInfo }
  | { type: "tile"; id: number; tile: Tile }
  | { type: "error"; id: number; name: string; message: string };

/** The part of `MessagePort`, `Worker` and a worker's global scope the protocol uses. */
export interface Port {
  postMessage(message: unknown, transfer?: Transferable[]): void;
  addEventListener(type: "message", listener: (event: MessageEvent) => void): void;
  start?: () => void;
}
