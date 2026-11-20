import { createReadStream, statSync } from "node:fs";
import { join, normalize } from "node:path";

import type { Plugin } from "vite";

/**
 * Serve the directory `root` at `prefix` as plain bytes with `Range` support. Vite would otherwise treat
 * zarr's extensionless chunk keys as JavaScript modules and try to transform them.
 */
export function serveStores(prefix: string, root: string): Plugin {
  return {
    name: `serve-stores:${prefix}`,
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const url = new URL(req.url ?? "/", "http://localhost");
        if (!url.pathname.startsWith(prefix)) {
          next();
          return;
        }
        const path = normalize(join(root, decodeURIComponent(url.pathname.slice(prefix.length))));
        let size: number;
        try {
          const stat = statSync(path);
          if (!path.startsWith(join(root, "/")) || !stat.isFile()) throw new Error("not a file");
          size = stat.size;
        } catch {
          res.statusCode = 404;
          res.end();
          return;
        }
        res.setHeader("Accept-Ranges", "bytes");
        res.setHeader("Content-Type", "application/octet-stream");
        const range = /^bytes=(\d*)-(\d*)$/.exec(req.headers.range ?? "");
        if (range === null) {
          res.setHeader("Content-Length", size);
          createReadStream(path).pipe(res);
          return;
        }
        const [, a = "", b = ""] = range;
        const start = a === "" ? Math.max(0, size - Number(b)) : Number(a);
        const end = a === "" || b === "" ? size - 1 : Math.min(size - 1, Number(b));
        res.statusCode = 206;
        res.setHeader("Content-Range", `bytes ${String(start)}-${String(end)}/${String(size)}`);
        res.setHeader("Content-Length", end - start + 1);
        createReadStream(path, { start, end }).pipe(res);
      });
    },
  };
}
