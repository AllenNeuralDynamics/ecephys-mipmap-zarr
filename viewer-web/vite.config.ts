import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { defineConfig } from "vitest/config";

import { serveStores } from "./scripts/serve-stores.js";

export default defineConfig({
  // 404 for missing keys, which zarrita reads as absent; an SPA fallback would answer with index.html.
  appType: "mpa",
  plugins: [
    serveStores("/test/fixtures/", fileURLToPath(new URL("test/fixtures", import.meta.url))),
    // `MIPMAP_STORE_DIR=/path/to/stores npm run dev` serves local stores at /stores/ for a manual session.
    ...(process.env.MIPMAP_STORE_DIR ? [serveStores("/stores/", resolve(process.env.MIPMAP_STORE_DIR))] : []),
  ],
  build: {
    lib: {
      entry: "src/index.ts",
      formats: ["es"],
      fileName: () => "mipmap-viewer.js",
    },
    target: "es2022",
  },
  test: {
    include: ["test/**/*.test.ts"],
    testTimeout: 60_000,
  },
});
