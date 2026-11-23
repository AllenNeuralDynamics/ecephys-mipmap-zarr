import { fileURLToPath } from "node:url";

import { defineConfig } from "vite";

/**
 * The standalone page, built into the Python package for `ecephys-mipmap view` to serve (hatch_build.py runs it).
 * Fixed file names let the server and tests name the entry; relative URLs let it run under a proxy's path prefix.
 */
export default defineConfig({
  root: "src/app",
  base: "./",
  build: {
    outDir: fileURLToPath(new URL("../src/ecephys_mipmap_zarr/static/viewer", import.meta.url)),
    emptyOutDir: true,
    target: "es2022",
    rolldownOptions: {
      output: {
        entryFileNames: "viewer.js",
        chunkFileNames: "[name].js",
        assetFileNames: "[name][extname]",
      },
    },
  },
});
