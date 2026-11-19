import { defineConfig } from "vitest/config";

export default defineConfig({
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
