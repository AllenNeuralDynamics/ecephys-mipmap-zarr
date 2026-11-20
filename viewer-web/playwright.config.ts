import { defineConfig, devices } from "@playwright/test";

const PORT = 5179;

export default defineConfig({
  testDir: "test/e2e",
  timeout: 120_000,
  use: { baseURL: `http://localhost:${String(PORT)}`, ...devices["Desktop Chrome"], deviceScaleFactor: 1 },
  webServer: {
    command: `npx vite --port ${String(PORT)} --strictPort`,
    url: `http://localhost:${String(PORT)}/test/e2e/page.html`,
    reuseExistingServer: !process.env.CI,
  },
});
