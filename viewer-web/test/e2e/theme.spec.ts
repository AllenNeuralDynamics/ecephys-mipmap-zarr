import { expect, test, type Page } from "@playwright/test";

/**
 * The ink of the viewer on the page: of the darkest and lightest 1% of opaque pixels, whichever is farther from
 * the median, which is the fill. Un-premultiplied, in 0..1.
 */
async function inkColour(page: Page): Promise<number[]> {
  return page.evaluate(async () => {
    const viewer = document.querySelector("mipmap-viewer");
    await viewer?.idle();
    const shot = viewer?.snapshot();
    if (shot === undefined) throw new Error("viewer did not render");
    const opaque: number[][] = [];
    for (let i = 0; i < shot.data.length; i += 4) {
      const alpha = shot.data[i + 3] ?? 0;
      if (alpha >= 250) opaque.push([0, 1, 2].map((k) => (shot.data[i + k] ?? 0) / alpha));
    }
    if (opaque.length < 100) return [];
    const luma = (c: number[]): number => 0.2126 * (c[0] ?? 0) + 0.7152 * (c[1] ?? 0) + 0.0722 * (c[2] ?? 0);
    opaque.sort((a, b) => luma(a) - luma(b));
    const n = Math.ceil(opaque.length / 100);
    const mean = (cs: number[][]): number[] =>
      [0, 1, 2].map((k) => cs.reduce((s, c) => s + (c[k] ?? 0), 0) / cs.length);
    const median = luma(opaque[Math.floor(opaque.length / 2)] ?? []);
    const dark = mean(opaque.slice(0, n));
    const light = mean(opaque.slice(-n));
    return median - luma(dark) > luma(light) - median ? dark : light;
  });
}

/** Open a fixture store under `theme` and return its ink and the host background. */
async function drawnColour(page: Page, theme: string | null): Promise<{ rgb: number[]; background: string }> {
  await page.goto("/test/e2e/page.html");
  await page.waitForFunction(() => (window as unknown as { viewerReady?: boolean }).viewerReady === true);
  const background = await page.evaluate(async (theme) => {
    const viewer = document.querySelector("mipmap-viewer");
    if (viewer === null) throw new Error("no viewer");
    if (theme !== null) viewer.setAttribute("theme", theme);
    viewer.store = "/test/fixtures/int16_raw-none_pyr-none.zarr";
    await viewer.idle();
    return getComputedStyle(viewer).backgroundColor;
  }, theme);
  return { rgb: await inkColour(page), background };
}

const isLight = (rgb: number[]): boolean => rgb.length === 3 && rgb.every((v) => v > 0.75);
const isDark = (rgb: number[]): boolean => rgb.length === 3 && rgb.every((v) => v < 0.35);

test("light by default, with dark ink", async ({ page }) => {
  await page.emulateMedia({ colorScheme: "no-preference" });
  const { rgb, background } = await drawnColour(page, null);
  expect(isDark(rgb)).toBe(true);
  expect(background).toBe("rgb(251, 251, 252)");
});

test("follows a dark browser setting", async ({ page }) => {
  await page.emulateMedia({ colorScheme: "dark" });
  const { rgb, background } = await drawnColour(page, null);
  expect(isLight(rgb)).toBe(true);
  expect(background).toBe("rgb(28, 29, 32)");
});

test("theme overrides the browser setting either way", async ({ page }) => {
  await page.emulateMedia({ colorScheme: "dark" });
  expect(isDark((await drawnColour(page, "light")).rgb)).toBe(true);
  await page.emulateMedia({ colorScheme: "light" });
  expect(isLight((await drawnColour(page, "dark")).rgb)).toBe(true);
});

test("redraws when the browser setting changes", async ({ page }) => {
  await page.emulateMedia({ colorScheme: "light" });
  expect(isDark((await drawnColour(page, null)).rgb)).toBe(true);
  await page.emulateMedia({ colorScheme: "dark" });
  // The media query's change event arrives asynchronously.
  await expect.poll(() => inkColour(page).then(isLight)).toBe(true);
});

test("a page's custom properties recolour the traces", async ({ page }) => {
  await drawnColour(page, "light");
  await page.evaluate(() => {
    const viewer = document.querySelector("mipmap-viewer");
    viewer?.style.setProperty("--mv-ink", "#ff0000");
    viewer?.setAttribute("theme", "auto");
    viewer?.setAttribute("theme", "light");
  });
  await expect.poll(() => inkColour(page).then(([r = 0, g = 1, b = 1]) => r > 0.8 && g < 0.3 && b < 0.3)).toBe(true);
});
