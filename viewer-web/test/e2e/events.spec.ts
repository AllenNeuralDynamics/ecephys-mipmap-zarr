import { expect, test, type Page } from "@playwright/test";

const FS = 30_000;
const spikeAt = (channel: number): number => 5_000 + 20_011 * channel;

/** Columns, per slot, holding a strongly coloured pixel (channel spread over 0.25 after un-premultiplying). */
async function tinted(page: Page, events: string | null, selected: string[] = []): Promise<number[][]> {
  await page.goto("/test/e2e/page.html");
  await page.waitForFunction(() => (window as unknown as { viewerReady?: boolean }).viewerReady === true);
  return page.evaluate(
    async ({ events, selected, range }) => {
      const viewer = document.querySelector("mipmap-viewer");
      if (viewer === null) throw new Error("no viewer");
      viewer.style.width = "800px";
      viewer.setAttribute("theme", "light");
      viewer.setAttribute("controls", "none");
      viewer.channels = [5, 2];
      viewer.order = "index";
      viewer.events = events;
      viewer.selectedGroups = selected;
      viewer.store = "/test/fixtures/int16_raw-none_pyr-none.zarr";
      viewer.timeRange = range;
      await viewer.idle();
      const shot = viewer.snapshot();
      const layout = viewer.layout();
      if (shot === undefined || layout === undefined) throw new Error("viewer did not render");
      const columns: number[][] = Array.from({ length: layout.view.nSlots }, () => []);
      for (let row = 0; row < shot.height; row++) {
        const slot = Math.floor((shot.height - 1 - row) / layout.slotPx);
        for (let x = 0; x < shot.width; x++) {
          const i = (row * shot.width + x) * 4;
          const alpha = shot.data[i + 3] ?? 0;
          if (alpha < 200) continue;
          const rgb = [0, 1, 2].map((k) => (shot.data[i + k] ?? 0) / alpha);
          if (Math.max(...rgb) - Math.min(...rgb) > 0.25 && !columns[slot]?.includes(x)) columns[slot]?.push(x);
        }
      }
      return columns.map((c) => c.sort((a, b) => a - b));
    },
    { events, selected, range: [(spikeAt(5) - 200) / FS, (spikeAt(5) + 200) / FS] as [number, number] },
  );
}

test("an event's group colours its footprint at the event and nowhere else", async ({ page }) => {
  const [own = [], other = []] = await tinted(page, "/test/fixtures/events.zarr");
  // 400 samples over 800 px: the spike sits at column 400, its -0.5..+1 ms window spans 30 px before to 60 after.
  expect(own.length).toBeGreaterThan(10);
  expect(Math.min(...own)).toBeGreaterThanOrEqual(400 - 32);
  expect(Math.max(...own)).toBeLessThanOrEqual(400 + 62);
  // Channel 3 is outside unit 6's footprint.
  expect(other).toEqual([]);
});

test("without events, or with another group selected, nothing is coloured", async ({ page }) => {
  expect((await tinted(page, null)).flat()).toEqual([]);
  expect((await tinted(page, "/test/fixtures/events.zarr", ["unit-1"])).flat()).toEqual([]);
  expect((await tinted(page, "/test/fixtures/events.zarr", ["unit-6"]))[0]?.length).toBeGreaterThan(10);
});

test("labels name the group at a narrow view", async ({ page }) => {
  await tinted(page, "/test/fixtures/events.zarr");
  const labels = await page.evaluate(() =>
    [...(document.querySelector("mipmap-viewer")?.shadowRoot?.querySelectorAll(".labels span") ?? [])].map(
      (s) => s.textContent,
    ),
  );
  expect(labels).toEqual(["unit-6"]);
});
