import { expect, test, type Page } from "@playwright/test";

const FS = 30_000;
const spikeAt = (channel: number): number => 5_000 + 20_011 * channel;
const EVENTS = "/test/fixtures/events.zarr";

interface Setup {
  events: string | null;
  selected?: string[];
  unselected?: "muted" | "hidden";
  colours?: Record<string, string>;
  markers?: string;
  raster?: "none";
}

/** Open the fixture around unit 6's spike, showing channels 5 (its own) and 2 (outside its footprint). */
async function open(page: Page, setup: Setup): Promise<void> {
  await page.goto("/test/e2e/page.html");
  await page.waitForFunction(() => (window as unknown as { viewerReady?: boolean }).viewerReady === true);
  await page.evaluate(
    async ({ setup, range }) => {
      const viewer = document.querySelector("mipmap-viewer");
      if (viewer === null) throw new Error("no viewer");
      viewer.style.width = "800px";
      viewer.setAttribute("theme", "light");
      viewer.setAttribute("controls", "none");
      viewer.unselected = setup.unselected ?? "muted";
      if (setup.raster === undefined) viewer.removeAttribute("raster");
      else viewer.setAttribute("raster", setup.raster);
      viewer.markers = setup.markers ?? null;
      viewer.channels = [5, 2];
      viewer.order = "index";
      viewer.events = setup.events;
      viewer.selectedGroups = setup.selected ?? [];
      viewer.groupColours = setup.colours ?? {};
      viewer.store = "/test/fixtures/int16_raw-none_pyr-none.zarr";
      viewer.timeRange = range;
      await viewer.idle();
    },
    { setup, range: [(spikeAt(5) - 200) / FS, (spikeAt(5) + 200) / FS] as [number, number] },
  );
}

interface Tinted {
  /** Per slot, columns holding a strongly coloured pixel (channel spread over 0.25 after un-premultiplying). */
  columns: number[][];
  /** Per slot, the largest channel spread of any opaque pixel, and that pixel's colour. */
  peak: { spread: number; rgb: number[] }[];
}

async function tinted(page: Page): Promise<Tinted> {
  return page.evaluate(async () => {
    const viewer = document.querySelector("mipmap-viewer");
    if (viewer === null) throw new Error("no viewer");
    await viewer.idle();
    const shot = viewer.snapshot();
    const layout = viewer.layout();
    if (shot === undefined || layout === undefined) throw new Error("viewer did not render");
    const columns: number[][] = Array.from({ length: layout.view.nSlots }, () => []);
    const peak = Array.from({ length: layout.view.nSlots }, () => ({ spread: 0, rgb: [0, 0, 0] }));
    for (let row = 0; row < shot.height; row++) {
      const slot = Math.floor((shot.height - 1 - row) / layout.slotPx);
      for (let x = 0; x < shot.width; x++) {
        const i = (row * shot.width + x) * 4;
        const alpha = shot.data[i + 3] ?? 0;
        if (alpha < 200) continue;
        const rgb = [0, 1, 2].map((k) => (shot.data[i + k] ?? 0) / alpha);
        const spread = Math.max(...rgb) - Math.min(...rgb);
        const best = peak[slot];
        if (best !== undefined && spread > best.spread) peak[slot] = { spread, rgb };
        if (spread > 0.25 && !columns[slot]?.includes(x)) columns[slot]?.push(x);
      }
    }
    return { columns: columns.map((c) => c.sort((a, b) => a - b)), peak };
  });
}

test("an event's group colours its footprint at the event and nowhere else", async ({ page }) => {
  await open(page, { events: EVENTS });
  const { columns } = await tinted(page);
  const [own = [], other = []] = columns;
  // 400 samples over 800 px: the spike sits at column 400, its -0.5..+1 ms window spans 30 px before to 60 after.
  expect(own.length).toBeGreaterThan(10);
  expect(Math.min(...own)).toBeGreaterThanOrEqual(400 - 32);
  expect(Math.max(...own)).toBeLessThanOrEqual(400 + 62);
  // Channel 3 is outside unit 6's footprint.
  expect(other).toEqual([]);
});

test("without events nothing is coloured; beside a selection a group is muted, or hidden", async ({ page }) => {
  await open(page, { events: null });
  expect((await tinted(page)).columns.flat()).toEqual([]);
  // Red makes unit 6's tint measurable against the bluish noise: how far red leads the other channels.
  const redness = async (setup: Setup): Promise<number> => {
    await open(page, { ...setup, events: EVENTS, colours: { "unit-6": "red" } });
    const [r = 0, g = 0, b = 0] = (await tinted(page)).peak[0]?.rgb ?? [];
    return r - Math.max(g, b);
  };
  const selected = await redness({ events: EVENTS, selected: ["unit-6"] });
  const muted = await redness({ events: EVENTS, selected: ["unit-1"] });
  const hidden = await redness({ events: EVENTS, selected: ["unit-1"], unselected: "hidden" });
  expect(selected).toBeGreaterThan(0.5);
  expect(muted).toBeGreaterThan(0.03);
  expect(muted).toBeLessThan(selected / 2);
  expect(hidden).toBeLessThanOrEqual(0);
});

test("a set colour replaces the group's own", async ({ page }) => {
  await open(page, { events: EVENTS, colours: { "unit-6": "red" } });
  const { rgb = [] } = (await tinted(page)).peak[0] ?? {};
  const [r = 0, g = 0, b = 0] = rgb;
  expect(r - Math.max(g, b)).toBeGreaterThan(0.4);
  const styles = await page.evaluate(() => document.querySelector("mipmap-viewer")?.groupStyles);
  expect(styles?.find((s) => s.id === "unit-6")?.colour).toBe("#ff0000");
  expect(styles?.length).toBe(9);
});

test("a click on a group's tint toggles it, unless a listener cancels; Escape clears", async ({ page }) => {
  await open(page, { events: EVENTS });
  const log = await page.evaluate(() => {
    const viewer = document.querySelector("mipmap-viewer");
    const canvas = viewer?.shadowRoot?.querySelector("canvas");
    if (viewer === null || canvas === null || canvas === undefined) throw new Error("no viewer");
    const out: string[] = [];
    viewer.addEventListener("groupclick", (e) =>
      out.push(`click ${(e as CustomEvent<{ groupId: string }>).detail.groupId}`),
    );
    viewer.addEventListener("selectionchange", (e) =>
      out.push(`selection ${(e as CustomEvent<{ selectedGroups: string[] }>).detail.selectedGroups.join(",")}`),
    );
    viewer.addEventListener("channelclick", () => out.push("channel"));
    const rect = canvas.getBoundingClientRect();
    const click = (fx: number, fy: number): void => {
      const init = {
        clientX: rect.left + fx * rect.width,
        clientY: rect.top + fy * rect.height,
        button: 0,
        pointerId: 1,
      };
      canvas.dispatchEvent(new PointerEvent("pointerdown", { ...init, bubbles: true }));
      canvas.dispatchEvent(new PointerEvent("pointerup", { ...init, bubbles: true }));
    };
    // The spike's column on its own channel (top slot), then empty trace on channel 2.
    click(0.52, 0.25);
    click(0.52, 0.75);
    const cancel = (e: Event): void => {
      e.preventDefault();
    };
    viewer.addEventListener("groupclick", cancel, { once: true });
    click(0.52, 0.25);
    out.push(`now ${viewer.selectedGroups.join(",")}`);
    viewer.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
    out.push(`after escape ${viewer.selectedGroups.join(",")}`);
    viewer.selectedGroups = ["unit-2"];
    return out;
  });
  expect(log).toEqual([
    "click unit-6",
    "selection unit-6",
    "channel",
    "click unit-6",
    "now unit-6",
    "selection ",
    "after escape ",
  ]);
});

test("labels name the group at a narrow view, and a click on one toggles it", async ({ page }) => {
  await open(page, { events: EVENTS });
  const labels = await page.evaluate(() =>
    [...(document.querySelector("mipmap-viewer")?.shadowRoot?.querySelectorAll(".labels span") ?? [])].map(
      (s) => s.textContent,
    ),
  );
  expect(labels).toEqual(["unit-6"]);
  const selected = await page.evaluate(async () => {
    const viewer = document.querySelector("mipmap-viewer");
    viewer?.shadowRoot?.querySelector<HTMLElement>(".labels span")?.click();
    await viewer?.idle();
    return [viewer?.selectedGroups, viewer?.shadowRoot?.querySelector(".labels span")?.className];
  });
  expect(selected).toEqual([["unit-6"], "selected"]);
});

/** Columns and rows of the raster's coloured pixels, device px, with the panel's size. */
async function rasterTicks(page: Page): Promise<{ width: number; height: number; ticks: [number, number][] }> {
  return page.evaluate(async () => {
    const viewer = document.querySelector("mipmap-viewer");
    await viewer?.idle();
    const canvas = viewer?.shadowRoot?.querySelector<HTMLCanvasElement>("canvas.raster");
    const ctx = canvas?.getContext("2d");
    if (canvas === null || canvas === undefined || ctx === null || ctx === undefined) throw new Error("no raster");
    const { data, width, height } = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const ticks: [number, number][] = [];
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const i = (y * width + x) * 4;
        const rgb = [data[i] ?? 0, data[i + 1] ?? 0, data[i + 2] ?? 0];
        if (Math.max(...rgb) - Math.min(...rgb) > 40) ticks.push([x, y]);
      }
    }
    return { width, height, ticks };
  });
}

test("the raster ticks each event at its column on its footprint's slot, and a click on one toggles it", async ({
  page,
}) => {
  // Red stands out from the background however pale a tick is.
  await open(page, { events: EVENTS, colours: { "unit-6": "red" } });
  const { height, ticks } = await rasterTicks(page);
  expect(ticks.length).toBeGreaterThan(0);
  // Unit 6 sits on channel 6, the top of the two slots; its spike at column 400.
  for (const [x, y] of ticks) {
    expect(Math.abs(x - 400)).toBeLessThanOrEqual(1);
    expect(Math.abs(y - height / 4)).toBeLessThanOrEqual(3);
  }
  const [x = 0, y = 0] = ticks[0] ?? [];
  const selected = await page.evaluate(
    ({ x, y }) => {
      const viewer = document.querySelector("mipmap-viewer");
      const canvas = viewer?.shadowRoot?.querySelector<HTMLCanvasElement>("canvas.raster");
      if (viewer === null || canvas === null || canvas === undefined) throw new Error("no raster");
      const rect = canvas.getBoundingClientRect();
      const init = {
        clientX: rect.left + ((x + 0.5) / canvas.width) * rect.width,
        clientY: rect.top + ((y + 0.5) / canvas.height) * rect.height,
        button: 0,
        pointerId: 1,
        bubbles: true,
      };
      canvas.dispatchEvent(new PointerEvent("pointerdown", init));
      canvas.dispatchEvent(new PointerEvent("pointerup", init));
      return viewer.selectedGroups;
    },
    { x, y },
  );
  expect(selected).toEqual(["unit-6"]);
});

test('raster="none" hides the raster and gives the traces its height', async ({ page }) => {
  const heights = async (): Promise<number[]> =>
    page.evaluate(() =>
      ["canvas.traces", "canvas.raster"].map(
        (q) => document.querySelector("mipmap-viewer")?.shadowRoot?.querySelector(q)?.clientHeight ?? -1,
      ),
    );
  await open(page, { events: EVENTS });
  const [withRaster = 0, raster = 0] = await heights();
  expect(raster).toBeGreaterThan(50);
  await open(page, { events: EVENTS, raster: "none" });
  const [traces = 0, hidden = -1] = await heights();
  expect(hidden).toBe(0);
  // Each height rounds on its own.
  expect(Math.abs(traces - (withRaster + raster))).toBeLessThanOrEqual(1);
});

test("markers tick the axis; ] and [ centre the view on the next and previous, and a click fires markerclick", async ({
  page,
}) => {
  await open(page, { events: null, markers: EVENTS });
  const log = await page.evaluate(async () => {
    const viewer = document.querySelector("mipmap-viewer");
    const axis = viewer?.shadowRoot?.querySelector<HTMLCanvasElement>("canvas.axis");
    if (viewer === null || axis === null || axis === undefined) throw new Error("no viewer");
    const out: string[] = [];
    viewer.addEventListener("viewchange", () => out.push("viewchange"));
    viewer.addEventListener("markerclick", (e) => {
      const { index, groupId } = (e as CustomEvent<{ index: number; groupId: string }>).detail;
      out.push(`markerclick ${String(index)} ${groupId}`);
    });
    const centre = (): string => {
      const [a = 0, b = 0] = viewer.timeRange ?? [];
      return ((a + b) / 2).toFixed(4);
    };
    out.push(`at ${centre()}`);
    viewer.dispatchEvent(new KeyboardEvent("keydown", { key: "]", bubbles: true }));
    out.push(`next ${centre()}`);
    viewer.dispatchEvent(new KeyboardEvent("keydown", { key: "[", bubbles: true }));
    out.push(`previous ${centre()}`);
    await viewer.idle();
    // The view is centred on a marker, so its tick is at the middle column.
    const rect = axis.getBoundingClientRect();
    const init = { clientX: rect.left + rect.width / 2, clientY: rect.top + 3, button: 0, pointerId: 1, bubbles: true };
    axis.dispatchEvent(new PointerEvent("pointerdown", init));
    axis.dispatchEvent(new PointerEvent("pointerup", init));
    return out;
  });
  // Unit 6's spike at sample 105055 (3.5018 s) is centred; the evenly spaced extras fall at 1000 + 9000 k.
  expect(log).toEqual([
    "at 3.5018",
    "viewchange",
    `next ${(109_000 / FS).toFixed(4)}`,
    "viewchange",
    `previous ${(105_055 / FS).toFixed(4)}`,
    "markerclick 17 unit-6",
    "viewchange",
  ]);
});
