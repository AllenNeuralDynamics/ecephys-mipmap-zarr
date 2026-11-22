import { expect, test, type Page } from "@playwright/test";

const FS = 30_000;
const spikeAt = (channel: number): number => 5_000 + 20_011 * channel;
const EVENTS = "/test/fixtures/events.zarr";

interface Setup {
  events: string | null;
  selected?: string[];
  unselected?: "muted" | "hidden";
  colours?: Record<string, string>;
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
