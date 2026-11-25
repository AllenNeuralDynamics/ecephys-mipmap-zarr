import { expect, test, type Page } from "@playwright/test";

import type { ViewChangeDetail } from "../../src/element/mipmap-viewer.js";

const STORE = "/test/fixtures/int16_raw-none_pyr-none.zarr";
/** The fixture's channels sit 20 µm apart with channel 1 deepest, so depth order reverses store order. */
const N_CHANNELS = 8;
const STORE_ORDER = Array.from({ length: N_CHANNELS }, (_, c) => c);
const DEPTH_ORDER = [...STORE_ORDER].reverse();

/** Open the fixture with `rows` channel rows, recording each `viewchange` in `window.changes`. */
async function open(page: Page, rows: number | null = null, controls: string | null = null): Promise<void> {
  await page.goto("/test/e2e/page.html");
  await page.waitForFunction(() => (window as unknown as { viewerReady?: boolean }).viewerReady === true);
  await page.evaluate(
    async ({ store, rows, controls }) => {
      const viewer = document.querySelector("mipmap-viewer");
      if (viewer === null) throw new Error("no viewer");
      const changes: ViewChangeDetail[] = [];
      (window as unknown as { changes: ViewChangeDetail[] }).changes = changes;
      viewer.addEventListener("viewchange", (event) => {
        changes.push((event as CustomEvent<ViewChangeDetail>).detail);
      });
      if (controls !== null) viewer.setAttribute("controls", controls);
      viewer.rows = rows;
      viewer.store = store;
      await viewer.idle();
    },
    { store: STORE, rows, controls },
  );
}

const state = (page: Page): Promise<{ order: string; displayed: number[]; firstSlot: number; width: number }> =>
  page.evaluate(() => {
    const viewer = document.querySelector("mipmap-viewer");
    const layout = viewer?.layout();
    const shot = viewer?.snapshot();
    if (viewer === null || layout === undefined || shot === undefined) throw new Error("viewer did not render");
    return {
      order: viewer.order,
      displayed: [...layout.displayed],
      firstSlot: layout.view.firstSlot,
      width: shot.width,
    };
  });

/** Wait for a `viewchange` and return the latest. */
async function lastChange(page: Page): Promise<ViewChangeDetail> {
  await page.waitForFunction(() => (window as unknown as { changes: unknown[] }).changes.length > 0);
  return page.evaluate(() => {
    const { changes } = window as unknown as { changes: ViewChangeDetail[] };
    const last = changes[changes.length - 1];
    if (last === undefined) throw new Error("no viewchange");
    return last;
  });
}

test("a store with channel locations opens in depth order", async ({ page }) => {
  await open(page);
  expect(await state(page)).toMatchObject({ order: "depth", displayed: DEPTH_ORDER });
  await expect(page.locator("mipmap-viewer select.order")).toHaveValue("depth");
});

test("each row is labelled with its own channel id, in either order", async ({ page }) => {
  await open(page);
  const ids = await page.evaluate(() => document.querySelector("mipmap-viewer")?.info?.channelIds ?? []);
  const labels = page.locator("mipmap-viewer .ids span");
  await expect(labels).toHaveText(DEPTH_ORDER.map((c) => ids[c] ?? ""));
  await page.locator("mipmap-viewer select.order").selectOption("index");
  await expect(labels).toHaveText(STORE_ORDER.map((c) => ids[c] ?? ""));
});

test("the order control switches to store order and reports it", async ({ page }) => {
  await open(page);
  await page.locator("mipmap-viewer select.order").selectOption("index");
  expect(await state(page)).toMatchObject({ order: "index", displayed: STORE_ORDER });
  expect((await lastChange(page)).order).toBe("index");
});

test("dragging the scroll thumb scrolls channels and the label follows", async ({ page }) => {
  await open(page, 4);
  const track = await page.locator("mipmap-viewer .track").boundingBox();
  const thumb = await page.locator("mipmap-viewer .thumb").boundingBox();
  if (track === null || thumb === null) throw new Error("no scroll bar");
  // Half the thumb's travel scrolls half the four hidden rows.
  const travel = track.height - thumb.height;
  await page.mouse.move(thumb.x + thumb.width / 2, thumb.y + thumb.height / 2);
  await page.mouse.down();
  await page.mouse.move(thumb.x + thumb.width / 2, thumb.y + thumb.height / 2 + travel / 2, { steps: 4 });
  await page.mouse.up();
  const { firstSlot } = await state(page);
  expect(firstSlot).toBeCloseTo(2, 1);
  expect((await lastChange(page)).visibleChannels).toEqual(DEPTH_ORDER.slice(2, 6));
  await expect(page.locator("mipmap-viewer .range")).toContainText("rows 3–6 of 8");
});

test("a press on the track below the thumb pages down", async ({ page }) => {
  await open(page, 3);
  const track = await page.locator("mipmap-viewer .track").boundingBox();
  if (track === null) throw new Error("no scroll bar");
  await page.mouse.click(track.x + track.width / 2, track.y + track.height - 2);
  expect((await state(page)).firstSlot).toBe(3);
  expect((await lastChange(page)).visibleChannels).toEqual(DEPTH_ORDER.slice(3, 6));
});

test('controls="none" hides the controls and the traces take the full width', async ({ page }) => {
  await open(page);
  const withControls = (await state(page)).width;
  await open(page, null, "none");
  await expect(page.locator("mipmap-viewer .toolbar")).toBeHidden();
  await expect(page.locator("mipmap-viewer .track")).toBeHidden();
  expect((await state(page)).width).toBe(800);
  expect(withControls).toBeLessThan(800);
});

test("the gain buttons step the gain, the readout resets it, and each change is reported", async ({ page }) => {
  await open(page);
  const gain = (): Promise<number> => page.evaluate(() => document.querySelector("mipmap-viewer")?.gain ?? 0);
  await page.locator("mipmap-viewer button.gain-up").click();
  await page.locator("mipmap-viewer button.gain-up").click();
  expect(await gain()).toBeCloseTo(1.5625);
  await expect(page.locator("mipmap-viewer .gain output")).toHaveText("×1.56");
  expect((await lastChange(page)).gain).toBeCloseTo(1.5625);
  await page.locator("mipmap-viewer button.gain-down").click();
  expect(await gain()).toBeCloseTo(1.25);
  await page.locator("mipmap-viewer .gain output").click();
  expect(await gain()).toBe(1);
  expect((await lastChange(page)).gain).toBe(1);
});

test("the rows menu sets how many rows are in view", async ({ page }) => {
  await open(page, 3);
  const rows = page.locator("mipmap-viewer select.rows");
  await expect(rows).toHaveValue("3");
  await expect(rows.locator("option")).toHaveText(["3", "All 8"]);
  await rows.selectOption("8");
  expect((await state(page)).firstSlot).toBe(0);
  expect(await page.evaluate(() => document.querySelector("mipmap-viewer")?.rows)).toBe(8);
  expect((await lastChange(page)).rows).toBe(8);
});
