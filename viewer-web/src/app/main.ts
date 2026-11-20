import { defineMipmapViewer, type ChannelOrder, type ViewChangeDetail } from "../element/mipmap-viewer.js";

defineMipmapViewer();

/** Parse `0-3,7` into channel indices. */
export function parseChannels(text: string): number[] {
  return text.split(",").flatMap((part) => {
    const [a, b] = part.split("-").map((s) => Number.parseInt(s, 10));
    if (a === undefined || Number.isNaN(a)) return [];
    if (b === undefined || Number.isNaN(b)) return [a];
    return Array.from({ length: Math.max(0, b - a + 1) }, (_, i) => a + i);
  });
}

const viewer = document.querySelector("mipmap-viewer");
const form = document.querySelector<HTMLFormElement>("#picker");
const input = document.querySelector<HTMLInputElement>("#store");
if (viewer === null || form === null || input === null) throw new Error("page is missing its viewer");

const params = new URLSearchParams(location.search);
const time = params.get("time")?.split(",").map(Number);
if (time?.length === 2 && time.every(Number.isFinite)) viewer.timeRange = [time[0] ?? 0, time[1] ?? 0];
const channels = params.get("channels");
if (channels !== null) viewer.channels = parseChannels(channels);
const order = params.get("order");
if (order !== null) viewer.order = order as ChannelOrder;
const gain = Number(params.get("gain"));
if (gain > 0) viewer.gain = gain;
const store = params.get("store");
if (store !== null) {
  input.value = store;
  viewer.store = store;
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const url = new URL(location.href);
  url.search = new URLSearchParams({ store: input.value }).toString();
  history.pushState(null, "", url);
  viewer.store = input.value;
});

viewer.addEventListener("viewchange", (event) => {
  const { timeRange, gain } = (event as CustomEvent<ViewChangeDetail>).detail;
  const url = new URL(location.href);
  url.searchParams.set("time", timeRange.map((t) => t.toFixed(6)).join(","));
  url.searchParams.set("gain", gain.toPrecision(4));
  history.replaceState(null, "", url);
});
