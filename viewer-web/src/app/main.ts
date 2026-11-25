import {
  defineMipmapViewer,
  type ChannelOrder,
  type SelectionChangeDetail,
  type ViewChangeDetail,
} from "../element/mipmap-viewer.js";
import { groupsWhere, parseGroupFilter } from "../events/filter.js";

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
const theme = params.get("theme");
if (theme === "light" || theme === "dark") {
  viewer.theme = theme;
  document.documentElement.dataset.theme = theme;
}
const gain = Number(params.get("gain"));
if (gain > 0) viewer.gain = gain;
const rows = Number(params.get("rows"));
if (rows >= 1) viewer.rows = rows;
const events = params.get("events");
if (events !== null) viewer.events = events;
const markers = params.get("markers");
if (markers !== null) viewer.markers = markers;
const selected = params.get("groups");
if (selected !== null) viewer.selectedGroups = selected.split(",").filter(Boolean);
if (params.get("unselected") === "hidden") viewer.unselected = "hidden";
const filter = params.get("filter");
if (filter !== null) {
  // Applied once per layer, when its groups arrive; setting it fires groupschange again for the same layer.
  const wanted = parseGroupFilter(filter);
  let filtered: unknown;
  viewer.addEventListener("groupschange", () => {
    const layer = viewer.eventLayer;
    if (layer === undefined || layer === filtered) return;
    filtered = layer;
    viewer.visibleGroups = groupsWhere(layer.groups, wanted);
  });
}
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
  const { timeRange, gain, rows, order } = (event as CustomEvent<ViewChangeDetail>).detail;
  const url = new URL(location.href);
  url.searchParams.set("time", timeRange.map((t) => t.toFixed(6)).join(","));
  url.searchParams.set("gain", gain.toPrecision(4));
  url.searchParams.set("rows", String(rows));
  url.searchParams.set("order", order);
  history.replaceState(null, "", url);
});

viewer.addEventListener("selectionchange", (event) => {
  const { selectedGroups } = (event as CustomEvent<SelectionChangeDetail>).detail;
  const url = new URL(location.href);
  if (selectedGroups.length > 0) url.searchParams.set("groups", selectedGroups.join(","));
  else url.searchParams.delete("groups");
  history.replaceState(null, "", url);
});
