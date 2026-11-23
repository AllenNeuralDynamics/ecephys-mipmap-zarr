import type { EventGroups } from "./layer.js";

/** Accepted values per group property: a group passes when every named property holds one of its values. */
export type GroupFilter = Record<string, readonly string[]>;

/** Parse `KSLabel:good,decoder_label:sua|mua`: properties joined by `,` must all match, values joined by `|` any. */
export function parseGroupFilter(text: string): GroupFilter {
  const filter: Record<string, string[]> = {};
  for (const term of text.split(",")) {
    const at = term.indexOf(":");
    if (at <= 0) continue;
    const values = term
      .slice(at + 1)
      .split("|")
      .filter((v) => v !== "");
    if (values.length > 0) filter[term.slice(0, at)] = values;
  }
  return filter;
}

/** A property value as filter text; booleans are stored as 1 and 0, so `true` and `false` match them too. */
function matches(value: string | number | undefined, wanted: ReadonlySet<string>): boolean {
  if (value === undefined) return false;
  if (typeof value === "string") return wanted.has(value);
  return wanted.has(String(value)) || (value === 1 && wanted.has("true")) || (value === 0 && wanted.has("false"));
}

/**
 * Ids of the groups passing `filter`, in group order. A property the layer lacks passes no group, so a typo shows
 * nothing rather than everything.
 */
export function groupsWhere(groups: EventGroups, filter: GroupFilter): string[] {
  const terms = Object.entries(filter).map(([name, values]) => ({
    values: groups.properties[name]?.values,
    wanted: new Set(values.map((v) => (/^(true|false)$/i.test(v) ? v.toLowerCase() : v))),
  }));
  return groups.ids.filter((_, g) => terms.every(({ values, wanted }) => matches(values?.[g], wanted)));
}
