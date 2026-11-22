/** Trace colours: the ramp a pixel's gate level maps through, and the colours events tint it with. */
export interface TracePalette {
  background: string;
  /** Between a column's extremes. */
  fill: string;
  /** An outline within the noise. */
  noise: string;
  /** An outline past the gate. */
  ink: string;
  /** Tints for event groups, replacing `ink` at the top of the ramp. */
  groups: string[];
  /** Outline width in CSS pixels within the noise and past the gate. */
  lineWidth: [number, number];
}

/** Levels the ramp passes through `fill` and `noise` at; the trace pass writes them. */
export const FILL_LEVEL = 0.22;
export const NOISE_LEVEL = 0.4;
/** Texels per ramp row. */
export const RAMP_STEPS = 256;

/** `#rgb`, `#rrggbb` or `rgb()`/`rgba()` with comma or space separators, as sRGB in 0..1. */
export function parseColor(text: string): [number, number, number] {
  const s = text.trim();
  const hex = /^#([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(s)?.[1];
  if (hex !== undefined) {
    const full = hex.length === 3 ? hex.replace(/./g, "$&$&") : hex;
    return [0, 2, 4].map((i) => parseInt(full.slice(i, i + 2), 16) / 255) as [number, number, number];
  }
  const parts = /^rgba?\(([^)]*)\)$/i
    .exec(s)?.[1]
    ?.split(/[\s,/]+/)
    .filter(Boolean);
  if (parts !== undefined && parts.length >= 3) {
    return parts.slice(0, 3).map((p) => (p.endsWith("%") ? parseFloat(p) / 100 : parseFloat(p) / 255)) as [
      number,
      number,
      number,
    ];
  }
  throw new Error(`unparsable colour: ${text}`);
}

const toLinear = (c: number): number => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
const toSrgb = (c: number): number => (c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055);
const WHITE = [0.95047, 1, 1.08883] as const;
const f = (t: number): number => (t > 216 / 24389 ? Math.cbrt(t) : (24389 / 27 / 116) * t + 16 / 116);
const fInverse = (t: number): number => (t > 6 / 29 ? t ** 3 : (116 / (24389 / 27)) * (t - 16 / 116));

/** sRGB in 0..1 to CIELAB (D65). */
export function toLab([r, g, b]: readonly number[]): [number, number, number] {
  const [lr, lg, lb] = [r ?? 0, g ?? 0, b ?? 0].map(toLinear) as [number, number, number];
  const x = (0.4124 * lr + 0.3576 * lg + 0.1805 * lb) / WHITE[0];
  const y = (0.2126 * lr + 0.7152 * lg + 0.0722 * lb) / WHITE[1];
  const z = (0.0193 * lr + 0.1192 * lg + 0.9505 * lb) / WHITE[2];
  return [116 * f(y) - 16, 500 * (f(x) - f(y)), 200 * (f(y) - f(z))];
}

/** CIELAB (D65) to sRGB in 0..1, clipped to the gamut. */
export function fromLab([l, a, b]: readonly number[]): [number, number, number] {
  const fy = ((l ?? 0) + 16) / 116;
  const x = fInverse(fy + (a ?? 0) / 500) * WHITE[0];
  const y = fInverse(fy) * WHITE[1];
  const z = fInverse(fy - (b ?? 0) / 200) * WHITE[2];
  const linear = [
    3.2406 * x - 1.5372 * y - 0.4986 * z,
    -0.9689 * x + 1.8758 * y + 0.0415 * z,
    0.0557 * x - 0.204 * y + 1.057 * z,
  ];
  return linear.map((c) => Math.min(1, Math.max(0, toSrgb(Math.min(1, Math.max(0, c)))))) as [number, number, number];
}

const mixLab = (a: readonly number[], b: readonly number[], t: number): number[] =>
  a.map((v, i) => v + t * ((b[i] ?? 0) - v));

/** Share of a colour's chroma a muted variant keeps, and how far its lightness moves toward the noise. */
const MUTED_CHROMA = 0.25;
const MUTED_LIGHTNESS = 0.7;

/** `#rrggbb` for sRGB in 0..1. */
export function toHex(rgb: readonly number[]): string {
  return `#${rgb
    .map((c) =>
      Math.round(c * 255)
        .toString(16)
        .padStart(2, "0"),
    )
    .join("")}`;
}

/** A quieter `colour` for groups drawn beside a selection: less chroma, lightness nearer `noise`'s, same hue. */
export function mutedColour(colour: string, noise: string): string {
  const [l, a, b] = toLab(parseColor(colour));
  const [ln] = toLab(parseColor(noise));
  return toHex(fromLab([l + MUTED_LIGHTNESS * (ln - l), a * MUTED_CHROMA, b * MUTED_CHROMA]));
}

/**
 * Ramp rows, premultiplied RGBA8, `RAMP_STEPS` texels each: row 0 ends in `ink`, row k in `groups[k - 1]`.
 * Below `FILL_LEVEL` a texel is the fill colour at partial alpha, so antialiased fringes blend into whatever is
 * behind the canvas; from there it is opaque, interpolated in Lab through `noise` to the row's top colour.
 */
export function buildRamp(palette: TracePalette): Uint8Array {
  const tops = [palette.ink, ...palette.groups].map((c) => toLab(parseColor(c)));
  const fill = toLab(parseColor(palette.fill));
  const noise = toLab(parseColor(palette.noise));
  const fillRgb = parseColor(palette.fill);
  const out = new Uint8Array(tops.length * RAMP_STEPS * 4);
  tops.forEach((top, row) => {
    for (let i = 0; i < RAMP_STEPS; i++) {
      const v = i / (RAMP_STEPS - 1);
      let rgb: readonly number[];
      let alpha = 1;
      if (v < FILL_LEVEL) {
        rgb = fillRgb;
        alpha = v / FILL_LEVEL;
      } else if (v < NOISE_LEVEL) {
        rgb = fromLab(mixLab(fill, noise, (v - FILL_LEVEL) / (NOISE_LEVEL - FILL_LEVEL)));
      } else {
        rgb = fromLab(mixLab(noise, top, (v - NOISE_LEVEL) / (1 - NOISE_LEVEL)));
      }
      const at = (row * RAMP_STEPS + i) * 4;
      for (let k = 0; k < 3; k++) out[at + k] = Math.round((rgb[k] ?? 0) * alpha * 255);
      out[at + 3] = Math.round(alpha * 255);
    }
  });
  return out;
}
