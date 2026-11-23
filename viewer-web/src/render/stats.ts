/**
 * Per-channel display statistics: the rules a store's `channel_stats` follows (`docs/store-format.md`), measured
 * here for stores that lack them.
 */

/** Quantile of a channel's excursion from its centre that reaches the slot's edge at gain 1. */
export const EXCURSION_QUANTILE = 0.95;

/** The `q` quantile of the finite `values`, interpolating between neighbours; NaN when none are finite. */
export function quantile(values: number[], q: number): number {
  const finite = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (finite.length === 0) return NaN;
  const at = q * (finite.length - 1);
  const below = Math.floor(at);
  const lo = finite[below] ?? NaN;
  return lo + (at - below) * ((finite[Math.min(below + 1, finite.length - 1)] ?? lo) - lo);
}

/**
 * Mean and standard deviation after repeatedly dropping values beyond 3 sd, so spikes do not count; unlike a
 * median absolute deviation, the sd does not snap to the step of coarsely quantized values.
 */
export function clippedStats(values: Float32Array): { mean: number; sd: number } {
  let lo = -Infinity;
  let hi = Infinity;
  let mean = NaN;
  let sd = NaN;
  for (let pass = 0; pass < 5; pass++) {
    let n = 0;
    let sum = 0;
    let squares = 0;
    // Indexed: a typed array's iterator is several times slower here, and this runs over a whole raw chunk.
    for (let i = 0; i < values.length; i++) {
      const v = values[i] ?? NaN;
      if (!(v >= lo && v <= hi)) continue;
      n++;
      sum += v;
      squares += v * v;
    }
    if (n < 2) return { mean: NaN, sd: NaN };
    mean = sum / n;
    sd = Math.sqrt(Math.max(0, squares / n - mean * mean));
    lo = mean - 3 * sd;
    hi = mean + 3 * sd;
  }
  return { mean, sd };
}

/**
 * A channel's centre and excursion from its coarsest-level block minima and maxima. The raw `baseline` is the
 * centre when known, since block midpoints lean toward a channel's spikes.
 */
export function centreAndExcursion(
  lows: number[],
  highs: number[],
  baseline: number,
): { centre: number; excursion: number } {
  const centre = Number.isFinite(baseline)
    ? baseline
    : quantile(
        lows.map((v, b) => 0.5 * (v + (highs[b] ?? NaN))),
        0.5,
      );
  const excursion = quantile(
    lows.map((v, b) => Math.max((highs[b] ?? NaN) - centre, centre - v)),
    EXCURSION_QUANTILE,
  );
  return { centre: centre || 0, excursion };
}

/** Value units per slot height at gain 1: twice the median channel excursion, or 1 when none is known. */
export function unitRangeOf(excursions: number[]): number {
  const unitRange = 2 * quantile(excursions, 0.5);
  return Number.isFinite(unitRange) && unitRange > 0 ? unitRange : 1;
}
