/** The newest store format version this reader decodes (`docs/store-format.md`). */
export const FORMAT_VERSION = 1;

/** Root attributes of a mipmap store, as the format defines them. */
export interface RootAttrs {
  formatVersion: number;
  samplingFrequency: number;
  nSamples: number;
  nChannels: number;
  strides: number[];
  units: string;
  dtype: string;
  /** Seconds of sample 0 on the acquisition clock. */
  tStart: number;
  rawSource: string | null;
  rawPath: string | null;
  sourceMetadata: unknown;
  /** Stored per-channel display statistics, in decoded units; null when the store predates them. */
  channelStats: ChannelStats | null;
}

/** A store's `channel_stats`, one entry per channel; an unknown excursion is NaN. */
export interface ChannelStats {
  centre: Float32Array;
  noiseSd: Float32Array;
  excursion: Float32Array;
}

export class FormatError extends Error {
  override name = "FormatError";
}

function field(attrs: Record<string, unknown>, key: string): unknown {
  if (!(key in attrs)) throw new FormatError(`store attrs lack ${key}`);
  return attrs[key];
}

function number(attrs: Record<string, unknown>, key: string): number {
  const value = field(attrs, key);
  if (typeof value !== "number" || !Number.isFinite(value)) throw new FormatError(`${key} is not a number`);
  return value;
}

function count(attrs: Record<string, unknown>, key: string): number {
  const value = number(attrs, key);
  if (!Number.isInteger(value) || value < 0) throw new FormatError(`${key} is not a count`);
  return value;
}

function optionalString(attrs: Record<string, unknown>, key: string): string | null {
  const value = attrs[key] ?? null;
  if (value !== null && typeof value !== "string") throw new FormatError(`${key} is not a string`);
  return value;
}

function channelList(stats: Record<string, unknown>, key: string, n: number, nullable: boolean): Float32Array {
  const values = stats[key];
  if (!Array.isArray(values) || values.length !== n)
    throw new FormatError(`channel_stats.${key} is not one per channel`);
  return Float32Array.from(values, (v: unknown) => {
    if (v === null && nullable) return NaN;
    if (typeof v !== "number" || !Number.isFinite(v)) throw new FormatError(`channel_stats.${key} holds a non-number`);
    return v;
  });
}

function parseChannelStats(value: unknown, n: number): ChannelStats | null {
  if (value === undefined || value === null) return null;
  if (typeof value !== "object" || Array.isArray(value)) throw new FormatError("channel_stats is not an object");
  const stats = value as Record<string, unknown>;
  return {
    centre: channelList(stats, "centre", n, false),
    noiseSd: channelList(stats, "noise_sd", n, false),
    excursion: channelList(stats, "excursion", n, true),
  };
}

/** Parse and validate a store's root attributes; throws `FormatError` on a store this reader cannot decode. */
export function parseRootAttrs(attrs: Record<string, unknown>): RootAttrs {
  const formatVersion = attrs.format_version === undefined ? 1 : count(attrs, "format_version");
  if (formatVersion > FORMAT_VERSION) {
    throw new FormatError(
      `store format version ${String(formatVersion)} is newer than this reader's ${String(FORMAT_VERSION)}`,
    );
  }
  const strides = field(attrs, "strides");
  if (!Array.isArray(strides) || !strides.every((s) => Number.isInteger(s) && (s as number) > 0)) {
    throw new FormatError("strides is not a list of positive integers");
  }
  const tStart = attrs.t_start ?? null;
  if (tStart !== null && typeof tStart !== "number") throw new FormatError("t_start is not a number");
  const nChannels = count(attrs, "n_channels");
  return {
    formatVersion,
    samplingFrequency: number(attrs, "sampling_frequency"),
    nSamples: count(attrs, "n_samples"),
    nChannels,
    strides: [...(strides as number[])].sort((a, b) => a - b),
    units: optionalString(attrs, "units") ?? "",
    dtype: optionalString(attrs, "dtype") ?? "",
    tStart: tStart ?? 0,
    rawSource: optionalString(attrs, "raw_source"),
    rawPath: optionalString(attrs, "raw_path"),
    sourceMetadata: attrs.source_metadata ?? null,
    channelStats: parseChannelStats(attrs.channel_stats, nChannels),
  };
}
