import type { SamplerKind } from "./textures.js";

/**
 * `extent-*` reduce a source into per-column extremes; `trace` draws the fill and outlines from them as a gate
 * level per pixel; `resolve` colours each level through the palette's ramp.
 */
export type ProgramKind = "extent-envelope" | "extent-raw" | "trace" | "resolve";

const SAMPLERS: Record<SamplerKind, string> = { int: "isampler2D", uint: "usampler2D", float: "sampler2D" };

/**
 * Extent passes draw into a texture one texel row per visible slot and one texel per canvas column, each texel
 * the column's (top, -bottom) in device pixels from the slot's centre, combined by MIN blending. x is the tile
 * origin, a float64 difference taken on the CPU, plus an integer offset within the tile, so float32 never holds
 * a session-scale sample index.
 */
const EXTENT_COMMON = `
uniform highp SAMPLER u_codes;
uniform highp sampler2D u_decode;  // per channel: scale, offset or min offset, max offset
uniform highp sampler2D u_slots;   // per channel: slot (negative when hidden), centre value
uniform int u_width;
uniform int u_rowsPerSeries;
uniform int u_c0;
uniform float u_originPx;
uniform float u_pxPerIndex;
uniform vec2 u_canvasPx;
uniform float u_pxPerUnit;
uniform int u_row0;
uniform int u_rows;
// Extremes are clamped just past the canvas, where nothing draws, to keep half floats precise.
uniform float u_limit;

float code(int series, int i) {
  return float(texelFetch(u_codes, ivec2(i % u_width, series * u_rowsPerSeries + i / u_width), 0).r);
}

// Device-pixel offset below the slot's centre of a value relative to the channel's centre.
float below(float value) {
  return clamp(-value * u_pxPerUnit, -u_limit, u_limit);
}

// The texel row of a channel, or -1 when it is not drawn.
int rowOf(int channel) {
  float slot = texelFetch(u_slots, ivec2(channel, 0), 0).r;
  int row = int(slot) - u_row0;
  return slot < 0.0 || row < 0 || row >= u_rows ? -1 : row;
}

vec4 clip(float xPx, float row) {
  return vec4(xPx / u_canvasPx.x * 2.0 - 1.0, row / float(u_rows) * 2.0 - 1.0, 0.0, 1.0);
}

const vec4 HIDDEN = vec4(2.0, 2.0, 2.0, 1.0);
const int CORNER_X[6] = int[6](0, 1, 0, 0, 1, 1);
const int CORNER_Y[6] = int[6](0, 0, 1, 1, 0, 1);
`;

/** One quad per block, its sides snapped outward to whole pixels, so every column it touches gets its extremes. */
const EXTENT_ENVELOPE = `
flat out vec2 v_extent;
void main() {
  int c = gl_InstanceID;
  int block = gl_VertexID / 6;
  int corner = gl_VertexID - block * 6;
  int channel = u_c0 + c;
  int row = rowOf(channel);
  if (row < 0) { gl_Position = HIDDEN; return; }
  vec4 decode = texelFetch(u_decode, ivec2(channel, 0), 0);
  float centre = texelFetch(u_slots, ivec2(channel, 0), 0).g;
  float lo = code(2 * c, block) * decode.x + decode.y - centre;
  float hi = code(2 * c + 1, block) * decode.x + decode.z - centre;
  v_extent = vec2(below(hi), -below(lo));
  float x0 = floor(u_originPx + float(block) * u_pxPerIndex);
  float x1 = max(x0 + 1.0, ceil(u_originPx + float(block + 1) * u_pxPerIndex));
  gl_Position = clip(CORNER_X[corner] == 1 ? x1 : x0, float(row + CORNER_Y[corner]));
}
`;

const EXTENT_ENVELOPE_FRAGMENT = `#version 300 es
precision highp float;
flat in vec2 v_extent;
out vec4 extent;
void main() { extent = vec4(v_extent, 0.0, 0.0); }
`;

/**
 * One quad per segment between consecutive samples, over the columns it crosses; each column takes the
 * segment's extremes within it, so lines join at any zoom.
 */
const EXTENT_RAW = `
flat out vec4 v_segment;  // x and offset below centre of both ends
void main() {
  int c = gl_InstanceID;
  int i = gl_VertexID / 6;
  int corner = gl_VertexID - i * 6;
  int channel = u_c0 + c;
  int row = rowOf(channel);
  if (row < 0) { gl_Position = HIDDEN; return; }
  vec4 decode = texelFetch(u_decode, ivec2(channel, 0), 0);
  float centre = texelFetch(u_slots, ivec2(channel, 0), 0).g;
  float xa = u_originPx + (float(i) + 0.5) * u_pxPerIndex;
  float xb = xa + u_pxPerIndex;
  v_segment = vec4(
    xa, below(code(c, i) * decode.x + decode.y - centre),
    xb, below(code(c, i + 1) * decode.x + decode.y - centre));
  float x0 = floor(xa);
  float x1 = max(x0 + 1.0, ceil(xb));
  gl_Position = clip(CORNER_X[corner] == 1 ? x1 : x0, float(row + CORNER_Y[corner]));
}
`;

const EXTENT_RAW_FRAGMENT = `#version 300 es
precision highp float;
flat in vec4 v_segment;
out vec4 extent;
void main() {
  float column = floor(gl_FragCoord.x);
  float span = v_segment.z - v_segment.x;
  vec2 t = clamp((vec2(column, column + 1.0) - v_segment.x) / span, 0.0, 1.0);
  vec2 y = mix(vec2(v_segment.y), vec2(v_segment.w), t);
  extent = vec4(min(y.x, y.y), -max(y.x, y.y), 0.0, 0.0);
}
`;

/**
 * Per visible slot (instance) and column: a fill from the column's top to bottom, then antialiased outlines
 * joining its top and bottom to the next column's. An outline within the noise is dim and thin; past the gate it
 * reaches full level and width. Drawn as a level with MAX blending, so overlaps keep the strongest.
 */
const TRACE = `#version 300 es
precision highp float;
precision highp int;
uniform highp sampler2D u_extent;
uniform highp sampler2D u_sigma;  // per slot: noise sd (0 when unknown) and the drawn array's step, value units
uniform vec2 u_canvasPx;
uniform float u_firstSlot;
uniform float u_slotPx;
uniform float u_pxPerUnit;
uniform int u_row0;
uniform int u_columns;
uniform vec2 u_halfWidth;  // device pixels within the noise and past the gate
uniform vec2 u_gate;  // sds from the centre where the outline starts to brighten, before quantization; sds to full
uniform float u_dim;  // outline strength inside the noise
uniform float u_fill;
flat out vec4 v_segment;  // ends, device pixels; x < -1e8 marks the fill
flat out float v_strength;
flat out float v_halfWidth;
flat out int v_row;

const vec4 HIDDEN = vec4(2.0, 2.0, 2.0, 1.0);
const float EMPTY = 1e4;

vec4 clip(vec2 p) {
  return vec4(p.x / u_canvasPx.x * 2.0 - 1.0, 1.0 - p.y / u_canvasPx.y * 2.0, 0.0, 1.0);
}

// How far past the gate an offset is, 0 to 1. A stored extreme may sit a quantization step further out than the
// signal's, so the gate starts that much later.
float gate(float offset, vec2 noise) {
  if (noise.x <= 0.0) return 1.0;
  float start = u_gate.x + noise.y / noise.x;
  return clamp((abs(offset) / (noise.x * u_pxPerUnit) - start) / u_gate.y, 0.0, 1.0);
}

void main() {
  int row = gl_InstanceID;
  int column = gl_VertexID / 18;
  int k = gl_VertexID - column * 18;
  vec2 here = texelFetch(u_extent, ivec2(column, row), 0).rg;
  if (here.x > EMPTY) { gl_Position = HIDDEN; return; }
  v_row = row;
  int slot = u_row0 + row;
  float centre = (float(slot) - u_firstSlot + 0.5) * u_slotPx;
  vec2 noise = texelFetch(u_sigma, ivec2(slot, 0), 0).rg;
  float x = float(column);
  if (k < 6) {
    float top = centre + here.x;
    float bottom = centre - here.y;
    v_segment = vec4(-1e9);
    v_strength = u_fill;
    v_halfWidth = 0.0;
    int corner = k;
    bool right = corner == 1 || corner == 4 || corner == 5;
    bool low = corner == 2 || corner == 3 || corner == 5;
    gl_Position = clip(vec2(right ? x + 1.0 : x, low ? bottom : top));
    return;
  }
  bool isTop = k < 12;
  int corner = k - (isTop ? 6 : 12);
  vec2 next = column + 1 < u_columns ? texelFetch(u_extent, ivec2(column + 1, row), 0).rg : vec2(2.0 * EMPTY);
  float a = isTop ? here.x : -here.y;
  float b = next.x > EMPTY ? a : (isTop ? next.x : -next.y);
  vec2 p0 = vec2(x + 0.5, centre + a);
  vec2 p1 = next.x > EMPTY ? p0 : vec2(x + 1.5, centre + b);
  v_segment = vec4(p0, p1);
  float past = max(gate(a, noise), gate(b, noise));
  v_strength = mix(u_dim, 1.0, past);
  v_halfWidth = mix(u_halfWidth.x, u_halfWidth.y, past);
  float reach = v_halfWidth + 1.0;
  vec2 along = length(p1 - p0) > 0.0 ? normalize(p1 - p0) : vec2(1.0, 0.0);
  vec2 across = vec2(-along.y, along.x);
  bool end = corner == 1 || corner == 4 || corner == 5;
  bool side = corner == 2 || corner == 3 || corner == 5;
  vec2 p = (end ? p1 + along * reach : p0 - along * reach) + across * (side ? reach : -reach);
  gl_Position = clip(p);
}
`;

const TRACE_FRAGMENT = `#version 300 es
precision highp float;
// A packed level is its ramp step times PACK plus the visible row that drew it.
const float PACK = 4096.0;
uniform highp vec2 u_canvasPx;
flat in highp vec4 v_segment;
flat in float v_strength;
flat in float v_halfWidth;
flat in int v_row;
uniform bool u_packed;
out vec4 level;
void main() {
  highp vec2 p = vec2(gl_FragCoord.x, u_canvasPx.y - gl_FragCoord.y);
  float coverage = 1.0;
  if (v_segment.x > -1e8) {
    highp vec2 d = v_segment.zw - v_segment.xy;
    highp float t = dot(d, d) > 0.0 ? clamp(dot(p - v_segment.xy, d) / dot(d, d), 0.0, 1.0) : 0.0;
    coverage = clamp(v_halfWidth + 0.5 - length(p - v_segment.xy - t * d), 0.0, 1.0);
  }
  // Packed, MAX keeps the strongest step and with it the row that drew it, so a trace over a neighbour's slot
  // keeps its own tint.
  level = u_packed ? vec4(floor(coverage * v_strength * 255.0 + 0.5) * PACK + float(v_row)) : vec4(coverage * v_strength);
}
`;

/**
 * One triangle over the canvas: each pixel's level indexes a ramp row, the ink's or, where an event tints the
 * pixel's slot and column, its group's, mixed by the tint's strength.
 */
const RESOLVE = `#version 300 es
precision highp float;
void main() {
  vec2 p = vec2(gl_VertexID == 1 ? 3.0 : -1.0, gl_VertexID == 2 ? 3.0 : -1.0);
  gl_Position = vec4(p, 0.0, 1.0);
}
`;

const RESOLVE_FRAGMENT = `#version 300 es
precision highp float;
const float PACK = 4096.0;
precision highp int;
uniform highp sampler2D u_level;
uniform highp sampler2D u_ramp;
uniform highp sampler2D u_tint;  // per visible row and column: group's ramp row / 255, strength
uniform bool u_tinted;
uniform vec2 u_canvasPx;
uniform float u_firstSlot;
uniform float u_slotPx;
uniform int u_row0;
uniform int u_rows;
uniform bool u_packed;
out vec4 color;
void main() {
  ivec2 pixel = ivec2(gl_FragCoord.xy);
  float level = texelFetch(u_level, pixel, 0).r;
  int step = u_packed ? int(level / PACK) : int(level * 255.0 + 0.5);
  vec4 ink = texelFetch(u_ramp, ivec2(step, 0), 0);
  color = ink;
  if (!u_tinted) return;
  float y = u_canvasPx.y - gl_FragCoord.y;
  int row = u_packed ? int(level) - step * int(PACK) : int(floor(y / u_slotPx + u_firstSlot)) - u_row0;
  if (row < 0 || row >= u_rows) return;
  vec2 tint = texelFetch(u_tint, ivec2(pixel.x, row), 0).rg;
  if (tint.g <= 0.0) return;
  vec4 group = texelFetch(u_ramp, ivec2(step, int(tint.r * 255.0 + 0.5)), 0);
  color = mix(ink, group, tint.g);
}
`;

export const UNIFORMS = [
  "u_codes",
  "u_decode",
  "u_slots",
  "u_width",
  "u_rowsPerSeries",
  "u_c0",
  "u_originPx",
  "u_pxPerIndex",
  "u_canvasPx",
  "u_pxPerUnit",
  "u_row0",
  "u_rows",
  "u_limit",
  "u_extent",
  "u_sigma",
  "u_firstSlot",
  "u_slotPx",
  "u_columns",
  "u_halfWidth",
  "u_gate",
  "u_dim",
  "u_fill",
  "u_level",
  "u_ramp",
  "u_tint",
  "u_tinted",
  "u_packed",
] as const;

export interface Program {
  program: WebGLProgram;
  uniforms: Record<(typeof UNIFORMS)[number], WebGLUniformLocation | null>;
}

function compile(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader {
  const shader = gl.createShader(type);
  if (shader === null) throw new Error("could not create a shader");
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (gl.getShaderParameter(shader, gl.COMPILE_STATUS) !== true) {
    throw new Error(`shader failed to compile: ${gl.getShaderInfoLog(shader) ?? ""}`);
  }
  return shader;
}

function sources(kind: ProgramKind, sampler: SamplerKind): [string, string] {
  if (kind === "trace") return [TRACE, TRACE_FRAGMENT];
  if (kind === "resolve") return [RESOLVE, RESOLVE_FRAGMENT];
  const vertex = `#version 300 es
precision highp float;
precision highp int;
${EXTENT_COMMON.replace("SAMPLER", SAMPLERS[sampler])}
${kind === "extent-envelope" ? EXTENT_ENVELOPE : EXTENT_RAW}`;
  return [vertex, kind === "extent-envelope" ? EXTENT_ENVELOPE_FRAGMENT : EXTENT_RAW_FRAGMENT];
}

/** Compiles each (kind, sampler) program on first use. */
export class Programs {
  readonly #programs = new Map<string, Program>();

  constructor(readonly gl: WebGL2RenderingContext) {}

  get(kind: ProgramKind, sampler: SamplerKind = "float"): Program {
    const id = `${kind}-${sampler}`;
    const cached = this.#programs.get(id);
    if (cached !== undefined) return cached;
    const gl = this.gl;
    const [vertex, fragment] = sources(kind, sampler);
    const program = gl.createProgram();
    gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, vertex));
    gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, fragment));
    gl.linkProgram(program);
    if (gl.getProgramParameter(program, gl.LINK_STATUS) !== true) {
      throw new Error(`program failed to link: ${gl.getProgramInfoLog(program) ?? ""}`);
    }
    const uniforms = Object.fromEntries(
      UNIFORMS.map((name) => [name, gl.getUniformLocation(program, name)]),
    ) as Program["uniforms"];
    const entry = { program, uniforms };
    this.#programs.set(id, entry);
    return entry;
  }
}
