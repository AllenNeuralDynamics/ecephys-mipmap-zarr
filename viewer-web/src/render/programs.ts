import type { SamplerKind } from "./textures.js";

export type ProgramKind = "envelope" | "raw";

const SAMPLERS: Record<SamplerKind, string> = { int: "isampler2D", uint: "usampler2D", float: "sampler2D" };

/**
 * Shared vertex code. Positions are in device pixels from the canvas's top left; x is the tile origin, a
 * float64 difference taken on the CPU, plus an integer offset within the tile, so float32 never holds a
 * session-scale sample index.
 */
const COMMON = `
uniform highp SAMPLER u_codes;
uniform highp sampler2D u_decode;  // per channel: scale, offset or min offset, max offset
uniform highp sampler2D u_slots;   // per channel: slot (negative when hidden), centre value
uniform int u_width;
uniform int u_rowsPerSeries;
uniform int u_c0;
uniform float u_originPx;
uniform float u_pxPerIndex;
uniform vec2 u_canvasPx;
uniform float u_firstSlot;
uniform float u_slotPx;
uniform float u_pxPerUnit;
// The slot's top and bottom rows; the fragment shader clips to them so one channel cannot paint over another.
flat out vec2 v_band;

float code(int series, int i) {
  return float(texelFetch(u_codes, ivec2(i % u_width, series * u_rowsPerSeries + i / u_width), 0).r);
}

vec4 clip(float xPx, float yPx) {
  return vec4(xPx / u_canvasPx.x * 2.0 - 1.0, 1.0 - yPx / u_canvasPx.y * 2.0, 0.0, 1.0);
}
`;

/**
 * Envelope: one quad (six vertices) per block, its sides snapped outward to whole pixels, so every pixel
 * column shows the full min and max of each block it touches. Interpolating between block centres instead
 * would undershoot the extremes at pixel centres.
 */
const ENVELOPE = `
const int CORNER_X[6] = int[6](0, 1, 0, 0, 1, 1);
const int CORNER_Y[6] = int[6](0, 0, 1, 1, 0, 1);
void main() {
  int c = gl_InstanceID;
  int block = gl_VertexID / 6;
  int corner = gl_VertexID - block * 6;
  int channel = u_c0 + c;
  vec4 decode = texelFetch(u_decode, ivec2(channel, 0), 0);
  vec2 slot = texelFetch(u_slots, ivec2(channel, 0), 0).rg;
  if (slot.x < 0.0) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
  float lo = code(2 * c, block) * decode.x + decode.y - slot.y;
  float hi = code(2 * c + 1, block) * decode.x + decode.z - slot.y;
  float centre = (slot.x - u_firstSlot + 0.5) * u_slotPx;
  v_band = vec2(centre - 0.5 * u_slotPx, centre + 0.5 * u_slotPx);
  float yLo = centre - lo * u_pxPerUnit;
  float yHi = centre - hi * u_pxPerUnit;
  // An envelope thinner than a pixel would vanish; widen it to one about its midpoint.
  if (yLo - yHi < 1.0) { float mid = 0.5 * (yLo + yHi); yLo = mid + 0.5; yHi = mid - 0.5; }
  float x0 = floor(u_originPx + float(block) * u_pxPerIndex);
  float x1 = max(x0 + 1.0, ceil(u_originPx + float(block + 1) * u_pxPerIndex));
  gl_Position = clip(CORNER_X[corner] == 1 ? x1 : x0, CORNER_Y[corner] == 1 ? yHi : yLo);
}
`;

const RAW = `
void main() {
  int c = gl_InstanceID;
  int i = gl_VertexID;
  int channel = u_c0 + c;
  vec4 decode = texelFetch(u_decode, ivec2(channel, 0), 0);
  vec2 slot = texelFetch(u_slots, ivec2(channel, 0), 0).rg;
  if (slot.x < 0.0) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
  float value = code(c, i) * decode.x + decode.y - slot.y;
  float centre = (slot.x - u_firstSlot + 0.5) * u_slotPx;
  v_band = vec2(centre - 0.5 * u_slotPx, centre + 0.5 * u_slotPx);
  gl_Position = clip(u_originPx + (float(i) + 0.5) * u_pxPerIndex, centre - value * u_pxPerUnit);
}
`;

const FRAGMENT = `#version 300 es
precision mediump float;
uniform highp vec2 u_canvasPx;
uniform vec4 u_color;
flat in highp vec2 v_band;
out vec4 color;
void main() {
  highp float y = u_canvasPx.y - gl_FragCoord.y;
  if (y < v_band.x || y > v_band.y) discard;
  color = u_color;
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
  "u_firstSlot",
  "u_slotPx",
  "u_pxPerUnit",
  "u_color",
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

/** Compiles each (kind, sampler) program on first use. */
export class Programs {
  readonly #programs = new Map<string, Program>();

  constructor(readonly gl: WebGL2RenderingContext) {}

  get(kind: ProgramKind, sampler: SamplerKind): Program {
    const id = `${kind}-${sampler}`;
    const cached = this.#programs.get(id);
    if (cached !== undefined) return cached;
    const gl = this.gl;
    const vertex = `#version 300 es
precision highp float;
precision highp int;
${COMMON.replace("SAMPLER", SAMPLERS[sampler])}
${kind === "envelope" ? ENVELOPE : RAW}`;
    const program = gl.createProgram();
    gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, vertex));
    gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, FRAGMENT));
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
