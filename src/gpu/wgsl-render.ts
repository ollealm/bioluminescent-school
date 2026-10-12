/**
 * WGSL render sources: instanced-quad sprites (a port of the reference's
 * point-sprite GLSL — same styles, velocity stretch, twinkle, in-hue flash
 * mix) accumulated additively into an HDR target, then a hue-preserving
 * shoulder tonemap onto the canvas.
 *
 * The tonemap is the structural "never white" law (HANDOFF §4.5): additive
 * pile-ups compress toward brightness 1 while keeping their hue ratios, so
 * a dense flash saturates *as the palette color*, not as white.
 *
 * Render uniform slots (struct RU):
 *   vp     view-projection matrix
 *   a      viewport.xy (device px), dpr, time
 *   b      size (density-compensated), opacity (compensated), flashBright, twinkle
 *   c      stretch, motionGlow, style, tonemapKnee
 *   flash  flash color rgb, camera z
 *   crowd  nRef (expected 27-fine-cell count at nominal density), crowdDim, 0, 0
 */
import { GPU_TUNING } from "../params";

// Fine collision grid geometry, duplicated from WGSL_COMMON in wgsl-sim.ts
// (both derive from GPU_TUNING, so the sprite shader indexes the same cells).
const FINE_GRID = /* wgsl */ `
const FINE_D: i32 = ${GPU_TUNING.fineGridDim};
const FINE_INV: f32 = ${1 / GPU_TUNING.fineCellSize};

fn fineCoord(v: f32) -> i32 {
  return clamp(i32(floor(v * FINE_INV)) + (FINE_D / 2), 0, FINE_D - 1);
}
`;

export const WGSL_SPRITES = /* wgsl */ `
struct RU {
  vp: mat4x4f,
  a: vec4f,
  b: vec4f,
  c: vec4f,
  flash: vec4f,
  crowd: vec4f,
}
${FINE_GRID}

struct Dyn {
  pe: vec4f,
  vf: vec4f,
  pu: vec4f,
}
struct SelfState {
  s1: vec4f,
  s2: vec4f,
}
struct Stat {
  colorSize: vec4f,
  traits: vec4f,
}

@group(0) @binding(0) var<uniform> R: RU;
@group(0) @binding(1) var<storage, read> dyn: array<Dyn>;
@group(0) @binding(2) var<storage, read> selfB: array<SelfState>;
@group(0) @binding(3) var<storage, read> stat: array<Stat>;
// The sim's gridData: fine-cell starts at offset 0, so a cell's population
// is gridData[cell + 1] - gridData[cell].
@group(0) @binding(4) var<storage, read> gridData: array<u32>;

struct VOut {
  @builtin(position) clip: vec4f,
  @location(0) color: vec3f,
  @location(1) misc: vec4f, // energy, tw, glow, seed
  @location(2) dirStretch: vec4f, // dir.xy, stretch, unused
  @location(3) uvc: vec2f,
  // Flat so a crowd fade of exactly 1.0 stays exactly 1.0 (no
  // interpolation round-off): crowdDim 0 must be bit-identical.
  @location(4) @interpolate(flat) crowdFade: f32,
}

@vertex
fn vsSprite(
  @builtin(vertex_index) vi: u32,
  @builtin(instance_index) ii: u32,
) -> VOut {
  let corner = vec2f(f32(vi & 1u) * 2.0 - 1.0, f32(vi >> 1u) * 2.0 - 1.0);
  let d = dyn[ii];
  let pos = d.pe.xyz;
  let energy = d.pe.w;
  let vel = d.vf.xyz;
  let st = stat[ii];
  let seed = st.traits.z;
  let glow = selfB[ii].s1.x;

  let time = R.a.w;
  let uTwinkle = R.b.w;
  let uStretch = R.c.x;

  // Slow per-particle brightness oscillation ("breathing" of individuals).
  let tw = 1.0 - uTwinkle * 0.5
    * (0.5 + 0.5 * sin(time * (0.5 + seed * 1.9) + seed * 87.0));

  // The camera has identity rotation, so view-space velocity is velocity:
  // screen-space swim direction + stretch fall out directly.
  let sp = length(vel.xy);
  let stretch = 1.0 + uStretch * min(sp * 0.16, 1.7);
  var dir = vec2f(1.0, 0.0);
  if (sp > 0.001) { dir = vel.xy / sp; }

  let depth = R.flash.w - pos.z; // -viewZ; camera looks down -z from camZ
  var sizePx = st.colorSize.w * R.b.x * (1.0 + energy * 0.45)
    * (0.72 + 0.28 * tw) * R.a.z * (340.0 / max(depth, 0.1)) * stretch;
  if (depth < 0.1) { sizePx = 0.0; } // behind the camera: collapse

  var clip = R.vp * vec4f(pos, 1.0);
  clip = vec4f(
    clip.xy + corner * (sizePx / R.a.xy) * clip.w,
    clip.zw,
  );

  // Crowd dim: perceived brightness is ~n * opacity, so a core 4x denser
  // than the nominal school would read 4x brighter and blow out into one
  // glowing spot. Fading by (nRef / nLocal)^(0.8 * crowdDim) makes it read
  // ~1.3x at full dim; sparse fish are never boosted. Uniform branch: no
  // grid reads and a factor of exactly 1 at crowdDim 0.
  //
  // nLocal must be position-continuous. A plain box sum over the 27 cells
  // around the fish's own cell jumps whenever the fish crosses a cell face
  // (a whole slab of 9 cells swaps in/out), and that step renders as
  // brightness banding along the grid planes in dense cores — the same
  // position-quantized signature as the reverted own-cell quota. Instead,
  // each cell's count is weighted by a separable tent of its center's
  // offset from the fish, in cell units: d = (o + 0.5) - frac per axis,
  // in [-1.5, 1.5], w = prod max(0, 1 - |d| / 1.5). At a face crossing the
  // slab leaving the stencil and the one entering it both sit at |d| = 1.5
  // (w = 0), so the weighted mean is continuous. Normalizing by sum(w) and
  // scaling by 27 keeps the old meaning (expected fish in 27 cells), so the
  // host's nRef stays valid. Off-grid cells get w = 0 (excluded from both
  // sums); sum(w) >= (1 - 0.5/1.5)^3 from the own cell alone. Cost per
  // vertex is the same 27 cells x 2 reads as before plus a few ALU ops.
  var crowdFade = 1.0;
  let crowdDim = R.crowd.y;
  if (crowdDim > 0.0) {
    let fx = fineCoord(pos.x);
    let fy = fineCoord(pos.y);
    let fz = fineCoord(pos.z);
    // Fractional position inside the own cell, [0, 1] (clamped so a fish
    // outside the grid, pinned to an edge cell by fineCoord, stays sane).
    let gpos = pos * FINE_INV + f32(FINE_D / 2);
    let fr = clamp(gpos - vec3f(f32(fx), f32(fy), f32(fz)), vec3f(0.0), vec3f(1.0));
    var wSum = 0.0;
    var nSum = 0.0;
    for (var oz = -1; oz <= 1; oz++) {
      for (var oy = -1; oy <= 1; oy++) {
        for (var ox = -1; ox <= 1; ox++) {
          let gx = fx + ox;
          let gy = fy + oy;
          let gz = fz + oz;
          if (gx < 0 || gx >= FINE_D || gy < 0 || gy >= FINE_D
              || gz < 0 || gz >= FINE_D) {
            continue;
          }
          let off = vec3f(f32(ox), f32(oy), f32(oz)) + 0.5 - fr;
          let t = max(vec3f(0.0), vec3f(1.0) - abs(off) / 1.5);
          let w = t.x * t.y * t.z;
          // fineCellOf's indexing, per neighbor cell.
          let c = u32(gx) + u32(gy) * u32(FINE_D) + u32(gz) * u32(FINE_D * FINE_D);
          nSum += w * f32(gridData[c + 1u] - gridData[c]);
          wSum += w;
        }
      }
    }
    let nLocal = nSum / max(wSum, 1e-6) * 27.0;
    crowdFade = pow(clamp(R.crowd.x / max(nLocal, 1.0), 0.0, 1.0), 0.8 * crowdDim);
  }

  var out: VOut;
  out.clip = clip;
  out.color = st.colorSize.rgb;
  out.misc = vec4f(energy, tw, glow, seed);
  out.dirStretch = vec4f(dir, stretch, 0.0);
  out.uvc = corner * 0.5;
  out.crowdFade = crowdFade;
  return out;
}

@fragment
fn fsSprite(in: VOut) -> @location(0) vec4f {
  let uvc = in.uvc;
  let dir = in.dirStretch.xy;
  let vStretch = in.dirStretch.z;
  // Rotate into the swim frame and compress the across-axis: the sprite
  // becomes a streak along the direction of motion.
  let pr = vec2f(
    dot(uvc, dir),
    (uvc.x * dir.y - uvc.y * dir.x) * vStretch,
  );
  let d = length(pr) * 2.0;
  if (d > 1.0) { discard; }

  let time = R.a.w;
  let style = u32(R.c.z);
  let seed = in.misc.w;
  let core = exp(-d * d * 12.0);
  var a: f32;

  if (style == 0u) {
    // glow — soft gaussian blob
    a = exp(-d * d * 4.5) * 0.8 + core * 0.6;
  } else if (style == 1u) {
    // orb — defined body with a faint halo
    a = smoothstep(0.55, 0.38, d) * 0.9 + exp(-d * d * 2.6) * 0.3;
  } else if (style == 2u) {
    // spark — bright core with a flare along the swim axis
    let c1 = exp(-pow(pr.x * 16.0, 2.0)) * exp(-pow(pr.y * 2.2, 2.0));
    let c2 = exp(-pow(pr.y * 16.0, 2.0)) * exp(-pow(pr.x * 2.2, 2.0));
    a = core * 0.85 + max(c1, c2) * 0.7;
  } else {
    // firefly — glow with slow individual blinking
    let blink = 0.18 + 0.82
      * pow(0.5 + 0.5 * sin(time * (0.9 + seed * 1.4) + seed * 92.0), 5.0);
    a = (exp(-d * d * 4.5) * 0.8 + core * 0.6) * blink;
  }

  // Excitement (cascade flash) + agitation (shear glow): both lean toward
  // the palette's saturated flash hue and brighten — the creature glows,
  // it doesn't detonate.
  let e = clamp(in.misc.x, 0.0, 1.0);
  let g = clamp(in.misc.z, 0.0, 1.0) * R.c.y;
  var col = mix(in.color, R.flash.rgb, min(1.0, e * 0.45 + g * 0.15));
  col *= 1.0 + e * R.b.z + g * 0.8 + core * 0.55;

  a *= R.b.y * (0.35 + 0.65 * in.misc.y) * (1.0 + e * 0.5 + g * 0.3);
  a *= in.crowdFade;
  return vec4f(col * a, a);
}
`;

export const WGSL_TONEMAP = /* wgsl */ `
struct RU {
  vp: mat4x4f,
  a: vec4f,
  b: vec4f,
  c: vec4f,
  flash: vec4f,
  crowd: vec4f,
}

@group(0) @binding(0) var<uniform> R: RU;
@group(0) @binding(1) var hdr: texture_2d<f32>;

struct VOut {
  @builtin(position) clip: vec4f,
}

@vertex
fn vsFull(@builtin(vertex_index) vi: u32) -> VOut {
  // One oversized triangle covering the screen.
  var out: VOut;
  let x = f32(i32(vi & 1u) * 4 - 1);
  let y = f32(i32(vi >> 1u) * 4 - 1);
  out.clip = vec4f(x, y, 0.0, 1.0);
  return out;
}

@fragment
fn fsTonemap(in: VOut) -> @location(0) vec4f {
  let t = textureLoad(hdr, vec2i(in.clip.xy), 0);
  let c = t.rgb;
  let knee = R.c.w;
  let m = max(c.r, max(c.g, c.b));
  var scale = 1.0;
  if (m > knee) {
    // Soft shoulder, asymptotic to 1: identity below the knee, compressed
    // above — applied to the max channel so hue ratios are preserved.
    let span = 1.0 - knee;
    let mapped = knee + span * (1.0 - exp(-(m - knee) / span));
    scale = mapped / m;
  }
  // The WebGL reference accumulated alpha additively into an alpha:true
  // (premultiplied) canvas — same composite: rgb + page·(1−a). rgb may
  // exceed a (brightness boosts), exactly as the reference emitted it.
  let a = clamp(t.a, 0.0, 1.0);
  return vec4f(c * scale, a);
}
`;
