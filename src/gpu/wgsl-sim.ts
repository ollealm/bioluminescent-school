import { GPU_TUNING, TUNING } from "../params";

/**
 * WGSL sources for the simulation: uniform grid build, event resolve, the
 * per-particle sim kernel (a line-for-line port of the CPU reference's
 * update loop — see src/cpu/ParticleSystem.ts), and the centroid reduction.
 *
 * TUNING constants are baked in at pipeline creation, exactly as they are
 * compile-time constants in the reference. Everything runtime-tweakable
 * arrives through the U[U_VEC4S] vec4 uniform array; the slot map below is
 * mirrored by the packing in GpuSwarm.update() - keep them in lockstep.
 *
 * Uniform slot map (flat f32 offsets in brackets):
 *   U[0]  dt, time, frame, activeCount
 *   U[1]  head.xyz, eventCount
 *   U[2]  dirAlign, dirCoh, driveW, dirSpeed
 *   U[3]  millF, gaitRate, breatheR, farR
 *   U[4]  roamTarget.xyz, fine-entries offset in gridData
 *   U[5]  pointer.xyz, pointerActive
 *   U[6]  lead.xyz, pvm                 (lead = pointer + velocity lead)
 *   U[7]  velHat.xyz, mouseRadius
 *   U[8]  splashBoost, igniteRate, perception, invCell
 *   U[9]  sepR, arriveR, cohesionW, alignmentW
 *   U[10] separationW, maxSpeedBase, dragF, energyDecay
 *   U[11] fearDecayF, fearSpreadDt, glowK, transmission
 *   U[12] refractory, kick, flowStrength, ft
 *   U[13] kx, ky, kz, swirl
 *   U[14] jitter, wanderPull, returnPull, repelForce
 *   U[15] fleeForce, regroup, containRadius, gait
 *   U[16] dirSep, shape, 0, 0           (lifelike levers; 1, 0 = v3 kernel)
 */

/** Uniform array length in vec4s (80 floats). Shared with GpuSwarm.ts. */
export const U_VEC4S = 20;

const D = TUNING.gridDim;

/** The 27 neighbor-cell offsets as a WGSL const array. */
function offsets27(): string {
  const rows: string[] = [];
  for (let z = -1; z <= 1; z++)
    for (let y = -1; y <= 1; y++)
      for (let x = -1; x <= 1; x++) rows.push(`vec3i(${x}, ${y}, ${z})`);
  return rows.join(", ");
}

/** Shared preamble: constants, structs, hash. */
export const WGSL_COMMON = /* wgsl */ `
const D: i32 = ${D};
const D2: u32 = ${D * D}u;
const CELLS: u32 = ${D * D * D}u;
const MAX_NEIGHBORS: u32 = ${TUNING.maxNeighbors}u;
const MAX_FORCE: f32 = ${TUNING.maxForce};
const CONTAGIOUS_ABOVE: f32 = ${TUNING.contagiousAbove};
const FLASH_ENDS_BELOW: f32 = ${TUNING.flashEndsBelow};
const GLOW_KNEE: f32 = ${TUNING.glowKnee};
const GLOW_GAIN: f32 = ${TUNING.glowGain};
const RADIAL_OUT: f32 = ${TUNING.radialGainOut};
const RADIAL_IN: f32 = ${TUNING.radialGainIn};
const FEAR_CEIL: f32 = ${TUNING.fearSpreadCeiling};
const E_SPEED_BOOST: f32 = ${TUNING.energySpeedBoost};
const F_SPEED_BOOST: f32 = ${TUNING.fearSpeedBoost};
const INBURST_RELAX: f32 = ${TUNING.inburstSepRelax};
const SCATTER_IGNITE_FRAC: f32 = ${GPU_TUNING.scatterIgniteFrac};
const PRED_SEP_RELAX: f32 = ${GPU_TUNING.predatorSepRelax};
const PAIR_PUSH: f32 = ${GPU_TUNING.pairPush};
const PAIR_RANGE: f32 = ${GPU_TUNING.pairPushRange};

// Fine collision grid (close-range personal space only; see GPU_TUNING).
const FINE_D: i32 = ${GPU_TUNING.fineGridDim};
const FINE_CELLS: u32 = ${GPU_TUNING.fineGridDim ** 3}u;
const FINE_INV: f32 = ${1 / GPU_TUNING.fineCellSize};
const FINE_CELL_SZ: f32 = ${GPU_TUNING.fineCellSize};

// The two grids share two buffers. gridData (u32): fine starts, coarse
// starts, coarse entries, fine entries — the fine-entries offset depends on
// the allocated capacity and arrives in U[19]. gridAtomics (atomic<u32>):
// coarse counts, fine counts, coarse cursors, fine cursors.
const FS_OFF: u32 = 0u;
const CS_OFF: u32 = ${GPU_TUNING.fineGridDim ** 3 + 1}u;
const CE_OFF: u32 = ${GPU_TUNING.fineGridDim ** 3 + 1 + TUNING.gridDim ** 3 + 1}u;
const CC_OFF: u32 = 0u;
const FC_OFF: u32 = ${TUNING.gridDim ** 3}u;
const CU_OFF: u32 = ${TUNING.gridDim ** 3 + GPU_TUNING.fineGridDim ** 3}u;
const FU_OFF: u32 = ${2 * TUNING.gridDim ** 3 + GPU_TUNING.fineGridDim ** 3}u;
const ATOMICS_LEN: u32 = ${2 * (TUNING.gridDim ** 3 + GPU_TUNING.fineGridDim ** 3)}u;
const WEDGE_AHEAD: f32 = ${GPU_TUNING.wedgeAhead};
const WEDGE_SIDE: f32 = ${GPU_TUNING.wedgeSide};
const WEDGE_SPEED_REF: f32 = ${GPU_TUNING.wedgeSpeedRef};

// Particle modes (match the CPU reference).
const FLOCK: u32 = 0u;
const WANDER: u32 = 1u;
const RETURN_: u32 = 2u;

// Event kinds (match EventKind in GpuSwarm.ts).
const EV_BURST: u32 = 1u;
const EV_STARTLE: u32 = 2u;
const EV_VANGUARD: u32 = 3u;
const EV_WANDER: u32 = 4u;
const EV_SCATTER: u32 = 5u;
const EV_IGNITE: u32 = 6u;
const EV_STRIKE: u32 = 7u;
const STRIKE_REACH: f32 = ${GPU_TUNING.strikeReach};
const STRIKE_APEX: f32 = ${GPU_TUNING.strikeApexWidth};

struct Dyn {
  pe: vec4f, // pos.xyz, energy
  vf: vec4f, // vel.xyz, fear
  pu: vec4f, // pulse epicenter.xyz, pulseSign
}

struct SelfState {
  s1: vec4f, // glow, refract, gaitClock, modeTimer
  s2: vec4f, // wanderDir.xyz, flags (mode | fired<<2, stored as f32)
}

struct Stat {
  colorSize: vec4f, // rgb, size
  traits: vec4f,    // speedFactor, social, seed, 0
}

struct Ev {
  tag: vec4u,   // kind, index, needsResolve, 0
  center: vec4f, // xyz, radius
  dir: vec4f,    // xyz, sign
  misc: vec4f,   // per-kind: WANDER(duration, prob) SCATTER(-) IGNITE(-)
}

fn pcg(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}

fn packFlags(mode: u32, fired: bool) -> f32 {
  return f32(mode | (select(0u, 1u, fired) << 2u));
}
fn flagsMode(w: f32) -> u32 { return u32(w) & 3u; }
fn flagsFired(w: f32) -> bool { return (u32(w) & 4u) != 0u; }

fn cellCoord(v: f32, invCell: f32) -> i32 {
  return clamp(i32(floor(v * invCell)) + (D / 2), 0, D - 1);
}

fn fineCoord(v: f32) -> i32 {
  return clamp(i32(floor(v * FINE_INV)) + (FINE_D / 2), 0, FINE_D - 1);
}

fn coarseCellOf(p: vec3f, invCell: f32) -> u32 {
  return u32(cellCoord(p.x, invCell))
    + u32(cellCoord(p.y, invCell)) * u32(D)
    + u32(cellCoord(p.z, invCell)) * D2;
}

fn fineCellOf(p: vec3f) -> u32 {
  return u32(fineCoord(p.x))
    + u32(fineCoord(p.y)) * u32(FINE_D)
    + u32(fineCoord(p.z)) * u32(FINE_D * FINE_D);
}
`;

/** Grid passes: clear counts, count, scan (starts + cursors), scatter. */
export const WGSL_GRID = /* wgsl */ `
${WGSL_COMMON}

@group(0) @binding(0) var<uniform> U: array<vec4f, ${U_VEC4S}>;
@group(0) @binding(1) var<storage, read> dynIn: array<Dyn>;
@group(0) @binding(2) var<storage, read_write> gridAtomics: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> gridData: array<u32>;

@compute @workgroup_size(256)
fn clearGrid(@builtin(global_invocation_id) gid: vec3u) {
  // Clears both grids' counts (cursors are initialized by the scans).
  if (gid.x < CELLS + FINE_CELLS) { atomicStore(&gridAtomics[gid.x], 0u); }
}

@compute @workgroup_size(256)
fn countGrid(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= u32(U[0].w)) { return; }
  let p = dynIn[i].pe.xyz;
  atomicAdd(&gridAtomics[CC_OFF + coarseCellOf(p, U[8].w)], 1u);
  atomicAdd(&gridAtomics[FC_OFF + fineCellOf(p)], 1u);
}

@compute @workgroup_size(256)
fn scatterGrid(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= u32(U[0].w)) { return; }
  let p = dynIn[i].pe.xyz;
  let cSlot = atomicAdd(&gridAtomics[CU_OFF + coarseCellOf(p, U[8].w)], 1u);
  gridData[CE_OFF + cSlot] = i;
  let fSlot = atomicAdd(&gridAtomics[FU_OFF + fineCellOf(p)], 1u);
  gridData[u32(U[4].w) + fSlot] = i; // U[4].w = fine-entries offset
}
`;

/**
 * Prefix scans for both grids: cell counts → start offsets + scatter
 * cursors. One workgroup each; threads walk contiguous chunks serially.
 */
export const WGSL_SCAN = /* wgsl */ `
${WGSL_COMMON}

@group(0) @binding(0) var<storage, read_write> gridAtomics: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> gridData: array<u32>;

var<workgroup> partial: array<u32, 256>;

fn scanRange(
  lid: u32,
  chunk: u32,
  total: u32,
  countOff: u32,
  startOff: u32,
  cursorOff: u32,
) {
  let base = lid * chunk;
  var local = 0u;
  for (var c = 0u; c < chunk; c++) {
    local += atomicLoad(&gridAtomics[countOff + base + c]);
  }
  partial[lid] = local;
  workgroupBarrier();
  if (lid == 0u) {
    var acc = 0u;
    for (var t = 0u; t < 256u; t++) {
      let v = partial[t];
      partial[t] = acc;
      acc += v;
    }
  }
  workgroupBarrier();
  var run = partial[lid];
  for (var c = 0u; c < chunk; c++) {
    let idx = base + c;
    let cnt = atomicLoad(&gridAtomics[countOff + idx]);
    gridData[startOff + idx] = run;
    atomicStore(&gridAtomics[cursorOff + idx], run);
    run += cnt;
  }
  if (lid == 255u) { gridData[startOff + total] = run; }
}

@compute @workgroup_size(256)
fn scanGrid(@builtin(local_invocation_index) lid: u32) {
  scanRange(lid, CELLS / 256u, CELLS, CC_OFF, CS_OFF, CU_OFF);
}

@compute @workgroup_size(256)
fn scanFine(@builtin(local_invocation_index) lid: u32) {
  scanRange(lid, FINE_CELLS / 256u, FINE_CELLS, FC_OFF, FS_OFF, FU_OFF);
}
`;

/**
 * Event resolve: events whose epicenter is "particle #index" (startle at a
 * random fish, turn vanguard, wander leader) get their center — and for
 * wander groups their outward heading — filled in on-GPU, so the host never
 * needs a position readback to schedule behavior.
 */
export const WGSL_RESOLVE = /* wgsl */ `
${WGSL_COMMON}

@group(0) @binding(0) var<uniform> U: array<vec4f, ${U_VEC4S}>;
@group(0) @binding(1) var<storage, read> dynIn: array<Dyn>;
@group(0) @binding(2) var<storage, read_write> events: array<Ev>;
@group(0) @binding(3) var<storage, read> centroid: array<vec4f>;

fn rndEv(ev: u32, salt: u32) -> f32 {
  let frame = u32(U[0].z);
  return f32(pcg(ev + pcg(salt + pcg(frame)))) * 2.3283064365386963e-10;
}

@compute @workgroup_size(${GPU_TUNING.maxEvents})
fn resolveEvents(@builtin(global_invocation_id) gid: vec3u) {
  let ev = gid.x;
  if (ev >= u32(U[1].w)) { return; }
  let E = events[ev];
  if (E.tag.z != 1u) { return; }
  let lp = dynIn[E.tag.y].pe.xyz;
  events[ev].center = vec4f(lp, E.center.w);
  if (E.tag.x == EV_WANDER) {
    // Head roughly outward from the centroid, with some tangent wobble
    // (port of spawnWanderGroup's direction pick).
    let cen = centroid[0].xyz;
    let d = vec3f(
      lp.x - cen.x + (rndEv(ev, 70u) - 0.5) * 2.0,
      lp.y - cen.y + (rndEv(ev, 71u) - 0.5) * 2.0,
      (lp.z - cen.z) * 0.6 + (rndEv(ev, 72u) - 0.5) * 1.5,
    );
    let dl = max(length(d), 1e-6);
    events[ev].dir = vec4f(d / dl, E.dir.w);
  }
}
`;

/**
 * Reduction over the just-written state. Two result slots:
 *   centroid[0] = flock centroid.xyz, flock count   (steers the school)
 *   centroid[1] = Σ|vel|, count(glow ≥ 0.8), 0, 0    (harness/debug stats)
 * Partials live at [2 .. 2+64) and [2+64 .. 2+128).
 */
export const WGSL_REDUCE = /* wgsl */ `
${WGSL_COMMON}

@group(0) @binding(0) var<uniform> U: array<vec4f, ${U_VEC4S}>;
@group(0) @binding(1) var<storage, read> dynOut: array<Dyn>;
@group(0) @binding(2) var<storage, read> selfB: array<SelfState>;
@group(0) @binding(3) var<storage, read_write> centroid: array<vec4f>;

var<workgroup> wsumA: array<vec4f, 256>;
var<workgroup> wsumB: array<vec4f, 256>;

@compute @workgroup_size(256)
fn reducePass1(
  @builtin(local_invocation_index) lid: u32,
  @builtin(workgroup_id) wg: vec3u,
) {
  let n = u32(U[0].w);
  var accA = vec4f(0.0);
  var accB = vec4f(0.0);
  var k = wg.x * 256u + lid;
  loop {
    if (k >= n) { break; }
    let d = dynOut[k];
    let s = selfB[k];
    if (flagsMode(s.s2.w) == FLOCK) {
      accA += vec4f(d.pe.xyz, 1.0);
    }
    accB += vec4f(length(d.vf.xyz), select(0.0, 1.0, s.s1.x >= 0.8), 0.0, 0.0);
    k += 64u * 256u;
  }
  wsumA[lid] = accA;
  wsumB[lid] = accB;
  workgroupBarrier();
  var s2 = 128u;
  while (s2 > 0u) {
    if (lid < s2) {
      wsumA[lid] += wsumA[lid + s2];
      wsumB[lid] += wsumB[lid + s2];
    }
    workgroupBarrier();
    s2 = s2 >> 1u;
  }
  if (lid == 0u) {
    centroid[2u + wg.x] = wsumA[0];
    centroid[2u + 64u + wg.x] = wsumB[0];
  }
}

var<workgroup> wsum2A: array<vec4f, 64>;
var<workgroup> wsum2B: array<vec4f, 64>;

@compute @workgroup_size(64)
fn reducePass2(@builtin(local_invocation_index) lid: u32) {
  wsum2A[lid] = centroid[2u + lid];
  wsum2B[lid] = centroid[2u + 64u + lid];
  workgroupBarrier();
  var s = 32u;
  while (s > 0u) {
    if (lid < s) {
      wsum2A[lid] += wsum2A[lid + s];
      wsum2B[lid] += wsum2B[lid + s];
    }
    workgroupBarrier();
    s = s >> 1u;
  }
  if (lid == 0u) {
    let r = wsum2A[0];
    if (r.w > 0.0) {
      centroid[0] = vec4f(r.xyz / r.w, r.w);
    } else {
      centroid[0] = vec4f(0.0);
    }
    centroid[1] = wsum2B[0];
  }
}
`;

/**
 * The sim kernel — the per-particle body of the CPU reference's update(),
 * ported clause by clause. Threads walk gridEntries (spatially sorted), so
 * workgroups process co-located fish: coherent branches, coherent loads.
 */
export const WGSL_SIM = /* wgsl */ `
${WGSL_COMMON}

@group(0) @binding(0) var<uniform> U: array<vec4f, ${U_VEC4S}>;
@group(0) @binding(1) var<storage, read> dynIn: array<Dyn>;
@group(0) @binding(2) var<storage, read_write> dynOut: array<Dyn>;
@group(0) @binding(3) var<storage, read_write> selfB: array<SelfState>;
@group(0) @binding(4) var<storage, read> stat: array<Stat>;
@group(0) @binding(5) var<storage, read> gridData: array<u32>;
@group(0) @binding(6) var<storage, read> events: array<Ev>;
@group(0) @binding(7) var<storage, read> centroid: array<vec4f>;

const OFFS = array<vec3i, 27>(${offsets27()});

fn rnd(i: u32, salt: u32) -> f32 {
  let frame = u32(U[0].z);
  return f32(pcg(i + pcg(salt + pcg(frame)))) * 2.3283064365386963e-10;
}

fn clampLen(v: vec3f, m: f32) -> vec3f {
  let l = length(v);
  if (l > m) { return v * (m / l); }
  return v;
}

// Body-shape anisotropy: soften the pull along the school's heading and
// stiffen it across, so the body relaxes into a traveling shoal instead of
// a ball. Callers skip it at shape 0 (the v3 kernel, untouched).
fn shapeAniso(v: vec3f, head: vec3f, shape: f32) -> vec3f {
  let par = dot(v, head) * head;
  return par * (1.0 - 0.6 * shape) + (v - par) * (1.0 + 0.8 * shape);
}

// Port of igniteIndex(): light up, remember the cascade's epicenter, and
// take the radial kick. force=true resets refractory (scatter seeds).
fn tryIgnite(
  i: u32, salt: u32, force: bool, c: vec3f, sign: f32, pos: vec3f, kick: f32,
  energy: ptr<function, f32>, fired: ptr<function, bool>,
  refract: ptr<function, f32>, pulse: ptr<function, vec3f>,
  pulseSign: ptr<function, f32>, vel: ptr<function, vec3f>,
  glow: ptr<function, f32>,
) {
  if (!force && (*refract > 0.0 || *energy > 0.6)) { return; }
  if (force) { *refract = 0.0; }
  *energy = 1.0;
  *fired = true;
  *pulse = c;
  *pulseSign = sign;
  if (kick <= 0.0) { return; }

  var k = (pos - c) * sign;
  let len = length(k);
  if (len > 1e-4) {
    k = k / len;
  } else {
    // The epicenter itself just shivers.
    let a = rnd(i, salt) * 6.2831853;
    let cc = rnd(i, salt + 1u) * 2.0 - 1.0;
    let s = sqrt(max(0.0, 1.0 - cc * cc)) * 0.4;
    k = vec3f(cos(a) * s, sin(a) * s, cc * 0.4);
  }
  let m = kick * (0.7 + rnd(i, salt + 2u) * 0.6);
  *vel += vec3f(
    (k.x + (rnd(i, salt + 3u) - 0.5) * 0.25) * m,
    (k.y + (rnd(i, salt + 4u) - 0.5) * 0.25) * m,
    (k.z + (rnd(i, salt + 5u) - 0.5) * 0.25) * m * 0.6,
  );
  *glow = 1.0;
}

@compute @workgroup_size(${GPU_TUNING.workgroupSize})
fn simStep(@builtin(global_invocation_id) gid: vec3u) {
  let n = u32(U[0].w);
  if (gid.x >= n) { return; }
  let i = gridData[CE_OFF + gid.x];

  // ---- Uniform unpack (slot map in wgsl-sim.ts header) ----
  let dt = U[0].x;
  let timeT = U[0].y;
  let head = U[1].xyz;
  let eventCount = u32(U[1].w);
  let dirAlign = U[2].x;
  let dirCoh = U[2].y;
  let driveW = U[2].z;
  let dirSpeed = U[2].w;
  let millF = U[3].x;
  let gaitRate = U[3].y;
  let breatheR = U[3].z;
  let farR = U[3].w;
  let roam = U[4].xyz;
  let pointerOn = U[5].w > 0.5;
  let lead = U[6].xyz;
  let pvm = U[6].w;
  let vhat = U[7].xyz;
  let mouseR = U[7].w;
  let splashBoost = U[8].x;
  let igniteRate = U[8].y;
  let perception = U[8].z;
  let invCell = U[8].w;
  let sepR = U[9].x;
  let arriveR = U[9].y;
  let cohesionW = U[9].z;
  let alignmentW = U[9].w;
  let separationW = U[10].x;
  let maxSpeedBase = U[10].y;
  let dragF = U[10].z;
  let energyDecay = U[10].w;
  let fearDecayF = U[11].x;
  let fearSpreadDt = U[11].y;
  let glowK = U[11].z;
  let transmission = U[11].w;
  let refractoryP = U[12].x;
  let kickP = U[12].y;
  let flowStrength = U[12].z;
  let ft = U[12].w;
  let kx = U[13].x;
  let ky = U[13].y;
  let kz = U[13].z;
  let swirl = U[13].w;
  let jitterP = U[14].x;
  let wanderPull = U[14].y;
  let returnPull = U[14].z;
  let repelForce = U[14].w;
  let fleeForce = U[15].x;
  let regroup = U[15].y;
  let containRadius = U[15].z;
  let gaitP = U[15].w;
  let dirSep = U[16].x;
  let shape = U[16].y;
  let perception2 = perception * perception;
  let cen = centroid[0].xyz;

  // ---- Load state ----
  let dIn = dynIn[i];
  var pos = dIn.pe.xyz;
  var energy = dIn.pe.w;
  var vel = dIn.vf.xyz;
  var fear = dIn.vf.w;
  var pulse = dIn.pu.xyz;
  var pulseSign = dIn.pu.w;
  let sf = selfB[i];
  var glow = sf.s1.x;
  var refract = sf.s1.y;
  var gaitClock = sf.s1.z;
  var modeTimer = sf.s1.w;
  var wanderDir = sf.s2.xyz;
  var mode = flagsMode(sf.s2.w);
  var fired = flagsFired(sf.s2.w);
  let st = stat[i];
  let spdF = st.traits.x;
  let social = st.traits.y;
  let seed = st.traits.z;

  // ---- Host events (burst / startle / vanguard / wander / scatter /
  // spontaneous ignition). The CPU applies these between frames; here they
  // apply at frame start, before anything reads own state. ----
  for (var ev = 0u; ev < eventCount; ev++) {
    let E = events[ev];
    let salt = 100u + ev * 16u;
    switch (E.tag.x) {
      case EV_BURST: {
        let dd = pos - E.center.xyz;
        let d = length(dd);
        if (d <= E.center.w) {
          let fall = 1.0 - d / E.center.w;
          let inv = 1.0 / max(0.2, d);
          let k = inv * (9.0 * fall + 2.0);
          vel += vec3f(dd.x * k, dd.y * k, dd.z * k * 0.5);
          refract = 0.0;
          energy = 1.0;
          fired = true;
          glow = 1.0;
          pulse = E.center.xyz;
          pulseSign = 1.0;
        }
      }
      case EV_STARTLE: {
        let dd = pos - E.center.xyz;
        let d = length(dd);
        if (d <= 3.4) {
          let f = 0.25 + 0.75 * (1.0 - d / 3.4);
          fear = max(fear, f);
          glow = max(glow, f * 0.8);
          if (d < 1.8 && rnd(i, salt) < 0.5) {
            tryIgnite(i, salt + 1u, false, E.center.xyz, 1.0, pos, kickP,
              &energy, &fired, &refract, &pulse, &pulseSign, &vel, &glow);
          }
        }
      }
      case EV_VANGUARD: {
        let dd = pos - E.center.xyz;
        if (dot(dd, dd) <= 3.5 * 3.5) {
          let dart = 2.0 + rnd(i, salt) * 1.4;
          vel += vec3f(head.x * dart, head.y * dart, head.z * dart * 0.6);
          gaitClock = 0.0;
          glow = max(glow, 0.5);
        }
      }
      case EV_WANDER: {
        if (mode == FLOCK) {
          let dd = pos - E.center.xyz;
          let leader = i == E.tag.y;
          if ((dot(dd, dd) <= 2.8 * 2.8 || leader)
              && (leader || rnd(i, salt) < E.misc.y)) {
            mode = WANDER;
            modeTimer = E.misc.x * (0.85 + rnd(i, salt + 1u) * 0.3);
            let w = vec3f(
              E.dir.x + (rnd(i, salt + 2u) - 0.5) * 0.35,
              E.dir.y + (rnd(i, salt + 3u) - 0.5) * 0.35,
              E.dir.z + (rnd(i, salt + 4u) - 0.5) * 0.35,
            );
            wanderDir = w / max(length(w), 1e-6);
          }
        }
      }
      case EV_SCATTER: {
        let d = max(0.5, length(pos));
        let jx = (rnd(i, salt) - 0.5) * 0.8;
        let jy = (rnd(i, salt + 1u) - 0.5) * 0.8;
        let jz = (rnd(i, salt + 2u) - 0.5) * 0.8;
        let s = 5.0 + rnd(i, salt + 3u) * 7.0;
        vel += vec3f(
          (pos.x / d + jx) * s,
          (pos.y / d + jy) * s,
          (pos.z / d + jz) * s * 0.6,
        );
        if (rnd(i, salt + 4u) < 0.35) {
          mode = WANDER;
          modeTimer = 2.0 + rnd(i, salt + 5u) * 2.5;
          wanderDir = vec3f(pos.x / d + jx, pos.y / d + jy, (pos.z / d + jz) * 0.5);
        }
        // Absolute seed count → fraction of the school (HANDOFF §5).
        if (rnd(i, salt + 6u) < SCATTER_IGNITE_FRAC) {
          tryIgnite(i, salt + 7u, true, pos, 1.0, pos, kickP,
            &energy, &fired, &refract, &pulse, &pulseSign, &vel, &glow);
        }
      }
      case EV_IGNITE: {
        if (i == E.tag.y) {
          tryIgnite(i, salt, false, pos, E.dir.w, pos, kickP,
            &energy, &fired, &refract, &pulse, &pulseSign, &vel, &glow);
        }
      }
      case EV_STRIKE: {
        // A hunter's dart: fish inside a forward-opening V along the strike
        // heading part hard away from the strike line, take contact-grade
        // fear, and a wake of outward ignitions lights the path. Depth uses
        // the same squash as the pointer column.
        let R = E.center.w;
        let dd = pos - E.center.xyz;
        let ddS = vec3f(dd.x, dd.y, dd.z * 0.45);
        let along = dot(ddS, E.dir.xyz);
        let reach = R * STRIKE_REACH;
        if (along > -R * 0.2 && along < reach) {
          let perp3 = ddS - along * E.dir.xyz;
          let perp = length(perp3);
          let t01 = clamp(along / reach, 0.0, 1.0);
          let sideR = R * mix(STRIKE_APEX, 1.0, t01); // the V opens forward
          if (perp < sideR) {
            // Strongest at the apex, fading toward the tip and edges.
            let fall = (1.0 - perp / sideR) * (1.0 - 0.55 * t01);
            var pdir: vec3f;
            if (perp > 1e-3) {
              pdir = perp3 / perp;
            } else {
              pdir = vec3f(-E.dir.y, E.dir.x, 0.0); // on the line: stable side
            }
            let blast = 9.0 * fall + 2.0;
            vel += vec3f(
              pdir.x * blast + E.dir.x * fall * 2.0,
              pdir.y * blast + E.dir.y * fall * 2.0,
              (pdir.z * blast + E.dir.z * fall * 2.0) * 0.6,
            );
            fear = max(fear, 0.35 + 0.65 * fall);
            glow = max(glow, 0.5 + 0.5 * fall);
            if (rnd(i, salt) < 0.35 * fall + 0.08) {
              tryIgnite(i, salt + 1u, false, E.center.xyz, 1.0, pos, kickP,
                &energy, &fired, &refract, &pulse, &pulseSign, &vel, &glow);
            }
          }
        }
      }
      default: {}
    }
  }

  // ---- Neighbor pass: boids accumulators + contagion pressure ----
  // Fair sampling under the neighbor cap: cell order and in-cell start are
  // hash-rotated per particle per frame, so the accepted subset is an
  // unbiased sample and anisotropy averages out (HANDOFF §5).
  var sep = vec3f(0.0);
  var hardPush = vec3f(0.0);
  var coh = vec3f(0.0);
  var aln = vec3f(0.0);
  var found = 0u;
  var contagion = 0.0;
  var bestJ = -1;
  var bestW = 0.0;
  var maxFear = 0.0;

  let cx = cellCoord(pos.x, invCell);
  let cy = cellCoord(pos.y, invCell);
  let cz = cellCoord(pos.z, invCell);
  let rot = pcg(i ^ pcg(u32(U[0].z) * 2654435761u));

  // Hash-rotated fair sweep over the 27-cell neighborhood. Deliberately NO
  // own-cell priority: any position-quantized sampling bias is static in
  // space and compounds — an earlier own-cell quota accreted the school
  // into visible sheets along the grid planes within a minute. Close-range
  // personal space is handled exhaustively by the fine collision grid
  // below, so flocking keeps the reference's clean 16-accept semantics.
  for (var oi = 0u; oi < 27u; oi++) {
    if (found >= MAX_NEIGHBORS) { break; }
    let o = OFFS[(oi + rot) % 27u];
    let gx = cx + o.x;
    let gy = cy + o.y;
    let gz = cz + o.z;
    if (gx < 0 || gx >= D || gy < 0 || gy >= D || gz < 0 || gz >= D) {
      continue;
    }
    let cell = u32(gx) + u32(gy) * u32(D) + u32(gz) * D2;
    let start = gridData[CS_OFF + cell];
    let cnt = gridData[CS_OFF + cell + 1u] - start;
    if (cnt == 0u) { continue; }
    let off = pcg(rot ^ cell) % cnt;
    for (var e = 0u; e < cnt; e++) {
      if (found >= MAX_NEIGHBORS) { break; }
      let j = gridData[CE_OFF + start + ((off + e) % cnt)];
      if (j == i) { continue; }
      let jd = dynIn[j];
      let d3 = jd.pe.xyz - pos;
      let d2 = dot(d3, d3);
      if (d2 > perception2) { continue; }

      let dist = max(sqrt(d2), 1e-4);
      if (dist < sepR) {
        let w = 1.0 - dist / sepR;
        sep -= d3 * (w * w / dist);
      }
      coh += jd.pe.xyz;
      aln += jd.vf.xyz;
      found++;

      if (jd.pe.w > CONTAGIOUS_ABOVE) {
        // Contagion is weighted by the source's remaining heat (v3 counted
        // every lit neighbor at full strength). At 20× density a lit fish
        // is a "spark carrier": the radial kick flies it a full perception
        // radius while still contagious, and with fuel everywhere the burn
        // never starves (HANDOFF §5). Fresh flashes ignite at full
        // strength; aging carriers fade — the click wavefront survives,
        // the wandering smolder does not.
        let srcHeat = (jd.pe.w - CONTAGIOUS_ABOVE) / (1.0 - CONTAGIOUS_ABOVE);
        let w = (1.0 - dist / perception) * srcHeat;
        contagion += w;
        if (w > bestW) {
          bestW = w;
          bestJ = i32(j);
        }
      }
      maxFear = max(maxFear, jd.vf.w);
    }
  }

  // ---- Fine collision grid: continuous close-range personal space ----
  // Every real overlap is seen every frame, from both sides — the
  // continuous response the reference got from scanning all neighbors,
  // which no probabilistic sampling reproduces at this density (sampled
  // kicks break pairs, alignment re-syncs them, overlap re-accumulates).
  {
    let range = min(sepR * PAIR_RANGE, FINE_CELL_SZ);
    let range2 = range * range;
    let fx = fineCoord(pos.x);
    let fy = fineCoord(pos.y);
    let fz = fineCoord(pos.z);
    let feOff = u32(U[4].w);
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
          let fcell = u32(gx) + u32(gy) * u32(FINE_D)
            + u32(gz) * u32(FINE_D * FINE_D);
          let fs = gridData[FS_OFF + fcell];
          let fcnt = gridData[FS_OFF + fcell + 1u] - fs;
          for (var k = 0u; k < fcnt; k++) {
            let j = gridData[feOff + fs + k];
            if (j == i) { continue; }
            let d3 = dynIn[j].pe.xyz - pos;
            let d2 = dot(d3, d3);
            if (d2 >= range2) { continue; }
            let dist = max(sqrt(d2), 1e-4);
            hardPush -= (d3 / dist) * ((1.0 - dist / range) * PAIR_PUSH);
          }
        }
      }
    }
  }

  // ---- Excitable medium: catch fire from glowing neighbors ----
  if (refract > 0.0) { refract -= dt; }
  if (contagion > 0.0 && bestJ >= 0 && !fired && refract <= 0.0
      && energy < CONTAGIOUS_ABOVE) {
    let p = 1.0 - exp(-contagion * transmission * dt);
    if (rnd(i, 40u) < p) {
      // Inherit the igniter's epicenter — the cascade stays one bubble.
      let bp = dynIn[u32(bestJ)].pu;
      tryIgnite(i, 41u, false, bp.xyz, bp.w, pos, kickP,
        &energy, &fired, &refract, &pulse, &pulseSign, &vel, &glow);
    }
  }

  // ---- Fear: wears off over time, spreads from panicked neighbors ----
  var fearI = fear * fearDecayF;
  if (maxFear * FEAR_CEIL > fearI) {
    fearI += (maxFear * FEAR_CEIL - fearI) * fearSpreadDt;
  }
  if (fearI < 0.005) { fearI = 0.0; }

  // Predator proximity, depth-squashed, measured against the lead point.
  // The danger zone is a hunter's: at rest it's the reference's sphere, at
  // speed it stretches into a lunge along the heading and narrows at the
  // flanks (a wedge). Behind-reach is untouched — dropping aft into the
  // wake is the fountain's escape and must stay safe. uNorm is the
  // dimensionless "how deep into the danger zone" (1 = boundary).
  var pd = vec3f(0.0);
  var pdist = 1e9;
  var uNorm = 1e9;
  if (pointerOn) {
    pd = pos - lead;
    let pdS = vec3f(pd.x, pd.y, pd.z * 0.45);
    pdist = max(length(pdS), 1e-4);
    uNorm = pdist / mouseR;
    let s01 = min(1.0, pvm / WEDGE_SPEED_REF);
    // Constant-folds away when both wedge dials are 0 (the shipped state:
    // the owner reverted early-reaction as illegible — see GPU_TUNING).
    if ((WEDGE_AHEAD > 0.0 || WEDGE_SIDE > 0.0) && s01 > 1e-3) {
      let along = dot(pdS, vhat);
      let perp = max(sqrt(max(dot(pdS, pdS) - along * along, 0.0)), 1e-5);
      let rAlong = mouseR * (1.0 + WEDGE_AHEAD * s01 * step(0.0, along));
      let rSide = mouseR * (1.0 - WEDGE_SIDE * s01);
      let a = along / rAlong;
      let p = perp / rSide;
      uNorm = max(sqrt(a * a + p * p), 1e-4);
    }
  }
  let threatNear = uNorm < 1.4;
  // Smoothstep falloff inside the danger zone — used by the escape steering
  // below and by the crowd-pressure compensation in separation (§5: at 50k
  // the dense wall around the cleared column would otherwise shove fresh
  // fish inward through pure separation pressure).
  var predFall = 0.0;
  if (pointerOn && uNorm < 1.0) {
    predFall = 1.0 - uNorm * uNorm * (3.0 - 2.0 * uNorm);
  }
  // Lasting fear keeps the reference sphere's semantics: the wedge's long
  // forward reach makes fish *evade* an approaching hunter early, but only
  // genuine proximity marks them (v3 fear = contact). Without this split
  // the pre-alarmed cohort ahead of a sweep — alarmed but not yet turned —
  // floods the fear population with cruise-heading fish.
  var fearFall = 0.0;
  if (pointerOn && pdist < mouseR) {
    let uS = pdist / mouseR;
    fearFall = 1.0 - uS * uS * (3.0 - 2.0 * uS);
  }

  // After a strike, far-flung survivors commit to returning.
  if (mode == FLOCK && fearI > 0.12 && !threatNear) {
    let h = pos - cen;
    if (dot(h, h) > breatheR * breatheR * 0.72) {
      mode = RETURN_;
      modeTimer = 30.0;
    }
  }

  // ---- Steering forces ----
  var acc = vec3f(0.0);
  var pairKick = vec3f(0.0);
  let maxSpeed = maxSpeedBase * spdF
    * (1.0 + energy * E_SPEED_BOOST + fearI * F_SPEED_BOOST);

  if (found > 0u) {
    // The director's separation gain: < 1 while the school contracts, a
    // brief > 1 pop on the release. Exactly 1.0 at collapse 0.
    let sepW = separationW * dirSep;
    // During an inburst, lit particles let the crowd condense — when the
    // flash fades, separation snaps back and the bubble pops.
    var relaxMul = 1.0;
    if (pulseSign < 0.0) {
      relaxMul *= 1.0 - min(1.0, energy) * INBURST_RELAX;
    }
    // Crowd-pressure relief around the predator: inside the column the
    // wall must not shove fish back in, and frightened fish near the
    // threat shoulder through the crowd instead of jamming against
    // personal space — at 20× density the panic wave has to move bodies.
    let panicRelax = max(predFall, select(0.0, fearI, threatNear));
    relaxMul *= 1.0 - panicRelax * PRED_SEP_RELAX;

    // Separation (all modes — personal space is universal)
    let sl = length(sep);
    if (sl > 1e-5) {
      let s = clampLen((sep / sl) * maxSpeed - vel, MAX_FORCE);
      acc += s * (sepW * relaxMul);
    }

    // Collision pushout obeys the same relaxes (inbursts must condense)
    // and vanishes with the user's separation slider. Fear damps it and
    // the threat's whole neighborhood disables it: threatened schools
    // pack tighter, collision noise would decohere the fountain's parting
    // velocity field, and shoves at the cleared column's compressed wall
    // would pump fish *into* the vacancy. It integrates beside acc rather
    // than through it: dodging a neighbor is not agitation, so it must
    // not trip the shear-glow knee.
    pairKick = hardPush
      * (min(1.0, sepW) * relaxMul * (1.0 - fearI)
        * select(1.0, 0.0, threatNear));

    if (mode == FLOCK) {
      // Cohesion with "arrival" damping (HANDOFF §4.1: without it the
      // swarm collapses into a ball).
      var t = coh / f32(found) - pos;
      // Anisotropic cohesion (shape > 0): before arrival damping, so the
      // damping still governs the body and the stretch is a gain, not a
      // spring. Uniform branch - free at shape 0.
      if (shape >= 0.001) { t = shapeAniso(t, head, shape); }
      let tl = length(t);
      if (tl > 1e-5) {
        let arrive = maxSpeed * min(1.0, tl / arriveR);
        t = clampLen((t / tl) * arrive - vel, MAX_FORCE);
        var cw = cohesionW * dirCoh * social;
        if (threatNear && fearI > 0.01) {
          cw *= 1.0 - fearI * 0.6;
        } else if (fearI > 0.08) {
          // Threat gone: fear flips into an urgent regroup drive.
          cw *= 1.0 + sqrt(fearI) * regroup * 0.8;
        }
        acc += t * cw;
      }
    }

    if (mode == FLOCK || mode == RETURN_) {
      // Alignment — match neighbors' *actual* pace (HANDOFF §4.2).
      var mv = aln / f32(found);
      let ml = length(mv);
      if (ml > 1e-5) {
        let matchSp = min(maxSpeed, ml * 1.15);
        mv = clampLen((mv / ml) * matchSp - vel, MAX_FORCE);
        let aw = alignmentW * dirAlign * social * (1.0 + fearI * 0.8);
        acc += mv * aw;
      }
    }
  }

  // ---- The school's collective will: cruise along the shared heading ----
  if (mode == FLOCK && driveW > 0.01) {
    let want = maxSpeedBase * spdF * dirSpeed;
    let w = clampLen(head * want - vel, MAX_FORCE);
    acc += w * (driveW * (0.7 + 0.3 * social));
  }

  // ---- Ambient flow field: layered sine currents + global swirl ----
  if (flowStrength > 0.0) {
    acc += vec3f(
      sin(ky * pos.y + ft) * cos(kz * pos.z - ft * 0.7),
      sin(kz * pos.z + ft * 1.1) * cos(kx * pos.x + ft * 0.6),
      sin(kx * pos.x - ft * 0.8) * cos(ky * pos.y + ft * 0.5) * 0.6,
    ) * flowStrength;
  }
  if (swirl > 0.0) {
    let rr = sqrt(pos.x * pos.x + pos.z * pos.z) + 1.0;
    acc.x += (-pos.z / rr) * swirl;
    acc.z += (pos.x / rr) * swirl;
  }
  if (jitterP > 0.0) {
    acc += vec3f(
      sin(timeT * 1.9 + seed * 61.0) * 1.2,
      sin(timeT * 1.6 + seed * 47.0) * 1.2,
      sin(timeT * 2.2 + seed * 29.0) * 0.8,
    ) * jitterP;
  }

  // ---- Mode behaviors ----
  let rCenter = length(pos);
  if (mode == FLOCK) {
    // Soft anchor plus group recentering, both chasing the roam target.
    // Recenter steers via the *centroid* (HANDOFF §4.4), computed here from
    // the GPU-resident centroid instead of a host constant.
    let h = pos - roam;
    let reCenter = (roam - cen) * 0.22;
    // The resting body elongates too: same anisotropy on the soft anchor.
    var anchor = -h * 0.02;
    if (shape >= 0.001) { anchor = shapeAniso(anchor, head, shape); }
    acc += anchor + reCenter;
    let rHome = length(h);
    if (rHome > breatheR) {
      let over = rHome - breatheR;
      acc += -h * ((over * over * 0.16) / rHome);
    }
    // Fish mill: coherent rotation around the shoal's own center.
    if (millF > 0.01) {
      let mv2 = pos.xy - cen.xy;
      let mr = length(mv2) + 0.5;
      acc.x += (-mv2.y / mr) * millF;
      acc.y += (mv2.x / mr) * millF;
    }
  } else if (mode == WANDER) {
    modeTimer -= dt;
    acc += wanderDir * (wanderPull * 3.2);
    // Far wall only — wanderers may leave the cloud, not the scene.
    if (rCenter > farR) {
      let over = rCenter - farR;
      acc += -pos * ((over * over * 0.3) / rCenter);
    }
    // Time's up — or a frightened wanderer gives up and heads home.
    if (modeTimer <= 0.0 || fearI > 0.35) {
      mode = RETURN_;
      modeTimer = 30.0;
    }
  } else {
    // RETURN — home in on the swarm centroid.
    modeTimer -= dt;
    let h = cen - pos;
    let hd = max(length(h), 1e-6);
    acc += (h / hd) * (returnPull * 3.5);
    // Release at the rim — arriving at dead center would clump it.
    if (hd < containRadius * 0.95 || modeTimer <= 0.0) {
      mode = FLOCK;
      // A soft rejoining glow (below the contagion threshold).
      if (energy < 0.5) { energy = 0.5; }
    }
  }

  // ---- Pointer: a predator to escape (fountain maneuver + wake) ----
  if (pointerOn) {
    if (uNorm < 1.0) {
      let fall = predFall;

      var e3 = pd / pdist;
      if (pvm > 1.5) {
        // Sideways component of the radial escape, relative to the
        // predator's heading: part around its line of travel.
        let along = dot(e3, vhat);
        var s3 = e3 - along * vhat;
        var sl = length(s3);
        if (sl < 0.15) {
          // Dead ahead on the path — dodge to a stable side.
          s3 = vec3f(-vhat.y, vhat.x, 0.0);
          sl = length(s3.xy);
          if (sl == 0.0) { sl = 1.0; }
        }
        s3 = s3 / sl;
        // Part sideways, drift back relative to its travel: fountain.
        let e = e3 * 0.45 + s3 - vhat * 0.35;
        e3 = e / max(length(e), 1e-6);
      }

      // Steer toward swimming at panic speed along the escape heading.
      let st3 = e3 * maxSpeed - vel;
      let stl = max(length(st3), 1e-6);
      let force = repelForce * fall * (1.0 + splashBoost);
      acc += vec3f(
        (st3.x / stl) * force,
        (st3.y / stl) * force,
        (st3.z / stl) * force * 0.6,
      );

      // Contact with the predator instills lasting fear.
      // Contact fear uses the sphere falloff, not the wedge's (see above).
      if (fearFall > fearI) { fearI = fearFall; }

      // Fast movement excites the swarm → cascades bursting away from the
      // predator's bow like a wake.
      if (igniteRate > 0.05 && !fired && refract <= 0.0) {
        if (rnd(i, 50u) < igniteRate * fall * dt) {
          tryIgnite(i, 51u, false, lead, 1.0, pos, kickP,
            &energy, &fired, &refract, &pulse, &pulseSign, &vel, &glow);
        }
      }
    }
    // Frightened particles keep swimming away — while the threat is close.
    if (fearI > 0.02 && threatNear) {
      let flee = fleeForce * fearI;
      acc += vec3f(
        (pd.x / pdist) * flee,
        (pd.y / pdist) * flee,
        (pd.z / pdist) * flee * 0.35,
      );
    }
  }

  // Regroup urgency: still afraid but the threat is out of range — swim
  // straight back to the shoal. Only outside the shoal body, so the center
  // doesn't clump.
  if (fearI > 0.02 && !threatNear) {
    let h = cen - pos;
    let hd = length(h);
    if (hd > containRadius * 0.75) {
      acc += h * ((regroup * 3.2 * sqrt(fearI)) / hd);
    }
  }

  // ---- Bubble pulse: sustained radial push from the cascade epicenter ----
  if (energy > 0.12 && kickP > 0.0) {
    let r3 = pos - pulse;
    let rd = length(r3);
    if (rd > 0.25) {
      // Collapses need extra gain to beat the crowd pressure.
      let gain = select(RADIAL_IN, RADIAL_OUT, pulseSign > 0.0);
      let f = (pulseSign * energy * kickP * gain) / rd;
      acc += vec3f(r3.x * f, r3.y * f, r3.z * f * 0.7);
    }
  }

  // ---- Burst-and-glide gait: thrust, then coast ----
  if (gaitP > 0.01) {
    gaitClock -= dt * gaitRate;
    if (gaitClock <= 0.0) {
      gaitClock = 0.35 + rnd(i, 60u) * 1.1;
      let sp = length(vel);
      let blend = select(0.2, dirSpeed * 1.2, mode == FLOCK);
      var d3: vec3f;
      if (sp > 0.2) { d3 = vel / sp + head * blend; }
      else { d3 = head + head * blend; }
      let dl = max(length(d3), 1e-6);
      let imp = gaitP * spdF * (0.4 + dirSpeed * 0.9 + fearI * 0.7)
        * maxSpeedBase * 0.34;
      vel += vec3f(
        (d3.x / dl) * imp,
        (d3.y / dl) * imp,
        (d3.z / dl) * imp * 0.7,
      );
      if (imp > 0.35 && glow < imp * 0.5) { glow = imp * 0.5; }
    }
  }

  // ---- Integrate ----
  vel = (vel + (acc + pairKick) * dt) * dragF;
  let sp2 = dot(vel, vel);
  let cap = maxSpeed * select(1.0, 0.85, mode == WANDER) + 0.4;
  if (sp2 > cap * cap) {
    vel *= cap / sqrt(sp2);
  }
  pos += vel * dt;

  // ---- Shear-triggered luminescence with the soft knee (HANDOFF §4.6) ----
  let aMag = length(acc);
  let gTarget = min(1.0, max(0.0, aMag - GLOW_KNEE) * GLOW_GAIN);
  glow += (gTarget - glow) * glowK;

  // ---- Flash bookkeeping ----
  var e = energy * energyDecay;
  if (e < 0.005) { e = 0.0; }
  energy = e;
  if (fired && e < FLASH_ENDS_BELOW) {
    fired = false;
    refract = refractoryP;
  }

  // ---- Write back ----
  dynOut[i] = Dyn(
    vec4f(pos, energy),
    vec4f(vel, fearI),
    vec4f(pulse, pulseSign),
  );
  selfB[i] = SelfState(
    vec4f(glow, refract, gaitClock, modeTimer),
    vec4f(wanderDir, packFlags(mode, fired)),
  );
}
`;
