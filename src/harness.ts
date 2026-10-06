import type { SwarmSystem } from "./system";

/**
 * Headless behavior harness — the acceptance tests that define "alive",
 * ported from the CPU reference (HANDOFF-GPU.md §6) to run against either
 * backend. The GPU backend forces it async: state comes back via buffer
 * readbacks, so every probe samples through a small adapter instead of
 * peeking at private arrays.
 *
 * Raw counts scale with the school (a healthy cascade at 50k lights ~19×
 * the particles it does at 2.6k), so the report carries fraction-of-count
 * versions of the count-valued metrics; compare *those* across backends.
 *
 * In the playground: `await harness()` (pauses the render loop; a 120 s
 * mood run takes a while — pass { moodSeconds: 30 } for a quick pass).
 *
 * The feel probes (`runFeelProbes`, below) are the other half: not pass/fail
 * invariants but shape readouts for tuning by eye - how stretched, how
 * crisp-edged, how internally structured the school is, whether it breathes,
 * and how fast it closes up behind the predator. They observe the live
 * school as-is, so tweak sliders, then measure.
 */

export interface HarnessReport {
  /** Lit count at ~0 / ~0.33 / ~0.66 / ~1.0 s after a click burst. */
  cascadeGrowth: number[];
  /** cascadeGrowth as fractions of count — comparable across backends. */
  cascadeGrowthFrac: number[];
  /** True if the burst cascade fully extinguished within 5 s. */
  cascadeDied: boolean;
  /** Fraction of lit particles moving away from their epicenter, ~0.33 s in. Healthy ≳ 0.7. */
  outburstOutwardFrac: number;
  /** Fraction moving inward ~0.3 s into a forced inburst. Healthy ≳ 0.55. */
  inburstInwardFrac: number;
  /** Mean dot(velocity, predator travel dir) among feared particles. Healthy < −0.2. */
  fountainDot: number;
  /**
   * Same, restricted to feared particles *behind* the predator — the
   * "part and drop behind" beat. Under the GPU fork's hunter wedge the
   * aggregate fountainDot cancels toward 0 by construction (prey ahead of
   * the lunge is herded forward-sideways before it parts), so this is the
   * signature to hold < −0.2 there. On the CPU reference both read alike.
   */
  fountainBehindDot: number;
  /** Particles left inside mouseRadius after the pointer parks 1.5 s. Healthy: ~0 (a mid-school park at 50k may still be evacuating). */
  predatorVacancy: number;
  /** Same count after 3 s — the settled, density-independent verdict. */
  predatorVacancy3s: number;
  /** Fraction of the school with fear > 0.1 after the park. Healthy ≳ 0.2. */
  panicFrac: number;
  /** Distinct director moods seen. Healthy: all 5. */
  moodsSeen: string[];
  /** Min/max of the ~1 s-averaged mean speed. Healthy: max ≳ 2× min. */
  meanSpeedRange: [number, number];
  /** Peak simultaneous count of shear-glow ≥ 0.8. Healthy: a small fraction. */
  maxHotGlow: number;
  /** maxHotGlow / count. The CPU reference sits under ~0.02 when healthy. */
  maxHotGlowFrac: number;
  /** Wall time per update() step over 300 steps (GPU: includes completion). */
  msPerStep: number;
  count: number;
  backend: "gpu" | "cpu";
}

export interface HarnessOptions {
  /** The live system to probe (it will be rebuilt and params touched). */
  system: SwarmSystem;
  count?: number;
  /** Length of the director observation run. 120 s is the acceptance bar. */
  moodSeconds?: number;
}

const DT = 1 / 60;

export interface ProbeState {
  pos: Float32Array;
  vel: Float32Array;
  energy: Float32Array;
  glow: Float32Array;
  fear: Float32Array;
  pulse: Float32Array;
  pointer: { x: number; y: number; z: number };
}

/** Uniform window into either backend. */
export interface Probe {
  backend: "gpu" | "cpu";
  read(): Promise<ProbeState>;
  /** Let queued GPU work drain (and mapAsync callbacks fire). CPU: no-op. */
  flush(): Promise<void>;
  /** Mean |vel| and count(glow ≥ 0.8) — cheap enough to call often. */
  stats(): Promise<{ meanSpeed: number; hotGlow: number }>;
}

export function makeProbe(system: SwarmSystem): Probe {
  const anySys = system as unknown as Record<string, unknown>;
  if (typeof anySys.readState === "function") {
    const gpu = system as SwarmSystem & {
      readState(): Promise<ProbeState>;
      waitIdle(): Promise<undefined>;
      stats(): { meanSpeed: number; hotGlow: number };
    };
    return {
      backend: "gpu",
      read: () => gpu.readState(),
      flush: async () => {
        await gpu.waitIdle();
        // One macrotask so pending mapAsync callbacks land.
        await new Promise((r) => setTimeout(r, 0));
      },
      stats: async () => gpu.stats(),
    };
  }
  // CPU reference: read the private simulation arrays directly, as the
  // original harness did.
  const cpu = system as unknown as {
    pos: Float32Array;
    vel: Float32Array;
    energy: Float32Array;
    glow: Float32Array;
    fear: Float32Array;
    pulse: Float32Array;
    pointer: { x: number; y: number; z: number };
  };
  const state = (): ProbeState => ({
    pos: cpu.pos,
    vel: cpu.vel,
    energy: cpu.energy,
    glow: cpu.glow,
    fear: cpu.fear,
    pulse: cpu.pulse,
    pointer: cpu.pointer,
  });
  return {
    backend: "cpu",
    read: async () => state(),
    flush: async () => {},
    stats: async () => {
      const { vel, glow } = state();
      const n = system.getCount();
      let sum = 0;
      let hot = 0;
      for (let i = 0; i < n; i++) {
        sum += Math.hypot(vel[i * 3], vel[i * 3 + 1], vel[i * 3 + 2]);
        if (glow[i] >= 0.8) hot++;
      }
      return { meanSpeed: sum / n, hotGlow: hot };
    },
  };
}

export function step(system: SwarmSystem, frames: number): void {
  for (let i = 0; i < frames; i++) system.update(DT);
}

/** Step with periodic drains so the GPU queue never runs unbounded. */
export async function stepFlushed(
  system: SwarmSystem,
  probe: Probe,
  frames: number,
): Promise<void> {
  let left = frames;
  while (left > 0) {
    const chunk = Math.min(30, left);
    step(system, chunk);
    left -= chunk;
    await probe.flush();
  }
}

function litCount(s: ProbeState, n: number, threshold = 0.3): number {
  let lit = 0;
  for (let i = 0; i < n; i++) if (s.energy[i] > threshold) lit++;
  return lit;
}

/** Fraction of lit particles whose velocity points away from their epicenter. */
function radialOutwardFrac(s: ProbeState, n: number): number {
  let out = 0;
  let total = 0;
  for (let i = 0; i < n; i++) {
    if (s.energy[i] <= 0.3) continue;
    const rx = s.pos[i * 3] - s.pulse[i * 3];
    const ry = s.pos[i * 3 + 1] - s.pulse[i * 3 + 1];
    const rz = s.pos[i * 3 + 2] - s.pulse[i * 3 + 2];
    const dot =
      rx * s.vel[i * 3] + ry * s.vel[i * 3 + 1] + rz * s.vel[i * 3 + 2];
    total++;
    if (dot > 0) out++;
  }
  return total > 0 ? out / total : 0;
}

/** Run every probe and return the report. */
export async function runHarness(opts: HarnessOptions): Promise<HarnessReport> {
  const sys = opts.system;
  const probe = makeProbe(sys);
  const n = opts.count ?? sys.getCount();
  const moodSeconds = opts.moodSeconds ?? 120;
  const P = sys.params;

  // --- Cascade growth + bubble coherence (outburst) ---
  // Isolate the probe: no spontaneous flashes or startles while measuring,
  // and a small seed so growth comes from transmission alone.
  const saved = {
    spontaneous: P.spontaneous,
    startle: P.startle,
    mouseRadius: P.mouseRadius,
  };
  P.spontaneous = 0;
  P.startle = 0;
  P.mouseRadius = 4;
  sys.rebuild(n);
  await stepFlushed(sys, probe, 480);
  sys.updateMousePosition({ x: -0.1, y: 0 });
  sys.burst();
  sys.clearPointer();
  // The GPU applies the burst inside the next update, so sample the seed
  // one frame in (the reference sampled synchronously; one 60th of decay
  // does not change the story).
  step(sys, 1);
  const cascadeGrowth = [litCount(await probe.read(), n)];
  step(sys, 19);
  let st = await probe.read();
  const outburstOutwardFrac = radialOutwardFrac(st, n);
  cascadeGrowth.push(litCount(st, n));
  step(sys, 20);
  cascadeGrowth.push(litCount(await probe.read(), n));
  step(sys, 20);
  cascadeGrowth.push(litCount(await probe.read(), n));
  await stepFlushed(sys, probe, 300);
  const cascadeDied = litCount(await probe.read(), n) === 0;
  P.spontaneous = saved.spontaneous;
  P.startle = saved.startle;
  P.mouseRadius = saved.mouseRadius;

  // --- Inburst coherence ---
  // Deviation from the v3 harness: measure on a fresh, settled field. The
  // reference ran this right after the cascade probe, which is benign at
  // 2.6k but at 50k (with the §4.3-rescaled refractory) leaves a large
  // partially-refractory population that starves the seeds and reduces the
  // measurement to a handful of coin-flip particles.
  const savedInburst = P.inburst;
  const savedSpont = P.spontaneous;
  const savedStartle = P.startle;
  P.inburst = 1;
  P.spontaneous = 0;
  P.startle = 0;
  sys.rebuild(n);
  await stepFlushed(sys, probe, 240);
  // ~6 seeds instead of the reference's ~3: a run whose lone seed fails to
  // spread reads as a false 0/1 (a single-particle bubble always moves
  // "outward" from its own ignition point).
  P.spontaneous = 30;
  step(sys, 12);
  P.spontaneous = 0;
  step(sys, 18);
  const inburstInwardFrac = 1 - radialOutwardFrac(await probe.read(), n);
  P.inburst = savedInburst;
  P.spontaneous = savedSpont;
  P.startle = savedStartle;

  // --- Predator: fountain maneuver during a sweep (+x travel) ---
  sys.rebuild(n);
  await stepFlushed(sys, probe, 480);
  for (let k = 0; k < 30; k++) {
    sys.updateMousePosition({ x: -0.6 + k * 0.04, y: 0 });
    sys.update(DT);
  }
  st = await probe.read();
  let dotSum = 0;
  let feared = 0;
  let behindSum = 0;
  let behindN = 0;
  for (let i = 0; i < n; i++) {
    if (st.fear[i] < 0.25) continue;
    const sp = Math.hypot(st.vel[i * 3], st.vel[i * 3 + 1], st.vel[i * 3 + 2]);
    if (sp < 0.3) continue;
    const dot = st.vel[i * 3] / sp; // travel direction is +x
    dotSum += dot;
    feared++;
    if (st.pos[i * 3] < st.pointer.x) {
      behindSum += dot;
      behindN++;
    }
  }
  const fountainDot = feared > 0 ? dotSum / feared : 0;
  const fountainBehindDot = behindN > 0 ? behindSum / behindN : 0;
  // Probe hygiene (v3 quirk that bites harder at 50k): without this the
  // vacancy probe settles around the sweep's stale active pointer and the
  // park begins with a violent pointer jump.
  sys.clearPointer();

  // --- Predator: parked-pointer vacancy + panic reach ---
  // Measured at the reference's 1.5 s mark and again at 3 s: evacuating a
  // 20× denser crowd is diffusion-limited, so a mid-school park may still
  // be clearing at 1.5 s. The 3 s mark is the settled truth at any density.
  sys.rebuild(n);
  await stepFlushed(sys, probe, 480);
  sys.updateMousePosition({ x: 0, y: 0 });
  await stepFlushed(sys, probe, 90);
  st = await probe.read();
  const countInside = (s: ProbeState): number => {
    let inside = 0;
    for (let i = 0; i < n; i++) {
      const dx = s.pos[i * 3] - s.pointer.x;
      const dy = s.pos[i * 3 + 1] - s.pointer.y;
      const dz = s.pos[i * 3 + 2] - s.pointer.z;
      if (Math.sqrt(dx * dx + dy * dy + dz * dz) < P.mouseRadius) inside++;
    }
    return inside;
  };
  const predatorVacancy = countInside(st);
  let panicked = 0;
  for (let i = 0; i < n; i++) if (st.fear[i] > 0.1) panicked++;
  const panicFrac = panicked / n;
  await stepFlushed(sys, probe, 90);
  const predatorVacancy3s = countInside(await probe.read());
  sys.clearPointer();

  // --- Director: moods, speed dynamics, glow discipline ---
  sys.rebuild(n);
  const moods = new Set<string>();
  let minSpeed = Infinity;
  let maxSpeed = 0;
  let maxHotGlow = 0;
  let windowAcc = 0;
  let windowSamples = 0;
  const totalFrames = moodSeconds * 60;
  for (let f = 0; f < totalFrames; f++) {
    sys.update(DT);
    moods.add(sys.moodName);
    // Sample the (cheap) reduced stats every half second of sim time and
    // average pairs into the reference's 1 s windows.
    if (f % 30 === 29) {
      await probe.flush();
      const s = await probe.stats();
      maxHotGlow = Math.max(maxHotGlow, s.hotGlow);
      windowAcc += s.meanSpeed;
      if (++windowSamples === 2) {
        const avg = windowAcc / 2;
        minSpeed = Math.min(minSpeed, avg);
        maxSpeed = Math.max(maxSpeed, avg);
        windowAcc = 0;
        windowSamples = 0;
      }
    }
  }

  // --- Perf ---
  await probe.flush();
  const t0 = performance.now();
  step(sys, 300);
  await probe.flush();
  const msPerStep = (performance.now() - t0) / 300;

  return {
    cascadeGrowth,
    cascadeGrowthFrac: cascadeGrowth.map((c) => Number((c / n).toFixed(4))),
    cascadeDied,
    outburstOutwardFrac: Number(outburstOutwardFrac.toFixed(2)),
    inburstInwardFrac: Number(inburstInwardFrac.toFixed(2)),
    fountainDot: Number(fountainDot.toFixed(2)),
    fountainBehindDot: Number(fountainBehindDot.toFixed(2)),
    predatorVacancy,
    predatorVacancy3s,
    panicFrac: Number(panicFrac.toFixed(2)),
    moodsSeen: [...moods],
    meanSpeedRange: [
      Number((minSpeed === Infinity ? 0 : minSpeed).toFixed(2)),
      Number(maxSpeed.toFixed(2)),
    ],
    maxHotGlow,
    maxHotGlowFrac: Number((maxHotGlow / n).toFixed(4)),
    msPerStep: Number(msPerStep.toFixed(2)),
    count: n,
    backend: probe.backend,
  };
}

// ====================================================================
// Feel probes - shape readouts for tuning toward "lifelike"
// ====================================================================

/**
 * What the school *looks* like over a stretch of sim time. Healthy ranges
 * are taste targets for the portfolio background ("a school constantly
 * moving and collapsing in on itself; the cursor is a predator the school
 * avoids and then closes up behind"), not invariants - compare runs at the
 * same count, before and after a slider change.
 */
export interface FeelReport {
  /**
   * Body stretch: sqrt(λ1/λ3) of the position covariance (longest axis over
   * shortest). A ball reads ≈ 1-1.3; a traveling shoal 1.5-2.5; past ~3.5
   * it reads as a ribbon or a split school. Mean and peak over samples.
   */
  elongation: { mean: number; max: number };
  /**
   * Edge crispness: radial width, in RMS radii, over which density falls
   * from 80% to 20% of the core's. A uniform ball with a hard skin ≈ 0.12;
   * a soft Gaussian haze ≈ 0.55. Shells are spherical, so a stretched body
   * reads softer than its skin is (a hard-skinned 2:1 ellipsoid ≈ 0.35).
   * Healthy school: ≲ 0.35 (a body with an edge, not a haze). Mean over
   * samples.
   */
  edgeSharpness: number;
  /**
   * Internal structure: coefficient of variation of fish-per-cell over a
   * 12³ grid on the school's (1-99 percentile) box, occupied cells only.
   * It also feels the radial profile, so read it beside edgeSharpness:
   * at 25k a featureless hard-edged ball reads ≈ 0.42 and a featureless
   * Gaussian haze ≈ 1.15. A crisp-edged school (edge ≲ 0.35) with CV 0.6-1.0
   * has visible streams, folds and gaps; past ~1.5 it is shredding into
   * clumps. Mean over samples.
   */
  densityCV: number;
  /**
   * The "collapsing in on itself" signature: the school's RMS radius over
   * time. `mean` in world units (tank radius 9). `amplitude` = (p90-p10)/mean:
   * < 0.05 is a static body, 0.1-0.3 visible breathing, > 0.4 the school
   * scatters and reforms. `period` (s) from zero crossings of the smoothed,
   * mean-removed series; 0 when it never crossed. A slow breath reads
   * lifelike at ~4-15 s; under ~2 s is jitter, not breathing.
   */
  breathing: { mean: number; amplitude: number; period: number };
  /**
   * Polarization: |mean velocity| / mean |velocity|. 1 = everyone swims the
   * same way, ~0 = milling or disordered. A traveling school cruises at
   * 0.7-0.95; a mill or contraction dips to 0.1-0.4. A lifelike run *swings*:
   * want max ≳ 0.8 and min ≲ 0.5 over 30 s, not a flat line either way.
   */
  speedCoherence: { mean: number; min: number; max: number };
  /**
   * "Closes up behind": seconds after the predator leaves (parked 2 s at the
   * centroid) until the vacated sphere is back to ≥ 70% of its pre-park
   * headcount. Healthy 1-3 s; under 0.5 s the cursor barely registered;
   * 8 = capped (the hole never closed, or the school swam off). -1 = skipped
   * (school not near the park point, or too sparse there to measure). -2 =
   * the park never cleared the sphere (still ≥ 70% full after 2 s), so there
   * was no hole to time.
   */
  refillSeconds: number;
  /** Director moods seen during the observation window. */
  moodsSeen: string[];
  /** Number of state samples taken (one per 0.5 s of sim time). */
  samples: number;
  seconds: number;
  count: number;
  backend: "gpu" | "cpu";
}

export interface FeelOptions {
  /** The live system to observe. */
  system: SwarmSystem;
  /**
   * Particle count. If it differs from the live count the school is rebuilt
   * at it (implies `fresh`). Default: the live count.
   */
  count?: number;
  /** Observation length in seconds of sim time. Default 30. */
  seconds?: number;
  /**
   * Rebuild and settle the school (8 s) before observing. Default false:
   * the live school is measured as-is, so slider tweaks carry straight in.
   */
  fresh?: boolean;
  /** Run the predator refill probe at the end (it disturbs the school). Default true. */
  refill?: boolean;
}

/** Sim frames between feel samples (0.5 s at the harness's fixed dt). */
const FEEL_SAMPLE_FRAMES = 30;

/** Fish count readable from a state window (the GPU count can shift between reads). */
function stateCount(sys: SwarmSystem, s: ProbeState): number {
  return Math.min(sys.getCount(), s.energy.length, Math.floor(s.pos.length / 3));
}

function centroidOf(s: ProbeState, n: number): [number, number, number] {
  let cx = 0;
  let cy = 0;
  let cz = 0;
  for (let i = 0; i < n; i++) {
    cx += s.pos[i * 3];
    cy += s.pos[i * 3 + 1];
    cz += s.pos[i * 3 + 2];
  }
  const k = n > 0 ? 1 / n : 0;
  return [cx * k, cy * k, cz * k];
}

/**
 * Eigenvalues of a symmetric 3×3 matrix, descending (closed-form
 * trigonometric solution - exact enough for a covariance, no iteration).
 */
function symEig3(
  a00: number, a01: number, a02: number,
  a11: number, a12: number, a22: number,
): [number, number, number] {
  const p1 = a01 * a01 + a02 * a02 + a12 * a12;
  if (p1 < 1e-18) {
    const d = [a00, a11, a22].sort((x, y) => y - x);
    return [d[0], d[1], d[2]];
  }
  const q = (a00 + a11 + a22) / 3;
  const p2 =
    (a00 - q) * (a00 - q) + (a11 - q) * (a11 - q) + (a22 - q) * (a22 - q) +
    2 * p1;
  const p = Math.sqrt(p2 / 6);
  const b00 = (a00 - q) / p;
  const b11 = (a11 - q) / p;
  const b22 = (a22 - q) / p;
  const b01 = a01 / p;
  const b02 = a02 / p;
  const b12 = a12 / p;
  const detB =
    b00 * (b11 * b22 - b12 * b12) -
    b01 * (b01 * b22 - b12 * b02) +
    b02 * (b01 * b12 - b11 * b02);
  const r = Math.max(-1, Math.min(1, detB / 2));
  const phi = Math.acos(r) / 3;
  const e1 = q + 2 * p * Math.cos(phi);
  const e3 = q + 2 * p * Math.cos(phi + (2 * Math.PI) / 3);
  return [e1, 3 * q - e1 - e3, e3];
}

/** sqrt(λ1/λ3) of the position covariance about c. */
function elongationOf(s: ProbeState, n: number, c: number[]): number {
  let xx = 0, xy = 0, xz = 0, yy = 0, yz = 0, zz = 0;
  for (let i = 0; i < n; i++) {
    const x = s.pos[i * 3] - c[0];
    const y = s.pos[i * 3 + 1] - c[1];
    const z = s.pos[i * 3 + 2] - c[2];
    xx += x * x; xy += x * y; xz += x * z;
    yy += y * y; yz += y * z; zz += z * z;
  }
  const k = n > 0 ? 1 / n : 0;
  const [l1, , l3] = symEig3(xx * k, xy * k, xz * k, yy * k, yz * k, zz * k);
  return Math.sqrt(Math.max(l1, 0) / Math.max(l3, 1e-9));
}

function rmsRadiusOf(s: ProbeState, n: number, c: number[]): number {
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const x = s.pos[i * 3] - c[0];
    const y = s.pos[i * 3 + 1] - c[1];
    const z = s.pos[i * 3 + 2] - c[2];
    sum += x * x + y * y + z * z;
  }
  return n > 0 ? Math.sqrt(sum / n) : 0;
}

const EDGE_SHELLS = 24;
const EDGE_REACH = 2.5; // shells span 0 .. 2.5 × RMS radius
const EDGE_CORE = 0.5; // the "inner plateau" is everything inside 0.5 × RMS

/**
 * Radial 80%→20% falloff width in RMS radii. The core density is pooled over
 * the inner shells (single inner shells are tiny and noisy). Each threshold
 * is crossed at its *outermost* downward crossing, so a hollow core (a mill
 * torus) still measures the outer skin rather than the hole.
 */
function edgeWidthOf(s: ProbeState, n: number, c: number[], rms: number): number {
  if (rms <= 1e-6) return 0;
  const w = (EDGE_REACH * rms) / EDGE_SHELLS;
  const counts = new Float64Array(EDGE_SHELLS);
  for (let i = 0; i < n; i++) {
    const x = s.pos[i * 3] - c[0];
    const y = s.pos[i * 3 + 1] - c[1];
    const z = s.pos[i * 3 + 2] - c[2];
    const sh = Math.floor(Math.sqrt(x * x + y * y + z * z) / w);
    if (sh < EDGE_SHELLS) counts[sh]++;
  }
  const dens = new Float64Array(EDGE_SHELLS);
  for (let k = 0; k < EDGE_SHELLS; k++) {
    const r0 = k * w;
    const r1 = r0 + w;
    dens[k] = counts[k] / ((4 / 3) * Math.PI * (r1 * r1 * r1 - r0 * r0 * r0));
  }
  const coreShells = Math.max(1, Math.round((EDGE_CORE * EDGE_SHELLS) / EDGE_REACH));
  let coreCount = 0;
  for (let k = 0; k < coreShells; k++) coreCount += counts[k];
  const coreR = coreShells * w;
  const plateau = coreCount / ((4 / 3) * Math.PI * coreR * coreR * coreR);
  if (plateau <= 0) return 0;
  const crossing = (frac: number): number => {
    const thr = frac * plateau;
    let j = -1;
    for (let k = EDGE_SHELLS - 1; k >= 0; k--) {
      if (dens[k] >= thr) {
        j = k;
        break;
      }
    }
    if (j < 0) return 0;
    if (j === EDGE_SHELLS - 1) return EDGE_REACH * rms;
    const t = (dens[j] - thr) / Math.max(1e-12, dens[j] - dens[j + 1]);
    return (j + 0.5 + t) * w;
  };
  return Math.max(0, crossing(0.2) - crossing(0.8)) / rms;
}

const CV_GRID = 12;

/** p-th percentile of a scratch array (sorted in place). */
function percentileSorted(a: Float32Array, p: number): number {
  if (a.length === 0) return 0;
  const idx = Math.min(a.length - 1, Math.max(0, Math.round(p * (a.length - 1))));
  return a[idx];
}

/**
 * CV of occupancy over a 12³ grid on the school's box. The box is the 1st to
 * 99th percentile per axis rather than the raw extent: a single wanderer
 * would otherwise stretch the grid and dilute every cell into noise.
 */
function densityCVOf(s: ProbeState, n: number, scratch: Float32Array): number {
  if (n < 2) return 0;
  const lo = [0, 0, 0];
  const span = [0, 0, 0];
  const axis = scratch.subarray(0, n);
  for (let a = 0; a < 3; a++) {
    for (let i = 0; i < n; i++) axis[i] = s.pos[i * 3 + a];
    axis.sort();
    lo[a] = percentileSorted(axis, 0.01);
    span[a] = Math.max(1e-6, percentileSorted(axis, 0.99) - lo[a]);
  }
  const cells = new Uint32Array(CV_GRID * CV_GRID * CV_GRID);
  for (let i = 0; i < n; i++) {
    const gx = Math.floor(((s.pos[i * 3] - lo[0]) / span[0]) * CV_GRID);
    const gy = Math.floor(((s.pos[i * 3 + 1] - lo[1]) / span[1]) * CV_GRID);
    const gz = Math.floor(((s.pos[i * 3 + 2] - lo[2]) / span[2]) * CV_GRID);
    // The top percentile lands exactly on the far face; fold it into the
    // last cell. Anything genuinely outside the box is a straggler: skip.
    const cx = gx === CV_GRID ? CV_GRID - 1 : gx;
    const cy = gy === CV_GRID ? CV_GRID - 1 : gy;
    const cz = gz === CV_GRID ? CV_GRID - 1 : gz;
    if (cx < 0 || cy < 0 || cz < 0 || cx >= CV_GRID || cy >= CV_GRID || cz >= CV_GRID) continue;
    cells[(cz * CV_GRID + cy) * CV_GRID + cx]++;
  }
  let occ = 0;
  let sum = 0;
  let sumSq = 0;
  for (let k = 0; k < cells.length; k++) {
    const v = cells[k];
    if (v === 0) continue;
    occ++;
    sum += v;
    sumSq += v * v;
  }
  if (occ === 0 || sum === 0) return 0;
  const mean = sum / occ;
  const variance = Math.max(0, sumSq / occ - mean * mean);
  return Math.sqrt(variance) / mean;
}

function coherenceOf(s: ProbeState, n: number): number {
  let vx = 0, vy = 0, vz = 0, speedSum = 0;
  for (let i = 0; i < n; i++) {
    const x = s.vel[i * 3];
    const y = s.vel[i * 3 + 1];
    const z = s.vel[i * 3 + 2];
    vx += x; vy += y; vz += z;
    speedSum += Math.sqrt(x * x + y * y + z * z);
  }
  return speedSum > 1e-9 ? Math.hypot(vx, vy, vz) / speedSum : 0;
}

/**
 * Breathing readout from an RMS-radius series sampled every dtSample s.
 * The period comes from zero crossings of the mean-removed series after a
 * 3-tap smooth (sample noise at 0.5 s spacing would otherwise read as a
 * 1 s "breath"): two crossings per cycle, so period = 2 × mean gap.
 */
function breathingOf(
  series: number[],
  dtSample: number,
): { mean: number; amplitude: number; period: number } {
  const m = series.length;
  if (m === 0) return { mean: 0, amplitude: 0, period: 0 };
  const mean = series.reduce((a, b) => a + b, 0) / m;
  const sorted = [...series].sort((a, b) => a - b);
  const pick = (p: number): number =>
    sorted[Math.min(m - 1, Math.max(0, Math.round(p * (m - 1))))];
  const amplitude = mean > 1e-9 ? (pick(0.9) - pick(0.1)) / mean : 0;
  const smooth = series.map((_, i) => {
    const a = series[Math.max(0, i - 1)];
    const b = series[i];
    const c = series[Math.min(m - 1, i + 1)];
    return (a + b + c) / 3 - mean;
  });
  const crossings: number[] = [];
  for (let i = 1; i < m; i++) {
    const a = smooth[i - 1];
    const b = smooth[i];
    if ((a < 0 && b >= 0) || (a >= 0 && b < 0)) {
      // Interpolated crossing time.
      crossings.push((i - 1 + a / (a - b)) * dtSample);
    }
  }
  let period = 0;
  if (crossings.length >= 2) {
    const gap =
      (crossings[crossings.length - 1] - crossings[0]) / (crossings.length - 1);
    period = 2 * gap;
  }
  return { mean, amplitude, period };
}

/**
 * The "closes up behind" beat: park the predator at NDC (0,0) - as the
 * vacancy probe does - for 2 s, release it, and time how long the vacated
 * sphere takes to refill to 70% of its pre-park headcount (cap 8 s).
 * Returns -1 when skipped (school off the park point or too sparse there)
 * and -2 when the park never cleared the sphere below that 70% mark.
 */
async function refillProbe(sys: SwarmSystem, probe: Probe): Promise<number> {
  const P = sys.params;
  const CAP_S = 8;
  const CHECK_FRAMES = 6; // 0.1 s resolution
  sys.clearPointer();
  // updateMousePosition projects synchronously on both backends, so one
  // read now gives the pre-park positions *and* the park point together.
  sys.updateMousePosition({ x: 0, y: 0 });
  const before = await probe.read();
  const park = { x: before.pointer.x, y: before.pointer.y, z: before.pointer.z };
  const n0 = stateCount(sys, before);
  const c = centroidOf(before, n0);
  const off = Math.hypot(c[0] - park.x, c[1] - park.y, c[2] - park.z);
  const countInside = (s: ProbeState): number => {
    const n = stateCount(sys, s);
    const r2 = P.mouseRadius * P.mouseRadius;
    let inside = 0;
    for (let i = 0; i < n; i++) {
      const dx = s.pos[i * 3] - park.x;
      const dy = s.pos[i * 3 + 1] - park.y;
      const dz = s.pos[i * 3 + 2] - park.z;
      if (dx * dx + dy * dy + dz * dz < r2) inside++;
    }
    return inside;
  };
  const pre = countInside(before);
  // Too sparse around the park point and "refilled" is just noise.
  const minPre = Math.max(10, n0 * 0.002);
  if (off > P.containRadius || pre < minPre) {
    sys.clearPointer();
    return -1;
  }
  await stepFlushed(sys, probe, 120);
  const target = 0.7 * pre;
  // Already "refilled" at release means the hole never opened: timing it
  // would report an instant 0.1 s refill for a predator nobody fled.
  const parked = countInside(await probe.read());
  sys.clearPointer();
  if (parked >= target) return -2;
  let frames = 0;
  while (frames < CAP_S * 60) {
    await stepFlushed(sys, probe, CHECK_FRAMES);
    frames += CHECK_FRAMES;
    if (countInside(await probe.read()) >= target) return frames / 60;
  }
  return CAP_S;
}

/**
 * Observe the school for `seconds` of sim time and report its feel. Does not
 * touch params. Rebuilds only on `fresh` (or a count change). Like
 * runHarness it steps the sim itself, so the caller should pause the render
 * loop for the duration.
 */
export async function runFeelProbes(opts: FeelOptions): Promise<FeelReport> {
  const sys = opts.system;
  const probe = makeProbe(sys);
  const seconds = opts.seconds ?? 30;
  const wantCount = opts.count ?? sys.getCount();
  if (opts.fresh || wantCount !== sys.getCount()) {
    sys.rebuild(wantCount);
    await stepFlushed(sys, probe, 480);
  }
  // Observe the undisturbed school, not a stale console-side pointer.
  sys.clearPointer();
  await probe.flush();

  const sampleCount = Math.max(1, Math.round((seconds * 60) / FEEL_SAMPLE_FRAMES));
  const dtSample = FEEL_SAMPLE_FRAMES / 60;
  const moods = new Set<string>();
  const elong: number[] = [];
  const edge: number[] = [];
  const cv: number[] = [];
  const radius: number[] = [];
  const coh: number[] = [];
  let scratch = new Float32Array(0);
  let n = sys.getCount();

  for (let k = 0; k < sampleCount; k++) {
    for (let f = 0; f < FEEL_SAMPLE_FRAMES; f++) {
      sys.update(DT);
      moods.add(sys.moodName);
    }
    await probe.flush();
    const s = await probe.read();
    n = stateCount(sys, s);
    if (n < 2) continue;
    if (scratch.length < n) scratch = new Float32Array(n);
    const c = centroidOf(s, n);
    const rms = rmsRadiusOf(s, n, c);
    elong.push(elongationOf(s, n, c));
    edge.push(edgeWidthOf(s, n, c, rms));
    cv.push(densityCVOf(s, n, scratch));
    radius.push(rms);
    coh.push(coherenceOf(s, n));
  }

  const refillSeconds = opts.refill === false ? -1 : await refillProbe(sys, probe);

  const avg = (a: number[]): number =>
    a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0;
  const r2 = (x: number): number => Number(x.toFixed(2));
  const r3 = (x: number): number => Number(x.toFixed(3));
  const br = breathingOf(radius, dtSample);
  return {
    elongation: { mean: r2(avg(elong)), max: r2(elong.length ? Math.max(...elong) : 0) },
    edgeSharpness: r3(avg(edge)),
    densityCV: r3(avg(cv)),
    breathing: { mean: r2(br.mean), amplitude: r3(br.amplitude), period: r2(br.period) },
    speedCoherence: {
      mean: r3(avg(coh)),
      min: r3(coh.length ? Math.min(...coh) : 0),
      max: r3(coh.length ? Math.max(...coh) : 0),
    },
    refillSeconds: r2(refillSeconds),
    moodsSeen: [...moods],
    samples: radius.length,
    seconds,
    count: n,
    backend: probe.backend,
  };
}

/** One-line digest of a FeelReport, for logging runs side by side. */
export function feelSummary(r: FeelReport): string {
  const k = r.count >= 1000 ? `${Math.round(r.count / 1000)}k` : `${r.count}`;
  const refill = r.refillSeconds < 0 ? "n/a" : `${r.refillSeconds}s`;
  return [
    `elong ${r.elongation.mean} (max ${r.elongation.max})`,
    `edge ${r.edgeSharpness}R`,
    `densCV ${r.densityCV}`,
    `breathe r${r.breathing.mean} ±${Math.round(r.breathing.amplitude * 100)}% T${r.breathing.period}s`,
    `coh ${r.speedCoherence.mean} [${r.speedCoherence.min}..${r.speedCoherence.max}]`,
    `refill ${refill}`,
    `moods ${r.moodsSeen.join("/")}`,
    `${k} ${r.backend} ${r.seconds}s`,
  ].join(" · ");
}
