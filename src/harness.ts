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

interface ProbeState {
  pos: Float32Array;
  vel: Float32Array;
  energy: Float32Array;
  glow: Float32Array;
  fear: Float32Array;
  pulse: Float32Array;
  pointer: { x: number; y: number; z: number };
}

/** Uniform window into either backend. */
interface Probe {
  backend: "gpu" | "cpu";
  read(): Promise<ProbeState>;
  /** Let queued GPU work drain (and mapAsync callbacks fire). CPU: no-op. */
  flush(): Promise<void>;
  /** Mean |vel| and count(glow ≥ 0.8) — cheap enough to call often. */
  stats(): Promise<{ meanSpeed: number; hotGlow: number }>;
}

function makeProbe(system: SwarmSystem): Probe {
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

function step(system: SwarmSystem, frames: number): void {
  for (let i = 0; i < frames; i++) system.update(DT);
}

/** Step with periodic drains so the GPU queue never runs unbounded. */
async function stepFlushed(
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
