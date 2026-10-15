import type { DevPanel } from "./DevPanel";
import { GPU_TUNING } from "./params";
import type { SwarmSystem } from "./system";

// --- Adaptive count governor ---
// Weaker devices (or a contended GPU or main thread) breathe the school's
// density down and back up instead of dropping frames. Resizes happen in
// place — the school keeps swimming, and the brightness compensation
// (the GPU renderer's own, the CPU spec's below) keeps the perceived mass
// steady — so both directions are invisible.
//
// Smooth first: a hitch is worse than a thinner school, so it sheds fast
// (a step a second while frames miss vsync) and grows back slowly (20 s
// of headroom per +15%).
//
// Frame times alone can't tell an overloaded device from a display the OS
// throttled (iOS Low Power Mode, Chrome Battery Saver: 30 Hz). So a step
// down needs two signals over the last second: frames missing vsync (more
// than LATE_SHARE of them), AND the school's own work (a backend's meter,
// below) nearly filling the frame budget. Or one: the meter over the whole
// budget, because frames can drop where rAF never sees them (iOS Safari
// keeps rAF at 60 while the GPU falls behind). A step down also has to
// lower the meter, or shedding stops.
//
// One per mount: it closes over that backend's system, panel and meter.

export interface Governor {
  /**
   * Call once per frame, right after the frame's last submit. `workMs` is
   * the school's cost this frame when the backend times it inline (the
   * CPU's update()); a probing meter ignores it.
   */
  tick: (rawMs: number, workMs?: number) => void;
  /** The count the user wants, not the governed-down one. A re-mount
   * carries this, so a fresh device gets a fresh shot at full density. */
  target: () => number;
  /** What the governor sees, for the diagnostics overlay. */
  status: () => GovernorStatus;
  /** Measures but never changes the count (?governOff). */
  observeOnly: boolean;
}

export interface GovernorStatus {
  /** The meter's label ("GPU", "sim") and its smoothed reading. */
  label: string;
  loadMs: number;
  /** Share of frames that missed vsync over the last decision window. */
  lateShare: number;
  /** The meter reading above which the school counts as bound. */
  boundMs: number;
  /** The frame budget: a meter reading over it sheds on its own. */
  budgetMs: number;
  /** One short phrase: what it's doing, or why it isn't shedding. */
  state: string;
}

/** A backend's meter and how far the governor may cut its school. */
export interface GovernorSpec {
  /** Never govern below this many fish. */
  floor: number;
  /** Governed counts are multiples of this. */
  quantum: number;
  /** What the meter measures, for the log lines ("GPU", "sim"). */
  label: string;
  /** The fastest display the governor defends: the frame budget is the
   * display's period, but never shorter than this rate's. */
  maxHz: number;
  /** Bound: the meter reads above this share of the frame budget, so fewer
   * fish would let frames land on time. */
  boundShare: number;
  /** Headroom: the meter reads below this share, so a step up fits. */
  headroomShare: number;
  /** An asynchronous meter: start one measurement now and call `done` with
   * its milliseconds when it lands. Without one, tick's workMs is the
   * sample. */
  probe?: (done: (ms: number) => void) => void;
  /** After every count the governor sets, and when the slider takes over
   * (governed = target then). */
  onCount?: (governed: number, target: number) => void;
}

/**
 * GPU meter: the time from the frame's last submit until
 * queue.onSubmittedWorkDone() resolves (~0.2 ms when idle, 1-2 frames
 * stale). A busy main thread also delays that callback, so it can read as
 * GPU load; that is what the step-down check catches.
 */
export function gpuSpec(device: GPUDevice): GovernorSpec {
  let probing = false;
  return {
    floor: 8000,
    quantum: 500,
    label: "GPU",
    // A 120 Hz display is defended at 120: 200k on an M2 Max ran at 107 fps
    // with every twentieth frame doubled, and it read as struggling.
    maxHz: 120,
    boundShare: 0.8,
    headroomShare: 0.5,
    probe: (done) => {
      if (probing) return;
      probing = true;
      const start = performance.now();
      device.queue
        .onSubmittedWorkDone()
        .then(() => done(performance.now() - start))
        .catch(() => {}) // device lost: the mount (and this governor) is replaced
        .finally(() => {
          probing = false;
        });
    },
  };
}

/**
 * CPU meter: update() timed inline, this frame's own cost, so no GPU
 * signal is needed. update() is only part of the main thread's frame
 * (render submission and the browser's own work share it), so the sim is
 * bound at 0.6 of a period instead of the GPU's 0.8, and headroom sits at
 * 0.4 so a +15% step (about +20% sim time) lands short of bound.
 *
 * Shed fish are compensated like the GPU renderer's density law
 * (opacityCompExp), but relative to the count the governor was asked for:
 * a feel's cpu block stays literal at its own count, and a thinned school
 * keeps its perceived mass.
 */
export function cpuSpec(app: { setOpacityScale: (k: number) => void }): GovernorSpec {
  return {
    floor: 1000,
    quantum: 100,
    label: "sim",
    // 60, not the display's 120: at an 8.3 ms budget headroom was out of
    // reach (2,000 fish cost ~4 ms of update() on an M2 Max, so a school
    // shed under a CPU spike never grew back; measured 2026-10-13).
    maxHz: 60,
    boundShare: 0.6,
    headroomShare: 0.4,
    onCount: (governed, target) =>
      app.setOpacityScale(
        Math.min(GPU_TUNING.maxComp, Math.pow(target / governed, GPU_TUNING.opacityCompExp)),
      ),
  };
}

// Shed when more than this share of the last second's frames missed vsync
// (at 60 Hz, more than two a second) while the school is bound. 200k on a 120 Hz
// M2 Max missed ~5% and felt it; 100k felt fine.
const LATE_SHARE = 0.04;
// Each step down keeps this share of the school. Fixed, not proportional
// to the meter: the GPU meter is submit-to-done latency, which includes
// frames queued behind each other, so it overstates per-frame work.
const STEP_DOWN = 0.8;
// Overloaded: the meter's median over the window exceeds this share of the
// budget, so the GPU has fallen behind whatever rAF says. Not 1: the meter
// is submit-to-done, and a GPU keeping up one frame behind reads just over
// a period. An iPhone 11 on iOS 26 read 16-19.6 ms of 16.7 at 8k, 20k and
// 30k alike (60 fps, no late frames; iOS clocks the GPU down to fit the
// frame), so at 1 it shed to the floor within 8 s; at 45k, really behind,
// it read 65 ms with frames late. The Mac's stutter read 1.7x (200k: 14
// of 8.3 ms; 160k: 6.5, clean). The case 1 was set for, an iPhone 13 Pro
// on iOS 18's flagged WebGPU dropping frames at 18.9 ms (1.13x) unseen by
// rAF, no longer sheds on the meter alone: that's an opt-in flag build,
// and iOS 26 shows real overload in rAF.
const OVER_SHARE = 1.5;
const STEP_UP = 1.15;
// Decisions look at a trailing window this long, never reaching back
// across the last step (plus SETTLE_MS for the resize to land) or stall.
const WINDOW_MS = 1000;
const SETTLE_MS = 250;
const HEADROOM_S = 20;
const WARMUP_S = 2;

/**
 * Frame cadence over the last few seconds: the display period, and which
 * frames missed a vsync. The period is the median of recent frame times,
 * capped at 60 Hz's, so a device that runs every frame slow can't take its
 * slow cadence for the display's. Not a low percentile: after a late
 * frame rAF often fires a short catch-up frame, and under tapping on an
 * iPhone 11 those put the 10th percentile at 9 ms on a 60 Hz display, so
 * every on-time frame read late and the school shed to the floor. Catch-up
 * frames follow late ones, so they can never be the median.
 */
export class FramePacer {
  private readonly at = new Float64Array(256);
  private readonly ms = new Float64Array(256);
  private head = 0;
  private len = 0;
  private sinceSort = 0;
  private readonly sorted: number[] = [];
  periodMs = 1000 / 60;

  push(now: number, rawMs: number): void {
    this.at[this.head] = now;
    this.ms[this.head] = rawMs;
    this.head = (this.head + 1) & 255;
    this.len = Math.min(256, this.len + 1);
    if (++this.sinceSort >= 16 && this.len >= 32) {
      this.sinceSort = 0;
      this.sorted.length = 0;
      for (let k = 0; k < this.len; k++) this.sorted.push(this.ms[k]);
      this.sorted.sort((a, b) => a - b);
      this.periodMs = Math.min(1000 / 60, this.sorted[this.len >> 1]);
    }
  }

  /** Frames since `t` that ran over `lateMs`: how many, out of how many,
   * and the share. */
  lateSince(t: number, lateMs: number): { late: number; n: number; share: number } {
    let n = 0;
    let late = 0;
    for (let k = 0; k < this.len; k++) {
      const idx = (this.head - 1 - k) & 255;
      if (this.at[idx] < t) break;
      n++;
      if (this.ms[idx] > lateMs) late++;
    }
    return { late, n, share: n > 0 ? late / n : 0 };
  }

  clear(): void {
    this.len = 0;
  }
}

export function makeGovernor(
  system: SwarmSystem,
  panel: Pick<DevPanel, "refresh">,
  spec: GovernorSpec,
  // ?governOff: keep the meter and the frame stats for ?diag, never step.
  observeOnly = false,
): Governor {
  const { floor, quantum, label } = spec;
  let target = system.getCount(); // what the user wants
  let governed = target; // what we've set
  const pacer = new FramePacer();
  // The frame budget: the display's period, but no shorter than maxHz's.
  const budgetMs = (): number => Math.max(pacer.periodMs, 1000 / spec.maxHz);
  let loadMs = 0; // smoothed meter reading per frame (outlier-clamped)
  // Recent raw samples (capped only at the stall threshold), for window
  // medians: loadMs's clamp would hide the drop on a GPU at 100 ms/frame,
  // a smoothed snapshot is too noisy to judge a few-percent drop, and a
  // mean lets one GC pause in the window fake a failed step and block
  // shedding for the session.
  const RING = 256;
  const sampleAt = new Float64Array(RING);
  const sampleMs = new Float64Array(RING);
  let ringHead = 0;
  let ringLen = 0;
  const windowMs: number[] = [];
  const medianSince = (t: number): { median: number; n: number } => {
    windowMs.length = 0;
    for (let k = 0; k < ringLen; k++) {
      const idx = (ringHead - 1 - k + RING) % RING;
      if (sampleAt[idx] < t) break;
      windowMs.push(sampleMs[idx]);
    }
    windowMs.sort((a, b) => a - b);
    const n = windowMs.length;
    return { median: n > 0 ? windowMs[n >> 1] : 0, n };
  };
  // Bumped on a stall so a probe that spans it is discarded on landing.
  let probeEpoch = 0;
  // Decisions only use frames after this (the last step's settle, a stall).
  let since = -Infinity;
  let lateShare = 0; // the last window's, for status()
  let under = 0; // seconds of headroom
  let warmup = WARMUP_S;
  // Step history for the oscillation guard: after down → up → down within
  // two minutes the lower level sticks for the session, so the count
  // can't visibly breathe up and down.
  const steps: { dir: -1 | 1; at: number }[] = [];
  let sticky = false;
  // Shedding has to pay off: the meter over the window after a step down
  // must fall by at least a quarter of the count cut (a 20% cut: 5%;
  // fixed costs like the tonemap pass don't shrink with the count). If it
  // doesn't, the load isn't the school's (a busy main thread delaying the
  // probe, another app on the GPU) and cutting more fish won't help, so
  // step-downs pause: 30 s, doubling each time it happens again. Not for
  // the session: load the school makes in bursts reads the same way (taps
  // light cascades, so a step judged mid-tapping "didn't help"), and a
  // phone that later heats up must still be able to shed (iPhone 17 Pro,
  // 2026-10-15). Doubling keeps a load that really isn't the school's
  // from bleeding it to the floor a step at a time.
  let verify: { at: number; before: number; cut: number } | null = null;
  let blockedUntil = -Infinity;
  let blockMs = 30_000;

  const record = (ms: number): void => {
    // Clamped so one outlier (a GC pause, a deferred completion) can't
    // flip the estimate on its own.
    loadMs += (Math.min(ms, 4 * budgetMs()) - loadMs) * 0.25;
    sampleAt[ringHead] = performance.now();
    sampleMs[ringHead] = Math.min(ms, 400);
    ringHead = (ringHead + 1) % RING;
    ringLen = Math.min(RING, ringLen + 1);
  };
  const measure = (workMs: number | undefined): void => {
    if (spec.probe) {
      const epoch = probeEpoch;
      spec.probe((ms) => {
        if (epoch === probeEpoch) record(ms);
      });
    } else if (workMs !== undefined) {
      record(workMs);
    }
  };

  const step = (dir: -1 | 1, count: number, why: string, meterMs: number): void => {
    if (observeOnly) return;
    const now = performance.now();
    since = now + SETTLE_MS;
    verify =
      dir < 0 ? { at: since + WINDOW_MS, before: meterMs, cut: 1 - count / governed } : null;
    governed = count;
    console.info(`[governor] ${why} — density ${dir < 0 ? "down" : "back up"} to ${governed}`);
    system.resize?.(governed);
    spec.onCount?.(governed, target);
    panel.refresh();
    under = 0;
    steps.push({ dir, at: now });
    while (steps.length && now - steps[0].at > 120_000) steps.shift();
    const n = steps.length;
    if (n >= 3 && steps[n - 3].dir < 0 && steps[n - 2].dir > 0 && steps[n - 1].dir < 0) {
      sticky = true;
      console.info(`[governor] oscillating — holding ${governed} for this session`);
    }
  };

  const tick = (rawMs: number, workMs?: number): void => {
    const now = performance.now();
    // A single long frame is a stall (tab switch, harness run, GC pause,
    // a modal), not overload: it would otherwise read as a late window and
    // drop density a notch per hiccup (observed: 25k → 9k over four
    // console harness runs). A hidden tab defers GPU completion, so
    // neither may feed the meter: a probe spanning it is dropped.
    if (document.hidden || rawMs > 400) {
      under = 0;
      loadMs = 0;
      ringLen = 0;
      pacer.clear();
      probeEpoch++;
      since = now + SETTLE_MS;
      // Judge a pending step on fresh samples after the stall.
      if (verify) verify.at = since + WINDOW_MS;
      return;
    }
    pacer.push(now, rawMs);
    measure(workMs);
    if (warmup > 0) {
      warmup -= rawMs / 1000;
      return;
    }
    const current = system.getCount();
    if (current !== governed) {
      // The user took the slider — that's the new target, and a fresh start.
      target = current;
      governed = current;
      under = 0;
      steps.length = 0;
      sticky = false;
      verify = null;
      blockedUntil = -Infinity;
      blockMs = 30_000;
      since = now + SETTLE_MS;
      spec.onCount?.(governed, target);
      return;
    }
    if (verify) {
      if (now < verify.at) return;
      const after = medianSince(since);
      if (after.n > 0 && after.median > verify.before * (1 - 0.25 * verify.cut)) {
        blockedUntil = now + blockMs;
        blockMs *= 2;
        console.info(
          `[governor] fewer fish didn't lower ${label} time (${verify.before.toFixed(1)} → ${after.median.toFixed(1)} ms) — holding ${governed}`,
        );
      }
      verify = null;
    }
    const budget = budgetMs();
    const start = Math.max(now - WINDOW_MS, since);
    const frames = pacer.lateSince(start, budget * 1.5);
    // Enough of a window to judge: three quarters of it, and ten frames.
    if (now - start >= 0.75 * WINDOW_MS && frames.n >= 10) {
      lateShare = frames.share;
      // Stutter, not a burst: every quarter of the window has a late
      // frame. One disturbance (another app's GPU spike, a page-side
      // hiccup) packs its late frames into a fraction of a second, and only
      // one over half a second long can reach all four; at 160k on the M2
      // Max one such spike cut a school that ran a clean 120 fps.
      let spread = true;
      let after = frames.late;
      for (let q = 1; q <= 3 && spread; q++) {
        const rest = pacer.lateSince(start + ((now - start) * q) / 4, budget * 1.5).late;
        spread = after > rest; // a late frame in quarter q
        after = rest;
      }
      spread &&= after > 0; // and in the last quarter
      const meter = medianSince(start);
      // Bound: the school's work nearly fills the budget, so fewer fish
      // would let frames land on time. Below that, late frames come from
      // something shedding can't fix (a 30 Hz cap with 4 ms of work).
      const bound = meter.n > 0 && meter.median > spec.boundShare * budget;
      const overloaded = meter.n >= 10 && meter.median > OVER_SHARE * budget;
      const stutter = lateShare > LATE_SHARE && spread && bound;
      if ((stutter || overloaded) && governed > floor && now >= blockedUntil) {
        step(
          -1,
          Math.max(floor, Math.round((governed * STEP_DOWN) / quantum) * quantum),
          stutter
            ? `${Math.round(lateShare * 100)}% of frames late (${label} ~${meter.median.toFixed(1)} ms)`
            : `${label} ~${meter.median.toFixed(1)} ms, over a ${budget.toFixed(1)} ms frame`,
          meter.median,
        );
        return;
      }
    }
    // Headroom: this frame's work would fit well inside the budget,
    // whatever the cadence (a throttled display still has room).
    under = loadMs < spec.headroomShare * budget ? under + rawMs / 1000 : 0;
    if (under > HEADROOM_S && governed < target && !sticky) {
      step(
        1,
        Math.min(target, Math.round((governed * STEP_UP) / quantum) * quantum),
        `headroom (${label} ~${loadMs.toFixed(1)} ms)`,
        loadMs,
      );
    }
  };

  return {
    tick,
    observeOnly,
    // A slider move since the last tick is the new target already.
    target: () => (system.getCount() !== governed ? system.getCount() : target),
    status: () => ({
      label,
      loadMs,
      lateShare,
      boundMs: spec.boundShare * budgetMs(),
      budgetMs: budgetMs(),
      state: observeOnly
        ? "off (measuring only)"
          : warmup > 0
            ? "warming up"
            : verify
              ? "checking last step"
              : performance.now() < blockedUntil
                ? `paused: fewer fish didn't help (${Math.ceil((blockedUntil - performance.now()) / 1000)} s)`
                : lateShare > LATE_SHARE && governed <= floor
                  ? "at floor"
                  : sticky
                    ? "holding: oscillated"
                    : governed < target
                      ? under > 1
                        ? `headroom ${under.toFixed(0)} of ${HEADROOM_S} s`
                        : "stepped down"
                      : "idle",
    }),
  };
}
