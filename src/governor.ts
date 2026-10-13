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
// Frame times alone can't tell an overloaded device from a display the OS
// throttled (iOS Low Power Mode, Chrome Battery Saver: 30 Hz). The old
// governor let its frame-period estimate drift up toward slow frames to
// avoid shedding fish under a throttle, and so it also normalized real
// overload: a steady 25-30 fps on a 60 Hz display never stepped down. So
// a step down needs two signals: frames late against the display, AND the
// school's own work (a backend's meter, below) nearly filling a 60 Hz
// period. A step down also has to lower the meter, or shedding stops.
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
}

/** A backend's meter and how far the governor may cut its school. */
export interface GovernorSpec {
  /** Never govern below this many fish. */
  floor: number;
  /** Governed counts are multiples of this. */
  quantum: number;
  /** What the meter measures, for the log lines ("GPU", "sim"). */
  label: string;
  /** Bound: the meter reads above this share of a 60 Hz period, so fewer
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
    boundShare: 0.6,
    headroomShare: 0.4,
    onCount: (governed, target) =>
      app.setOpacityScale(
        Math.min(GPU_TUNING.maxComp, Math.pow(target / governed, GPU_TUNING.opacityCompExp)),
      ),
  };
}

// The budget the meter is judged against: a 60 Hz period on any display.
// Late frames are judged against 45 fps on a 120 Hz display too, so the
// governor only defends 60 fps there, and an 8.3 ms period made headroom
// unreachable: 2,000 CPU fish cost ~4 ms of update() on an M2 Max, so a
// school shed under a CPU spike never grew back (measured, 2026-10-13).
const BUDGET_MS = 1000 / 60;

export function makeGovernor(
  system: SwarmSystem,
  panel: Pick<DevPanel, "refresh">,
  spec: GovernorSpec,
): Governor {
  const { floor, quantum, label } = spec;
  let target = system.getCount(); // what the user wants
  let governed = target; // what we've set
  // The display's frame period: the fastest frame seen, but never slower
  // than 60 Hz, so a device that boots already overloaded can't take its
  // slow cadence for the display's. It drifts up only while the meter
  // reads unbound (a 120 Hz window moved to a 60 Hz screen), never under
  // load.
  let displayMs = 1000 / 60;
  let loadMs = 0; // smoothed meter reading per frame (outlier-clamped)
  // Recent raw samples (capped only at the stall threshold), for the shed
  // check's window medians: loadMs's clamp (~67 ms at 60 Hz) would hide the
  // drop on a GPU at 100 ms/frame, a smoothed snapshot is too noisy to judge
  // a few-percent drop, and a mean lets one GC pause in the window fake a
  // failed step and block shedding for the session.
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
  let over = 0; // seconds of late, bound frames
  let under = 0; // seconds of headroom
  let warmup = 5;
  // Step history for the oscillation guard: after down → up → down within
  // two minutes the lower level sticks for the session, so the count
  // can't visibly breathe up and down.
  const steps: { dir: -1 | 1; at: number }[] = [];
  let sticky = false;
  // Shedding has to pay off: the meter over the second after a step down
  // must fall by at least a quarter of the count cut (a 22% cut: 5.5%;
  // fixed costs like the tonemap pass don't shrink with the count). If it
  // doesn't, the load isn't the school's (a busy main thread delaying the
  // probe, another app on the GPU) and cutting more fish won't help, so
  // step-downs stop until the slider moves.
  let verify: { from: number; at: number; before: number; cut: number } | null = null;
  let shedBlocked = false;

  const record = (ms: number): void => {
    // Clamped so one outlier (a GC pause, a deferred completion) can't
    // flip the estimate on its own.
    loadMs += (Math.min(ms, 4 * displayMs) - loadMs) * 0.25;
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

  const step = (dir: -1 | 1, count: number, why: string): void => {
    const now = performance.now();
    if (dir < 0) {
      const before = medianSince(now - 1000);
      verify = {
        from: now + 250, // let the resize land before judging
        at: now + 1250,
        before: before.n > 0 ? before.median : loadMs,
        cut: 1 - count / governed,
      };
    }
    governed = count;
    console.info(`[governor] ${why} — density ${dir < 0 ? "down" : "back up"} to ${governed}`);
    system.resize?.(governed);
    spec.onCount?.(governed, target);
    panel.refresh();
    over = 0;
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
    // A single long frame is a stall (tab switch, harness run, GC pause,
    // a modal), not sustained overload: one such frame would otherwise
    // count as "2 s over budget" by itself and drop density a notch per
    // hiccup (observed: 25k → 9k over four console harness runs). Only
    // sustained misses accumulate. A hidden tab defers GPU completion, so
    // neither may feed the meter: a probe spanning it is dropped.
    if (document.hidden || rawMs > 400) {
      over = 0;
      under = 0;
      loadMs = 0;
      ringLen = 0;
      probeEpoch++;
      // Judge a pending step on fresh samples after the stall.
      if (verify) {
        verify.from = performance.now() + 250;
        verify.at = verify.from + 1000;
      }
      return;
    }
    measure(workMs);
    if (warmup > 0) {
      warmup -= rawMs / 1000;
      displayMs = Math.min(displayMs, rawMs);
      return;
    }
    const current = system.getCount();
    if (current !== governed) {
      // The user took the slider — that's the new target, and a fresh start.
      target = current;
      governed = current;
      over = 0;
      under = 0;
      steps.length = 0;
      sticky = false;
      verify = null;
      shedBlocked = false;
      spec.onCount?.(governed, target);
      return;
    }
    if (verify && performance.now() >= verify.at) {
      const after = medianSince(verify.from);
      if (after.n > 0 && after.median > verify.before * (1 - 0.25 * verify.cut)) {
        shedBlocked = true;
        console.info(
          `[governor] fewer fish didn't lower ${label} time (${verify.before.toFixed(1)} → ${after.median.toFixed(1)} ms) — holding ${governed}`,
        );
      }
      verify = null;
    }
    // Bound: the school's work nearly fills the budget, so fewer fish
    // would let frames land on time. Below that, late frames come from
    // something shedding can't fix (a 30 Hz cap with 12 ms of work).
    const bound = loadMs > spec.boundShare * BUDGET_MS;
    if (rawMs < displayMs) displayMs = rawMs;
    else if (!bound) displayMs = Math.min(1000 / 60, displayMs + (rawMs - displayMs) * 0.005);
    const dtS = rawMs / 1000;
    // Late = over 1.5 display periods and under 45 fps, so 60 fps on a
    // 120 Hz display is never late.
    const late = rawMs > Math.max(displayMs * 1.5, 1000 / 45);
    if (late && bound) {
      over += dtS;
      under = 0;
    } else {
      over = Math.max(0, over - dtS);
      // Headroom: this frame's work would fit well inside the budget,
      // whatever the cadence (a throttled display still has room).
      under = loadMs < spec.headroomShare * BUDGET_MS ? under + dtS : 0;
    }
    if (over > 2 && governed > floor && !verify && !shedBlocked) {
      step(
        -1,
        Math.max(floor, Math.round((governed * 0.78) / quantum) * quantum),
        `late frames (~${rawMs.toFixed(0)} ms, ${label} ~${loadMs.toFixed(1)} ms)`,
      );
    } else if (under > 20 && governed < target && !sticky) {
      step(
        1,
        Math.min(target, Math.round((governed * 1.15) / quantum) * quantum),
        `headroom (${label} ~${loadMs.toFixed(1)} ms)`,
      );
    }
  };
  return {
    tick,
    // A slider move since the last tick is the new target already.
    target: () => (system.getCount() !== governed ? system.getCount() : target),
  };
}
