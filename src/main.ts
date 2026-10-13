import { DevPanel, PANEL_STORAGE_KEY } from "./DevPanel";
import { installFeelSwitcher, type FeelSwitcher } from "./feel";
import type { FilmstripOptions } from "./review";
import type { Snapshot, SwarmSystem } from "./system";

/**
 * Bootstrap: pick a backend, wire pointer/resize/loop around it.
 *
 * Backend ladder (docs/HANDOFF-GPU.md §5 "degrading gracefully"):
 *   1. WebGPU on a real (non-software) adapter → the compute rewrite
 *      (src/gpu, no three.js).
 *   2. otherwise → the CPU engine (v3 plus the fork's levers) at a few
 *      thousand fish (src/cpu, lazy chunk that carries three.js).
 *
 * A lost GPU device re-mounts the whole backend: one more try on a fresh
 * adapter, then the CPU fallback, carrying the live snapshot across so the
 * tuned params survive.
 *
 * URL overrides for side-by-side comparison during tuning:
 *   ?backend=cpu|gpu   force a backend (gpu also admits a software adapter)
 *   ?count=NNN         initial particle count
 */

interface App {
  system: SwarmSystem;
  render: () => void;
  resize: (width: number, height: number, dpr: number) => void;
  dispose: () => void;
}

/** Everything that belongs to one backend instance and is rebuilt on re-mount. */
interface Mounted {
  application: App;
  system: SwarmSystem;
  panel: DevPanel;
  feel: FeelSwitcher;
  governor: Governor | null;
}

const app = document.getElementById("app")!;
const backendTag = document.getElementById("backend-tag");

const query = new URLSearchParams(location.search);
const forced = query.get("backend");
const countOverride = Number(query.get("count")) || 0;

// Count is decided by input type only. Width is a layout fact, not a device
// fact: a hidden tab, embed or preview pane can report a narrow window at
// boot, and the school would stay pinned at the phone count forever.
const coarsePointer = window.matchMedia("(pointer: coarse)").matches;
const GPU_COUNT = coarsePointer ? 12000 : 25000;
const CPU_COUNT = coarsePointer ? 2200 : 4000;

let live: Mounted | null = null;
// The async harness/feel probes step the sim themselves. A depth, not a
// flag: overlapping console runs must not unpause each other early.
let pauseDepth = 0;
let gpuRetried = false;

/** True for software / fallback adapters (SwiftShader, WARP, ...). They run
 * WebGPU compute far slower than the CPU reference runs on the CPU, so they
 * count as "no WebGPU". Checked three ways for older implementations:
 * `adapter.info` (current spec), the removed `adapter.isFallbackAdapter`,
 * and the removed `requestAdapterInfo()`. */
async function isFallbackAdapter(adapter: GPUAdapter): Promise<boolean> {
  const a = adapter as GPUAdapter & {
    isFallbackAdapter?: boolean;
    requestAdapterInfo?: () => Promise<{ isFallbackAdapter?: boolean }>;
  };
  if (a.isFallbackAdapter || a.info?.isFallbackAdapter) return true;
  try {
    return Boolean((await a.requestAdapterInfo?.())?.isFallbackAdapter);
  } catch {
    return false;
  }
}

async function detectWebGPU(): Promise<GPUAdapter | null> {
  if (forced === "cpu") return null;
  if (!("gpu" in navigator)) return null;
  try {
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: "high-performance",
    });
    if (adapter && forced !== "gpu" && (await isFallbackAdapter(adapter))) {
      console.info(
        "[boot] WebGPU adapter is a software fallback - using the CPU backend (?backend=gpu to force it)",
      );
      return null;
    }
    return adapter;
  } catch {
    return null;
  }
}

/**
 * Boot one backend into #app with its own panel and governor, and make it
 * the live one. `carry` is the previous backend's snapshot on a re-mount:
 * params/presets/palette survive, the count is re-derived per backend (a
 * 25k GPU count is not a CPU count).
 */
async function mountBackend(
  tryGpu: boolean,
  carry?: Snapshot,
  panelOpen = !coarsePointer && window.innerWidth >= 768,
): Promise<void> {
  let application: App | null = null;
  let device: GPUDevice | null = null;
  let label = "webgpu";
  let countMax = 100000;
  let count = 0;

  if (tryGpu) {
    const adapter = await detectWebGPU();
    if (adapter) {
      try {
        const { createGpuApp } = await import("./gpu/GpuSwarm");
        count = carry?.count || countOverride || GPU_COUNT;
        const gpuApp = await createGpuApp(app, adapter, count);
        gpuApp.lost.then(onDeviceLost);
        Object.assign(window as object, { __gpuDevice: gpuApp.device });
        application = gpuApp;
        device = gpuApp.device;
      } catch (err) {
        console.warn("WebGPU init failed, falling back to CPU:", err);
      }
    }
  }

  if (!application) {
    const { createCpuApp } = await import("./cpu/CpuSwarm");
    count = countOverride || CPU_COUNT;
    application = createCpuApp(app, count);
    label = "cpu·three";
    countMax = 8000;
    Object.assign(window as object, { __gpuDevice: undefined });
  }

  const { system } = application;
  if (backendTag) backendTag.textContent = `v4 · ${label}`;

  const reduceMotion =
    !carry && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const applyReducedMotion = (): void => {
    system.params.timeScale = 0.5;
    system.params.spontaneous *= 0.4;
  };
  if (reduceMotion) applyReducedMotion();

  const panel = new DevPanel(system, {
    startOpen: panelOpen,
    countMax,
    countStep: countMax > 10000 ? 500 : 100,
    backendLabel: label,
  });
  // Feel A/B (src/feel.ts) applies any ?feel= selection itself.
  // A plain visit boots into the shipped feel; a dev session that saved
  // slider tweaks (the panel's localStorage) keeps those instead.
  let hasSavedTweaks = false;
  try {
    hasSavedTweaks = !!localStorage.getItem(PANEL_STORAGE_KEY);
  } catch {
    /* storage blocked: treat as a fresh visit */
  }
  const feel = installFeelSwitcher(system, panel, {
    apply: (snap) => system.applySnapshot(snap),
    useDefault: !hasSavedTweaks && !carry,
  });
  // A ?feel= snapshot replaces every param, the slowdown included: re-apply
  // it on top so reduced motion still wins. Only then: without a feel the
  // panel's localStorage restore may hold a saved reduction already, and a
  // second pass would compound it.
  if (reduceMotion && feel.applied) {
    applyReducedMotion();
    panel.refresh();
  }
  if (carry) {
    // Last, after the panel's localStorage restore and the ?feel= apply:
    // on a re-mount the live state (slider tweaks included) wins.
    system.applySnapshot({ ...carry, count });
    panel.refresh();
  }

  live = {
    application,
    system,
    panel,
    feel,
    // Created after every count-changing step above, so its target is the
    // count this mount actually settled on.
    governor: device ? makeGovernor(system, panel, device) : null,
  };
  Object.assign(window as object, { swarm: system });

  applySize();
}

/** The device is gone (driver reset, GPU process crash, or a debugging
 * `__gpuDevice.destroy()`): tear the backend down and re-mount. One more
 * GPU attempt per page, then the CPU fallback. */
async function onDeviceLost(info: GPUDeviceLostInfo): Promise<void> {
  const old = live;
  if (!old) return;
  console.warn(
    `[webgpu] device lost (${info.reason}): ${info.message || "no message"} - re-mounting`,
  );
  live = null; // the loop idles until the new backend is in
  const snapshot = old.system.getSnapshot();
  // Carry what the user wants, not what a struggling device was governed
  // down to; the new mount's governor starts from this.
  if (old.governor) snapshot.count = old.governor.target();
  const panelOpen =
    document.querySelector<HTMLElement>(".blm-panel")?.style.display !== "none";
  old.feel.dispose();
  old.panel.dispose();
  try {
    old.application.dispose(); // removes the canvas
  } catch (err) {
    console.warn("[webgpu] dispose after loss threw:", err);
  }
  const retryGpu = !gpuRetried;
  gpuRetried = true;
  await mountBackend(retryGpu, snapshot, panelOpen);
}

// --- Adaptive count governor (GPU only) ---
// Weaker devices (or a contended GPU) breathe the school's density down
// and back up instead of dropping frames. Resizes happen in place — the
// school keeps swimming; density compensation keeps the perceived mass
// steady — so both directions are invisible.
//
// Frame times alone can't tell an overloaded GPU from a display the OS
// throttled (iOS Low Power Mode, Chrome Battery Saver: 30 Hz). The old
// governor let its frame-period estimate drift up toward slow frames to
// avoid shedding fish under a throttle, and so it also normalized real
// overload: a steady 25-30 fps on a 60 Hz display never stepped down. So
// a step down now needs two signals: frames late against the display, AND
// GPU work that nearly fills a display period. The GPU estimate is the
// time from the frame's last submit until queue.onSubmittedWorkDone()
// resolves (~0.2 ms when idle, 1-2 frames stale). A busy main thread also
// delays that callback, so it can read as GPU load; a step down therefore
// has to lower the estimate, or shedding stops.
//
// One per mount: it closes over that backend's system, panel and device.
interface Governor {
  /** Call right after the frame's last submit. */
  tick: (rawMs: number) => void;
  /** The count the user wants, not the governed-down one. A re-mount
   * carries this, so a fresh device gets a fresh shot at full density. */
  target: () => number;
}

function makeGovernor(
  system: SwarmSystem,
  panel: DevPanel,
  device: GPUDevice,
): Governor | null {
  if (countOverride) return null;
  const FLOOR = 8000;
  let target = system.getCount(); // what the user wants
  let governed = target; // what we've set
  // The display's frame period: the fastest frame seen, but never slower
  // than 60 Hz, so a device that boots already overloaded can't take its
  // slow cadence for the display's. It drifts up only while the GPU is
  // idle (a 120 Hz window moved to a 60 Hz screen), never under load.
  let displayMs = 1000 / 60;
  let gpuMs = 0; // smoothed GPU time per frame (outlier-clamped)
  // The same, capped only at the stall threshold: judges whether a step
  // down helped. gpuMs's clamp (~67 ms at 60 Hz) would hide the drop on a
  // GPU at 100 ms/frame and block the shedding it exists for.
  let gpuLongMs = 0;
  let probing = false;
  // Bumped on a stall so a probe that spans it is discarded on landing.
  let probeEpoch = 0;
  let over = 0; // seconds of late, GPU-bound frames
  let under = 0; // seconds of GPU headroom
  let warmup = 5;
  // Step history for the oscillation guard: after down → up → down within
  // two minutes the lower level sticks for the session, so the count
  // can't visibly breathe up and down.
  const steps: { dir: -1 | 1; at: number }[] = [];
  let sticky = false;
  // Shedding has to pay off: a second after a step down the GPU estimate
  // must have fallen. If it hasn't, the load isn't the school's (a busy
  // main thread delaying the probe, another app on the GPU) and cutting
  // more fish won't help, so step-downs stop until the slider moves.
  let verify: { at: number; before: number } | null = null;
  let shedBlocked = false;

  const probe = (): void => {
    if (probing) return;
    probing = true;
    const epoch = probeEpoch;
    const start = performance.now();
    device.queue
      .onSubmittedWorkDone()
      .then(() => {
        if (epoch !== probeEpoch) return;
        // Clamped so one outlier (a GC pause, a deferred completion) can't
        // flip the estimate on its own.
        const elapsed = performance.now() - start;
        gpuMs += (Math.min(elapsed, 4 * displayMs) - gpuMs) * 0.25;
        gpuLongMs += (Math.min(elapsed, 400) - gpuLongMs) * 0.25;
      })
      .catch(() => {}) // device lost: the mount (and this governor) is replaced
      .finally(() => {
        probing = false;
      });
  };

  const step = (dir: -1 | 1, count: number, why: string): void => {
    governed = count;
    console.info(`[governor] ${why} — density ${dir < 0 ? "down" : "back up"} to ${governed}`);
    system.resize?.(governed);
    panel.refresh();
    over = 0;
    under = 0;
    const now = performance.now();
    if (dir < 0) verify = { at: now + 1000, before: gpuLongMs };
    steps.push({ dir, at: now });
    while (steps.length && now - steps[0].at > 120_000) steps.shift();
    const n = steps.length;
    if (n >= 3 && steps[n - 3].dir < 0 && steps[n - 2].dir > 0 && steps[n - 1].dir < 0) {
      sticky = true;
      console.info(`[governor] oscillating — holding ${governed} for this session`);
    }
  };

  const tick = (rawMs: number): void => {
    // A single long frame is a stall (tab switch, harness run, GC pause,
    // a modal), not sustained overload: one such frame would otherwise
    // count as "2 s over budget" by itself and drop density a notch per
    // hiccup (observed: 25k → 9k over four console harness runs). Only
    // sustained misses accumulate. A hidden tab defers GPU completion, so
    // neither may feed the GPU estimate: a probe spanning it is dropped.
    if (document.hidden || rawMs > 400) {
      over = 0;
      under = 0;
      gpuMs = 0;
      gpuLongMs = 0;
      probeEpoch++;
      // The estimate restarts from zero: judge the last step on fresh samples.
      if (verify) verify.at = performance.now() + 1000;
      return;
    }
    probe();
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
      return;
    }
    if (verify && performance.now() >= verify.at) {
      if (gpuLongMs > 0.9 * verify.before) {
        shedBlocked = true;
        console.info(
          `[governor] fewer fish didn't lower GPU time (~${gpuLongMs.toFixed(1)} ms) — holding ${governed}`,
        );
      }
      verify = null;
    }
    // GPU-bound: the GPU work nearly fills a display period, so fewer fish
    // would let frames land on time. Below that, late frames come from
    // something shedding can't fix (a 30 Hz cap with 12 ms of work).
    const gpuBound = gpuMs > 0.8 * displayMs;
    if (rawMs < displayMs) displayMs = rawMs;
    else if (!gpuBound) displayMs = Math.min(1000 / 60, displayMs + (rawMs - displayMs) * 0.005);
    const dtS = rawMs / 1000;
    // Late = over 1.5 display periods and under 45 fps, so 60 fps on a
    // 120 Hz display is never late.
    const late = rawMs > Math.max(displayMs * 1.5, 1000 / 45);
    if (late && gpuBound) {
      over += dtS;
      under = 0;
    } else {
      over = Math.max(0, over - dtS);
      // Headroom: the GPU would fit this frame's work in half a display
      // period, whatever the cadence (a throttled display still has room).
      under = gpuMs < 0.5 * displayMs ? under + dtS : 0;
    }
    if (over > 2 && governed > FLOOR && !verify && !shedBlocked) {
      step(
        -1,
        Math.max(FLOOR, Math.round((governed * 0.78) / 500) * 500),
        `late frames (~${rawMs.toFixed(0)} ms, GPU ~${gpuMs.toFixed(1)} ms)`,
      );
    } else if (under > 20 && governed < target && !sticky) {
      step(
        1,
        Math.min(target, Math.round((governed * 1.15) / 500) * 500),
        `headroom (GPU ~${gpuMs.toFixed(1)} ms)`,
      );
    }
  };
  return {
    tick,
    // A slider move since the last tick is the new target already.
    target: () => (system.getCount() !== governed ? system.getCount() : target),
  };
}

// --- Pointer --- (always talks to whichever backend is live)
function toNDC(e: PointerEvent): { x: number; y: number } {
  return {
    x: (e.clientX / window.innerWidth) * 2 - 1,
    y: -(e.clientY / window.innerHeight) * 2 + 1,
  };
}

// While a probe owns the sim, a stray mouse move would re-park the
// predator mid-measurement.
window.addEventListener("pointermove", (e) => {
  if (pauseDepth > 0) return;
  live?.system.updateMousePosition(toNDC(e));
});

window.addEventListener("pointerdown", (e) => {
  if (pauseDepth > 0) return;
  // Clicks on the panel shouldn't detonate the swarm.
  if ((e.target as HTMLElement).closest?.(".blm-panel, .blm-toggle, #mail")) return;
  live?.system.updateMousePosition(toNDC(e));
  live?.system.burst();
});

document.documentElement.addEventListener("mouseleave", () => {
  if (pauseDepth > 0) return;
  live?.system.clearPointer();
});
window.addEventListener("blur", () => {
  if (pauseDepth > 0) return;
  live?.system.clearPointer();
});

// --- Resize ---
function applySize(): void {
  if (!live || window.innerWidth < 1 || window.innerHeight < 1) return;
  live.application.resize(
    window.innerWidth,
    window.innerHeight,
    window.devicePixelRatio,
  );
}
window.addEventListener("resize", applySize);
// A tab can boot with zero layout (hidden tab, display:none embed) and
// never receive a window resize — watch the container for its first real
// size instead of trusting boot-time innerWidth.
new ResizeObserver(applySize).observe(app);

// --- Loop ---
let last = performance.now();
let lastRaf = 0;
let frames = 0;
let fpsTime = 0;

function body(now: number): void {
  const rawMs = now - last;
  const dt = Math.min(rawMs / 1000, 0.05);
  last = now;
  if (pauseDepth > 0 || !live) return;

  live.system.update(dt);
  live.application.render();
  live.governor?.tick(rawMs);

  frames++;
  fpsTime += dt;
  if (fpsTime >= 0.5) {
    live.panel.setFps(frames / fpsTime);
    frames = 0;
    fpsTime = 0;
  }
}

function frame(now: number): void {
  lastRaf = now;
  requestAnimationFrame(frame);
  body(now);
}

// Handy for poking at the swarm from the console. `harness()` runs the
// headless behavior probes from src/harness.ts and returns the report —
// the same acceptance metrics as the CPU reference. It pauses the render
// loop while running (the GPU harness is async, unlike v3's). `swarm` is
// re-pointed on every mount; `__render` and `harness` resolve the live
// backend at call time.
Object.assign(window as object, {
  __render: () => live?.application.render(),
  harness: async (opts?: object) => {
    if (!live) throw new Error("no live backend (re-mounting?)");
    const { runHarness } = await import("./harness");
    pauseDepth++;
    try {
      return await runHarness({ system: live.system, ...opts });
    } finally {
      pauseDepth--;
    }
  },
  // Shape/feel probes on the *live* school (no rebuild): tweak sliders,
  // then `await feel()` for elongation, edge, density structure, breathing,
  // coherence and the refill-behind-the-cursor time. { seconds: 60 } etc.
  feel: async (opts?: object) => {
    if (!live) throw new Error("no live backend (re-mounting?)");
    const { runFeelProbes, feelSummary } = await import("./harness");
    pauseDepth++;
    try {
      const report = await runFeelProbes({ system: live.system, ...opts });
      console.info(feelSummary(report));
      return report;
    } finally {
      pauseDepth--;
    }
  },
  // Filmstrip review (src/review.ts): N frames of the *running* loop in one
  // JPEG plus glow stats, optionally with a scripted pointer. `__sweep()` is
  // the standard predator pass.
  __filmstrip: async (opts?: FilmstripOptions) =>
    (await import("./review")).runFilmstrip(opts),
  __sweep: async (opts?: FilmstripOptions) =>
    (await import("./review")).runSweep(opts),
});

// The address in the corner copies itself; "copied" shows for 3 s.
const mail = document.getElementById("mail");
if (mail) {
  const address = "hi@olle.work";
  let timer = 0;
  mail.addEventListener("click", async (e) => {
    e.stopPropagation();
    try {
      await navigator.clipboard.writeText(address);
    } catch {
      /* clipboard blocked: the address is still readable */
    }
    mail.textContent = "copied";
    mail.classList.add("copied");
    window.clearTimeout(timer);
    timer = window.setTimeout(() => {
      mail.textContent = address;
      mail.classList.remove("copied");
    }, 3000);
  });
}

mountBackend(true)
  .then(() => {
    requestAnimationFrame(frame);
    // Headless/background contexts suspend rAF; keep the creature living on
    // a timer that yields the moment rAF ticks again.
    setInterval(() => {
      const now = performance.now();
      if (now - lastRaf > 250) body(now);
    }, 1000 / 60);
  })
  .catch((err) => {
    console.error("bioluminescent boot failed:", err);
    if (backendTag) backendTag.textContent = "v4 · boot failed";
  });
