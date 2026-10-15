import { DevPanel } from "./DevPanel";
import { installDiagnostics, type Diagnostics } from "./diagnostics";
import { installFeelSwitcher, type FeelSwitcher } from "./feel";
import { cpuSpec, FramePacer, gpuSpec, makeGovernor, type Governor, type GovernorSpec } from "./governor";
import type { FilmstripOptions } from "./review";
import type { Snapshot, SwarmSystem } from "./system";

/**
 * The school as a library: `mountSchool()` picks a backend and wires
 * pointer/resize/loop around it. The demo page (src/main.ts) is one call;
 * a site embedding the school calls it the same way and adds its own UI.
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
 *   ?count=NNN         initial particle count (still governed)
 *   ?governOff         turn the adaptive count governor off (pin the count)
 *   ?diag              diagnostics overlay: backend, adapter, fps, count
 */

interface App {
  system: SwarmSystem;
  render: () => void;
  resize: (width: number, height: number, dpr: number) => void;
  dispose: () => void;
}

export interface SchoolOptions {
  /** Element the canvas mounts into, sized to the window. Default `#app`. */
  container?: HTMLElement;
  /** Element that shows the backend label ("v4 · webgpu"), if any. Default `#backend-tag`. */
  backendTag?: HTMLElement | null;
  /** Selector for the page's own UI: a pointerdown inside it never bursts
   * the school. The dev panel is always excluded. */
  ignore?: string;
}

/** Everything that belongs to one backend instance and is rebuilt on re-mount. */
interface Mounted {
  application: App;
  system: SwarmSystem;
  panel: DevPanel;
  feel: FeelSwitcher;
  governor: Governor | null;
}

let app: HTMLElement;
let backendTag: HTMLElement | null = null;
let ignoreSelector = ".blm-panel, .blm-toggle";

const query = new URLSearchParams(location.search);
const forced = query.get("backend");
const countOverride = Number(query.get("count")) || 0;
// Flags are bare: ?diag, ?governOff, ?alive.
const governOff = query.has("governOff");
const diagnostics: Diagnostics | null =
  query.has("diag") ? installDiagnostics() : null;
// Why the last boot didn't get WebGPU, for the diagnostics overlay.
let cpuReason = "";
// ?alive (or window.__keepAlive, which the filmstrip sets while it runs)
// keeps the sim stepping in a hidden tab whose rAF is suspended: test and
// agent panes report hidden but still need a live school.
const keepAliveParam = query.has("alive");

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
  if (forced === "cpu") {
    cpuReason = "forced by ?backend=cpu";
    return null;
  }
  if (!("gpu" in navigator)) {
    cpuReason = "no WebGPU in this browser";
    return null;
  }
  try {
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: "high-performance",
    });
    if (!adapter) cpuReason = "no WebGPU adapter";
    if (adapter && forced !== "gpu" && (await isFallbackAdapter(adapter))) {
      console.info(
        "[boot] WebGPU adapter is a software fallback - using the CPU backend (?backend=gpu to force it)",
      );
      cpuReason = "software WebGPU adapter";
      return null;
    }
    return adapter;
  } catch {
    cpuReason = "WebGPU adapter request failed";
    return null;
  }
}

/** The adapter as one short line ("apple · metal-3"), for diagnostics.
 * `adapter.info` is current; older Chromes only had requestAdapterInfo(). */
async function describeAdapter(adapter: GPUAdapter): Promise<string> {
  const a = adapter as GPUAdapter & {
    requestAdapterInfo?: () => Promise<GPUAdapterInfo>;
  };
  let info: GPUAdapterInfo | undefined = a.info;
  if (!info) {
    try {
      info = await a.requestAdapterInfo?.();
    } catch {
      /* no info: the row says so */
    }
  }
  if (!info) return "unknown adapter";
  return [info.vendor, info.architecture || info.description].filter(Boolean).join(" · ") || "unknown adapter";
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
  // Closed on every device: the school is the page; D opens the panel.
  panelOpen = false,
): Promise<void> {
  let application: App | null = null;
  let governorSpec: GovernorSpec | null = null;
  let label = "webgpu";
  let countMax = 100000;
  let count = 0;
  let detail = "";

  if (!tryGpu) cpuReason = "GPU device lost twice";
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
        governorSpec = gpuSpec(gpuApp.device);
        if (diagnostics) detail = await describeAdapter(adapter);
      } catch (err) {
        console.warn("WebGPU init failed, falling back to CPU:", err);
        cpuReason = "WebGPU init failed";
      }
    }
  }

  if (!application) {
    const { createCpuApp } = await import("./cpu/CpuSwarm");
    count = countOverride || CPU_COUNT;
    countMax = 8000;
    // Allocated to the slider's range, so neither the slider nor the
    // governor ever respawns the school.
    const cpuApp = createCpuApp(app, count, countMax);
    application = cpuApp;
    governorSpec = cpuSpec(cpuApp);
    label = "cpu·three";
    detail = cpuReason;
    Object.assign(window as object, { __gpuDevice: undefined });
  }

  const { system } = application;
  if (backendTag) backendTag.textContent = `v4 · ${label}`;
  const gpu = label === "webgpu";
  diagnostics?.setBackend(gpu ? "gpu · webgpu" : "cpu · three.js", detail, gpu);

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
  // Feel A/B (src/feel.ts) applies any ?feel= selection itself, and a
  // plain visit always boots into the shipped feel. Only a re-mount after
  // a lost device skips it: that carries the live state instead.
  const feel = installFeelSwitcher(system, panel, {
    apply: (snap) => system.applySnapshot(snap),
    useDefault: !carry,
  });
  // A feel snapshot replaces every param, the slowdown included: re-apply
  // it on top so reduced motion still wins.
  if (reduceMotion && feel.applied) {
    applyReducedMotion();
    panel.refresh();
  }
  if (carry) {
    // Last, after any ?feel= apply: on a re-mount the live state (slider
    // tweaks included) wins.
    system.applySnapshot({ ...carry, count });
    panel.refresh();
  }

  live = {
    application,
    system,
    panel,
    feel,
    // Created after every count-changing step above, so its target is the
    // count this mount actually settled on (a ?count= is its target).
    // ?governOff pins the count instead.
    governor:
      governOff || !governorSpec ? null : makeGovernor(system, panel, governorSpec),
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
  if ((e.target as HTMLElement).closest?.(ignoreSelector)) return;
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

// --- Loop ---
let last = performance.now();
let lastRaf = 0;
let frames = 0;
let fpsTime = 0;
// For the diagnostics: a frame over 1.5 display periods missed a vsync.
const pacer = new FramePacer();
let worstMs = 0;

function body(now: number): void {
  const rawMs = now - last;
  const dt = Math.min(rawMs / 1000, 0.05);
  last = now;
  if (pauseDepth > 0 || !live) return;

  const t0 = performance.now();
  live.system.update(dt);
  const updateMs = performance.now() - t0;
  live.application.render();
  live.governor?.tick(rawMs, updateMs);

  frames++;
  if (diagnostics) {
    pacer.push(now, rawMs);
    worstMs = Math.max(worstMs, rawMs);
  }
  // Wall time, not the clamped sim dt: under 20 fps the clamp read as 20.
  fpsTime += rawMs / 1000;
  if (fpsTime >= 0.5) {
    live.panel.setFps(frames / fpsTime);
    if (diagnostics) {
      const late = pacer.lateSince(now - fpsTime * 1000, pacer.periodMs * 1.5);
      diagnostics.setFrames(frames / fpsTime, late.share, worstMs);
      diagnostics.setCount(live.system.getCount(), live.governor?.target() ?? null);
      diagnostics.setGovernor(live.governor?.status() ?? null);
    }
    frames = 0;
    fpsTime = 0;
    worstMs = 0;
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

let mounted = false;

/** Boot the school. Call once per page. */
export function mountSchool(options: SchoolOptions = {}): Promise<void> {
  if (mounted) throw new Error("mountSchool: the school is already mounted");
  mounted = true;
  const container = options.container ?? document.getElementById("app");
  if (!container) throw new Error("mountSchool: no container (pass one, or add #app)");
  app = container;
  backendTag =
    options.backendTag !== undefined ? options.backendTag : document.getElementById("backend-tag");
  if (options.ignore) ignoreSelector += `, ${options.ignore}`;
  // A tab can boot with zero layout (hidden tab, display:none embed) and
  // never receive a window resize — watch the container for its first real
  // size instead of trusting boot-time innerWidth.
  new ResizeObserver(applySize).observe(app);

  return mountBackend(true)
    .then(() => {
      requestAnimationFrame(frame);
      // Headless contexts suspend rAF; keep the creature living on a timer
      // that yields the moment rAF ticks again. Not in a hidden tab (a
      // background tab nobody sees should cost nothing) unless kept alive.
      setInterval(() => {
        const now = performance.now();
        const keepAlive =
          keepAliveParam || (window as { __keepAlive?: boolean }).__keepAlive === true;
        if (now - lastRaf > 250 && (!document.hidden || keepAlive)) body(now);
      }, 1000 / 60);
    })
    .catch((err) => {
      console.error("bioluminescent boot failed:", err);
      if (backendTag) backendTag.textContent = "v4 · boot failed";
    });
}
