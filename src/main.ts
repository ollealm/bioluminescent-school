import { DevPanel } from "./DevPanel";
import type { SwarmSystem } from "./system";

/**
 * Bootstrap: pick a backend, wire pointer/resize/loop around it.
 *
 * Backend ladder (HANDOFF-GPU.md §5 "degrading gracefully"):
 *   1. WebGPU  → the 50k compute rewrite (src/gpu, no three.js).
 *   2. no WebGPU → the untouched CPU v3 reference at its original counts
 *      (src/cpu, lazy chunk that carries three.js).
 *
 * URL overrides for side-by-side comparison during tuning:
 *   ?backend=cpu|gpu   force a backend
 *   ?count=NNN         initial particle count
 */

interface App {
  system: SwarmSystem;
  render: () => void;
  resize: (width: number, height: number, dpr: number) => void;
  dispose: () => void;
}

const app = document.getElementById("app")!;
const backendTag = document.getElementById("backend-tag");

const query = new URLSearchParams(location.search);
const forced = query.get("backend");

const isMobile =
  window.matchMedia("(pointer: coarse)").matches || window.innerWidth < 768;

async function detectWebGPU(): Promise<GPUAdapter | null> {
  if (forced === "cpu") return null;
  if (!("gpu" in navigator)) return null;
  try {
    return await navigator.gpu.requestAdapter({
      powerPreference: "high-performance",
    });
  } catch {
    return null;
  }
}

async function boot(): Promise<void> {
  const adapter = await detectWebGPU();
  const countOverride = Number(query.get("count")) || 0;

  let application: App;
  let label: string;
  let countMax: number;

  let gpuApp: App | null = null;
  if (adapter) {
    try {
      const { createGpuApp } = await import("./gpu/GpuSwarm");
      const count = countOverride || (isMobile ? 16000 : 50000);
      gpuApp = await createGpuApp(app, adapter, count);
    } catch (err) {
      console.warn("WebGPU init failed, falling back to CPU:", err);
    }
  }

  if (gpuApp) {
    application = gpuApp;
    label = "webgpu";
    countMax = 100000;
  } else {
    const { createCpuApp } = await import("./cpu/CpuSwarm");
    const count = countOverride || (isMobile ? 1400 : 2600);
    application = createCpuApp(app, count);
    label = "cpu·three";
    countMax = 6000;
  }

  const { system } = application;
  if (backendTag) backendTag.textContent = `v4 · ${label}`;

  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    system.params.timeScale = 0.5;
    system.params.spontaneous *= 0.4;
  }

  const panel = new DevPanel(system, {
    startOpen: !isMobile,
    countMax,
    countStep: countMax > 10000 ? 500 : 100,
    backendLabel: label,
  });

  // --- Pointer ---
  function toNDC(e: PointerEvent): { x: number; y: number } {
    return {
      x: (e.clientX / window.innerWidth) * 2 - 1,
      y: -(e.clientY / window.innerHeight) * 2 + 1,
    };
  }

  window.addEventListener("pointermove", (e) => {
    system.updateMousePosition(toNDC(e));
  });

  window.addEventListener("pointerdown", (e) => {
    // Clicks on the panel shouldn't detonate the swarm.
    if ((e.target as HTMLElement).closest?.(".blm-panel, .blm-toggle")) return;
    system.updateMousePosition(toNDC(e));
    system.burst();
  });

  document.documentElement.addEventListener("mouseleave", () =>
    system.clearPointer(),
  );
  window.addEventListener("blur", () => system.clearPointer());

  // --- Resize ---
  window.addEventListener("resize", () => {
    application.resize(
      window.innerWidth,
      window.innerHeight,
      window.devicePixelRatio,
    );
  });

  // --- Loop ---
  let last = performance.now();
  let lastRaf = 0;
  let frames = 0;
  let fpsTime = 0;
  let paused = false; // the async harness steps the sim itself

  // --- Adaptive count governor (GPU only) ---
  // Weaker WebGPU devices step the school down instead of dropping frames:
  // a sustained >17.5 ms frame time sheds ~22% of the count (floor 8k).
  // One-way ratchet — stepping back up would visibly reset the school for
  // no benefit — and it stands down the moment the user takes the slider.
  const governor = (() => {
    if (label !== "webgpu" || countOverride) return null;
    const FLOOR = 8000;
    let governed = system.getCount();
    let ema = 12;
    let overBudget = 0;
    let warmup = 5;
    return (rawMs: number) => {
      if (warmup > 0) {
        warmup -= rawMs / 1000;
        return;
      }
      if (system.getCount() !== governed) return; // user took control
      ema += (Math.min(rawMs, 100) - ema) * 0.05;
      if (ema > 17.5) overBudget += rawMs / 1000;
      else overBudget = Math.max(0, overBudget - rawMs / 1000);
      if (overBudget > 2 && governed > FLOOR) {
        governed = Math.max(FLOOR, Math.round((governed * 0.78) / 500) * 500);
        console.info(
          `[governor] ${ema.toFixed(1)} ms/frame — school steps down to ${governed}`,
        );
        system.rebuild(governed);
        panel.refresh();
        ema = 12;
        overBudget = 0;
      }
    };
  })();

  function body(now: number): void {
    const rawMs = now - last;
    const dt = Math.min(rawMs / 1000, 0.05);
    last = now;
    if (paused) return;

    system.update(dt);
    application.render();
    governor?.(rawMs);

    frames++;
    fpsTime += dt;
    if (fpsTime >= 0.5) {
      panel.setFps(frames / fpsTime);
      frames = 0;
      fpsTime = 0;
    }
  }

  function frame(now: number): void {
    lastRaf = now;
    requestAnimationFrame(frame);
    body(now);
  }
  requestAnimationFrame(frame);

  // Headless/background contexts suspend rAF; keep the creature living on a
  // timer that yields the moment rAF ticks again.
  setInterval(() => {
    const now = performance.now();
    if (now - lastRaf > 250) body(now);
  }, 1000 / 60);

  // Handy for poking at the swarm from the console. `harness()` runs the
  // headless behavior probes from src/harness.ts and returns the report —
  // the same acceptance metrics as the CPU reference. It pauses the render
  // loop while running (the GPU harness is async, unlike v3's).
  Object.assign(window as object, {
    swarm: system,
    __render: () => application.render(),
    harness: async (opts?: object) => {
      const { runHarness } = await import("./harness");
      paused = true;
      try {
        return await runHarness({ system, ...opts });
      } finally {
        paused = false;
      }
    },
  });
}

boot().catch((err) => {
  console.error("bioluminescent boot failed:", err);
  if (backendTag) backendTag.textContent = "v4 · boot failed";
});
