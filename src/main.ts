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
  const applySize = () => {
    if (window.innerWidth < 1 || window.innerHeight < 1) return;
    application.resize(
      window.innerWidth,
      window.innerHeight,
      window.devicePixelRatio,
    );
  };
  window.addEventListener("resize", applySize);
  // A tab can boot with zero layout (hidden tab, display:none embed) and
  // never receive a window resize — watch the container for its first real
  // size instead of trusting boot-time innerWidth.
  new ResizeObserver(applySize).observe(app);
  applySize();

  // --- Loop ---
  let last = performance.now();
  let lastRaf = 0;
  let frames = 0;
  let fpsTime = 0;
  let paused = false; // the async harness steps the sim itself

  // --- Adaptive count governor (GPU only) ---
  // Weaker devices (or a contended GPU) breathe the school's density down
  // and back up instead of dropping frames. Resizes happen in place — the
  // school keeps swimming; density compensation keeps the perceived mass
  // steady — so both directions are invisible.
  //
  // "Over budget" is measured against the display's own observed frame
  // period (rolling min), not an absolute number: on a 60 Hz display the
  // normal cadence is 16.7 ms and any absolute ~17 ms trigger is a hair
  // trigger that fires on ambient jitter (the bug that kept resetting the
  // school). A frame only counts as bad when it clearly misses vsync.
  const governor = (() => {
    if (label !== "webgpu" || countOverride) return null;
    const FLOOR = 8000;
    let target = system.getCount(); // what the user wants
    let governed = target; // what we've set
    let minMs = 50; // rolling estimate of the display's frame period
    let over = 0; // seconds of clearly-missed frames
    let under = 0; // seconds of comfortable headroom
    let warmup = 5;
    return (rawMs: number) => {
      if (document.hidden) {
        over = 0;
        under = 0;
        return;
      }
      if (warmup > 0) {
        warmup -= rawMs / 1000;
        minMs = Math.min(minMs, rawMs);
        return;
      }
      const current = system.getCount();
      if (current !== governed) {
        // The user took the slider — that's the new target.
        target = current;
        governed = current;
        over = 0;
        under = 0;
        return;
      }
      minMs = Math.min(minMs, rawMs);
      minMs += (rawMs - minMs) * 0.005; // relax upward slowly
      const dtS = rawMs / 1000;
      // "Bad" = worse than ~half the display's best rate (with margin so a
      // ProMotion 120→60 Hz downshift never reads as overload).
      if (rawMs > minMs * 1.8 + 3) {
        over += dtS;
        under = 0;
      } else if (rawMs < minMs * 1.25 + 0.5) {
        under += dtS;
        over = Math.max(0, over - dtS);
      } else {
        over = Math.max(0, over - dtS);
        under = 0;
      }
      if (over > 2 && governed > FLOOR) {
        governed = Math.max(FLOOR, Math.round((governed * 0.78) / 500) * 500);
        console.info(
          `[governor] missing vsync (~${rawMs.toFixed(0)} ms vs ${minMs.toFixed(1)} ms base) — density down to ${governed}`,
        );
        system.resize?.(governed);
        panel.refresh();
        over = 0;
        under = 0;
      } else if (under > 20 && governed < target) {
        governed = Math.min(
          target,
          Math.round((governed * 1.15) / 500) * 500,
        );
        console.info(`[governor] headroom — density back up to ${governed}`);
        system.resize?.(governed);
        panel.refresh();
        over = 0;
        under = 0;
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
