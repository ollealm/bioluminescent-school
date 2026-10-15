import type { GovernorStatus } from "./governor";

/**
 * ?diag: a small read-only overlay for testing on real devices (older
 * phones, laptops without WebGPU) where the console is out of reach. It
 * shows which backend booted and why, the frame rate and how many frames
 * came in late (an average fps hides stutter: 50 fps can be every fifth
 * frame doubled), the particle count against the count the governor is
 * defending, and what the governor sees.
 *
 * Styled after the olle.work document (thin border in the UI's main color,
 * transparent, the page's own font) and reads the site's --ui-main /
 * --ui-size when they exist, so it sits in the same family on the site and
 * the demo page. Never takes pointer input: it can't eat a hunt or a click.
 */

const CSS = `
.blm-diag {
  position: fixed; left: 24px; bottom: 24px; z-index: 6;
  padding: 12px 16px 13px;
  border: 1px solid var(--ui-main, rgb(228, 255, 240));
  background: transparent;
  color: var(--ui-main, rgb(228, 255, 240));
  font: inherit; font-size: var(--ui-size, 13px); line-height: 1.75; letter-spacing: 0.04em;
  font-variant-numeric: tabular-nums;
  pointer-events: none; user-select: none;
  white-space: nowrap;
}
.blm-diag dl { display: grid; grid-template-columns: auto auto; column-gap: 1.4em; }
.blm-diag dt { opacity: 0.55; }
.blm-diag dd { margin: 0; }
@media (max-width: 640px) { .blm-diag { left: 16px; bottom: 16px; } }
`;

export interface Diagnostics {
  /** The backend that booted ("webgpu", "cpu · three") and a detail line:
   * the adapter for WebGPU, the reason for a CPU fallback. */
  setBackend: (backend: string, detail: string, gpu: boolean) => void;
  /** Frame rate, share of frames late, and the worst frame (ms), over
   * the last half second. */
  setFrames: (fps: number, lateShare: number, worstMs: number) => void;
  /** Live count, and the governor's target (null: not governed). */
  setCount: (count: number, target: number | null) => void;
  /** Drawing-buffer size and the device pixel ratio it renders at. */
  setResolution: (width: number, height: number, dpr: number) => void;
  /** The governor's view (null: ?governOff). */
  setGovernor: (status: GovernorStatus | null) => void;
}

export function installDiagnostics(): Diagnostics {
  const style = document.createElement("style");
  style.textContent = CSS;
  document.head.appendChild(style);

  const root = document.createElement("div");
  root.className = "blm-diag";
  root.setAttribute("aria-hidden", "true");
  const list = document.createElement("dl");
  root.appendChild(list);
  const row = (label: string): [HTMLElement, HTMLElement] => {
    const dt = document.createElement("dt");
    dt.textContent = label;
    const dd = document.createElement("dd");
    dd.textContent = "—";
    list.append(dt, dd);
    return [dt, dd];
  };
  const [, backendEl] = row("backend");
  const [detailLabel, detailEl] = row("adapter");
  const [, fpsEl] = row("fps");
  const [, lateEl] = row("late");
  const [, countEl] = row("particles");
  const [, resolutionEl] = row("pixels");
  const [meterLabel, meterEl] = row("gpu");
  const [, governorEl] = row("governor");
  document.body.appendChild(root);

  const fmt = (n: number): string => n.toLocaleString("en-US");

  return {
    setBackend: (backend, detail, gpu) => {
      backendEl.textContent = backend;
      detailLabel.textContent = gpu ? "adapter" : "fallback";
      detailEl.textContent = detail || "—";
    },
    setFrames: (fps, lateShare, worstMs) => {
      fpsEl.textContent = String(Math.round(fps));
      lateEl.textContent = `${Math.round(lateShare * 100)}% · worst ${Math.round(worstMs)} ms`;
    },
    setCount: (count, target) => {
      countEl.textContent =
        target === null
          ? `${fmt(count)} (fixed)`
          : count < target
            ? `${fmt(count)} of ${fmt(target)}`
            : fmt(count);
    },
    setResolution: (width, height, dpr) => {
      resolutionEl.textContent = `${width}×${height} · ${dpr}x`;
    },
    setGovernor: (status) => {
      if (!status) {
        governorEl.textContent = "off";
        return;
      }
      meterLabel.textContent = status.label.toLowerCase();
      meterEl.textContent = `${status.loadMs.toFixed(1)} of ${status.budgetMs.toFixed(1)} ms`;
      governorEl.textContent = status.state;
    },
  };
}
