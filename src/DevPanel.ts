import type { SwarmSystem } from "./system";
import { DEFAULT_PARAMS, PALETTES, STYLES, type PaletteName, type StyleName, type SwarmParams } from "./params";
import { DEFAULT_PRESETS, PRESET_REGISTRY, type PresetCategory } from "./presets";
import { urlFeel } from "./feel";

const CATEGORY_LABELS: Record<PresetCategory, string> = {
  motion: "Movement",
  reaction: "Reactions",
};

interface SliderDef {
  key: keyof SwarmParams;
  label: string;
  min: number;
  max: number;
  step: number;
  /** One-line description shown under the slider (toggleable). */
  hint: string;
  /** Importance for the overall feel: 3 = defining, 2 = shaping, 1 = fine-tuning. */
  impact: 1 | 2 | 3;
}

interface Group {
  title: string;
  /** A few sentences explaining what this group of controls governs. */
  blurb: string;
  open?: boolean;
  sliders: SliderDef[];
}

const GROUPS: Group[] = [
  {
    title: "Swarm",
    blurb:
      "The flocking substrate — how each fish relates to its neighbors. Cohesion, alignment and separation set the school's texture; perception is how far 'local' reaches. Max speed and drag set the overall energy level, and nearly every other system acts through these forces.",
    open: true,
    sliders: [
      { key: "maxSpeed", label: "Max speed", min: 0.4, max: 8, step: 0.1, impact: 3, hint: "Top swimming speed — the single biggest energy knob" },
      { key: "alignment", label: "Alignment", min: 0, max: 3, step: 0.01, impact: 3, hint: "Match neighbors' direction — how much they move as one" },
      { key: "cohesion", label: "Cohesion", min: 0, max: 3, step: 0.01, impact: 2, hint: "Pull toward neighbors — how tightly the group holds together" },
      { key: "separation", label: "Separation", min: 0, max: 4, step: 0.01, impact: 2, hint: "Personal space — resistance to crowding" },
      { key: "perception", label: "Perception", min: 1.2, max: 4, step: 0.05, impact: 2, hint: "How far each particle sees its neighbors" },
      { key: "drag", label: "Drag", min: 0, max: 2, step: 0.01, impact: 2, hint: "Water resistance — how quickly motion coasts off" },
    ],
  },
  {
    title: "School behavior",
    blurb:
      "The school's collective mind. On its own schedule it cruises, turns as one, mills into a vortex, rests, and occasionally startles at nothing — this group is why things happen when you aren't touching it. Drive is how strongly the shared will is enforced; tempo is how often it changes.",
    open: true,
    sliders: [
      { key: "schoolDrive", label: "School drive", min: 0, max: 3, step: 0.05, impact: 3, hint: "How hard everyone follows the school's shared heading" },
      { key: "tempo", label: "Tempo", min: 0.1, max: 2.5, step: 0.05, impact: 3, hint: "How often the school changes its mind (cruise / turn / mill / hover)" },
      { key: "startle", label: "Startles", min: 0, max: 2, step: 0.05, impact: 2, hint: "Rate of spontaneous false alarms sweeping the school" },
      { key: "gait", label: "Gait", min: 0, max: 2, step: 0.05, impact: 2, hint: "Burst-and-glide swimming — thrust impulses, then coasting" },
      { key: "collapse", label: "Collapse", min: 0, max: 2, step: 0.05, impact: 3, hint: "How often and how hard the school contracts in on itself, then releases" },
      { key: "shape", label: "Body shape", min: 0, max: 1, step: 0.01, impact: 2, hint: "Stretch the school along its heading - a traveling shoal instead of a ball" },
    ],
  },
  {
    title: "Currents",
    blurb:
      "The water and the stage. Ambient flow, swirl and jitter keep individuals moving; breathing and cloud size shape the resting school; roam lets its home region travel the frame. Time scale is the master speed knob for everything at once.",
    sliders: [
      { key: "timeScale", label: "Time scale", min: 0.1, max: 2, step: 0.05, impact: 3, hint: "Global simulation speed — slows or hurries everything" },
      { key: "flowStrength", label: "Flow strength", min: 0, max: 4, step: 0.05, impact: 2, hint: "Force of the ambient water current" },
      { key: "roam", label: "Roam", min: 0, max: 1.5, step: 0.05, impact: 2, hint: "How far the school's home region wanders around the frame" },
      { key: "mill", label: "Mill", min: 0, max: 2, step: 0.05, impact: 2, hint: "Standing vortex — rotation around the school's own center" },
      { key: "containRadius", label: "Cloud size", min: 5, max: 16, step: 0.5, impact: 2, hint: "Resting size of the school" },
      { key: "flowScale", label: "Flow scale", min: 0.02, max: 0.25, step: 0.005, impact: 1, hint: "Eddy size of the current — lower = broader swirls" },
      { key: "swirl", label: "Swirl", min: 0, max: 2, step: 0.02, impact: 1, hint: "Slow global rotation of the water" },
      { key: "jitter", label: "Jitter", min: 0, max: 2, step: 0.02, impact: 1, hint: "Per-particle wobble — nervous small-scale motion" },
      { key: "breathe", label: "Breathing", min: 0, max: 0.2, step: 0.005, impact: 1, hint: "Slow expand/contract of the whole school" },
    ],
  },
  {
    title: "Wanderers",
    blurb:
      "Small groups that break away, explore, and come home — the drift-away-and-reunite behavior. Mostly texture and story; the rate and the pull back home matter most.",
    sliders: [
      { key: "wanderRate", label: "Wander rate", min: 0, max: 1, step: 0.01, impact: 2, hint: "How often a small group breaks away to explore" },
      { key: "returnPull", label: "Return pull", min: 0.2, max: 5, step: 0.05, impact: 2, hint: "Homing strength when they head back to reunite" },
      { key: "groupSize", label: "Group size", min: 1, max: 30, step: 1, impact: 1, hint: "How many leave together" },
      { key: "wanderPull", label: "Wander pull", min: 0, max: 3, step: 0.05, impact: 1, hint: "How determinedly wanderers push away from the school" },
    ],
  },
  {
    title: "Predator (pointer)",
    blurb:
      "How the cursor hunts and how the school survives it. Radius and repel force are the immediate escape; fear makes the response persist, panic spread makes it contagious, and the fountain maneuver emerges from fleeing fish parting around your path. Regroup is the snap-back once the threat passes.",
    open: true,
    sliders: [
      { key: "mouseRadius", label: "Radius", min: 1, max: 12, step: 0.1, impact: 3, hint: "Size of the danger zone around the cursor" },
      { key: "bite", label: "Bite", min: 0, max: 0.9, step: 0.01, impact: 3, hint: "How crisp the danger zone's edge is - 0 = soft gravity-like falloff, high = nothing until close, then full bolt" },
      { key: "flight", label: "Flight", min: 0, max: 1, step: 0.01, impact: 3, hint: "How far fear carries a fish out of the school before it regroups - with panic spread, the hit fish take neighbors with them" },
      { key: "repelForce", label: "Repel force", min: 0, max: 90, step: 1, impact: 3, hint: "Escape steering strength inside the danger zone" },
      { key: "regroup", label: "Regroup drive", min: 0, max: 4, step: 0.05, impact: 3, hint: "Post-threat urge to rush back to the school" },
      { key: "fleeForce", label: "Flee force", min: 0, max: 80, step: 1, impact: 2, hint: "Extra push while frightened and the threat is still close" },
      { key: "fearSpread", label: "Panic spread", min: 0, max: 4, step: 0.05, impact: 2, hint: "How contagious fear is between neighbors" },
      { key: "fearDecay", label: "Fear decay", min: 0.3, max: 4, step: 0.05, impact: 2, hint: "How quickly fear wears off" },
      { key: "splash", label: "Splash", min: 0, max: 3, step: 0.05, impact: 2, hint: "How much cursor speed excites the swarm (light + force)" },
    ],
  },
  {
    title: "Chain reactions",
    blurb:
      "Bioluminescent cascades — bubbles of light and motion expanding (or collapsing) from a shared center. Transmission sets how far a flash chains and pulse power how hard it shoves; decay and refractory shape each wave's lifetime and stop endless echoes.",
    open: true,
    sliders: [
      { key: "transmission", label: "Transmission", min: 0, max: 12, step: 0.1, impact: 3, hint: "How easily a flash ignites its neighbors — cascade reach" },
      { key: "kick", label: "Pulse power", min: 0, max: 4, step: 0.05, impact: 3, hint: "Physical shove of each cascade — the bubble's strength" },
      { key: "spontaneous", label: "Spontaneous", min: 0, max: 4, step: 0.05, impact: 2, hint: "Rate of random flashes when idle" },
      { key: "flashDecay", label: "Flash decay", min: 0.5, max: 6, step: 0.05, impact: 2, hint: "How fast a flash fades out" },
      { key: "refractory", label: "Refractory", min: 0.2, max: 5, step: 0.05, impact: 2, hint: "Cooldown before a particle can flash again" },
      { key: "flashBrightness", label: "Flash glow", min: 0, max: 3, step: 0.05, impact: 2, hint: "How bright a flash glows" },
      { key: "inburst", label: "Inburst odds", min: 0, max: 1, step: 0.01, impact: 1, hint: "Chance a cascade collapses inward instead of bursting out" },
    ],
  },
  {
    title: "Appearance",
    blurb:
      "Pure rendering — no physics. Size and opacity are the additive brightness budget (raise the count, lower these). Motion glow and stretch are what make speed readable as light and streaks.",
    sliders: [
      { key: "particleSize", label: "Size", min: 0.4, max: 4, step: 0.05, impact: 3, hint: "Base particle size" },
      { key: "opacity", label: "Opacity", min: 0.1, max: 1, step: 0.01, impact: 3, hint: "Overall brightness / translucency" },
      { key: "motionGlow", label: "Motion glow", min: 0, max: 2, step: 0.05, impact: 2, hint: "Shear-triggered light — agitated particles glow" },
      { key: "stretch", label: "Stretch", min: 0, max: 3, step: 0.05, impact: 2, hint: "Fast particles render as streaks along their motion" },
      { key: "crowdDim", label: "Crowd dim", min: 0, max: 1, step: 0.01, impact: 2, hint: "Dim fish in dense cores so pile-ups don't blow out into one glowing spot" },
      { key: "twinkle", label: "Twinkle", min: 0, max: 1, step: 0.01, impact: 1, hint: "Slow individual brightness oscillation" },
    ],
  },
];

const CSS = `
.blm-toggle {
  position: fixed; right: 18px; bottom: 18px; z-index: 40;
  width: 42px; height: 42px; border-radius: 50%;
  border: 1px solid rgba(96, 255, 178, 0.25);
  background: rgba(4, 18, 12, 0.72);
  color: #7dffb0; font-size: 17px; line-height: 1;
  cursor: pointer; backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
  transition: transform .15s ease, border-color .15s ease;
}
.blm-toggle:hover { transform: scale(1.08); border-color: rgba(96, 255, 178, 0.6); }
.blm-panel {
  position: fixed; right: 18px; top: 18px; z-index: 41;
  width: 296px; max-height: calc(100vh - 96px);
  overflow-y: auto; overscroll-behavior: contain;
  border-radius: 14px;
  border: 1px solid rgba(96, 255, 178, 0.16);
  background: rgba(3, 14, 10, 0.78);
  backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
  color: #cfeee0;
  font: 12px/1.45 ui-monospace, "SF Mono", SFMono-Regular, Menlo, monospace;
  padding: 14px 14px 12px;
  scrollbar-width: thin; scrollbar-color: rgba(96,255,178,.25) transparent;
}
.blm-panel::-webkit-scrollbar { width: 6px; }
.blm-panel::-webkit-scrollbar-thumb { background: rgba(96,255,178,.2); border-radius: 3px; }
.blm-panel * { box-sizing: border-box; }
.blm-head { display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 10px; }
.blm-title { font-size: 12px; letter-spacing: .08em; color: #7dffb0; text-transform: lowercase; }
.blm-fps { font-size: 11px; opacity: .9; }
.blm-sub { font-size: 10px; letter-spacing: .1em; text-transform: uppercase; color: rgba(140, 230, 190, 0.55); margin: 10px 0 6px; }
.blm-presets { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
.blm-preset {
  border: 1px solid rgba(96, 255, 178, 0.18); border-radius: 8px;
  background: rgba(10, 40, 28, 0.35); color: #bdf5d8;
  font: 11px ui-monospace, Menlo, monospace; letter-spacing: .02em;
  padding: 6px 4px; cursor: pointer; transition: all .12s ease;
}
.blm-preset:hover { border-color: rgba(96, 255, 178, 0.55); }
.blm-preset.active { background: rgba(38, 210, 130, 0.22); border-color: rgba(96, 255, 178, 0.75); color: #eafff3; }
.blm-desc { font-size: 10px; color: rgba(150, 220, 185, 0.5); min-height: 26px; margin: 6px 2px 2px; }
.blm-row { display: grid; grid-template-columns: 86px 1fr 40px; align-items: center; gap: 8px; margin: 5px 0; }
.blm-hint { font-size: 9.5px; line-height: 1.4; color: rgba(150, 220, 185, 0.45); margin: -3px 0 7px 0; }
.blm-imp { font-size: 6px; letter-spacing: 2px; color: rgba(53, 224, 140, 0.8); vertical-align: 1px; margin-right: 5px; }
.blm-group-desc { font-size: 10px; line-height: 1.45; color: rgba(150, 220, 185, 0.55); margin: 2px 0 9px; }
.blm-panel.blm-hide-desc .blm-hint, .blm-panel.blm-hide-desc .blm-group-desc { display: none; }
.blm-head-right { display: flex; align-items: center; gap: 8px; }
.blm-desc-toggle {
  border: 1px solid rgba(96, 255, 178, 0.25); border-radius: 6px;
  background: rgba(10, 40, 28, 0.35); color: #7dffb0;
  font: 9px ui-monospace, Menlo, monospace; letter-spacing: .06em;
  padding: 3px 7px; cursor: pointer; transition: opacity .12s ease;
}
.blm-desc-toggle.off { opacity: .4; }
.blm-row label { color: rgba(190, 240, 215, 0.75); font-size: 11px; white-space: nowrap; }
.blm-row output { text-align: right; font-size: 10px; color: #8be8b8; }
.blm-panel input[type="range"] { width: 100%; accent-color: #35e08c; height: 14px; cursor: ew-resize; background: transparent; }
.blm-panel details { border-top: 1px solid rgba(96, 255, 178, 0.1); padding: 4px 0 6px; }
.blm-panel summary {
  cursor: pointer; list-style: none; padding: 6px 0 4px;
  font-size: 10px; letter-spacing: .1em; text-transform: uppercase; color: rgba(140, 230, 190, 0.6);
}
.blm-panel summary::before { content: "▸ "; color: rgba(96,255,178,.5); }
.blm-panel details[open] summary::before { content: "▾ "; }
.blm-selects { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin: 4px 0 2px; }
.blm-select-wrap { display: flex; flex-direction: column; gap: 3px; }
.blm-select-wrap span { font-size: 9px; letter-spacing: .1em; text-transform: uppercase; color: rgba(140,230,190,.5); }
.blm-panel select {
  width: 100%; padding: 5px 6px; border-radius: 7px;
  background: rgba(10, 40, 28, 0.5); color: #d4ffe9;
  border: 1px solid rgba(96, 255, 178, 0.2);
  font: 11px ui-monospace, Menlo, monospace;
}
.blm-count-note { font-size: 9px; color: rgba(150,220,185,.4); margin-top: 2px; }
.blm-actions { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-top: 10px; }
.blm-action {
  border: 1px solid rgba(96, 255, 178, 0.2); border-radius: 8px;
  background: rgba(10, 40, 28, 0.4); color: #cdf7e2;
  font: 11px ui-monospace, Menlo, monospace;
  padding: 7px 4px; cursor: pointer; transition: all .12s ease;
}
.blm-action:hover { border-color: rgba(96, 255, 178, 0.6); background: rgba(30, 120, 80, 0.3); }
.blm-feels { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
.blm-feel { text-align: left; padding-left: 8px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.blm-feel-key { color: rgba(96, 255, 178, 0.45); margin-right: 6px; }
.blm-feel.active .blm-feel-key { color: #7dffb0; }
.blm-gpu-only { color: rgba(255, 209, 102, 0.6); margin-left: 4px; }
.blm-foot { margin-top: 10px; font-size: 9px; color: rgba(150, 220, 185, 0.35); text-align: center; }
@media (max-width: 640px) {
  .blm-panel { right: 10px; left: 10px; top: 10px; width: auto; max-height: 62vh; }
}
`;

/** localStorage key for the auto-saved slider state (shared by both backends). */
export const PANEL_STORAGE_KEY = "olle-biolume-gpu-v1";

export interface DevPanelOptions {
  storageKey?: string;
  startOpen?: boolean;
  /** Count slider range — the GPU backend raises the ceiling far above v3. */
  countMin?: number;
  countMax?: number;
  countStep?: number;
  /** Short backend label for the footer, e.g. "webgpu" / "cpu·three". */
  backendLabel?: string;
}

/** Self-contained dev control panel. `new DevPanel(system)` and done. */
export class DevPanel {
  private system: SwarmSystem;
  private storageKey: string;
  private root: HTMLDivElement;
  private toggleBtn: HTMLButtonElement;
  private fpsEl!: HTMLSpanElement;
  private descEl!: HTMLDivElement;
  private sliderInputs = new Map<string, { input: HTMLInputElement; out: HTMLOutputElement }>();
  private presetBtns = new Map<string, HTMLButtonElement>();
  private styleSel!: HTMLSelectElement;
  private paletteSel!: HTMLSelectElement;
  private countInput!: HTMLInputElement;
  private countOut!: HTMLOutputElement;
  private rebuildTimer: number | undefined;
  private onKey: (e: KeyboardEvent) => void;
  private opts: Required<Pick<DevPanelOptions, "countMin" | "countMax" | "countStep" | "backendLabel">>;
  private feelWrap!: HTMLDivElement;
  private feelGrid!: HTMLDivElement;
  private feelBtns = new Map<string, HTMLButtonElement>();
  private feelName: string | null = null;
  /** Set by installFeelSwitcher (src/feel.ts); Feel A/B buttons route through it. */
  onFeelSelect: ((name: string) => void) | null = null;
  /** Set by installFeelSwitcher: Reset lands on the shipped feel, not bare v3 defaults. */
  onReset?: () => void;

  constructor(system: SwarmSystem, options: DevPanelOptions = {}) {
    this.system = system;
    this.storageKey = options.storageKey ?? PANEL_STORAGE_KEY;
    this.opts = {
      countMin: options.countMin ?? 500,
      countMax: options.countMax ?? 6000,
      countStep: options.countStep ?? 100,
      backendLabel: options.backendLabel ?? "",
    };

    this.restore();

    if (!document.getElementById("blm-style")) {
      const style = document.createElement("style");
      style.id = "blm-style";
      style.textContent = CSS;
      document.head.appendChild(style);
    }

    this.toggleBtn = document.createElement("button");
    this.toggleBtn.className = "blm-toggle";
    this.toggleBtn.title = "Toggle dev panel (D)";
    this.toggleBtn.textContent = "⚙";
    document.body.appendChild(this.toggleBtn);

    this.root = document.createElement("div");
    this.root.className = "blm-panel";
    this.build();
    document.body.appendChild(this.root);

    const startOpen = options.startOpen ?? window.innerWidth >= 900;
    this.root.style.display = startOpen ? "" : "none";

    this.toggleBtn.addEventListener("click", () => this.toggle());
    this.onKey = (e) => {
      const target = e.target as HTMLElement;
      if (/^(input|select|textarea)$/i.test(target?.tagName ?? "")) return;
      if (e.key === "d" || e.key === "D") this.toggle();
    };
    window.addEventListener("keydown", this.onKey);
  }

  toggle(): void {
    this.root.style.display = this.root.style.display === "none" ? "" : "none";
  }

  setFps(fps: number): void {
    const rounded = Math.round(fps);
    this.fpsEl.textContent = `${rounded} fps`;
    this.fpsEl.style.color =
      rounded >= 55 ? "#7dffb0" : rounded >= 30 ? "#ffd166" : "#ff7b7b";
  }

  /** Populate the Feel A/B row; index i gets hotkey i+1 (first nine). */
  setFeels(names: readonly string[]): void {
    this.feelGrid.replaceChildren();
    this.feelBtns.clear();
    names.forEach((name, i) => {
      const btn = el("button", "blm-preset blm-feel") as HTMLButtonElement;
      if (i < 9) btn.appendChild(el("span", "blm-feel-key", String(i + 1)));
      btn.appendChild(document.createTextNode(name));
      btn.title = i < 9 ? `${name} (key ${i + 1})` : name;
      btn.addEventListener("click", () => this.onFeelSelect?.(name));
      this.feelBtns.set(name, btn);
      this.feelGrid.appendChild(btn);
    });
    this.feelWrap.style.display = names.length ? "" : "none";
    this.setFeelName(this.feelName);
  }

  /** Highlight the active feel (null = none, e.g. after a Reset). */
  setFeelName(name: string | null): void {
    this.feelName = name;
    for (const [n, btn] of this.feelBtns) btn.classList.toggle("active", n === name);
  }

  /** The count slider's range - feels clamp their count to it (CPU stays small). */
  get countRange(): { min: number; max: number } {
    return { min: this.opts.countMin, max: this.opts.countMax };
  }

  dispose(): void {
    // A pending count debounce would resize a system that is being torn down.
    window.clearTimeout(this.rebuildTimer);
    window.removeEventListener("keydown", this.onKey);
    this.root.remove();
    this.toggleBtn.remove();
  }

  // ------------------------------------------------------------------

  private build(): void {
    const sys = this.system;
    const root = this.root;

    // Header — title, description toggle, fps
    const head = el("div", "blm-head");
    head.appendChild(el("div", "blm-title", "swarm controls"));
    const headRight = el("div", "blm-head-right");
    const descToggle = el("button", "blm-desc-toggle", "ⓘ info") as HTMLButtonElement;
    descToggle.title = "Show/hide the descriptions under sliders and groups";
    const descKey = this.storageKey + ":descs";
    const applyDescs = (on: boolean) => {
      root.classList.toggle("blm-hide-desc", !on);
      descToggle.classList.toggle("off", !on);
      try {
        localStorage.setItem(descKey, on ? "1" : "0");
      } catch {
        /* best-effort */
      }
    };
    descToggle.addEventListener("click", () =>
      applyDescs(root.classList.contains("blm-hide-desc")),
    );
    headRight.appendChild(descToggle);
    this.fpsEl = el("span", "blm-fps", "— fps");
    headRight.appendChild(this.fpsEl);
    head.appendChild(headRight);
    root.appendChild(head);
    applyDescs(localStorage.getItem(descKey) !== "0");

    // Feel A/B - whole tuned snapshots, one keypress apart. Empty and
    // hidden until installFeelSwitcher hands over the names (setFeels).
    this.feelWrap = el("div", "") as HTMLDivElement;
    this.feelWrap.style.display = "none";
    this.feelWrap.appendChild(el("div", "blm-sub", "Feel A/B"));
    this.feelGrid = el("div", "blm-feels") as HTMLDivElement;
    this.feelWrap.appendChild(this.feelGrid);
    this.feelWrap.appendChild(
      el("div", "blm-hint", "Full snapshots from snapshots/*.json · keys 1-9 switch · ?feel=name in the URL shares one"),
    );
    root.appendChild(this.feelWrap);

    // Presets — one row of buttons per category, freely combinable.
    for (const cat of Object.keys(PRESET_REGISTRY) as PresetCategory[]) {
      root.appendChild(el("div", "blm-sub", CATEGORY_LABELS[cat]));
      const grid = el("div", "blm-presets");
      for (const [name, preset] of Object.entries(PRESET_REGISTRY[cat])) {
        const btn = el("button", "blm-preset", preset.label) as HTMLButtonElement;
        btn.title = preset.description;
        btn.addEventListener("click", () => {
          sys.applyPreset(cat, name);
          this.descEl.textContent = preset.description;
          this.refresh();
          this.save();
        });
        this.presetBtns.set(`${cat}:${name}`, btn);
        grid.appendChild(btn);
      }
      root.appendChild(grid);
    }
    this.descEl = el("div", "blm-desc", "mix a movement with a reaction · style & palette below") as HTMLDivElement;
    root.appendChild(this.descEl);

    // Style / palette selects
    const selects = el("div", "blm-selects");
    selects.appendChild(
      this.makeSelect("Style", STYLES.map((s) => [s, s]), (v) => {
        sys.setStyle(v as StyleName);
        this.save();
      }, (sel) => (this.styleSel = sel), "Sprite shape: soft glow, hard orb, star spark, or blinking firefly"),
    );
    selects.appendChild(
      this.makeSelect(
        "Palette",
        Object.entries(PALETTES).map(([k, p]) => [k, p.label]),
        (v) => {
          sys.setPalette(v as PaletteName);
          this.save();
        },
        (sel) => (this.paletteSel = sel),
        "Color scheme — also sets the flash hue",
      ),
    );
    root.appendChild(selects);

    // Particle count (debounced rebuild)
    const countRow = el("div", "blm-row");
    countRow.appendChild(el("label", "", "Particles"));
    this.countInput = document.createElement("input");
    this.countInput.type = "range";
    this.countInput.min = String(this.opts.countMin);
    this.countInput.max = String(this.opts.countMax);
    this.countInput.step = String(this.opts.countStep);
    this.countInput.value = String(sys.getCount());
    this.countOut = el("output", "", String(sys.getCount())) as HTMLOutputElement;
    this.countInput.addEventListener("input", () => {
      this.countOut.textContent = this.countInput.value;
      window.clearTimeout(this.rebuildTimer);
      this.rebuildTimer = window.setTimeout(() => {
        const next = Number(this.countInput.value);
        // Thin/grow the living school when the backend can; only fall back
        // to a full respawn (v3 behavior) when it can't.
        if (sys.resize) sys.resize(next);
        else sys.rebuild(next);
        this.save();
      }, 350);
    });
    countRow.appendChild(this.countInput);
    countRow.appendChild(this.countOut);
    root.appendChild(countRow);
    const countHint = el("div", "blm-hint");
    const countDots = el("span", "blm-imp", "●●●");
    countDots.title = "defines the feel";
    countHint.appendChild(countDots);
    countHint.appendChild(
      document.createTextNode(
        "Particle count — first knob to turn if FPS drops",
      ),
    );
    root.appendChild(countHint);

    // Slider groups
    for (const group of GROUPS) {
      const details = document.createElement("details");
      if (group.open) details.open = true;
      const summary = document.createElement("summary");
      summary.textContent = group.title;
      details.appendChild(summary);
      details.appendChild(el("div", "blm-group-desc", group.blurb));

      // Most important first (3 = defining the feel, 1 = fine-tuning).
      const ranked = [...group.sliders].sort((a, b) => b.impact - a.impact);
      for (const def of ranked) {
        details.appendChild(this.makeSlider(def));
      }
      root.appendChild(details);
    }

    // Actions
    const actions = el("div", "blm-actions");
    actions.appendChild(
      this.makeAction("💥 Burst", () => sys.burst(),
        "Detonate a chain reaction at the cursor (same as clicking the swarm)"),
    );
    actions.appendChild(
      this.makeAction("⚡ Startle", () => sys.startle(),
        "Trigger a false alarm — a pocket of panic sweeps the school"),
    );
    actions.appendChild(
      this.makeAction("🌊 Scatter", () => sys.scatter(),
        "Blow the swarm apart and watch it regroup"),
    );
    actions.appendChild(
      this.makeAction("↺ Reset", () => {
        localStorage.removeItem(this.storageKey);
        // Every key, not just the preset-owned ones: a feel or slider tweak
        // outside the presets' keys would otherwise survive the reset.
        Object.assign(sys.params, DEFAULT_PARAMS);
        sys.setPalette("emerald");
        sys.setStyle("glow");
        for (const [cat, name] of Object.entries(DEFAULT_PRESETS)) {
          sys.applyPreset(cat as PresetCategory, name);
        }
        this.descEl.textContent = "mix a movement with a reaction · style & palette below";
        // Drop the feel selection too, or a reload would re-apply it.
        this.setFeelName(null);
        const query = new URLSearchParams(location.search);
        if (query.has("feel")) {
          query.delete("feel");
          const qs = query.toString();
          history.replaceState(history.state, "", `${location.pathname}${qs ? "?" + qs : ""}${location.hash}`);
        }
        this.refresh();
        // Back to the shipped feel; the callback re-highlights it.
        this.onReset?.();
      }, "Restore all settings to defaults and clear saved tuning"),
    );
    actions.appendChild(
      this.makeAction("⧉ Copy JSON", (btn) => {
        const json = JSON.stringify(sys.getSnapshot(), null, 2);
        navigator.clipboard?.writeText(json).then(
          () => flash(btn, "copied!"),
          () => flash(btn, "copy failed"),
        );
      }, "Copy the current settings as a JSON snapshot for applySnapshot()"),
    );
    actions.appendChild(
      this.makeAction("⤓ Save feel…", () => this.saveFeel(),
        "Download the current settings as <name>.json - drop it into snapshots/ to make it a Feel A/B button"),
    );
    root.appendChild(actions);

    const foot = this.opts.backendLabel
      ? `${this.opts.backendLabel} · press D to toggle · settings auto-save`
      : "press D to toggle · settings auto-save";
    root.appendChild(el("div", "blm-foot", foot));

    this.refresh();
  }

  private makeSlider(def: SliderDef): HTMLElement {
    const sys = this.system;
    const wrap = el("div", "");
    const row = el("div", "blm-row");
    row.appendChild(el("label", "", def.label));

    const input = document.createElement("input");
    input.type = "range";
    input.min = String(def.min);
    input.max = String(def.max);
    input.step = String(def.step);
    input.value = String(sys.params[def.key]);

    const out = el("output", "", fmt(sys.params[def.key])) as HTMLOutputElement;

    input.addEventListener("input", () => {
      const v = Number(input.value);
      (sys.params as unknown as Record<string, number>)[def.key] = v;
      out.textContent = fmt(v);
      this.save();
    });

    this.sliderInputs.set(def.key, { input, out });
    row.appendChild(input);
    row.appendChild(out);
    wrap.appendChild(row);

    // Visible description + importance dots (●●● defining … ● fine-tuning).
    const hint = el("div", "blm-hint");
    const dots = el("span", "blm-imp", "●".repeat(def.impact));
    dots.title = ["fine-tuning", "shapes the feel", "defines the feel"][def.impact - 1];
    hint.appendChild(dots);
    hint.appendChild(document.createTextNode(def.hint));
    if (GPU_ONLY_KEYS.has(def.key)) {
      const tag = el("span", "blm-gpu-only", "· GPU only");
      tag.title = "The CPU fallback (v3 reference) ignores this parameter";
      hint.appendChild(tag);
    }
    wrap.appendChild(hint);
    return wrap;
  }

  private makeSelect(
    label: string,
    options: [string, string][],
    onChange: (v: string) => void,
    ref: (sel: HTMLSelectElement) => void,
    hint?: string,
  ): HTMLElement {
    const wrap = el("div", "blm-select-wrap");
    if (hint) wrap.title = hint;
    wrap.appendChild(el("span", "", label));
    const sel = document.createElement("select");
    for (const [value, text] of options) {
      const opt = document.createElement("option");
      opt.value = value;
      opt.textContent = text;
      sel.appendChild(opt);
    }
    sel.addEventListener("change", () => onChange(sel.value));
    ref(sel);
    wrap.appendChild(sel);
    return wrap;
  }

  private makeAction(
    label: string,
    fn: (btn: HTMLButtonElement) => void,
    hint?: string,
  ): HTMLButtonElement {
    const btn = el("button", "blm-action", label) as HTMLButtonElement;
    if (hint) btn.title = hint;
    btn.addEventListener("click", () => fn(btn));
    return btn;
  }

  /** Sync every control with the system's current state. */
  refresh(): void {
    const sys = this.system;
    for (const [key, { input, out }] of this.sliderInputs) {
      const v = (sys.params as unknown as Record<string, number>)[key];
      input.value = String(v);
      out.textContent = fmt(v);
    }
    for (const [key, btn] of this.presetBtns) {
      const [cat, name] = key.split(":");
      btn.classList.toggle(
        "active",
        sys.presetNames[cat as PresetCategory] === name,
      );
    }
    this.styleSel.value = sys.styleName;
    this.paletteSel.value = sys.paletteName;
    this.countInput.value = String(sys.getCount());
    this.countOut.textContent = String(sys.getCount());
  }

  private saveFeel(): void {
    // Suggest the next letter after the active feel (lifelike-a -> lifelike-b).
    const cur = this.feelName ?? "lifelike-a";
    const m = /^(.*-)([a-y])$/.exec(cur);
    const suggestion = m ? m[1] + String.fromCharCode(m[2].charCodeAt(0) + 1) : `${cur}-b`;
    const raw = window.prompt("Save feel as (downloads <name>.json for snapshots/):", suggestion);
    if (!raw) return;
    const name = raw.trim().replace(/\.json$/i, "").replace(/[^\w.-]+/g, "-");
    if (!name) return;
    const json = JSON.stringify(this.system.getSnapshot(), null, 2) + "\n";
    const url = URL.createObjectURL(new Blob([json], { type: "application/json" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = `${name}.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  private save(): void {
    try {
      localStorage.setItem(this.storageKey, JSON.stringify(this.system.getSnapshot()));
    } catch {
      // Private mode / quota — persistence is best-effort.
    }
  }

  private restore(): void {
    // A ?feel= link is an explicit selection: the switcher applies it, and
    // last session's auto-saved tweaks must not paint over it first.
    if (urlFeel()) return;
    try {
      const raw = localStorage.getItem(this.storageKey);
      // Count is a density decision owned by boot (device class, ?count=,
      // the governor), not a saved tweak: restoring a stale count would
      // pin a desktop at last session's phone-sized school.
      if (raw) {
        const { count: _count, ...rest } = JSON.parse(raw) as Record<string, unknown>;
        void _count;
        this.system.applySnapshot(rest);
      }
    } catch {
      localStorage.removeItem(this.storageKey);
    }
  }
}

/** Params only the GPU kernel implements (added after the v3 port). */
const GPU_ONLY_KEYS = new Set<string>(["collapse", "shape", "bite", "flight", "crowdDim"]);

function el(tag: string, cls: string, text?: string): HTMLElement {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

function fmt(v: number): string {
  return Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2);
}

function flash(btn: HTMLButtonElement, msg: string): void {
  const orig = btn.textContent;
  btn.textContent = msg;
  setTimeout(() => (btn.textContent = orig), 900);
}
