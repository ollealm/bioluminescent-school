import * as THREE from "three";
import { baseStops, flashFromBase, hexToRgb, mulberry32 } from "../color";
import {
  DEFAULT_PARAMS,
  GPU_TUNING,
  PALETTES,
  STYLE_IDS,
  TUNING,
  type Palette,
  type PaletteName,
  type StyleName,
  type SwarmParams,
} from "../params";
import {
  CATEGORY_KEYS,
  DEFAULT_PRESETS,
  PRESET_REGISTRY,
  type PresetCategory,
} from "../presets";
import { SNAPSHOT_VERSION, type Snapshot, type SwarmSystem } from "../system";

// Behavioral modes (per particle)
const FLOCK = 0;
const WANDER = 1;
const RETURN = 2;

// School moods (whole-swarm director). CONTRACT is the `collapse` lever's
// mood, unreachable at collapse 0, so the default FSM is v3's.
const M_CRUISE = 0;
const M_TURN = 1;
const M_MILL = 2;
const M_HOVER = 3;
const M_STARTLE = 4;
const M_CONTRACT = 5;

const MOOD_NAMES = ["cruise", "turn", "mill", "hover", "startle", "contract"];

// Spatial grid: dense, directly indexed. D³ cells of edge `perception`,
// centered on the origin; positions outside clamp to border cells (distance
// checks still filter, so far-flung wanderers stay correct, just unbucketed).
const D = TUNING.gridDim;
const D2 = D * D;
const CELLS = D * D * D;
const MAX_NEIGHBORS = TUNING.maxNeighbors;
const MAX_FORCE = TUNING.maxForce;

// Stratified neighbor sampling (GPU_TUNING.sampleCandidates): candidates
// examined per fish, split over the 27 cells by population.
const SAMPLE_CAND = GPU_TUNING.sampleCandidates;
// The 27 neighbor-cell offsets, x fastest (the GPU kernel's OFFS order).
const OFF_X = new Int8Array(27);
const OFF_Y = new Int8Array(27);
const OFF_Z = new Int8Array(27);
for (let k = 0; k < 27; k++) {
  OFF_X[k] = (k % 3) - 1;
  OFF_Y[k] = (((k / 3) | 0) % 3) - 1;
  OFF_Z[k] = ((k / 9) | 0) - 1;
}
// Scratch for the fish being stepped: flat cell index per offset (-1 =
// off-grid), and the reservoir of accepted neighbor indices.
const nbCell = new Int32Array(27);
const nbJ = new Int32Array(MAX_NEIGHBORS);

/** PCG hash, u32 in and out: the GPU kernel's `pcg`, bit for bit. */
function pcg(v: number): number {
  const s = (Math.imul(v, 747796405) + 2891336453) >>> 0;
  const w = Math.imul((s >>> ((s >>> 28) + 4)) ^ s, 277803737) >>> 0;
  return ((w >>> 22) ^ w) >>> 0;
}

// Contagion gain on this engine, calibrated separately from the GPU's
// GPU_TUNING.transmissionScale. With heat-weighted contagion, the kick and
// last-frame neighbor reads (the GPU's mechanics), a centroid burst on
// olle-work (12 runs each) grows 2.5× in its first ⅓ s at 1.0, 3.0× at
// 1.2, 3.4× at 1.4; the GPU grows 3.5×. So the mechanics agree near the
// GPU's own 1.35. The CPU still peaks earlier and dies sooner (~1.75 s vs
// 2.75 s) because its smaller school (RMS 3.9 vs 4.4 u) gives the burst
// twice the seed share and less fuel; that is the feel's geometry, not
// contagion. At 1.35 the flash also stays near the old CPU's (lit 0.67 vs
// 0.73 of the school at ⅓ s).
const CPU_TRANSMISSION_SCALE = 1.35;

// Fine collision grid (GPU_TUNING.fineGridDims cells of fineCellSize): close-range
// personal space, every overlap seen every frame. v3's flocking scan stopped
// at the first 16 neighbors in cell order; in a dense school those all sat
// in one corner cell, separation then pushes every fish the same way (no
// divergence, so it can't hold pressure) and herd/sheet/contract crushed
// the school to a 1 u pancake. Same fix, cells and extent as the GPU
// kernel: the counting sort touches only occupied cells, so the screen-wide
// extent costs memory, not time (a 64³ cube clamped wanderers into its edge
// cells and a school moved past ±14.4 u checked 677k pairs instead of 77k).
const [FX, FY, FZ] = GPU_TUNING.fineGridDims;
const FXY = FX * FY;
const FCELLS = FX * FY * FZ;
const F_INV = 1 / GPU_TUNING.fineCellSize;

// Crowd-dim density grid: counts only, cells of half a perception radius.
// The flocking scan stops at 16 neighbors, so it can't tell a crowded core
// from a merely full one; this grid can, for a few lookups per fish. It
// spans the fine grid's screen-wide extent at the panel's lowest perception
// (1.2; a wider perception only widens it), and a fish outside is neither
// counted nor dimmed: a 48³ cube clamped fish into its edge cells, and a
// wander group crossing x ≈ ±14.4 at perception 1.2 dimmed to 0.3×.
const CROWD_CELL_MIN = 1.2 * 0.5;
const CX = Math.round((FX * GPU_TUNING.fineCellSize) / CROWD_CELL_MIN); // 108
const CY = Math.round((FY * GPU_TUNING.fineCellSize) / CROWD_CELL_MIN); // 72
const CZ = Math.round((FZ * GPU_TUNING.fineCellSize) / CROWD_CELL_MIN); // 48
const CXY = CX * CY;

function smoothstep(e0: number, e1: number, x: number): number {
  const t = Math.max(0, Math.min(1, (x - e0) / (e1 - e0)));
  return t * t * (3 - 2 * t);
}

// Scratch for shapeAniso (no per-fish allocation in the hot loop).
const ANISO = new Float64Array(3);

/**
 * Body-shape anisotropy: soften the pull along the school's heading and
 * stiffen it across, so the body relaxes into a traveling shoal instead of
 * a ball. Same as the GPU kernel's shapeAniso; result lands in ANISO.
 */
function shapeAniso(
  x: number, y: number, z: number,
  hx: number, hy: number, hz: number,
  shape: number,
): void {
  const d = x * hx + y * hy + z * hz;
  const px = d * hx;
  const py = d * hy;
  const pz = d * hz;
  const a = 1 - 0.6 * shape;
  const b = 1 + 0.8 * shape;
  ANISO[0] = px * a + (x - px) * b;
  ANISO[1] = py * a + (y - py) * b;
  ANISO[2] = pz * a + (z - pz) * b;
}

/**
 * Counting sort of fish into grid cells by cellOf, touching only cells
 * that hold fish: O(n) per frame instead of O(cells) (the 262k-cell fine
 * grid's clear and scan were 0.5 ms of a 6.3 ms step at 2,500 fish).
 * Ranges are laid out in first-touched order; every reader goes through
 * start[c] and count[c], and each cell's entries keep ascending fish order,
 * so lookups are unchanged. On entry `count` is zero except the cells in
 * touched[0, touchedN), which are cleared first. Returns the new touchedN.
 */
function sortIntoCells(
  n: number,
  cellOf: Int32Array,
  count: Int32Array,
  start: Int32Array,
  entries: Int32Array,
  touched: Int32Array,
  touchedN: number,
): number {
  for (let k = 0; k < touchedN; k++) count[touched[k]] = 0;
  let tn = 0;
  for (let i = 0; i < n; i++) {
    const c = cellOf[i];
    if (count[c]++ === 0) touched[tn++] = c;
  }
  let sum = 0;
  for (let k = 0; k < tn; k++) {
    const c = touched[k];
    start[c] = sum;
    sum += count[c];
  }
  // Starts double as scatter cursors, then shift back.
  for (let i = 0; i < n; i++) entries[start[cellOf[i]]++] = i;
  for (let k = 0; k < tn; k++) start[touched[k]] -= count[touched[k]];
  return tn;
}

/**
 * Set a THREE.Color to a "#rrggbb" as raw sRGB values, as the GPU backend
 * and the vertex colors (hexToRgb) take it. A plain set() converts sRGB to
 * linear under ColorManagement, which made the CPU's flash hue darker and
 * more saturated than the GPU's; declaring the input linear skips that.
 */
function setRaw(color: THREE.Color, hex: string): THREE.Color {
  return color.setStyle(hex, THREE.LinearSRGBColorSpace);
}

function detectMobile(): boolean {
  if (typeof window === "undefined") return false;
  return (
    window.matchMedia?.("(pointer: coarse)").matches ||
    window.innerWidth < 768
  );
}

export interface ParticleSystemOptions {
  /** Camera used to project the pointer onto the swarm plane. */
  camera?: THREE.PerspectiveCamera;
  /** Particle count. Defaults to 2600 desktop / 1400 mobile. */
  count?: number;
  /**
   * Slots to allocate up front (at least the count), so resize() up to
   * this many thins and revives the living school instead of respawning it.
   */
  capacity?: number;
  /** Starting presets per category, e.g. { motion: "shoal", look: "reef" }. */
  presets?: Partial<Record<PresetCategory, string>>;
}

type Vec3 = { x: number; y: number; z: number };

/**
 * A living school of bioluminescent motes.
 *
 * Four interacting layers produce the "alive" feel:
 *  1. Boids flocking (separation / alignment / cohesion) over a spatial hash
 *     grid, plus a curl-like flow field for water.
 *  2. A swarm director — a mood machine (cruise → turn-as-one → mill →
 *     hover, with spontaneous startles) that gives the school long-term
 *     temporal drama: it travels, changes its mind, rests, panics at
 *     nothing. Turns are seeded through a vanguard cluster so direction
 *     changes propagate as visible waves.
 *  3. Individuals: burst-and-glide swimming gaits, per-fish speed and
 *     sociability, wanderer groups that leave and come home.
 *  4. Light bound to motion: an excitable flash medium (chain-reaction
 *     bubbles with shared epicenters) *plus* shear-triggered luminescence —
 *     agitated particles glow, so every maneuver paints itself in light,
 *     the way real dinoflagellates fire when the water shears.
 *
 * This is the v3 engine carrying the GPU fork's levers (bite, flight,
 * rally, evade, herd + contract, shiver, sheet, shape, roamDepth, the turn
 * wave, the strike, crowd dim, accent/base overrides), ported from
 * src/gpu/wgsl-sim.ts and GpuSwarm.ts so the fallback feels like the GPU
 * school with fewer fish. Every lever defaults to 0. Also ported: the fine
 * collision grid, because the levers pack the school dense enough that
 * v3's 16-neighbor cap starves separation here too, and the GPU's
 * stratified neighbor sampling, because v3's fixed cell-order scan is a
 * position-quantized bias that bands the school along the grid planes. Not
 * ported: the other compensations for 20× density (transmission /
 * fear-spread / refractory scales, predator crowd relief).
 */
export class ParticleSystem implements SwarmSystem {
  /** Live-tweakable parameters (the dev panel binds directly to this). */
  readonly params: SwarmParams;
  presetNames: Record<PresetCategory, string> = { ...DEFAULT_PRESETS };
  paletteName: PaletteName;
  styleName: StyleName;

  private scene: THREE.Scene;
  private camera?: THREE.PerspectiveCamera;
  private count = 0;
  /** Allocated slots; fish in [count, capacity) are dormant. */
  private capacity = 0;
  private minCapacity: number;
  /** Multiplies the Opacity slider; see setOpacityScale(). */
  private opacityScale = 1;

  private geometry!: THREE.BufferGeometry;
  private material: THREE.ShaderMaterial;
  private points!: THREE.Points;

  // Simulation state — pos/vel/energy/glow alias the GPU attribute arrays.
  private pos!: Float32Array;
  private vel!: Float32Array;
  private energy!: Float32Array;
  private glow!: Float32Array;
  /** Crowd-dim alpha per fish (1 = untouched); a render attribute. */
  private fade!: Float32Array;
  private refract!: Float32Array;
  private fired!: Uint8Array;
  private mode!: Uint8Array;
  private modeTimer!: Float32Array;
  private wanderDir!: Float32Array;
  private seeds!: Float32Array;
  /** Predator fear per particle (0..1) — drives fleeing and panic spread. */
  private fear!: Float32Array;
  /** Epicenter of the cascade that lit each particle; pulses are radial. */
  private pulse!: Float32Array;
  /** +1 = outburst (bubble bursting), -1 = inburst (bubble collapsing). */
  private pulseSign!: Float32Array;
  // Last frame's state. Neighbors read these, never the live arrays:
  // updating in place let a cascade or a panic hop several fish per frame
  // along increasing indices, and made motion depend on array order (two
  // fish swapping indices reversed their drift). The GPU reads last frame's
  // buffer the same way.
  private posPrev!: Float32Array;
  private velPrev!: Float32Array;
  private energyPrev!: Float32Array;
  private fearPrev!: Float32Array;
  private pulsePrev!: Float32Array;
  private pulseSignPrev!: Float32Array;
  // Individuality
  private speedFactor!: Float32Array;
  private social!: Float32Array;
  private gaitClock!: Float32Array;

  // Spatial grid (counting sort over occupied cells, rebuilt every frame)
  private gridCount = new Int32Array(CELLS);
  private gridStart = new Int32Array(CELLS);
  private gridEntries!: Int32Array;
  private cellOf!: Int32Array;
  /** Cells holding fish last frame (cleared at the next sort). */
  private gridTouched!: Int32Array;
  private gridTouchedN = 0;
  /** Stepped-frame counter: seeds the per-frame neighbor sampling hash. */
  private frameNo = 0;
  // Fine collision grid (same counting sort)
  private fineCount = new Int32Array(FCELLS);
  private fineStart = new Int32Array(FCELLS);
  private fineEntries!: Int32Array;
  private fineCellOf!: Int32Array;
  private fineTouched!: Int32Array;
  private fineTouchedN = 0;

  // Crowd dim (counts only; cleared through the touched list like the grids)
  private crowdCount = new Int32Array(CXY * CZ);
  private crowdTouched!: Int32Array;
  private crowdTouchedN = 0;
  /** Smoothed mean local count over the school; -1 = not measured yet. */
  private nRef = -1;
  private crowdOn = false;

  // Colors
  private _accent: string | null = null;
  private _base: string | null = null;
  private lastAccentShare = NaN;
  private colorSeed = 0;

  private time = 0;
  private pointer = new THREE.Vector3();
  private prevPointerW = new THREE.Vector3();
  private pointerVel = new THREE.Vector3();
  private hadPointerFrame = false;
  private pointerActive = false;
  private mouseSpeed = 0;
  private lastPointerTime = 0;
  private prevNDC = new THREE.Vector2();
  private hasPrevNDC = false;
  private aspect = 1;

  private centroidX = 0;
  private centroidY = 0;
  private centroidZ = 0;
  /** RMS distance of flock fish from the centroid: the school's live spread. */
  private rmsRadius = 0;
  /** Mean per-fish fear: the director's alarm signal. */
  private meanFear = 0;
  private wanderClock = 5;

  // Evade: smoothed displacement of the school's center away from a parked
  // predator (stays zero at evade 0).
  private evadeX = 0;
  private evadeY = 0;
  private evadeZ = 0;

  // Strike (flick) detection.
  private strikeCooldown = 0;
  private lastSlowTime = 0;

  // --- Swarm director state ---
  private mood = M_CRUISE;
  private moodTimer = 6;
  /** Dev test control: pin the director to one mood (index), null = the FSM. */
  private forced: number | null = null;
  private headX = 1;
  private headY = 0;
  private headZ = 0;
  // Turn wave: fish outside a sphere growing from the vanguard fish keep
  // the pre-turn heading, so the turn spreads from a small group.
  private turnOldX = 1;
  private turnOldY = 0;
  private turnOldZ = 0;
  private turnIndex = 0;
  private turnRadius = 0;
  private turnActive = 0;
  /** An alarm collapse waiting for a running turn to finish. */
  private alarmPending = false;
  /** Consecutive turns (through cruise) since another mood; capped at 2. */
  private turnsInARow = 0;
  private startleClock = 25;
  /** Countdown to the next resting agitation wave (shiver). */
  private shiverClock = 8;
  // Smoothed mood multipliers (live) and their targets
  private dirAlign = 1;
  private dirCoh = 1;
  private dirMill = 0;
  private dirDrive = 1;
  private dirSpeed = 0.6;
  /** Separation gain: 1 except the 1.25 pop released after a contraction. */
  private dirSep = 1;
  /** Herd-drive gain: swells while contracting, 1 otherwise. */
  private dirHerd = 1;
  private tgtAlign = 1;
  private tgtCoh = 1;
  private tgtMill = 0;
  private tgtDrive = 1;
  private tgtSpeed = 0.6;
  private tgtSep = 1;
  private tgtHerd = 1;

  // Alarm-triggered collapse: a scare that subsides pulls the whole school
  // into a bait ball.
  /** meanFear smoothed over ~0.15 s. */
  private alarm = 0;
  /** Running max of alarm, decaying back toward it at 1/s. */
  private alarmPeak = 0;
  private alarmCooldown = 0;
  /** Set right before enterMood(M_CONTRACT) to pick the harder contraction. */
  private contractFromAlarm = false;

  constructor(scene: THREE.Scene, options: ParticleSystemOptions = {}) {
    this.scene = scene;
    this.camera = options.camera;
    this.aspect =
      typeof window !== "undefined"
        ? window.innerWidth / Math.max(1, window.innerHeight)
        : 1;

    this.params = { ...DEFAULT_PARAMS };
    this.paletteName = "emerald";
    this.styleName = "glow";

    // Random initial heading so every load starts differently.
    const a = Math.random() * Math.PI * 2;
    this.headX = Math.cos(a);
    this.headY = Math.sin(a);
    this.headZ = 0;

    this.material = this.createMaterial();
    this.setStyle(this.styleName);
    this.minCapacity = options.capacity ?? 0;
    this.rebuild(options.count ?? (detectMobile() ? 1400 : 2600));

    if (options.presets) {
      for (const [cat, name] of Object.entries(options.presets)) {
        if (name) this.applyPreset(cat as PresetCategory, name);
      }
    }
  }

  getCount(): number {
    return this.count;
  }

  /** Current director mood, for debugging / curiosity. */
  getMood(): string {
    return MOOD_NAMES[this.mood];
  }

  /** Current director mood, for harnesses and debugging. */
  get moodName(): string {
    return MOOD_NAMES[this.mood];
  }

  get moodNames(): readonly string[] {
    return MOOD_NAMES;
  }

  get forcedMood(): string | null {
    return this.forced === null ? null : MOOD_NAMES[this.forced];
  }

  /**
   * Pin the director to one mood so it can be judged in isolation; null
   * hands control back to the FSM (the current mood runs out its timer).
   */
  forceMood(name: string | null): void {
    const i = name === null ? -1 : MOOD_NAMES.indexOf(name);
    this.forced = i < 0 ? null : i;
    if (this.forced !== null) this.enterForced();
  }

  /** Re-enter the pinned mood; startle goes through startle() so its pocket fires again. */
  private enterForced(): void {
    if (this.forced === M_STARTLE) this.startle();
    else this.enterMood(this.forced!);
  }

  /** RMS radius of a uniform ball filling 0.8 × the tank: R·√(3/5). */
  private nominalRmsRadius(): number {
    return this.params.containRadius * 0.8 * Math.sqrt(3 / 5);
  }

  // ------------------------------------------------------------------
  // Presets / palettes / styles
  // ------------------------------------------------------------------

  /**
   * Apply one preset. Only the params belonging to that category are reset
   * to defaults and overwritten, so motion / reaction / look combine freely.
   */
  applyPreset(category: PresetCategory, name: string): void {
    const preset = PRESET_REGISTRY[category]?.[name];
    if (!preset) return;
    this.presetNames[category] = name;

    const params = this.params as unknown as Record<string, number>;
    const defaults = DEFAULT_PARAMS as unknown as Record<string, number>;
    for (const key of CATEGORY_KEYS[category]) {
      params[key] = (preset.params as Record<string, number>)[key] ?? defaults[key];
    }
  }

  setPalette(name: PaletteName): void {
    if (!PALETTES[name]) return;
    this.paletteName = name;
    this.syncFlashColor();
    this.applyPaletteColors();
  }

  /** The accent override, or null while the palette's own accent shows. */
  get accentColor(): string | null {
    return this._accent;
  }

  /** Override the palette's accent ("#rrggbb"); null restores it. */
  setAccent(hex: string | null): void {
    if (hex !== null && !/^#[0-9a-f]{6}$/i.test(hex)) return;
    const next = hex === null ? null : hex.toLowerCase();
    if (next === this._accent) return;
    this._accent = next;
    this.applyPaletteColors();
  }

  /** The base override, or null while the palette's own gradient shows. */
  get baseColor(): string | null {
    return this._base;
  }

  /** Derive the gradient (and flash hue) from one color; null restores the palette's. */
  setBase(hex: string | null): void {
    if (hex !== null && !/^#[0-9a-f]{6}$/i.test(hex)) return;
    const next = hex === null ? null : hex.toLowerCase();
    if (next === this._base) return;
    this._base = next;
    this.syncFlashColor();
    this.applyPaletteColors();
  }

  /** Flash hue ("#rrggbb"): the palette's, or derived from the base override. */
  get flashColor(): string {
    return this._base ? flashFromBase(this._base) : PALETTES[this.paletteName].flash;
  }

  private syncFlashColor(): void {
    setRaw(this.material.uniforms.uFlashColor.value as THREE.Color, this.flashColor);
  }

  setStyle(name: StyleName): void {
    if (!(name in STYLE_IDS)) return;
    this.styleName = name;
    this.material.uniforms.uStyle.value = STYLE_IDS[name];
  }

  getSnapshot(): Snapshot {
    return {
      version: SNAPSHOT_VERSION,
      presets: { ...this.presetNames },
      palette: this.paletteName,
      style: this.styleName,
      count: this.count,
      params: { ...this.params },
      // Only when set: a snapshot without one follows the palette.
      ...(this._accent ? { accent: this._accent } : {}),
      ...(this._base ? { base: this._base } : {}),
    };
  }

  /**
   * Snapshots are applied key-by-key with validation, so older/newer
   * snapshots degrade gracefully: unknown presets/palettes are skipped and
   * unknown params are simply extra keys. `version` is informational.
   */
  applySnapshot(s: Partial<Snapshot>): void {
    if (s.presets) {
      for (const [cat, name] of Object.entries(s.presets)) {
        if (name) this.applyPreset(cat as PresetCategory, name);
      }
    }
    if (s.params) {
      const params = this.params as unknown as Record<string, number>;
      for (const [k, v] of Object.entries(s.params)) {
        if (k in DEFAULT_PARAMS && typeof v === "number") params[k] = v;
      }
    }
    if (s.accent !== undefined) this.setAccent(s.accent);
    if (s.base !== undefined) this.setBase(s.base);
    if (s.palette) this.setPalette(s.palette);
    if (s.style) this.setStyle(s.style);
    if (s.count && s.count !== this.count) this.rebuild(s.count);
  }

  // ------------------------------------------------------------------
  // Construction
  // ------------------------------------------------------------------

  private createMaterial(): THREE.ShaderMaterial {
    const vertexShader = /* glsl */ `
      attribute float aSize;
      attribute float aEnergy;
      attribute float aSeed;
      attribute float aGlow;
      attribute float aFade;
      attribute vec3 aVel;

      uniform float uTime;
      uniform float uSize;
      uniform float uDpr;
      uniform float uTwinkle;
      uniform float uStretch;
      uniform float uFlashSize;

      varying vec3 vColor;
      varying float vEnergy;
      varying float vSeed;
      varying float vTw;
      varying float vGlow;
      varying float vFade;
      varying vec2 vDir;
      varying float vStretch;

      void main() {
        vColor = color;
        vEnergy = aEnergy;
        vSeed = aSeed;
        vGlow = aGlow;
        vFade = aFade;

        // Slow per-particle brightness oscillation ("breathing" of individuals)
        float tw = 1.0 - uTwinkle * 0.5 * (0.5 + 0.5 * sin(uTime * (0.5 + aSeed * 1.9) + aSeed * 87.0));
        vTw = tw;

        vec4 mv = modelViewMatrix * vec4(position, 1.0);

        // Velocity in view space → screen-space swim direction + stretch.
        // Fast particles render as streaks along their motion; still ones
        // stay round motes.
        vec3 vv = (modelViewMatrix * vec4(aVel, 0.0)).xyz;
        float sp = length(vv.xy);
        float st = 1.0 + uStretch * min(sp * 0.16, 1.7);
        vDir = sp > 0.001 ? vv.xy / sp : vec2(1.0, 0.0);
        vStretch = st;

        // uFlashSize: how much a lit fish swells (v3 hard-coded 0.45).
        float size = aSize * uSize * (1.0 + aEnergy * uFlashSize) * (0.72 + 0.28 * tw);
        gl_PointSize = size * uDpr * (340.0 / -mv.z) * st;
        gl_Position = projectionMatrix * mv;
      }
    `;

    const fragmentShader = /* glsl */ `
      uniform int uStyle;
      uniform float uTime;
      uniform float uOpacity;
      uniform float uFlashBright;
      uniform float uMotionGlow;
      uniform vec3 uFlashColor;

      varying vec3 vColor;
      varying float vEnergy;
      varying float vSeed;
      varying float vTw;
      varying float vGlow;
      varying float vFade;
      varying vec2 vDir;
      varying float vStretch;

      void main() {
        vec2 uvc = gl_PointCoord - 0.5;
        // Rotate into the swim frame and compress the across-axis: the
        // sprite becomes a streak along the direction of motion.
        vec2 pr = vec2(
          dot(uvc, vDir),
          (uvc.x * vDir.y - uvc.y * vDir.x) * vStretch
        );
        float d = length(pr) * 2.0;
        if (d > 1.0) discard;

        float core = exp(-d * d * 12.0);
        float a;

        if (uStyle == 0) {
          // glow — soft gaussian blob
          a = exp(-d * d * 4.5) * 0.8 + core * 0.6;
        } else if (uStyle == 1) {
          // orb — defined body with a faint halo
          a = smoothstep(0.55, 0.38, d) * 0.9 + exp(-d * d * 2.6) * 0.3;
        } else if (uStyle == 2) {
          // spark — bright core with a flare along the swim axis
          float c1 = exp(-pow(pr.x * 16.0, 2.0)) * exp(-pow(pr.y * 2.2, 2.0));
          float c2 = exp(-pow(pr.y * 16.0, 2.0)) * exp(-pow(pr.x * 2.2, 2.0));
          a = core * 0.85 + max(c1, c2) * 0.7;
        } else {
          // firefly — glow with slow individual blinking
          float blink = 0.18 + 0.82 * pow(0.5 + 0.5 * sin(uTime * (0.9 + vSeed * 1.4) + vSeed * 92.0), 5.0);
          a = (exp(-d * d * 4.5) * 0.8 + core * 0.6) * blink;
        }

        // Excitement (cascade flash) + agitation (shear-triggered glow):
        // both lean toward the palette's saturated flash hue and brighten —
        // the creature glows, it doesn't detonate.
        float e = clamp(vEnergy, 0.0, 1.0);
        float g = clamp(vGlow, 0.0, 1.0) * uMotionGlow;
        vec3 col = mix(vColor, uFlashColor, min(1.0, e * 0.45 + g * 0.15));
        col *= 1.0 + e * uFlashBright + g * 0.8 + core * 0.55;

        a *= uOpacity * (0.35 + 0.65 * vTw) * (1.0 + e * 0.5 + g * 0.3);
        // Crowd dim: fish in cores denser than the school's mean fade, so
        // additive pile-ups read as density, not one blown-out spot.
        a *= vFade;
        gl_FragColor = vec4(col * a, a);
      }
    `;

    const dpr =
      typeof window !== "undefined"
        ? Math.min(window.devicePixelRatio || 1, 2)
        : 1;

    return new THREE.ShaderMaterial({
      uniforms: {
        uTime: { value: 0 },
        uSize: { value: this.params.particleSize },
        uDpr: { value: dpr },
        uTwinkle: { value: this.params.twinkle },
        uStretch: { value: this.params.stretch },
        uFlashSize: { value: this.params.flashSize },
        uMotionGlow: { value: this.params.motionGlow },
        uStyle: { value: 0 },
        uOpacity: { value: this.params.opacity },
        uFlashBright: { value: this.params.flashBrightness },
        uFlashColor: { value: setRaw(new THREE.Color(), this.flashColor) },
      },
      vertexShader,
      fragmentShader,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthTest: false,
      depthWrite: false,
      vertexColors: true,
    });
  }

  /**
   * Respawn the school at a new count (the harness relies on this for a
   * clean field). Params are untouched. Buffers are sized to the capacity,
   * so later resizes within it keep the school.
   */
  rebuild(count: number): void {
    count = Math.max(64, Math.floor(count));
    const cap = Math.max(count, this.minCapacity);
    if (this.points) {
      this.scene.remove(this.points);
      this.geometry.dispose();
    }

    this.count = count;
    this.capacity = cap;
    const positions = new Float32Array(cap * 3);
    const colors = new Float32Array(cap * 3);
    const sizes = new Float32Array(cap);
    const energies = new Float32Array(cap);
    const glows = new Float32Array(cap);
    const seeds = new Float32Array(cap);
    const fades = new Float32Array(cap).fill(1);

    this.vel = new Float32Array(cap * 3);
    this.refract = new Float32Array(cap);
    this.fired = new Uint8Array(cap);
    this.mode = new Uint8Array(cap);
    this.modeTimer = new Float32Array(cap);
    this.wanderDir = new Float32Array(cap * 3);
    this.fear = new Float32Array(cap);
    this.pulse = new Float32Array(cap * 3);
    this.pulseSign = new Float32Array(cap);
    this.posPrev = new Float32Array(cap * 3);
    this.velPrev = new Float32Array(cap * 3);
    this.energyPrev = new Float32Array(cap);
    this.fearPrev = new Float32Array(cap);
    this.pulsePrev = new Float32Array(cap * 3);
    this.pulseSignPrev = new Float32Array(cap);
    this.speedFactor = new Float32Array(cap);
    this.social = new Float32Array(cap);
    this.gaitClock = new Float32Array(cap);
    this.gridEntries = new Int32Array(cap);
    this.cellOf = new Int32Array(cap);
    this.fineEntries = new Int32Array(cap);
    this.fineCellOf = new Int32Array(cap);
    // New touched lists for the new capacity: the counts they'd clear go now.
    this.gridCount.fill(0);
    this.fineCount.fill(0);
    this.crowdCount.fill(0);
    this.gridTouched = new Int32Array(cap);
    this.fineTouched = new Int32Array(cap);
    this.crowdTouched = new Int32Array(cap);
    this.gridTouchedN = this.fineTouchedN = this.crowdTouchedN = 0;
    this.seeds = seeds;
    this.fade = fades;
    this.crowdOn = false;
    this.nRef = -1;
    this.turnActive = 0;
    this.rmsRadius = this.nominalRmsRadius();
    this.meanFear = 0;

    const R = this.params.containRadius;
    // Dormant slots spawn too: their statics (size, seed, personality,
    // color) are what resize() wakes them with.
    for (let i = 0; i < cap; i++) {
      // Spawn in a soft sphere; a few in an outer shell for loose whiskers.
      // A 1% fringe of strays, not v3's 15% halo out to 1.4R: the halo
      // read as "a sphere inside a sparser sphere" on every reload and the
      // first seconds were always the same collapse (owner's note).
      const shell = Math.random() < 0.01;
      const r = shell
        ? R * (0.85 + Math.random() * 0.2)
        : R * 0.8 * Math.cbrt(Math.random());
      const theta = Math.random() * Math.PI * 2;
      const cosPhi = Math.random() * 2 - 1;
      const sinPhi = Math.sqrt(1 - cosPhi * cosPhi);

      positions[i * 3] = r * sinPhi * Math.cos(theta);
      positions[i * 3 + 1] = r * sinPhi * Math.sin(theta);
      positions[i * 3 + 2] = r * cosPhi;

      this.vel[i * 3] = (Math.random() - 0.5) * 0.6;
      this.vel[i * 3 + 1] = (Math.random() - 0.5) * 0.6;
      this.vel[i * 3 + 2] = (Math.random() - 0.5) * 0.6;

      // Mostly small motes, a few large ones.
      sizes[i] = 0.5 + Math.pow(Math.random(), 1.6) * 0.9;
      seeds[i] = Math.random();

      // Individuality: pace, sociability, staggered swim rhythm.
      this.speedFactor[i] = 0.82 + Math.random() * 0.36;
      this.social[i] = 0.7 + Math.random() * 0.6;
      this.gaitClock[i] = Math.random() * 1.5;
    }

    this.geometry = new THREE.BufferGeometry();
    this.geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    this.geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    this.geometry.setAttribute("aSize", new THREE.BufferAttribute(sizes, 1));
    this.geometry.setAttribute("aEnergy", new THREE.BufferAttribute(energies, 1));
    this.geometry.setAttribute("aGlow", new THREE.BufferAttribute(glows, 1));
    this.geometry.setAttribute("aSeed", new THREE.BufferAttribute(seeds, 1));
    this.geometry.setAttribute("aFade", new THREE.BufferAttribute(fades, 1));
    // The physics velocity array doubles as the GPU attribute for stretch.
    this.geometry.setAttribute("aVel", new THREE.BufferAttribute(this.vel, 3));

    this.pos = positions;
    this.energy = energies;
    this.glow = glows;

    this.colorSeed = (Math.random() * 2 ** 32) >>> 0;
    this.applyPaletteColors();

    this.geometry.setDrawRange(0, count);
    this.points = new THREE.Points(this.geometry, this.material);
    this.points.frustumCulled = false;
    this.scene.add(this.points);
  }

  /**
   * Change the count without resetting the school, as GpuSwarm.resize does:
   * shrinking puts the tail to sleep (the sim, the grids and the draw range
   * all stop at count), growing wakes dormant slots near the centroid. Only
   * a count past the allocated capacity respawns.
   */
  resize(count: number): void {
    count = Math.max(64, Math.floor(count));
    if (count > this.capacity) {
      this.rebuild(count);
      return;
    }
    const grow = count > this.count;
    if (grow) this.reviveRange(this.count, count);
    this.count = count;
    this.geometry.setDrawRange(0, count);
    // Upload the woken fish now: a paused school (timeScale 0) returns from
    // update() before it queues them, and they'd draw where they slept.
    if (grow) {
      const attrs = this.geometry.attributes;
      for (const name of ["position", "aVel", "aEnergy", "aGlow", "aFade"]) {
        this.markLive(attrs[name] as THREE.BufferAttribute);
      }
    }
  }

  /**
   * Wake dormant slots [from, to) in a soft ball around the school's
   * center, at rest and unlit; their statics were spawned with the
   * capacity. The previous-frame copies get the same state, so no
   * neighbor reads a woken fish where it slept.
   */
  private reviveRange(from: number, to: number): void {
    const R = this.params.containRadius;
    for (let i = from; i < to; i++) {
      const r = R * 0.7 * Math.cbrt(Math.random());
      const theta = Math.random() * Math.PI * 2;
      const cosPhi = Math.random() * 2 - 1;
      const sinPhi = Math.sqrt(1 - cosPhi * cosPhi);
      const i3 = i * 3;
      this.pos[i3] = this.centroidX + r * sinPhi * Math.cos(theta);
      this.pos[i3 + 1] = this.centroidY + r * sinPhi * Math.sin(theta);
      this.pos[i3 + 2] = this.centroidZ + r * cosPhi;
      this.vel[i3] = (Math.random() - 0.5) * 0.6;
      this.vel[i3 + 1] = (Math.random() - 0.5) * 0.6;
      this.vel[i3 + 2] = (Math.random() - 0.5) * 0.6;
      this.wanderDir[i3] = this.wanderDir[i3 + 1] = this.wanderDir[i3 + 2] = 0;
      this.pulse[i3] = this.pulse[i3 + 1] = this.pulse[i3 + 2] = 0;
      this.pulseSign[i] = 0;
      this.energy[i] = 0;
      this.glow[i] = 0;
      this.fear[i] = 0;
      this.refract[i] = 0;
      this.fired[i] = 0;
      this.mode[i] = FLOCK;
      this.modeTimer[i] = 0;
      this.fade[i] = 1;
      this.gaitClock[i] = Math.random() * 1.5;
    }
    const a3 = from * 3;
    const b3 = to * 3;
    this.posPrev.set(this.pos.subarray(a3, b3), a3);
    this.velPrev.set(this.vel.subarray(a3, b3), a3);
    this.pulsePrev.set(this.pulse.subarray(a3, b3), a3);
    this.energyPrev.fill(0, from, to);
    this.fearPrev.fill(0, from, to);
    this.pulseSignPrev.fill(0, from, to);
  }

  /**
   * Brightness for a school the count governor has thinned: multiplies the
   * Opacity slider (1 = literal), so fewer fish keep the perceived mass.
   */
  setOpacityScale(k: number): void {
    this.opacityScale = k;
  }

  /**
   * Same draws as the GPU backend: every fish takes five draws from the
   * spawn's seed whichever branch it lands in, so dragging the accent share
   * flips only the fish crossing it and a new accent recolors only accent
   * fish. accentShare replaces the palette's own accentChance.
   */
  private applyPaletteColors(): void {
    if (!this.geometry) return;
    const palette: Palette = PALETTES[this.paletteName];
    const stops = this._base ? baseStops(this._base) : palette.stops.map(hexToRgb);
    const accent = hexToRgb(this._accent ?? palette.accent);
    const share = this.params.accentShare ?? palette.accentChance;
    this.lastAccentShare = this.params.accentShare;
    const rand = mulberry32(this.colorSeed);
    const colors = this.geometry.attributes.color.array as Float32Array;

    // The whole capacity, so a woken fish already has its color.
    for (let i = 0; i < this.capacity; i++) {
      const pick = rand();
      // Sample the gradient, biased toward the brighter end.
      const t = Math.pow(rand(), 0.8) * (stops.length - 1);
      let r: number, g: number, b: number;
      if (pick < share) {
        [r, g, b] = accent;
      } else {
        const idx = Math.min(stops.length - 2, Math.floor(t));
        const f = t - idx;
        r = stops[idx][0] + (stops[idx + 1][0] - stops[idx][0]) * f;
        g = stops[idx][1] + (stops[idx + 1][1] - stops[idx][1]) * f;
        b = stops[idx][2] + (stops[idx + 1][2] - stops[idx][2]) * f;
      }
      colors[i * 3] = Math.min(1, r * (0.9 + rand() * 0.2));
      colors[i * 3 + 1] = Math.min(1, g * (0.9 + rand() * 0.2));
      colors[i * 3 + 2] = Math.min(1, b * (0.9 + rand() * 0.2));
    }
    this.geometry.attributes.color.needsUpdate = true;
  }

  // ------------------------------------------------------------------
  // Pointer
  // ------------------------------------------------------------------

  /** NDC coordinates in [-1, 1]. Same signature as the old system. */
  updateMousePosition(ndc: { x: number; y: number }): void {
    const now =
      typeof performance !== "undefined" ? performance.now() : Date.now();

    if (this.hasPrevNDC) {
      const dt = Math.max(0.008, (now - this.lastPointerTime) / 1000);
      const dx = ndc.x - this.prevNDC.x;
      const dy = ndc.y - this.prevNDC.y;
      const speed = Math.min(8, Math.sqrt(dx * dx + dy * dy) / dt);
      this.mouseSpeed += (speed - this.mouseSpeed) * 0.35;
    }
    this.prevNDC.set(ndc.x, ndc.y);
    this.lastPointerTime = now;
    this.hasPrevNDC = true;
    this.pointerActive = true;

    if (this.camera) {
      // Aim at the school's mid-depth, not a fixed z=0 plane, so the
      // predator column tracks wherever the school swims (as on the GPU).
      // Clamped so a degenerate centroid can't fling the pointer.
      const R = this.params.containRadius;
      const planeZ = Math.max(-R, Math.min(R, this.centroidZ));
      // unproject reads matrixWorld, which only a render refreshes: before
      // the first frame (a harness run at load) it is identity and the
      // pointer collapsed ~60x toward the center.
      this.camera.updateMatrixWorld();
      const v = new THREE.Vector3(ndc.x, ndc.y, 0.5).unproject(this.camera);
      const dir = v.sub(this.camera.position).normalize();
      if (Math.abs(dir.z) > 1e-4) {
        const t = (planeZ - this.camera.position.z) / dir.z;
        this.pointer.copy(this.camera.position).addScaledVector(dir, t);
        return;
      }
    }
    // Fallback without a camera: approximate screen→world mapping.
    this.pointer.set(ndc.x * 13 * this.aspect, ndc.y * 13, 0);
  }

  clearPointer(): void {
    this.pointerActive = false;
    this.hasPrevNDC = false;
    this.mouseSpeed = 0;
  }

  updateRendererSize(halfX: number, halfY: number): void {
    this.aspect = halfX / Math.max(1, halfY);
  }

  // ------------------------------------------------------------------
  // Actions
  // ------------------------------------------------------------------

  /** Detonate a chain reaction at the pointer (or swarm center). */
  burst(at?: Vec3): void {
    const cx = at ? at.x : this.pointerActive ? this.pointer.x : 0;
    const cy = at ? at.y : this.pointerActive ? this.pointer.y : 0;
    const cz = at ? at.z : this.pointerActive ? this.pointer.z : 0;
    // Small seed — the chain reaction itself should do the spreading.
    const radius = Math.max(2, this.params.mouseRadius * TUNING.burstSeedRatio);

    for (let i = 0; i < this.count; i++) {
      const dx = this.pos[i * 3] - cx;
      const dy = this.pos[i * 3 + 1] - cy;
      const dz = this.pos[i * 3 + 2] - cz;
      const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (d > radius) continue;

      const fall = 1 - d / radius;
      const inv = 1 / Math.max(0.2, d);
      this.vel[i * 3] += dx * inv * (9 * fall + 2);
      this.vel[i * 3 + 1] += dy * inv * (9 * fall + 2);
      this.vel[i * 3 + 2] += dz * inv * (9 * fall + 2) * 0.5;

      this.refract[i] = 0;
      this.energy[i] = 1;
      this.fired[i] = 1;
      this.glow[i] = 1;
      this.pulse[i * 3] = cx;
      this.pulse[i * 3 + 1] = cy;
      this.pulse[i * 3 + 2] = cz;
      this.pulseSign[i] = 1;
    }
  }

  /** Blow the whole swarm apart; it will drift back and reunite. */
  scatter(): void {
    for (let i = 0; i < this.count; i++) {
      const x = this.pos[i * 3];
      const y = this.pos[i * 3 + 1];
      const z = this.pos[i * 3 + 2];
      const d = Math.max(0.5, Math.sqrt(x * x + y * y + z * z));
      const jx = (Math.random() - 0.5) * 0.8;
      const jy = (Math.random() - 0.5) * 0.8;
      const jz = (Math.random() - 0.5) * 0.8;
      const s = 5 + Math.random() * 7;
      this.vel[i * 3] += (x / d + jx) * s;
      this.vel[i * 3 + 1] += (y / d + jy) * s;
      this.vel[i * 3 + 2] += (z / d + jz) * s * 0.6;

      if (Math.random() < 0.35) {
        this.mode[i] = WANDER;
        this.modeTimer[i] = 2 + Math.random() * 2.5;
        this.wanderDir[i * 3] = x / d + jx;
        this.wanderDir[i * 3 + 1] = y / d + jy;
        this.wanderDir[i * 3 + 2] = (z / d + jz) * 0.5;
      }
    }
    for (let k = 0; k < 12; k++) {
      const i = (Math.random() * this.count) | 0;
      this.igniteIndex(
        i,
        true,
        this.pos[i * 3],
        this.pos[i * 3 + 1],
        this.pos[i * 3 + 2],
        1,
      );
    }
  }

  /**
   * A false alarm: fear erupts at a point inside the school and a wave of
   * panic + light sweeps outward, the school tightens, then calms down.
   * Happens spontaneously (rate = params.startle); callable for fun.
   */
  startle(at?: Vec3): void {
    this.startlePocket(at);
    // While pinned to another mood a startle is just its pocket, so the
    // forced mood stays isolated.
    if (this.forced === null || this.forced === M_STARTLE) this.enterMood(M_STARTLE);
  }

  private startlePocket(at?: Vec3): void {
    let ex: number, ey: number, ez: number;
    if (at) {
      ex = at.x; ey = at.y; ez = at.z;
    } else {
      const i = (Math.random() * this.count) | 0;
      ex = this.pos[i * 3]; ey = this.pos[i * 3 + 1]; ez = this.pos[i * 3 + 2];
    }
    for (let i = 0; i < this.count; i++) {
      const dx = this.pos[i * 3] - ex;
      const dy = this.pos[i * 3 + 1] - ey;
      const dz = this.pos[i * 3 + 2] - ez;
      const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
      // Ragged pocket (see seedTurnVanguard): per-fish radius jitter.
      const R = 3.4 * (0.6 + 0.8 * Math.random());
      if (d > R) continue;
      const f = 0.25 + 0.75 * (1 - d / R);
      if (f > this.fear[i]) this.fear[i] = f;
      this.glow[i] = Math.max(this.glow[i], f * 0.8);
      if (d < 1.8 && Math.random() < 0.5) {
        this.igniteIndex(i, false, ex, ey, ez, 1);
      }
    }
  }

  /**
   * The strike: a flick of the pointer detonates a directional V-blast at
   * the lunge tip. Fish inside a forward-opening V part hard away from the
   * strike line, take contact-grade fear, and a wake of outward ignitions
   * lights the path. Depth uses the pointer column's squash.
   */
  private strike(cx: number, cy: number, cz: number, hx: number, hy: number, hz: number): void {
    const R = this.params.mouseRadius;
    const reach = R * GPU_TUNING.strikeReach;
    const apex = GPU_TUNING.strikeApexWidth;
    for (let i = 0; i < this.count; i++) {
      const i3 = i * 3;
      const dx = this.pos[i3] - cx;
      const dy = this.pos[i3 + 1] - cy;
      const dzS = (this.pos[i3 + 2] - cz) * 0.45;
      const along = dx * hx + dy * hy + dzS * hz;
      if (along <= -R * 0.2 || along >= reach) continue;
      const px = dx - along * hx;
      const py = dy - along * hy;
      const pz = dzS - along * hz;
      const perp = Math.sqrt(px * px + py * py + pz * pz);
      const t01 = Math.max(0, Math.min(1, along / reach));
      const sideR = R * (apex + (1 - apex) * t01); // the V opens forward
      if (perp >= sideR) continue;
      // Strongest at the apex, fading toward the tip and edges.
      const fall = (1 - perp / sideR) * (1 - 0.55 * t01);
      let ux: number, uy: number, uz: number;
      if (perp > 1e-3) {
        ux = px / perp; uy = py / perp; uz = pz / perp;
      } else {
        ux = -hy; uy = hx; uz = 0; // on the line: stable side
      }
      const blast = 9 * fall + 2;
      this.vel[i3] += ux * blast + hx * fall * 2;
      this.vel[i3 + 1] += uy * blast + hy * fall * 2;
      this.vel[i3 + 2] += (uz * blast + hz * fall * 2) * 0.6;
      this.fear[i] = Math.max(this.fear[i], 0.35 + 0.65 * fall);
      this.glow[i] = Math.max(this.glow[i], 0.5 + 0.5 * fall);
      if (Math.random() < 0.35 * fall + 0.08) {
        this.igniteIndex(i, false, cx, cy, cz, 1);
      }
    }
  }

  /**
   * Resting agitation: a pocket around a random fish takes fear and darts
   * outward; fear contagion carries the shiver on. No ignition or glow, so
   * movement comes before (and here without) light.
   */
  private spawnShiver(): void {
    // Arm the alarm: one pocket rarely lifts the school's mean fear past
    // the collapse threshold on its own, and the re-contraction as the
    // wave dies is the point of a shiver.
    this.alarm = Math.max(this.alarm, 0.12);
    this.alarmPeak = Math.max(this.alarmPeak, 0.12);
    const amp0 = 0.55 + 0.3 * Math.random();
    const rMul = (2 + Math.random()) * Math.max(0.5, this.params.perception);
    const c = (Math.random() * this.count) | 0;
    const ex = this.pos[c * 3];
    const ey = this.pos[c * 3 + 1];
    const ez = this.pos[c * 3 + 2];
    for (let i = 0; i < this.count; i++) {
      const i3 = i * 3;
      const dx = this.pos[i3] - ex;
      const dy = this.pos[i3 + 1] - ey;
      const dz = this.pos[i3 + 2] - ez;
      const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
      // Ragged pocket: jittered radius per fish.
      const r = rMul * (0.6 + 0.8 * Math.random());
      if (d >= r) continue;
      const amp = amp0 * (1 - d / r);
      if (amp > this.fear[i]) this.fear[i] = amp;
      // The epicenter fish has no outward direction: it darts ahead.
      let ux = this.headX, uy = this.headY, uz = this.headZ;
      if (d > 1e-3) {
        ux = dx / d; uy = dy / d; uz = dz / d;
      }
      const kick = 2.5 * amp;
      this.vel[i3] += ux * kick;
      this.vel[i3 + 1] += uy * kick;
      this.vel[i3 + 2] += uz * kick * 0.6;
    }
  }

  private igniteIndex(
    i: number,
    force: boolean,
    cx: number,
    cy: number,
    cz: number,
    sign: number,
  ): void {
    if (!force && (this.refract[i] > 0 || this.energy[i] > 0.6)) return;
    if (force) this.refract[i] = 0;
    this.energy[i] = 1;
    this.fired[i] = 1;
    this.pulse[i * 3] = cx;
    this.pulse[i * 3 + 1] = cy;
    this.pulse[i * 3 + 2] = cz;
    this.pulseSign[i] = sign;

    const kick = this.params.kick;
    if (kick <= 0) return;

    let kx = (this.pos[i * 3] - cx) * sign;
    let ky = (this.pos[i * 3 + 1] - cy) * sign;
    let kz = (this.pos[i * 3 + 2] - cz) * sign;
    const len = Math.sqrt(kx * kx + ky * ky + kz * kz);
    if (len > 1e-4) {
      kx /= len;
      ky /= len;
      kz /= len;
    } else {
      // The epicenter itself just shivers.
      const a = Math.random() * Math.PI * 2;
      const c = Math.random() * 2 - 1;
      const s = Math.sqrt(1 - c * c) * 0.4;
      kx = Math.cos(a) * s;
      ky = Math.sin(a) * s;
      kz = c * 0.4;
    }
    const m = kick * (0.7 + Math.random() * 0.6);
    this.vel[i * 3] += (kx + (Math.random() - 0.5) * 0.25) * m;
    this.vel[i * 3 + 1] += (ky + (Math.random() - 0.5) * 0.25) * m;
    this.vel[i * 3 + 2] += (kz + (Math.random() - 0.5) * 0.25) * m * 0.6;
    this.glow[i] = 1;
  }

  /** Spawn a spontaneous cascade at particle i — sometimes an implosion. */
  private igniteSpontaneous(i: number): void {
    const sign = Math.random() < this.params.inburst ? -1 : 1;
    this.igniteIndex(
      i,
      false,
      this.pos[i * 3],
      this.pos[i * 3 + 1],
      this.pos[i * 3 + 2],
      sign,
    );
  }

  // ------------------------------------------------------------------
  // Swarm director — the school's changing mind
  // ------------------------------------------------------------------

  private enterMood(mood: number): void {
    const P = this.params;
    const tempo = Math.max(0.25, P.tempo);
    const prev = this.mood;
    this.mood = mood;
    // Released from a contraction, the school pops outward (separation
    // 1.25) whatever mood it enters. The turn wave is not cleared here: it
    // outlives the turn mood so the school never snaps onto the heading.
    const popAfterContract = prev === M_CONTRACT && mood !== M_CONTRACT;
    this.tgtSep = 1;
    this.tgtHerd = 1;

    switch (mood) {
      case M_CRUISE:
        this.moodTimer = (7 + Math.random() * 8) / tempo;
        this.tgtAlign = 1.1;
        this.tgtCoh = 1.0;
        this.tgtMill = 0;
        this.tgtDrive = 1.0;
        this.tgtSpeed = 0.5 + Math.random() * 0.35;
        // Sometimes ease into a slightly new direction.
        if (Math.random() < 0.45) this.pickNewHeading(10, 40);
        break;
      case M_TURN: {
        this.turnsInARow++;
        // The wave's crossing time plus a beat to settle, not a tempo-
        // scaled constant: a turn that ends before the front has crossed
        // snaps the rest of the school and reads as a second turn.
        this.moodTimer =
          (2.4 * this.rmsRadius) / GPU_TUNING.turnWaveSpeed + 0.8 + Math.random() * 0.6;
        this.tgtAlign = 1.6;
        this.tgtCoh = 0.95;
        this.tgtMill = 0;
        // Softer than v3's 1.5: alignment carries the wave, the will
        // doesn't swing the whole school at once.
        this.tgtDrive = 1.2;
        this.tgtSpeed = 0.95;
        this.turnOldX = this.headX;
        this.turnOldY = this.headY;
        this.turnOldZ = this.headZ;
        this.pickNewHeading(70, 155);
        const leader = (Math.random() * this.count) | 0;
        this.turnIndex = leader;
        this.turnRadius = 0;
        this.turnActive = 1;
        this.seedTurnVanguard(leader);
        break;
      }
      case M_MILL:
        this.turnsInARow = 0;
        this.moodTimer = (6 + Math.random() * 7) / tempo;
        this.tgtAlign = 1.25;
        this.tgtCoh = 1.05;
        this.tgtMill = 1.1;
        this.tgtDrive = 0.4;
        this.tgtSpeed = 0.5;
        break;
      case M_HOVER:
        this.turnsInARow = 0;
        this.moodTimer = (3.5 + Math.random() * 5) / tempo;
        this.tgtAlign = 0.85;
        this.tgtCoh = 1.15;
        this.tgtMill = 0.08;
        this.tgtDrive = 0.15;
        this.tgtSpeed = 0.14;
        break;
      case M_STARTLE:
        this.moodTimer = 1.3 + Math.random() * 0.5;
        this.tgtAlign = 1.9;
        this.tgtCoh = 1.0;
        this.tgtMill = 0;
        this.tgtDrive = 0.9;
        this.tgtSpeed = 1.0;
        break;
      case M_CONTRACT: {
        // The school draws in on itself: cohesion swells, the shared will
        // slackens, a slight swirl keeps the condensing body alive.
        // Personal space never gives (relaxing it crushed the GPU school
        // into one glowing mass); arrival damping bounds the squeeze.
        const c = P.collapse;
        this.tgtAlign = 0.7;
        this.tgtMill = 0.25;
        this.tgtDrive = 0.25;
        if (this.contractFromAlarm) {
          // After a scare the whole school balls up: shorter, harder, and
          // the herd drive surges.
          this.contractFromAlarm = false;
          this.moodTimer = (1.6 + Math.random() * 1.0) / tempo;
          this.tgtCoh = 1.0 + 1.4 * c;
          this.tgtSpeed = 0.45;
          this.tgtHerd = 1 + 2.5 * c;
        } else {
          this.moodTimer = (2.2 + Math.random() * 1.8) / tempo;
          this.tgtCoh = 1.0 + 0.9 * c;
          this.tgtSpeed = 0.35;
          this.tgtHerd = 1 + 1.2 * c;
        }
        break;
      }
    }
    if (popAfterContract) this.tgtSep = 1.25;
  }

  private pickNextMood(): number {
    // Collapse diverts a share of cruise/mill exits into a contraction.
    // No draw at 0: v3's FSM, untouched.
    const c = this.params.collapse;
    if (
      c > 0 &&
      (this.mood === M_CRUISE || this.mood === M_MILL) &&
      Math.random() < Math.min(0.5, 0.2 * c)
    ) {
      return M_CONTRACT;
    }
    const r = Math.random();
    switch (this.mood) {
      case M_CRUISE:
        // Turns chain through cruise; after two in a row the school does
        // something else first (back-to-back turns read as disruptive).
        if (this.turnsInARow >= 2) {
          this.turnsInARow = 0;
          return r < 0.5 ? M_MILL : M_HOVER;
        }
        return r < 0.5 ? M_TURN : r < 0.72 ? M_MILL : M_HOVER;
      case M_TURN:
        return M_CRUISE;
      case M_MILL:
        return r < 0.55 ? M_TURN : r < 0.75 ? M_HOVER : M_CRUISE;
      case M_HOVER:
        return r < 0.55 ? M_CRUISE : M_TURN;
      case M_CONTRACT:
        // Release into cruise, not a turn: contract → turn → cruise → turn
        // chained into near-continuous turning.
        return M_CRUISE;
      default:
        return M_CRUISE;
    }
  }

  private pickNewHeading(minDeg: number, maxDeg: number): void {
    const ang =
      ((minDeg + Math.random() * (maxDeg - minDeg)) * Math.PI) / 180 *
      (Math.random() < 0.5 ? -1 : 1);
    const c = Math.cos(ang);
    const s = Math.sin(ang);
    const hx = this.headX * c - this.headY * s;
    const hy = this.headX * s + this.headY * c;
    // Pitch scales with roamDepth (1 = v3's ±0.3).
    const zd = Math.max(0, Math.min(1, this.params.roamDepth ?? 1));
    let hz = this.headZ + (Math.random() - 0.5) * 0.4 * zd;
    hz = Math.max(-0.3 * zd, Math.min(0.3 * zd, hz));
    const l = Math.sqrt(hx * hx + hy * hy + hz * hz) || 1;
    this.headX = hx / l;
    this.headY = hy / l;
    this.headZ = hz / l;
  }

  /**
   * Seed a turn through a vanguard: one pocket darts onto the new heading
   * and the wave front grows from its leader. A ragged pocket with a soft
   * falloff and no glow: a hard disc darting and lighting up together read
   * as "a perfect circle turning" once the school became a sheet.
   */
  private seedTurnVanguard(leader: number): void {
    const lx = this.pos[leader * 3];
    const ly = this.pos[leader * 3 + 1];
    const lz = this.pos[leader * 3 + 2];
    for (let i = 0; i < this.count; i++) {
      const dx = this.pos[i * 3] - lx;
      const dy = this.pos[i * 3 + 1] - ly;
      const dz = this.pos[i * 3 + 2] - lz;
      const rv = 3.5 * (0.55 + 0.9 * Math.random());
      const dv = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (dv > rv) continue;
      const w = 1 - dv / rv;
      const dart = (1.9 + Math.random() * 1.0) * (0.4 + 0.6 * w);
      this.vel[i * 3] += this.headX * dart;
      this.vel[i * 3 + 1] += this.headY * dart;
      this.vel[i * 3 + 2] += this.headZ * dart * 0.6;
      this.gaitClock[i] = 0;
    }
  }

  private updateDirector(dt: number, homeX: number, homeY: number): void {
    const P = this.params;

    // Spontaneous false alarms.
    this.startleClock -= dt;
    if (this.startleClock <= 0) {
      this.startleClock = (14 + Math.random() * 26) / Math.max(0.05, P.startle);
      if (P.startle > 0.02 && this.mood !== M_STARTLE) this.startle();
    }

    // Resting shivers: per-fish fear and a dart, never a mood change.
    const shiver = P.shiver ?? 0;
    if (shiver > 0.02) {
      this.shiverClock -= dt;
      if (this.shiverClock <= 0) {
        this.shiverClock = (6 + Math.random() * 6) / Math.max(0.05, shiver);
        this.spawnShiver();
      }
    }

    // The front keeps growing after the turn mood ends until it has
    // crossed the school; a new turn re-arms it from zero.
    if (this.turnActive > 0) {
      this.turnRadius += GPU_TUNING.turnWaveSpeed * dt;
      if (this.turnRadius > 3.0 * this.rmsRadius + 2) this.turnActive = 0;
    }
    this.moodTimer -= dt;
    if (this.moodTimer <= 0) {
      // Pinned: re-enter so turn keeps turning and contract keeps contracting.
      if (this.forced !== null) this.enterForced();
      else this.enterMood(this.mood === M_STARTLE ? M_CRUISE : this.pickNextMood());
    }

    // Alarm collapse: once a real scare subsides (alarm under half its
    // recent peak), the school collapses in on itself. Unreachable at
    // collapse 0.
    this.alarm += (this.meanFear - this.alarm) * (1 - Math.exp(-dt / 0.15));
    this.alarmPeak += (this.alarm - this.alarmPeak) * (1 - Math.exp(-dt));
    if (this.alarm > this.alarmPeak) this.alarmPeak = this.alarm;
    this.alarmCooldown -= dt;
    if (
      P.collapse > 0 &&
      this.forced === null &&
      this.alarmPeak > 0.08 &&
      this.alarm < 0.5 * this.alarmPeak &&
      this.mood !== M_CONTRACT &&
      this.mood !== M_STARTLE &&
      this.alarmCooldown <= 0
    ) {
      // A running turn finishes first: a contraction cutting it short read
      // as a new turn starting before the last one ended.
      this.alarmPending = true;
      this.alarmPeak = this.alarm;
      this.alarmCooldown = 4;
    }
    if (
      this.alarmPending &&
      this.mood !== M_TURN &&
      this.mood !== M_CONTRACT &&
      this.mood !== M_STARTLE
    ) {
      this.alarmPending = false;
      this.contractFromAlarm = true;
      this.enterMood(M_CONTRACT);
    }

    // Bend the heading toward home when the school strays — it turns back
    // because it "decides" to, no invisible walls.
    const hx = homeX - this.centroidX;
    const hy = homeY - this.centroidY;
    const hd = Math.sqrt(hx * hx + hy * hy);
    const R = P.containRadius;
    if (hd > R * 0.4) {
      const k = Math.min(1, ((hd - R * 0.4) / R) * dt * 2.8);
      const nx = this.headX + (hx / hd - this.headX) * k;
      const ny = this.headY + (hy / hd - this.headY) * k;
      const nz = this.headZ * (1 - k * 0.5);
      const l = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1;
      this.headX = nx / l;
      this.headY = ny / l;
      this.headZ = nz / l;
    }

    // Ease the live multipliers toward the mood's targets.
    const k = 1 - Math.exp(-dt * 1.6);
    this.dirAlign += (this.tgtAlign - this.dirAlign) * k;
    this.dirCoh += (this.tgtCoh - this.dirCoh) * k;
    this.dirMill += (this.tgtMill - this.dirMill) * k;
    this.dirDrive += (this.tgtDrive - this.dirDrive) * k;
    this.dirSpeed += (this.tgtSpeed - this.dirSpeed) * k;
    // Snap home so an idle lever leaves separation and herd at exactly the
    // user's weight (the ease alone only approaches it).
    this.dirSep += (this.tgtSep - this.dirSep) * k;
    if (Math.abs(this.tgtSep - this.dirSep) < 1e-4) this.dirSep = this.tgtSep;
    this.dirHerd += (this.tgtHerd - this.dirHerd) * k;
    if (Math.abs(this.tgtHerd - this.dirHerd) < 1e-4) this.dirHerd = this.tgtHerd;
  }

  // ------------------------------------------------------------------
  // Wanderers
  // ------------------------------------------------------------------

  private spawnWanderGroup(): void {
    const P = this.params;
    let leader = -1;
    for (let tries = 0; tries < 12; tries++) {
      const i = (Math.random() * this.count) | 0;
      if (this.mode[i] === FLOCK) {
        leader = i;
        break;
      }
    }
    if (leader < 0) return;

    const lx = this.pos[leader * 3];
    const ly = this.pos[leader * 3 + 1];
    const lz = this.pos[leader * 3 + 2];

    // Head roughly outward from the centroid, with some tangent wobble.
    let dx = lx - this.centroidX + (Math.random() - 0.5) * 2;
    let dy = ly - this.centroidY + (Math.random() - 0.5) * 2;
    let dz = (lz - this.centroidZ) * 0.6 + (Math.random() - 0.5) * 1.5;
    const dl = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1;
    dx /= dl;
    dy /= dl;
    dz /= dl;

    const duration = 6 + Math.random() * 7;
    // Large groups recruit a bigger pocket so they leave as a sub-school,
    // not a thinned-out cloud (v3's 2.8 at its sizes). groupSize is read
    // as a fish count here; the GPU reads it per 2,600 fish.
    const recruitR = 2.8 + 2.0 * Math.min(1, P.groupSize / 200);
    const recruitR2 = recruitR * recruitR;
    let recruited = 0;
    const cap = Math.max(1, Math.round(P.groupSize));

    for (let i = 0; i < this.count && recruited < cap; i++) {
      if (this.mode[i] !== FLOCK) continue;
      const ddx = this.pos[i * 3] - lx;
      const ddy = this.pos[i * 3 + 1] - ly;
      const ddz = this.pos[i * 3 + 2] - lz;
      if (i !== leader && ddx * ddx + ddy * ddy + ddz * ddz > recruitR2) continue;

      this.mode[i] = WANDER;
      this.modeTimer[i] = duration * (0.85 + Math.random() * 0.3);
      // Shared heading with individual variation, so the group stays loose.
      let wx = dx + (Math.random() - 0.5) * 0.35;
      let wy = dy + (Math.random() - 0.5) * 0.35;
      let wz = dz + (Math.random() - 0.5) * 0.35;
      const wl = Math.sqrt(wx * wx + wy * wy + wz * wz) || 1;
      this.wanderDir[i * 3] = wx / wl;
      this.wanderDir[i * 3 + 1] = wy / wl;
      this.wanderDir[i * 3 + 2] = wz / wl;
      recruited++;
    }
  }

  // ------------------------------------------------------------------
  // Spatial grid
  // ------------------------------------------------------------------

  /** Clamped cell coordinate along one axis. */
  private static cellCoord(v: number, invCell: number): number {
    const c = Math.floor(v * invCell) + (D >> 1);
    return c < 0 ? 0 : c >= D ? D - 1 : c;
  }

  private buildGrid(invCell: number): void {
    const n = this.count;
    for (let i = 0; i < n; i++) {
      this.cellOf[i] =
        ParticleSystem.cellCoord(this.pos[i * 3], invCell) +
        ParticleSystem.cellCoord(this.pos[i * 3 + 1], invCell) * D +
        ParticleSystem.cellCoord(this.pos[i * 3 + 2], invCell) * D2;
      this.fineCellOf[i] =
        ParticleSystem.fineCoord(this.pos[i * 3], FX) +
        ParticleSystem.fineCoord(this.pos[i * 3 + 1], FY) * FX +
        ParticleSystem.fineCoord(this.pos[i * 3 + 2], FZ) * FXY;
    }
    this.gridTouchedN = sortIntoCells(
      n, this.cellOf, this.gridCount, this.gridStart, this.gridEntries,
      this.gridTouched, this.gridTouchedN,
    );
    this.fineTouchedN = sortIntoCells(
      n, this.fineCellOf, this.fineCount, this.fineStart, this.fineEntries,
      this.fineTouched, this.fineTouchedN,
    );
  }

  /** Clamped fine-grid coordinate along one axis. */
  private static fineCoord(v: number, dim: number): number {
    const c = Math.floor(v * F_INV) + (dim >> 1);
    return c < 0 ? 0 : c >= dim ? dim - 1 : c;
  }

  /**
   * Crowd dim: fade each fish by (nRef / nLocal)^(0.8·crowdDim), where
   * nLocal is the fish count around it and nRef the school's mean, so only
   * local excess dims and a contracted school keeps its brightness.
   *
   * nLocal must be position-continuous or the fade bands along grid
   * planes: each cell's count is weighted by a separable tent of its
   * center's offset from the fish (in cells, w = 0 at |d| = 1.5), the same
   * stencil as the GPU sprite shader, so a face crossing swaps slabs at
   * zero weight. Fish outside the grid keep full brightness and stay out
   * of the mean.
   */
  private updateCrowdFade(dt: number): void {
    const dim = this.params.crowdDim ?? 0;
    const fade = this.fade;
    const attr = this.geometry.attributes.aFade as THREE.BufferAttribute;
    if (!(dim > 0)) {
      if (this.crowdOn) {
        // The whole buffer, dormant slots too: with crowd dim off nothing
        // re-uploads a fish's fade when it wakes.
        fade.fill(1);
        attr.clearUpdateRanges();
        attr.needsUpdate = true;
        this.crowdOn = false;
        this.nRef = -1;
      }
      return;
    }
    this.crowdOn = true;
    const n = this.count;
    const pos = this.pos;
    const inv = 1 / (Math.max(0.5, this.params.perception) * 0.5);
    const hx = CX >> 1;
    const hy = CY >> 1;
    const hz = CZ >> 1;
    const counts = this.crowdCount;
    const touched = this.crowdTouched;
    for (let k = 0; k < this.crowdTouchedN; k++) counts[touched[k]] = 0;
    let tn = 0;
    for (let i = 0; i < n; i++) {
      const cx = Math.floor(pos[i * 3] * inv) + hx;
      const cy = Math.floor(pos[i * 3 + 1] * inv) + hy;
      const cz = Math.floor(pos[i * 3 + 2] * inv) + hz;
      if (cx < 0 || cx >= CX || cy < 0 || cy >= CY || cz < 0 || cz >= CZ) continue;
      const c = cx + cy * CX + cz * CXY;
      if (counts[c]++ === 0) touched[tn++] = c;
    }
    this.crowdTouchedN = tn;

    // Pass 1: nLocal into fade (as scratch, -1 = outside the grid) and the
    // school's mean.
    let sum = 0;
    let inGrid = 0;
    for (let i = 0; i < n; i++) {
      const gx = pos[i * 3] * inv + hx;
      const gy = pos[i * 3 + 1] * inv + hy;
      const gz = pos[i * 3 + 2] * inv + hz;
      const fx = Math.floor(gx);
      const fy = Math.floor(gy);
      const fz = Math.floor(gz);
      if (fx < 0 || fx >= CX || fy < 0 || fy >= CY || fz < 0 || fz >= CZ) {
        fade[i] = -1;
        continue;
      }
      const rx = gx - fx;
      const ry = gy - fy;
      const rz = gz - fz;
      let wSum = 0;
      let nSum = 0;
      for (let oz = -1; oz <= 1; oz++) {
        const cz = fz + oz;
        if (cz < 0 || cz >= CZ) continue;
        const wz = Math.max(0, 1 - Math.abs(oz + 0.5 - rz) / 1.5);
        for (let oy = -1; oy <= 1; oy++) {
          const cy = fy + oy;
          if (cy < 0 || cy >= CY) continue;
          const wzy = wz * Math.max(0, 1 - Math.abs(oy + 0.5 - ry) / 1.5);
          const row = cy * CX + cz * CXY;
          for (let ox = -1; ox <= 1; ox++) {
            const cx = fx + ox;
            if (cx < 0 || cx >= CX) continue;
            const w = wzy * Math.max(0, 1 - Math.abs(ox + 0.5 - rx) / 1.5);
            nSum += w * counts[cx + row];
            wSum += w;
          }
        }
      }
      const nLocal = (nSum / Math.max(wSum, 1e-6)) * 27;
      fade[i] = nLocal;
      sum += nLocal;
      inGrid++;
    }
    // One-pole smoothed (0.5 s) so the reference never flickers.
    const nRefNow = sum / Math.max(1, inGrid);
    this.nRef = this.nRef < 0 ? nRefNow : this.nRef + (nRefNow - this.nRef) * (1 - Math.exp(-dt / 0.5));

    // Pass 2: the fade itself. Sparse fish are never boosted.
    const k = 0.8 * dim;
    const nRef = this.nRef;
    for (let i = 0; i < n; i++) {
      fade[i] = fade[i] < 0 ? 1 : Math.pow(Math.min(1, nRef / Math.max(fade[i], 1)), k);
    }
    this.markLive(attr);
  }

  // ------------------------------------------------------------------
  // Main update
  // ------------------------------------------------------------------

  update(deltaTime: number): void {
    const P = this.params;
    // The panel writes params directly, so the share is polled. Ahead of
    // the dt check so a paused school still recolors.
    if (P.accentShare !== this.lastAccentShare) this.applyPaletteColors();
    const dt = Math.min(deltaTime, 0.05) * P.timeScale;
    if (dt <= 0) return;
    this.time += dt;
    this.frameNo = (this.frameNo + 1) >>> 0;

    const t = this.time;
    const n = this.count;
    const pos = this.pos;
    const vel = this.vel;
    const energy = this.energy;
    const glow = this.glow;
    const fear = this.fear;

    // Frame constants
    const perception = Math.max(0.5, P.perception);
    const perception2 = perception * perception;
    const sepR = perception * TUNING.separationRatio;
    const arriveR = perception * TUNING.arriveRatio;
    const pushR = Math.min(sepR * GPU_TUNING.pairPushRange, GPU_TUNING.fineCellSize);
    const pushR2 = pushR * pushR;
    const invCell = 1 / perception;
    const dragF = Math.exp(-P.drag * dt);
    const energyDecay = Math.exp(-P.flashDecay * dt);
    const fearDecayF = Math.exp(-P.fearDecay * dt);
    const fearSpreadDt = Math.min(1, P.fearSpread * dt);
    const glowK = Math.min(1, dt * 5);
    const breatheR = P.containRadius * (1 + P.breathe * Math.sin(t * 0.45));
    const farR = breatheR * 2.1;
    const kx = P.flowScale;
    const ky = P.flowScale * 1.33;
    const kz = P.flowScale * 0.77;
    const ft = t * 0.6;
    const maxSpeedBase = P.maxSpeed;
    const mouseR = P.mouseRadius;
    const pointerOn = this.pointerActive;
    const px = this.pointer.x;
    const py = this.pointer.y;
    const pz = this.pointer.z;
    const splashBoost = Math.min(2.5, this.mouseSpeed * P.splash * 0.6);
    const igniteRate = this.mouseSpeed * P.splash * 1.2;

    // The levers (each 0 = v3).
    const shape = P.shape ?? 0;
    const sheet = P.sheet ?? 0;
    const herd = P.herd ?? 0;
    const rally = P.rally ?? 0;
    const evade = P.evade ?? 0;
    const flight = P.flight ?? 0;
    // Below 1: smoothstep's edges must not meet.
    const bite = Math.min(0.9, Math.max(0, P.bite ?? 0));
    // The "still threatened" band shrinks with bite: a crisp predator must
    // not hold a wide halo open around a resting cursor. v3's 1.4 at 0.
    const threatBand = 1.4 - 0.4 * bite;

    this.mouseSpeed *= Math.exp(-2.5 * dt);

    // World-space pointer velocity (smoothed) — the predator's heading.
    if (pointerOn) {
      if (this.hadPointerFrame) {
        const k = Math.min(1, dt * 8);
        this.pointerVel.x += ((px - this.prevPointerW.x) / dt - this.pointerVel.x) * k;
        this.pointerVel.y += ((py - this.prevPointerW.y) / dt - this.pointerVel.y) * k;
        this.pointerVel.z += ((pz - this.prevPointerW.z) / dt - this.pointerVel.z) * k;
        if (this.pointerVel.lengthSq() > 80 * 80) this.pointerVel.setLength(80);
      }
      this.prevPointerW.set(px, py, pz);
      this.hadPointerFrame = true;
    } else {
      this.hadPointerFrame = false;
      this.pointerVel.multiplyScalar(Math.exp(-4 * dt));
    }
    const pvm = this.pointerVel.length();
    const vhx = pvm > 1e-3 ? this.pointerVel.x / pvm : 0;
    const vhy = pvm > 1e-3 ? this.pointerVel.y / pvm : 0;
    const vhz = pvm > 1e-3 ? this.pointerVel.z / pvm : 0;
    // The danger zone leads a moving predator — prey react to its approach.
    const lead = Math.min(pvm * 0.22, mouseR * 0.9);
    const tpx = px + vhx * lead;
    const tpy = py + vhy * lead;
    const tpz = pz + vhz * lead;

    // The strike: a flick (rest bursting into speed) detonates a V-blast
    // at the lunge tip. Sustained fast movement never triggers it.
    this.strikeCooldown -= dt;
    if (pvm < GPU_TUNING.strikeSpeed * 0.45) this.lastSlowTime = t;
    if (
      pointerOn &&
      pvm > GPU_TUNING.strikeSpeed &&
      t - this.lastSlowTime < GPU_TUNING.strikeFlickWindow &&
      this.strikeCooldown <= 0
    ) {
      this.strikeCooldown = GPU_TUNING.strikeCooldown;
      this.strike(tpx, tpy, tpz, vhx, vhy, vhz);
    }

    // Roaming home region: a slow-wandering anchor the school patrols around.
    const roamR = (P.containRadius * 0.75 * P.roam) / 1.5;
    const targetX =
      (Math.sin(t * 0.13) + 0.5 * Math.sin(t * 0.23 + 2.1)) * roamR;
    const targetY =
      (Math.sin(t * 0.107 + 1.3) + 0.5 * Math.sin(t * 0.19)) * roamR * 0.8;
    const targetZ = Math.sin(t * 0.147 + 4.2) * roamR * 0.45
      * Math.max(0, Math.min(1, P.roamDepth ?? 1));

    // Evade: the school dodges a parked predator as a body. Its herd center
    // and roam target shift away from the pointer and the heading bends off
    // it, so the threat ends up at the edge with the body streaming past.
    let evW = 0;
    let evNX = 0;
    let evNY = 0;
    let evNZ = 0;
    if (pointerOn && evade > 0) {
      const dx = this.centroidX - px;
      const dy = this.centroidY - py;
      const dz = this.centroidZ - pz;
      const dist = Math.sqrt(dx * dx + dy * dy + dz * 0.45 * (dz * 0.45));
      // Full strength until the pointer is outside the school, then a short
      // ramp: a linear fade from the center settled with the cursor still
      // inside the edge.
      const edge = Math.max(2, this.rmsRadius * 1.6);
      evW = evade * Math.min(1, Math.max(0, (edge * 1.25 - dist) / (edge * 0.25)));
      if (dist < 0.3) {
        // Pointer on the centroid: step aside across the heading.
        const hl = Math.hypot(this.headX, this.headY) || 1;
        evNX = -this.headY / hl;
        evNY = this.headX / hl;
      } else {
        const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
        evNX = dx / len;
        evNY = dy / len;
        evNZ = dz / len;
      }
    }
    const evTarget = P.containRadius * 0.9 * evW;
    // One pole (0.35 s) so the center never jumps.
    const ek = 1 - Math.exp(-dt / 0.35);
    this.evadeX += (evNX * evTarget - this.evadeX) * ek;
    this.evadeY += (evNY * evTarget - this.evadeY) * ek;
    this.evadeZ += (evNZ * evTarget - this.evadeZ) * ek;
    if (
      !(evade > 0) &&
      Math.abs(this.evadeX) + Math.abs(this.evadeY) + Math.abs(this.evadeZ) < 1e-4
    ) {
      this.evadeX = this.evadeY = this.evadeZ = 0;
    }
    const roamX = targetX + this.evadeX;
    const roamY = targetY + this.evadeY;
    const roamZ = targetZ + this.evadeZ;

    // The school's mind: moods, heading, false alarms.
    this.updateDirector(dt, targetX, targetY);

    // Bend the heading away from the predator, like the home bend.
    if (evW > 0) {
      const xl = Math.hypot(evNX, evNY);
      if (xl > 1e-6) {
        const k = Math.min(1, evW * dt * 4);
        const nx = this.headX + (evNX / xl - this.headX) * k;
        const ny = this.headY + (evNY / xl - this.headY) * k;
        const nz = this.headZ * (1 - k * 0.5);
        const l = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1;
        this.headX = nx / l;
        this.headY = ny / l;
        this.headZ = nz / l;
      }
    }

    const headX = this.headX;
    const headY = this.headY;
    const headZ = this.headZ;
    const dirAlign = this.dirAlign;
    const dirCoh = this.dirCoh;
    const dirDrive = this.dirDrive;
    const dirSpeed = this.dirSpeed;
    const driveW = P.schoolDrive * dirDrive;
    const gaitRate = 0.55 + dirSpeed * 1.3;
    const millF = (P.mill + this.dirMill) * 2.6;
    // The director's separation gain: 1 except the release pop.
    const sepW = P.separation * this.dirSep;
    const herdGain = this.dirHerd;

    // Group-level recentering: gentle, the cruise dynamics do the framing.
    const reCenterX = (roamX - this.centroidX) * 0.22;
    const reCenterY = (roamY - this.centroidY) * 0.22;
    const reCenterZ = (roamZ - this.centroidZ) * 0.22;
    const cenX = this.centroidX;
    const cenY = this.centroidY;
    const cenZ = this.centroidZ;
    // Herd/rally/regroup aim here: with evade on, the center shifted off
    // a parked predator, so the school pulls away from the threat instead
    // of into it.
    const evOn = evade > 0.001;
    const hcX = evOn ? cenX + this.evadeX : cenX;
    const hcY = evOn ? cenY + this.evadeY : cenY;
    const hcZ = evOn ? cenZ + this.evadeZ : cenZ;

    // The turn wave: until the front (a sphere growing from the vanguard
    // fish) reaches a fish it keeps the old heading, so the turn is led by
    // a small group and spreads by alignment. Last frame's leader position.
    const turnOn = this.turnActive > 0 && this.turnIndex < n;
    const tcx = turnOn ? pos[this.turnIndex * 3] : 0;
    const tcy = turnOn ? pos[this.turnIndex * 3 + 1] : 0;
    const tcz = turnOn ? pos[this.turnIndex * 3 + 2] : 0;
    const turnR2 = this.turnRadius * this.turnRadius;
    const oldHX = this.turnOldX;
    const oldHY = this.turnOldY;
    const oldHZ = this.turnOldZ;

    this.buildGrid(invCell);
    // Snapshot before this frame's ignitions, so a seed lights its
    // neighbors next frame, as on the GPU.
    this.posPrev.set(pos.subarray(0, n * 3));
    this.velPrev.set(vel.subarray(0, n * 3));
    this.energyPrev.set(energy.subarray(0, n));
    this.fearPrev.set(fear.subarray(0, n));
    this.pulsePrev.set(this.pulse.subarray(0, n * 3));
    this.pulseSignPrev.set(this.pulseSign.subarray(0, n));
    const posPrev = this.posPrev;
    const velPrev = this.velPrev;
    const energyPrev = this.energyPrev;
    const fearPrev = this.fearPrev;

    // Spontaneous flashes: the swarm glimmers on its own.
    let spontAcc = P.spontaneous * dt;
    while (spontAcc > 0) {
      if (Math.random() < spontAcc) {
        this.igniteSpontaneous((Math.random() * n) | 0);
      }
      spontAcc -= 1;
    }

    // Occasionally a small group breaks away.
    if (P.wanderRate > 0.001) {
      this.wanderClock -= dt;
      if (this.wanderClock <= 0) {
        const interval = 18 - 15 * Math.min(1, P.wanderRate);
        this.wanderClock = interval * (0.6 + Math.random() * 0.8);
        this.spawnWanderGroup();
      }
    }

    // Fresh sampling order every frame, so no subset is favored over time.
    const frameHash = pcg(Math.imul(this.frameNo, 2654435761) >>> 0);

    let cSumX = 0;
    let cSumY = 0;
    let cSumZ = 0;
    let cNum = 0;
    let spreadSum = 0;
    let fearSum = 0;

    for (let i = 0; i < n; i++) {
      const i3 = i * 3;
      const x = pos[i3];
      const y = pos[i3 + 1];
      const z = pos[i3 + 2];
      let vx = vel[i3];
      let vy = vel[i3 + 1];
      let vz = vel[i3 + 2];
      const m = this.mode[i];
      const social = this.social[i];
      const spdF = this.speedFactor[i];
      const seed = this.seeds[i];

      let mhX = headX, mhY = headY, mhZ = headZ;
      if (turnOn) {
        const ddx = x - tcx;
        const ddy = y - tcy;
        const ddz = z - tcz;
        if (ddx * ddx + ddy * ddy + ddz * ddz > turnR2) {
          mhX = oldHX; mhY = oldHY; mhZ = oldHZ;
        }
      }

      // ---- Neighbor pass: boids accumulators + contagion pressure ----
      let sepX = 0, sepY = 0, sepZ = 0;
      let cohX = 0, cohY = 0, cohZ = 0;
      let alnX = 0, alnY = 0, alnZ = 0;
      let found = 0;
      let contagion = 0;
      let bestJ = -1; // strongest glowing neighbor — its cascade is inherited
      let bestW = 0;
      let maxFear = 0;

      const cx = ParticleSystem.cellCoord(x, invCell);
      const cy = ParticleSystem.cellCoord(y, invCell);
      const cz = ParticleSystem.cellCoord(z, invCell);

      // Stratified fair sample (the GPU kernel's scheme, see wgsl-sim.ts):
      // v3 took the first 16 in fixed cell order, so the sample averaged
      // over cells, not fish. That makes cohesion/alignment depend on where
      // a fish sits inside its cell, a static force with the grid pitch
      // that banded the school along the grid planes. Each cell gets a
      // candidate quota proportional to its population (dithered so the
      // expectation is exact) from a hashed in-cell start, so the candidates
      // are a uniform sample of the 27-cell block.
      // No own-cell priority: any position-quantized bias compounds.
      let total = 0;
      for (let oi = 0; oi < 27; oi++) {
        const gx = cx + OFF_X[oi];
        const gy = cy + OFF_Y[oi];
        const gz = cz + OFF_Z[oi];
        if (gx < 0 || gx >= D || gy < 0 || gy >= D || gz < 0 || gz >= D) {
          nbCell[oi] = -1;
          continue;
        }
        const cell = gx + gy * D + gz * D2;
        nbCell[oi] = cell;
        total += this.gridCount[cell];
      }
      // Unlike the GPU kernel, every candidate is examined and the 16
      // accepts are a reservoir sample of all in-range ones. In a flattened
      // school (olle-work's sheet, ~33 of 96 candidates in range) the cap
      // binds about halfway, and stopping there kept the first cells of the
      // walk: still ±5% banding. Hashes are masked to 30 bits so V8 keeps `%` in
      // integer registers (a full u32 is a double and `%` becomes fmod).
      let inRange = 0;
      if (total > 0) {
        const candScale = SAMPLE_CAND / total;
        let rs = pcg((i ^ frameHash) >>> 0);
        for (let oi = 0; oi < 27; oi++) {
          const cell = nbCell[oi];
          if (cell < 0) continue;
          const cnt = this.gridCount[cell];
          if (cnt === 0) continue;
          const h = pcg((rs ^ cell) >>> 0);
          const dither = (pcg(h) >>> 8) * (1 / 16777216);
          const quota = Math.min(cnt, Math.floor(cnt * candScale + dither));
          const start = this.gridStart[cell];
          const off = (h & 0x3fffffff) % cnt;
          for (let e = 0; e < quota; e++) {
            const oe = off + e;
            const j = this.gridEntries[start + (oe < cnt ? oe : oe - cnt)];
            if (j === i) continue;
            const j3 = j * 3;
            const dx = posPrev[j3] - x;
            const dy = posPrev[j3 + 1] - y;
            const dz = posPrev[j3 + 2] - z;
            if (dx * dx + dy * dy + dz * dz > perception2) continue;
            if (inRange < MAX_NEIGHBORS) {
              nbJ[inRange] = j;
            } else {
              rs = pcg(rs);
              const slot = (rs & 0x3fffffff) % (inRange + 1);
              if (slot < MAX_NEIGHBORS) nbJ[slot] = j;
            }
            inRange++;
          }
        }
      }
      const accepted = inRange < MAX_NEIGHBORS ? inRange : MAX_NEIGHBORS;
      for (let k = 0; k < accepted; k++) {
        const j = nbJ[k];
        const j3 = j * 3;
        const dx = posPrev[j3] - x;
        const dy = posPrev[j3 + 1] - y;
        const dz = posPrev[j3 + 2] - z;
        const d = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1e-4;
        if (d < sepR) {
          const w = 1 - d / sepR;
          const inv = (w * w) / d;
          sepX -= dx * inv;
          sepY -= dy * inv;
          sepZ -= dz * inv;
        }
        cohX += posPrev[j3];
        cohY += posPrev[j3 + 1];
        cohZ += posPrev[j3 + 2];
        alnX += velPrev[j3];
        alnY += velPrev[j3 + 1];
        alnZ += velPrev[j3 + 2];
        found++;

        if (energyPrev[j] > TUNING.contagiousAbove) {
          // Weighted by the source's remaining heat, as on the GPU: the
          // radial kick flies a lit fish a perception radius while it is
          // still contagious, and at full weight those spark carriers keep
          // re-seeding the burn so cascades never die.
          const srcHeat =
            (energyPrev[j] - TUNING.contagiousAbove) / (1 - TUNING.contagiousAbove);
          const w = (1 - d / perception) * srcHeat;
          contagion += w;
          if (w > bestW) {
            bestW = w;
            bestJ = j;
          }
        }
        if (fearPrev[j] > maxFear) maxFear = fearPrev[j];
      }

      // ---- Fine collision grid: continuous close-range personal space ----
      let hpX = 0, hpY = 0, hpZ = 0;
      {
        const fx = ParticleSystem.fineCoord(x, FX);
        const fy = ParticleSystem.fineCoord(y, FY);
        const fz = ParticleSystem.fineCoord(z, FZ);
        for (let oz = fz > 0 ? fz - 1 : 0; oz <= fz + 1 && oz < FZ; oz++) {
          for (let oy = fy > 0 ? fy - 1 : 0; oy <= fy + 1 && oy < FY; oy++) {
            const row = oy * FX + oz * FXY;
            for (let ox = fx > 0 ? fx - 1 : 0; ox <= fx + 1 && ox < FX; ox++) {
              const cell = ox + row;
              const start = this.fineStart[cell];
              const end = start + this.fineCount[cell];
              for (let e = start; e < end; e++) {
                const j = this.fineEntries[e];
                if (j === i) continue;
                const dx = posPrev[j * 3] - x;
                const dy = posPrev[j * 3 + 1] - y;
                const dz = posPrev[j * 3 + 2] - z;
                const d2 = dx * dx + dy * dy + dz * dz;
                if (d2 >= pushR2) continue;
                const d = Math.max(Math.sqrt(d2), 1e-4);
                const k = ((1 - d / pushR) * GPU_TUNING.pairPush) / d;
                hpX -= dx * k;
                hpY -= dy * k;
                hpZ -= dz * k;
              }
            }
          }
        }
      }

      // ---- Excitable medium: catch fire from glowing neighbors ----
      if (this.refract[i] > 0) this.refract[i] -= dt;
      if (
        contagion > 0 &&
        bestJ >= 0 &&
        this.fired[i] === 0 &&
        this.refract[i] <= 0 &&
        energy[i] < TUNING.contagiousAbove
      ) {
        const p = 1 - Math.exp(-contagion * P.transmission * CPU_TRANSMISSION_SCALE * dt);
        // Inherit the igniter's epicenter — the whole cascade stays one bubble.
        if (Math.random() < p) {
          this.igniteIndex(
            i,
            false,
            this.pulsePrev[bestJ * 3],
            this.pulsePrev[bestJ * 3 + 1],
            this.pulsePrev[bestJ * 3 + 2],
            this.pulseSignPrev[bestJ],
          );
          // igniteIndex kicks this.vel; the locals would overwrite it at
          // integration (v3 lost every contagion kick that way).
          vx = vel[i3];
          vy = vel[i3 + 1];
          vz = vel[i3 + 2];
        }
      }

      // ---- Fear: wears off over time, spreads from panicked neighbors ----
      let fearI = fear[i] * fearDecayF;
      if (maxFear * TUNING.fearSpreadCeiling > fearI) {
        fearI += (maxFear * TUNING.fearSpreadCeiling - fearI) * fearSpreadDt;
      }
      fear[i] = fearI < 0.005 ? 0 : fearI;
      if (fearI < 0.005) fearI = 0;

      // Predator proximity, depth-squashed, measured against the lead point
      // (needed before steering: fear means scatter while the threat is
      // close, regroup once it's gone). uNorm is "how deep into the danger
      // zone" (1 = its edge).
      let pdx = 0, pdy = 0, pdz = 0, pd = 1e9, uNorm = 1e9;
      if (pointerOn) {
        pdx = x - tpx;
        pdy = y - tpy;
        pdz = z - tpz;
        const pdzS = pdz * 0.45;
        pd = Math.sqrt(pdx * pdx + pdy * pdy + pdzS * pdzS) || 1e-4;
        // Each fish has its own zone radius so the hollow is never a clean
        // circle: a static per-fish jitter (0.75-1.25x) and an upstream
        // stretch (a fish swimming toward the predator shies off up to 60%
        // earlier). A real vacuole is a teardrop against the flow.
        const spd0 = Math.sqrt(vx * vx + vy * vy + vz * vz);
        const toward =
          spd0 > 0.2 ? Math.max(0, -(vx * pdx + vy * pdy + vz * pdz) / (spd0 * pd)) : 0;
        uNorm = pd / (mouseR * (0.75 + 0.5 * seed) * (1 + 0.6 * toward));
      }
      const threatNear = uNorm < threatBand;
      // Bite moves the falloff's inner edge out: nothing until a fish is
      // close, then the full bolt. Bite 0 is v3's u*u*(3-2u) exactly.
      const predFall = pointerOn && uNorm < 1 ? 1 - smoothstep(bite, 1, uNorm) : 0;
      // Flight: fear above 1 - flight keeps a fish bolting past the threat
      // band, so hit fish (and those panic spread infects) stream out of
      // the school before regrouping. Needs the pointer for a direction.
      const inFlight = flight > 0.001 && pointerOn && fearI > 1 - flight;

      // After a strike, far-flung survivors commit to returning: RETURN mode
      // persists until they actually reach the shoal, outliving the fear
      // itself (takes effect next frame).
      if (m === FLOCK && fearI > 0.12 && !threatNear && !inFlight) {
        const hx = x - cenX;
        const hy = y - cenY;
        const hz = z - cenZ;
        if (hx * hx + hy * hy + hz * hz > breatheR * breatheR * 0.72) {
          this.mode[i] = RETURN;
          this.modeTimer[i] = 30;
        }
      }

      // ---- Steering forces ----
      let ax = 0, ay = 0, az = 0;
      // Lifts the hard speed cap for the herd and rally rushes; 0 = v3.
      let rallyBoost = 0;
      let pkX = 0, pkY = 0, pkZ = 0;
      const maxSpeed =
        maxSpeedBase *
        spdF *
        (1 + energy[i] * TUNING.energySpeedBoost + fearI * TUNING.fearSpeedBoost);

      // During an inburst, lit particles allow the crowd to condense —
      // when the flash fades, separation snaps back and the bubble pops.
      const relaxMul =
        this.pulseSign[i] < 0 ? 1 - Math.min(1, energy[i]) * TUNING.inburstSepRelax : 1;

      if (found > 0) {
        // Separation (all modes — personal space is universal)
        const sl = Math.sqrt(sepX * sepX + sepY * sepY + sepZ * sepZ);
        if (sl > 1e-5) {
          let sx = (sepX / sl) * maxSpeed - vx;
          let sy = (sepY / sl) * maxSpeed - vy;
          let sz = (sepZ / sl) * maxSpeed - vz;
          const sm = Math.sqrt(sx * sx + sy * sy + sz * sz);
          if (sm > MAX_FORCE) {
            const f = MAX_FORCE / sm;
            sx *= f; sy *= f; sz *= f;
          }
          ax += sx * sepW * relaxMul;
          ay += sy * sepW * relaxMul;
          az += sz * sepW * relaxMul;
        }

        // Collision pushout: obeys the inburst relax (bubbles must
        // condense) and the user's separation slider, not the release pop.
        // Fear damps it and a near threat disables it: threatened schools
        // pack tighter, and shoves at the cleared column's wall would pump
        // fish into it. Integrated beside the steering sum, because
        // dodging a neighbor is not agitation and must not trip the glow.
        if (!threatNear) {
          const k = Math.min(1, P.separation) * relaxMul * (1 - fearI);
          pkX = hpX * k;
          pkY = hpY * k;
          pkZ = hpZ * k;
        }

        // Wanderers cohere too, so a recruited group peels off as a
        // sub-school instead of dust; away from the cloud their only
        // neighbors are each other, so they can still leave it.
        if (m === FLOCK || m === WANDER) {
          // Cohesion — steer toward local center of mass, with "arrival"
          // damping so particles already inside the cluster relax instead
          // of endlessly compressing it.
          let tx = cohX / found - x;
          let ty = cohY / found - y;
          // Sheet: pull harder toward the neighbors' depth. x1 at 0.
          let tz = (cohZ / found - z) * (1 + 2 * sheet);
          // Shape: anisotropic before arrival damping, so damping still
          // governs the body and the stretch is a gain, not a spring.
          if (shape >= 0.001) {
            shapeAniso(tx, ty, tz, headX, headY, headZ, shape);
            tx = ANISO[0]; ty = ANISO[1]; tz = ANISO[2];
          }
          let tl = Math.sqrt(tx * tx + ty * ty + tz * tz);
          if (tl > 1e-5) {
            const arrive = maxSpeed * Math.min(1, tl / arriveR);
            tx = (tx / tl) * arrive - vx;
            ty = (ty / tl) * arrive - vy;
            tz = (tz / tl) * arrive - vz;
            tl = Math.sqrt(tx * tx + ty * ty + tz * tz);
            if (tl > MAX_FORCE) {
              const f = MAX_FORCE / tl;
              tx *= f; ty *= f; tz *= f;
            }
            // Threat close (or in flight): panic scatters the shoal. Threat
            // gone: fear flips into an urgent drive to get back to the group.
            let coh = P.cohesion * dirCoh * social;
            if ((threatNear && fearI > 0.01) || inFlight) {
              coh *= 1 - fearI * 0.6;
            } else if (fearI > 0.08) {
              // sqrt keeps the regroup urge alive — but not into the tail,
              // or the shoal slowly packs itself into a center clump.
              coh *= 1 + Math.sqrt(fearI) * P.regroup * 0.8;
            }
            ax += tx * coh;
            ay += ty * coh;
            az += tz * coh;
          }
        }

        // Alignment — match the neighbors' heading at their actual pace
        // (normalizing to maxSpeed would make an idle swarm race around).
        // Every mode, so a wander group swims as one body.
        let mx = alnX / found;
        let my = alnY / found;
        let mz = alnZ / found;
        const ml = Math.sqrt(mx * mx + my * my + mz * mz);
        if (ml > 1e-5) {
          const match = Math.min(maxSpeed, ml * 1.15);
          mx = (mx / ml) * match - vx;
          my = (my / ml) * match - vy;
          mz = (mz / ml) * match - vz;
          const mm = Math.sqrt(mx * mx + my * my + mz * mz);
          if (mm > MAX_FORCE) {
            const f = MAX_FORCE / mm;
            mx *= f; my *= f; mz *= f;
          }
          // ...but a fleeing (or turning) school moves as one.
          const aln = P.alignment * dirAlign * social * (1 + fearI * 0.8);
          ax += mx * aln;
          ay += my * aln;
          az += mz * aln;
        }
      }

      // ---- The school's collective will: cruise along the shared heading ----
      if (m === FLOCK && driveW > 0.01) {
        const want = maxSpeedBase * spdF * dirSpeed;
        let wx = mhX * want - vx;
        let wy = mhY * want - vy;
        let wz = mhZ * want - vz;
        const wl = Math.sqrt(wx * wx + wy * wy + wz * wz);
        if (wl > MAX_FORCE) {
          const f = MAX_FORCE / wl;
          wx *= f; wy *= f; wz *= f;
        }
        const w = driveW * (0.7 + 0.3 * social);
        ax += wx * w;
        ay += wy * w;
        az += wz * w;
      }

      // ---- Ambient flow field: layered sine currents + global swirl ----
      if (P.flowStrength > 0) {
        ax += Math.sin(ky * y + ft) * Math.cos(kz * z - ft * 0.7) * P.flowStrength;
        ay += Math.sin(kz * z + ft * 1.1) * Math.cos(kx * x + ft * 0.6) * P.flowStrength;
        az += Math.sin(kx * x - ft * 0.8) * Math.cos(ky * y + ft * 0.5) * P.flowStrength * 0.6;
      }
      if (P.swirl > 0) {
        const rr = Math.sqrt(x * x + z * z) + 1;
        ax += (-z / rr) * P.swirl;
        az += (x / rr) * P.swirl;
      }
      if (P.jitter > 0) {
        ax += Math.sin(t * 1.9 + seed * 61.0) * P.jitter * 1.2;
        ay += Math.sin(t * 1.6 + seed * 47.0) * P.jitter * 1.2;
        az += Math.sin(t * 2.2 + seed * 29.0) * P.jitter * 0.8;
      }

      // ---- Sheet: flatten the school toward the view plane ----
      // A school many fish deep along the camera axis averages its density
      // waves away; a few deep, waves, holes and collapses show. A damped
      // spring in z toward the school's mean depth.
      if (sheet > 0.001 && (m === FLOCK || m === WANDER)) {
        az += (cenZ - z) * (sheet * 6) - vz * (sheet * 2);
      }

      // ---- Mode behaviors ----
      const rCenter = Math.sqrt(x * x + y * y + z * z);

      if (m === FLOCK) {
        // Soft anchor plus group recentering, both chasing the roam target.
        const hx = x - roamX;
        const hy = y - roamY;
        const hz = z - roamZ;
        let anX = -hx * 0.02, anY = -hy * 0.02, anZ = -hz * 0.02;
        // The resting body elongates too: same anisotropy on the anchor.
        if (shape >= 0.001) {
          shapeAniso(anX, anY, anZ, headX, headY, headZ, shape);
          anX = ANISO[0]; anY = ANISO[1]; anZ = ANISO[2];
        }
        ax += anX + reCenterX;
        ay += anY + reCenterY;
        az += anZ + reCenterZ;
        // Selfish herd: every fish wants deeper into the school; rel² makes
        // edge fish rush and core fish idle. No arrival damping on purpose:
        // separation and crowd dim push back, and that standoff is the pulse.
        if (herd > 0.001) {
          const hvx = hcX - x;
          const hvy = hcY - y;
          const hvz = hcZ - z;
          const hdc = Math.sqrt(hvx * hvx + hvy * hvy + hvz * hvz);
          if (hdc > 0.3) {
            const rel = Math.min(2, hdc / Math.max(1, P.containRadius * 0.8));
            const g = (herd * herdGain * 12 * rel * rel) / hdc;
            ax += hvx * g;
            ay += hvy * g;
            az += hvz * g;
          }
          // An alarm collapse lifts the speed cap so the rush in is
          // visible. Nothing at herdGain 1 (every mood but contract).
          rallyBoost = Math.max(rallyBoost, 0.5 * herd * (herdGain - 1));
        }
        const rHome = Math.sqrt(hx * hx + hy * hy + hz * hz);
        if (rHome > breatheR) {
          const over = rHome - breatheR;
          const k = (over * over * 0.16) / rHome;
          ax += -hx * k;
          ay += -hy * k;
          az += -hz * k;
        }
        // Fish mill: coherent rotation around the shoal's own center.
        if (millF > 0.01) {
          const mvx = x - cenX;
          const mvy = y - cenY;
          const mr = Math.sqrt(mvx * mvx + mvy * mvy) + 0.5;
          ax += (-mvy / mr) * millF;
          ay += (mvx / mr) * millF;
        }
      } else if (m === WANDER) {
        this.modeTimer[i] -= dt;
        ax += this.wanderDir[i3] * P.wanderPull * 3.2;
        ay += this.wanderDir[i3 + 1] * P.wanderPull * 3.2;
        az += this.wanderDir[i3 + 2] * P.wanderPull * 3.2;
        // Far wall only — wanderers may leave the cloud, not the scene.
        if (rCenter > farR) {
          const over = rCenter - farR;
          const k = (over * over * 0.3) / rCenter;
          ax += -x * k;
          ay += -y * k;
          az += -z * k;
        }
        // Time's up — or a frightened wanderer gives up and heads home.
        if (this.modeTimer[i] <= 0 || fearI > 0.35) {
          this.mode[i] = RETURN;
          this.modeTimer[i] = 30;
        }
      } else {
        // RETURN — home in on the swarm centroid.
        this.modeTimer[i] -= dt;
        const hx = cenX - x;
        const hy = cenY - y;
        const hz = cenZ - z;
        const hd = Math.sqrt(hx * hx + hy * hy + hz * hz) || 1;
        ax += (hx / hd) * P.returnPull * 3.5;
        ay += (hy / hd) * P.returnPull * 3.5;
        az += (hz / hd) * P.returnPull * 3.5;
        // Release at the rim — arriving at dead center would clump it.
        if (hd < P.containRadius * 0.95 || this.modeTimer[i] <= 0) {
          this.mode[i] = FLOCK;
          // A soft rejoining glow (below the contagion threshold).
          if (energy[i] < 0.5) energy[i] = 0.5;
        }
      }

      // ---- Pointer: a predator to escape ----
      // Not a radial force field: threatened particles pick an escape
      // heading and are steered to *swim* along it — turning tail like fish.
      // When the predator is moving, the heading parts sideways around its
      // line of travel and drops behind it: the fountain maneuver.
      if (pointerOn) {
        let fdx = pdx / pd;
        let fdy = pdy / pd;
        let fdz = pdz / pd;
        // Evade around a slow predator: blend the radial escape toward the
        // fish's own sideways momentum, so the hollow is a ring of fish
        // sliding past rather than a static shell. Radial at evade 0.
        if (evOn && pvm <= 1.5) {
          const pl = Math.max(Math.sqrt(pdx * pdx + pdy * pdy + pdz * pdz), 1e-6);
          const erx = pdx / pl;
          const ery = pdy / pl;
          const erz = pdz / pl;
          const vr = vx * erx + vy * ery + vz * erz;
          let tgx = vx - erx * vr;
          let tgy = vy - ery * vr;
          let tgz = vz - erz * vr;
          let tl = Math.sqrt(tgx * tgx + tgy * tgy + tgz * tgz);
          if (tl < 0.15) {
            // Barely moving sideways: pick a side per fish so both occur.
            const side = seed > 0.5 ? 1 : -1;
            tgx = -ery * side;
            tgy = erx * side;
            tgz = 0;
            tl = Math.sqrt(tgx * tgx + tgy * tgy);
          }
          // tl ~ 0 only when the predator is straight along z; stay radial.
          if (tl > 1e-5) {
            const ex = erx * (1 - 0.6 * evade) + (tgx / tl) * (1.2 * evade);
            const ey = ery * (1 - 0.6 * evade) + (tgy / tl) * (1.2 * evade);
            const ez = erz * (1 - 0.6 * evade) + (tgz / tl) * (1.2 * evade);
            const el = Math.max(Math.sqrt(ex * ex + ey * ey + ez * ez), 1e-6);
            fdx = ex / el;
            fdy = ey / el;
            fdz = ez / el;
          }
        }

        if (uNorm < 1) {
          const fall = predFall;

          let ex = fdx;
          let ey = fdy;
          let ez = fdz;
          if (pvm > 1.5) {
            // Sideways component of the radial escape, relative to the
            // predator's heading.
            const along = ex * vhx + ey * vhy + ez * vhz;
            let sx = ex - along * vhx;
            let sy = ey - along * vhy;
            let sz = ez - along * vhz;
            let sl = Math.sqrt(sx * sx + sy * sy + sz * sz);
            if (sl < 0.15) {
              // Dead ahead on the path — dodge to a stable side.
              sx = -vhy;
              sy = vhx;
              sz = 0;
              sl = Math.sqrt(sx * sx + sy * sy) || 1;
            }
            sx /= sl;
            sy /= sl;
            sz /= sl;
            // Part sideways, drift back relative to its travel: fountain.
            ex = ex * 0.45 + sx - vhx * 0.35;
            ey = ey * 0.45 + sy - vhy * 0.35;
            ez = ez * 0.45 + sz - vhz * 0.35;
            const el = Math.sqrt(ex * ex + ey * ey + ez * ez) || 1;
            ex /= el;
            ey /= el;
            ez /= el;
          }

          // Steer toward swimming at panic speed along the escape heading.
          let stx = ex * maxSpeed - vx;
          let sty = ey * maxSpeed - vy;
          let stz = ez * maxSpeed - vz;
          const stl = Math.sqrt(stx * stx + sty * sty + stz * stz) || 1;
          const force = P.repelForce * fall * (1 + splashBoost);
          ax += (stx / stl) * force;
          ay += (sty / stl) * force;
          az += (stz / stl) * force * 0.6;

          // Contact with the predator instills lasting fear.
          if (fall > fearI) fear[i] = fearI = fall;

          // Fast movement excites the swarm → cascades bursting away
          // from the predator's bow like a wake.
          if (igniteRate > 0.05 && this.fired[i] === 0 && this.refract[i] <= 0) {
            if (Math.random() < igniteRate * fall * dt) {
              this.igniteIndex(i, false, tpx, tpy, tpz, 1);
              vx = vel[i3];
              vy = vel[i3 + 1];
              vz = vel[i3 + 2];
            }
          }
        }
        // Frightened particles keep swimming away even outside the radius —
        // but only while the threat is still close; regrouping takes over
        // the moment it is not.
        if (fearI > 0.02 && threatNear) {
          const flee = P.fleeForce * fearI;
          ax += fdx * flee;
          ay += fdy * flee;
          az += fdz * flee * 0.35;
        }
        // In flight past the threat band: keep bolting, weaker at low
        // flight so the carry-out grows with the slider.
        if (inFlight && !threatNear) {
          const flee = P.fleeForce * fearI * (0.35 + 0.65 * flight);
          ax += (pdx / pd) * flee;
          ay += (pdy / pd) * flee;
          az += (pdz / pd) * flee * 0.35;
        }
      }

      // Regroup urgency: still afraid but the threat is out of range —
      // swim straight back to the shoal (works with the pointer gone too).
      // Only particles still *outside* the shoal body are pulled home —
      // once inside, the urge releases so the center doesn't clump. Waits
      // until flight has burned off.
      if (fearI > 0.02 && !threatNear && !inFlight) {
        const hx = hcX - x;
        const hy = hcY - y;
        const hz = hcZ - z;
        const hd = Math.sqrt(hx * hx + hy * hy + hz * hz);
        if (hd > P.containRadius * 0.75) {
          const g = (P.regroup * 3.2 * Math.sqrt(fearI)) / hd;
          ax += hx * g;
          ay += hy * g;
          az += hz * g;
        }
        // Rally: frightened fish want the school's center, not its edge.
        // No distance gate and no arrival damping, so they overshoot
        // through the center; separation and crowd dim absorb the pile-up.
        if (rally > 0.001 && hd > 0.5) {
          const g = (P.regroup * 3.2 * Math.sqrt(fearI) * rally * 1.5) / hd;
          ax += hx * g;
          ay += hy * g;
          az += hz * g;
          rallyBoost = Math.max(rallyBoost, rally * Math.sqrt(fearI));
        }
      }

      // ---- Bubble pulse: sustained radial push from the cascade epicenter,
      // outward for a burst (+1), inward for a collapse (-1) ----
      const ei = energy[i];
      if (ei > 0.12 && P.kick > 0) {
        const rx = x - this.pulse[i3];
        const ry = y - this.pulse[i3 + 1];
        const rz = z - this.pulse[i3 + 2];
        const rd = Math.sqrt(rx * rx + ry * ry + rz * rz);
        if (rd > 0.25) {
          // Collapses need extra gain to beat the crowd pressure.
          const gain =
            this.pulseSign[i] > 0 ? TUNING.radialGainOut : TUNING.radialGainIn;
          const f = (this.pulseSign[i] * ei * P.kick * gain) / rd;
          ax += rx * f;
          ay += ry * f;
          az += rz * f * 0.7;
        }
      }

      // ---- Burst-and-glide gait: thrust, then coast ----
      if (P.gait > 0.01) {
        this.gaitClock[i] -= dt * gaitRate;
        if (this.gaitClock[i] <= 0) {
          this.gaitClock[i] = 0.35 + Math.random() * 1.1;
          // Thrust along current motion, blended with the fish's heading
          // (the turn wave's, so fish the front hasn't reached don't get
          // thrust onto the new heading early).
          const sp = Math.sqrt(vx * vx + vy * vy + vz * vz);
          const blend = m === FLOCK ? dirSpeed * 1.2 : 0.2;
          const dxT = (sp > 0.2 ? vx / sp : mhX) + mhX * blend;
          const dyT = (sp > 0.2 ? vy / sp : mhY) + mhY * blend;
          const dzT = (sp > 0.2 ? vz / sp : mhZ) + mhZ * blend;
          const dl = Math.sqrt(dxT * dxT + dyT * dyT + dzT * dzT) || 1;
          const imp =
            P.gait * spdF * (0.4 + dirSpeed * 0.9 + fearI * 0.7) *
            maxSpeedBase * 0.34;
          vx += (dxT / dl) * imp;
          vy += (dyT / dl) * imp;
          vz += (dzT / dl) * imp * 0.7;
          if (imp > 0.35 && glow[i] < imp * 0.5) glow[i] = imp * 0.5;
        }
      }

      // ---- Integrate ----
      vx = (vx + (ax + pkX) * dt) * dragF;
      vy = (vy + (ay + pkY) * dt) * dragF;
      vz = (vz + (az + pkZ) * dt) * dragF;

      const sp2 = vx * vx + vy * vy + vz * vz;
      // maxSpeed also feeds the steering targets; only the hard cap rises.
      const cap = maxSpeed * (1 + 0.8 * rallyBoost) * (m === WANDER ? 0.85 : 1) + 0.4;
      if (sp2 > cap * cap) {
        const f = cap / Math.sqrt(sp2);
        vx *= f; vy *= f; vz *= f;
      }

      vel[i3] = vx;
      vel[i3 + 1] = vy;
      vel[i3 + 2] = vz;
      pos[i3] = x + vx * dt;
      pos[i3 + 1] = y + vy * dt;
      pos[i3 + 2] = z + vz * dt;

      // ---- Shear-triggered luminescence: agitation glows ----
      // (like dinoflagellates firing when the water shears around them)
      // Soft knee: steady cruising forces stay dark — only above-baseline
      // acceleration (turn waves, thrusts, strikes) fires the light.
      const aMag = Math.sqrt(ax * ax + ay * ay + az * az);
      const gTarget = Math.min(
        1,
        Math.max(0, aMag - TUNING.glowKnee) * TUNING.glowGain,
      );
      glow[i] += (gTarget - glow[i]) * glowK;

      // ---- Flash bookkeeping ----
      let e = energy[i] * energyDecay;
      if (e < 0.005) e = 0;
      energy[i] = e;
      if (this.fired[i] === 1 && e < TUNING.flashEndsBelow) {
        this.fired[i] = 0;
        this.refract[i] = P.refractory;
      }

      fearSum += fearI;
      if (this.mode[i] === FLOCK) {
        const nx = pos[i3];
        const ny = pos[i3 + 1];
        const nz = pos[i3 + 2];
        cSumX += nx;
        cSumY += ny;
        cSumZ += nz;
        cNum++;
        // Spread about last frame's centroid (the GPU reduction's choice;
        // the bias is negligible at swimming speed).
        const sx = nx - cenX;
        const sy = ny - cenY;
        const sz = nz - cenZ;
        spreadSum += sx * sx + sy * sy + sz * sz;
      }
    }

    // Centroid used next frame by returning wanderers.
    if (cNum > 0) {
      this.centroidX = cSumX / cNum;
      this.centroidY = cSumY / cNum;
      this.centroidZ = cSumZ / cNum;
      this.rmsRadius = Math.sqrt(spreadSum / cNum);
    } else {
      this.centroidX = this.centroidY = this.centroidZ = 0;
    }
    this.meanFear = fearSum / Math.max(1, n);

    this.updateCrowdFade(dt);

    // Push cheap uniforms every frame so panel tweaks apply instantly.
    const u = this.material.uniforms;
    u.uTime.value = t;
    u.uSize.value = P.particleSize;
    u.uOpacity.value = P.opacity * this.opacityScale;
    u.uFlashBright.value = P.flashBrightness;
    u.uTwinkle.value = P.twinkle;
    u.uStretch.value = P.stretch;
    u.uFlashSize.value = P.flashSize ?? 0.45;
    u.uMotionGlow.value = P.motionGlow;

    const attrs = this.geometry.attributes;
    this.markLive(attrs.position as THREE.BufferAttribute);
    this.markLive(attrs.aEnergy as THREE.BufferAttribute);
    this.markLive(attrs.aGlow as THREE.BufferAttribute);
    this.markLive(attrs.aVel as THREE.BufferAttribute);
  }

  /**
   * Queue an upload of the live fish only: the dormant tail isn't drawn.
   * Cleared first, because the harness steps many frames per render and
   * three.js keeps every queued range until the next upload.
   */
  private markLive(attr: THREE.BufferAttribute): void {
    attr.clearUpdateRanges();
    attr.addUpdateRange(0, this.count * attr.itemSize);
    attr.needsUpdate = true;
  }

  setPixelRatio(dpr: number): void {
    this.material.uniforms.uDpr.value = Math.min(dpr, 2);
  }

  dispose(): void {
    this.scene.remove(this.points);
    this.geometry.dispose();
    this.material.dispose();
  }
}
