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
import { SwarmCamera } from "./camera";
import {
  U_VEC4S,
  WGSL_GRID,
  WGSL_REDUCE,
  WGSL_RESOLVE,
  WGSL_SCAN,
  WGSL_SIM,
} from "./wgsl-sim";
import { GpuRenderer } from "./renderer";

// School moods (whole-swarm director) - identical to the CPU reference,
// plus CONTRACT (GPU-only, the `collapse` lever): the school condenses on
// its centroid, then releases. Unreachable at collapse 0, so the default
// FSM is exactly the reference's.
const M_CRUISE = 0;
const M_TURN = 1;
const M_MILL = 2;
const M_HOVER = 3;
const M_STARTLE = 4;
const M_CONTRACT = 5;
const MOOD_NAMES = ["cruise", "turn", "mill", "hover", "startle", "contract"];

// Event kinds — mirror the EV_* constants in wgsl-sim.ts.
const EV_BURST = 1;
const EV_STARTLE = 2;
const EV_VANGUARD = 3;
const EV_WANDER = 4;
const EV_SCATTER = 5;
const EV_IGNITE = 6;
const EV_STRIKE = 7;
const EV_SHIVER = 8;

const D = TUNING.gridDim;
const CELLS = D * D * D;
const FINE_CELLS = GPU_TUNING.fineGridDim ** 3;
const MAX_EVENTS = GPU_TUNING.maxEvents;
const EVENT_STRIDE = 16; // floats/u32s per event (4 × vec4)
const WG = GPU_TUNING.workgroupSize;
const REDUCE_WGS = 64;

interface PendingEvent {
  kind: number;
  index?: number;
  needsResolve?: boolean;
  center?: [number, number, number];
  radius?: number;
  dir?: [number, number, number];
  sign?: number;
  misc?: [number, number, number, number];
}

interface Vec3 {
  x: number;
  y: number;
  z: number;
}

/** Full sim state pulled back to the host — the harness's window in. */
export interface GpuStateSnapshot {
  pos: Float32Array;
  vel: Float32Array;
  energy: Float32Array;
  glow: Float32Array;
  fear: Float32Array;
  pulse: Float32Array;
  pulseSign: Float32Array;
  mode: Uint8Array;
  pointer: Vec3;
}

/**
 * The GPU rewrite of the living school. Same four behavior layers as the
 * CPU reference; the split follows HANDOFF-GPU.md §2.2: the swarm director
 * (a handful of scalars) stays host-side, every per-particle quantity lives
 * in GPU storage buffers, and host→swarm interventions travel through a
 * small event queue resolved on-GPU (no readback in the loop; the centroid
 * comes back asynchronously, 1–2 frames stale — same semantics as the CPU
 * version's "last frame's centroid").
 */
export class GpuSwarm implements SwarmSystem {
  readonly params: SwarmParams;
  presetNames: Record<PresetCategory, string> = { ...DEFAULT_PRESETS };
  paletteName: PaletteName = "emerald";
  styleName: StyleName = "glow";

  readonly camera: SwarmCamera;
  /** Bumped on rebuild so the renderer refreshes its bind groups. */
  generation = 0;

  private device: GPUDevice;
  private count = 0;
  /** Allocated particle slots; count ≤ capacity. Grows only on demand. */
  private capacity = 0;

  // --- GPU resources (rebuilt with the particle count) ---
  private dyn: [GPUBuffer, GPUBuffer] | null = null;
  private selfB!: GPUBuffer;
  private statB!: GPUBuffer;
  /** Counts + scatter cursors for both grids (atomic u32, const offsets). */
  private gridAtomics!: GPUBuffer;
  /** Cell starts + sorted entries for both grids (u32; layout in wgsl-sim). */
  private gridData!: GPUBuffer;
  private eventsB!: GPUBuffer;
  private uniformsB!: GPUBuffer;
  private centroidB!: GPUBuffer;
  private parity = 0; // dyn[parity] is the read side this frame

  private pipelines!: {
    clear: GPUComputePipeline;
    count: GPUComputePipeline;
    scan: GPUComputePipeline;
    scanFine: GPUComputePipeline;
    scatter: GPUComputePipeline;
    resolve: GPUComputePipeline;
    sim: GPUComputePipeline;
    reduce1: GPUComputePipeline;
    reduce2: GPUComputePipeline;
  };
  private groups!: {
    clear: GPUBindGroup;
    count: [GPUBindGroup, GPUBindGroup];
    scan: GPUBindGroup;
    scanFine: GPUBindGroup;
    scatter: [GPUBindGroup, GPUBindGroup];
    resolve: [GPUBindGroup, GPUBindGroup];
    sim: [GPUBindGroup, GPUBindGroup];
    reduce: [GPUBindGroup, GPUBindGroup];
    reduce2: GPUBindGroup;
  };

  // Host mirrors for statics so palette switches don't regenerate sizes.
  private staticsHost!: Float32Array;
  /** Accent override ("#rrggbb"); null = the palette's own accent. */
  private _accent: string | null = null;
  /** accentShare the colors were last generated with (update() polls it). */
  private lastAccentShare = NaN;
  /** Per-spawn seed so a recolor reuses each fish's draws. */
  private colorSeed = 0;

  private uniforms = new Float32Array(U_VEC4S * 4);
  private eventData = new ArrayBuffer(MAX_EVENTS * EVENT_STRIDE * 4);
  private eventF32 = new Float32Array(this.eventData);
  private eventU32 = new Uint32Array(this.eventData);
  private pending: PendingEvent[] = [];

  // Centroid readback ring.
  private staging: { buf: GPUBuffer; state: "free" | "pending" | "mapping" }[] =
    [];
  private centroid: Vec3 = { x: 0, y: 0, z: 0 };
  private flockN = 0;
  /** GPU-reduced stats, 1–2 frames stale (readback ring). */
  private meanSpeed = 0;
  private hotGlowCount = 0;
  /** Mean per-fish fear (readback): the director's alarm signal. */
  private meanFear = 0;
  /**
   * RMS distance of flock fish from the centroid (readback, ~2 frames
   * stale). Seeded with a uniform ball of 0.8 × tank (RMS = R·√(3/5)) until
   * the first readback lands; set in rebuild() once params exist.
   */
  private rmsRadius = 0;

  // --- Simulation clock ---
  private time = 0;
  private frame = 0;

  // --- Pointer state (port of the CPU fields) ---
  private pointer: Vec3 = { x: 0, y: 0, z: 0 };
  private prevPointerW: Vec3 = { x: 0, y: 0, z: 0 };
  private pointerVel: Vec3 = { x: 0, y: 0, z: 0 };
  private hadPointerFrame = false;
  private pointerActive = false;
  private mouseSpeed = 0;
  private lastPointerTime = 0;
  private prevNDC = { x: 0, y: 0 };
  private hasPrevNDC = false;

  private wanderClock = 5;

  // Evade: smoothed displacement of the school's center away from a parked
  // predator (stays zero at evade 0).
  private evadeX = 0;
  private evadeY = 0;
  private evadeZ = 0;

  // Strike (flick) detection.
  private strikeCooldown = 0;
  private lastSlowTime = 0;

  // --- Swarm director state (identical numbers to the reference) ---
  private mood = M_CRUISE;
  private moodTimer = 6;
  private headX = 1;
  private headY = 0;
  private headZ = 0;
  private startleClock = 25;
  /** Shiver (GPU-only): countdown to the next resting agitation wave. */
  private shiverClock = 8;
  private dirAlign = 1;
  private dirCoh = 1;
  private dirMill = 0;
  private dirDrive = 1;
  private dirSpeed = 0.6;
  /** Separation gain (GPU-only): dips while contracting, pops on release. */
  private dirSep = 1;
  /** Herd-drive gain (GPU-only): swells while contracting, 1 otherwise. */
  private dirHerd = 1;
  private tgtAlign = 1;
  private tgtCoh = 1;
  private tgtMill = 0;
  private tgtDrive = 1;
  private tgtSpeed = 0.6;
  private tgtSep = 1;
  private tgtHerd = 1;

  // Alarm-triggered collapse (GPU-only, the `collapse` lever): a scare
  // that subsides pulls the whole school into a bait ball.
  /** meanFear smoothed over ~0.15 s. */
  private alarm = 0;
  /** Running max of alarm, decaying back toward it at 1/s. */
  private alarmPeak = 0;
  private alarmCooldown = 0;
  /** Set right before enterMood(M_CONTRACT) to pick the harder contraction. */
  private contractFromAlarm = false;

  constructor(device: GPUDevice, aspect: number, count: number) {
    this.device = device;
    this.camera = new SwarmCamera(aspect);
    this.params = { ...DEFAULT_PARAMS };

    // Random initial heading so every load starts differently.
    const a = Math.random() * Math.PI * 2;
    this.headX = Math.cos(a);
    this.headY = Math.sin(a);

    this.createPipelines();
    this.rebuild(count);
  }

  getCount(): number {
    return this.count;
  }

  get moodName(): string {
    return MOOD_NAMES[this.mood];
  }

  /** Latest known centroid (async readback; 1–2 frames stale). */
  getCentroid(): Vec3 {
    return this.centroid;
  }

  /**
   * GPU-reduced frame stats (mean |vel|, count of glow ≥ 0.8, school RMS
   * radius, mean fear); ~2 frames stale.
   */
  stats(): { meanSpeed: number; hotGlow: number; rmsRadius: number; meanFear: number } {
    return {
      meanSpeed: this.meanSpeed,
      hotGlow: this.hotGlowCount,
      rmsRadius: this.rmsRadius,
      meanFear: this.meanFear,
    };
  }

  /**
   * RMS distance of flock fish from their centroid — the school's live
   * spread (contracts with the director's contract mood). ~2 frames stale.
   */
  schoolRmsRadius(): number {
    return this.rmsRadius;
  }

  /** RMS radius of a uniform ball filling 0.8 × the tank: R·√(3/5). */
  private nominalRmsRadius(): number {
    return this.params.containRadius * 0.8 * Math.sqrt(3 / 5);
  }

  // ------------------------------------------------------------------
  // Presets / palettes / styles / snapshots — same logic as the reference
  // ------------------------------------------------------------------

  applyPreset(category: PresetCategory, name: string): void {
    const preset = PRESET_REGISTRY[category]?.[name];
    if (!preset) return;
    this.presetNames[category] = name;
    const params = this.params as unknown as Record<string, number>;
    const defaults = DEFAULT_PARAMS as unknown as Record<string, number>;
    for (const key of CATEGORY_KEYS[category]) {
      params[key] =
        (preset.params as Record<string, number>)[key] ?? defaults[key];
    }
  }

  setPalette(name: PaletteName): void {
    if (!PALETTES[name]) return;
    this.paletteName = name;
    this.recolor();
  }

  /** The accent override, or null while the palette's own accent shows. */
  get accentColor(): string | null {
    return this._accent;
  }

  /**
   * Override the palette's accent color ("#rrggbb"); null restores the
   * palette's own. Malformed input is ignored rather than clearing.
   */
  setAccent(hex: string | null): void {
    if (hex !== null && !/^#[0-9a-f]{6}$/i.test(hex)) return;
    const next = hex === null ? null : hex.toLowerCase();
    if (next === this._accent) return;
    this._accent = next;
    this.recolor();
  }

  setStyle(name: StyleName): void {
    if (!(name in STYLE_IDS)) return;
    this.styleName = name;
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
    };
  }

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
    if (s.palette) this.setPalette(s.palette);
    if (s.style) this.setStyle(s.style);
    if (s.count && s.count !== this.count) this.rebuild(s.count);
  }

  // ------------------------------------------------------------------
  // Construction
  // ------------------------------------------------------------------

  private makeModule(code: string, label: string): GPUShaderModule {
    const module = this.device.createShaderModule({ code, label });
    module.getCompilationInfo().then((info) => {
      for (const m of info.messages) {
        if (m.type === "error") {
          console.error(`[wgsl:${label}] ${m.lineNum}:${m.linePos} ${m.message}`);
        }
      }
    });
    return module;
  }

  private createPipelines(): void {
    const dev = this.device;
    const mk = (code: string, entry: string, label: string) =>
      dev.createComputePipeline({
        label,
        layout: "auto",
        compute: { module: this.makeModule(code, label), entryPoint: entry },
      });

    const gridModCode = WGSL_GRID;
    this.pipelines = {
      clear: mk(gridModCode, "clearGrid", "clearGrid"),
      count: mk(gridModCode, "countGrid", "countGrid"),
      scan: mk(WGSL_SCAN, "scanGrid", "scanGrid"),
      scanFine: mk(WGSL_SCAN, "scanFine", "scanFine"),
      scatter: mk(gridModCode, "scatterGrid", "scatterGrid"),
      resolve: mk(WGSL_RESOLVE, "resolveEvents", "resolveEvents"),
      sim: mk(WGSL_SIM, "simStep", "simStep"),
      reduce1: mk(WGSL_REDUCE, "reducePass1", "reducePass1"),
      reduce2: mk(WGSL_REDUCE, "reducePass2", "reducePass2"),
    };
  }

  /**
   * Reset the swarm to a fresh spawn at the given count (v3 rebuild
   * semantics — the harness relies on this giving every probe a clean
   * field). Buffers are only reallocated when the count outgrows the
   * allocated capacity; otherwise the respawn happens in place.
   */
  rebuild(count: number): void {
    count = Math.max(64, Math.floor(count));
    if (this.dyn && count <= this.capacity) {
      this.count = count;
      this.initParticles();
      this.parity = 0;
      this.centroid = { x: 0, y: 0, z: 0 };
      this.flockN = count;
      this.rmsRadius = this.nominalRmsRadius();
      this.meanFear = 0;
      this.evadeX = this.evadeY = this.evadeZ = 0;
      return;
    }
    this.destroyBuffers();
    this.capacity = count;
    this.count = count;
    const dev = this.device;

    const mkStorage = (bytes: number, label: string, extra: number = 0) =>
      dev.createBuffer({
        label,
        size: bytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | extra,
      });

    this.dyn = [
      mkStorage(this.capacity * 48, "dynA"),
      mkStorage(this.capacity * 48, "dynB"),
    ];
    this.selfB = mkStorage(this.capacity * 32, "self");
    this.statB = mkStorage(this.capacity * 32, "stat");
    this.gridAtomics = mkStorage(2 * (CELLS + FINE_CELLS) * 4, "gridAtomics");
    // fine starts + coarse starts + coarse entries + fine entries.
    this.gridData = mkStorage(
      (FINE_CELLS + 1 + CELLS + 1 + 2 * this.capacity) * 4,
      "gridData",
    );
    this.eventsB = mkStorage(this.eventData.byteLength, "events");
    // [0] centroid+flockN, [1] stats (Σ|vel|, hotGlow, Σ|pos−cen|², 0),
    // then 2×64 partial blocks.
    this.centroidB = mkStorage(16 * (2 + 2 * REDUCE_WGS), "centroid");
    this.uniformsB = dev.createBuffer({
      label: "uniforms",
      size: this.uniforms.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.staging = Array.from({ length: 3 }, (_, k) => ({
      buf: dev.createBuffer({
        label: `centroid-staging-${k}`,
        size: 32,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      }),
      state: "free" as const,
    }));

    this.initParticles();
    this.createBindGroups();
    this.parity = 0;
    this.centroid = { x: 0, y: 0, z: 0 };
    this.flockN = count;
    this.rmsRadius = this.nominalRmsRadius();
    this.meanFear = 0;
    this.evadeX = this.evadeY = this.evadeZ = 0;
    this.generation++;
  }

  /**
   * Change the particle count *without* resetting the school — the living
   * swarm keeps swimming. Shrinking simply deactivates the tail; growing
   * revives dormant slots in a soft sphere around the current centroid so
   * newcomers surface inside the school instead of teleporting the world.
   * The density-compensated brightness keeps the perceived mass steady, so
   * the frame-time governor can breathe the count up and down invisibly.
   * Falls back to a full respawn only when outgrowing the allocation.
   */
  resize(count: number): void {
    count = Math.max(64, Math.floor(count));
    if (!this.dyn || count > this.capacity) {
      this.rebuild(count);
      return;
    }
    if (count > this.count) this.reviveRange(this.count, count);
    this.count = count;
  }

  /** Wake dormant slots [from, to) near the school's current center. */
  private reviveRange(from: number, to: number): void {
    const n = to - from;
    const cen = this.centroid;
    const R = this.params.containRadius;
    const dyn = new Float32Array(n * 12);
    const selfD = new Float32Array(n * 8);
    for (let i = 0; i < n; i++) {
      const r = R * 0.7 * Math.cbrt(Math.random());
      const theta = Math.random() * Math.PI * 2;
      const cosPhi = Math.random() * 2 - 1;
      const sinPhi = Math.sqrt(1 - cosPhi * cosPhi);
      const d = i * 12;
      dyn[d] = cen.x + r * sinPhi * Math.cos(theta);
      dyn[d + 1] = cen.y + r * sinPhi * Math.sin(theta);
      dyn[d + 2] = cen.z + r * cosPhi;
      dyn[d + 4] = (Math.random() - 0.5) * 0.6;
      dyn[d + 5] = (Math.random() - 0.5) * 0.6;
      dyn[d + 6] = (Math.random() - 0.5) * 0.6;
      // energy/fear/pulse zero.
      selfD[i * 8 + 2] = Math.random() * 1.5; // gaitClock; rest zero = FLOCK
    }
    const q = this.device.queue;
    q.writeBuffer(this.dyn![0], from * 48, dyn);
    q.writeBuffer(this.dyn![1], from * 48, dyn);
    q.writeBuffer(this.selfB, from * 32, selfD);
    // Statics (color/size/personality) were initialized for the full
    // capacity and stay valid across sleep/wake cycles.
  }

  /**
   * Port of the reference's spawn: soft sphere + a few shell whiskers.
   * Fills the full allocated capacity so resize() can activate dormant
   * particles later without touching statics.
   */
  private initParticles(): void {
    const n = this.capacity;
    const R = this.params.containRadius;
    const dyn = new Float32Array(n * 12);
    const selfD = new Float32Array(n * 8);
    this.staticsHost = new Float32Array(n * 8);

    for (let i = 0; i < n; i++) {
      const shell = Math.random() < 0.15;
      const r = shell
        ? R * (0.9 + Math.random() * 0.5)
        : R * 0.9 * Math.cbrt(Math.random());
      const theta = Math.random() * Math.PI * 2;
      const cosPhi = Math.random() * 2 - 1;
      const sinPhi = Math.sqrt(1 - cosPhi * cosPhi);

      const d = i * 12;
      dyn[d] = r * sinPhi * Math.cos(theta);
      dyn[d + 1] = r * sinPhi * Math.sin(theta);
      dyn[d + 2] = r * cosPhi;
      dyn[d + 3] = 0; // energy
      dyn[d + 4] = (Math.random() - 0.5) * 0.6;
      dyn[d + 5] = (Math.random() - 0.5) * 0.6;
      dyn[d + 6] = (Math.random() - 0.5) * 0.6;
      dyn[d + 7] = 0; // fear
      // pulse xyz + sign stay 0.

      const s = i * 8;
      selfD[s] = 0; // glow
      selfD[s + 1] = 0; // refract
      selfD[s + 2] = Math.random() * 1.5; // gaitClock
      selfD[s + 3] = 0; // modeTimer
      // wanderDir 0, flags 0 (FLOCK, not fired)

      const st = i * 8;
      // colors filled by regenerateColors()
      this.staticsHost[st + 3] = 0.5 + Math.pow(Math.random(), 1.6) * 0.9; // size
      this.staticsHost[st + 4] = 0.82 + Math.random() * 0.36; // speedFactor
      this.staticsHost[st + 5] = 0.7 + Math.random() * 0.6; // social
      this.staticsHost[st + 6] = Math.random(); // seed
    }
    this.colorSeed = (Math.random() * 2 ** 32) >>> 0;
    this.regenerateColors();

    const q = this.device.queue;
    q.writeBuffer(this.dyn![0], 0, dyn);
    q.writeBuffer(this.dyn![1], 0, dyn); // both sides start identical
    q.writeBuffer(this.selfB, 0, selfD);
    q.writeBuffer(this.statB, 0, this.staticsHost);
    q.writeBuffer(this.centroidB, 0, new Float32Array(4 * (2 + 2 * REDUCE_WGS)));
  }

  /**
   * Port of applyPaletteColors: writes rgb lanes of the statics mirror.
   * Every fish takes the same five draws from the spawn's seed whichever
   * branch it lands in, so dragging the share flips only the fish crossing
   * it and a new accent recolors only accent fish, not the whole school.
   */
  private regenerateColors(): void {
    const palette: Palette = PALETTES[this.paletteName];
    const stops = palette.stops.map(hexToRgb);
    const accent = hexToRgb(this._accent ?? palette.accent);
    const share = this.params.accentShare;
    this.lastAccentShare = share;
    const rand = mulberry32(this.colorSeed);
    for (let i = 0; i < this.capacity; i++) {
      const pick = rand();
      const t = Math.pow(rand(), 0.8) * (stops.length - 1);
      let rgb: [number, number, number];
      if (pick < share) {
        rgb = accent;
      } else {
        const idx = Math.min(stops.length - 2, Math.floor(t));
        const f = t - idx;
        rgb = [
          stops[idx][0] + (stops[idx + 1][0] - stops[idx][0]) * f,
          stops[idx][1] + (stops[idx + 1][1] - stops[idx][1]) * f,
          stops[idx][2] + (stops[idx + 1][2] - stops[idx][2]) * f,
        ];
      }
      const st = i * 8;
      this.staticsHost[st] = Math.min(1, rgb[0] * (0.9 + rand() * 0.2));
      this.staticsHost[st + 1] = Math.min(1, rgb[1] * (0.9 + rand() * 0.2));
      this.staticsHost[st + 2] = Math.min(1, rgb[2] * (0.9 + rand() * 0.2));
    }
  }

  /** Regenerate and upload colors (no-op before the first spawn). */
  private recolor(): void {
    if (!this.staticsHost) return;
    this.regenerateColors();
    this.device.queue.writeBuffer(this.statB, 0, this.staticsHost);
  }

  private createBindGroups(): void {
    const dev = this.device;
    const p = this.pipelines;
    const dyn = this.dyn!;
    const bg = (
      pipeline: GPUComputePipeline,
      entries: (GPUBuffer | null)[],
    ): GPUBindGroup =>
      dev.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: entries.flatMap((buf, i) =>
          buf ? [{ binding: i, resource: { buffer: buf } }] : [],
        ),
      });

    const both = (
      pipeline: GPUComputePipeline,
      make: (dynIn: GPUBuffer, dynOut: GPUBuffer) => (GPUBuffer | null)[],
    ): [GPUBindGroup, GPUBindGroup] => [
      bg(pipeline, make(dyn[0], dyn[1])),
      bg(pipeline, make(dyn[1], dyn[0])),
    ];

    this.groups = {
      clear: bg(p.clear, [null, null, this.gridAtomics]),
      count: both(p.count, (dynIn) => [
        this.uniformsB,
        dynIn,
        this.gridAtomics,
      ]),
      scan: bg(p.scan, [this.gridAtomics, this.gridData]),
      scanFine: bg(p.scanFine, [this.gridAtomics, this.gridData]),
      scatter: both(p.scatter, (dynIn) => [
        this.uniformsB,
        dynIn,
        this.gridAtomics,
        this.gridData,
      ]),
      resolve: both(p.resolve, (dynIn) => [
        this.uniformsB,
        dynIn,
        this.eventsB,
        this.centroidB,
      ]),
      sim: both(p.sim, (dynIn, dynOut) => [
        this.uniformsB,
        dynIn,
        dynOut,
        this.selfB,
        this.statB,
        this.gridData,
        this.eventsB,
        this.centroidB,
      ]),
      reduce: both(p.reduce1, (_dynIn, dynOut) => [
        this.uniformsB,
        dynOut,
        this.selfB,
        this.centroidB,
      ]),
      // reducePass2 only touches `centroid`, declared at binding 3.
      reduce2: bg(p.reduce2, [null, null, null, this.centroidB]),
    };
  }

  private destroyBuffers(): void {
    if (!this.dyn) return;
    for (const b of [
      ...this.dyn,
      this.selfB,
      this.statB,
      this.gridAtomics,
      this.gridData,
      this.eventsB,
      this.uniformsB,
      this.centroidB,
      ...this.staging.map((s) => s.buf),
    ]) {
      b?.destroy();
    }
    this.dyn = null;
    this.staging = [];
  }

  // ------------------------------------------------------------------
  // Pointer — verbatim port of the reference host logic
  // ------------------------------------------------------------------

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
    this.prevNDC = { x: ndc.x, y: ndc.y };
    this.lastPointerTime = now;
    this.hasPrevNDC = true;
    this.pointerActive = true;

    // Aim at the school's mid-depth, not the fixed z=0 plane: the predator
    // column tracks wherever the school is actually swimming. Clamped so a
    // degenerate centroid (boot, empty flock) can't fling the pointer.
    const planeZ = Math.max(
      -this.params.containRadius,
      Math.min(this.params.containRadius, this.centroid.z),
    );
    this.pointer = this.camera.unprojectToSwarmPlane(ndc, planeZ);
  }

  clearPointer(): void {
    this.pointerActive = false;
    this.hasPrevNDC = false;
    this.mouseSpeed = 0;
  }

  updateRendererSize(halfX: number, halfY: number): void {
    this.camera.setAspect(halfX / Math.max(1, halfY));
  }

  setPixelRatio(_dpr: number): void {
    // The renderer reads devicePixelRatio directly; kept for interface parity.
  }

  // ------------------------------------------------------------------
  // Actions → events
  // ------------------------------------------------------------------

  burst(at?: Vec3): void {
    const c = at ?? (this.pointerActive ? this.pointer : { x: 0, y: 0, z: 0 });
    this.pending.push({
      kind: EV_BURST,
      center: [c.x, c.y, c.z],
      radius: Math.max(2, this.params.mouseRadius * TUNING.burstSeedRatio),
    });
  }

  scatter(): void {
    this.pending.push({ kind: EV_SCATTER });
  }

  startle(at?: Vec3): void {
    if (at) {
      this.pending.push({ kind: EV_STARTLE, center: [at.x, at.y, at.z] });
    } else {
      this.pending.push({
        kind: EV_STARTLE,
        index: (Math.random() * this.count) | 0,
        needsResolve: true,
      });
    }
    this.enterMood(M_STARTLE);
  }

  /** Spawn a spontaneous cascade at a random fish — sometimes an implosion. */
  private igniteSpontaneous(): void {
    this.pending.push({
      kind: EV_IGNITE,
      index: (Math.random() * this.count) | 0,
      sign: Math.random() < this.params.inburst ? -1 : 1,
    });
  }

  /** A shiver pocket around a random fish; radius in units of perception. */
  private spawnShiver(): void {
    // Arm the alarm: a single pocket rarely lifts the school's mean fear
    // past the threshold (linear falloff averages 0.25 over the sphere and
    // contagion halves per hop), so without this the wave would fade
    // without the re-contraction that is the point of a shiver. The alarm
    // then tracks the (lower) mean fear, crosses half its peak as the wave
    // dies, and the existing logic fires the collapse.
    this.alarm = Math.max(this.alarm, 0.12);
    this.alarmPeak = Math.max(this.alarmPeak, 0.12);
    this.pending.push({
      kind: EV_SHIVER,
      index: (Math.random() * this.count) | 0,
      needsResolve: true,
      misc: [0.55 + 0.3 * Math.random(), 2.0 + Math.random(), 0, 0],
    });
  }

  private spawnWanderGroup(): void {
    const P = this.params;
    // groupSize is "fish per referenceCount" (HANDOFF §5: fractions, not
    // absolute counts). Recruit within rRecruit of the leader with
    // probability targetGroup / expectedInRadius; both scale with count, so
    // p doesn't. Large groups recruit a bigger pocket so they leave as a
    // sub-school, not a thinned-out cloud (2.8 at the reference's sizes).
    const rRecruit = 2.8 + 2.0 * Math.min(1, P.groupSize / 200);
    const schoolR = Math.max(2, P.containRadius * 0.85);
    const vSchool = (4 / 3) * Math.PI * schoolR ** 3;
    const vRecruit = (4 / 3) * Math.PI * rRecruit ** 3;
    const prob = Math.min(1, (P.groupSize * vSchool) / (GPU_TUNING.referenceCount * vRecruit));
    this.pending.push({
      kind: EV_WANDER,
      index: (Math.random() * this.count) | 0,
      needsResolve: true,
      misc: [6 + Math.random() * 7, prob, rRecruit, 0],
    });
  }

  // ------------------------------------------------------------------
  // Swarm director — verbatim port
  // ------------------------------------------------------------------

  private enterMood(mood: number): void {
    const P = this.params;
    const tempo = Math.max(0.25, P.tempo);
    const prev = this.mood;
    this.mood = mood;
    // Personal space is the reference's everywhere except the contraction
    // and the turn that releases it; the herd gain idles outside contract.
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
        if (Math.random() < 0.45) this.pickNewHeading(10, 40);
        break;
      case M_TURN:
        this.moodTimer = 1.2 + Math.random() * 0.8;
        this.tgtAlign = 1.6;
        this.tgtCoh = 0.95;
        this.tgtMill = 0;
        this.tgtDrive = 1.5;
        this.tgtSpeed = 0.95;
        // Released from a contraction, the school pops outward as it darts
        // onto the new heading.
        if (prev === M_CONTRACT) this.tgtSep = 1.25;
        this.pickNewHeading(70, 155);
        // Seed the turn through a vanguard cluster: one pocket darts onto
        // the new heading and alignment propagates it as a visible wave.
        this.pending.push({
          kind: EV_VANGUARD,
          index: (Math.random() * this.count) | 0,
          needsResolve: true,
        });
        break;
      case M_MILL:
        this.moodTimer = (6 + Math.random() * 7) / tempo;
        this.tgtAlign = 1.25;
        this.tgtCoh = 1.05;
        this.tgtMill = 1.1;
        this.tgtDrive = 0.4;
        this.tgtSpeed = 0.5;
        break;
      case M_HOVER:
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
        // slackens, and a slight swirl keeps the condensing body alive.
        // Personal space does NOT give: relaxing soft separation was tried
        // and let the fine collision grid dominate, so the condensed school
        // crushed into one glowing mass (the owner's "smushed" complaint)
        // and settled onto the grid lattice. Cohesion alone condenses; the
        // arrival damping (HANDOFF §4.1) bounds the squeeze.
        const c = P.collapse;
        this.tgtAlign = 0.7;
        this.tgtMill = 0.25;
        this.tgtDrive = 0.25;
        this.tgtSep = 1;
        if (this.contractFromAlarm) {
          // After a scare the whole school balls up, not just the fish that
          // fled: shorter, harder, and the herd drive surges.
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
  }

  private pickNextMood(): number {
    // Collapse diverts a share of cruise/mill exits into a contraction
    // (35% at collapse 1). No draw at 0: the reference FSM, untouched.
    const c = this.params.collapse;
    if (
      c > 0 &&
      (this.mood === M_CRUISE || this.mood === M_MILL) &&
      Math.random() < Math.min(0.7, 0.35 * c)
    ) {
      return M_CONTRACT;
    }
    const r = Math.random();
    switch (this.mood) {
      case M_CRUISE:
        return r < 0.5 ? M_TURN : r < 0.72 ? M_MILL : M_HOVER;
      case M_TURN:
        return M_CRUISE;
      case M_MILL:
        return r < 0.55 ? M_TURN : r < 0.75 ? M_HOVER : M_CRUISE;
      case M_HOVER:
        return r < 0.55 ? M_CRUISE : M_TURN;
      case M_CONTRACT:
        return r < 0.7 ? M_TURN : M_CRUISE;
      default:
        return M_CRUISE;
    }
  }

  private pickNewHeading(minDeg: number, maxDeg: number): void {
    const ang =
      (((minDeg + Math.random() * (maxDeg - minDeg)) * Math.PI) / 180) *
      (Math.random() < 0.5 ? -1 : 1);
    const c = Math.cos(ang);
    const s = Math.sin(ang);
    const hx = this.headX * c - this.headY * s;
    const hy = this.headX * s + this.headY * c;
    let hz = this.headZ + (Math.random() - 0.5) * 0.4;
    hz = Math.max(-0.3, Math.min(0.3, hz));
    const l = Math.sqrt(hx * hx + hy * hy + hz * hz) || 1;
    this.headX = hx / l;
    this.headY = hy / l;
    this.headZ = hz / l;
  }

  private updateDirector(dt: number, homeX: number, homeY: number): void {
    const P = this.params;

    // Spontaneous false alarms.
    this.startleClock -= dt;
    if (this.startleClock <= 0) {
      this.startleClock =
        (14 + Math.random() * 26) / Math.max(0.05, P.startle);
      if (P.startle > 0.02 && this.mood !== M_STARTLE) this.startle();
    }

    // Resting shivers: per-particle fear and a dart, never a mood change,
    // so the director keeps its course and nothing lights up.
    const shiver = P.shiver ?? 0;
    if (shiver > 0.02) {
      this.shiverClock -= dt;
      if (this.shiverClock <= 0) {
        this.shiverClock = (6 + Math.random() * 6) / Math.max(0.05, shiver);
        this.spawnShiver();
      }
    }

    this.moodTimer -= dt;
    if (this.moodTimer <= 0) {
      this.enterMood(this.mood === M_STARTLE ? M_CRUISE : this.pickNextMood());
    }

    // Alarm collapse: once a real scare subsides (alarm under half its
    // recent peak), the school collapses in on itself. Unreachable at
    // collapse 0, so the reference FSM stays untouched.
    this.alarm += (this.meanFear - this.alarm) * (1 - Math.exp(-dt / 0.15));
    this.alarmPeak += (this.alarm - this.alarmPeak) * (1 - Math.exp(-dt));
    if (this.alarm > this.alarmPeak) this.alarmPeak = this.alarm;
    this.alarmCooldown -= dt;
    if (
      P.collapse > 0 &&
      this.alarmPeak > 0.08 &&
      this.alarm < 0.5 * this.alarmPeak &&
      this.mood !== M_CONTRACT &&
      this.mood !== M_STARTLE &&
      this.alarmCooldown <= 0
    ) {
      this.contractFromAlarm = true;
      this.enterMood(M_CONTRACT);
      this.alarmPeak = this.alarm;
      this.alarmCooldown = 4;
    }

    // Bend the heading toward home when the school strays.
    const hx = homeX - this.centroid.x;
    const hy = homeY - this.centroid.y;
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
    this.dirSep += (this.tgtSep - this.dirSep) * k;
    // Snap home so an idle lever leaves the kernel's separation at exactly
    // the user's weight (the ease alone only approaches 1 asymptotically).
    if (Math.abs(this.tgtSep - this.dirSep) < 1e-4) this.dirSep = this.tgtSep;
    this.dirHerd += (this.tgtHerd - this.dirHerd) * k;
    if (Math.abs(this.tgtHerd - this.dirHerd) < 1e-4) this.dirHerd = this.tgtHerd;
  }

  // ------------------------------------------------------------------
  // Main update
  // ------------------------------------------------------------------

  update(deltaTime: number): void {
    if (!this.dyn) return;
    const P = this.params;
    // The panel writes params directly, so the share is polled. Ahead of
    // the dt check so a paused school still recolors.
    if (P.accentShare !== this.lastAccentShare) this.recolor();
    const dt = Math.min(deltaTime, 0.05) * P.timeScale;
    if (dt <= 0) return;
    this.time += dt;
    this.frame++;

    const t = this.time;

    // Frame constants (order matches the reference).
    const perception = Math.max(0.5, P.perception);
    const breatheR = P.containRadius * (1 + P.breathe * Math.sin(t * 0.45));
    const splashBoost = Math.min(2.5, this.mouseSpeed * P.splash * 0.6);
    const igniteRate = this.mouseSpeed * P.splash * 1.2;
    this.mouseSpeed *= Math.exp(-2.5 * dt);

    // World-space pointer velocity (smoothed) — the predator's heading.
    const px = this.pointer.x;
    const py = this.pointer.y;
    const pz = this.pointer.z;
    if (this.pointerActive) {
      if (this.hadPointerFrame) {
        const k = Math.min(1, dt * 8);
        this.pointerVel.x += ((px - this.prevPointerW.x) / dt - this.pointerVel.x) * k;
        this.pointerVel.y += ((py - this.prevPointerW.y) / dt - this.pointerVel.y) * k;
        this.pointerVel.z += ((pz - this.prevPointerW.z) / dt - this.pointerVel.z) * k;
        const m2 =
          this.pointerVel.x ** 2 + this.pointerVel.y ** 2 + this.pointerVel.z ** 2;
        if (m2 > 80 * 80) {
          const f = 80 / Math.sqrt(m2);
          this.pointerVel.x *= f;
          this.pointerVel.y *= f;
          this.pointerVel.z *= f;
        }
      }
      this.prevPointerW = { x: px, y: py, z: pz };
      this.hadPointerFrame = true;
    } else {
      this.hadPointerFrame = false;
      const f = Math.exp(-4 * dt);
      this.pointerVel.x *= f;
      this.pointerVel.y *= f;
      this.pointerVel.z *= f;
    }
    const pvm = Math.sqrt(
      this.pointerVel.x ** 2 + this.pointerVel.y ** 2 + this.pointerVel.z ** 2,
    );
    const vhx = pvm > 1e-3 ? this.pointerVel.x / pvm : 0;
    const vhy = pvm > 1e-3 ? this.pointerVel.y / pvm : 0;
    const vhz = pvm > 1e-3 ? this.pointerVel.z / pvm : 0;
    // The danger zone leads a moving predator.
    const lead = Math.min(pvm * 0.22, P.mouseRadius * 0.9);
    const tpx = px + vhx * lead;
    const tpy = py + vhy * lead;
    const tpz = pz + vhz * lead;

    // The strike: a flick — rest bursting into speed — detonates a
    // directional V-blast at the lunge tip (see GPU_TUNING.strikeSpeed).
    // Sustained fast movement never triggers it; the flick edge does.
    this.strikeCooldown -= dt;
    if (pvm < GPU_TUNING.strikeSpeed * 0.45) this.lastSlowTime = t;
    if (
      this.pointerActive &&
      pvm > GPU_TUNING.strikeSpeed &&
      t - this.lastSlowTime < GPU_TUNING.strikeFlickWindow &&
      this.strikeCooldown <= 0
    ) {
      this.strikeCooldown = GPU_TUNING.strikeCooldown;
      this.pending.push({
        kind: EV_STRIKE,
        center: [tpx, tpy, tpz],
        radius: P.mouseRadius,
        dir: [vhx, vhy, vhz],
      });
    }

    // Roaming home region.
    const roamR = (P.containRadius * 0.75 * P.roam) / 1.5;
    const targetX = (Math.sin(t * 0.13) + 0.5 * Math.sin(t * 0.23 + 2.1)) * roamR;
    const targetY =
      (Math.sin(t * 0.107 + 1.3) + 0.5 * Math.sin(t * 0.19)) * roamR * 0.8;
    const targetZ = Math.sin(t * 0.147 + 4.2) * roamR * 0.45;

    // Evade: the school dodges a parked predator as a body. Its herd
    // center and roam target shift away from the pointer and the heading
    // bends off it, so the threat ends up at the edge with the body
    // streaming past instead of a static shell around it.
    const cen = this.centroid;
    let evW = 0;
    let evNX = 0;
    let evNY = 0;
    let evNZ = 0;
    if (this.pointerActive && P.evade > 0) {
      const dx = cen.x - px;
      const dy = cen.y - py;
      const dz = cen.z - pz;
      // Depth-squashed metric, like the kernel's danger zone.
      const dist = Math.sqrt(dx * dx + dy * dy + dz * 0.45 * (dz * 0.45));
      evW =
        P.evade *
        Math.min(1, Math.max(0, 1 - dist / (P.containRadius * 0.9)));
      if (dist < 0.3) {
        // Pointer on the centroid: no direction to read, so step aside
        // perpendicular to the heading (stable frame to frame).
        const hl = Math.hypot(this.headX, this.headY) || 1;
        evNX = -this.headY / hl;
        evNY = this.headX / hl;
      } else {
        // Direction from the true offset; the squash only sets the range.
        const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
        evNX = dx / len;
        evNY = dy / len;
        evNZ = dz / len;
      }
    }
    const evTarget = P.containRadius * 0.7 * evW;
    // One pole (0.35 s) so the center never jumps; with the pointer gone
    // the target is 0 and it decays at the same rate.
    const ek = 1 - Math.exp(-dt / 0.35);
    this.evadeX += (evNX * evTarget - this.evadeX) * ek;
    this.evadeY += (evNY * evTarget - this.evadeY) * ek;
    this.evadeZ += (evNZ * evTarget - this.evadeZ) * ek;
    // Snap home so evade 0 is the reference exactly (the pole only
    // approaches zero).
    const evRest =
      Math.abs(this.evadeX) + Math.abs(this.evadeY) + Math.abs(this.evadeZ);
    if (!(P.evade > 0) && evRest < 1e-4) {
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

    // Spontaneous flashes: swarm-wide rate, NOT per-particle (HANDOFF §4.8).
    let spontAcc = P.spontaneous * dt;
    while (spontAcc > 0) {
      if (Math.random() < spontAcc) this.igniteSpontaneous();
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

    // ---- Pack uniforms (slot map in wgsl-sim.ts) ----
    const U = this.uniforms;
    U[0] = dt;
    U[1] = t;
    U[2] = this.frame % 1e6;
    U[3] = this.count;
    U[4] = this.headX;
    U[5] = this.headY;
    U[6] = this.headZ;
    // U[7] eventCount — set below.
    U[8] = this.dirAlign;
    U[9] = this.dirCoh;
    U[10] = P.schoolDrive * this.dirDrive;
    U[11] = this.dirSpeed;
    U[12] = (P.mill + this.dirMill) * 2.6;
    U[13] = 0.55 + this.dirSpeed * 1.3;
    U[14] = breatheR;
    U[15] = breatheR * 2.1;
    U[16] = roamX;
    U[17] = roamY;
    U[18] = roamZ;
    // Fine-entries offset inside gridData (capacity-dependent; exact in f32).
    U[19] = FINE_CELLS + 1 + CELLS + 1 + this.capacity;
    U[20] = px;
    U[21] = py;
    U[22] = pz;
    U[23] = this.pointerActive ? 1 : 0;
    U[24] = tpx;
    U[25] = tpy;
    U[26] = tpz;
    U[27] = pvm;
    U[28] = vhx;
    U[29] = vhy;
    U[30] = vhz;
    U[31] = P.mouseRadius;
    U[32] = splashBoost;
    U[33] = igniteRate;
    U[34] = perception;
    U[35] = 1 / perception;
    U[36] = perception * TUNING.separationRatio;
    U[37] = perception * TUNING.arriveRatio;
    U[38] = P.cohesion;
    U[39] = P.alignment;
    U[40] = P.separation;
    U[41] = P.maxSpeed;
    U[42] = Math.exp(-P.drag * dt);
    U[43] = Math.exp(-P.flashDecay * dt);
    U[44] = Math.exp(-P.fearDecay * dt);
    U[45] = Math.min(1, P.fearSpread * GPU_TUNING.fearSpreadScale * dt);
    U[46] = Math.min(1, dt * 5);
    U[47] = P.transmission * GPU_TUNING.transmissionScale;
    U[48] = P.refractory * GPU_TUNING.refractoryScale;
    U[49] = P.kick;
    U[50] = P.flowStrength;
    U[51] = t * 0.6;
    U[52] = P.flowScale;
    U[53] = P.flowScale * 1.33;
    U[54] = P.flowScale * 0.77;
    U[55] = P.swirl;
    U[56] = P.jitter;
    U[57] = P.wanderPull;
    U[58] = P.returnPull;
    U[59] = P.repelForce;
    U[60] = P.fleeForce;
    U[61] = P.regroup;
    U[62] = P.containRadius;
    U[63] = P.gait;
    U[64] = this.dirSep;
    U[65] = P.shape;
    // Clamped below 1: WGSL smoothstep is undefined when its edges meet.
    U[66] = Math.min(0.9, Math.max(0, P.bite ?? 0));
    U[67] = P.flight ?? 0;
    U[68] = P.rally ?? 0;
    U[69] = P.herd ?? 0;
    U[70] = this.dirHerd;
    U[71] = P.evade ?? 0;
    U[72] = cen.x + this.evadeX;
    U[73] = cen.y + this.evadeY;
    U[74] = cen.z + this.evadeZ;
    U[75] = P.sheet ?? 0;

    // ---- Pack events ----
    const batch = this.pending.splice(0, MAX_EVENTS);
    U[7] = batch.length;
    this.eventU32.fill(0);
    for (let k = 0; k < batch.length; k++) {
      const ev = batch[k];
      const b = k * EVENT_STRIDE;
      this.eventU32[b] = ev.kind;
      this.eventU32[b + 1] = ev.index ?? 0;
      this.eventU32[b + 2] = ev.needsResolve ? 1 : 0;
      const c = ev.center ?? [0, 0, 0];
      this.eventF32[b + 4] = c[0];
      this.eventF32[b + 5] = c[1];
      this.eventF32[b + 6] = c[2];
      this.eventF32[b + 7] = ev.radius ?? 0;
      const dir = ev.dir ?? [0, 0, 0];
      this.eventF32[b + 8] = dir[0];
      this.eventF32[b + 9] = dir[1];
      this.eventF32[b + 10] = dir[2];
      this.eventF32[b + 11] = ev.sign ?? 1;
      const misc = ev.misc ?? [0, 0, 0, 0];
      this.eventF32[b + 12] = misc[0];
      this.eventF32[b + 13] = misc[1];
      this.eventF32[b + 14] = misc[2];
      this.eventF32[b + 15] = misc[3];
    }

    const q = this.device.queue;
    q.writeBuffer(this.uniformsB, 0, U);
    q.writeBuffer(this.eventsB, 0, this.eventData);

    // ---- Encode the frame ----
    const g = this.groups;
    const pr = this.parity as 0 | 1;
    const enc = this.device.createCommandEncoder({ label: "sim-frame" });
    const pass = enc.beginComputePass({ label: "sim" });
    const run = (
      pipeline: GPUComputePipeline,
      group: GPUBindGroup,
      workgroups: number,
    ) => {
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(workgroups);
    };
    const nWg256 = Math.ceil(this.count / 256);
    run(this.pipelines.clear, g.clear, Math.ceil((CELLS + FINE_CELLS) / 256));
    run(this.pipelines.count, g.count[pr], nWg256);
    run(this.pipelines.scan, g.scan, 1);
    run(this.pipelines.scanFine, g.scanFine, 1);
    run(this.pipelines.scatter, g.scatter[pr], nWg256);
    if (batch.length > 0) run(this.pipelines.resolve, g.resolve[pr], 1);
    run(this.pipelines.sim, g.sim[pr], Math.ceil(this.count / WG));
    run(this.pipelines.reduce1, g.reduce[pr], REDUCE_WGS);
    run(this.pipelines.reduce2, g.reduce2, 1);
    pass.end();

    // Centroid + stats readback (ring; skipped while all buffers are busy).
    const st = this.staging[this.frame % 3];
    if (st && st.state === "free") {
      enc.copyBufferToBuffer(this.centroidB, 0, st.buf, 0, 32);
      st.state = "pending";
    }
    q.submit([enc.finish()]);

    if (st && st.state === "pending") {
      st.state = "mapping";
      st.buf
        .mapAsync(GPUMapMode.READ)
        .then(() => {
          const v = new Float32Array(st.buf.getMappedRange());
          this.centroid = { x: v[0], y: v[1], z: v[2] };
          this.flockN = v[3];
          this.meanSpeed = v[4] / Math.max(1, this.count);
          this.hotGlowCount = v[5];
          // [1].z = Σ|pos − cen|² over flock fish (see WGSL_REDUCE).
          this.rmsRadius = Math.sqrt(Math.max(0, v[6]) / Math.max(1, v[3]));
          // [1].w = Σfear over active fish.
          this.meanFear = v[7] / Math.max(1, this.count);
          st.buf.unmap();
          st.state = "free";
        })
        .catch(() => {
          st.state = "free"; // buffer destroyed mid-map (rebuild)
        });
    }

    this.parity = 1 - this.parity;
  }

  /** Buffers holding the *latest* particle state (after update's swap). */
  currentDyn(): GPUBuffer {
    return this.dyn![this.parity];
  }
  /** Which of dynBuffers() is current — renderer picks its bind group by it. */
  currentParity(): 0 | 1 {
    return this.parity as 0 | 1;
  }
  dynBuffers(): [GPUBuffer, GPUBuffer] {
    return this.dyn!;
  }
  selfBuffer(): GPUBuffer {
    return this.selfB;
  }
  staticsBuffer(): GPUBuffer {
    return this.statB;
  }
  /** Grid buffer; fine-cell starts sit at offset 0 (crowd dim reads them). */
  gridDataBuffer(): GPUBuffer {
    return this.gridData;
  }
  /** Simulation clock — drives twinkle/blink phases in the renderer. */
  get simTime(): number {
    return this.time;
  }

  /** Await all submitted GPU work (harness timing). */
  waitIdle(): Promise<undefined> {
    return this.device.queue.onSubmittedWorkDone();
  }

  /**
   * Pull the full sim state back to the host — the harness's probe window.
   * De-interleaved into the same tight arrays the CPU harness reads.
   */
  async readState(): Promise<GpuStateSnapshot> {
    const n = this.count;
    const dev = this.device;
    const dynStage = dev.createBuffer({
      label: "readState-dyn",
      size: n * 48,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const selfStage = dev.createBuffer({
      label: "readState-self",
      size: n * 32,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const enc = dev.createCommandEncoder();
    enc.copyBufferToBuffer(this.currentDyn(), 0, dynStage, 0, n * 48);
    enc.copyBufferToBuffer(this.selfB, 0, selfStage, 0, n * 32);
    dev.queue.submit([enc.finish()]);
    await Promise.all([
      dynStage.mapAsync(GPUMapMode.READ),
      selfStage.mapAsync(GPUMapMode.READ),
    ]);
    const dyn = new Float32Array(dynStage.getMappedRange());
    const selfD = new Float32Array(selfStage.getMappedRange());

    const out: GpuStateSnapshot = {
      pos: new Float32Array(n * 3),
      vel: new Float32Array(n * 3),
      energy: new Float32Array(n),
      glow: new Float32Array(n),
      fear: new Float32Array(n),
      pulse: new Float32Array(n * 3),
      pulseSign: new Float32Array(n),
      mode: new Uint8Array(n),
      pointer: { ...this.pointer },
    };
    for (let i = 0; i < n; i++) {
      const d = i * 12;
      out.pos[i * 3] = dyn[d];
      out.pos[i * 3 + 1] = dyn[d + 1];
      out.pos[i * 3 + 2] = dyn[d + 2];
      out.energy[i] = dyn[d + 3];
      out.vel[i * 3] = dyn[d + 4];
      out.vel[i * 3 + 1] = dyn[d + 5];
      out.vel[i * 3 + 2] = dyn[d + 6];
      out.fear[i] = dyn[d + 7];
      out.pulse[i * 3] = dyn[d + 8];
      out.pulse[i * 3 + 1] = dyn[d + 9];
      out.pulse[i * 3 + 2] = dyn[d + 10];
      out.pulseSign[i] = dyn[d + 11];
      out.glow[i] = selfD[i * 8];
      out.mode[i] = selfD[i * 8 + 7] & 3;
    }
    dynStage.destroy();
    selfStage.destroy();
    return out;
  }

  dispose(): void {
    this.destroyBuffers();
  }
}

/** Small seeded PRNG: the same seed replays the same color draws. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function hexToRgb(hex: string): [number, number, number] {
  const v = parseInt(hex.slice(1), 16);
  return [((v >> 16) & 255) / 255, ((v >> 8) & 255) / 255, (v & 255) / 255];
}

export interface GpuApp {
  system: GpuSwarm;
  render: () => void;
  resize: (width: number, height: number, dpr: number) => void;
  dispose: () => void;
}

/** Boot the WebGPU backend: device, sim, canvas, renderer. Throws if the
 * device can't be created — the caller falls back to the CPU backend.
 *
 * `lost` resolves when the device goes away under us (driver reset, GPU
 * process crash, tab backgrounded on some mobile GPUs, or a debugging
 * `device.destroy()` from the console) and never resolves for our own
 * `dispose()` - so the caller can re-mount on it without filtering out
 * intentional teardown. `device` is exposed for that debugging poke. */
export async function createGpuApp(
  container: HTMLElement,
  adapter: GPUAdapter,
  count: number,
): Promise<
  GpuApp & { device: GPUDevice; lost: Promise<GPUDeviceLostInfo> }
> {
  const device = await adapter.requestDevice({ label: "bioluminescent" });
  device.addEventListener("uncapturederror", (e) => {
    console.error("[webgpu]", (e as GPUUncapturedErrorEvent).error.message);
  });
  let disposing = false;
  const lost = new Promise<GPUDeviceLostInfo>((resolve) => {
    device.lost.then((info) => {
      if (!disposing) resolve(info);
    });
  });

  const canvas = document.createElement("canvas");
  container.appendChild(canvas);

  const width = window.innerWidth;
  const height = window.innerHeight;
  let system: GpuSwarm;
  let renderer: GpuRenderer;
  try {
    system = new GpuSwarm(device, width / Math.max(1, height), count);
    renderer = new GpuRenderer(device, canvas, system);
    renderer.resize(width, height, window.devicePixelRatio || 1);
  } catch (err) {
    // The caller falls back to the CPU backend: leave no dead canvas over
    // it and no live device behind it.
    disposing = true;
    canvas.remove();
    device.destroy();
    throw err;
  }

  return {
    system,
    device,
    lost,
    render: () => renderer.render(),
    resize: (w, h, dpr) => {
      system.updateRendererSize(w / 2, h / 2);
      renderer.resize(w, h, dpr);
    },
    dispose: () => {
      disposing = true;
      renderer.dispose(); // also removes the canvas
      system.dispose();
      device.destroy();
    },
  };
}
