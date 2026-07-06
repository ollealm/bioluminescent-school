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
  WGSL_GRID,
  WGSL_REDUCE,
  WGSL_RESOLVE,
  WGSL_SCAN,
  WGSL_SCATTER,
  WGSL_SIM,
} from "./wgsl-sim";
import { GpuRenderer } from "./renderer";

// School moods (whole-swarm director) — identical to the CPU reference.
const M_CRUISE = 0;
const M_TURN = 1;
const M_MILL = 2;
const M_HOVER = 3;
const M_STARTLE = 4;
const MOOD_NAMES = ["cruise", "turn", "mill", "hover", "startle"];

// Event kinds — mirror the EV_* constants in wgsl-sim.ts.
const EV_BURST = 1;
const EV_STARTLE = 2;
const EV_VANGUARD = 3;
const EV_WANDER = 4;
const EV_SCATTER = 5;
const EV_IGNITE = 6;

const D = TUNING.gridDim;
const CELLS = D * D * D;
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

  // --- GPU resources (rebuilt with the particle count) ---
  private dyn: [GPUBuffer, GPUBuffer] | null = null;
  private selfB!: GPUBuffer;
  private statB!: GPUBuffer;
  private gridCount!: GPUBuffer;
  private gridStart!: GPUBuffer;
  private gridCursor!: GPUBuffer;
  private gridEntries!: GPUBuffer;
  private eventsB!: GPUBuffer;
  private uniformsB!: GPUBuffer;
  private centroidB!: GPUBuffer;
  private parity = 0; // dyn[parity] is the read side this frame

  private pipelines!: {
    clear: GPUComputePipeline;
    count: GPUComputePipeline;
    scan: GPUComputePipeline;
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
    scatter: [GPUBindGroup, GPUBindGroup];
    resolve: [GPUBindGroup, GPUBindGroup];
    sim: [GPUBindGroup, GPUBindGroup];
    reduce: [GPUBindGroup, GPUBindGroup];
    reduce2: GPUBindGroup;
  };

  // Host mirrors for statics so palette switches don't regenerate sizes.
  private staticsHost!: Float32Array;

  private uniforms = new Float32Array(64);
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

  // --- Swarm director state (identical numbers to the reference) ---
  private mood = M_CRUISE;
  private moodTimer = 6;
  private headX = 1;
  private headY = 0;
  private headZ = 0;
  private startleClock = 25;
  private dirAlign = 1;
  private dirCoh = 1;
  private dirMill = 0;
  private dirDrive = 1;
  private dirSpeed = 0.6;
  private tgtAlign = 1;
  private tgtCoh = 1;
  private tgtMill = 0;
  private tgtDrive = 1;
  private tgtSpeed = 0.6;

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

  /** GPU-reduced frame stats (mean |vel|, count of glow ≥ 0.8); ~2 frames stale. */
  stats(): { meanSpeed: number; hotGlow: number } {
    return { meanSpeed: this.meanSpeed, hotGlow: this.hotGlowCount };
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
    if (this.staticsHost) {
      this.regenerateColors();
      this.device.queue.writeBuffer(this.statB, 0, this.staticsHost);
    }
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
      scatter: mk(WGSL_SCATTER, "scatterGrid", "scatterGrid"),
      resolve: mk(WGSL_RESOLVE, "resolveEvents", "resolveEvents"),
      sim: mk(WGSL_SIM, "simStep", "simStep"),
      reduce1: mk(WGSL_REDUCE, "reducePass1", "reducePass1"),
      reduce2: mk(WGSL_REDUCE, "reducePass2", "reducePass2"),
    };
  }

  /** Recreate all buffers for a new particle count. Params are untouched. */
  rebuild(count: number): void {
    count = Math.max(64, Math.floor(count));
    this.destroyBuffers();
    this.count = count;
    const dev = this.device;

    const mkStorage = (bytes: number, label: string, extra: number = 0) =>
      dev.createBuffer({
        label,
        size: bytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | extra,
      });

    this.dyn = [
      mkStorage(count * 48, "dynA"),
      mkStorage(count * 48, "dynB"),
    ];
    this.selfB = mkStorage(count * 32, "self");
    this.statB = mkStorage(count * 32, "stat");
    this.gridCount = mkStorage(CELLS * 4, "gridCount");
    this.gridStart = mkStorage((CELLS + 1) * 4, "gridStart");
    this.gridCursor = mkStorage(CELLS * 4, "gridCursor");
    this.gridEntries = mkStorage(count * 4, "gridEntries");
    this.eventsB = mkStorage(this.eventData.byteLength, "events");
    // [0] centroid+flockN, [1] stats, then 2×64 partial blocks.
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
    this.generation++;
  }

  /** Port of the reference's spawn: soft sphere + a few shell whiskers. */
  private initParticles(): void {
    const n = this.count;
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
    this.regenerateColors();

    const q = this.device.queue;
    q.writeBuffer(this.dyn![0], 0, dyn);
    q.writeBuffer(this.dyn![1], 0, dyn); // both sides start identical
    q.writeBuffer(this.selfB, 0, selfD);
    q.writeBuffer(this.statB, 0, this.staticsHost);
    q.writeBuffer(this.centroidB, 0, new Float32Array(4 * (2 + 2 * REDUCE_WGS)));
  }

  /** Port of applyPaletteColors — writes rgb lanes of the statics mirror. */
  private regenerateColors(): void {
    const palette: Palette = PALETTES[this.paletteName];
    const stops = palette.stops.map(hexToRgb);
    const accent = hexToRgb(palette.accent);
    for (let i = 0; i < this.count; i++) {
      let rgb: [number, number, number];
      if (Math.random() < palette.accentChance) {
        rgb = [...accent];
      } else {
        const t = Math.pow(Math.random(), 0.8) * (stops.length - 1);
        const idx = Math.min(stops.length - 2, Math.floor(t));
        const f = t - idx;
        rgb = [
          stops[idx][0] + (stops[idx + 1][0] - stops[idx][0]) * f,
          stops[idx][1] + (stops[idx + 1][1] - stops[idx][1]) * f,
          stops[idx][2] + (stops[idx + 1][2] - stops[idx][2]) * f,
        ];
      }
      const st = i * 8;
      this.staticsHost[st] = Math.min(1, rgb[0] * (0.9 + Math.random() * 0.2));
      this.staticsHost[st + 1] = Math.min(1, rgb[1] * (0.9 + Math.random() * 0.2));
      this.staticsHost[st + 2] = Math.min(1, rgb[2] * (0.9 + Math.random() * 0.2));
    }
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
      clear: bg(p.clear, [null, null, this.gridCount]),
      count: both(p.count, (dynIn) => [this.uniformsB, dynIn, this.gridCount]),
      scan: bg(p.scan, [this.gridCount, this.gridStart, this.gridCursor]),
      scatter: both(p.scatter, (dynIn) => [
        this.uniformsB,
        dynIn,
        this.gridCursor,
        this.gridEntries,
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
        this.gridStart,
        this.gridEntries,
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
      this.gridCount,
      this.gridStart,
      this.gridCursor,
      this.gridEntries,
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

    this.pointer = this.camera.unprojectToSwarmPlane(ndc);
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

  private spawnWanderGroup(): void {
    const P = this.params;
    // groupSize is "fish per referenceCount" (HANDOFF §5: fractions, not
    // absolute counts). Recruit within r=2.8 of the leader with probability
    // targetGroup / expectedInRadius; both scale with count, so p doesn't.
    const schoolR = Math.max(2, P.containRadius * 0.85);
    const vSchool = (4 / 3) * Math.PI * schoolR ** 3;
    const vRecruit = (4 / 3) * Math.PI * 2.8 ** 3;
    const prob = Math.min(1, (P.groupSize * vSchool) / (GPU_TUNING.referenceCount * vRecruit));
    this.pending.push({
      kind: EV_WANDER,
      index: (Math.random() * this.count) | 0,
      needsResolve: true,
      misc: [6 + Math.random() * 7, prob, 0, 0],
    });
  }

  // ------------------------------------------------------------------
  // Swarm director — verbatim port
  // ------------------------------------------------------------------

  private enterMood(mood: number): void {
    const P = this.params;
    const tempo = Math.max(0.25, P.tempo);
    this.mood = mood;

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
    }
  }

  private pickNextMood(): number {
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

    this.moodTimer -= dt;
    if (this.moodTimer <= 0) {
      this.enterMood(this.mood === M_STARTLE ? M_CRUISE : this.pickNextMood());
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
  }

  // ------------------------------------------------------------------
  // Main update
  // ------------------------------------------------------------------

  update(deltaTime: number): void {
    if (!this.dyn) return;
    const P = this.params;
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

    // Roaming home region.
    const roamR = (P.containRadius * 0.75 * P.roam) / 1.5;
    const targetX = (Math.sin(t * 0.13) + 0.5 * Math.sin(t * 0.23 + 2.1)) * roamR;
    const targetY =
      (Math.sin(t * 0.107 + 1.3) + 0.5 * Math.sin(t * 0.19)) * roamR * 0.8;
    const targetZ = Math.sin(t * 0.147 + 4.2) * roamR * 0.45;

    // The school's mind: moods, heading, false alarms.
    this.updateDirector(dt, targetX, targetY);

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
    U[16] = targetX;
    U[17] = targetY;
    U[18] = targetZ;
    U[19] = 0;
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
    run(this.pipelines.clear, g.clear, Math.ceil(CELLS / 256));
    run(this.pipelines.count, g.count[pr], nWg256);
    run(this.pipelines.scan, g.scan, 1);
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
 * device can't be created — the caller falls back to the CPU backend. */
export async function createGpuApp(
  container: HTMLElement,
  adapter: GPUAdapter,
  count: number,
): Promise<GpuApp> {
  const device = await adapter.requestDevice({ label: "bioluminescent" });
  device.addEventListener("uncapturederror", (e) => {
    console.error("[webgpu]", (e as GPUUncapturedErrorEvent).error.message);
  });

  const canvas = document.createElement("canvas");
  container.appendChild(canvas);

  const width = window.innerWidth;
  const height = window.innerHeight;
  const system = new GpuSwarm(device, width / Math.max(1, height), count);
  const renderer = new GpuRenderer(device, canvas, system);
  renderer.resize(width, height, window.devicePixelRatio || 1);

  return {
    system,
    render: () => renderer.render(),
    resize: (w, h, dpr) => {
      system.updateRendererSize(w / 2, h / 2);
      renderer.resize(w, h, dpr);
    },
    dispose: () => {
      renderer.dispose();
      system.dispose();
      device.destroy();
    },
  };
}
