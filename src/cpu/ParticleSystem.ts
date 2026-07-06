import * as THREE from "three";
import {
  DEFAULT_PARAMS,
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

// Behavioral modes (per particle)
const FLOCK = 0;
const WANDER = 1;
const RETURN = 2;

// School moods (whole-swarm director)
const M_CRUISE = 0;
const M_TURN = 1;
const M_MILL = 2;
const M_HOVER = 3;
const M_STARTLE = 4;

const MOOD_NAMES = ["cruise", "turn", "mill", "hover", "startle"];

// Spatial grid: dense, directly indexed. D³ cells of edge `perception`,
// centered on the origin; positions outside clamp to border cells (distance
// checks still filter, so far-flung wanderers stay correct, just unbucketed).
const D = TUNING.gridDim;
const D2 = D * D;
const CELLS = D * D * D;
const MAX_NEIGHBORS = TUNING.maxNeighbors;
const MAX_FORCE = TUNING.maxForce;

function detectMobile(): boolean {
  if (typeof window === "undefined") return false;
  return (
    window.matchMedia?.("(pointer: coarse)").matches ||
    window.innerWidth < 768
  );
}

export interface ParticleSystemOptions {
  /** Camera used to project the pointer onto the swarm plane (z=0). */
  camera?: THREE.PerspectiveCamera;
  /** Particle count. Defaults to 2600 desktop / 1400 mobile. */
  count?: number;
  /** Starting presets per category, e.g. { motion: "shoal", look: "reef" }. */
  presets?: Partial<Record<PresetCategory, string>>;
}

/** Bump when the snapshot shape or param semantics change incompatibly. */
export const SNAPSHOT_VERSION = 3;

export interface Snapshot {
  /** Format version — lets forks and future versions port tuned settings. */
  version: number;
  presets: Record<PresetCategory, string>;
  palette: PaletteName;
  style: StyleName;
  count: number;
  params: SwarmParams;
}

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
 */
export class ParticleSystem {
  /** Live-tweakable parameters (the dev panel binds directly to this). */
  readonly params: SwarmParams;
  presetNames: Record<PresetCategory, string> = { ...DEFAULT_PRESETS };
  paletteName: PaletteName;
  styleName: StyleName;

  private scene: THREE.Scene;
  private camera?: THREE.PerspectiveCamera;
  private count = 0;

  private geometry!: THREE.BufferGeometry;
  private material: THREE.ShaderMaterial;
  private points!: THREE.Points;

  // Simulation state — pos/vel/energy/glow alias the GPU attribute arrays.
  private pos!: Float32Array;
  private vel!: Float32Array;
  private energy!: Float32Array;
  private glow!: Float32Array;
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
  // Individuality
  private speedFactor!: Float32Array;
  private social!: Float32Array;
  private gaitClock!: Float32Array;

  // Spatial grid (counting sort, rebuilt every frame)
  private gridCount = new Int32Array(CELLS);
  private gridStart = new Int32Array(CELLS + 1);
  private gridCursor = new Int32Array(CELLS);
  private gridEntries!: Int32Array;
  private cellOf!: Int32Array;

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
  private wanderClock = 5;

  // --- Swarm director state ---
  private mood = M_CRUISE;
  private moodTimer = 6;
  private headX = 1;
  private headY = 0;
  private headZ = 0;
  private startleClock = 25;
  // Smoothed mood multipliers (live) and their targets
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
    const palette = PALETTES[name];
    (this.material.uniforms.uFlashColor.value as THREE.Color).set(
      palette.flash,
    );
    if (this.geometry) this.applyPaletteColors();
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
    if (s.palette) this.setPalette(s.palette);
    if (s.style) this.setStyle(s.style);
    if (s.count && s.count !== this.count) this.rebuild(s.count);
  }

  /** Current director mood, for harnesses and debugging. */
  get moodName(): string {
    return MOOD_NAMES[this.mood];
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
      attribute vec3 aVel;

      uniform float uTime;
      uniform float uSize;
      uniform float uDpr;
      uniform float uTwinkle;
      uniform float uStretch;

      varying vec3 vColor;
      varying float vEnergy;
      varying float vSeed;
      varying float vTw;
      varying float vGlow;
      varying vec2 vDir;
      varying float vStretch;

      void main() {
        vColor = color;
        vEnergy = aEnergy;
        vSeed = aSeed;
        vGlow = aGlow;

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

        float size = aSize * uSize * (1.0 + aEnergy * 0.45) * (0.72 + 0.28 * tw);
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
        uMotionGlow: { value: this.params.motionGlow },
        uStyle: { value: 0 },
        uOpacity: { value: this.params.opacity },
        uFlashBright: { value: this.params.flashBrightness },
        uFlashColor: { value: new THREE.Color(PALETTES[this.paletteName].flash) },
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

  /** Recreate all buffers for a new particle count. Params are untouched. */
  rebuild(count: number): void {
    count = Math.max(64, Math.floor(count));
    if (this.points) {
      this.scene.remove(this.points);
      this.geometry.dispose();
    }

    this.count = count;
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);
    const sizes = new Float32Array(count);
    const energies = new Float32Array(count);
    const glows = new Float32Array(count);
    const seeds = new Float32Array(count);

    this.vel = new Float32Array(count * 3);
    this.refract = new Float32Array(count);
    this.fired = new Uint8Array(count);
    this.mode = new Uint8Array(count);
    this.modeTimer = new Float32Array(count);
    this.wanderDir = new Float32Array(count * 3);
    this.fear = new Float32Array(count);
    this.pulse = new Float32Array(count * 3);
    this.pulseSign = new Float32Array(count);
    this.speedFactor = new Float32Array(count);
    this.social = new Float32Array(count);
    this.gaitClock = new Float32Array(count);
    this.gridEntries = new Int32Array(count);
    this.cellOf = new Int32Array(count);
    this.seeds = seeds;

    const R = this.params.containRadius;
    for (let i = 0; i < count; i++) {
      // Spawn in a soft sphere; a few in an outer shell for loose whiskers.
      const shell = Math.random() < 0.15;
      const r = shell
        ? R * (0.9 + Math.random() * 0.5)
        : R * 0.9 * Math.cbrt(Math.random());
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
    // The physics velocity array doubles as the GPU attribute for stretch.
    this.geometry.setAttribute("aVel", new THREE.BufferAttribute(this.vel, 3));

    this.pos = positions;
    this.energy = energies;
    this.glow = glows;

    this.applyPaletteColors();

    this.points = new THREE.Points(this.geometry, this.material);
    this.points.frustumCulled = false;
    this.scene.add(this.points);
  }

  private applyPaletteColors(): void {
    const palette: Palette = PALETTES[this.paletteName];
    const stops = palette.stops.map((c) => new THREE.Color(c));
    const accent = new THREE.Color(palette.accent);
    const colors = this.geometry.attributes.color.array as Float32Array;
    const tmp = new THREE.Color();

    for (let i = 0; i < this.count; i++) {
      if (Math.random() < palette.accentChance) {
        tmp.copy(accent);
      } else {
        // Sample the gradient, biased toward the brighter end.
        const t = Math.pow(Math.random(), 0.8) * (stops.length - 1);
        const idx = Math.min(stops.length - 2, Math.floor(t));
        tmp.copy(stops[idx]).lerp(stops[idx + 1], t - idx);
      }
      colors[i * 3] = Math.min(1, tmp.r * (0.9 + Math.random() * 0.2));
      colors[i * 3 + 1] = Math.min(1, tmp.g * (0.9 + Math.random() * 0.2));
      colors[i * 3 + 2] = Math.min(1, tmp.b * (0.9 + Math.random() * 0.2));
    }
    this.geometry.attributes.color.needsUpdate = true;
  }

  // ------------------------------------------------------------------
  // Pointer
  // ------------------------------------------------------------------

  /** NDC coordinates in [-1, 1]. Same signature as the old system. */
  updateMousePosition(ndc: THREE.Vector2): void {
    const now =
      typeof performance !== "undefined" ? performance.now() : Date.now();

    if (this.hasPrevNDC) {
      const dt = Math.max(0.008, (now - this.lastPointerTime) / 1000);
      const dx = ndc.x - this.prevNDC.x;
      const dy = ndc.y - this.prevNDC.y;
      const speed = Math.min(8, Math.sqrt(dx * dx + dy * dy) / dt);
      this.mouseSpeed += (speed - this.mouseSpeed) * 0.35;
    }
    this.prevNDC.copy(ndc);
    this.lastPointerTime = now;
    this.hasPrevNDC = true;
    this.pointerActive = true;

    if (this.camera) {
      // Project the pointer ray onto the z=0 plane the swarm lives on.
      const v = new THREE.Vector3(ndc.x, ndc.y, 0.5).unproject(this.camera);
      const dir = v.sub(this.camera.position).normalize();
      if (Math.abs(dir.z) > 1e-4) {
        const t = -this.camera.position.z / dir.z;
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
  burst(at?: THREE.Vector3): void {
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
  startle(at?: THREE.Vector3): void {
    let ex: number, ey: number, ez: number;
    if (at) {
      ex = at.x; ey = at.y; ez = at.z;
    } else {
      const i = (Math.random() * this.count) | 0;
      ex = this.pos[i * 3]; ey = this.pos[i * 3 + 1]; ez = this.pos[i * 3 + 2];
    }
    const R = 3.4;
    for (let i = 0; i < this.count; i++) {
      const dx = this.pos[i * 3] - ex;
      const dy = this.pos[i * 3 + 1] - ey;
      const dz = this.pos[i * 3 + 2] - ez;
      const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (d > R) continue;
      const f = 0.25 + 0.75 * (1 - d / R);
      if (f > this.fear[i]) this.fear[i] = f;
      this.glow[i] = Math.max(this.glow[i], f * 0.8);
      if (d < 1.8 && Math.random() < 0.5) {
        this.igniteIndex(i, false, ex, ey, ez, 1);
      }
    }
    this.enterMood(M_STARTLE);
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
    this.mood = mood;

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
      case M_TURN:
        this.moodTimer = 1.2 + Math.random() * 0.8;
        this.tgtAlign = 1.6;
        this.tgtCoh = 0.95;
        this.tgtMill = 0;
        this.tgtDrive = 1.5;
        this.tgtSpeed = 0.95;
        this.pickNewHeading(70, 155);
        this.seedTurnVanguard();
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
      ((minDeg + Math.random() * (maxDeg - minDeg)) * Math.PI) / 180 *
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

  /**
   * Seed a turn through a vanguard: one cluster darts onto the new heading
   * and alignment cascades the turn through the school as a visible wave.
   */
  private seedTurnVanguard(): void {
    const leader = (Math.random() * this.count) | 0;
    const lx = this.pos[leader * 3];
    const ly = this.pos[leader * 3 + 1];
    const lz = this.pos[leader * 3 + 2];
    const r2 = 3.5 * 3.5;
    for (let i = 0; i < this.count; i++) {
      const dx = this.pos[i * 3] - lx;
      const dy = this.pos[i * 3 + 1] - ly;
      const dz = this.pos[i * 3 + 2] - lz;
      if (dx * dx + dy * dy + dz * dz > r2) continue;
      const dart = 2.0 + Math.random() * 1.4;
      this.vel[i * 3] += this.headX * dart;
      this.vel[i * 3 + 1] += this.headY * dart;
      this.vel[i * 3 + 2] += this.headZ * dart * 0.6;
      this.gaitClock[i] = 0;
      this.glow[i] = Math.max(this.glow[i], 0.5);
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

    this.moodTimer -= dt;
    if (this.moodTimer <= 0) {
      this.enterMood(this.mood === M_STARTLE ? M_CRUISE : this.pickNextMood());
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
    const recruitR2 = 2.8 * 2.8;
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
    this.gridCount.fill(0);
    for (let i = 0; i < n; i++) {
      const idx =
        ParticleSystem.cellCoord(this.pos[i * 3], invCell) +
        ParticleSystem.cellCoord(this.pos[i * 3 + 1], invCell) * D +
        ParticleSystem.cellCoord(this.pos[i * 3 + 2], invCell) * D2;
      this.cellOf[i] = idx;
      this.gridCount[idx]++;
    }
    let sum = 0;
    for (let c = 0; c < CELLS; c++) {
      this.gridStart[c] = sum;
      this.gridCursor[c] = sum;
      sum += this.gridCount[c];
    }
    this.gridStart[CELLS] = sum;
    for (let i = 0; i < n; i++) {
      this.gridEntries[this.gridCursor[this.cellOf[i]]++] = i;
    }
  }

  // ------------------------------------------------------------------
  // Main update
  // ------------------------------------------------------------------

  update(deltaTime: number): void {
    const P = this.params;
    const dt = Math.min(deltaTime, 0.05) * P.timeScale;
    if (dt <= 0) return;
    this.time += dt;

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

    // Roaming home region: a slow-wandering anchor the school patrols around.
    const roamR = (P.containRadius * 0.75 * P.roam) / 1.5;
    const targetX =
      (Math.sin(t * 0.13) + 0.5 * Math.sin(t * 0.23 + 2.1)) * roamR;
    const targetY =
      (Math.sin(t * 0.107 + 1.3) + 0.5 * Math.sin(t * 0.19)) * roamR * 0.8;
    const targetZ = Math.sin(t * 0.147 + 4.2) * roamR * 0.45;

    // The school's mind: moods, heading, false alarms.
    this.updateDirector(dt, targetX, targetY);
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

    // Group-level recentering: gentle, the cruise dynamics do the framing.
    const reCenterX = (targetX - this.centroidX) * 0.22;
    const reCenterY = (targetY - this.centroidY) * 0.22;
    const reCenterZ = (targetZ - this.centroidZ) * 0.22;
    const cenX = this.centroidX;
    const cenY = this.centroidY;

    this.buildGrid(invCell);

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

    let cSumX = 0;
    let cSumY = 0;
    let cSumZ = 0;
    let cNum = 0;

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
      const x0 = cx > 0 ? cx - 1 : 0;
      const x1 = cx < D - 1 ? cx + 1 : D - 1;
      const y0 = cy > 0 ? cy - 1 : 0;
      const y1 = cy < D - 1 ? cy + 1 : D - 1;
      const z0 = cz > 0 ? cz - 1 : 0;
      const z1 = cz < D - 1 ? cz + 1 : D - 1;

      outer: for (let gz = z0; gz <= z1; gz++) {
        for (let gy = y0; gy <= y1; gy++) {
          const rowBase = gy * D + gz * D2;
          for (let gx = x0; gx <= x1; gx++) {
            const cell = gx + rowBase;
            const start = this.gridStart[cell];
            const end = start + this.gridCount[cell];
            for (let e = start; e < end; e++) {
              const j = this.gridEntries[e];
              if (j === i) continue;
              const j3 = j * 3;
              const dx = pos[j3] - x;
              const dy = pos[j3 + 1] - y;
              const dz = pos[j3 + 2] - z;
              const d2 = dx * dx + dy * dy + dz * dz;
              if (d2 > perception2) continue;

              const d = Math.sqrt(d2) || 1e-4;
              if (d < sepR) {
                const w = 1 - d / sepR;
                const inv = (w * w) / d;
                sepX -= dx * inv;
                sepY -= dy * inv;
                sepZ -= dz * inv;
              }
              cohX += pos[j3];
              cohY += pos[j3 + 1];
              cohZ += pos[j3 + 2];
              alnX += vel[j3];
              alnY += vel[j3 + 1];
              alnZ += vel[j3 + 2];
              found++;

              if (energy[j] > TUNING.contagiousAbove) {
                const w = 1 - d / perception;
                contagion += w;
                if (w > bestW) {
                  bestW = w;
                  bestJ = j;
                }
              }
              if (fear[j] > maxFear) maxFear = fear[j];

              if (found >= MAX_NEIGHBORS) break outer;
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
        const p = 1 - Math.exp(-contagion * P.transmission * dt);
        // Inherit the igniter's epicenter — the whole cascade stays one bubble.
        if (Math.random() < p) {
          this.igniteIndex(
            i,
            false,
            this.pulse[bestJ * 3],
            this.pulse[bestJ * 3 + 1],
            this.pulse[bestJ * 3 + 2],
            this.pulseSign[bestJ],
          );
        }
      }

      // ---- Fear: wears off over time, spreads from panicked neighbors ----
      let fearI = fear[i] * fearDecayF;
      if (maxFear * TUNING.fearSpreadCeiling > fearI) {
        fearI += (maxFear * TUNING.fearSpreadCeiling - fearI) * fearSpreadDt;
      }
      fear[i] = fearI < 0.005 ? 0 : fearI;

      // Predator proximity, depth-squashed, measured against the lead point
      // (needed before steering: fear means scatter while the threat is
      // close, regroup once it's gone).
      let pdx = 0, pdy = 0, pdz = 0, pd = 1e9;
      if (pointerOn) {
        pdx = x - tpx;
        pdy = y - tpy;
        pdz = z - tpz;
        const pdzS = pdz * 0.45;
        pd = Math.sqrt(pdx * pdx + pdy * pdy + pdzS * pdzS) || 1e-4;
      }
      const threatNear = pd < mouseR * 1.4;

      // After a strike, far-flung survivors commit to returning: RETURN mode
      // persists until they actually reach the shoal, outliving the fear
      // itself (takes effect next frame).
      if (m === FLOCK && fearI > 0.12 && !threatNear) {
        const hx = x - cenX;
        const hy = y - cenY;
        const hz = z - this.centroidZ;
        if (hx * hx + hy * hy + hz * hz > breatheR * breatheR * 0.72) {
          this.mode[i] = RETURN;
          this.modeTimer[i] = 30;
        }
      }

      // ---- Steering forces ----
      let ax = 0, ay = 0, az = 0;
      const maxSpeed =
        maxSpeedBase *
        spdF *
        (1 + energy[i] * TUNING.energySpeedBoost + fearI * TUNING.fearSpeedBoost);

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
          // During an inburst, lit particles allow the crowd to condense —
          // when the flash fades, separation snaps back and the bubble pops.
          let sep = P.separation;
          if (this.pulseSign[i] < 0) {
            sep *= 1 - Math.min(1, energy[i]) * TUNING.inburstSepRelax;
          }
          ax += sx * sep;
          ay += sy * sep;
          az += sz * sep;
        }

        if (m === FLOCK) {
          // Cohesion — steer toward local center of mass, with "arrival"
          // damping so particles already inside the cluster relax instead
          // of endlessly compressing it.
          let tx = cohX / found - x;
          let ty = cohY / found - y;
          let tz = cohZ / found - z;
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
            // Threat close: panic scatters the shoal. Threat gone: fear
            // flips into an urgent drive to get back to the group.
            let coh = P.cohesion * dirCoh * social;
            if (threatNear && fearI > 0.01) {
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

        if (m === FLOCK || m === RETURN) {
          // Alignment — match the neighbors' heading at their actual pace
          // (normalizing to maxSpeed would make an idle swarm race around).
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
      }

      // ---- The school's collective will: cruise along the shared heading ----
      if (m === FLOCK && driveW > 0.01) {
        const want = maxSpeedBase * spdF * dirSpeed;
        let wx = headX * want - vx;
        let wy = headY * want - vy;
        let wz = headZ * want - vz;
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
        const seed = this.seeds[i];
        ax += Math.sin(t * 1.9 + seed * 61.0) * P.jitter * 1.2;
        ay += Math.sin(t * 1.6 + seed * 47.0) * P.jitter * 1.2;
        az += Math.sin(t * 2.2 + seed * 29.0) * P.jitter * 0.8;
      }

      // ---- Mode behaviors ----
      const rCenter = Math.sqrt(x * x + y * y + z * z);

      if (m === FLOCK) {
        // Soft anchor plus group recentering, both chasing the roam target.
        const hx = x - targetX;
        const hy = y - targetY;
        const hz = z - targetZ;
        ax += -hx * 0.02 + reCenterX;
        ay += -hy * 0.02 + reCenterY;
        az += -hz * 0.02 + reCenterZ;
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
        let hx = this.centroidX - x;
        let hy = this.centroidY - y;
        let hz = this.centroidZ - z;
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
        if (pd < mouseR) {
          const u = pd / mouseR;
          const fall = 1 - u * u * (3 - 2 * u); // 1 at cursor, 0 at edge

          let ex = pdx / pd;
          let ey = pdy / pd;
          let ez = pdz / pd;
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
            }
          }
        }
        // Frightened particles keep swimming away even outside the radius —
        // but only while the threat is still close; regrouping takes over
        // the moment it is not.
        if (fearI > 0.02 && threatNear) {
          const flee = P.fleeForce * fearI;
          ax += (pdx / pd) * flee;
          ay += (pdy / pd) * flee;
          az += (pdz / pd) * flee * 0.35;
        }
      }

      // Regroup urgency: still afraid but the threat is out of range —
      // swim straight back to the shoal (works with the pointer gone too).
      // Only particles still *outside* the shoal body are pulled home —
      // once inside, the urge releases so the center doesn't clump.
      if (fearI > 0.02 && !threatNear) {
        const hx = cenX - x;
        const hy = cenY - y;
        const hz = this.centroidZ - z;
        const hd = Math.sqrt(hx * hx + hy * hy + hz * hz);
        if (hd > P.containRadius * 0.75) {
          const g = (P.regroup * 3.2 * Math.sqrt(fearI)) / hd;
          ax += hx * g;
          ay += hy * g;
          az += hz * g;
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
          // Thrust along current motion, blended with the school heading.
          const sp = Math.sqrt(vx * vx + vy * vy + vz * vz);
          const blend = m === FLOCK ? dirSpeed * 1.2 : 0.2;
          let dxT = (sp > 0.2 ? vx / sp : headX) + headX * blend;
          let dyT = (sp > 0.2 ? vy / sp : headY) + headY * blend;
          let dzT = (sp > 0.2 ? vz / sp : headZ) + headZ * blend;
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
      vx = (vx + ax * dt) * dragF;
      vy = (vy + ay * dt) * dragF;
      vz = (vz + az * dt) * dragF;

      const sp2 = vx * vx + vy * vy + vz * vz;
      const cap = maxSpeed * (m === WANDER ? 0.85 : 1) + 0.4;
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

      if (this.mode[i] === FLOCK) {
        cSumX += pos[i3];
        cSumY += pos[i3 + 1];
        cSumZ += pos[i3 + 2];
        cNum++;
      }
    }

    // Centroid used next frame by returning wanderers.
    if (cNum > 0) {
      this.centroidX = cSumX / cNum;
      this.centroidY = cSumY / cNum;
      this.centroidZ = cSumZ / cNum;
    } else {
      this.centroidX = this.centroidY = this.centroidZ = 0;
    }

    // Push cheap uniforms every frame so panel tweaks apply instantly.
    const u = this.material.uniforms;
    u.uTime.value = t;
    u.uSize.value = P.particleSize;
    u.uOpacity.value = P.opacity;
    u.uFlashBright.value = P.flashBrightness;
    u.uTwinkle.value = P.twinkle;
    u.uStretch.value = P.stretch;
    u.uMotionGlow.value = P.motionGlow;

    this.geometry.attributes.position.needsUpdate = true;
    this.geometry.attributes.aEnergy.needsUpdate = true;
    this.geometry.attributes.aGlow.needsUpdate = true;
    this.geometry.attributes.aVel.needsUpdate = true;
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
