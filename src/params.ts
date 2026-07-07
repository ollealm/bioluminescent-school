// Shared parameter/palette/style definitions for the swarm.

/**
 * The load-bearing constants that aren't user-facing sliders — the real spec
 * of the simulation, hoisted out of the code so a rewrite (see
 * HANDOFF-GPU.md) can diff against it instead of excavating literals.
 * Changing anything here changes behavior; each entry says what breaks.
 */
export const TUNING = {
  /** Spatial grid cells per axis (cells have edge = perception). */
  gridDim: 32,
  /** Neighbor cap per particle. Higher = smoother boids, linear cost. */
  maxNeighbors: 16,
  /** Steering clamp (u/s²) for each boids force before its weight. */
  maxForce: 14,
  /** Separation radius as a fraction of perception. */
  separationRatio: 0.66,
  /**
   * Cohesion "arrival" radius as a fraction of perception. Without arrival
   * damping the swarm collapses into a ball (see HANDOFF-GPU.md §4.1).
   */
  arriveRatio: 1.2,
  /** A flash is contagious while energy is above this. */
  contagiousAbove: 0.55,
  /** The flash "ends" (refractory starts) when energy drops below this. */
  flashEndsBelow: 0.35,
  /** Shear-glow knee (u/s²): accelerations below this stay dark. */
  glowKnee: 22,
  /** Shear-glow gain per u/s² above the knee. */
  glowGain: 0.04,
  /** Sustained radial pulse gain for outbursts (× energy × kick). */
  radialGainOut: 2.4,
  /** Inbursts need extra gain to beat crowd pressure. */
  radialGainIn: 3.4,
  /** Fear spreads toward this fraction of the scariest neighbor's fear. */
  fearSpreadCeiling: 0.92,
  /** Speed-cap boost at full flash energy. */
  energySpeedBoost: 1.1,
  /** Speed-cap boost at full fear. */
  fearSpeedBoost: 1.2,
  /** How much a lit inburst particle relaxes separation (0..1). */
  inburstSepRelax: 0.6,
  /** Click-burst seed radius as a fraction of mouseRadius. */
  burstSeedRatio: 0.5,
} as const;

/**
 * Fork-specific constants for the GPU rewrite. The behavior constants above
 * are shared verbatim with the CPU reference; everything here exists only
 * because of the 20× density jump (HANDOFF-GPU.md §5) or the GPU execution
 * model. Each entry says what breaks.
 */
export const GPU_TUNING = {
  /**
   * The particle count all aesthetic params were tuned at (the canonical
   * snapshot). Density compensation is relative to this.
   */
  referenceCount: 2600,
  /**
   * Perceived brightness ≈ count × size² × opacity under additive blending.
   * Rendered size/opacity are scaled by (referenceCount/count)^exponent so
   * the same tuned params read the same at any count. The two exponents sum
   * to ~1 in the size²·opacity product; size takes the larger share so 50k
   * reads as *finer grain*, not just dimmer motes.
   */
  sizeCompExp: 0.35,
  opacityCompExp: 0.3,
  /** Compensation is clamped so tiny counts don't blow up (≤ this factor). */
  maxComp: 2.0,
  /**
   * HDR shoulder tonemap knee: below this the transfer is identity, above it
   * brightness compresses asymptotically to 1 while preserving hue. This is
   * the structural "never white" enforcement — additive pile-ups saturate
   * toward the palette hue instead of clipping RGB to white.
   */
  tonemapKnee: 0.72,
  /**
   * Absolute-count params expressed as fractions of referenceCount
   * (HANDOFF §5): scatter's ignition seeds scale with count through this,
   * and params.groupSize is read as "fish per referenceCount".
   */
  scatterIgniteFrac: 12 / 2600,
  /**
   * Contagion normalization at 20× density (HANDOFF §5). The GPU kernel
   * weights each lit neighbor by its remaining heat — (energy − 0.55)/0.45
   * — where v3 counted every lit neighbor at full strength. That starves
   * "spark carriers" (lit fish that fly a perception radius while still
   * contagious and re-seed the burn indefinitely in dense fuel — the
   * cascades-never-die failure). At 1.0 the click-cascade growth curve
   * matches the measured CPU reference: the heat-weighting's front slowdown
   * and the own-cell quota's closer (hotter) samples cancel. Retune this
   * first if the sampling scheme changes — it's the calibration dial.
   */
  transmissionScale: 1.25,
  /**
   * Refractory re-derivation at 50k (invariant §4.3: refractory must exceed
   * the cascade's crossing time, "re-derive this at 50k"). The wave crosses
   * the denser school in ~1.5× the reference time, and the dense fringe can
   * sustain a circulating smolder (a spiral wave) whose lap time marginally
   * outlives the tuned 2.2 s refractory — cascades then never die (the §6
   * cascadeDied probe fails ~1 run in 3). Scaling effective refractory by
   * the same ~1.5× restores the reference's refractory/crossing ratio and
   * starves the spiral. Applied host-side; the user slider keeps meaning.
   */
  refractoryScale: 1.75,
  /**
   * Crowd-pressure compensation around the predator (HANDOFF §5 "kick vs
   * crowd pressure", §6 predator signatures). Two jams at 50k the reference
   * never felt: the dense wall around the cleared column shoves fresh fish
   * inward through pure separation pressure, and the panic wave can't
   * *move* bodies — fleeing fish stall against personal space instead of
   * scattering. Fish relax their own separation by this × max(column
   * falloff, fear-while-threat-near), so evacuation isn't diffusion-jammed
   * and a strike visibly shatters the crowd ("react hard, return hard").
   */
  predatorSepRelax: 0.8,
  /**
   * Fear-spread compensation for sampled neighbors (same §5 family and
   * mechanism as transmissionScale): panic spreads toward 0.92× the
   * *sampled* max neighbor fear, which lags the true front of a moving
   * panic wave. The own-cell quota supplies front-hugging samples (like
   * v3's cell-order scan), so only a small rate boost remains — tuned so a
   * fast sweep panics slightly more of the school than the reference
   * (~14%), erring lively because 20× more non-reacting bodies dilute the
   * visible reaction. The 0.92-of-max ceiling semantics are untouched.
   */
  fearSpreadScale: 1.0,
  /**
   * Of the 16-neighbor cap, this many are taken from the particle's own
   * grid cell (random in-cell offset) before the hash-rotated fair sweep
   * fills the rest. Pure fair sampling starves close-range separation at
   * 50k — knot-mates inside a clump almost never sample each other (16 of
   * ~800 candidates), so fish pair up into overlapping blobs: measured 36%
   * of the school with a nearest neighbor < 0.12 u vs ~7% for random
   * placement. The quota guarantees personal space sees the actual
   * neighbors; the remaining samples keep cohesion/alignment fair. Costs
   * nothing — same 16 reads, re-aimed. The v3 reference's cell-order scan
   * had this locality implicitly (it read its own cell first).
   */
  ownCellQuota: 8,
  /**
   * Collision pushout for sampled personal-space violators, the other half
   * of the anti-clump story. Even with the own-cell quota, a pair-mate is
   * one fish among ~130 in the cell — sampled maybe 6% of frames — while
   * cohesion pulls every frame, so overlapped pairs persist (the v3 scan
   * caught mates every frame; sampling can't). Compensate rarity with
   * strength: when a violator inside pairPushRange×sepR *is* sampled, it
   * gets a hard shove (u/s², deliberately above MAX_FORCE — a collision
   * response, not steering) whose expected time-average matches the
   * reference's continuous separation. Same §5 sampling-normalization
   * family as transmissionScale / fearSpreadScale. Respects the inburst
   * and predator separation relaxes, so condensation still pops.
   */
  pairPush: 40,
  pairPushRange: 0.3,
  /**
   * The hunter wedge (owner's ask): at speed, the predator's danger zone
   * elongates along its heading and narrows at the flanks — a lunge, not a
   * bubble — relaxing back to the reference's sphere at rest. Ahead reach
   * grows by wedgeAhead×, flank reach shrinks by wedgeSide× at full speed;
   * full speed = wedgeSpeedRef in world u/s of pointer velocity. The
   * behind-reach is untouched: prey dropping aft of the predator is the
   * fountain's "safe wake" beat and must stay safe.
   */
  wedgeAhead: 0.6,
  wedgeSide: 0.1,
  wedgeSpeedRef: 25,
  /** Compute workgroup size for the per-particle kernels. */
  workgroupSize: 64,
  /** Max host→GPU events applied per frame (burst/startle/vanguard/...). */
  maxEvents: 16,
} as const;

export type StyleName = "glow" | "orb" | "spark" | "firefly";

export const STYLES: StyleName[] = ["glow", "orb", "spark", "firefly"];

export const STYLE_IDS: Record<StyleName, number> = {
  glow: 0,
  orb: 1,
  spark: 2,
  firefly: 3,
};

export interface Palette {
  label: string;
  /** Gradient stops, dark → bright. Particles sample along this. */
  stops: string[];
  /** Rare accent color sprinkled in for depth. */
  accent: string;
  accentChance: number;
  /**
   * Color a particle leans toward when excited. Keep it a saturated version
   * of the palette's own hue — never white, so flashes read as the creature
   * brightening rather than fireworks.
   */
  flash: string;
}

export type PaletteName =
  | "emerald"
  | "reef"
  | "firefly"
  | "abyss"
  | "ghost"
  | "violet";

export const PALETTES: Record<PaletteName, Palette> = {
  emerald: {
    label: "Emerald",
    stops: ["#0a5c3f", "#12b56f", "#41f2a0", "#a5ffd2"],
    accent: "#4fc3ff",
    accentChance: 0.05,
    flash: "#57ffae",
  },
  reef: {
    label: "Cyan Reef",
    stops: ["#083b52", "#0fa3b8", "#39e8e0", "#b7fff4"],
    accent: "#58ffb0",
    accentChance: 0.06,
    flash: "#4fe3f2",
  },
  firefly: {
    label: "Firefly",
    stops: ["#4c5a10", "#96a821", "#e8ff5e", "#fbffc9"],
    accent: "#ffb347",
    accentChance: 0.05,
    flash: "#ffe066",
  },
  abyss: {
    label: "Abyss",
    stops: ["#06253d", "#0b5e66", "#19b39b", "#7dffd9"],
    accent: "#3d7bff",
    accentChance: 0.06,
    flash: "#38dfc0",
  },
  ghost: {
    label: "Ghost",
    stops: ["#22343a", "#5f8d8a", "#aee8d8", "#f2fffa"],
    accent: "#8fd0ff",
    accentChance: 0.04,
    flash: "#bdf2de",
  },
  violet: {
    label: "Violet",
    stops: ["#2a0b52", "#6b2fd6", "#b06bff", "#ead2ff"],
    accent: "#41f2a0",
    accentChance: 0.06,
    flash: "#c894ff",
  },
};

export interface SwarmParams {
  /** Global simulation speed multiplier. */
  timeScale: number;

  // --- Flocking (boids) ---
  cohesion: number;
  alignment: number;
  separation: number;
  /** Neighbor perception radius in world units. */
  perception: number;
  maxSpeed: number;
  drag: number;

  // --- School behavior (the swarm director) ---
  /** Weight of the school's collective will (cruise/turn propulsion). */
  schoolDrive: number;
  /** How often the school changes its mind (mood transition rate). */
  tempo: number;
  /** Rate of spontaneous false alarms sweeping the school. */
  startle: number;
  /** Burst-and-glide swimming intensity (thrust impulses, then coasting). */
  gait: number;

  // --- Ambient motion ---
  flowStrength: number;
  flowScale: number;
  swirl: number;
  jitter: number;
  breathe: number;
  containRadius: number;
  /** How far the swarm's roaming attractor travels — the shoal cruises. */
  roam: number;
  /** Rotating-vortex (fish mill) force around the swarm's own center. */
  mill: number;

  // --- Wanderers (drift away & reunite) ---
  wanderRate: number;
  groupSize: number;
  wanderPull: number;
  returnPull: number;

  // --- Pointer (predator) ---
  mouseRadius: number;
  repelForce: number;
  splash: number;
  /** How hard frightened particles keep swimming away from the cursor. */
  fleeForce: number;
  /** How quickly fear wears off (per second). */
  fearDecay: number;
  /** How strongly panic spreads to neighbors. */
  fearSpread: number;
  /**
   * Post-threat cohesion surge: once the predator is out of range, still-
   * frightened particles rush back to the shoal instead of dispersing.
   */
  regroup: number;

  // --- Chain reactions (excitable medium) ---
  transmission: number;
  spontaneous: number;
  flashDecay: number;
  refractory: number;
  flashBrightness: number;
  kick: number;
  /** Chance a spontaneous cascade collapses inward instead of bursting out. */
  inburst: number;

  // --- Appearance ---
  particleSize: number;
  opacity: number;
  twinkle: number;
  /** Shear-triggered luminescence: agitated particles glow brighter. */
  motionGlow: number;
  /** Velocity stretching: fast particles render as streaks, not dots. */
  stretch: number;
}

export const DEFAULT_PARAMS: SwarmParams = {
  timeScale: 1.0,

  cohesion: 1.0,
  alignment: 1.3,
  separation: 1.5,
  perception: 2.3,
  maxSpeed: 3.2,
  drag: 0.55,

  schoolDrive: 1.0,
  tempo: 1.0,
  startle: 0.5,
  gait: 0.8,

  flowStrength: 0.9,
  flowScale: 0.09,
  swirl: 0.45,
  jitter: 0.4,
  breathe: 0.06,
  containRadius: 9,
  roam: 0.45,
  mill: 0.2,

  wanderRate: 0.45,
  groupSize: 12,
  wanderPull: 1.0,
  returnPull: 1.6,

  mouseRadius: 6.5,
  repelForce: 38,
  splash: 1.0,
  fleeForce: 30,
  fearDecay: 1.1,
  fearSpread: 1.3,
  regroup: 1.6,

  transmission: 2.8,
  spontaneous: 0.45,
  flashDecay: 2.4,
  refractory: 2.2,
  flashBrightness: 0.9,
  kick: 2.2,
  inburst: 0.35,

  particleSize: 1.6,
  opacity: 0.85,
  twinkle: 0.55,
  motionGlow: 0.9,
  stretch: 1.3,
};
