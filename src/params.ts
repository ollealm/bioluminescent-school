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
   * cascades-never-die failure) but also slows the fresh wavefront by the
   * mean heat factor; this scale restores the click-cascade growth curve to
   * the measured CPU reference under fair sampling. It's the calibration
   * dial — retune it first if the sampling scheme ever changes.
   */
  transmissionScale: 1.35,
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
  refractoryScale: 2.0,
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
   * Fear-spread compensation for fair neighbor sampling (same §5 family and
   * mechanism as transmissionScale): panic spreads toward 0.92× the
   * *sampled* max neighbor fear, and 16-of-~800 fair samples lag the true
   * front of a moving panic wave — without compensation a fast sweep panics
   * ~4× less of the school than the reference (whose cell-order neighbors
   * hugged the front) and the strike reads as a foreground-only flinch.
   * Scaling the approach rate restores the reference's spread reach; the
   * 0.92-of-max ceiling semantics are untouched.
   *
   * Do NOT trade this for locality-biased sampling: an own-cell quota was
   * tried and any position-quantized bias is static in space — it accreted
   * the school into visible sheets along the grid-cell planes.
   *
   * Back to 1.0 since stratified sampling (sampleCandidates): the sampled
   * max-fear now hugs the front like v3's full scan, and at 2.0 a parked
   * cursor panicked 89% of the school (reference ≈ 0.22, bar ≳ 0.2) and the
   * spread-feared crowd swamped the fountain signature (dot −0.08).
   * Measured at 1.0: panic 0.26–0.39, fountain −0.17…−0.41.
   */
  fearSpreadScale: 1.0,
  /**
   * Collision pushout — the anti-clump force, driven by the fine collision
   * grid below. Fair sampling starves close-range separation at 50k (an
   * overlapped pair-mate is one of ~800 candidates for 16 samples), so fish
   * paired up into blobs: 36% of the school had a nearest neighbor < 0.12 u
   * vs ~7% for random placement. Every fish inside pairPushRange×sepR
   * (clamped to one fine cell) pushes away continuously at up to this many
   * u/s² — the v3 separation-clamp magnitude. Fear damps it (threatened
   * schools pack tighter; collision noise would decohere the fountain), the
   * threat's neighborhood disables it (wall shoves would pump fish into the
   * cleared column), and it bypasses the shear-glow sense (dodging a
   * neighbor is not agitation).
   */
  pairPush: 30,
  pairPushRange: 0.3,
  /**
   * The collision grid: a second, fine counting-sort grid (cells of
   * fineCellSize world units, fineGridDim per axis) built each frame beside
   * the perception grid, used ONLY for close-range personal space. Within
   * pairPushRange×sepR (clamped to one fine cell), every real overlap is
   * detected *every frame, symmetrically* — the continuous response v3 got
   * from scanning all ~40 neighbors at 2.6k, which no probabilistic
   * sampling can reproduce at 50k: sampled hard shoves break pairs but
   * alignment re-syncs them and overlap re-accumulates (measured 4.7% at
   * 15 s → 29% at 60 s; the fine grid holds 0% flat). Fine cells hold
   * ~0.5–1.5 fish, so the pass costs ~a dozen position reads per particle
   * and three small dispatches. pairPush is now a continuous force in the
   * v3 separation-clamp family (~21 u/s²), not a rarity-compensated kick.
   */
  fineGridDim: 64,
  fineCellSize: 0.45,
  /**
   * The hunter wedge: at speed the danger zone would elongate along the
   * heading (wedgeAhead×) and taper at the flanks (wedgeSide×), relaxing to
   * the reference's sphere at rest. Tried and REVERTED to 0 by the owner:
   * fish reacting ahead of the visible cursor read as *randomness*, not
   * hunting — casual mouse movement triggered reactions with no legible
   * cause. Cause-and-effect legibility beats predator realism; the strike
   * (below) carries the hunter fantasy instead, because it's tied to a
   * deliberate gesture. Machinery kept in the kernel for experimentation;
   * both at 0 the zone is exactly v3's sphere.
   */
  wedgeAhead: 0,
  wedgeSide: 0,
  wedgeSpeedRef: 25,
  /**
   * The strike (owner's ask, round two): a flick — pointer at rest bursting
   * past strikeSpeed (u/s) within strikeFlickWindow (s) — detonates a
   * directional V-blast at the lunge tip: fish inside a forward wedge part
   * hard away from the strike line, take contact-grade fear, and a wake of
   * outward ignitions lights the path. Cooldown keeps it a *move* (stalk →
   * lunge → strike), not a mode; sustained fast circling never triggers it.
   * Magnitudes sit in the click-burst family (blast ≈ 9·fall+2).
   */
  strikeSpeed: 32,
  strikeFlickWindow: 0.35,
  strikeCooldown: 1.2,
  /** Strike wedge: reach ×mouseRadius along the heading; V half-width opens
   * from apexWidth×mouseRadius at the cursor to 1×mouseRadius at the tip. */
  strikeReach: 2.2,
  strikeApexWidth: 0.35,
  /**
   * Neighbor candidates examined per fish per frame, stratified across the
   * 27 perception cells in proportion to each cell's population; accepted
   * neighbors still cap at TUNING.maxNeighbors. The first GPU sweep took the
   * first 16 in-range fish in a hash-rotated cell order — but the first cell
   * with in-range fish usually filled all 16 slots, so the sample averaged
   * over *cells*, not fish, and a fish's cohesion target depended on where
   * it sat inside its cell: a static force field with the grid period
   * (measured ±10% density modulation, visible as sheets along the grid
   * planes once the core's mid-tones show). Population-proportional quotas
   * make every fish in the block equally likely to be examined.
   *
   * Sized to the reference's neighbor count: cells have edge = perception,
   * so only 4π/3 ÷ 27 ≈ 15.5% of a uniformly dense 27-cell block lies inside
   * the perception sphere, and 96 candidates accept ~15 in the core (fewer
   * at the fringe, all of them where the block holds < 96 fish). When the
   * cap binds it trims only a few accepts off the tail of the cell order —
   * a second-order echo of the old bias. Cost is about what the old sweep
   * spent reaching 16 accepts. Lower it and contagion (a sum over accepted
   * neighbors) and the sampled max-fear weaken proportionally.
   */
  sampleCandidates: 96,
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
  /**
   * GPU-only. How often and how hard the school contracts in on itself, then
   * releases (the director's CONTRACT mood). 0 = the reference FSM exactly;
   * 1 sends ~35% of cruise/mill exits into a contraction.
   */
  collapse: number;
  /**
   * GPU-only. Stretches the school along its heading: anisotropic cohesion
   * (weaker along, stronger across) so it travels as a shoal, not a ball.
   * 0 = the reference's isotropic cohesion.
   */
  shape: number;
  /**
   * GPU-only. Selfish-herd drive: every fish wants deeper into the school,
   * edge fish hardest, so the body pulses against its own crowd pressure.
   * The director swells it during a contraction. 0 = the reference.
   */
  herd: number;
  /**
   * GPU-only. Waves of agitation at rest: on a random clock a pocket of fish
   * takes fear and darts outward; fear contagion carries the shiver through
   * the school and the regroup/alarm machinery re-contracts it. No light.
   * Scales the event rate. 0 = no events (the reference).
   */
  shiver: number;
  /**
   * GPU-only. Flattens the school toward the view plane: a damped z spring
   * toward the school's mean depth plus stronger depth cohesion. 25k fish
   * stacked ~20 deep along the camera axis average density waves away; a
   * school a few fish deep shows them. 0 = the reference.
   */
  sheet: number;

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
  /**
   * GPU-only. Crispness of the danger zone's edge: the escape and fear
   * falloffs ramp over [bite, 1] of the radius instead of [0, 1]. 0 = the
   * reference's soft gravity-like push; high = nothing until a fish is
   * close, then a full-strength bolt (a predator, not a well).
   */
  bite: number;
  /**
   * GPU-only. How far fear carries a fish out of the school before it
   * regroups: fear above 1 - flight keeps it fleeing past the threat band,
   * so with panic spread the hit fish take neighbors with them. 0 = v3.
   */
  flight: number;
  /**
   * GPU-only. How violently frightened fish rush back to the school's
   * center once the threat is gone: they drive for the centroid (not just
   * the edge) and overshoot through it. 0 = v3.
   */
  rally: number;
  /**
   * GPU-only. The school evades a parked predator as a body: its center
   * shifts away and fish slide around the threat instead of holding a
   * static shell. 0 = v3.
   */
  evade: number;
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
  /**
   * Crowd dim (GPU only, 0..1): fish in cores denser than the nominal
   * school fade by (nominal / local)^(0.8 * crowdDim), so additive pile-ups
   * read as a denser school rather than one blown-out glowing spot. 0 = off
   * (bit-identical to before).
   */
  crowdDim: number;
  /**
   * Accent share (GPU only, appearance only): fraction of fish that take the
   * palette's accent color. 0 removes them. Replaces the palette's own
   * accentChance on the GPU backend; the CPU fallback keeps v3's.
   */
  accentShare: number;
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
  collapse: 0,
  shape: 0,
  herd: 0,
  shiver: 0,
  sheet: 0,

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
  bite: 0,
  flight: 0,
  rally: 0,
  evade: 0,
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
  crowdDim: 0,
  accentShare: 0.05,
};
