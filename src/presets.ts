import { DEFAULT_PARAMS, type SwarmParams } from "./params";

/**
 * Presets come in two orthogonal categories so movement and reactions can be
 * mixed freely:
 *
 *  - motion    — how the swarm moves (director, flocking, currents, wanderers)
 *  - reaction  — how it responds (predator, chain reactions, kicks)
 *
 * Applying a preset resets only its own category's params, leaving the other
 * untouched. Looks are not preset-managed: style and palette are picked
 * directly, and the appearance sliders do the rest.
 */
export type PresetCategory = "motion" | "reaction";

export interface Preset {
  label: string;
  description: string;
  params: Partial<SwarmParams>;
}

export const MOTION_KEYS: (keyof SwarmParams)[] = [
  "timeScale",
  "cohesion",
  "alignment",
  "separation",
  "perception",
  "maxSpeed",
  "drag",
  "schoolDrive",
  "tempo",
  "startle",
  "gait",
  "flowStrength",
  "flowScale",
  "swirl",
  "jitter",
  "breathe",
  "containRadius",
  "roam",
  "mill",
  "wanderRate",
  "groupSize",
  "wanderPull",
  "returnPull",
];

export const REACTION_KEYS: (keyof SwarmParams)[] = [
  "mouseRadius",
  "repelForce",
  "splash",
  "fleeForce",
  "fearDecay",
  "fearSpread",
  "regroup",
  "transmission",
  "spontaneous",
  "flashDecay",
  "refractory",
  "kick",
  "inburst",
];

export const CATEGORY_KEYS: Record<PresetCategory, (keyof SwarmParams)[]> = {
  motion: MOTION_KEYS,
  reaction: REACTION_KEYS,
};

// ---------------------------------------------------------------------------
// Motion — how the swarm moves
// ---------------------------------------------------------------------------

export const MOTION_PRESETS: Record<string, Preset> = {
  lagoon: {
    label: "Lagoon",
    description: "Balanced default — cohesive drift with a gentle cruise.",
    params: {},
  },
  shoal: {
    label: "Shoal",
    description: "A traveling shoal — cruises hard, turns as one.",
    params: {
      cohesion: 1.3,
      alignment: 2.0,
      schoolDrive: 1.9,
      tempo: 1.2,
      startle: 0.6,
      gait: 1.0,
      separation: 1.4,
      perception: 2.5,
      maxSpeed: 3.8,
      drag: 0.5,
      flowStrength: 0.7,
      swirl: 0.3,
      jitter: 0.4,
      breathe: 0.05,
      containRadius: 8.5,
      roam: 1.2,
      mill: 0.15,
      wanderRate: 0.3,
      groupSize: 10,
      returnPull: 2.6,
    },
  },
  mill: {
    label: "The Mill",
    description: "A rotating vortex — the shoal turns in on itself.",
    params: {
      cohesion: 1.1,
      alignment: 1.7,
      schoolDrive: 0.5,
      tempo: 0.6,
      startle: 0.3,
      gait: 0.7,
      separation: 1.5,
      perception: 2.5,
      maxSpeed: 3.4,
      drag: 0.5,
      flowStrength: 0.4,
      swirl: 0.2,
      jitter: 0.3,
      breathe: 0.05,
      containRadius: 8,
      roam: 0.15,
      mill: 1.5,
      wanderRate: 0.2,
      returnPull: 2.2,
    },
  },
  baitball: {
    label: "Baitball",
    description: "Tight, fast, nervous — a ball that turns as one.",
    params: {
      cohesion: 2.2,
      alignment: 2.6,
      schoolDrive: 2.2,
      tempo: 1.7,
      startle: 1.5,
      gait: 1.2,
      separation: 1.6,
      perception: 2.6,
      maxSpeed: 5.2,
      drag: 0.42,
      flowStrength: 0.4,
      swirl: 0.6,
      jitter: 0.25,
      breathe: 0.04,
      containRadius: 6.5,
      roam: 0.6,
      mill: 0.35,
      wanderRate: 0.12,
      returnPull: 3.0,
    },
  },
  riptide: {
    label: "Riptide",
    description: "Strong currents drag wisps into streaks and eddies.",
    params: {
      cohesion: 0.8,
      alignment: 1.3,
      schoolDrive: 0.35,
      tempo: 0.8,
      startle: 0.3,
      gait: 0.5,
      separation: 1.4,
      perception: 2.4,
      maxSpeed: 2.2,
      drag: 0.5,
      flowStrength: 2.2,
      flowScale: 0.13,
      swirl: 0.8,
      jitter: 0.7,
      breathe: 0.08,
      containRadius: 10,
      roam: 0.5,
      mill: 0,
      wanderRate: 0.6,
      groupSize: 16,
      returnPull: 1.4,
    },
  },
  meadow: {
    label: "Meadow",
    description: "Loosely social hoverers with lots of lone drifters.",
    params: {
      cohesion: 0.5,
      alignment: 0.35,
      schoolDrive: 0.15,
      tempo: 0.5,
      startle: 0.12,
      gait: 1.1,
      separation: 1.8,
      perception: 2.6,
      maxSpeed: 1.4,
      drag: 0.8,
      flowStrength: 0.7,
      flowScale: 0.06,
      swirl: 0.2,
      jitter: 0.9,
      breathe: 0.03,
      roam: 0.25,
      mill: 0,
      wanderRate: 0.8,
      groupSize: 5,
      returnPull: 1.0,
    },
  },
  abyssal: {
    label: "Abyssal",
    description: "Sparse, slow, deep — barely-moving drift.",
    params: {
      timeScale: 0.85,
      cohesion: 0.7,
      schoolDrive: 0.3,
      tempo: 0.35,
      startle: 0.05,
      gait: 0.3,
      alignment: 0.6,
      separation: 2.0,
      perception: 3.0,
      maxSpeed: 1.1,
      drag: 0.9,
      flowStrength: 0.8,
      flowScale: 0.05,
      swirl: 0.3,
      jitter: 0.3,
      breathe: 0.09,
      containRadius: 10.5,
      roam: 0.15,
      mill: 0.1,
      wanderRate: 0.25,
    },
  },
};

// ---------------------------------------------------------------------------
// Reactions — how it responds to you and to itself
// ---------------------------------------------------------------------------

export const REACTION_PRESETS: Record<string, Preset> = {
  still: {
    label: "Still",
    description: "Barely reactive — mild unease, rare lone glimmers.",
    params: {
      mouseRadius: 5,
      repelForce: 16,
      splash: 0.5,
      fleeForce: 10,
      fearDecay: 1.6,
      fearSpread: 0.4,
      regroup: 0.8,
      transmission: 0.8,
      spontaneous: 0.12,
      flashDecay: 2.0,
      refractory: 3.0,
      kick: 1.0,
      inburst: 0.2,
    },
  },
  ripple: {
    label: "Ripple",
    description: "Wary swarm, gentle bubbling pulses that fade.",
    params: {
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
      kick: 2.2,
      inburst: 0.35,
    },
  },
  surge: {
    label: "Surge",
    description: "Real panic — fear spreads, big rolling bubbles.",
    params: {
      mouseRadius: 7.5,
      repelForce: 50,
      splash: 1.6,
      fleeForce: 40,
      fearDecay: 0.9,
      fearSpread: 2.0,
      regroup: 2.4,
      transmission: 5.0,
      spontaneous: 0.8,
      flashDecay: 3.0,
      refractory: 1.6,
      kick: 3.0,
      inburst: 0.4,
    },
  },
  frenzy: {
    label: "Frenzy",
    description: "Terror — the swarm shatters around you, hard pulses.",
    params: {
      mouseRadius: 8,
      repelForce: 65,
      splash: 2.2,
      fleeForce: 55,
      fearDecay: 0.7,
      fearSpread: 3.0,
      regroup: 3.2,
      transmission: 7.0,
      spontaneous: 1.6,
      flashDecay: 3.5,
      refractory: 1.2,
      kick: 3.6,
      inburst: 0.5,
    },
  },
};

export const PRESET_REGISTRY: Record<PresetCategory, Record<string, Preset>> = {
  motion: MOTION_PRESETS,
  reaction: REACTION_PRESETS,
};

export const DEFAULT_PRESETS: Record<PresetCategory, string> = {
  motion: "lagoon",
  reaction: "ripple",
};

/** Sanity: the default pair must equal DEFAULT_PARAMS so "defaults" mean one thing. */
for (const cat of ["motion", "reaction"] as const) {
  const preset = PRESET_REGISTRY[cat][DEFAULT_PRESETS[cat]];
  for (const [k, v] of Object.entries(preset.params)) {
    if (DEFAULT_PARAMS[k as keyof SwarmParams] !== v) {
      // eslint-disable-next-line no-console
      console.warn(`Preset default mismatch: ${cat}/${k} = ${v}`);
    }
  }
}
