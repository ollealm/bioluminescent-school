import type { PaletteName, StyleName, SwarmParams } from "./params";
import type { PresetCategory } from "./presets";

/**
 * The backend-agnostic contract both simulations implement.
 *
 * The GPU rewrite (src/gpu) and the vendored CPU reference (src/cpu) are
 * interchangeable behind this interface: the bootstrap picks one by feature
 * detection, and the dev panel / harness / pointer plumbing never know which
 * they're driving. It mirrors the public surface of the CPU ParticleSystem
 * (which predates the interface), so the CPU class satisfies it as-is.
 */
export interface SwarmSystem {
  /** Live-tweakable parameters — the dev panel binds directly to this. */
  readonly params: SwarmParams;
  presetNames: Record<PresetCategory, string>;
  paletteName: PaletteName;
  styleName: StyleName;

  getCount(): number;
  /** Current director mood, for harnesses and debugging. */
  readonly moodName: string;
  /**
   * Optional dev test control: every mood the director knows, the one it
   * is pinned to (null = the normal FSM), and the pin itself.
   */
  readonly moodNames?: readonly string[];
  readonly forcedMood?: string | null;
  forceMood?(name: string | null): void;

  applyPreset(category: PresetCategory, name: string): void;
  setPalette(name: PaletteName): void;
  setStyle(name: StyleName): void;
  /**
   * Accent color override ("#rrggbb"), null = the palette's own. Optional
   * (the panel feature-detects it); both backends implement it.
   */
  readonly accentColor?: string | null;
  /** Override the palette's accent color; null restores the palette's own. */
  setAccent?(hex: string | null): void;
  /**
   * Base color override ("#rrggbb"), null = the palette's own gradient.
   * Optional for the same reason as the accent.
   */
  readonly baseColor?: string | null;
  /** Derive the gradient from one color; null restores the palette's stops. */
  setBase?(hex: string | null): void;
  getSnapshot(): Snapshot;
  applySnapshot(s: Partial<Snapshot>): void;
  /** Recreate all particle state for a new count. Params are untouched. */
  rebuild(count: number): void;
  /**
   * Optional: change the count without resetting the school (both
   * backends thin and revive in place). Callers that can't tolerate a full
   * respawn (count slider, frame-time governor) prefer this when present;
   * the harness always uses rebuild for a clean field.
   */
  resize?(count: number): void;

  /** Pointer in NDC ([-1,1]²); projected onto the z=0 swarm plane inside. */
  updateMousePosition(ndc: { x: number; y: number }): void;
  clearPointer(): void;
  updateRendererSize(halfX: number, halfY: number): void;
  setPixelRatio(dpr: number): void;

  /** Detonate a chain reaction at `at`, else the pointer (or swarm center). */
  burst(at?: { x: number; y: number; z: number }): void;
  /** Blow the whole swarm apart; it will drift back and reunite. */
  scatter(): void;
  /** A false alarm — a pocket of panic sweeps the school. */
  startle(): void;

  /** Advance the simulation by dt seconds (already frame-clamped upstream). */
  update(dt: number): void;

  dispose(): void;
}

/** Bump when the snapshot shape or param semantics change incompatibly. */
export const SNAPSHOT_VERSION = 3;

/**
 * Structurally identical to the CPU reference's Snapshot — snapshots flow
 * freely between the two versions (that's the point: snapshots/site-default
 * .json is the canonical tuned state both must reproduce).
 */
export interface Snapshot {
  version: number;
  presets: Record<PresetCategory, string>;
  palette: PaletteName;
  style: StyleName;
  count: number;
  params: SwarmParams;
  /**
   * Accent color override ("#rrggbb"). getSnapshot() omits it while the
   * palette's own accent shows; in applySnapshot() absent leaves the
   * current override and null clears it. Additive, so the v3 reference and
   * older snapshots stay valid at version 3.
   */
  accent?: string | null;
  /**
   * Base color override ("#rrggbb"): the school's gradient and flash hue
   * are derived from it. Same omit/absent/null rules as accent.
   */
  base?: string | null;
  /**
   * Per-backend overrides for the CPU fallback, merged over `params` when a
   * feel is selected on it (src/feel.ts). A feel tuned at 25k on the GPU
   * reads too fast and too dim at a few thousand fish; this keeps one feel
   * for both. `count` is the CPU fish count unless ?count= overrides it.
   */
  cpu?: Partial<SwarmParams> & { count?: number };
}
