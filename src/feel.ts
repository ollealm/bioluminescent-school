import type { DevPanel } from "./DevPanel";
import { DEFAULT_PARAMS, type SwarmParams } from "./params";
import { CATEGORY_KEYS, PRESET_REGISTRY, type PresetCategory } from "./presets";
import type { Snapshot, SwarmSystem } from "./system";

/**
 * Feel A/B: compare whole tuned states live, and share them by URL.
 *
 * Every `snapshots/*.json` is a "feel" (a full Snapshot, the same shape the
 * panel's Copy JSON produces), keyed by basename. Number keys 1..9 switch
 * between them in sorted order with site-default always on 1, so the
 * canonical v3 state is one keypress away from whatever is being tuned.
 * The active feel is mirrored into `?feel=<name>`, so a link reproduces it.
 *
 * Adding a feel is a file drop: "Save feel..." in the panel downloads
 * `<name>.json`, move it into snapshots/ and Vite picks it up.
 */

const modules = import.meta.glob<Snapshot>("../snapshots/*.json", {
  eager: true,
  import: "default",
});

export const FEELS: Record<string, Snapshot> = {};
for (const [path, snap] of Object.entries(modules)) {
  const name = path.split("/").pop()!.replace(/\.json$/, "");
  FEELS[name] = snap;
}

/** Hotkey order: site-default first (the canonical reference), rest by name. */
/** Own keys only: `?feel=toString` must not resolve to a prototype member.
 * (Object.hasOwn is ES2022; the tsconfig lib is ES2020.) */
function isFeel(name: string): boolean {
  return Object.prototype.hasOwnProperty.call(FEELS, name);
}

export const FEEL_NAMES: string[] = Object.keys(FEELS).sort((a, b) =>
  a === "site-default" ? -1 : b === "site-default" ? 1 : a.localeCompare(b),
);

/** The `?feel=` selection, only if it names a known snapshot. */
export function urlFeel(): string | null {
  const name = new URLSearchParams(location.search).get("feel");
  return name && isFeel(name) ? name : null;
}

/** The feel a plain visit boots into: the one shipped on olle.work. */
export const DEFAULT_FEEL = "olle-work";

export interface FeelSwitcherOptions {
  /** How a feel's snapshot (count stripped) reaches the system. */
  apply?: (snap: Partial<Snapshot>) => void;
  /** Apply DEFAULT_FEEL when the URL names none (the production boot). */
  useDefault?: boolean;
}

export interface FeelSwitcher {
  /** Apply a feel by name; false if unknown. */
  select(name: string): boolean;
  readonly names: readonly string[];
  /** The feel applied at install (from ?feel= or the default), if any. */
  readonly applied: string | null;
  /** Unbind hotkeys and detach from the panel (call before a re-mount). */
  dispose(): void;
}

export function installFeelSwitcher(
  system: SwarmSystem,
  panel: DevPanel,
  opts: FeelSwitcherOptions = {},
): FeelSwitcher {
  const apply = opts.apply ?? ((snap) => system.applySnapshot(snap));

  const select = (name: string, o: { updateUrl?: boolean } = {}): boolean => {
    if (!isFeel(name)) return false;
    const snap = FEELS[name];
    // A feel is a temperament, not a density: the snapshot's count is
    // ignored so A/B comparisons happen at equal density (the boot default,
    // the count slider, or ?count= decide how many fish there are).
    const { count: _count, ...rest } = snap;
    void _count;
    // Replace, not merge: keys the snapshot predates (crowdDim in
    // site-default) must return to defaults, not keep the last feel's.
    apply({ ...rest, params: { ...DEFAULT_PARAMS, ...rest.params } });
    // A feel usually carries the preset names it was derived from, but its
    // sliders have moved on. Highlighting "Lagoon" over olle-work's values
    // misleads (and clicking it wipes the feel's tuning), so a category
    // whose values no longer match its preset shows no highlight.
    for (const cat of Object.keys(CATEGORY_KEYS) as PresetCategory[]) {
      const preset = PRESET_REGISTRY[cat][system.presetNames[cat]];
      if (!preset) continue;
      const live = system.params as unknown as Record<keyof SwarmParams, number>;
      const base = { ...DEFAULT_PARAMS, ...preset.params } as Record<keyof SwarmParams, number>;
      if (CATEGORY_KEYS[cat].some((k) => live[k] !== base[k])) {
        system.presetNames[cat] = "";
      }
    }

    // Already in the URL (boot apply, same key twice): no history churn.
    const query = new URLSearchParams(location.search);
    if (o.updateUrl !== false && query.get("feel") !== name) {
      query.set("feel", name);
      history.replaceState(history.state, "", `${location.pathname}?${query}${location.hash}`);
    }

    panel.refresh();
    panel.setFeelName(name);
    return true;
  };

  panel.setFeels(FEEL_NAMES);
  panel.onFeelSelect = (name) => void select(name);
  panel.onReset = () => void select(DEFAULT_FEEL, { updateUrl: false });

  let applied: string | null = null;
  const requested = new URLSearchParams(location.search).get("feel");
  if (requested && select(requested)) {
    applied = requested;
  } else {
    // A typo'd link still gets the shipped look rather than bare defaults.
    if (requested) console.warn(`[feel] unknown ?feel=${requested}; have: ${FEEL_NAMES.join(", ")}`);
    if (opts.useDefault && isFeel(DEFAULT_FEEL)) {
      // The shipped look: a plain visit gets the final feel, silently (no
      // URL rewrite). A dev session that saved its own tweaks passes
      // useDefault false and keeps them.
      if (select(DEFAULT_FEEL, { updateUrl: false })) applied = DEFAULT_FEEL;
    }
  }

  const onKey = (e: KeyboardEvent) => {
    // A held key would re-apply the snapshot every autorepeat tick.
    if (e.repeat) return;
    const target = e.target as HTMLElement;
    if (/^(input|select|textarea)$/i.test(target?.tagName ?? "") || target?.isContentEditable) return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (!/^[1-9]$/.test(e.key)) return;
    const name = FEEL_NAMES[Number(e.key) - 1];
    if (name) select(name);
  };
  window.addEventListener("keydown", onKey);

  return {
    select,
    names: FEEL_NAMES,
    applied,
    dispose() {
      window.removeEventListener("keydown", onKey);
      panel.onFeelSelect = null;
      panel.onReset = undefined;
    },
  };
}
