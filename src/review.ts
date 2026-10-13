/**
 * Filmstrip review: one image of N frames over time plus per-frame numbers,
 * for reviewers that can't watch motion (an LLM orchestrator, a PR thread).
 *
 *   const r = await __filmstrip({ frames: 8, intervalMs: 400 });
 *   const r = await __sweep();   // the standard predator test
 *
 * It observes the live animation and never pauses the loop. Each capture
 * calls `window.__render()` and `drawImage`s the canvas in the same task:
 * a WebGPU canvas is only reliably readable in the task that rendered it.
 * The canvas composites additively over the page gradient; tiles show it
 * over black instead, so stats measure the swarm's own light.
 *
 * Self-contained on purpose: talks to the sim only through `window.swarm`.
 */

export interface FilmstripPoint {
  /** Pointer in NDC [-1,1]². */
  x: number;
  y: number;
  /** Seconds from the start of the strip. */
  t: number;
}

export interface FilmstripOptions {
  frames?: number;
  /** Wall-clock gap between captures. */
  intervalMs?: number;
  cols?: number;
  /** Tile size as a fraction of the canvas pixel size. */
  scale?: number;
  /** Scripted pointer, linearly interpolated at ~60 Hz. */
  path?: FilmstripPoint[];
  /** Seconds at which to call `swarm.burst()`. */
  click?: number[];
  label?: string;
}

export interface FrameStat {
  /** Seconds since the strip started (actual capture time). */
  t: number;
  mood: string;
  /** Fraction of tile pixels with max channel >= 0.92: blown-out glow. */
  hotFrac: number;
  /** Fraction with max channel >= 0.7. */
  brightFrac: number;
  /** Fraction with max channel >= 0.12: the school's footprint. */
  litFrac: number;
  /** Largest 4-connected hot blob as a fraction of lit pixels. */
  peakBlob: number;
  /** Centroid of lit pixels in tile pixel coords (null if nothing lit). */
  centroid: { x: number; y: number } | null;
}

export interface FilmstripResult {
  /** image/jpeg, quality 0.85. */
  dataUrl: string;
  frames: FrameStat[];
  summary: string;
}

interface LooseSwarm {
  moodName?: string;
  getCount?: () => number;
  updateMousePosition: (ndc: { x: number; y: number }) => void;
  clearPointer: () => void;
  burst: () => void;
}

type Win = Window & { swarm?: LooseSwarm; __render?: () => void; __keepAlive?: boolean };

const HOT = 0.92 * 255;
const BRIGHT = 0.7 * 255;
const LIT = 0.12 * 255;
const TAIL_S = 0.25;
const HEADER_PX = 18;

function samplePath(path: FilmstripPoint[], t: number): { x: number; y: number } | null {
  if (path.length === 0 || t < path[0].t || t > path[path.length - 1].t) return null;
  for (let i = 1; i < path.length; i++) {
    const a = path[i - 1];
    const b = path[i];
    if (t <= b.t) {
      const k = b.t > a.t ? (t - a.t) / (b.t - a.t) : 1;
      return { x: a.x + (b.x - a.x) * k, y: a.y + (b.y - a.y) * k };
    }
  }
  return { x: path[0].x, y: path[0].y };
}

/** Pixel stats over one tile's RGBA. Blob search is an iterative flood fill
 * over hot pixels only, so its cost scales with the glow, not the tile. */
function tileStats(
  img: ImageData,
): Omit<FrameStat, "t" | "mood"> & { maxValue: number } {
  const { width: w, height: h, data } = img;
  const n = w * h;
  const hotMask = new Uint8Array(n);
  let hot = 0;
  let bright = 0;
  let lit = 0;
  let sx = 0;
  let sy = 0;
  let maxValue = 0;
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    const m = Math.max(data[p], data[p + 1], data[p + 2]);
    if (m > maxValue) maxValue = m;
    if (m >= LIT) {
      lit++;
      sx += i % w;
      sy += (i / w) | 0;
      if (m >= BRIGHT) {
        bright++;
        if (m >= HOT) {
          hot++;
          hotMask[i] = 1;
        }
      }
    }
  }
  let biggest = 0;
  if (hot > 0) {
    const stack = new Int32Array(hot);
    for (let i = 0; i < n; i++) {
      if (hotMask[i] !== 1) continue;
      hotMask[i] = 2;
      let top = 0;
      stack[top++] = i;
      let size = 0;
      while (top > 0) {
        const j = stack[--top];
        size++;
        const x = j % w;
        if (x > 0 && hotMask[j - 1] === 1) { hotMask[j - 1] = 2; stack[top++] = j - 1; }
        if (x < w - 1 && hotMask[j + 1] === 1) { hotMask[j + 1] = 2; stack[top++] = j + 1; }
        if (j >= w && hotMask[j - w] === 1) { hotMask[j - w] = 2; stack[top++] = j - w; }
        if (j + w < n && hotMask[j + w] === 1) { hotMask[j + w] = 2; stack[top++] = j + w; }
      }
      if (size > biggest) biggest = size;
    }
  }
  return {
    hotFrac: hot / n,
    brightFrac: bright / n,
    litFrac: lit / n,
    peakBlob: lit > 0 ? biggest / lit : 0,
    centroid: lit > 0 ? { x: sx / lit, y: sy / lit } : null,
    maxValue,
  };
}

function stamp(ctx: CanvasRenderingContext2D, text: string, x: number, y: number): void {
  ctx.font = "11px ui-monospace, Menlo, monospace";
  ctx.textBaseline = "top";
  ctx.lineJoin = "round";
  ctx.lineWidth = 3;
  ctx.strokeStyle = "rgba(0,0,0,0.85)";
  ctx.strokeText(text, x, y);
  ctx.fillStyle = "#fff";
  ctx.fillText(text, x, y);
}

const pct = (v: number): string => `${(v * 100).toFixed(1)}%`;

/** Filmstrips in flight, and the keep-alive value from before the first. */
let filming = 0;
let keepAliveBefore: boolean | undefined;

export async function runFilmstrip(opts: FilmstripOptions = {}): Promise<FilmstripResult> {
  // The strip films the running loop. A hidden pane (rAF suspended) only
  // steps through main.ts's fallback timer, which skips hidden tabs unless
  // kept alive; without this every frame is the same frozen school. Counted,
  // not saved per call: with overlapping strips, a save/restore pair turned
  // keep-alive off under the second strip and then left it on for good.
  const win = window as Win;
  if (filming++ === 0) keepAliveBefore = win.__keepAlive;
  win.__keepAlive = true;
  try {
    return await filmstrip(opts);
  } finally {
    if (--filming === 0) win.__keepAlive = keepAliveBefore;
  }
}

async function filmstrip(opts: FilmstripOptions): Promise<FilmstripResult> {
  const win = window as Win;
  const swarm = win.swarm;
  if (!swarm) throw new Error("no live swarm (window.swarm missing)");
  const source = document.querySelector<HTMLCanvasElement>("#app canvas");
  if (!source) throw new Error("no swarm canvas in #app");

  // NaN or Infinity would schedule nothing (or forever) and never resolve.
  const num = (v: number | undefined, d: number): number => (Number.isFinite(v) ? (v as number) : d);
  const frames = Math.max(1, Math.floor(num(opts.frames, 8)));
  const intervalMs = Math.max(0, num(opts.intervalMs, 400));
  const cols = Math.max(1, Math.min(frames, Math.floor(num(opts.cols, 4))));
  const rows = Math.ceil(frames / cols);
  const scale = num(opts.scale, 0.3) > 0 ? num(opts.scale, 0.3) : 0.3;
  const path = [...(opts.path ?? [])].sort((a, b) => a.t - b.t);
  const clicks = opts.click ?? [];

  const tw = Math.max(1, Math.round(source.width * scale));
  const th = Math.max(1, Math.round(source.height * scale));
  const header = opts.label ? HEADER_PX : 0;
  const strip = document.createElement("canvas");
  strip.width = cols * tw + (cols - 1);
  strip.height = header + rows * th + (rows - 1);
  const ctx = strip.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("2D canvas unavailable");
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, strip.width, strip.height);
  if (opts.label) stamp(ctx, opts.label, 4, 3);

  const start = performance.now();
  const elapsed = (): number => (performance.now() - start) / 1000;
  // Recent pointer positions (NDC) for the tile's tail.
  const trail: { x: number; y: number; t: number }[] = [];
  let pointerActive = false;
  const timers: number[] = [];

  const drivePointer = (): void => {
    if (path.length === 0) return;
    const t = elapsed();
    const p = samplePath(path, t);
    if (p) {
      swarm.updateMousePosition(p);
      pointerActive = true;
      trail.push({ ...p, t });
      while (trail.length && trail[0].t < t - TAIL_S) trail.shift();
    } else if (pointerActive && t > path[path.length - 1].t) {
      swarm.clearPointer();
      pointerActive = false;
      trail.length = 0;
    }
  };
  let pointerTimer = 0;
  // Any throw inside a timer lands here: stop everything and reject, or the
  // caller's await hangs forever on a resolve that never comes.
  let fail: (err: unknown) => void = () => {};
  const stopAll = (): void => {
    timers.forEach((id) => window.clearTimeout(id));
    if (pointerTimer) window.clearInterval(pointerTimer);
    pointerTimer = 0;
    try {
      swarm.clearPointer();
    } catch {
      // Already failing; the original error is the one worth reporting.
    }
    pointerActive = false;
  };
  const guarded = (fn: () => void) => (): void => {
    try {
      fn();
    } catch (err) {
      fail(err);
    }
  };

  const stats: FrameStat[] = [];
  const blank: number[] = [];

  const capture = (i: number): void => {
    // Render and read in one task: the WebGPU canvas texture is only
    // guaranteed readable before this task yields.
    win.__render?.();
    const t = elapsed();
    const col = i % cols;
    const row = Math.floor(i / cols);
    const ox = col * (tw + 1);
    const oy = header + row * (th + 1);
    ctx.drawImage(source, 0, 0, source.width, source.height, ox, oy, tw, th);
    // Stats before any overlay so the stamp and pointer ring aren't counted.
    const { maxValue, ...s } = tileStats(ctx.getImageData(ox, oy, tw, th));
    if (maxValue === 0) blank.push(i);
    const mood = swarm.moodName ?? "?";
    const n = swarm.getCount?.() ?? 0;
    stats.push({ t, mood, ...s });

    if (pointerActive && trail.length) {
      const toTile = (p: { x: number; y: number }): [number, number] => [
        ox + ((p.x + 1) / 2) * tw,
        oy + ((1 - p.y) / 2) * th,
      ];
      ctx.save();
      ctx.beginPath();
      ctx.rect(ox, oy, tw, th);
      ctx.clip();
      ctx.strokeStyle = "rgba(255,60,60,0.55)";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      trail.forEach((p, k) => {
        const [x, y] = toTile(p);
        if (k === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      const [hx, hy] = toTile(trail[trail.length - 1]);
      ctx.strokeStyle = "#ff3c3c";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(hx, hy, 5, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    }
    stamp(
      ctx,
      `t=${t.toFixed(1)}s mood=${mood} n=${n} hot=${pct(s.hotFrac)}`,
      ox + 4,
      oy + 3,
    );
  };

  try {
    await new Promise<void>((resolve, reject) => {
      fail = (err) => {
        stopAll();
        reject(err);
      };
      if (path.length) {
        guarded(drivePointer)();
        pointerTimer = window.setInterval(guarded(drivePointer), 1000 / 60);
      }
      for (const s of clicks) {
        if (Number.isFinite(s)) timers.push(window.setTimeout(guarded(() => swarm.burst()), s * 1000));
      }
      for (let i = 0; i < frames; i++) {
        timers.push(
          window.setTimeout(
            guarded(() => {
              capture(i);
              if (i === frames - 1) resolve();
            }),
            i * intervalMs,
          ),
        );
      }
    });
  } finally {
    // Repeats harmlessly after fail(); on success only a pointer this strip
    // drove gets cleared.
    timers.forEach((id) => window.clearTimeout(id));
    if (pointerTimer) window.clearInterval(pointerTimer);
    if (pointerActive) swarm.clearPointer();
  }

  const lines = stats.map(
    (s, i) =>
      `#${i} t=${s.t.toFixed(2)}s ${s.mood.padEnd(7)} lit=${pct(s.litFrac)} bright=${pct(s.brightFrac)} hot=${pct(s.hotFrac)} blob=${pct(s.peakBlob)}` +
      (s.centroid ? ` c=(${s.centroid.x.toFixed(0)},${s.centroid.y.toFixed(0)})` : " c=-"),
  );
  const maxHot = Math.max(...stats.map((s) => s.hotFrac));
  const maxBlob = Math.max(...stats.map((s) => s.peakBlob));
  const head = opts.label ? [`[${opts.label}] ${tw}x${th} tiles`] : [`${tw}x${th} tiles`];
  const tail = [`max hot=${pct(maxHot)} max blob=${pct(maxBlob)}`];
  if (blank.length) {
    tail.push(
      `WARNING: tiles ${blank.join(",")} are fully black - canvas read likely failed (hidden tab or device lost?)`,
    );
  }
  return {
    dataUrl: strip.toDataURL("image/jpeg", 0.85),
    frames: stats,
    summary: [...head, ...lines, ...tail].join("\n"),
  };
}

/** The standard predator test: pointer enters from the left at y=0, glides
 * to x=+0.3 by 1.6 s, parks until 3.2 s, then leaves. */
export function runSweep(opts: FilmstripOptions = {}): Promise<FilmstripResult> {
  return runFilmstrip({
    frames: 8,
    intervalMs: 450,
    label: "sweep",
    path: [
      { x: -1, y: 0, t: 0 },
      { x: 0.3, y: 0, t: 1.6 },
      { x: 0.3, y: 0, t: 3.2 },
    ],
    ...opts,
  });
}
