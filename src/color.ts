/**
 * Color helpers shared by both backends, so an accent or base override
 * derives the same gradient and flash hue on the GPU and the CPU fallback.
 */

export type Rgb = [number, number, number];

/** Small seeded PRNG: the same seed replays the same color draws. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function hexToRgb(hex: string): Rgb {
  const v = parseInt(hex.slice(1), 16);
  return [((v >> 16) & 255) / 255, ((v >> 8) & 255) / 255, (v & 255) / 255];
}

export function rgbToHex(rgb: Rgb): string {
  return "#" + rgb.map((c) => Math.round(c * 255).toString(16).padStart(2, "0")).join("");
}

/** rgb in 0..1 → [h 0..1, s, l]. */
export function rgbToHsl([r, g, b]: Rgb): Rgb {
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const l = (max + min) / 2;
  const d = max - min;
  if (d === 0) return [0, 0, l];
  const s = d / (1 - Math.abs(2 * l - 1));
  let h: number;
  if (max === r) h = ((g - b) / d + 6) % 6;
  else if (max === g) h = (b - r) / d + 2;
  else h = (r - g) / d + 4;
  return [h / 6, s, l];
}

/** [h 0..1, s, l] → rgb in 0..1. */
export function hslToRgb(h: number, s: number, l: number): Rgb {
  const k = (n: number) => (n + h * 12) % 12;
  const a = s * Math.min(l, 1 - l);
  const f = (n: number) => l - a * Math.max(-1, Math.min(k(n) - 3, 9 - k(n), 1));
  return [f(0), f(8), f(4)];
}

/**
 * Four dark → bright stops in the shape of the palettes' own: base sits at
 * stops[2] (the palette's main body color), the top stop washes out a bit.
 */
export function baseStops(hex: string): Rgb[] {
  const [h, s, l] = rgbToHsl(hexToRgb(hex));
  return [
    hslToRgb(h, s * 0.9, l * 0.45),
    hslToRgb(h, s, l * 0.8),
    hslToRgb(h, s, l),
    hslToRgb(h, s * 0.7, Math.min(0.92, l + 0.3)),
  ];
}

/** Flash hue for a base override: saturated and a touch lighter, so
 * flashes read as the same hue, lit. */
export function flashFromBase(hex: string): string {
  const [h, s, l] = rgbToHsl(hexToRgb(hex));
  return rgbToHex(hslToRgb(h, Math.min(1, s * 1.1), Math.min(0.85, l + 0.12)));
}
