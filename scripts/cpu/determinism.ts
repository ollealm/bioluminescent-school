// Determinism check for src/cpu/ParticleSystem.ts: seeded Math.random, a
// fake clock, a scripted session (every mood, pointer, a strike, bursts,
// resizes), and a hash of every typed array on the instance after each
// phase. A refactor that must not change behavior prints the same hashes
// before and after (the 2026-10-15 split of update() did, bit for bit):
//
//   sh scripts/cpu/run.sh determinism > before.txt   # on the old code
//   sh scripts/cpu/run.sh determinism > after.txt    # on the new code
//   diff before.txt after.txt
//
// It boots the olle-work snapshot whole (count included, so 8,000 fish at
// the capacity cap): slow (~2 min), but it exercises every lever.
import { createHash } from "node:crypto";

// A fake clock, one frame per step: pointer kinematics read performance.now().
let clock = 0;
Object.defineProperty(globalThis, "performance", { value: { now: () => clock }, configurable: true });
let seed = 12345;
Math.random = () => {
  seed = (seed + 0x6d2b79f5) | 0;
  let t = seed;
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};

import * as THREE from "three";
import { ParticleSystem } from "../../src/cpu/ParticleSystem";
import snap from "../../snapshots/olle-work.json";

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(55, 16 / 9, 0.1, 200);
camera.position.set(0, 0, 26);
camera.updateMatrixWorld();
const sys = new ParticleSystem(scene, { camera, count: 2200, capacity: 8000 });
sys.applySnapshot(snap);

const hash = (): string => {
  const h = createHash("sha256");
  for (const [k, v] of Object.entries(sys).sort(([a], [b]) => (a < b ? -1 : 1))) {
    if (ArrayBuffer.isView(v)) {
      h.update(k);
      h.update(new Uint8Array(v.buffer, v.byteOffset, v.byteLength));
    }
  }
  h.update(String(sys.moodName));
  return h.digest("hex").slice(0, 16);
};

const dt = 1 / 60;
const out: string[] = [];
const run = (label: string, frames: number, each?: (f: number) => void) => {
  for (let f = 0; f < frames; f++) {
    each?.(f);
    clock += dt * 1000;
    sys.update(dt);
  }
  out.push(`${label.padEnd(12)} ${hash()}  mood ${sys.moodName}`);
};

run("boot", 120);
for (const mood of sys.moodNames as string[]) {
  sys.forceMood(mood);
  run(mood, 90);
}
sys.forceMood(null);
run("auto", 300);
run("pointer", 120, (f) => sys.updateMousePosition({ x: Math.sin(f / 15) * 0.6, y: Math.cos(f / 21) * 0.4 }));
run("strike", 30, (f) => sys.updateMousePosition({ x: -0.8 + f * 0.06, y: 0.1 }));
sys.clearPointer();
sys.burst();
run("burst", 120);
sys.scatter();
sys.startle();
run("scatter", 120);
sys.resize(1000);
run("shrink", 120);
sys.resize(3000);
run("grow", 180);
console.log(out.join("\n"));
