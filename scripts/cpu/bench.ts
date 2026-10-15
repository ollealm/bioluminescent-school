// CPU engine benchmark in both browser JS engines, mood-cycled, ms per frame
// per segment at the CPU feel's settings and 2,200 fish:
//
//   sh scripts/cpu/run.sh bench        # V8 (Chrome) via node, and
//                                      # JavaScriptCore (Safari) via macOS's jsc
//
// JavaScriptCore is the one to watch: a long update() fell to its slow tier
// on mood switches and stayed there (22-35 ms vs 3.7 ms here; 40-80 ms vs
// 5 ms on iPhones), which V8 never showed. macOS's jsc reproduces it.
import * as THREE from "three";
import { ParticleSystem } from "../../src/cpu/ParticleSystem";
import snap from "../../snapshots/olle-work.json";

const g = globalThis as any;
const now: () => number = g.preciseTime ? () => g.preciseTime() * 1000 : () => performance.now();
const log: (s: string) => void = g.print ?? ((s: string) => console.log(s));
if (!g.performance) g.performance = { now };

const sys = new ParticleSystem(new THREE.Scene(), { count: 2200, capacity: 8000 });
// As src/feel.ts does on the CPU backend: no GPU count, then the cpu block.
const { count: _gpuCount, cpu, ...rest } = snap as any;
sys.applySnapshot(rest);
const { count: _cpuCount, ...cpuParams } = cpu;
Object.assign(sys.params, cpuParams);
if (sys.getCount() !== 2200) sys.resize(2200);
const dt = 1 / 60;
const seg = (label: string, frames: number) => {
  const t0 = now();
  for (let f = 0; f < frames; f++) sys.update(dt);
  log(`${label.padEnd(9)} ${((now() - t0) / frames).toFixed(2)} ms`);
};
seg("boot", 120);
for (let round = 0; round < 2; round++) {
  for (const mood of ["cruise", "mill", "hover", "contract", "startle", "turn"]) {
    sys.forceMood(mood);
    seg(mood, 120);
  }
}
sys.forceMood(null);
seg("auto", 600);
