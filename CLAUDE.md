# Bioluminescent swarm — GPU fork (v4)

WebGPU-compute rewrite (25k particles by default, 50k+ capable) of `~/bioluminescent` (v3 = untouched
CPU reference, also bundled here as the no-WebGPU fallback). The behavior
contract is `~/bioluminescent/HANDOFF-GPU.md` — the four behavior layers, the
taste law (in-hue glow, never white; movement before light), and the eight
dynamical invariants all survive the port. Read it before changing behavior
or tuning.

Verify with `npm run typecheck` and the acceptance probes (`window.harness()`).
`snapshots/site-default.json` is the canonical tuned state — **both backends
must reproduce it**. `?backend=cpu` forces the fallback; `?count=N` overrides
particle count; `?feel=<name>` boots a `snapshots/*.json` feel (keys 1–9
switch live; `await feel()` measures shape/breathing/coherence of the live
school; `await __sweep()` returns a filmstrip JPEG of the predator pass with
hot-pixel stats for visual review). `snapshots/olle-work.json` is the
shipped feel and the boot default; `site-default` is the v3 state.

@README.md
