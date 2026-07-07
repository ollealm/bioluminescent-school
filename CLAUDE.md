# Bioluminescent swarm — GPU fork (v4)

WebGPU-compute rewrite (50k particles) of `~/bioluminescent` (v3 = untouched
CPU reference, also bundled here as the no-WebGPU fallback). The behavior
contract is `~/bioluminescent/HANDOFF-GPU.md` — the four behavior layers, the
taste law (in-hue glow, never white; movement before light), and the eight
dynamical invariants all survive the port. Read it before changing behavior
or tuning.

Verify with `npm run typecheck` and the acceptance probes (`window.harness()`).
`snapshots/site-default.json` is the canonical tuned state — **both backends
must reproduce it**. `?backend=cpu` forces the fallback; `?count=N` overrides
particle count.

@README.md
