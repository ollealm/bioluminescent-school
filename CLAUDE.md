# Bioluminescent swarm — GPU fork (v4)

WebGPU-compute rewrite (25k particles by default, 50k+ capable) of the v3 CPU
swarm. The CPU engine in `src/cpu` is no longer the untouched v3 reference: it
is the v3 engine with every GPU lever ported (see `docs/NOTES.md` §3), shipped
as the no-WebGPU fallback. The untouched v3 lives in a separate private repo
and is not needed here. The behavior contract is `docs/HANDOFF-GPU.md` — the
four behavior layers, the taste law (in-hue glow, never white; movement before
light), and the eight dynamical invariants all survive the port. Read it
before changing behavior or tuning; `docs/NOTES.md` has the design chronicle
(why each lever exists, what was tried and reverted).

Verify with `npm run typecheck` and the acceptance probes (`window.harness()`).
`snapshots/site-default.json` is v3's canonical state (the GPU backend
reproduces it; the CPU engine keeps its always-on ports, see docs/NOTES.md §3)
and `snapshots/olle-work.json` is the shipped feel. `?backend=cpu` forces the fallback; `?count=N` overrides
particle count; `?alive=1` keeps a hidden tab simulating; `?feel=<name>`
boots a `snapshots/*.json` feel (`await feel()` measures shape/breathing/coherence of the live
school; `await __sweep()` returns a filmstrip JPEG of the predator pass with
hot-pixel stats for visual review). `snapshots/olle-work.json` is the
shipped feel and the boot default; `site-default` is the v3 state.

@README.md
