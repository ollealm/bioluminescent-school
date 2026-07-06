# bioluminescent swarm · GPU fork (v4)

An experimental rewrite of [bioluminescent-swarm](https://bioluminescent.vercel.app)
(v3, ~2,600 particles, CPU) as **WebGPU compute**: the same living school of
bioluminescent fish at **50,000 particles**, targeting 60 fps on Apple-silicon
laptops and degrading gracefully below that.

The behavior contract is `HANDOFF-GPU.md` in the reference repo: the four
behavior layers, the taste law (in-hue glow never white, movement before
light, green over cyan, bias fast), and the eight dynamical invariants all
survive the port. The CPU version remains the untouched reference
implementation — and ships inside this fork as the no-WebGPU fallback.

```
npm install
npm run dev        # http://localhost:5209
npm run typecheck
```

`?backend=cpu` forces the fallback; `?count=N` overrides the particle count.
Press **D** for the dev panel (same presets/palettes/sliders as v3; snapshots
flow between versions unchanged — `snapshots/site-default.json` is the
canonical tuned state both backends reproduce).

## Architecture

```
src/
  params.ts        shared spec: sliders, palettes, TUNING (v3 verbatim) + GPU_TUNING
  presets.ts       v3 verbatim
  system.ts        SwarmSystem — the interface both backends implement
  main.ts          boot: WebGPU detect → gpu/ ; else lazy-load cpu/ (three.js chunk)
  harness.ts       acceptance probes (async, backend-agnostic port of v3's)
  DevPanel.ts      v3 panel, count ceiling raised for the GPU backend
  gpu/             the rewrite — no three.js, no dependencies
    GpuSwarm.ts    host side: director FSM, event queue, uniforms, buffers
    wgsl-sim.ts    grid build, event resolve, sim kernel, reductions (WGSL)
    wgsl-render.ts instanced sprites + hue-preserving tonemap (WGSL)
    renderer.ts    HDR pipeline, canvas, density-compensated brightness
    camera.ts      fixed camera + pointer unprojection (replaces three.js)
  cpu/             v3 ParticleSystem verbatim + a thin adapter
```

**Split.** Whole-school state stays host-side, exactly as HANDOFF §2.2
prescribes: the mood FSM (cruise/turn/mill/hover/startle), shared heading,
roam anchor, pointer kinematics, and spontaneous-flash scheduling are a
handful of scalars per frame, written into one 256-byte uniform buffer.
Everything per-particle lives in GPU storage buffers and never crosses back
— except one 32-byte async readback (centroid + flock count + debug stats),
1–2 frames stale, which matches the CPU version's "last frame's centroid"
semantics.

**Frame anatomy** (one compute pass, ~8 dispatches): clear grid → count →
prefix-scan → scatter (counting-sort uniform grid, cell edge = perception)
→ resolve events → sim kernel → 2-stage reduction. The sim kernel walks
particles in grid order, so workgroups process spatially-coherent fish.
Rendering reads the sim's storage buffers directly: instanced quads with the
v3 sprite math (styles, velocity stretch, twinkle, in-hue flash mix),
accumulated additively into an rgba16float target, then resolved through a
shoulder tonemap.

**Host→swarm interventions** (click burst, startle, turn vanguard, wander
recruitment, scatter, spontaneous ignitions) travel through a 16-slot event
queue. Events that need a fish's position as their epicenter carry a
particle index; a one-workgroup "resolve" pre-pass fills in the position
on-GPU, so scheduling behavior never waits on a readback.

## What deliberately changed at 20× density (HANDOFF §5)

Every change is a named constant in `GPU_TUNING` (src/params.ts) with the
rationale in place; user-facing sliders keep their v3 meaning throughout.

- **Brightness budget** — automatic: rendered size/opacity scale by
  `(2600/count)^k` (`sizeCompExp`/`opacityCompExp`), so the tuned params
  mean the same perceived brightness at any count, and the canonical
  snapshot stays valid verbatim.
- **Never-white** is now structural, not just tuned: the HDR shoulder
  tonemap (`tonemapKnee`) compresses additive pile-ups toward brightness 1
  while preserving hue ratios — dense cores saturate *as the palette
  color*. Screen sampling during full-school cascades finds zero
  near-white pixels.
- **Neighbor sampling** — the 16-neighbor cap survives, but candidates are
  sampled with per-particle-per-frame hash-randomized cell order and
  in-cell offsets instead of v3's grid-cell-order bias (which would read
  as anisotropy at 50k).
- **Contagion is heat-weighted** (`transmissionScale`). At 50k a lit fish
  is a *spark carrier*: the radial kick flies it a full perception radius
  while still contagious, and with fuel everywhere the burn re-seeds
  itself indefinitely — cascades stopped dying (~1 run in 3), which is
  invariant §4.3's "flickers forever" failure. Weighting each lit
  neighbor's contribution by its remaining heat starves aging carriers
  while `transmissionScale = 2.0` restores the fresh wavefront to the
  measured v3 growth curve. (Measured today, the v3 reference itself
  occasionally fails its own extinction probe — the mechanism exists at
  2.6k too, just rarely finds fuel.)
- **Refractory re-derived** (`refractoryScale = 1.6`), per invariant §4.3's
  explicit instruction: the wave crosses the denser school ~1.5× slower,
  so the refractory wall scales to keep re-entry impossible.
- **Predator-column crowd pressure** (`predatorSepRelax`): the dense wall
  around the cleared column pushes fresh fish inward through pure
  separation pressure; fish inside the danger zone relax their own
  separation by `fall × 0.8` so evacuation isn't diffusion-jammed.
  Two honest density effects remain: a mid-school park is still *clearing*
  at the reference's 1.5 s mark (20× more bodies moving through each
  other), and the long-park equilibrium leaks a rim shell ~3% of the
  school at 6 s — newly measured at ~1.7% in the untouched v3, so a shared
  trait scaled by crowd pressure, not a regression.
- **Absolute counts → fractions** — wander group size is read as "fish per
  2,600" and recruited by probability; scatter's 12 ignition seeds became
  `scatterIgniteFrac`.
- **World scale** — same tank (containRadius 9, camera z 26, fov 55°):
  denser and more liquid, per the fork brief; all pointer/fear radii keep
  their meaning.

## Acceptance (measured 2026-07-06, same harness, 120 s mood runs)

| Probe (§6) | CPU v3 @2,600 | GPU fork @50,000 | Bar |
| --- | --- | --- | --- |
| cascade growth (frac of school at ~0/⅓/⅔/1 s) | .005 / .031 / .099 / .240 | .020 / .067 / .105 / .160 | grows, then dies |
| cascade extinguishes ≤ 5 s | true (fails ~1 in 4 runs) | true (10/10 runs) | true |
| outburst outward coherence | 0.86 | 0.99 | ≳ 0.7 |
| inburst inward coherence | 0.49 | 0.85 | ≳ 0.55 |
| fountain (mean dot vs travel) | −0.58 | −0.33 … −0.72 across runs | < −0.2 |
| predator vacancy @1.5 s / 3 s | 0 / 8 | 93 / 706 (mid-school park; 0 when off-school) | ~0 (see §5 notes) |
| panic reach | 0.22 | 0.59 | ≳ 0.2 |
| moods seen in 120 s | all 5 | all 5 | all 5 |
| mean-speed swing | 2.6× | 3.4× | ≥ 2× |
| peak hot-glow (frac) | 0.037 | 0.040 | small |

Perf on Apple silicon (Metal 3, Chrome): **4.1 ms sim / 5.3 ms sim+render
per step at 50,000** — the CPU reference spends 5.3 ms on 2,600. 100,000
runs at 12.5 ms. Weaker WebGPU devices degrade via a frame-time governor
that steps the count down (floor 8k); no WebGPU falls back to the vendored
CPU v3 at its original counts.

## Verification

`await harness()` in the console runs the v3 acceptance probes (async —
GPU state comes back via buffer readbacks) and returns the same report on
either backend, plus fraction-of-count versions of the count-valued metrics
for cross-density comparison. Two probes were hardened for validity at 50k
(fresh field + more seeds for the inburst probe; pointer-state hygiene and
a second 3 s vacancy mark) — deviations are commented in src/harness.ts.
`window.swarm` and `window.__render()` are exposed for poking, same as v3.
