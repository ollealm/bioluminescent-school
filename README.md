# bioluminescent swarm · GPU fork (v4)

An experimental rewrite of [bioluminescent-swarm](https://bioluminescent.vercel.app)
(v3, ~2,600 particles, CPU) as **WebGPU compute**: the same living school of
bioluminescent fish at **25,000 particles** by default (50k+ runs fine on
Apple silicon; the owner's taste call was that 20-30k larger fish read more
like a school than 50k dust), targeting 60 fps and degrading gracefully
below that. Destined to be the background of olle.work.

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

`?backend=cpu` forces the fallback; `?count=N` overrides the particle count;
`?feel=<name>` boots into one of the `snapshots/*.json` feels (see below).
Press **D** for the dev panel (same presets/palettes/sliders as v3; snapshots
flow between versions unchanged — `snapshots/site-default.json` is the
canonical tuned state both backends reproduce).

## Calibrating the feel (the A/B loop)

The hard part of this project is taste, so the tooling is built around
comparing full parameter sets live:

- **Feels** — every `snapshots/*.json` is a named feel. Keys **1–9** switch
  between them in place (the school keeps swimming; a feel is a temperament,
  not a density, so the snapshot's `count` is ignored), the URL updates to
  `?feel=<name>` for sharing, and the panel's *Feel A/B* row shows which is
  active. *Save feel…* downloads the current sliders as `<name>.json`; drop
  it into `snapshots/` and it appears on reload. `site-default` is the v3
  canonical state; **`olle-work` is the shipped feel** - a plain visit
  boots into it (a dev session with saved slider tweaks keeps those; press
  Reset or use `?feel=olle-work` to get back).
- **Four GPU-only levers** added for the brief (all default 0, so v3
  snapshots are bit-identical):
  - `collapse` adds a *contract* mood to the director: the school condenses
    toward its centroid through cohesion alone, then releases into a turn
    with a separation pop. Personal space never relaxes during it: that was
    tried and crushed the condensed school into one glowing mass.
  - `shape` makes cohesion anisotropic along the school heading so a
    traveling shoal stretches instead of staying a ball.
  - `bite` sharpens the danger zone's edge: the v3 zone is a smoothstep
    sphere nearly the size of the school, which reads as a gravity well
    displacing the school as a fluid (owner's complaint). With bite, nothing
    happens until a fish is genuinely close, then it bolts at full strength;
    the hole it leaves is made by bolting, fear and contagion, not by a wide
    gentle push. Pair it with a small `mouseRadius`.
  - `crowdDim` is the density-aware brightness budget: each sprite's opacity
    scales by `(nRef / nLocal)^(0.8·crowdDim)` where nLocal is a smooth
    tent-weighted fish count over the 27 fine collision cells around it and
    nRef the count at nominal school density. Dense cores stop piling up
    additively into one glowing spot while the resting body keeps its
    brightness. A box count was tried first and banded along the grid
    planes; the tent kernel is position-continuous.
- **Visual review** - `await __sweep()` (standard predator pass: the cursor
  enters from the left, crosses into the school, parks, leaves) and
  `await __filmstrip({ frames, intervalMs, path, click })` return one
  contact-sheet JPEG (data URL) of the live animation with per-frame stats:
  `hotFrac` (pixels near white, the "smushed glowing spot" signal), the
  largest hot blob, lit footprint and centroid. This is how the feel was
  reviewed without watching it move.
- **`await feel()`** in the console measures the live school for 30 s
  (no rebuild, so tweak sliders first): elongation, edge sharpness, internal
  density structure, breathing amplitude/period, velocity coherence swing,
  and the refill time behind a parked cursor. Measured 2026-10-06 at 25k:

  | | site-default | lifelike-a |
  | --- | --- | --- |
  | elongation (mean / max) | 1.21 / 1.28 | 1.42 / 1.71 |
  | breathing amplitude | ±3 % | ±23 % |
  | coherence swing (min…max) | 0.75…0.97 | 0.57…0.99 |
  | moods in 45 s | cruise/turn/startle/hover | + contract |

  The refill probe parks at the origin, so it is only meaningful while the
  school sits there; with roam it mostly measures the school moving away.

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
  **stratified by cell population**: a first pass counts the 27 cells, then
  each cell is examined for a share of `sampleCandidates` (96) proportional
  to its count, from a hash-randomized in-cell offset, in hash-rotated cell
  order. The earlier "first cell to fill the cap wins" sweep averaged over
  *cells* rather than *fish*, which is a periodic force with the cell period
  - measured as a ±10% density modulation at the perception-cell pitch that
  showed as sheets along the grid planes once crowd dim revealed the core's
  mid-tones (now ±3%, noise). Fair sampling must stay *pure*: an own-cell
  quota was tried for close-range locality and any position-quantized bias
  is static in space - it accreted the school into visible sheets within a
  minute. Both calibration scales below were re-derived for it.
- **The fine collision grid** (`fineGridDim`/`fineCellSize`/`pairPush`) —
  anti-clumping (owner's ask). Fair sampling starves close-range
  separation (an overlapped pair-mate is one of ~800 candidates for 16
  samples) and fish paired into blobs: 36% of the school had a nearest
  neighbor < 0.12 u vs ~7% for random placement — and no probabilistic
  push can fix it (sampled kicks break pairs, alignment re-syncs them;
  measured re-accumulating to 29% within a minute). A second counting-sort
  grid at 0.45 u cells detects every real overlap every frame from both
  sides — the continuous response v3 got by scanning all its neighbors —
  for ~a dozen extra position reads per particle. Overlap holds at **0%
  flat** with no grid-plane signature. The pushout damps with fear
  (threatened schools pack tighter; collision noise would decohere the
  fountain), disables near the predator (wall shoves would pump fish into
  the cleared column), and bypasses the shear-glow sense — dodging a
  neighbor is not agitation.
- **The hunter wedge — tried and reverted** (`wedgeAhead`/`wedgeSide`, now
  0): at speed the danger zone stretched into a lunge along the heading.
  Owner verdict after use: fish reacting ahead of the visible cursor read
  as *randomness*, not hunting — casual mouse movement triggered reactions
  with no legible cause. Cause-and-effect legibility beats predator
  realism; the influence zone is exactly v3's sphere again (the branch
  constant-folds out of the kernel) and the strike carries the hunter
  fantasy instead. The aggregate fountainDot still reads near 0 on this
  fork — the rate-compensated fear spread panics ~2× more of the school
  than v3 (lively by taste), and the spread-feared crowd swamps the
  directly-fleeing cohort — so the fountain signature to hold is
  `fountainBehindDot` (see acceptance).
- **The strike** (`strikeSpeed`/`strikeCooldown`, owner's ask) — a flick
  (pointer at rest bursting past 32 u/s) detonates a directional V-blast
  at the lunge tip: fish part hard away from the strike line with
  contact-grade fear and a wake of outward ignitions along the path
  (measured: one flick sends ~30% of a 20k school bolting at >4 u/s).
  A 1.2 s cooldown and the rest-to-fast edge detector keep it a *move* —
  stalk, lunge, strike — never a mode; sustained fast circling can't
  trigger it. Blast magnitudes sit in the click-burst family.
- **Contagion is heat-weighted** (`transmissionScale`). At 50k a lit fish
  is a *spark carrier*: the radial kick flies it a full perception radius
  while still contagious, and with fuel everywhere the burn re-seeds
  itself indefinitely — cascades stopped dying (~1 run in 3), which is
  invariant §4.3's "flickers forever" failure. Weighting each lit
  neighbor's contribution by its remaining heat starves aging carriers
  while `transmissionScale` (1.35 under stratified sampling) restores the
  fresh wavefront to the measured v3 growth curve. (Measured today, the v3 reference itself
  occasionally fails its own extinction probe — the mechanism exists at
  2.6k too, just rarely finds fuel.)
- **Refractory re-derived** (`refractoryScale = 1.6`), per invariant §4.3's
  explicit instruction: the wave crosses the denser school ~1.5× slower,
  so the refractory wall scales to keep re-entry impossible.
- **Predator crowd pressure** (`predatorSepRelax`): two jams the reference
  never felt at 20× density — the dense wall around the cleared column
  pushes fresh fish inward through pure separation pressure, and the panic
  wave can't *move* bodies (fleeing fish stall against personal space).
  Fish relax their own separation by `max(column falloff, fear while the
  threat is near) × 0.8`, so evacuation isn't diffusion-jammed and a strike
  visibly shatters the crowd. One honest density effect remains: a
  mid-school park is still *clearing* at the reference's 1.5 s mark (20×
  more bodies moving through each other).
- **Fear spread under sampling** (`fearSpreadScale`): panic spreads toward
  0.92× the *sampled* max neighbor fear. Under pure fair sampling that
  lagged the front of a moving panic wave ~4× (the strike read as a
  foreground-only flinch, needing a ×2.5 rate boost); the own-cell quota
  supplies front-hugging samples like v3's cell-order scan did; stratified
  sampling now does the same without the quota's bias, so the compensation
  is 1.0 (at 2.0 a parked cursor panicked 89% of the school). Ceiling
  semantics untouched throughout.
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
| fountain (mean dot vs travel) | −0.58 | ~0 by construction (hunter wedge herds prey forward before it parts) | < −0.2 on v3's zone shape |
| fountain, behind cohort (drop-behind beat) | ≈ fountainDot | −0.45 … −0.49 across runs | < −0.2 |
| predator vacancy @1.5 s / 3 s | 0 / 8 | 93 / 706 (mid-school park; 0 when off-school) | ~0 (see §5 notes) |
| panic reach | 0.22 | 0.59 | ≳ 0.2 |
| moods seen in 120 s | all 5 | all 5 | all 5 |
| mean-speed swing | 2.6× | 3.4× | ≥ 2× |
| peak hot-glow (frac) | 0.037 | 0.040 | small |

Round three (same day, owner: "the swarm should move a lot more, glow less
in interaction, and the stationary cursor is still a large circle"): the
feel's energy went up (maxSpeed 4.2, schoolDrive 1.4, tempo 1.8, gait 1.0,
startle 0.9, mill 0.4, flow 1.4, collapse 1.0, drag 0.45; mean speed
0.76…1.86 u/s vs 0.54…1.67 before), interaction light went down (splash
0.3, motionGlow 0.5, flashBrightness 0.7, stretch 1.3), and the cursor
shrank (mouseRadius 2.2, bite 0.7, fleeForce 25, fearDecay 2.0). Two code
changes came with it: the "still threatened" band that keeps frightened
fish fleeing outside the zone now shrinks with bite (1.4× → 1.12× the
radius at bite 0.7), and crowd dim's reference density is the school's
*live* mean density (Σ|pos − cen|² reduced on-GPU → RMS radius) instead of
the nominal tank volume, so a contracted school keeps its brightness and
only local excess is dimmed. Gait is capped at 1.0 on purpose: at 1.3 the
thrust impulses crossed the shear-glow knee and 13% of the school ran hot
(invariant §4.6); at 1.0 it is 1.7%.

Re-measured 2026-10-12 at 25,000 with the shipped `olle-work` feel after
stratified sampling and the re-derived scales (30 s mood run): cascade
.022 / .095 / .209 / .275 then extinct; outburst 0.97; inburst 0.91;
fountain −0.27 (bar < −0.2, back from ~0 under the old fear-spread
compensation); vacancy 0 / 67; panic reach 0.39; mean-speed swing 3.1×;
peak hot-glow 0.018; 1.75 ms per step.

Perf on Apple silicon (Metal 3, Chrome): **4.1 ms sim / 5.3 ms sim+render
per step at 50,000** — the CPU reference spends 5.3 ms on 2,600. 100,000
runs at 12.5 ms. Weaker WebGPU devices degrade via a frame-time governor
that steps the count down (floor 8k; a single stalled frame such as a tab
switch is ignored, only sustained misses count); no WebGPU, or a
software/fallback adapter, boots the vendored CPU v3 at its original counts.
A lost GPU device (driver reset, GPU process crash) re-mounts the backend
once with the current snapshot carried over, then falls back to CPU.

## Verification

`await harness()` in the console runs the v3 acceptance probes (async —
GPU state comes back via buffer readbacks) and returns the same report on
either backend, plus fraction-of-count versions of the count-valued metrics
for cross-density comparison. Two probes were hardened for validity at 50k
(fresh field + more seeds for the inburst probe; pointer-state hygiene and
a second 3 s vacancy mark) — deviations are commented in src/harness.ts.
`window.swarm` and `window.__render()` are exposed for poking, same as v3.
