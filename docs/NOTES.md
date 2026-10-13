# Design notes

This is the working chronicle of the GPU fork: what had to change when the
v3 CPU school (2,600 fish) became 25,000–50,000 fish in WebGPU compute, and
then how the feel was tuned, round by round, against Olle's notes until it
was ready to sit behind olle.work. The levers it describes all exist in the
code (`src/params.ts`, `snapshots/olle-work.json`), but the notes keep the
reasons, the measurements and the wrong turns that the code alone does not.
Olle's quotes are kept as given: they are the brief each round answered.
The behavior contract everything here is measured against is
[`HANDOFF-GPU.md`](HANDOFF-GPU.md); section references ("HANDOFF §5",
"invariant §4.3") point into it.

## Contents

1. [What deliberately changed at 20× density](#1-what-deliberately-changed-at-20-density-handoff-5)
2. [Calibrating the feel, round by round](#2-calibrating-the-feel-round-by-round)
3. [The CPU port](#3-the-cpu-port)
4. [Measurements](#4-measurements)

---

## 1. What deliberately changed at 20× density (HANDOFF §5)

Every change is a named constant in `GPU_TUNING` (src/params.ts) with the
rationale in place; user-facing sliders keep their v3 meaning throughout.

- **Brightness budget** — automatic: rendered size/opacity scale by
  `(2600/count)^k` (`sizeCompExp`/`opacityCompExp`), so the tuned params
  mean the same perceived brightness at any count, and the canonical
  snapshot stays valid verbatim.
  *Later superseded (2026-10-13):* scaling sprites with count made a phone
  at 12k draw bigger, brighter sprites than a desktop at 25k, which read as
  blur. Size compensation is now 0 (Size is literal at any count); opacity
  compensation came back on Olle's call ("opacity makes sense, size
  doesn't"), relative to 25,000: Opacity scales by
  `(25000/count)^0.35`, so 0.4 at 25k reads as ~0.52 at 12k.
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
  — measured as a ±10% density modulation at the perception-cell pitch that
  showed as sheets along the grid planes once crowd dim revealed the core's
  mid-tones (now ±3%, noise). Fair sampling must stay *pure*: an own-cell
  quota was tried for close-range locality and any position-quantized bias
  is static in space — it accreted the school into visible sheets within a
  minute. Both calibration scales below were re-derived for it.
- **The fine collision grid** (`fineGridDims`/`fineCellSize`/`pairPush`) —
  anti-clumping (Olle's ask). Fair sampling starves close-range
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
  neighbor is not agitation. The grid spans ±32.4 × ±21.6 × ±14.4 u
  (144 × 96 × 64 cells): the first 64³ cube (±14.4 u) clamped wander
  groups on a 16:9 screen into its edge cells, and crowd dim read the
  pile-up as density, dimming departing groups to 0.2–0.6× at x ≈ ±14.4.
- **The hunter wedge — tried and reverted** (`wedgeAhead`/`wedgeSide`, now
  0): at speed the danger zone stretched into a lunge along the heading.
  Olle's verdict after use: fish reacting ahead of the visible cursor read
  as *randomness*, not hunting — casual mouse movement triggered reactions
  with no legible cause. Cause-and-effect legibility beats predator
  realism; the influence zone is exactly v3's sphere again (the branch
  constant-folds out of the kernel) and the strike carries the hunter
  fantasy instead. The aggregate fountainDot read near 0 on this fork for
  a while — the rate-compensated fear spread panicked ~2× more of the
  school than v3 (lively by taste), and the spread-feared crowd swamped the
  directly-fleeing cohort — so the fountain signature held then was
  `fountainBehindDot`. With stratified sampling and fear spread back at
  1.0, fountainDot is −0.27 again (see §4).
- **The strike** (`strikeSpeed`/`strikeCooldown`, Olle's ask) — a flick
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
  fresh wavefront to the measured v3 growth curve. (Measured at the time,
  the v3 reference itself occasionally fails its own extinction probe —
  the mechanism exists at 2.6k too, just rarely finds fuel.)
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
  supplied front-hugging samples like v3's cell-order scan did; stratified
  sampling now does the same without the quota's bias, so the compensation
  is 1.0 (at 2.0 a parked cursor panicked 89% of the school). Ceiling
  semantics untouched throughout.
- **Absolute counts → fractions** — wander group size is read as "fish per
  2,600" and recruited by probability; scatter's 12 ignition seeds became
  `scatterIgniteFrac`.
- **World scale** — same tank (containRadius 9, camera z 26, fov 55°):
  denser and more liquid, per the fork brief; all pointer/fear radii keep
  their meaning.

## 2. Calibrating the feel, round by round

The hard part of this project is taste, so the tooling is built around
comparing full parameter sets live:

- **Feels** — every `snapshots/*.json` is a named feel. Keys **1–9** switch
  between them in place (the school keeps swimming; a feel is a temperament,
  not a density, so the snapshot's `count` is ignored), the URL updates to
  `?feel=<name>` for sharing, and the panel's *Feel A/B* row shows which is
  active. *Save feel…* downloads the current sliders as `<name>.json`; drop
  it into `snapshots/` and it appears on reload. `site-default` is the v3
  canonical state; **`olle-work` is the shipped feel** — a plain visit
  boots into it (a dev session with saved slider tweaks keeps those; press
  Reset or use `?feel=olle-work` to get back).
- **Visual review** — `await __sweep()` (standard predator pass: the cursor
  enters from the left, crosses into the school, parks, leaves) and
  `await __filmstrip({ frames, intervalMs, path, click })` return one
  contact-sheet JPEG (data URL) of the live animation with per-frame stats:
  `hotFrac` (pixels near white, the "smushed glowing spot" signal), the
  largest hot blob, lit footprint and centroid. This is how the feel was
  reviewed without watching it move.
- **`await feel()`** measures the live school for 30 s (no rebuild, so
  tweak sliders first): elongation, edge sharpness, internal density
  structure, breathing amplitude/period, velocity coherence swing, and the
  refill time behind a parked cursor.

### Rounds one and two — the first GPU-only levers

The brief for the olle.work relaunch was a school "constantly moving and
collapsing in on itself". Four GPU-only levers came out of it (all default
0, so v3 snapshots are bit-identical):

- `collapse` adds a *contract* mood to the director: the school condenses
  toward its centroid through cohesion alone, then releases into a turn
  with a separation pop. Personal space never relaxes during it: that was
  tried and crushed the condensed school into one glowing mass.
- `shape` makes cohesion anisotropic along the school heading so a
  traveling shoal stretches instead of staying a ball.
- `bite` sharpens the danger zone's edge: the v3 zone is a smoothstep
  sphere nearly the size of the school, which reads as a gravity well
  displacing the school as a fluid (Olle's complaint). With bite, nothing
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

Round one's candidate feel, `lifelike-a` (collapse 1, shape 0.6, a
tighter, faster school; since retired), measured against `site-default`
in §4. Round two's feedback on every preset was that the pointer read as a
gravity push and dense cores smushed into one glowing spot; `bite` and
`crowdDim` answered it, and crowd dim in turn revealed the ±10% sampling
bias described in §1 (the sheets along the grid planes).

**Accent fish** (appearance only): `accentShare` (default 0.05, 0 removes
them) is the fraction of fish in the palette's accent color, replacing each
palette's fixed `accentChance`; the panel's *Accent* picker overrides the
color itself, saved as the snapshot's optional `accent` key (a feel without
one shows the palette's own). Recolors replay each fish's seeded draws, so
only the fish that change role change color. Both backends.

### Round three — more motion, less light, a smaller cursor

Olle: "the swarm should move a lot more, glow less in interaction, and the
stationary cursor is still a large circle". The feel's energy went up
(maxSpeed 4.2, schoolDrive 1.4, tempo 1.8, gait 1.0, startle 0.9, mill 0.4,
flow 1.4, collapse 1.0, drag 0.45; mean speed 0.76…1.86 u/s vs 0.54…1.67
before), interaction light went down (splash 0.3, motionGlow 0.5,
flashBrightness 0.7, stretch 1.3), and the cursor shrank (mouseRadius 2.2,
bite 0.7, fleeForce 25, fearDecay 2.0). Two code changes came with it: the
"still threatened" band that keeps frightened fish fleeing outside the zone
now shrinks with bite (1.4× → 1.12× the radius at bite 0.7), and crowd
dim's reference density is the school's *live* mean density
(Σ|pos − cen|² reduced on-GPU → RMS radius) instead of the nominal tank
volume, so a contracted school keeps its brightness and only local excess
is dimmed. Gait is capped at 1.0 on purpose: at 1.3 the thrust impulses
crossed the shear-glow knee and 13% of the school ran hot (invariant §4.6);
at 1.0 it is 1.7%.

### Round four — `flight`, and the park-and-stir rule

Olle: "keep the small area around the cursor but make the reaction much
stronger, so hit fish leave the school and take others with them, with a
dynamic delay". A fish whose fear exceeds 1 − flight keeps bolting away
from the cursor *outside* the threat band instead of flipping into
regroup, and the regroup surge, the commit-to-return and the post-strike
homing all wait until its fear has decayed. With panic spread the wave
reaches neighbors hop by hop (0.92× per hop), so flight sets the reach in
hops and fearSpread sets the delay. olle-work: flight 0.7, spread 2.0,
decay 2.0, repel 90, flee 60, regroup 4.0 — the intensity comes from
force, the brevity from fast decay.

Two wrong turns are worth remembering: a low flight threshold with slow
decay (0.9 / 0.35) kept lingering fear damping cohesion for seconds (the
school went broad and slow), and spread 4.0 let a cursor that *stays* over
the school re-infect fear faster than it decays, so every fish stayed in
flight radially away from the pointer (a sphere again) and nothing
regrouped until the cursor left — a pass that exits after 2 s looked great
in the filmstrip and hid it. **Test every predator change with a parked
cursor *and* a continuous stir.**

Spontaneous light was made subtle at the same time (spontaneous 0.2,
flashBrightness 0.55, kick 1.6; transmission stays 2.8 because at 2.2
click cascades no longer grew and smoldered past the 5 s bar).

### Round five — `rally`

Olle: "the regroup should be dynamic, chaotic, violent — fish wanting the
center of the school, not just the edge". Once flight ends, fish drive
straight at the centroid with no distance gate and no arrival damping,
with a raised speed cap, so they overshoot through it and the body pulses.
Chosen by a 16-combination sweep (rally × regroup × fearDecay × flight,
each run as a 3 s cursor stir then 5 s of recovery, with the school's RMS
radius traced to measure overshoot) plus one Haiku reviewer per filmstrip
and a Fable pass over the top candidates. Three candidate feels shipped
beside `olle-work` (since retired): `rally-tight` (rally 0.5, regroup 3,
decay 1.5, flight 0.5: a clean snap into a core at ~60% of resting size),
`rally-pulse` (0.7 / 4 / 2.0 / 0.6: the school is carved into lobes, then
compact again within ~2 s) and `rally-storm` (1.0 / 5 / 2.5 / 0.8: tears
into lobes and voids for two seconds, then reassembles). Full rally with
slow decay (1.0 / 5 / 1.5) implodes into one bright ball and was rejected.

### Round six — `herd` and the alarm collapse

Olle: "real schools all seek the center; the regroup should collapse in on
itself and be more violent all the time". The selfish-herd drive pulls
every flocking fish toward the centroid with strength ∝ (r/R)² and no
arrival damping, so edge fish rush in and the body pulses against its own
crowd pressure; and the director now reads the school's mean fear (reduced
on-GPU) and, when a scare subsides, triggers a hard contraction (cohesion
1 + 1.4·collapse, herd gain 1 + 2.5·collapse, 4 s cooldown) so the whole
school implodes on return, not only the fish that fled. olle-work =
rally-tight + herd 0.4 + collapse 1.0. At herd 0.5 the resting school
became a hard-edged ball and a startle implosion ran 3% hot.

### Round seven — why no amount of center-pull made it alive, and what did

Olle: "herd just compacts the school; the cursor is still a sphere; it
needs to feel alive". Diagnosis from the filmstrips: (1) a cursor parked
inside a herd-pulled school reaches a static equilibrium shell; (2) 25,000
fish in a 3D ball seen face-on are stacked ~20 deep along the view axis, so
every density wave, hole and collapse averages away into a uniform disc
whatever the dynamics do. Four levers followed:

- `evade`: the school avoids the predator as a body. The roam target,
  heading, herd and rally centers shift away from a cursor inside the
  school, and a still predator's escape is tangential, so the hollow is a
  moving arc, not a shell.
- `shiver`: silent agitation waves at rest. A pocket darts, fear contagion
  carries it, and the alarm is armed so each wave ends in a
  re-contraction — no light.
- `sheet`: a damped z-spring toward the school's mean depth plus
  depth-weighted cohesion, so the school is a few fish deep and its
  structure shows. At 0.7 25k fish spread into a faint veil because the
  tank cannot hold them a few layers deep; 0.4 is the compromise.
- Cohesive wander groups: wanderers keep cohesion and alignment among
  themselves and recruit a pocket up to 4.8 u, so hundreds peel off as a
  sub-school and merge back. This one changes wander behavior at every
  setting; the CPU backend recruits up to groupSize fish as a plain count.

olle-work = rally-tight + sheet 0.4, cohesion 1.3, separation 1.3, herd
0.2, shiver 1.0, collapse 1.0, evade 0.6, groupSize 300, wanderRate 0.8.
Measured: elongation 4.8 mean, breathing ±59%, coherence swinging
0.22…0.95; a 3 s stir blows the body into a turbulent wave, implodes it
into a tight core at 1.4–2.8 s, reformed by 4.2 s.

### Round eight — the turn is a wave

Olle: "turn disrupts the flow; the vanguard pocket is too noticeable and
isn't leading; turns are too short". Before, a turn flipped the shared
heading for all 25k fish at once (drive 1.5) and the vanguard pocket was a
lit flash on top. Now the director remembers the old heading, the pocket
(ragged, no glow, a modest dart) takes the new one, and a wavefront grows
from the vanguard fish at `turnWaveSpeed` 3.2 u/s: fish outside it keep the
old heading in the drive and in the gait (the gait was a leak that turned
the wave back into a whole-school rotation), fish inside follow the new
one, and alignment carries the bend ahead of the front. Turns last
2.6–4 s / tempo at drive 1.2. Mood pin buttons in the panel (auto + each
mood) exist to test moods in isolation.

Pacing (Olle: "a new turn starts before the previous one is finished;
ideally one or two turns before the next mood"): a turn's duration is the
wave's crossing time (2.4·RMS radius / turnWaveSpeed + ~1 s), not
tempo-scaled; the front keeps growing after the mood ends so the school
never snaps onto the heading; an alarm collapse queues behind a running
turn instead of cutting it; a contraction releases into cruise (the
separation pop now applies to whatever mood follows it) instead of 70%
into a turn; cruise allows at most two turns in a row before mill or
hover; the spontaneous contract share is 0.2·collapse. Logged over 40 s:
two full-length turns, no chaining.

### Round nine — the cursor disc

In a sheet the hollow around a parked cursor became a clean 2D disc with a
crisp rim: fish just outside the zone are pushed in by cohesion and herd
and pop back out, and the tangential slide makes the rim an orbit.
Per-fish zone radii (0.75–1.25× static jitter, up to 1.6× upstream for a
fish swimming toward the predator) are in the kernel now but cannot break
the circle on their own: the visible edge is the smallest radius. What
fixed it is evade at full strength until the pointer is outside the live
school (1.6× RMS radius, then a short ramp) with a 0.9·containRadius
displacement, so the school leaves: the hole exists for ~1 s, becomes a
notch at the edge, and by ~3 s the cursor is outside a compact school.
olle-work at the time: bite 0.5, evade 0.85. Hand-tuning afterwards
settled lower (evade 0.2, bite 0.3, mouseRadius 1.7: a full-strength push
felt like too much displacement); `snapshots/olle-work.json` is the
current truth.

## 3. The CPU port

Olle found the no-WebGPU path was used more than expected and wanted it to
feel like the GPU version with fewer fish. The vendored v3 engine
(`src/cpu/ParticleSystem.ts`) now carries every GPU lever: bite, flight,
rally, evade, herd + the contract mood + the alarm collapse, shiver, sheet,
cohesive wander groups, shape, the turn wave, ragged pockets, roamDepth,
flashSize, crowdDim (on a count grid), accent/base/flash colors, mood pins,
the strike and the pointer depth. Only the compensations for GPU neighbor
sampling and 20× density are left out. The v3 loop stopped at the first 16
neighbors it found, so the GPU's fine collision grid came too: herd and
sheet otherwise crushed the school to a point. `site-default` on the CPU is
therefore no longer exactly v3 (turn wave, ragged pockets, per-fish zone,
alignment in every mode, strike, pointer depth and collision pushout are
always on); the untouched v3 lives in its own repo.

A feel's optional `cpu` block (`count` plus param overrides) is merged over
its params on the CPU backend, because the CPU has no hue-preserving
tonemap: at GPU opacity the sheet piled up to white. Measured (CPU 4,000 vs
GPU 25k): elongation 6.3 vs 4.8, breathing ±30% vs ±59%, coherence
0.29…0.97 vs 0.22…0.95; aimed probes: fountain −0.62, a park clears 476 → 3
fish in 1.5 s.

The same sampling bias showed up here: the v3 scan took the first 16
neighbors in fixed cell order, a periodic force with the cell pitch (±8–11%
density modulation at 2,500, visible as banding in the sheet). The CPU now
does a count pass, a hashed cell walk with population-proportional quotas
out of 96 candidates, and keeps the 16 accepts as a reservoir sample of
every in-range candidate: ±1% (noise), at 6.5 ms per step at 2,500 (was
4.2).

## 4. Measurements

**`feel()`, 2026-10-06 at 25k** (round one):

| | site-default | lifelike-a |
| --- | --- | --- |
| elongation (mean / max) | 1.21 / 1.28 | 1.42 / 1.71 |
| breathing amplitude | ±3 % | ±23 % |
| coherence swing (min…max) | 0.75…0.97 | 0.57…0.99 |
| moods in 45 s | cruise/turn/startle/hover | + contract |

The refill probe parks at the origin, so it is only meaningful while the
school sits there; with roam it mostly measures the school moving away.

**Acceptance, 2026-07-06** (same harness, 120 s mood runs, before any feel
tuning; the fountain row reflects the hunter wedge, since reverted):

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

**Re-measured 2026-10-12** at 25,000 with the shipped `olle-work` feel
after stratified sampling and the re-derived scales (30 s mood run):
cascade .022 / .095 / .209 / .275 then extinct; outburst 0.97; inburst
0.91; fountain −0.27 (bar < −0.2, back from ~0 under the old fear-spread
compensation); vacancy 0 / 67; panic reach 0.39; mean-speed swing 3.1×;
peak hot-glow 0.018; 1.75 ms per step.

**Perf** on Apple silicon (Metal 3, Chrome): 4.1 ms sim / 5.3 ms
sim+render per step at 50,000 — the CPU reference spends 5.3 ms on 2,600.
100,000 runs at 12.5 ms.
