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
  to its count, from a hash-randomized in-cell offset, and the 16 accepts
  are a reservoir sample of every in-range candidate. The earlier "first
  cell to fill the cap wins" sweep averaged over *cells* rather than *fish*,
  which is a periodic force with the cell period — measured as a ±10%
  density modulation at the perception-cell pitch that showed as sheets
  along the grid planes once crowd dim revealed the core's mid-tones.
  Quotas alone still stopped the walk at the 16th accept, so cells early in
  the walk won whenever more than 16 candidates were in range (94% of fish
  in olle-work's sheet): a uniform ~0.18 u cohesion pull toward −x/−y/−z
  and a ±3–4% low-side slope in the within-cell density. The reservoir
  removed both (within-cell ±1–3%, no slope; +0.11 ms per step at 25k).
  A pointer sweep through the school's center reads the same on both
  kernels, about −0.3 for the behind cohort whichever way it sweeps. Fair sampling must stay *pure*: an own-cell
  quota was tried for close-range locality and any position-quantized bias
  is static in space — it accreted the school into visible sheets within a
  minute. Both calibration scales below were re-derived for it; the
  reservoir moved click-cascade growth by under 3%, so they stand.
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

- **Feels** — every `snapshots/*.json` is a named feel. The panel's *Feel
  A/B* row switches between them in place and shows which is active (the
  school keeps swimming; a feel is a temperament, not a density, so the
  snapshot's `count` is ignored), and the URL updates to `?feel=<name>`
  for sharing. *Save feel…* downloads the current sliders as `<name>.json`; drop
  it into `snapshots/` and it appears on reload. `site-default` is the v3
  canonical state; **`olle-work` is the shipped feel** — every plain visit
  boots into it. Slider tweaks are not saved across reloads; keep one with
  *Save feel…*.
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
its params on the CPU backend: a few thousand fish need their own speed,
size and brightness. It began as the workaround for a CPU path with no
hue-preserving tonemap, where the sheet piled up to white at GPU opacity
(the tonemap came later, below). Measured (CPU 4,000 vs
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

The CPU now steps like the GPU (2026-10-13). Three v3 behaviors went:
neighbors read last frame's position, velocity, energy, fear and pulse
center instead of arrays already updated earlier in the frame (a cascade
or panic could hop several fish per frame along increasing indices, and
two fish swapping indices reversed their drift); a fish lit by contagion
or the predator's wake keeps its radial kick (the update loop's local
velocity used to overwrite it); and contagion is heat-weighted like the
GPU's. `CPU_TRANSMISSION_SCALE` (1.35) was calibrated on a centroid burst:
a cascade grows 3.4× in its first ⅓ s, like the GPU's 3.5×. It still
peaks earlier and dies sooner (~1.75 s vs 2.75 s) because the CPU school
is smaller (RMS 3.9 vs 4.4 u), so the burst seeds twice the share and runs
out of fuel; that is the feel's geometry, not contagion, so match the
timing through the cpu block's school size, never by tuning this scale. Old vs new CPU on
olle-work: elongation 10.7 vs 10.8, breathing ±53% vs ±54%, coherence
0.20…0.97 vs 0.20…0.95; a 3 s stir blows out 1.19× vs 1.27×, implodes to
0.84× vs 0.92× and reforms by 2.8 s vs 3.0 s (the GPU: 1.08×, 0.81×,
4.6 s). Also: the grids sort only occupied cells (grid build 0.47 →
0.05 ms), the fine grid uses the GPU's screen-wide extent, and MSAA is off
(point sprites render pixel-identical without it, at a third of the cost).

**The CPU governs its count and tonemaps like the GPU** (2026-10-13). The
fallback ran at a fixed count on exactly the weakest devices: under
Chrome's 4× CPU throttle, 2,500 fish held ~42 fps without adapting. The
governor (`src/governor.ts`) is now one state machine with a meter per
backend: the GPU's `onSubmittedWorkDone` probe, or on the CPU `update()`
timed inline. The rest is shared: the 5 s warmup, the stall guard, "late"
(over 1.5 display periods and under 45 fps), the step down that must lower
the meter's median by a quarter of the cut, the down/up/down hold, and
the slider as the new target. The CPU sheds from 1,000 fish up, in steps
of −22% / +15% rounded to 100. It counts as bound when `update()` takes
over 0.6 of a 60 Hz period, and has headroom under 0.4. Those shares sit
below the GPU's 0.8 / 0.5 because `update()` is only part of the main
thread's frame, and a +15% step has to land short of bound. Both meters are
now judged against a 60 Hz period on any display. "Late" already meant
under 45 fps on a 120 Hz screen, and against its 8.3 ms period headroom was
out of reach: in headless Chrome at 120 Hz, a school shed under the
throttle held 2,000 at 120 fps for good, and a mocked GPU spike stayed at
15,000 (now back to 25,000). At 60 Hz the GPU governor makes the same
decisions as before; a mocked-clock replay of the old and new code agrees
step for step in five scenarios. `ParticleSystem` allocates the slider's
8,000 slots once. Shrinking puts the tail to sleep: the sim, the grids,
the previous-frame copies, the uploads and the draw range all stop at the
count. Growing wakes dormant slots in a 0.7·R ball at the centroid, with
their previous-frame copies set, as in `GpuSwarm.reviveRange`. So the
count slider no longer respawns the CPU school either, and at
capacity = count the sim is bit-identical to before. Shed fish are
compensated with the GPU's density law, relative to the governed target:
opacity × (target/governed)^0.35, so a feel's `cpu` block stays literal at
its own count. In the replay (vsync-quantized frames) a 4× load sheds
2,500 → 1,200 in 11 s and grows back once it lifts; a busy page, stalls,
or `update()` cost that doesn't scale with the count shed nothing (the
last blocks after one step). One corner is shared with the GPU: under a
30 Hz display cap with a sim over 10 ms, frames are late and the meter
bound, so it sheds (2,200 → 1,300) without gaining a frame.

**The governor sheds on stutter, fast** (2026-10-15). Two misses showed up
in `?diag` on real devices. An iPhone 13 Pro at 25k stuttered visibly at
~50 fps and never shed: "over" grew by late time and on-time frames paid
it back one for one, so every fifth frame doubled (a third of the time
late) never added up to the 2 s trigger. And a 120 Hz M2 Max at 200k ran
107 fps with ~5% of frames at 13-25 ms, which felt like struggling, while
"late" meant under 45 fps there. Now: the frame budget is the display's
period (the 10th percentile of recent frame times, capped at 60 Hz's) but
no shorter than `maxHz` (GPU 120, CPU 60, which keeps the CPU's reachable
headroom from 2026-10-13); a frame is late past 1.5 budgets; and a
trailing 1 s window sheds 20% when over 4% of its frames are late, every
quarter of it has one (a burst under half a second, like one GPU spike
from another app, can't qualify: at 160k one cut a school running a
clean 120 fps), and the meter's median is bound. The window never
reaches back across a step's 250 ms settle or a stall, so a struggling
school steps about once a second, and warmup is 2 s instead of 5. Steps
are a fixed −20%, not proportional to the meter: the GPU meter is
submit-to-done latency, which counts frames queued behind each other
(14 ms at 200k while frames landed every 8.3 ms). Growing back is
unchanged, slow on purpose. Measured on the M2 Max: 200k → 160k 2.5 s
after load, then 120 fps with no late frames. In a mocked-clock replay
the iPhone case goes 25k → 20k → 16k by 5 s; a 30 Hz cap, a GC hitch a
second and jittery rAF timestamps at 25k shed nothing.
The iPhone then showed the other blind spot: at 20k it read 60 fps, 0%
late, GPU 18.9 ms against a 16.7 ms period, and still stuttered. iOS
Safari keeps rAF at 60 while the GPU falls behind, so frames drop at
presentation where no frame timing can see them. So a meter median over
the whole budget (`OVER_SHARE`) now sheds on its own; the Mac agrees from
the other side (200k: 14 ms against 8.3, stuttering; 160k: 6.5, clean).
In the replay the measured iPhone goes 25k → 20k → 16k by 3.3 s, and the
12k phone default is untouched.

The points now accumulate into a half-float target with the same
`SRC_ALPHA, ONE` blend, then resolve through the GPU's `fsTonemap`: the
0.72 knee on the max channel, hue ratios kept, alpha clamped, the same
premultiplied composite onto the page as before. Without
`EXT_color_buffer_float`/`_half_float` the points draw straight to the
canvas, as before. On frozen cascade frames at 2880×1800, near-white
pixels (min channel ≥ 0.9) fell from up to 0.12% of the frame to
≤ 0.005%, and hot pixels keep the flash hue's saturation (0.21 vs 0.06
direct; the flash, #bdf2de, is 0.22). The 8-bit canvas also dropped
sprite tails below one level, which the half-float target keeps. The
resolve costs 0.28 ms per frame at 2880×1800 on M2 Max (timer query:
0.17 → 0.45 ms). Also fixed: the flash uniform went through
`THREE.Color`, whose `set()` linearizes sRGB, so CPU flashes were darker
and more saturated than the GPU's (#bdf2de drawn as 0.51/0.89/0.73
instead of 0.74/0.95/0.87). The palette stops, accent and base were
already raw. olle-work's `cpu` block keeps opacity 0.6. In 12-frame stirs
with two cascades (CPU 2,500 vs GPU 25k), near-white stayed ≤ 0.005% at
0.6 and at 0.5 (before the tonemap: 0.05–0.10%). Hot pixels (max channel
≥ 0.92) ran 0.08–0.14% at 0.6 against the GPU's 0.05%, but in the flash's
own hue. 0.6 reads closest to the GPU's brightness; 0.5 read faint.
The crowd-dim grid now spans the fine grid's screen-wide extent at the
panel's lowest perception (1.2). Fish outside it keep full brightness;
the old 48³ cube clamped them into its edge cells, and a group crossing
x ≈ ±14.4 dimmed to 0.4×.

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
