# Handoff: 50k-particle GPU-compute fork

This document is the behavior contract for a GPU rewrite. It deliberately says
nothing about *how* to build the GPU version — it captures what the current
CPU version (v3, ~2,600 particles) has learned, what must survive the port,
what breaks at 20× density, and what to change here before forking.

Ground truth for every number in this doc: `src/params.ts` (params, palettes),
`src/presets.ts` (the tuned preset matrix), `src/ParticleSystem.ts` (inline
constants). Live reference: <https://bioluminescent.vercel.app>.

---

## 1. What this thing is (protect this)

A **living school**, not a particle cloud. The identity, in priority order:

1. **It does things on its own.** The school travels, turns as one, mills,
   rests, and startles at nothing. If a viewer watches for 60 seconds and
   nothing *happens*, the port has failed — regardless of particle count.
2. **The cursor is a predator, not a magnet.** Contact causes lasting fear;
   panic spreads; fleeing fish part around the threat's line of travel and
   drop behind it (fountain maneuver); the moment the threat passes, fear
   flips into an urgent regroup. React hard, return hard.
3. **Light is bound to motion.** Nothing glows for decorative reasons. Flash
   cascades shove; agitation luminesces; speed streaks. If a maneuver
   happens, it paints itself in light — and vice versa.
4. **Taste law (owner's explicit preferences, learned the hard way):**
   - Color changes stay **inside the palette's hue. Never white.** Every
     palette has a saturated `flash` color for this purpose.
   - Movement first, glow second. A cascade you can only *see* (not feel in
     the motion) is wrong.
   - Green-dominant (emerald default); cyan is an accent/alternative.
   - Bias fast. When a tuning choice is ambiguous, pick the livelier one.

## 2. The four behavior layers (the contract)

Keep the *dynamics*, not the code. Everything below is stated scale-free
where possible (ratios of `perception`, `maxSpeed`, `containRadius`).

### 2.1 Flocking substrate
Reynolds boids over a uniform spatial grid; neighbor radius `perception`
(default 2.3 u), hard cap of 16 accepted neighbors per particle.
- Separation radius = **0.66 × perception**, weight falls off as (1−d/r)².
- Cohesion uses **arrival damping**: desired speed scales with
  min(1, dist/1.2·perception). *Load-bearing — see §4.*
- Alignment matches neighbors' **actual average speed** (×1.15, capped at
  maxSpeed), never normalized to full speed. *Load-bearing — see §4.*
- All steering clamped to MAX_FORCE = 14 u/s²; drag `exp(−0.55·dt)`;
  speed cap = maxSpeed × (1 + 1.1·energy + 1.2·fear).

### 2.2 The swarm director (global, tiny, stays host-side by nature)
A whole-school mood FSM: **cruise → turn → mill → hover**, plus **startle**
interrupts. It owns a shared heading and blends five global gains (alignment,
cohesion, mill, drive, target speed) toward per-mood targets — the per-particle
sim just multiplies its weights by these. Two details that matter:
- **Turns are seeded through a vanguard**: one spatial cluster receives the
  new heading first and darts; alignment propagates the turn as a visible
  wave. A turn where everyone rotates simultaneously looks robotic.
- **Startle** = a false alarm at a random point: local fear + light + a
  moment of scatter, then the director tightens the school. This is what
  makes it feel like an animal rather than a screensaver.
- The school follows a **roaming home attractor** (slow sum-of-sines path,
  radius ≈ containRadius·0.75·roam/1.5) and is recentered by steering the
  *centroid* toward it — never by per-particle springs (§4).

### 2.3 Individuals
- **Burst-and-glide gait**: per-fish clocks fire thrust impulses along the
  current heading, then coast. Staggered phases; this is most of the visual
  texture inside a calm school.
- **Personalities**: per-fish speed factor and sociability (static randoms)
  scale their max speed and cohesion. Cheap, high payoff.
- **Wanderer lifecycle**: FLOCK → WANDER (small group recruited around a
  leader, shared heading ± jitter, ~6–13 s) → RETURN (home toward centroid)
  → FLOCK with a sub-threshold rejoining glow. Frightened wanderers abort
  and come home.

### 2.4 Light = motion
- **Excitable flash medium** (dinoflagellate model): ignite → energy 1,
  contagious while energy > 0.55 → decays at `flashDecay` (default 2.4/s,
  half-life ≈ 0.29 s) → below 0.35 enters refractory (default 2.2 s).
  Contagion probability per second = transmission × Σ(1−d/perception) over
  lit neighbors.
- **Cascades are bubbles**: every ignition inherits the cascade's shared
  epicenter from its strongest lit neighbor and both kicks *and keeps
  pushing* radially from it — sign +1 outward (burst), −1 inward (collapse).
  Inbursts relax separation ×(1−0.6·energy) while lit so the crowd truly
  condenses, then pops when separation snaps back. Radial gain 2.4 out /
  3.4 in (collapse must beat crowd pressure). Sources: spontaneous flashes
  (own position, inburst chance = `inburst`), cursor wake (epicenter =
  pointer, always outward), click burst (small seed, radius ≈ 0.5×mouseRadius
  — let transmission do the spreading), scatter, startle.
- **Shear glow**: a second, independent glow channel driven by acceleration
  magnitude with a **soft knee**: target = clamp((|a| − 22) × 0.04), smoothed.
  The knee is why steady cruising stays dark; without it the school is a
  white smear (this bug shipped once — see §4).
- **Velocity stretch**: sprites elongate along screen-space velocity
  (`stretch` param). Fast = glinting streak, still = soft mote. This single
  effect carries most of the "school of fish" read.

### 2.5 The predator
- Pointer unprojected to the z=0 swarm plane; influence uses a
  **depth-squashed metric** (dz × 0.45) so it clears a visible column, not an
  invisible sphere.
- Inside `mouseRadius`: smoothstep-falloff escape steering (`repelForce`),
  fear = max(fear, falloff). Cursor *speed* multiplies force and ignites
  outward cascades (`splash`).
- Fear: decays `exp(−fearDecay·dt)`; spreads toward 0.92 × max neighbor fear
  at rate min(1, fearSpread·dt); boosts alignment (+0.8·fear), cuts cohesion
  while the threat is near, raises speed cap.
- **Fountain maneuver**: fleeing fish part perpendicular to the predator's
  travel direction (pointer velocity is tracked for this) and fall in behind.
- **Regroup**: once the threat is out of range, remaining fear converts to
  cohesion surge (×(1+√fear·regroup·0.8)) plus direct homing to the centroid
  (∝ regroup·√fear). This is the "attacked shoal snaps back together" beat —
  the owner asked for it explicitly; don't lose it.

## 3. State inventory

What actually exists per particle (all Float32 unless noted). This is the
list that must become GPU-resident; everything global in §2.2 is a handful of
scalars per frame.

| Field | Written by | Notes |
| --- | --- | --- |
| pos ×3, vel ×3 | integrator | vel doubles as the stretch attribute |
| energy | flash medium | also a render attribute |
| glow | shear luminescence | render attribute, smoothed |
| refract, fired (u8) | flash bookkeeping | refractory timing |
| pulse ×3, pulseSign | ignition | shared cascade epicenter |
| fear | predator/panic | reads neighbors' fear (gather) |
| mode (u8), modeTimer, wanderDir ×3 | wanderer FSM | recruitment is a host-driven scatter event |
| speedFactor, social, gaitClock, seeds | init / gait | personalities; mostly static |

Per-frame global inputs: mood gains (5), shared heading (3), roam home (3),
pointer pos/vel/active, splash level, centroid (a reduction over positions —
the only global *output* the sim needs back each frame).

Frame anatomy (order matters): build grid → global updates (roam, director,
spontaneous ignitions, wander scheduling) → per-particle neighbor gather
(boids sums + contagion + strongest-lit-neighbor + max fear) → fear update →
steering (boids × director gains, flow/swirl/mill, gait, mode forces,
predator, radial pulse) → integrate + speed clamp → shear glow → flash
bookkeeping → centroid reduction → upload pos/vel/energy/glow.

## 4. Hard-won invariants (each of these was a shipped bug)

1. **Cohesion without arrival damping ⇒ collapse.** Normalizing "toward local
   center of mass" to full speed makes the swarm implode into a white ball
   within seconds. Damp desired speed near the target.
2. **Alignment normalized to maxSpeed ⇒ perpetual racing.** Match neighbors'
   actual pace; the school must be able to idle.
3. **Refractory must exceed the cascade's crossing time** of the densest
   cluster, or waves re-enter and the swarm flickers forever. At default
   tuning: crossing ≈ 1 s, refractory 2.2 s. Re-derive this at 50k — density
   changes wave speed.
4. **Recenter the centroid, not the particles.** Per-particle springs to the
   origin compress the school; steering the group's mean preserves internal
   structure. Same principle for the roam attractor.
5. **Additive blending saturates to white.** The taste law ("never white")
   is enforced by *headroom*, not just hue: flash mix ≤ 0.45 toward the
   palette flash color, modest brightness/alpha boosts. Density spikes
   (inburst condensation!) are where it blows out.
6. **Shear glow needs the knee** (|a| − 22, gain 0.04). Baseline forces —
   cruising drive, containment, drag — must sit below it, or everything glows.
7. **Depth-squash the pointer metric**, or interaction only touches the thin
   z≈0 slice and feels weak on screen.
8. **Spontaneous ignition rate is swarm-wide** (flashes/sec), not
   per-particle. Keep it that way or 50k particles = constant lightning.

## 5. What changes at 20× density (50,000 particles)

Things that will NOT port as-is, by design:

- **Brightness budget.** Perceived brightness ≈ count × size² × opacity under
  additive blending. At 50k, per-particle opacity and size must drop roughly
  an order of magnitude, and flash/glow gains with them. Retune visually
  against the never-white law; the current values are only valid at ~2.6k.
- **Contagion normalization.** Contagion is a *sum* over lit neighbors; more
  neighbors ⇒ faster, hotter cascades at the same `transmission`. Either
  normalize by the neighbor cap or retune transmission down.
- **Neighbor cap semantics.** The 16-neighbor cap currently biases sampling
  by grid-cell order; at 50k a biased cap becomes visible as anisotropy.
  Sample fairly (any unbiased subset is fine).
- **Absolute counts → fractions.** Wanderer group size (12), vanguard size,
  burst seed counts are absolute today; express as fractions of count.
- **Kick vs crowd pressure.** Separation pressure rises with density, so
  pulse kicks/radial gains that read at 2.6k may vanish at 50k. The
  inburst/outburst *contrast* is what matters; retune to preserve it.
- **World scale decision.** Either a bigger tank (keep density, more
  presence) or same tank (denser, more liquid). The behavior constants are
  ratios of `perception`, so they survive a world-rescale; the *aesthetic*
  constants (sizes, opacity) do not. Current frame: containRadius 9,
  camera z = 26, fov 55° ⇒ ≈ 27 world units visible height.
- **The mouse gets relatively smaller.** At 50k in a bigger tank,
  `mouseRadius` and fear parameters need to scale with the school's size or
  the predator stops mattering.

## 6. Verification playbook (port this, it's how "alive" was measured)

**Implemented:** `src/harness.ts` runs all of the probes below headlessly and
returns a typed report; in the playground call `window.harness()` (or import
`runHarness` — no renderer or DOM needed). Run it against the fork with the
same options and compare reports. For ad-hoc poking, `window.swarm` and
`window.__render()` are also exposed. The signatures that define "correct":

- **Cascade growth curve**: seed a click burst, count energy > 0.3 at 0 / 0.33
  / 0.66 / 1 s. Healthy: grows ~25 → ~100 → ~500, then decays to zero. Zero
  growth = transmission broken; no decay = refractory broken.
- **Bubble coherence**: among lit particles, sign of dot(vel, pos − pulse
  epicenter). Outbursts mostly positive, inbursts mostly negative at ~0.3 s.
- **Sweep panic**: a predator pass through the school frightens ≥ 10% of it
  at peak (robust). The fountain maneuver itself — parting around the travel
  line — is a real force in the sim but resists a stable scalar signature
  (rear-drift and perp/par ratios both drown in the director's collective
  heading; historic "−0.31..−0.49" readings were mood-luck). Judge the
  parting shape visually.
- **Predator vacancy** (advisory — mood-dependent): park the pointer with
  verified contact for 1.5 s → typically a handful of stragglers inside the
  radius and peak panic ≳ 15%. But in fast-cruise moods the school
  legitimately *flows through* a parked pointer, and past ~1.5 s a thin rim
  shell re-forms under crowd pressure (~2% at 6 s here; more at density).
  Treat as advisory, or gate it on hover/mill moods.
- **Mood log**: run 2 minutes, log mood transitions and mean speed. Expect
  every mood to occur and mean speed to swing ≥ 2× between hover and turn.
- **Never-white check**: during a full-school turn, count pixels/particles at
  glow saturation — should be a handful, not hundreds.
- **Perf**: current CPU cost ~5 ms/frame at 2,600 (Apple silicon). The fork's
  target at 50k should be measured the same way (sim step wall time).

**The probes are stochastic — re-measure, don't trust these notes.** Run the
harness against this reference several times and use *those* numbers as the
baseline; single-run values in this doc have real variance. Known findings
from the 2026 GPU fork doing exactly that: this reference itself occasionally
fails the cascade-extinction probe (kicked "spark carriers" fly still-hot
into fresh fuel a perception-radius away — rare at 2.6k, chronic at 50k; the
fork's fix was weighting contagion by the source's remaining heat), and the
parked-pointer rim shell above. Density scales both.

## 7. Pre-fork prep — DONE in this repo

All four prep items are implemented here; the fork inherits them:

1. **Canonical snapshot** — `snapshots/site-default.json` (version 3, the
   default Lagoon + Ripple + emerald/glow tuning at 2,600). Re-export via the
   panel's ⧉ Copy JSON whenever the owner settles on a new favorite; that
   file plus `presets.ts` is the entire tuned state a fork must reproduce.
2. **Snapshot versioning** — `Snapshot.version` / `SNAPSHOT_VERSION` in
   `ParticleSystem.ts`; `applySnapshot` validates key-by-key so foreign or
   stale snapshots degrade gracefully.
3. **`TUNING` table** — `src/params.ts` exports every load-bearing non-slider
   constant (grid dim, neighbor cap, max force, separation/arrive ratios,
   contagious window, glow knee/gain, radial gains, fear ceiling, speed
   boosts, inburst relax, burst seed ratio), each with a comment saying what
   breaks. Diff the fork against this table, not against code.
4. **Metrics harness** — `src/harness.ts` (§6). Same probes, same report
   type, runnable in both versions.

Still optional: a **fixed-timestep accumulator** (sim at fixed 1/60, render
interpolates) for tighter side-by-side comparison. Skip if it fights the
site integration.

Explicitly **not** worth doing here: refactoring the sim loop into
GPU-shaped passes, making the CPU version count-scalable, or adding tests
beyond the harness — the fork renumbers everything anyway, and this version's
job is to stay the tuned, working reference.

---

*Repo: github.com/ollealm/bioluminescent-swarm · v3, ~2,600 particles CPU ·
history of why each system exists is in the commit messages.*
