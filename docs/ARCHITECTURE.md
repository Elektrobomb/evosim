# Evosim architecture and design rationale

## 1. Big picture

```
              ┌──────────── server.py (thread) ────────────┐
 browser  ◀──▶│ HTTP: /api/state (poll ~30 fps), /control  │
 index.html   │ Runner: ticks World at chosen speed         │
              └───────────────┬────────────────────────────┘
                              ▼
 World.tick()  (every BRAIN_DT = 0.05 sim-s)
   1. _rebuild()     if population changed: concat all creatures into flat arrays
   2. _sense()       RGB light per eye cluster (numba, sim/vision.py)
   3. brains.step()  every brain in one numba call (BrainBatch)
   4. _physics()     10 substeps in one numba call (sim/physics.py)
   5. _metabolism()  upkeep, eating, predation, death, reproduction
   6. _spawn_food()
```

Genome → phenotype happens once, at birth: `Genome.body.cppn` → `body.cells` (cached) →
`Body` (lattice); `Genome.brain` + `Body` → `Brain`.

## 2. Units and conventions

- 1 length unit = 1 cell; time in sim seconds; energy arbitrary.
- **y-up**; the viewer flips y.
- Cell `(i, j)` spans `[i, i+1] × [j, j+1]`. The **core is cell (0,0)**; cells not
  4-connected to it are dropped.
- The seed eel faces **+x**, so **+y is its left**.
- Cell corners are counter-clockwise, so exposed edge a→b has outward normal `(dy, −dx)`.
  Drag and eye direction rely on this.

## 3. Body physics (`sim/body.py`, `sim/physics.py`)

**Lattice.** Each cell is 4 corner masses with 4 edge and 2 diagonal springs; neighbours
share corners (the glue). Cell mass is split equally over its corners.

**Shared edges.** Each owner adds its own copy of the spring (stiffnesses add). Rest length:
1. any owner bone or core → **rigid**;
2. else any owner muscle → follows that muscle;
3. else a plain spring.

**Muscle.** For activation `a ∈ [0,1]` and unit axis `u`, each of its spring vectors `r0`
rests at `r0 − 0.5·a·(r0·u)u`: the square squashed 50% along `u`, diagonals included.
Activation follows the brain's target with time constant `MUSCLE_TAU` (stability).
`MUSCLE_SLACK` (softer relaxed muscle) is off: it swam worse.

**Integration.** Semi-implicit Euler, dt = 0.005 (10 substeps/tick). Spring force
`k(L−rest) + c·(Δv·û)`, `c = SPRING_DAMP·√k`. Walls clamp and reflect at 20%. Creatures
with non-finite positions are removed (`_kill_unstable`).

**Anti-inversion.** A square has the same spring lengths as its mirror image, so a shove
could flip a cell inside out, where it stuck at zero energy (thousands of times per few
hundred sim-s in mutated bodies). Each corner triangle `(prev, corner, next)` has signed
area +0.5 at rest; below `AREA_MIN` (0.1) a damped force along the area gradient pushes it
back. The seed eel's thinnest corner while swimming is ~0.14, so normal motion never
engages it (the tadpole dips to ~0.09).

**Fluid.** Drag on every exposed edge. Normal (pressure) drag acts only when the face moves
outward, `max(v·n, 0)`, with linear and quadratic terms; tangential drag is weak (0.01 vs
2.0). The asymmetry turns undulation into thrust.

**Collisions.** A uniform grid (cell `COLLIDE_R`) finds close node pairs, pushed apart by a
soft spring every `COLLIDE_EVERY` (2) substeps. Different creatures repel within
`COLLIDE_R` (0.9). Same-creature nodes that don't share a cell (rest distance > 1.5) repel
within `SELF_COLLIDE_R` (0.7), so a tail can't pass through its body; the radius is smaller
because a contracted muscle brings nodes two cells apart to ~1.0.

### What failed during tuning

| Attempt | What happened | Fix |
|---|---|---|
| Cartilage with soft diagonals (a "shear hinge") | One-sided contraction sheared the spine into parallelograms; it never bent | Cartilage isotropically ~4× softer than bone (k 150/150) |
| Normal drag on all faces | Blunt head and tail faces dominated, ~0.2 cells/s | Pressure drag on leading faces only → ~1.6 cells/s |
| Brain I/O binding σ = 1.2 | Each side's muscles picked up the other side's motors and the wave cancelled | σ = 0.7 |
| Alternating bone/cartilage spine with bone tail fin | Near-reciprocal C-flapping, almost no thrust | Rigid bony head, all-cartilage spine, no fin |
| Steering by excite-same + inhibit-other side | ~5°/s, and it killed thrust | Excite same side only: ~20°/s |
| Springs alone holding cell shape | Cells flipped inside out and stuck | Corner-triangle area barrier + same-creature contact |

Speed is modest (~0.2 body lengths/s): a 3-wide body with 50% contraction can only bend so
far. Evolution can find better shapes.

## 4. Vision (`World._sense`, `sim/vision.py`)

- Eye cells form 4-connected **clusters**. Per tick each has a centroid, a facing (unit sum
  of exposed-edge normals; none → sees nothing) and range `EYE_RANGE_PER_CELL × n_cells`.
- **Point sources** (food, and the centre of every cell of *other* creatures) add
  `colour · brightness · cone(θ) · (1 − d/range)`, cone linear in cos θ from 1 on-axis to 0
  at ±65°. An eye never sees its own body.
  - Food is `FOOD_RGB` (green); rich food `RICH_FOOD_RGB` (orange-gold), which the seed
    food sensor reads at exactly 0, so liking it has to evolve. Brightness is
    `min(1, energy / fresh energy of that kind)`, so carrion is dim.
  - Every cell shows the creature's evolved colour `Genome.colour` at `CELL_BRIGHT` (0.5).
    Seeds start red, `COLOUR_INIT`, which the seed food sensor reads at ~−0.49, so hunting
    has to evolve its own tuning. Cyan (the earlier start) read ~0.57 against ~0.99 for a
    pellet, making predation findable from the start; green read more than a pellet.
- **Walls** are sampled by `WALL_RAYS` (7) rays across the cone, each ray's nearness
  `1 − hit/range` triangle-weighted toward the axis, times `WALL_RGB`. Walls glow violet, a
  colour nothing else emits, which the food sensor reads strongly negative, so the seed eel
  steers away. The old dim blue-grey read ≈ 0 and was nearly invisible. (One 600 s run:
  with violet walls the tadpoles died out and eels overshot to 217; single seed, so only
  suggestive.)
- The eye outputs **unsquashed** RGB; the sensor squashes after colour tuning (§5). Squashing
  per channel first made a colour-opponent sensor read a dense patch *weaker* than one
  pellet, because the patch saturates every channel.
- The kernel loops over all cluster–source pairs with a range cut: ~0.5 ms at pop 60 with
  2,000 pellets. For much larger worlds, reuse the collision grid.
- Seed eel, left eye: a pellet 6 cells away reads rgb ≈ (0.26, 0.47, 0.16), food sensor
  ≈ +0.43; a wall 2.5 cells away ≈ (0.18, 0.05, 0.24), sensor ≈ −0.29.

## 5. Brain (`sim/genome.py`, `sim/brain.py`)

**Genome.** NEAT-style graph.
- `NodeGene(id, kind, bias, tau, pos, tune, reach)`, kind ∈ `in | sensor | hidden | mod | motor`.
  - Fixed inputs: 0 energy fraction, 1 osc sin, 2 osc cos, 3 "ate". A meal of energy e sets
    `ate ← 1 − (1 − ate)·exp(−e/FOOD_ENERGY)` (plain ≈ 0.63, rich ≈ 1.0), decaying over
    `ATE_TAU` = 1 s; a one-tick pulse would barely move a neuron.
- `ConnGene(src, dst, w, enabled, innov, eta, hebb)`, keyed by `(src, dst)`. `w` is the
  weight at birth; innovation numbers come from a global `(src, dst)` table.
- `osc_freq` is evolvable.

**Runtime (CTRNN).** Nodes ordered `[fixed inputs, sensors | hidden, mods, motors]`; inputs
clamped. `y += dt/τ · (−y + W·[inputs, tanh(y+b)])`, outputs `tanh(y+b)`. τ ≥ `BRAIN_DT`
keeps Euler stable. Loops allowed. `W` is dense, which is fine at these sizes.

**Plasticity (ABCD rule).** After the new outputs, each link with `eta > 0` updates its live
weight: `dw/dt = η·(A·pre·post + B·pre + C·post + D)`, clipped to ±`HEBB_W_MAX` (8, the
genome's bound). η ∈ [0, `HEBB_ETA_MAX`], A–D ∈ [−1, 1]. D is a constant drift, usable as a
maturation timer. No decay to the birth weight; clipping keeps it bounded.
- Learned weights live only in `Brain.W`, never the genome (no Lamarckism).
- Plastic edges are a sparse list (`p_flat` indexes `W.ravel()`), so non-plastic brains pay
  nothing. 2-D fancy indexing + `np.clip` cost ~3× more than flat `take`/`put`.

**Neuromodulation** (Soltoggio et al. 2008; eligibility after Izhikevich 2007).
- A `mod` node's **outgoing** links are not in `W`. They set each target's
  `m_j = clip(1 + Σ w·out_mod, ±MOD_MAX)`. Unmodulated neurons have m = 1 (plain Hebbian);
  a weak new mod link is near-neutral; m < 0 is punishment.
- A reward arrives after the activity that earned it, so plastic links into a *modulated*
  neuron keep `elig += dt/ELIG_TAU·(hebb − elig)` (2 s) and learn `dw/dt = η·m_j·elig`.
- Modulatory links never learn, and "split connection" skips them.
- Check: a link gated by a mod node fed from "ate" moves +0.002 in 2 s unfed, +0.79 fed.

**Positional I/O binding.** Lets body and brain co-evolve. The body CPPN paints every cell
a **wiring address** (§6a); an eye cluster's or muscle's address is its cells' mean. Each
sensor/motor has an address `pos` and an evolvable **reach** σ (default `BIND_SIGMA` 0.7,
log-normal drift, within [0.3, 4]).
- `bind_s[s, c] = gauss(|pos_s − addr_c|, σ_s)`; input = `tanh(tune · Σ_c bind_s·rgb_c)`.
  `tune` is the sensor's RGB sensitivity, like an opsin. The seed `FOOD_TUNE` is
  (−0.9, 1.8, −0.9): food ≈ 1, bone/cartilage ≈ 0.04, muscle and walls negative.
- `bind_m[m, k] = gauss(|addr_m − pos_k|, σ_k) / (Σ_k + 0.05)`: a normalised blend; the
  +0.05 gives a muscle far from every motor ~0 drive. Motor output → `(tanh + 1)/2`.
- Adding, removing or retyping a cell never invalidates the brain.

Why addresses rather than grid positions: body mutations moved cells out from under fixed
I/O nodes. After 10 unselected mutations, 49% (eel) / 61% (tadpole) of muscles had drive
< 0.3, and unwired muscle still costs upkeep, so selection purged nearly every body change.
Addresses painted by the same CPPN move a region *and* its wiring (a periodic u gives
repeated segments the same motors); that cut the figures to 24% / 41%. Seeds have no links
into u, v, so their addresses equal grid positions. The viewer draws each I/O node at
`brain.at`, the grid centre of what it binds. In 3000 s runs, generation time (~200 s,
mean generation only ~15–17), not wiring, looks like the bottleneck.

## 6. Genome mutation (`Genome.mutate`)

Mutation deep-copies the parent.
1. `mut_rate *= exp(N(0, 0.15))`, clipped to [0.2, 4]; it scales every probability and step.
2. Colour: each channel N(0, `COLOUR_STEP`·rate), clipped to [0, 1].
3. Body CPPN (§6a); `sym_prob` drifts N(0, 0.05).
4. Brain:
   - weights: 80% perturbed (5% full reset); bias, τ (log-normal), address, reach, osc freq drift
   - add connection 12%; split into hidden 4%; remove or toggle 3%
   - add motor near a muscle's address 2%; add sensor near an eye 2% (random N(0, 1) tuning);
     drop hidden/mod 1%
   - add mod node 1%: fed from "ate" half the time, with one weak N(0, 0.3) modulatory link out.
     Later links from a mod node are modulatory too.
   - sensor tuning: each component N(0, 0.15) at 20%, clipped to ±3
   - learning rules: A–D drift N(0, 0.15) at 20% *even while η = 0* (hidden variation). An off
     rule switches on (2%) at `HEBB_ETA_INIT` 0.05, near-neutral; an on rule drifts (30%) and
     is weakened (1%) by halving, zeroed below 0.02. No step is a big jump.
   - a split copies the rule to the new hidden→dst link

Older genome JSON loads with defaults: rules off, sensors `FOOD_TUNE`, node 3 added,
u/v unlinked, an old `hue` converted to RGB. Direct-`cells` genomes are rejected.

## 6a. Body encoding: the CPPN (`sim/cppn.py`)

An acyclic net *paints* the body. Each cell in |i|, |j| ≤ `CPPN_EXTENT` (8) is queried with
`x, y, |y|, d, 1` (x = i·s, y = j·s, s = 0.25, so the box spans ±2). Outputs:
- `exists > CPPN_EXISTS_SLOPE·d` (1.5) → a cell (see *Smoothness*);
- argmax of 4 logits → bone, muscle, cartilage or eye;
- muscle axis `(π/2)·tanh(angle)·sgn(j)`, sgn(0) = +1;
- wiring address `(i + u, j + sgn(j)·v)`.

Hidden nodes use tanh, sin, gauss, abs or linear. `BodyGenome.cells` / `.addr` cache the
result, so everything downstream sees a plain cell dict.

**Symmetry.** The mirror is j → −j. A net that never reads signed `y` is exactly symmetric,
and sgn(j) mirrors angles and addresses to match. New links from `y` are refused with
probability `sym_prob`, so each lineage evolves its appetite for asymmetry.

**Mutation** (probabilities × rate):
- each weight and bias: 30% N(0, 0.3) drift, or 2% N(0, 2) kick (slides a boundary ~1 cell)
- add link 8%, weak N(0, 0.5)
- add pattern node 4%: random activation fed by one input (N(0, 2)), weak link out
- split a link 3% through a **linear** node (no body change; later swaps bend it)
- start activation swap 1.5%: output `(1−mix)·act + mix·act2`, mix starts at 0, drifts, commits at 1
- weaken link 1.5%: halve, or delete if < 0.1 or disabled (no toggle: re-enabling is a jump)
- links only if they keep the net acyclic, counting disabled links

**Smoothness.** Steep tanh steps saturate into plateaus (the old seed's `exists` took 3
values over 289 cells), so threshold moves changed none or 20+ cells. Now: the distance
slope orders a plateau's cells so the edge moves a cell or two at a time; seed outputs of
±4 give margins of ~2; swaps and removals blend.

| Genomes (children, 2026-09-27) | Change | 1–5 cells (of changes) | >15 cells |
|---|---|---|---|
| Seed, old | 14% | 57% | 33% |
| Seed, new | 16% | 73% | 10% |
| Evolved 800 s, old | 30–40% | 48–60% | 20–45% |
| Evolved 800 s, new | 14–30% | 72–85% | 2–4% |

Type logits have no tie-break, so re-typing a region can still be lumpy.

| Attempt | What happened | Fix |
|---|---|---|
| NEAT split with random activation, new links N(0, 1.5), seed outputs ±1 | Half of body changes rewrote 20+ cells; a weight drift could delete a muscle row (margin ~0.45) | Linear splits, separate weak pattern-node op, weak new links |
| Plateau-only fields, instant swap, remove/toggle | Every operator bimodal; swaps/removals rewrote 15+ cells ~30% of the time | Distance slope, seed weights ±4, blended swaps, halve-then-prune |
| Drift only, no kicks | ~5% of children changed shape; growing a tail took ~150 generations | 2% N(0, 2) kick per parameter |

**Seeds** (`presets.SEEDS`). The world starts with mutants of each in turn, each with
2 × `CHILD_ENERGY_FRACTION` × build cost (~44 for the eel); they are also the floor's
`seed_genomes`.
- Written as boxes (`x0..x1`, `|y|` in `a0..a1`, one type) painted by `box_cppn`: each box
  edge is a shared tanh step (position = bias / weight, so sliding a boundary is one
  parameter); one tanh node per box ANDs its steps; outputs read `4·(box+1)` (type margin 8,
  `exists` margin ≥ 2). A long body needs a link from `d` to cancel part of the slope.
  `_boxed` asserts the CPPN paints exactly `box_cells`.
- `wave_brain`: motors at `(x, ±1)`, an oscillator-driven travelling wave, sides in
  antiphase, and **differential steering**: per side, hidden `tanh(8·(own − other) − 3)`
  excites that side's motors, with the motor bias cancelling its rest value (≈ −1).
  - Why: eyes wired straight to motors saturated in a patch (~0.94 each), mean muscle drive
    rose 0.47 → 0.85, the eel co-contracted, slowed to ~0.8 cells/s, and ate half what a
    *blind* eel did. Pellets eaten (lone eel, 120 s at 0.003 / 0.01 / 0.04 per cell², then
    180 s over four distant patches): direct 5.0 / 18.1 / 15.8 / 76, differential
    8.4 / 18.9 / 41.4 / 143, blind 1.5 / 4.6 / 27.4 / 1. Speed and turn rate unchanged.
  - Rejected: compressing eye brightness (worse in sparse food, no help in patches);
    motor mapping `max(0, tanh)` (−4% speed, −25% turn, foraged worse).
- **Eel** (22 cells): core nose, eyes flanking a bone, 6-long cartilage spine flanked by
  muscle. ~1.78 cells/s, ~22°/s. Core-in-front replaced eyes-flanking-the-core: ~10 vs ~9
  pellets/min in a patch, and far more kills.
- **Tadpole** (11 cells, just small enough for the eel to eat): the eel's head, two muscle
  segments beating in unison at 2 Hz, a one-cell cartilage tail tip that doubles its speed
  (~1.14 cells/s). It eats ~5.6× its running cost in a patch (eel ~3.3×). Steering must
  stay below the stroke (steer 4, amp 5) or it circles near food and starves. Rejected: the
  first 6-cell tadpole (~0.4 cells/s), and a 12-cell tipless mini-eel (2.9× its cost).
- Mixed 2000 s runs were winner-take-all (either lineage could win). Eel-only runs stay at
  10–60. A 49-cell pike seed died out within ~100 s: its 14 muscles cost ~1.9× what it
  could forage.

## 7. Ecology (`World._metabolism`, `_spawn_food`)

**Energy.** Upkeep = per-cell costs + `MUSCLE_ACTIVE_COST × Σ activation`. A creature eats
food within `eat_r = sqrt(n_cells)` of its **core centre** (the mouth): 1.0 for a bare
core, ~4.69 for the eel. Bigger mouths for bigger bodies make tiny bodies earn less. Energy caps at `max_energy = 2.5·build + 2·child_energy + 10` (eel ~90), where
`child_energy = CHILD_ENERGY_FRACTION (1.5) × build`.

**Reproduction.** At energy ≥ `build + child_energy + PARENT_RESERVE (0.3)·max_energy`
(eel ~64) the parent pays the child's build cost plus `child_energy`, and the child spawns
nearby at a random rotation. The eel's child gets ~22, about 60 s of swimming.
- Why: with a 0.65 reserve, newborns had ~25 s to find food. In 4000 s runs 93% of deaths
  were starvation, half of them newborns that never ate, the population sat at the floor
  9–28% of the time, and stockpiled food fuelled boom-and-crash cycles.
- No crossover yet.

**Death.** At energy ≤ 0 or age > `AGE_PER_CELL × n_cells` (25 s per cell: eel 550 s, bare
core 25 s). Large bodies die of energy long before age; this stops tiny bodies holding
slots. Corpses drop `CARRION_FRACTION × build` as ~2-energy pellets.

**Predation** (`World._predation`, `physics.core_contacts`). A creature with more than
`PREDATION_SIZE_RATIO` (2) times the prey's cells kills it if a prey node is within
`PREDATION_REACH_FRACTION` (0.4) × `eat_r` of its core centre. A node pressed against the
core sits ~0.9–1.5 from its centre, so the eel (reach 1.88) strikes just before contact,
the tadpole (1.33) needs contact, and a 3-cell body (0.69) can't kill. The prey becomes
carrion at the killer's mouth: the usual carrion plus `KILL_ENERGY_FRACTION` (2.5) of its
stored energy. If several predators qualify, the largest wins; one that dies the same tick
doesn't kill. Early runs: chance collisions only (7 kills in 913 deaths over 2500 s);
there is no hunting behaviour yet.

**Floor.** Below `MIN_POPULATION`, a mutant is injected: of the top eater (70%) or a seed.

**Food.** Pellets spawn at `food_rate`/s up to `FOOD_MAX`, never inside a core's eat
radius. 70% land around 5 hotspots that random-walk and all jump to new spots every
`HOTSPOT_RELOCATE_PERIOD` (300 s), so a lineage can't park on one patch forever.
- **Soft population limit:** `d ln(rate)/dt = −FOOD_RATE_GAIN·(pop − target)/target` toward
  `POP_TARGET` (100), clamped to [2, 15]. The population lags food by about a lifespan, so
  this overshoots by design (one 3000 s run: pop 60–180, rate 3.4–7.4/s). `MAX_POPULATION`
  (300) is only a safety cap. `World.pop_target = None` disables it.
- **Rich food:** 1% of pellets, 80 energy (10×), orange-gold; one every ~25 s at 4/s, ~10%
  of energy input. `food_k` marks kind (0 plain, 1 rich); carrion and viewer food are plain.

## 8. Server and viewer

**`server.py`** (standard library only).
- Listens on `127.0.0.1` and `::1` with HTTP/1.1 keep-alive. Without both, every
  `localhost` poll on Windows stalled 0.3–2 s.
- `Runner` ticks the world in a daemon thread, one lock per tick. Speed is sim-s per real
  s; 0 pauses, −1 is max. Backlog is capped.
- `GET /api/state?epoch=&since=&sel=`, `GET /api/genome?id=`,
  `POST /api/control {cmd: speed|reset|food_rate|pop_target|feed|clone|kill}`
  (`pop_target` 0 disables the controller).

**Snapshots.** Static structure (`q`, `t`, `m`) is sent only for ids > `since`; the client
caches it. Ids increase within an `epoch`; reset bumps it. Positions and activations are
sent every frame.

**`static/index.html`** (one file).
- Redraws every animation frame, blending each creature toward the newest snapshot over
  about one snapshot interval (~50 ms lag).
- Cells batched into one path per colour (muscle activation quantised to 12 shades).
- Panel: stats, controls, history chart, and for the selected creature a body map with
  I/O rings and a live brain graph (mod nodes and links purple, plastic links dashed). Eye
  cones are tinted by what each eye sees; sensor rings by their tuning.
- Stale `selected` detail is discarded to avoid a selection race.

## 9. Performance

Measured 2026-10-01 on a Ryzen 9 5900X (12 cores, two 6-core chiplets), headless:

| population | tick before | tick now | Max speed |
|---|---|---|---|
| 60 (~1,400 nodes)  | 4.4 ms  | 1.15 ms | ~43× real time |
| 150 (~4,500 nodes) | 13.3 ms | 2.65 ms | ~19× real time |

Biggest wins first:
- **Brains** batched: `BrainBatch` packs all brains; `step_all` steps them in one numba call
  (4.6 → ~0.1 ms at pop 150). Each `Brain`'s arrays become views, so the viewer and
  `Brain.step` see live state.
- **Eating**: `physics.food_in_reach` returns only in-reach pairs (2.9 → ~0.1 ms).
- **Physics** parallel by creature. Contact sums per node from a position snapshot (pairs
  computed from both ends), so no two threads write one node. A parallel pass costs
  ~15–100 µs to start, about a substep, so one pass covers a contact period: 5 per tick.
- **Collision grid** stays all −1 between uses (`_bin`/`_unbin`); wiping the ~490×313 grid
  cost more than binning.
- **Vision** parallel over clusters, prep in numba (0.9 → 0.25 ms at pop 150).
- **`_rebuild`** caches per-creature static arrays and shifts indices vectorised (13 → ~5 ms).
- **Threads**: one per physical core (SMT only added sync cost); at most half below ~2,000
  nodes (one chiplet beat two). TBB beat OpenMP, whose spinning starved the Python thread.
- Deterministic per seed: no two threads ever sum into one value.

Left: serial Python (per-creature bookkeeping, `_rebuild`, snapshot + JSON at ~1.6 ms per
poll at pop 60, holding the GIL). Next step: creature scalars (energy, age, ate) as world
arrays.

Numba kernels cache in `sim/__pycache__`; the first start after editing a kernel file
takes a few seconds longer.
