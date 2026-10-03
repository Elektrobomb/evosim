# CLAUDE.md: orientation for AI assistants working on Evosim

James's for-fun Python artificial-life sim: a 2D pond of soft-bodied grid-cell creatures
evolving open-endedly, watched in the browser. Read `docs/ARCHITECTURE.md` before changing
physics, brain or genome code: it records why the parameters are what they are and what failed.

## Commands
- The environment is pixi (`pixi.toml`, locked in `pixi.lock`): Python 3.13, numpy, numba.
  Run everything through it; a bare `python` on this machine may lack numpy.
- Run: `pixi run serve` → http://localhost:8000 (`--seed`, `--pop`, `--seeds eel,tadpole`, `--genome file.json`, `--port`)
- Regression check after any physics, brain or config change: `pixi run check`.
  Expect forward speed ~+1.78 cells/s, turn rate ~+22.6°/s, tadpole ~+1.14 cells/s, no NaNs (~20 s).
- Add dependencies with `pixi add <pkg>` (conda-forge), and mirror them in `requirements.txt`.

## James's original spec (keep faithful to it)
- 2D continuous space. Creatures are described on a grid of cells, but once simulated
  every node moves continuously. Cells in contact are glued.
- **Bone:** inert, rigid filler, low cost.
- **Muscle:** contracts 50% along one axis when signalled, costs extra energy when
  contracting, and is expensive to maintain.
- **Cartilage:** flexible, low cost like bone.
- **Photoreceptor:** senses brightness up to a range X. Touching photoreceptors form one
  sensor whose range grows with the number of cells.
- One **core** cell per creature, which is the brain: a sparse, flexible-architecture
  neural net whose traits vary continuously, so it evolves sensibly together with the body.
- Evolution, not gradient descent. The goal for now is finding food.
- He cares more about flexibility and evolvability than efficiency.

## Map
- `sim/config.py`: all constants. Change behaviour here first.
- `sim/body.py`: cell dict → lattice (nodes, springs, drag edges, eye clusters).
- `sim/brain.py`: CTRNN runtime plus positional sensor/motor binding. `BrainBatch` + `step_all`
  (numba) step every brain in one call; `Brain.step` is the reference version.
- `sim/genome.py`: genomes, brain mutation ops, and `to_dict`/`from_dict`.
- `sim/cppn.py`: the body CPPN: evaluation over the grid (`express`, which returns cells and
  each cell's wiring address) and its mutation ops.
- `sim/ids.py`: global node-id counter and innovation table (brain and CPPN share it).
- `sim/world.py`: flat per-population arrays, metabolism, reproduction, snapshots.
- `sim/vision.py`: numba kernel for the RGB light at every eye (food, other creatures' cells, walls).
- `sim/physics.py`: numba kernel running all substeps of a tick (springs, drag,
  anti-inversion, contact, integration, walls), multi-core per creature; also the eating and
  predation contact queries.
- `sim/presets.py`: the seed creatures (`SEEDS`): the hand-tuned eel (core in front) and the
  tadpole, each painted from boxes by `box_cppn`, with `wave_brain` controllers.
- `server.py`, `static/index.html`: the viewer.
- `tools/check.py`: headless checks.

## Invariants and gotchas
- The core is always cell (0,0); other cells must be 4-connected to it (`Body` drops
  orphans). Space is y-up, and the seed eel faces +x with +y as its left.
- Cell corners are listed counter-clockwise, so an exposed edge a→b has outward normal
  `(dy, −dx)`. Drag and eye direction both depend on this.
- The global arrays are rebuilt whenever the population changes. Before changing
  `self.creatures`, call `_sync_out()` to copy state back into the creatures, then set
  `_dirty = True`. `spawn()` does both.
- Node ids are global. After creating nodes with explicit ids, call `reserve_node_ids`.
  Innovation numbers are global per `(src, dst)` pair (the CPPN uses its own `"cppn"` namespace).
- Brain sensors/motors bind by **wiring address** (CPPN outputs u, v), not grid position.
  Build a `Body` with `Body(g.body.cells, g.body.addr)`. Leaving out `addr` falls back to grid
  positions, which is only right for a CPPN with no links into u, v.
- `BodyGenome.cells` is the CPPN's painted body, cached. Only change a genome through
  `mutate()` (which clears the cache), and keep the CPPN acyclic: `_order()` silently skips
  nodes in a cycle.
- The seeds must stay representable: `_boxed` asserts each seed's CPPN paints exactly its
  `box_cells` layout (the CPPN box is |i|, |j| ≤ 8).
  `Genome.lineage` names the seed a creature descends from (display only). Old genome JSON (direct `cells` encoding) is rejected by `from_dict`.
- `BRAIN_DT` (0.05) is the minimum τ. Don't lower τ without also lowering `BRAIN_DT`.
- Physics is tuned for dt = 0.005. Stiffer springs (k > ~800) or larger dt become
  unstable, and unstable creatures are auto-killed rather than crashing the sim.
- Eyes output unsquashed RGB. Sensors apply `tanh(tune · rgb)`. Don't squash per channel
  in the eye, because it breaks colour-opponent tuning (ARCHITECTURE §4).
- `sim/physics.py` (and `sim/vision.py`) take every constant as an argument. Don't read `config` globals inside
  an `@njit` function: numba bakes globals into its on-disk cache, so later edits to
  `config.py` would be silently ignored.
- Cells can't flip inside out: `AREA_MIN`/`AREA_K` push back any cell corner triangle whose
  area collapses. The seed eel's thinnest corner while swimming is ~0.14, so keep
  `AREA_MIN` below that or the barrier starts to change the tuned swimming. (The tadpole
  dips to ~0.09, so its tuned stroke already leans slightly on the barrier.)
- Viewer speed: the server listens on both `127.0.0.1` and `::1` and uses HTTP/1.1
  keep-alive. Removing either brings back a ~0.3–2 s per-request stall on Windows
  (`localhost` tries `::1` first).
- The population cap is soft: `_spawn_food` runs an integral controller that moves
  `food_rate` toward `POP_TARGET` (ARCHITECTURE §7). A test that needs a fixed food rate
  must also set `w.pop_target = None`.
- `snapshot()` output is the viewer's contract. Keep the keys in sync with `index.html`
  (`n, a, q, t, m, rgb, e`, where `rgb` is the evolved body colour, `selected.brain.{kinds, names, v, e, pos}`, `selected.grid`,
  `selected.eyes`, `selected.cppn`). `brain.e` entries are `[src, dst, live_w, plastic]`,
  `brain.tune` is each sensor's RGB tuning (null for other nodes), and `eyes[].s` is RGB.
  `brain.pos` is an I/O node's wiring address, `brain.reach` its reach, and `brain.at` where it
  acts in grid space (null if it binds nothing). `grid` entries are `[i, j, type, u, v]`.
  Top-level `food` is plain pellets, `rich` is rich ones, and `stats.rich` is their count.
  `stats.kills` and `selected.kills` count predation kills, and `selected.lineage` is its seed.
  `stats.top_speed` (also the 5th `history` column) is the population's mean best speed,
  and `selected.speed` is `{now, avg, top, parent}` in cells/s (display-only tracking).
  `history` rows are `[t, pop, mean_gen, food, top_speed, food_rate]`, and `stats.pop_target`
  is the soft-limit target (null if off).
- Food arrays `food_pos`, `food_e` and `food_k` must be filtered together. Always add food
  through `add_food()`.
- Fixed inputs are 0–3 (energy, osc sin, osc cos, ate). Sensors start at input index
  `len(FIXED_INPUTS)`, not a literal 3. Links out of a `mod` node are modulatory, are kept
  outside `W` (`Brain.md_*`), and never learn.
- Plastic weights live only in `Brain.W`. Never write them back into the genome, or
  learning becomes Lamarckian inheritance.
- After `_rebuild`, each `Brain`'s `W`, `y`, `values`, `elig` and `m` are views into the
  world's `BrainBatch`. Update them in place (`a[:] = ...`), never rebind them, or the brain
  silently stops seeing the batched state. Brain maths changes go in both `Brain.step` and
  `step_all`.
- Parallel kernels (`prange`) must never have two threads write the same element. That
  keeps runs deterministic per seed and race-free. Physics splits by creature (each creature's
  springs, edges and triangles only touch its own nodes), and contact is summed per node.
- The collision grid (`_grid_head`) must be all −1 between uses: pair every `_bin` with an
  `_unbin` on the same positions.
- Background processes: never `pkill -f server.py` from a shell whose own command line
  contains that string, because it kills the shell. Use the PID instead.

## Status (2026-10-03)
Done: physics, RGB vision, CTRNN brain with positional I/O, reproduction, collisions,
viewer, CPPN bodies, Hebbian plasticity + neuromodulation, predation, performance work.
Open issue: sessile 1–3-cell "plants" can take over hotspots in long runs (ARCHITECTURE §7).

## Roadmap (rough order)
1. ~~CPPN body encoding~~ (as `f(x, y, |y|, d, 1)`: bodies face +x, so the mirror axis is y)
2. ~~Evolved Hebbian plasticity and neuromodulation~~
3. **Speciation and crossover** (NEAT). Innovation numbers already exist.
4. **Touch sensing.** Mutual visibility and predation are done.
5. **World save/load and a headless long-run mode.**
6. **Performance:** mostly done (ARCHITECTURE §9). What's left is serial Python.
