# Evosim

A 2D pond of soft-bodied creatures built from grid cells (core, bone, muscle, cartilage,
eye). They swim in continuous space, hunt food and each other, and evolve body and brain
with no fitness function: a creature that banks enough energy buds off a mutated child,
and one that runs out dies.

## Run

```bash
pixi run serve    # http://localhost:8000
pixi run check    # headless regression checks (~20 s)
```

Options for `serve`: `--port`, `--seed` (reproducible world), `--pop` (starting
population), `--seeds eel,tadpole` (which built-in seeds), `--genome saved.json` (start
from a genome exported with the viewer's **Genome JSON** button).

Without pixi: `pip install -r requirements.txt`, then `python server.py`.

**Viewer:** click a creature to follow it and see its body map and live brain. Drag to
pan, scroll to zoom, double-click to drop food, space to pause.

## Overview

- **Bodies:** each cell is a square of 4 point masses joined by springs, and neighbours
  share corners. Muscles contract 50% along an evolved axis. Drag acts on leading faces
  only, so undulation produces thrust. A small evolved net (a CPPN) paints the body.
- **Eyes:** clusters of eye cells see RGB light from food, other creatures and the walls,
  with range proportional to cluster size.
- **Brain:** a sparse NEAT-style CTRNN in the core, with evolved Hebbian plasticity and
  reward neuromodulation. Sensors and motors bind to nearby cells, so the brain survives
  body changes.
- **Ecology:** steady-state reproduction, food hotspots, a soft population target, and
  predation by creatures more than twice the prey's size.

## Docs

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): design, parameters, and what failed.
- [CLAUDE.md](CLAUDE.md): orientation for AI assistants (invariants, status, roadmap).
