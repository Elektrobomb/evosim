"""Headless regression checks. Run after touching physics, brain or config.

    pixi run check                   # ~20 s
    pixi run check --eco 600         # longer ecology run (sim seconds)

Baselines (seed eel unless noted):
    forward speed   ~1.78 cells/s (tadpole ~1.14)
    turn rate       ~+22.6 deg/s toward a lit left eye
    vision          food-tuned left sensor: pellet ~+0.43, rich pellet ~0, wall ~-0.29
                    (violet), another eel ~-0.49 (red); its own body is ignored
    plasticity      dw/dt = eta*D gives dw = eta*D*t; genome weight unchanged; JSON round trip
    reward          mod-gated link: ~0 unfed, ~+0.8 fed over 2 s; ate ~0.63 plain, ~1.0 rich;
                    JSON without the "ate" node loads
    predation       eel kills and eats a bare core; an 11-cell block (half its size) survives
    ecology         no NaNs, population stays above MIN_POPULATION
"""
import argparse
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from sim import config as C  # noqa: E402
from sim.genome import Genome, IN_ATE  # noqa: E402
from sim.presets import SEEDS, swimmer  # noqa: E402
from sim.world import World  # noqa: E402


def lone(genome):
    w = World(seed=1)
    w.food_rate = 0
    w.pop_target = None
    w.allow_births = False
    c = w.spawn(genome, 50, 35, 0.0, energy=1e6)
    w.tick()
    return w, c


def heading(w, c):
    tail = np.argmin(c.body.grid_pos[:, 0])
    h = w.core_centres()[0] - w.pos[tail + c.noff]
    return math.atan2(h[1], h[0]), h / np.linalg.norm(h)


def forward_speed(T=200, genome=None):
    w, c = lone(genome or swimmer(steer=0))
    prev, tot = w.core_centres()[0].copy(), 0.0
    for _ in range(T):
        w.tick()
        core = w.core_centres()[0]
        tot += float((core - prev) @ heading(w, c)[1])
        prev = core.copy()
    return tot / (T * C.BRAIN_DT), bool(np.isfinite(w.pos).all())


def turn_rate(T=100):
    """Yaw rate (deg/s) with the left (+y) eye forced to see bright food."""
    w, c = lone(swimmer())
    left = int(np.argmax(c.body.cluster_pos[:, 1]))
    orig = w._sense

    def fake():
        orig()
        w.signals = np.zeros((w.n_clusters, 3))
        w.signals[left] = 3 * np.array(C.FOOD_RGB)      # sensor reads tanh(~3) ~ 1
    w._sense = fake
    prev, tot = heading(w, c)[0], 0.0
    for _ in range(T):
        w.tick()
        h = heading(w, c)[0]
        tot += (h - prev + math.pi) % (2 * math.pi) - math.pi
        prev = h
    return math.degrees(tot) / (T * C.BRAIN_DT)


def vision():
    """{scene: (left eye rgb, left sensor input)} for the seed eel."""
    out = {}
    for scene in ("empty", "food", "rich", "wall", "eel"):
        w = World(seed=1)
        w.food_rate = 0
        w.pop_target = None
        w.allow_births = False
        y = C.WORLD_H - 4 if scene == "wall" else 35
        c = w.spawn(swimmer(), 50, y, 0.0, energy=1e6)
        if scene == "food":
            w.add_food([[50, y + 6]], [C.FOOD_ENERGY])
        if scene == "rich":
            w.add_food([[50, y + 6]], [C.RICH_FOOD_ENERGY], 1)
        if scene == "eel":
            w.spawn(swimmer(), 50, y + 6, 0.0, energy=1e6)
        w._rebuild()
        w._sense()
        left = c.coff + int(np.argmax(c.body.cluster_pos[:, 1]))
        c.brain.step(1.0, 0.0, 0.0, w.signals[c.coff:c.coff + len(c.body.clusters)], C.BRAIN_DT)
        sensor = c.brain.ids.index(10)                  # swimmer()'s left-eye sensor node
        out[scene] = (w.signals[left].round(3), round(float(c.brain.values[sensor]), 3))
    return out


def plasticity(T=20):
    """Constant-drift rule (eta=0.5, D=1) on an oscillator->motor link.
    Returns (live dw, expected dw, genome and JSON ok)."""
    g = swimmer()
    conn = next(c for c in g.brain.conns.values() if c.src == 1 and c.w > 0)
    conn.eta, conn.hebb = 0.5, [0.0, 0.0, 0.0, 1.0]
    w0 = conn.w
    w, c = lone(g)                      # lone() already ran one tick
    for _ in range(T - 1):
        w.tick()
    i = c.brain.ids.index(conn.src), c.brain.ids.index(conn.dst)
    live = c.brain.W[i[1] - c.brain.n_in, i[0]]
    back = Genome.from_dict(g.to_dict()).brain.conns[(conn.src, conn.dst)]
    ok = conn.w == w0 and back.eta == 0.5 and back.hebb == [0.0, 0.0, 0.0, 1.0]
    return live - w0, min(0.5 * T * C.BRAIN_DT, C.HEBB_W_MAX - w0), ok


def reward(T=40):
    """The plasticity() link, gated by a mod node fed from "ate" (m ~0 unfed, ~2 fed).
    Returns (dw unfed, dw fed, ate after plain meal, after rich meal, old JSON loads)."""
    def gated():
        g = swimmer()
        conn = next(c for c in g.brain.conns.values() if c.src == 1 and c.w > 0)
        conn.eta, conn.hebb = 0.5, [0.0, 0.0, 0.0, 1.0]
        m = g.brain.add_node("mod", bias=-3.0, tau=0.1)
        g.brain.connect(IN_ATE, m, 6.0)
        g.brain.connect(m, conn.dst, 1.0)
        return g, conn

    dws = []
    for fed in (False, True):
        g, conn = gated()
        w, c = lone(g)
        for _ in range(T):
            if fed:
                c.ate = 1.0
            w.tick()
        i = c.brain.ids.index(conn.src), c.brain.ids.index(conn.dst)
        dws.append(c.brain.W[i[1] - c.brain.n_in, i[0]] - conn.w)

    ates = []
    for e, k in ((C.FOOD_ENERGY, 0), (C.RICH_FOOD_ENERGY, 1)):
        w, c = lone(swimmer())
        c.energy = 1.0                                  # lone() makes it too full to eat
        w.add_food([w.core_centres()[0]], [e], k)
        w.tick()
        ates.append(c.ate)

    d = gated()[0].to_dict()
    d["brain"]["nodes"] = [n for n in d["brain"]["nodes"] if n["id"] != IN_ATE]
    try:
        w, c = lone(Genome.from_dict(d))
        w.tick()
        old_ok = bool(np.isfinite(c.brain.values).all())
    except Exception as e:  # noqa: BLE001
        print(f"  old JSON: {e!r}")
        old_ok = False
    return dws[0], dws[1], ates[0], ates[1], old_ok


def predation(T=100):
    """A swimming seed eel meets an inert block. Returns {block cells: (killed, eel kills,
    pellets eaten)}."""
    out = {}
    for n in (1, 11):
        w = World(seed=1)
        w.food_rate = 0
        w.pop_target = None
        w.allow_births = False
        eel = w.spawn(swimmer(steer=0), 50, 35, 0.0, energy=20)
        g = swimmer()
        cells = {(0, 0): (C.CORE, 0.0)}
        cells.update({(i, j): (C.BONE, 0.0) for i in range(4) for j in (-1, 0, 1)
                      if (i, j) != (0, 0)})
        g.body._cells = dict(list(cells.items())[:n])           # test-only: bypass the CPPN
        block = w.spawn(g, 55, 35.5, 0.0, energy=10)
        for _ in range(T):
            w.tick()
        out[n] = (block.dead, eel.kills, eel.eaten)
    return out


def ecology(sim_seconds, seeds):
    w = World(seed=3)
    w.populate([SEEDS[s]() for s in seeds], 24)
    w.add_food(w.rng.uniform([0, 0], [w.W, w.H], (120, 2)), np.full(120, C.FOOD_ENERGY))
    t0 = time.time()
    n = int(sim_seconds / C.BRAIN_DT)
    for i in range(n):
        w.tick()
        if i % int(100 / C.BRAIN_DT) == 0 or i == n - 1:
            s = w.stats
            kinds = {}
            for cr in w.creatures:
                kinds[cr.genome.lineage] = kinds.get(cr.genome.lineage, 0) + 1
            mix = " ".join(f"{k}={v}" for k, v in sorted(kinds.items()))
            print(f"  t={w.t:6.0f}s pop={len(w.creatures):3d} ({mix}) food={len(w.food_e):3d} "
                  f"births={s['births']:4d} deaths={s['deaths']:4d} max_gen={s['max_gen']:3d} "
                  f"rich eaten={s['rich_eaten']:3d} kills={s['kills']:3d} "
                  f"wall={time.time() - t0:5.0f}s")
    conns = [cn for cr in w.creatures for cn in cr.genome.brain.conns.values() if cn.enabled]
    etas = [cn.eta for cn in conns if cn.eta > 0]
    n_mod = sum(n.kind == "mod" for cr in w.creatures for n in cr.genome.brain.nodes.values())
    n_ate = sum(cn.src == IN_ATE for cn in conns)
    print(f"  mod nodes: {n_mod}, links from 'ate': {n_ate}")
    print(f"  plastic links: {len(etas)}/{len(conns)}"
          + (f", median eta {np.median(etas):.3f}" if etas else ""))
    return bool(np.isfinite(w.pos).all()), len(w.creatures)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eco", type=float, default=200.0)
    ap.add_argument("--seeds", default=",".join(SEEDS), help="preset seeds for the ecology run")
    a = ap.parse_args()
    v, ok = forward_speed()
    print(f"forward speed: {v:+.2f} cells/s  finite={ok}   {'OK' if v > 0.5 and ok else 'FAIL'}")
    for name, lo in (("tadpole", 0.6),):
        v, ok = forward_speed(genome=SEEDS[name]())
        print(f"forward speed ({name}): {v:+.2f} cells/s  finite={ok}   {'OK' if v > lo and ok else 'FAIL'}")
    t = turn_rate()
    print(f"turn rate (left eye lit): {t:+.1f} deg/s   {'OK' if t > 5 else 'FAIL'}")
    vis = vision()
    for k, (rgb, sen) in vis.items():
        print(f"vision {k:5s}: left eye rgb {rgb}  food sensor {sen:+.3f}")
    rgb = vis["rich"][0]
    good = (abs(vis["empty"][1]) < 0.02 and vis["food"][1] > 0.3 and vis["wall"][1] < -0.1
            and abs(vis["rich"][1]) < 0.05 and rgb[0] > rgb[1] > rgb[2] > 0
            and vis["wall"][0][2] > vis["wall"][0][0] > vis["wall"][0][1] > 0 and vis["eel"][1] < -0.3 and vis["eel"][0][0] > vis["eel"][0][1])
    print(f"vision: {'OK' if good else 'FAIL'}")
    dw, want, ok = plasticity()
    good = ok and abs(dw - want) < 1e-6
    print(f"plasticity: dw {dw:+.3f} (want {want:+.3f}), genome/JSON {'ok' if ok else 'BAD'}   "
          f"{'OK' if good else 'FAIL'}")
    unfed, fed, ate, ate_rich, old_ok = reward()
    good = abs(unfed) < 0.02 and fed > 0.4 and 0.55 < ate < 0.7 and ate_rich > 0.99 and old_ok
    print(f"reward: dw unfed {unfed:+.3f}, fed {fed:+.3f}; ate after a meal {ate:.2f} "
          f"(rich {ate_rich:.2f}); "
          f"old JSON {'ok' if old_ok else 'BAD'}   {'OK' if good else 'FAIL'}")
    pr = predation()
    good = pr[1][0] and pr[1][1] == 1 and pr[1][2] > 0 and not pr[11][0] and pr[11][1] == 0
    print(f"predation: bare core killed={pr[1][0]} (carrion eaten {pr[1][2]}), "
          f"11-cell block killed={pr[11][0]}   {'OK' if good else 'FAIL'}")
    print("ecology:")
    ok, pop = ecology(a.eco, a.seeds.split(","))
    print(f"ecology finite={ok} final pop={pop}   {'OK' if ok and pop >= C.MIN_POPULATION else 'FAIL'}")
