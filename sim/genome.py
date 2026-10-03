"""Genomes and mutation.

Brain: NEAT-style sparse graph. Links out of a "mod" node gate their target's plasticity
instead of feeding its activation. Body: a CPPN (cppn.py); `sym_prob` is how often it
refuses a new link from signed y. The mutation rate is itself mutated.
"""
import colorsys
import copy
import math

import numpy as np

from .config import (MUSCLE, EYE, HEBB_ETA_MAX, HEBB_ETA_INIT, FOOD_TUNE, BIND_SIGMA,
                     REACH_MIN, REACH_MAX, COLOUR_INIT, COLOUR_STEP)
from .cppn import Cppn
from .ids import innovation, new_node_id, reserve_node_ids  # noqa: F401 (re-exported)

IN_ENERGY, IN_OSC_S, IN_OSC_C, IN_ATE = 0, 1, 2, 3
FIXED_INPUTS = (IN_ENERGY, IN_OSC_S, IN_OSC_C, IN_ATE)
FIXED_NAMES = {IN_ENERGY: "energy", IN_OSC_S: "osc sin", IN_OSC_C: "osc cos", IN_ATE: "ate"}
DYN_KINDS = ("hidden", "mod", "motor")         # nodes with CTRNN dynamics


class NodeGene:
    """pos, reach: sensor/motor wiring address and binding width. tune: a sensor's RGB
    sensitivity. Other kinds ignore them."""
    __slots__ = ("id", "kind", "bias", "tau", "pos", "tune", "reach")

    def __init__(self, id, kind, bias=0.0, tau=0.2, pos=(0.0, 0.0), tune=FOOD_TUNE,
                 reach=BIND_SIGMA):
        self.id, self.kind, self.bias, self.tau = id, kind, bias, tau
        self.pos = (float(pos[0]), float(pos[1]))
        self.tune = [float(t) for t in tune]
        self.reach = float(reach)

    def to_dict(self):
        d = dict(id=self.id, kind=self.kind, bias=self.bias, tau=self.tau, pos=list(self.pos))
        if self.kind in ("sensor", "motor"):
            d["reach"] = self.reach
        if self.kind == "sensor":
            d["tune"] = self.tune
        return d


class ConnGene:
    """w is the weight at birth; eta > 0 makes it learn with hebb = (A, B, C, D)."""
    __slots__ = ("src", "dst", "w", "enabled", "innov", "eta", "hebb")

    def __init__(self, src, dst, w, enabled=True, eta=0.0, hebb=(0.0, 0.0, 0.0, 0.0)):
        self.src, self.dst, self.w, self.enabled = src, dst, w, enabled
        self.eta, self.hebb = float(eta), [float(h) for h in hebb]
        self.innov = innovation(src, dst)

    def to_dict(self):
        return dict(src=self.src, dst=self.dst, w=self.w, enabled=self.enabled, innov=self.innov,
                    eta=self.eta, hebb=self.hebb)


class BrainGenome:
    def __init__(self):
        self.nodes = {i: NodeGene(i, "in") for i in FIXED_INPUTS}
        self.conns = {}
        self.osc_freq = 0.8

    def add_node(self, kind, **kw):
        nid = kw.pop("id", None) or new_node_id()
        self.nodes[nid] = NodeGene(nid, kind, **kw)
        return nid

    def connect(self, src, dst, w, **kw):
        self.conns[(src, dst)] = ConnGene(src, dst, w, **kw)

    def ids(self, *kinds):
        return sorted(n.id for n in self.nodes.values() if n.kind in kinds)

    def to_dict(self):
        return dict(osc_freq=self.osc_freq,
                    nodes=[n.to_dict() for n in self.nodes.values()],
                    conns=[c.to_dict() for c in self.conns.values()])


class BodyGenome:
    def __init__(self, cppn=None, sym_prob=0.7):
        self.cppn = cppn or Cppn()
        self.sym_prob = sym_prob
        self._cells = self._addr = None

    @property
    def cells(self):
        """(i, j) -> (type, axis angle). Cached; mutate() clears it."""
        if self._cells is None:
            self._cells, self._addr = self.cppn.express()
        return self._cells

    @property
    def addr(self):
        """(i, j) -> wiring address. Cached with `cells`."""
        if self._cells is None:
            self._cells, self._addr = self.cppn.express()
        return self._addr

    def to_dict(self):
        return dict(sym_prob=self.sym_prob, cppn=self.cppn.to_dict())


class Genome:
    def __init__(self, body, brain, mut_rate=1.0, colour=None, lineage=None):
        self.body, self.brain = body, brain
        self.mut_rate = mut_rate
        self.colour = [float(x) for x in (colour if colour is not None else COLOUR_INIT)]  # RGB, 0-1
        self.lineage = lineage or "custom"      # seed it descends from (display only)

    def to_dict(self):
        return dict(mut_rate=self.mut_rate, colour=self.colour, lineage=self.lineage,
                    body=self.body.to_dict(), brain=self.brain.to_dict())

    @staticmethod
    def from_dict(d):
        """Inverse of to_dict(); also loads older files."""
        d = d.get("genome", d)
        br = BrainGenome()
        br.osc_freq = d["brain"]["osc_freq"]
        br.nodes = {i: NodeGene(i, "in") for i in FIXED_INPUTS}    # older files lack "ate"
        for n in d["brain"]["nodes"]:
            br.nodes[n["id"]] = NodeGene(n["id"], n["kind"], n["bias"], n["tau"], n["pos"],
                                         n.get("tune", FOOD_TUNE),     # pre-colour files: food-tuned
                                         n.get("reach", BIND_SIGMA))
        for c in d["brain"]["conns"]:
            cg = ConnGene(c["src"], c["dst"], c["w"], c["enabled"],     # pre-plasticity files: fixed
                          c.get("eta", 0.0), c.get("hebb", (0.0, 0.0, 0.0, 0.0)))
            br.conns[(c["src"], c["dst"])] = cg
        if "cppn" not in d["body"]:
            raise ValueError("this genome uses the old direct cell encoding; bodies are CPPNs "
                             "now, so it can't be loaded")
        body = BodyGenome(Cppn.from_dict(d["body"]["cppn"]), d["body"]["sym_prob"])
        reserve_node_ids(max(list(br.nodes) + list(body.cppn.nodes)))
        colour = d.get("colour")
        if colour is None:                     # pre-RGB files: an HSL hue at 70% sat, 55% light
            colour = colorsys.hls_to_rgb(d.get("hue", 180.0) / 360.0, 0.55, 0.7)
        return Genome(body, br, d["mut_rate"], colour, d.get("lineage"))

    # ------------------------------------------------------------------ mutation
    def mutate(self, rng):
        g = copy.deepcopy(self)
        g.mut_rate = float(np.clip(g.mut_rate * math.exp(rng.normal(0, 0.15)), 0.2, 4.0))
        r = g.mut_rate
        g.colour = np.clip(np.array(g.colour) + rng.normal(0, COLOUR_STEP * r, 3), 0.0, 1.0).tolist()
        _mutate_body(g.body, rng, r)
        _mutate_brain(g.brain, g.body, rng, r)
        return g


# ---------------------------------------------------------------------- body ops
def _mutate_body(b, rng, r):
    b.sym_prob = float(np.clip(b.sym_prob + rng.normal(0, 0.05), 0.0, 1.0))
    b.cppn.mutate(rng, r, b.sym_prob)
    b._cells = b._addr = None


# --------------------------------------------------------------------- brain ops
def _mutate_brain(br, body, rng, r):
    br.osc_freq = float(np.clip(br.osc_freq * math.exp(rng.normal(0, 0.08 * r)), 0.1, 4.0))
    for c in br.conns.values():
        if rng.random() < 0.8:
            if rng.random() < 0.05:
                c.w = float(rng.normal(0, 1.5))
            else:
                c.w = float(np.clip(c.w + rng.normal(0, 0.25 * r), -8, 8))
        if br.nodes[c.src].kind != "mod":                               # modulatory links don't learn
            _mutate_rule(c, rng, r)
    for n in br.nodes.values():
        if n.kind == "in":
            continue
        if rng.random() < 0.5:
            n.bias = float(np.clip(n.bias + rng.normal(0, 0.15 * r), -4, 4))
        if rng.random() < 0.3:
            n.tau = float(np.clip(n.tau * math.exp(rng.normal(0, 0.2 * r)), 0.05, 5.0))
        if n.kind in ("sensor", "motor") and rng.random() < 0.3:
            n.pos = (n.pos[0] + rng.normal(0, 0.3 * r), n.pos[1] + rng.normal(0, 0.3 * r))
        if n.kind in ("sensor", "motor") and rng.random() < 0.3:
            n.reach = float(np.clip(n.reach * math.exp(rng.normal(0, 0.15 * r)), REACH_MIN, REACH_MAX))
        if n.kind == "sensor":
            for k in range(3):
                if rng.random() < 0.2:
                    n.tune[k] = float(np.clip(n.tune[k] + rng.normal(0, 0.15 * r), -3, 3))

    ins = br.ids("in", "sensor")
    dyn = br.ids(*DYN_KINDS)

    if rng.random() < 0.12 * r and dyn:                                # add connection (from a mod: modulatory)
        src = int(rng.choice(ins + dyn))
        dst = int(rng.choice(dyn))
        if (src, dst) not in br.conns:
            br.connect(src, dst, float(rng.normal(0, 1.0)))

    enabled = [c for c in br.conns.values() if c.enabled and br.nodes[c.src].kind != "mod"]
    if rng.random() < 0.04 * r and enabled:                            # split -> hidden node
        c = enabled[rng.integers(len(enabled))]
        c.enabled = False
        h = br.add_node("hidden", bias=0.0, tau=float(np.exp(rng.uniform(np.log(0.1), np.log(1.0)))))
        br.connect(c.src, h, 1.0)
        br.connect(h, c.dst, c.w, eta=c.eta, hebb=c.hebb)             # the out-link keeps the rule

    if rng.random() < 0.03 * r and br.conns:                           # remove / toggle connection
        keys = sorted(br.conns)
        key = keys[rng.integers(len(keys))]
        if rng.random() < 0.5:
            del br.conns[key]
        else:
            br.conns[key].enabled = not br.conns[key].enabled

    # New I/O nodes go near a cell's wiring address.
    if rng.random() < 0.02 * r:                                        # new motor node near a muscle
        mus = [body.addr[c] for c, (t, _) in body.cells.items() if t == MUSCLE]
        if mus:
            p = mus[rng.integers(len(mus))]
            m = br.add_node("motor", bias=float(rng.normal(0, 0.5)), tau=0.1,
                            pos=(p[0] + rng.normal(0, 0.5), p[1] + rng.normal(0, 0.5)))
            src = int(rng.choice(br.ids("in", "sensor", "hidden")))
            br.connect(src, m, float(rng.normal(0, 2.0)))

    if rng.random() < 0.02 * r:                                        # new sensor node near an eye
        eyes = [body.addr[c] for c, (t, _) in body.cells.items() if t == EYE]
        if eyes:
            p = eyes[rng.integers(len(eyes))]
            s = br.add_node("sensor", pos=(p[0] + rng.normal(0, 0.5), p[1] + rng.normal(0, 0.5)),
                            tune=rng.normal(0, 1.0, 3))
            dyn = br.ids(*DYN_KINDS)
            if dyn:
                br.connect(s, int(rng.choice(dyn)), float(rng.normal(0, 2.0)))

    if rng.random() < 0.01 * r:                                        # new modulatory node
        # fed from "ate" half the time: reward is its main use
        src = IN_ATE if rng.random() < 0.5 else int(rng.choice(br.ids("in", "sensor", "hidden")))
        targets = br.ids("hidden", "motor")
        if targets:
            m = br.add_node("mod", bias=float(rng.normal(0, 0.5)), tau=0.2)
            br.connect(src, m, float(rng.normal(0, 2.0)))
            # weak out-link keeps the target's m near 1 (neutral)
            br.connect(m, int(rng.choice(targets)), float(rng.normal(0, 0.3)))

    if rng.random() < 0.01 * r:                                        # drop a hidden or mod node
        hid = br.ids("hidden", "mod")
        if hid:
            h = int(rng.choice(hid))
            del br.nodes[h]
            for key in [k for k in br.conns if h in k]:
                del br.conns[key]


def _mutate_rule(c, rng, r):
    """Coefficients drift even while eta = 0 (hidden variation). Switch-on starts slow;
    switch-off halves eta until it's tiny. No step is a big jump."""
    for k in range(4):
        if rng.random() < 0.2:
            c.hebb[k] = float(np.clip(c.hebb[k] + rng.normal(0, 0.15 * r), -1, 1))
    if c.eta == 0.0:
        if rng.random() < 0.02 * r:                                     # switch plasticity on
            c.eta = HEBB_ETA_INIT
    else:
        if rng.random() < 0.3:
            c.eta = float(min(HEBB_ETA_MAX, c.eta * math.exp(rng.normal(0, 0.3 * r))))
        if rng.random() < 0.01 * r:                                     # weaken, then switch off
            c.eta = 0.0 if c.eta < 0.02 else c.eta / 2
