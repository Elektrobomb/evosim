"""CPPN body encoding: an acyclic net queried at every cell (i, j), |i|, |j| <= CPPN_EXTENT.

    inputs   x, y, |y|, d, 1       x = i·s, y = j·s, d = √(x² + y²), s = CPPN_COORD_SCALE
    exists   cell present if > CPPN_EXISTS_SLOPE·d
    4 logits argmax picks bone / muscle / cartilage / eye
    angle    muscle axis = (π/2)·tanh(out)·sgn(j), sgn(0) = +1
    u, v     wiring address (i + u, j + sgn(j)·v), which brain I/O nodes bind by

Only cells 4-connected to the core at (0,0) are kept. A net that never reads signed y is
mirror-symmetric (j -> -j). Structural mutations are gradual: activation swaps blend in via
`mix`, and removals halve a link before deleting it.
"""
import math

import numpy as np

from .body import connected_to_core
from .config import (CORE, BONE, MUSCLE, CART, EYE, CPPN_EXTENT, CPPN_COORD_SCALE,
                     CPPN_EXISTS_SLOPE)
from .ids import innovation, new_node_id

IN_X, IN_Y, IN_AY, IN_D, IN_BIAS = range(5)
OUT_EXISTS, OUT_BONE, OUT_MUSCLE, OUT_CART, OUT_EYE, OUT_ANGLE, OUT_U, OUT_V = range(5, 13)
INPUTS = (IN_X, IN_Y, IN_AY, IN_D, IN_BIAS)
OUTPUTS = (OUT_EXISTS, OUT_BONE, OUT_MUSCLE, OUT_CART, OUT_EYE, OUT_ANGLE, OUT_U, OUT_V)
TYPE_OUTPUTS = (OUT_BONE, OUT_MUSCLE, OUT_CART, OUT_EYE)
TYPE_OF_LOGIT = np.array([BONE, MUSCLE, CART, EYE])
NAMES = {IN_X: "x", IN_Y: "y", IN_AY: "|y|", IN_D: "d", IN_BIAS: "1",
         OUT_EXISTS: "exists", OUT_BONE: "bone", OUT_MUSCLE: "muscle", OUT_CART: "cartilage",
         OUT_EYE: "eye", OUT_ANGLE: "angle", OUT_U: "addr u", OUT_V: "addr v"}

ACTS = {
    "tanh": np.tanh,
    "sin": np.sin,
    "gauss": lambda s: np.exp(-s * s),
    "abs": np.abs,
    "linear": lambda s: s,
}
ACT_NAMES = sorted(ACTS)


PRUNE_W = 0.1           # weaker links are deleted rather than halved


class CNode:
    """Output is (1-mix)·act + mix·act2 while a swap is in progress."""
    __slots__ = ("id", "kind", "act", "bias", "act2", "mix")

    def __init__(self, id, kind, act="linear", bias=0.0, act2=None, mix=0.0):
        self.id, self.kind, self.act, self.bias = id, kind, act, float(bias)
        self.act2, self.mix = act2 or act, float(mix)

    def to_dict(self):
        return dict(id=self.id, kind=self.kind, act=self.act, bias=self.bias,
                    act2=self.act2, mix=self.mix)


class CConn:
    __slots__ = ("src", "dst", "w", "enabled", "innov")

    def __init__(self, src, dst, w, enabled=True):
        self.src, self.dst, self.w, self.enabled = src, dst, float(w), enabled
        self.innov = innovation(src, dst, "cppn")

    def to_dict(self):
        return dict(src=self.src, dst=self.dst, w=self.w, enabled=self.enabled, innov=self.innov)


class Cppn:
    def __init__(self):
        self.nodes = {i: CNode(i, "in") for i in INPUTS}
        self.nodes.update({o: CNode(o, "out") for o in OUTPUTS})
        self.conns = {}

    def add_node(self, act, bias=0.0, id=None):
        nid = id or new_node_id()
        self.nodes[nid] = CNode(nid, "hidden", act, bias)
        return nid

    def connect(self, src, dst, w):
        self.conns[(src, dst)] = CConn(src, dst, w)

    def ids(self, *kinds):
        return sorted(n.id for n in self.nodes.values() if n.kind in kinds)

    def to_dict(self):
        return dict(nodes=[n.to_dict() for n in self.nodes.values()],
                    conns=[c.to_dict() for c in self.conns.values()])

    @staticmethod
    def from_dict(d):
        net = Cppn()
        net.nodes = {n["id"]: CNode(n["id"], n["kind"], n["act"], n["bias"], n.get("act2"),
                                    n.get("mix", 0.0)) for n in d["nodes"]}
        for o in OUTPUTS:                  # files saved before wiring addresses lack u, v
            net.nodes.setdefault(o, CNode(o, "out"))
        for c in d["conns"]:
            cg = CConn(c["src"], c["dst"], c["w"], c["enabled"])
            net.conns[(c["src"], c["dst"])] = cg
        return net

    # ------------------------------------------------------------------ evaluation
    def _order(self):
        """Hidden and output nodes in topological order. Nodes in a cycle are skipped."""
        live = [c for c in self.conns.values() if c.enabled]
        indeg = {n: 0 for n in self.nodes}
        out = {n: [] for n in self.nodes}
        for c in live:
            indeg[c.dst] += 1
            out[c.src].append(c.dst)
        ready = sorted(n for n, k in indeg.items() if k == 0)
        order = []
        while ready:
            n = ready.pop()
            order.append(n)
            for m in out[n]:
                indeg[m] -= 1
                if indeg[m] == 0:
                    ready.append(m)
        return [n for n in order if self.nodes[n].kind != "in"]

    def evaluate(self, x, y):
        """node id -> output array, for arrays of scaled coordinates."""
        vals = {IN_X: x, IN_Y: y, IN_AY: np.abs(y), IN_D: np.hypot(x, y),
                IN_BIAS: np.ones_like(x)}
        incoming = {}
        for c in self.conns.values():
            if c.enabled:
                incoming.setdefault(c.dst, []).append(c)
        for n in self._order():
            s = np.full_like(x, self.nodes[n].bias)
            for c in incoming.get(n, ()):
                s = s + c.w * vals[c.src]
            nd = self.nodes[n]
            vals[n] = ACTS[nd.act](s)
            if nd.mix > 0:
                vals[n] = (1 - nd.mix) * vals[n] + nd.mix * ACTS[nd.act2](s)
        return vals

    def express(self, extent=CPPN_EXTENT, scale=CPPN_COORD_SCALE):
        """(cells, addr): (i, j) -> (type, muscle angle) and (i, j) -> wiring address."""
        r = np.arange(-extent, extent + 1)
        ii, jj = (a.ravel() for a in np.meshgrid(r, r, indexing="ij"))
        v = self.evaluate(ii * scale, jj * scale)
        exists = v[OUT_EXISTS] > CPPN_EXISTS_SLOPE * scale * np.hypot(ii, jj)
        types = TYPE_OF_LOGIT[np.argmax(np.stack([v[o] for o in TYPE_OUTPUTS]), axis=0)]
        sgn = np.where(jj < 0, -1.0, 1.0)
        ang = 0.5 * math.pi * np.tanh(v[OUT_ANGLE]) * sgn
        au, av = ii + v[OUT_U], jj + sgn * v[OUT_V]
        cells, addr = {}, {}
        for i, j, t, a, e, u, w in zip(ii, jj, types, ang, exists, au, av):
            if e or (i == 0 and j == 0):
                cells[(int(i), int(j))] = (int(t), float(a) if t == MUSCLE else 0.0)
                addr[(int(i), int(j))] = (float(u), float(w))
        cells[(0, 0)] = (CORE, 0.0)
        keep = sorted(connected_to_core(cells))
        return {c: cells[c] for c in keep}, {c: addr[c] for c in keep}

    # -------------------------------------------------------------------- mutation
    def _reaches(self, a, b):
        """Path a -> ... -> b, counting disabled links."""
        stack, seen = [a], {a}
        while stack:
            n = stack.pop()
            if n == b:
                return True
            for (s, d) in self.conns:
                if s == n and d not in seen:
                    seen.add(d)
                    stack.append(d)
        return False

    def mutate(self, rng, r, sym_prob):
        """Mutate in place. Ops are chosen so most body changes are local (ARCHITECTURE §6a)."""
        def drift(v):
            # rare kicks slide a boundary ~1 cell
            if rng.random() < 0.02 * r:
                v += rng.normal(0, 2.0)
            elif rng.random() < 0.3:
                v += rng.normal(0, 0.3 * r)
            return float(np.clip(v, -30, 30))

        for c in self.conns.values():
            c.w = drift(c.w)
        for n in self.nodes.values():
            if n.kind != "in":
                n.bias = drift(n.bias)
            if n.act2 != n.act:                    # swap in progress
                if rng.random() < 0.02 * r:
                    n.mix += rng.normal(0, 0.5)
                elif rng.random() < 0.3:
                    n.mix += rng.normal(0, 0.15 * r)
                n.mix = float(np.clip(n.mix, 0.0, 1.0))
                if n.mix >= 1.0:
                    n.act, n.mix = n.act2, 0.0

        if rng.random() < 0.08 * r:                                    # add connection
            src = int(rng.choice(self.ids("in", "hidden")))
            dst = int(rng.choice(self.ids("hidden", "out")))
            if (src != dst and (src, dst) not in self.conns and not self._reaches(dst, src)
                    and not self._refuse(src, rng, sym_prob)):
                self.connect(src, dst, float(rng.normal(0, 0.5)))

        if rng.random() < 0.04 * r:                                    # add a pattern node
            src = INPUTS[rng.integers(len(INPUTS))]
            if not self._refuse(src, rng, sym_prob):
                h = self.add_node(ACT_NAMES[rng.integers(len(ACT_NAMES))], float(rng.normal(0, 1)))
                self.connect(src, h, float(rng.normal(0, 2.0)))
                dsts = [n for n in self.ids("hidden", "out") if n != h]
                self.connect(h, int(rng.choice(dsts)), float(rng.normal(0, 0.5)))

        enabled = [c for c in self.conns.values() if c.enabled]
        if rng.random() < 0.03 * r and enabled:                        # split -> linear node
            c = enabled[rng.integers(len(enabled))]
            c.enabled = False
            h = self.add_node("linear")
            self.connect(c.src, h, 1.0)
            self.connect(h, c.dst, c.w)

        hid = self.ids("hidden")
        if rng.random() < 0.015 * r and hid:                           # start an activation swap
            n = self.nodes[int(rng.choice(hid))]
            if n.mix == 0.0:                       # (re)target only while the blend is at 0
                n.act2 = ACT_NAMES[rng.integers(len(ACT_NAMES))]

        if rng.random() < 0.015 * r and self.conns:                    # weaken / prune a link
            keys = sorted(self.conns)
            c = self.conns[keys[rng.integers(len(keys))]]
            if not c.enabled or abs(c.w) < PRUNE_W:
                del self.conns[(c.src, c.dst)]
            else:
                c.w *= 0.5

    @staticmethod
    def _refuse(src, rng, sym_prob):
        """Refuse new links from signed y (they break symmetry) with probability sym_prob."""
        return src == IN_Y and rng.random() < sym_prob
