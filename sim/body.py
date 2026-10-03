"""Grid of cells -> mass-spring lattice.

Each cell is 4 corner nodes; neighbours share corners (that's the glue). Corners are
counter-clockwise (y-up), so edge a->b has outward normal (dy, -dx).
"""
import math
from collections import defaultdict

import numpy as np

from .config import CELL, CORE, MUSCLE, EYE, RIGID_TYPES, SPRING_DAMP

NEIGH4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def connected_to_core(cells):
    """Set of cell coords 4-connected to the core at (0,0)."""
    if (0, 0) not in cells:
        return set()
    seen, stack = {(0, 0)}, [(0, 0)]
    while stack:
        i, j = stack.pop()
        for di, dj in NEIGH4:
            n = (i + di, j + dj)
            if n in cells and n not in seen:
                seen.add(n)
                stack.append(n)
    return seen


class Body:
    """Static structure of a creature's body (no world state)."""

    def __init__(self, cells, addr=None):
        # cells: (i, j) -> (type, axis_angle); addr: (i, j) -> wiring address (default: grid position)
        addr = addr or {}
        keep = connected_to_core(cells)
        self.cells = sorted(((c, cells[c][0], cells[c][1]) for c in keep),
                            key=lambda x: (x[0] != (0, 0), x[0]))
        corner_id = {}

        def nid(p):
            if p not in corner_id:
                corner_id[p] = len(corner_id)
            return corner_id[p]

        quads, ctype, cmus = [], [], []
        mus_u, mus_pos, mus_addr = [], [], []
        for (i, j), t, ang in self.cells:
            q = [nid((i, j)), nid((i + 1, j)), nid((i + 1, j + 1)), nid((i, j + 1))]
            quads.append(q)
            ctype.append(t)
            if t == MUSCLE:
                cmus.append(len(mus_u))
                mus_u.append((math.cos(ang), math.sin(ang)))
                mus_pos.append((i, j))
                mus_addr.append(addr.get((i, j), (i, j)))
            else:
                cmus.append(-1)
        self.quads = np.array(quads, dtype=np.int64)
        self.ctype = np.array(ctype, dtype=np.int64)
        self.cmus = np.array(cmus, dtype=np.int64)
        self.core_quad = 0
        # Corner triangles (prev, corner, next), counter-clockwise: positive area while the
        # cell is the right way out. Used by the anti-inversion barrier.
        q = self.quads
        self.tri = np.stack([np.roll(q, 1, 1), q, np.roll(q, -1, 1)], -1).reshape(-1, 3)
        self.n_nodes = len(corner_id)
        self.n_muscles = len(mus_u)
        self.mus_u = np.array(mus_u, dtype=float).reshape(-1, 2)
        self.mus_pos = np.array(mus_pos, dtype=float).reshape(-1, 2)
        self.mus_addr = np.array(mus_addr, dtype=float).reshape(-1, 2)

        # Rest positions, centred on the core cell's centre.
        gp = np.zeros((self.n_nodes, 2))
        for p, k in corner_id.items():
            gp[k] = p
        self.grid_pos = gp - 0.5

        mass = np.zeros(self.n_nodes)
        for q, t in zip(quads, ctype):
            mass[q] += CELL[t]["density"] / 4.0
        self.mass = mass

        # Gather each edge's owners so shared edges follow the glue rules.
        edge_owners = defaultdict(list)       # key -> [(cell_idx, a, b)]
        for ci, q in enumerate(quads):
            for e in range(4):
                a, b = q[e], q[(e + 1) % 4]
                edge_owners[(min(a, b), max(a, b))].append((ci, a, b))

        sa, sb, k, smus, own = [], [], [], [], []

        def add(a, b, kk, m, mine=False):
            sa.append(a); sb.append(b); k.append(kk); smus.append(m); own.append(mine)

        ea, eb = [], []
        exposed_by_cell = defaultdict(list)
        for key, owners in edge_owners.items():
            types = [ctype[ci] for ci, _, _ in owners]
            rigid = any(t in RIGID_TYPES for t in types)
            first_muscle = next((cmus[ci] for ci, _, _ in owners if ctype[ci] == MUSCLE), -1)
            for ci, a, b in owners:
                t = ctype[ci]
                if rigid:
                    m = -1
                elif t == MUSCLE:
                    m = cmus[ci]
                else:
                    m = first_muscle                  # muscle wins over soft neighbours
                add(a, b, CELL[t]["k_edge"], m, t == MUSCLE and m >= 0)
            if len(owners) == 1:
                ci, a, b = owners[0]
                ea.append(a); eb.append(b)
                exposed_by_cell[ci].append((a, b))
        for ci, q in enumerate(quads):
            t = ctype[ci]
            m = cmus[ci] if t == MUSCLE else -1
            add(q[0], q[2], CELL[t]["k_diag"], m, m >= 0)
            add(q[1], q[3], CELL[t]["k_diag"], m, m >= 0)

        self.sa = np.array(sa, dtype=np.int64)
        self.sb = np.array(sb, dtype=np.int64)
        self.k = np.array(k, dtype=float)
        self.c = SPRING_DAMP * np.sqrt(self.k)
        self.smus = np.array(smus, dtype=np.int64)
        self.s_own = np.array(own, dtype=bool)   # a muscle's own springs (for MUSCLE_SLACK)
        self.r0 = self.grid_pos[self.sb] - self.grid_pos[self.sa]
        self.ea = np.array(ea, dtype=np.int64)
        self.eb = np.array(eb, dtype=np.int64)

        eye_cells = {c for (c, t, _) in self.cells if t == EYE}
        idx_of = {c: ci for ci, (c, _, _) in enumerate(self.cells)}
        self.clusters = []
        seen = set()
        for c in sorted(eye_cells):
            if c in seen:
                continue
            comp, stack = [], [c]
            seen.add(c)
            while stack:
                cur = stack.pop()
                comp.append(cur)
                for di, dj in NEIGH4:
                    n = (cur[0] + di, cur[1] + dj)
                    if n in eye_cells and n not in seen:
                        seen.add(n)
                        stack.append(n)
            cis = [idx_of[x] for x in comp]
            nodes = np.unique(self.quads[cis].ravel())
            edges = [e for ci in cis for e in exposed_by_cell[ci]]
            self.clusters.append(dict(
                cells=comp, n=len(comp), nodes=nodes,
                ea=np.array([e[0] for e in edges], dtype=np.int64),
                eb=np.array([e[1] for e in edges], dtype=np.int64),
                pos=np.mean(np.array(comp, dtype=float), axis=0),
                addr=np.mean(np.array([addr.get(x, x) for x in comp], dtype=float), axis=0)))
        self.cluster_pos = (np.array([c["pos"] for c in self.clusters]).reshape(-1, 2))
        self.cluster_addr = (np.array([c["addr"] for c in self.clusters]).reshape(-1, 2))

        self.upkeep = float(sum(CELL[t]["upkeep"] for t in ctype))
        self.build_cost = float(sum(CELL[t]["build"] for t in ctype))
        self.radius = float(np.max(np.linalg.norm(self.grid_pos, axis=1)))

    @property
    def n_cells(self):
        return len(self.cells)

    def place(self, x, y, angle):
        c, s = math.cos(angle), math.sin(angle)
        R = np.array([[c, -s], [s, c]])
        return self.grid_pos @ R.T + np.array([x, y])
