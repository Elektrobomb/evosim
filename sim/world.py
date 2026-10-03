"""The world: flat per-population arrays (rebuilt when the population changes), metabolism,
food, reproduction and snapshots."""
import math
from collections import deque

import numba
import numpy as np

from . import config as C
from . import physics
from . import vision
from .body import Body
from .brain import Brain, BrainBatch


def cell_colours(ctype, colour):
    """RGB of each cell as other eyes see it."""
    return np.tile(np.array(colour, dtype=float), (len(ctype), 1))


def eat_radius(n_cells):
    return C.EAT_RADIUS_SCALE * math.sqrt(n_cells)


class Creature:
    def __init__(self, cid, genome, body, brain, pos, energy, gen, parent):
        self.id, self.genome, self.body, self.brain = cid, genome, body, brain
        self.pos = pos
        self.vel = np.zeros_like(pos)
        self.act = np.zeros(body.n_muscles)
        self.energy = energy
        self.gen, self.parent = gen, parent
        self.age = 0.0
        self.children = 0
        self.eaten = 0
        self.kills = 0
        self.killed_by = None     # predator id
        self.ate = 0.0            # brain input, see config.ATE_TAU
        self.dead = False
        self.track = deque(maxlen=C.SPEED_WINDOW + 1)   # core centre every SPEED_SAMPLE s
        self.odometer = 0.0
        self.legs = 0                                   # samples summed into odometer
        self.speed = self.top_speed = 0.0
        self.child_energy = C.CHILD_ENERGY_FRACTION * body.build_cost
        self.max_energy = 2.5 * body.build_cost + 2 * self.child_energy + 10.0
        self.max_age = C.AGE_PER_CELL * body.n_cells
        self.eat_r = eat_radius(body.n_cells)
        self.noff = self.moff = self.coff = 0
        # Creature-local arrays for _rebuild, beyond the Body's own.
        cl = body.clusters
        ints = lambda xs: np.concatenate(xs).astype(np.int64) if xs else np.zeros(0, dtype=np.int64)
        self.static = dict(
            cln=ints([c["nodes"] for c in cl]),
            clni=np.repeat(np.arange(len(cl)), [len(c["nodes"]) for c in cl]),
            clea=ints([c["ea"] for c in cl]), cleb=ints([c["eb"] for c in cl]),
            clei=np.repeat(np.arange(len(cl)), [len(c["ea"]) for c in cl]),
            cl_cells=np.array([c["n"] for c in cl], dtype=float),
            ccol=cell_colours(body.ctype, genome.colour))


class World:
    def __init__(self, seed=None, w=C.WORLD_W, h=C.WORLD_H):
        self.rng = np.random.default_rng(seed)
        self.W, self.H = w, h
        self.t = 0.0
        self.creatures = []
        self.next_id = 1
        self.food_pos = np.zeros((0, 2))
        self.food_e = np.zeros(0)
        self.food_k = np.zeros(0, dtype=np.int64)       # 0 plain, 1 rich
        self._food_rgb = np.array([C.FOOD_RGB, C.RICH_FOOD_RGB])
        self._food_base = np.array([C.FOOD_ENERGY, C.RICH_FOOD_ENERGY])
        self.hotspots = self.rng.uniform([10, 10], [w - 10, h - 10], (C.N_HOTSPOTS, 2))
        self._hotspot_clock = 0.0
        self.food_rate = C.FOOD_RATE
        self.pop_target = C.POP_TARGET          # None: food-rate controller off
        self._grid_head = np.full(physics.grid_shape(w, h, C.COLLIDE_R), -1, dtype=np.int64)
        self.food_max = C.FOOD_MAX
        self.seed_genomes = []
        self.stats = dict(births=0, deaths=0, max_gen=0, rich_eaten=0, kills=0)
        self.history = deque(maxlen=900)
        self._top_speed = 0.0
        self._food_acc = 0.0
        self.allow_births = True
        self._dirty = True
        self._n = 0
        self.signals = np.zeros((0, 3))
        self.cl_cent = np.zeros((0, 2))
        self.cl_dir = np.zeros((0, 2))
        self.cl_range = np.zeros(0)

    # ---------------------------------------------------------------- population
    def spawn(self, genome, x=None, y=None, angle=None, energy=None, gen=0, parent=None,
              min_dist=0.0):
        """Add a creature min_dist from (x, y) in a random direction, or anywhere."""
        body = Body(genome.body.cells, genome.body.addr)
        brain = Brain(genome.brain, body, self.rng)
        if energy is None:
            energy = C.CHILD_ENERGY_FRACTION * body.build_cost
        self._sync_out()
        pos = self._find_spot(body, x, y, angle, min_dist)
        self._clear_food(pos[body.quads[body.core_quad]].mean(0),
                         max(body.radius, eat_radius(body.n_cells)) + C.SPAWN_FOOD_CLEAR)
        c = Creature(self.next_id, genome, body, brain, pos, energy, gen, parent)
        self.next_id += 1
        self.creatures.append(c)
        self._dirty = True
        self.stats["max_gen"] = max(self.stats["max_gen"], gen)
        return c

    def populate(self, genomes, n):
        """n mutants of `genomes` in turn, each with 2 × CHILD_ENERGY_FRACTION × build cost.
        The genomes become the floor's seed_genomes."""
        self.seed_genomes = list(genomes)
        for i in range(n):
            g = self.seed_genomes[i % len(self.seed_genomes)].mutate(self.rng)
            self.spawn(g, energy=2 * C.CHILD_ENERGY_FRACTION * Body(g.body.cells).build_cost)

    def _find_spot(self, body, x, y, angle, min_dist):
        """Node positions keeping SPAWN_CLEARANCE from other creatures, else the least
        crowded try. Needs _sync_out first."""
        r = body.radius + 1
        others = [c.pos for c in self.creatures if not c.dead]
        others = np.concatenate(others) if others else np.zeros((0, 2))
        best, best_gap = None, -1.0
        for i in range(C.SPAWN_TRIES):
            if x is None:
                cx, cy = self.rng.uniform(r, self.W - r), self.rng.uniform(r, self.H - r)
            else:
                d = min_dist + i * C.SPAWN_STEP
                a = self.rng.uniform(0, 2 * math.pi)
                cx, cy = x + d * math.cos(a), y + d * math.sin(a)
            cx, cy = float(np.clip(cx, r, self.W - r)), float(np.clip(cy, r, self.H - r))
            ang = self.rng.uniform(0, 2 * math.pi) if angle is None else angle
            pos = body.place(cx, cy, ang)
            if len(others) == 0:
                return pos
            near = others[(np.abs(others[:, 0] - cx) < r + C.SPAWN_CLEARANCE)
                          & (np.abs(others[:, 1] - cy) < r + C.SPAWN_CLEARANCE)]
            if len(near) == 0:
                return pos
            gap = float(np.sqrt(((pos[:, None, :] - near[None, :, :]) ** 2).sum(-1).min()))
            if gap >= C.SPAWN_CLEARANCE:
                return pos
            if gap > best_gap:
                best, best_gap = pos, gap
        return best

    def _clear_food(self, centre, r):
        self.food_pos = self._push_out(self.food_pos, np.reshape(centre, (1, 2)), r)

    def _push_out(self, pts, centres, r, passes=3):
        """Move points within r of a centre radially out to r (r scalar or per centre).
        Repeats, since a push can land in another centre's range."""
        pts = pts.copy()
        if len(pts) == 0 or len(centres) == 0:
            return pts
        r = np.broadcast_to(np.asarray(r, dtype=float), (len(centres),))
        for _ in range(passes):
            d = pts[:, None, :] - centres[None, :, :]
            dist = np.sqrt((d ** 2).sum(-1))
            j = (dist - r).argmin(1)
            idx = np.nonzero(dist[np.arange(len(pts)), j] < r[j])[0]
            if len(idx) == 0:
                break
            jj = j[idx]
            dv, dn = d[idx, jj], dist[idx, jj][:, None]
            a = self.rng.uniform(0, 2 * math.pi, len(idx))
            u = np.where(dn > 1e-6, dv / np.maximum(dn, 1e-6), np.stack([np.cos(a), np.sin(a)], 1))
            pts[idx] = np.clip(centres[jj] + u * r[jj, None], [1, 1], [self.W - 1, self.H - 1])
        return pts

    def _sync_out(self):
        """Copy global state back into creatures. Call before changing self.creatures."""
        if self._dirty or self._n == 0:
            return
        for c in self.creatures:
            n = c.body.n_nodes
            c.pos = self.pos[c.noff:c.noff + n].copy()
            c.vel = self.vel[c.noff:c.noff + n].copy()
            c.act = self.act[c.moff:c.moff + c.body.n_muscles].copy()

    def _rebuild(self):
        """Concatenate all creatures into the flat arrays, shifting local indices."""
        crs = self.creatures
        bs = [cr.body for cr in crs]
        st = [cr.static for cr in crs]
        ids = np.arange(len(crs))

        def cat(arrs, shape=(0,), dtype=float):
            return np.concatenate(arrs) if arrs else np.zeros(shape, dtype=dtype)

        def starts(counts):
            return np.concatenate([[0], np.cumsum(counts, dtype=np.int64)]).astype(np.int64)

        def shifted(arrs, start, shape=(0,)):
            a = cat(arrs, shape, np.int64)
            sh = np.repeat(start[:-1], [len(x) for x in arrs])
            return a + sh.reshape((-1,) + (1,) * (a.ndim - 1))

        # Each creature's slice of each array (physics threads by creature).
        self.node_rng = starts([b.n_nodes for b in bs])
        self.spring_rng = starts([len(b.sa) for b in bs])
        self.edge_rng = starts([len(b.ea) for b in bs])
        self.tri_rng = starts([len(b.tri) for b in bs])
        self.mus_rng = starts([b.n_muscles for b in bs])
        cl_rng = starts([len(b.clusters) for b in bs])
        for cr, n0, m0, c0 in zip(crs, self.node_rng.tolist(), self.mus_rng.tolist(), cl_rng.tolist()):
            cr.noff, cr.moff, cr.coff = n0, m0, c0

        self.pos = cat([cr.pos for cr in crs], (0, 2)); self.vel = cat([cr.vel for cr in crs], (0, 2))
        self.act = cat([cr.act for cr in crs])
        self.mass = cat([b.mass for b in bs])
        self.sa = shifted([b.sa for b in bs], self.node_rng)
        self.sb = shifted([b.sb for b in bs], self.node_rng)
        self.r0 = cat([b.r0 for b in bs], (0, 2)); self.k = cat([b.k for b in bs]); self.c = cat([b.c for b in bs])
        smus = cat([b.smus for b in bs], dtype=np.int64)
        self.smus = np.where(smus >= 0, smus + np.repeat(self.mus_rng[:-1], [len(b.smus) for b in bs]), -1)
        self.mu = cat([b.mus_u for b in bs], (0, 2))
        self.s_own = cat([b.s_own for b in bs], dtype=bool)
        self.ea = shifted([b.ea for b in bs], self.node_rng)
        self.eb = shifted([b.eb for b in bs], self.node_rng)
        self.owner = np.repeat(ids, np.diff(self.node_rng))
        tri = shifted([b.tri for b in bs], self.node_rng, (0, 3))
        self.ta, self.tb, self.tc = tri[:, 0].copy(), tri[:, 1].copy(), tri[:, 2].copy()
        self.gpos = cat([b.grid_pos for b in bs], (0, 2))
        self.cell_quads = shifted([b.quads for b in bs], self.node_rng, (0, 4))
        self.cell_col = C.CELL_BRIGHT * cat([s["ccol"] for s in st], (0, 3))
        self.cell_owner = np.repeat(ids, [len(b.ctype) for b in bs])
        self.cl_owner = np.repeat(ids, np.diff(cl_rng))
        self.cln = shifted([s["cln"] for s in st], self.node_rng)
        self.clni = shifted([s["clni"] for s in st], cl_rng)
        self.clea = shifted([s["clea"] for s in st], self.node_rng)
        self.cleb = shifted([s["cleb"] for s in st], self.node_rng)
        self.clei = shifted([s["clei"] for s in st], cl_rng)
        self.cl_cells = cat([s["cl_cells"] for s in st])
        self.n_clusters = int(cl_rng[-1])
        self._n = off = int(self.node_rng[-1])
        self.inv_mass = 1.0 / np.maximum(self.mass, 1e-6)
        self.rest0 = np.linalg.norm(self.r0, axis=1)
        # rest vector = r0 - activation * proj
        self.proj = np.zeros_like(self.r0)
        m = self.smus >= 0
        if m.any():
            u = self.mu[self.smus[m]]
            self.proj[m] = C.MUSCLE_CONTRACTION * (self.r0[m] * u).sum(1, keepdims=True) * u
        self.core_nodes = (np.array([b.quads[b.core_quad] for b in bs], dtype=np.int64).reshape(-1, 4)
                           + self.node_rng[:-1, None])
        self.mus_owner = np.repeat(ids, np.diff(self.mus_rng))
        self.upkeep = np.array([b.upkeep for b in bs])
        self.size = np.array([b.n_cells for b in bs], dtype=float)
        self.eat_r = np.array([cr.eat_r for cr in crs], dtype=float)
        self._killer = np.full(len(crs), -1, dtype=np.int64)
        self.target = np.zeros_like(self.act)
        self._force, self._cforce, self._snap, self._grid_next = physics.scratch(off)
        self.brains = BrainBatch([cr.brain for cr in crs], [cr.coff for cr in crs], [cr.moff for cr in crs])
        self._dirty = False

    # ------------------------------------------------------------------- physics
    def _physics(self, dt):
        if self._n == 0:
            return
        sub = dt / C.SUBSTEPS
        physics.run_substeps(
            self.pos, self.vel, self.inv_mass, self.act, self.target, C.SUBSTEPS, sub,
            min(1.0, sub / C.MUSCLE_TAU),
            self.sa, self.sb, self.k, self.c, self.rest0, self.r0, self.proj, self.smus,
            self.s_own, C.MUSCLE_SLACK,
            self.ea, self.eb, C.DRAG_NORMAL, C.DRAG_QUAD, C.DRAG_TANGENT, C.GLOBAL_DAMP,
            self.ta, self.tb, self.tc, C.AREA_MIN, C.AREA_K, C.AREA_DAMP * math.sqrt(C.AREA_K),
            self.owner, self.gpos, C.COLLIDE_R, C.SELF_COLLIDE_R, C.COLLIDE_K, C.COLLIDE_EVERY,
            float(self.W), float(self.H), self._force, self._cforce, self._snap, self._grid_head, self._grid_next,
            self.node_rng, self.spring_rng, self.edge_rng, self.tri_rng, self.mus_rng)

    # -------------------------------------------------------------------- vision
    def _sense(self):
        nc = self.n_clusters
        if nc == 0:
            self.signals = np.zeros((0, 3))
            return
        self.cl_cent, self.cl_dir = vision.eye_geometry(self.pos, self.cln, self.clni, self.clea,
                                                        self.cleb, self.clei, nc)
        self.cl_range = C.EYE_RANGE_PER_CELL * self.cl_cells
        # brightness relative to a fresh pellet of its kind (carrion is dim)
        bright = np.minimum(1.0, self.food_e / self._food_base[self.food_k])
        nf = len(self.food_e)
        src_pos = np.empty((nf + len(self.cell_quads), 2))
        src_pos[:nf] = self.food_pos
        vision.cell_centres(self.pos, self.cell_quads, src_pos[nf:])
        src_col = np.concatenate([bright[:, None] * self._food_rgb[self.food_k], self.cell_col])
        src_own = np.concatenate([np.full(len(self.food_e), -1, dtype=np.int64), self.cell_owner])
        self.signals = np.empty((nc, 3))
        vision.see(self.cl_cent, self.cl_dir, self.cl_range, self.cl_owner, src_pos, src_col,
                   src_own, math.cos(math.radians(C.EYE_HALF_ANGLE_DEG)), float(self.W),
                   float(self.H), np.array(C.WALL_RGB), C.WALL_RAYS, self.signals)

    # ---------------------------------------------------------------------- food
    def _spawn_food(self, dt):
        self._hotspot_clock += dt
        if C.HOTSPOT_RELOCATE_PERIOD > 0 and self._hotspot_clock >= C.HOTSPOT_RELOCATE_PERIOD:
            self._hotspot_clock -= C.HOTSPOT_RELOCATE_PERIOD
            self.hotspots = self.rng.uniform([10, 10], [self.W - 10, self.H - 10],
                                             self.hotspots.shape)
        self.hotspots += self.rng.normal(0, C.HOTSPOT_DRIFT, self.hotspots.shape) * math.sqrt(dt)
        self.hotspots = np.clip(self.hotspots, [10, 10], [self.W - 10, self.H - 10])
        if self.pop_target:
            err = (len(self.creatures) - self.pop_target) / self.pop_target
            self.food_rate = float(np.clip(self.food_rate * math.exp(-C.FOOD_RATE_GAIN * err * dt),
                                           C.FOOD_RATE_MIN, C.FOOD_RATE_MAX))
        self._food_acc += self.food_rate * dt
        new, rich = [], []
        while self._food_acc >= 1.0:
            self._food_acc -= 1.0
            if len(self.food_e) + len(new) >= self.food_max:
                continue
            if self.rng.random() < C.FOOD_PATCH_FRACTION and len(self.hotspots):
                h = self.hotspots[self.rng.integers(len(self.hotspots))]
                p = h + self.rng.normal(0, 7.0, 2)
            else:
                p = self.rng.uniform([0, 0], [self.W, self.H])
            new.append(np.clip(p, [1, 1], [self.W - 1, self.H - 1]))
            rich.append(self.rng.random() < C.RICH_FOOD_FRACTION)
        if new:
            k = np.array(rich, dtype=np.int64)
            reach = np.array([c.eat_r for c in self.creatures], dtype=float) + C.FOOD_CORE_CLEAR
            self.add_food(self._push_out(np.array(new), self._live_cores(), reach),
                          self._food_base[k], k)

    def _live_cores(self):
        """Core centres, from creature.pos if the global arrays are stale (_dirty)."""
        if not self._dirty:
            return self.core_centres()
        cores = [c.pos[c.body.quads[c.body.core_quad]].mean(0) for c in self.creatures]
        return np.array(cores).reshape(-1, 2)

    def add_food(self, pts, energies, kinds=0):
        """`kinds`: 0 plain or 1 rich, per pellet or one for all."""
        pts = np.asarray(pts, dtype=float).reshape(-1, 2)
        self.food_pos = np.concatenate([self.food_pos, pts])
        self.food_e = np.concatenate([self.food_e, np.asarray(energies, dtype=float)])
        self.food_k = np.concatenate([self.food_k, np.broadcast_to(np.asarray(kinds, dtype=np.int64), len(pts))])

    # ---------------------------------------------------------------------- tick
    def tick(self):
        dt = C.BRAIN_DT
        if self._dirty:
            self._rebuild()
        numba.set_num_threads(physics.threads_for(self._n))
        self._sense()

        crs = self.creatures
        self.brains.step(np.array([c.energy / c.max_energy for c in crs]), np.array([c.ate for c in crs]),
                         np.array([c.age for c in crs]), self.signals, dt, self.target)

        self._physics(dt)
        if self._n and not np.all(np.isfinite(self.pos)):
            self._kill_unstable()

        self._metabolism(dt)
        self._spawn_food(dt)
        self.t += dt
        if int(self.t / dt) % 20 == 0:
            gens = [c.gen for c in self.creatures] or [0]
            self.history.append((round(self.t, 1), len(self.creatures),
                                 round(float(np.mean(gens)), 2), len(self.food_e),
                                 self.mean_top_speed(), round(self.food_rate, 2)))

    def core_centres(self):
        return self.pos[self.core_nodes].mean(1)

    def _kill_unstable(self):
        for cr in self.creatures:
            p = self.pos[cr.noff:cr.noff + cr.body.n_nodes]
            if not np.all(np.isfinite(p)):
                cr.dead = True
                self.pos[cr.noff:cr.noff + cr.body.n_nodes] = 0.0
                self.vel[cr.noff:cr.noff + cr.body.n_nodes] = 0.0

    def _metabolism(self, dt):
        cores = self.core_centres()
        eat_mask = np.zeros(len(self.food_e), dtype=bool)
        burn = ((self.upkeep + C.MUSCLE_ACTIVE_COST
                 * np.bincount(self.mus_owner, self.act, len(self.creatures))) * dt).tolist()
        meals = {}
        for i, f in zip(*(a.tolist() for a in physics.food_in_reach(cores, self.food_pos, self.eat_r))):
            meals.setdefault(i, []).append(f)
        ate_decay = math.exp(-dt / C.ATE_TAU)
        for i, cr in enumerate(self.creatures):
            if int((cr.age + dt) / C.SPEED_SAMPLE) != int(cr.age / C.SPEED_SAMPLE):
                self._track_speed(cr, cores[i])
            cr.age += dt
            cr.energy -= burn[i]
            cr.ate *= ate_decay
            if i in meals and cr.energy < cr.max_energy:
                for f in meals[i]:
                    if eat_mask[f]:
                        continue
                    if cr.energy >= cr.max_energy:
                        break
                    cr.energy = min(cr.max_energy, cr.energy + self.food_e[f])
                    cr.ate = 1.0 - (1.0 - cr.ate) * math.exp(-self.food_e[f] / C.FOOD_ENERGY)
                    cr.eaten += 1
                    self.stats["rich_eaten"] += int(self.food_k[f])
                    eat_mask[f] = True
            if cr.energy <= 0 or cr.age > cr.max_age:
                cr.dead = True
        if eat_mask.any():
            self.food_pos = self.food_pos[~eat_mask]
            self.food_e = self.food_e[~eat_mask]
            self.food_k = self.food_k[~eat_mask]
        self._predation(cores)

        births = []
        n_alive = sum(not c.dead for c in self.creatures)
        for i, cr in enumerate(self.creatures):
            if cr.dead or not self.allow_births or n_alive + len(births) >= C.MAX_POPULATION:
                continue
            need = cr.body.build_cost + cr.child_energy + C.PARENT_RESERVE * cr.max_energy
            if cr.energy >= need:
                births.append((cr, cores[i]))

        dead = [c for c in self.creatures if c.dead]
        if dead or births:
            self._sync_out()
            for cr in dead:
                self.stats["deaths"] += 1
                pts = self.pos[cr.noff:cr.noff + cr.body.n_nodes] if not self._dirty else cr.pos
                e = C.CARRION_FRACTION * cr.body.build_cost
                if cr.killed_by is not None:
                    e += C.KILL_ENERGY_FRACTION * max(cr.energy, 0.0)
                k = max(1, int(round(e / 2.0)))
                pick = pts[self.rng.integers(len(pts), size=k)]
                if np.all(np.isfinite(pick)):
                    self.add_food(np.clip(pick, 1, [self.W - 1, self.H - 1]), np.full(k, e / k))
            self.creatures = [c for c in self.creatures if not c.dead]
            self._dirty = True
            for parent, p in births:
                self._reproduce(parent, p)

        if len(self.creatures) < C.MIN_POPULATION and self.seed_genomes:
            src = self.rng.integers(len(self.seed_genomes))
            best = max(self.creatures, key=lambda c: c.eaten, default=None)
            g = best.genome if (best is not None and self.rng.random() < 0.7) else self.seed_genomes[src]
            child = g.mutate(self.rng)
            self.spawn(child, energy=2 * C.CHILD_ENERGY_FRACTION * Body(child.body.cells).build_cost,
                       gen=(best.gen + 1 if best is not None and g is best.genome else 0))

    def _predation(self, cores):
        """Mark prey dead (see config); _metabolism turns them into carrion. A predator that
        died this tick can't kill."""
        if len(self.creatures) < 2:
            return
        physics.core_contacts(self.pos, self.owner, cores, C.PREDATION_REACH_FRACTION * self.eat_r,
                              self.size, C.PREDATION_SIZE_RATIO, C.COLLIDE_R,
                              self._grid_head, self._grid_next, self._killer)
        for j in np.nonzero(self._killer >= 0)[0]:
            prey, pred = self.creatures[j], self.creatures[self._killer[j]]
            if prey.dead or pred.dead:
                continue
            prey.dead = True
            prey.killed_by = pred.id
            pred.kills += 1
            self.stats["kills"] += 1

    @staticmethod
    def _track_speed(cr, core):
        if not np.all(np.isfinite(core)):
            return
        if cr.track:
            cr.odometer += float(np.linalg.norm(core - cr.track[-1]))
            cr.legs += 1
        cr.track.append(core.copy())
        if len(cr.track) > 1:
            span = (len(cr.track) - 1) * C.SPEED_SAMPLE
            cr.speed = float(np.linalg.norm(cr.track[-1] - cr.track[0])) / span
            if len(cr.track) == cr.track.maxlen:
                cr.top_speed = max(cr.top_speed, cr.speed)

    def mean_top_speed(self):
        """Mean best speed (cells/s) over creatures with a full speed window; keeps the last
        value if there are none."""
        tops = [c.top_speed for c in self.creatures if c.legs >= C.SPEED_WINDOW]
        if tops:
            self._top_speed = round(float(np.mean(tops)), 2)
        return self._top_speed

    @staticmethod
    def mean_speed(cr):
        """Lifetime average speed, cells/s."""
        return cr.odometer / (C.SPEED_SAMPLE * cr.legs) if cr.legs else 0.0

    def _reproduce(self, parent, at):
        g = parent.genome.mutate(self.rng)
        child_cost = Body(g.body.cells).build_cost
        child_energy = C.CHILD_ENERGY_FRACTION * child_cost
        parent.energy -= child_cost + child_energy
        parent.children += 1
        self.stats["births"] += 1
        self.spawn(g, at[0], at[1], min_dist=parent.body.radius + 4,
                   energy=child_energy, gen=parent.gen + 1, parent=parent.id)

    # ------------------------------------------------------------------ snapshot
    def snapshot(self, selected=None, since=None):
        """Viewer state (keys are index.html's contract). Static structure only for ids > since."""
        if self._dirty:
            self._rebuild()
        crs = []
        pos, act = np.round(self.pos, 2).ravel().tolist(), np.round(self.act, 2).tolist()
        for cr in self.creatures:
            b = cr.body
            d = dict(id=cr.id, gen=cr.gen, rgb=[round(x, 3) for x in cr.genome.colour],
                     e=round(cr.energy / cr.max_energy, 3),
                     n=pos[2 * cr.noff:2 * (cr.noff + b.n_nodes)],
                     a=act[cr.moff:cr.moff + b.n_muscles])
            if since is None or cr.id > since:
                d["q"] = b.quads.ravel().tolist()
                d["t"] = b.ctype.tolist()
                d["m"] = b.cmus.tolist()
            crs.append(d)
        gens = [c.gen for c in self.creatures] or [0]
        out = dict(t=round(self.t, 2), W=self.W, H=self.H,
                   food=np.round(self.food_pos[self.food_k == 0], 2).ravel().tolist(),
                   rich=np.round(self.food_pos[self.food_k == 1], 2).ravel().tolist(),
                   creatures=crs,
                   stats=dict(pop=len(self.creatures), food=len(self.food_e),
                              rich=int(self.food_k.sum()),
                              births=self.stats["births"], deaths=self.stats["deaths"],
                              kills=self.stats["kills"],
                              max_gen=self.stats["max_gen"],
                              mean_gen=round(float(np.mean(gens)), 2),
                              top_speed=self.mean_top_speed(),
                              food_rate=round(self.food_rate, 2),
                              pop_target=self.pop_target),
                   history=list(self.history)[-300:])
        sel = next((c for c in self.creatures if c.id == selected), None)
        if sel is not None:
            out["selected"] = self._detail(sel)
        return out

    def _detail(self, cr):
        br = cr.brain
        cl = slice(cr.coff, cr.coff + len(cr.body.clusters))
        eyes = []
        if self.n_clusters and len(self.cl_cent):
            for c, d, r, s in zip(self.cl_cent[cl], self.cl_dir[cl], self.cl_range[cl],
                                  self.signals[cl] if len(self.signals) else []):
                eyes.append(dict(p=np.round(c, 2).tolist(), d=np.round(d, 3).tolist(),
                                 r=float(r), s=np.round(s, 3).tolist()))
        from .genome import FIXED_NAMES
        names = []
        for nid, kind in zip(br.ids, br.kinds):
            names.append(FIXED_NAMES.get(nid, kind))
        counts = np.bincount(cr.body.ctype, minlength=5).tolist()
        g = cr.genome.brain
        pos = [list(g.nodes[i].pos) if g.nodes[i].kind in ("sensor", "motor") else None
               for i in br.ids]
        tune = [np.round(g.nodes[i].tune, 2).tolist() if g.nodes[i].kind == "sensor" else None
                for i in br.ids]
        reach = [round(g.nodes[i].reach, 2) if g.nodes[i].kind in ("sensor", "motor") else None
                 for i in br.ids]
        # Grid-space centre of what each I/O node binds (pos is in address space).
        at = [None] * len(br.ids)
        for k, nid in enumerate(br.sensor_ids):
            w = br.bind_s[k]
            if w.sum() > 0.05:
                at[br.ids.index(nid)] = np.round(w @ cr.body.cluster_pos / w.sum(), 2).tolist()
        for k, nid in enumerate(br.motor_ids):
            w = br.bind_m[:, k]
            if w.sum() > 0.05:
                at[br.ids.index(nid)] = np.round(w @ cr.body.mus_pos / w.sum(), 2).tolist()
        addr = cr.genome.body.addr
        grid = [[c[0], c[1], t, *np.round(addr.get(c, c), 2).tolist()] for c, t, _ in cr.body.cells]
        bg = cr.genome.body
        cppn = dict(hidden=len(bg.cppn.ids("hidden")),
                    links=sum(c.enabled for c in bg.cppn.conns.values()),
                    sym=round(bg.sym_prob, 2))
        par = next((c for c in self.creatures if c.id == cr.parent), None)
        speed = dict(now=round(cr.speed, 2), avg=round(self.mean_speed(cr), 2),
                     top=round(cr.top_speed, 2),
                     parent=round(self.mean_speed(par), 2) if par is not None else None)
        return dict(id=cr.id, gen=cr.gen, parent=cr.parent, lineage=cr.genome.lineage, age=round(cr.age, 1), speed=speed,
                    energy=round(cr.energy, 2), max_energy=round(cr.max_energy, 1),
                    max_age=round(cr.max_age),
                    children=cr.children, eaten=cr.eaten, kills=cr.kills, cells=counts,
                    mut_rate=round(cr.genome.mut_rate, 3), osc=round(br.freq, 3),
                    upkeep=round(cr.body.upkeep, 3),
                    brain=dict(kinds=br.kinds, names=names,
                               v=np.round(br.values, 3).tolist(),
                               e=[(s, d, round(w, 2), int(pl)) for (s, d, pl), w
                                  in zip(br.edges, br.edge_weights())],
                               pos=pos, tune=tune, reach=reach, at=at),
                    grid=grid, cppn=cppn,
                    eyes=eyes)
