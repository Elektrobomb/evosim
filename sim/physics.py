"""Numba physics kernels, plus the eating and predation queries.

Parallel by creature: a creature's internal forces touch only its own nodes. Contact is
summed per node from a position snapshot, so no two threads write one element.
Constants are arguments: numba bakes globals into its disk cache.
"""
import math

import numpy as np
import numba
from numba import njit, prange


@njit(cache=True, nogil=True, parallel=True)
def run_substeps(pos, vel, inv_mass, act, target, n_sub, dt, act_alpha,
                 sa, sb, k, c, rest0, r0, proj, smus, own, slack,
                 ea, eb, drag_n, drag_q, drag_t, global_damp,
                 ta, tb, tc, area_min, area_k, area_c,
                 owner, gpos, collide_r, self_r, collide_k, collide_every,
                 W, H, force, cforce, snap, head, nxt,
                 node_rng, spring_rng, edge_rng, tri_rng, mus_rng):
    """All substeps of one tick. Creature b owns node_rng[b]:node_rng[b + 1] (likewise
    springs, edges, triangles, muscles). A thread pass costs about a substep's work, so
    each pass covers one contact period of collide_every substeps."""
    nb = node_rng.shape[0] - 1
    period = collide_every if collide_every > 0 else n_sub
    for s0 in range(0, n_sub, period):
        s1 = min(s0 + period, n_sub)
        if collide_every > 0:
            if s0 > 0:
                _unbin(snap, collide_r, head)
            snap[:, :] = pos
            _bin(snap, collide_r, head, nxt)
        for b in prange(nb):
            if collide_every > 0:
                for i in range(node_rng[b], node_rng[b + 1]):
                    _contact(i, snap, owner, gpos, collide_r, self_r, collide_k, head, nxt, cforce)
            for s in range(s0, s1):
                contact = collide_every > 0 and s == s0
                _creature_substep(b, pos, vel, inv_mass, act, target, dt, act_alpha,
                                  sa, sb, k, c, rest0, r0, proj, smus, own, slack,
                                  ea, eb, drag_n, drag_q, drag_t, global_damp,
                                  ta, tb, tc, area_min, area_k, area_c,
                                  W, H, force, cforce, contact,
                                  node_rng, spring_rng, edge_rng, tri_rng, mus_rng)
    if collide_every > 0:
        _unbin(snap, collide_r, head)


@njit(cache=True, nogil=True)
def _creature_substep(b, pos, vel, inv_mass, act, target, dt, act_alpha,
                      sa, sb, k, c, rest0, r0, proj, smus, own, slack,
                      ea, eb, drag_n, drag_q, drag_t, global_damp,
                      ta, tb, tc, area_min, area_k, area_c,
                      W, H, force, cforce, contact,
                      node_rng, spring_rng, edge_rng, tri_rng, mus_rng):
    """One substep of creature b; touches only b's nodes and muscles."""
    for m in range(mus_rng[b], mus_rng[b + 1]):
        act[m] += (target[m] - act[m]) * act_alpha
    n0, n1 = node_rng[b], node_rng[b + 1]
    for i in range(n0, n1):
        force[i, 0] = 0.0
        force[i, 1] = 0.0

    # Springs. A muscle's rest vector is r0 squashed along its axis.
    for i in range(spring_rng[b], spring_rng[b + 1]):
        a, q = sa[i], sb[i]
        dx, dy = pos[q, 0] - pos[a, 0], pos[q, 1] - pos[a, 1]
        L = math.sqrt(dx * dx + dy * dy) + 1e-9
        ux, uy = dx / L, dy / L
        kk, cc, rest = k[i], c[i], rest0[i]
        m = smus[i]
        if m >= 0:
            am = act[m]
            rx, ry = r0[i, 0] - am * proj[i, 0], r0[i, 1] - am * proj[i, 1]
            rest = math.sqrt(rx * rx + ry * ry)
            if own[i] and slack != 1.0:
                sc = slack + (1.0 - slack) * am
                kk *= sc
                cc *= math.sqrt(sc)
        f = kk * (L - rest) + cc * ((vel[q, 0] - vel[a, 0]) * ux + (vel[q, 1] - vel[a, 1]) * uy)
        force[a, 0] += f * ux
        force[a, 1] += f * uy
        force[q, 0] -= f * ux
        force[q, 1] -= f * uy

    # Drag on exposed edges; pressure drag on leading faces only.
    for i in range(edge_rng[b], edge_rng[b + 1]):
        a, q = ea[i], eb[i]
        ex, ey = pos[q, 0] - pos[a, 0], pos[q, 1] - pos[a, 1]
        el = math.sqrt(ex * ex + ey * ey) + 1e-9
        tx, ty = ex / el, ey / el
        nx, ny = ty, -tx
        vx, vy = 0.5 * (vel[a, 0] + vel[q, 0]), 0.5 * (vel[a, 1] + vel[q, 1])
        vp = max(vx * nx + vy * ny, 0.0)
        fn = -(drag_n * vp + drag_q * vp * vp) * el
        ft = -drag_t * (vx * tx + vy * ty) * el
        Dx, Dy = 0.5 * (fn * nx + ft * tx), 0.5 * (fn * ny + ft * ty)
        force[a, 0] += Dx
        force[a, 1] += Dy
        force[q, 0] += Dx
        force[q, 1] += Dy

    # Anti-inversion barrier on corner triangles. d(area)/d(vertex) = half the opposite
    # edge turned 90°.
    for i in range(tri_rng[b], tri_rng[b + 1]):
        a, p, q = ta[i], tb[i], tc[i]
        abx, aby = pos[p, 0] - pos[a, 0], pos[p, 1] - pos[a, 1]
        acx, acy = pos[q, 0] - pos[a, 0], pos[q, 1] - pos[a, 1]
        area = 0.5 * (abx * acy - aby * acx)
        if area >= area_min:
            continue
        gax, gay = -0.5 * (pos[q, 1] - pos[p, 1]), 0.5 * (pos[q, 0] - pos[p, 0])
        gbx, gby = -0.5 * (pos[a, 1] - pos[q, 1]), 0.5 * (pos[a, 0] - pos[q, 0])
        gcx, gcy = -0.5 * aby, 0.5 * abx
        rate = (gax * vel[a, 0] + gay * vel[a, 1] + gbx * vel[p, 0] + gby * vel[p, 1]
                + gcx * vel[q, 0] + gcy * vel[q, 1])
        f = max(area_k * (area_min - area) - area_c * rate, 0.0)
        force[a, 0] += f * gax
        force[a, 1] += f * gay
        force[p, 0] += f * gbx
        force[p, 1] += f * gby
        force[q, 0] += f * gcx
        force[q, 1] += f * gcy

    # Semi-implicit Euler; walls reflect at 20%.
    damp = 1.0 - global_damp * dt
    for i in range(n0, n1):
        for d in range(2):
            fd = force[i, d] + cforce[i, d] if contact else force[i, d]
            v = (vel[i, d] + fd * inv_mass[i] * dt) * damp
            p = pos[i, d] + v * dt
            hi = W if d == 0 else H
            if p < 0.0:
                p, v = 0.0, abs(v) * 0.2
            elif p > hi:
                p, v = hi, -abs(v) * 0.2
            vel[i, d] = v
            pos[i, d] = p


@njit(cache=True, nogil=True)
def _contact(i, pos, owner, gpos, collide_r, self_r, collide_k, head, nxt, cforce):
    """Set cforce[i]. Same-creature nodes repel within self_r only if they don't share a
    cell (rest distance > 1.5)."""
    gx = head.shape[0]
    gy = head.shape[1]
    fx = fy = 0.0
    x, y = pos[i, 0], pos[i, 1]
    if math.isfinite(x) and math.isfinite(y):
        r2 = collide_r * collide_r
        cx = min(max(int(x / collide_r), 0), gx - 1)
        cy = min(max(int(y / collide_r), 0), gy - 1)
        for ox in range(max(cx - 1, 0), min(cx + 2, gx)):
            for oy in range(max(cy - 1, 0), min(cy + 2, gy)):
                j = head[ox, oy]
                while j >= 0:
                    if j != i:
                        dx, dy = pos[j, 0] - x, pos[j, 1] - y
                        d2 = dx * dx + dy * dy
                        if d2 < r2:
                            r = collide_r
                            hit = True
                            if owner[i] == owner[j]:
                                gdx, gdy = gpos[j, 0] - gpos[i, 0], gpos[j, 1] - gpos[i, 1]
                                hit = d2 < self_r * self_r and gdx * gdx + gdy * gdy > 2.25
                                r = self_r
                            if hit:
                                L = math.sqrt(d2) + 1e-9
                                f = collide_k * (r - L) / L
                                fx -= f * dx
                                fy -= f * dy
                    j = nxt[j]
    cforce[i, 0] = fx
    cforce[i, 1] = fy


@njit(cache=True, nogil=True)
def _bin(pos, cell, head, nxt):
    """Fill the grid's linked lists (head, nxt). head must be all -1 before, and every
    _bin must be undone by _unbin with the same positions (wiping it is too slow)."""
    gx, gy = head.shape[0], head.shape[1]
    for i in range(pos.shape[0]):
        x, y = pos[i, 0], pos[i, 1]
        if not (math.isfinite(x) and math.isfinite(y)):
            continue                                  # unstable, about to be culled
        cx = min(max(int(x / cell), 0), gx - 1)
        cy = min(max(int(y / cell), 0), gy - 1)
        nxt[i] = head[cx, cy]
        head[cx, cy] = i


@njit(cache=True, nogil=True)
def _unbin(pos, cell, head):
    """Undo _bin(pos, ...)."""
    gx, gy = head.shape[0], head.shape[1]
    for i in range(pos.shape[0]):
        x, y = pos[i, 0], pos[i, 1]
        if math.isfinite(x) and math.isfinite(y):
            head[min(max(int(x / cell), 0), gx - 1), min(max(int(y / cell), 0), gy - 1)] = -1


@njit(cache=True, nogil=True)
def core_contacts(pos, owner, cores, reach, size, ratio, cell, head, nxt, killer):
    """killer[j] = the largest creature i with a node of j within reach[i] of its core and
    more than `ratio` times j's size, else -1."""
    _bin(pos, cell, head, nxt)
    gx, gy = head.shape[0], head.shape[1]
    killer[:] = -1
    for i in range(cores.shape[0]):
        x, y = cores[i, 0], cores[i, 1]
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        r2 = reach[i] * reach[i]
        span = int(math.ceil(reach[i] / cell))
        cx = min(max(int(x / cell), 0), gx - 1)
        cy = min(max(int(y / cell), 0), gy - 1)
        for ox in range(max(cx - span, 0), min(cx + span + 1, gx)):
            for oy in range(max(cy - span, 0), min(cy + span + 1, gy)):
                j = head[ox, oy]
                while j >= 0:
                    o = owner[j]
                    if o != i and size[i] > ratio * size[o]:
                        dx, dy = pos[j, 0] - x, pos[j, 1] - y
                        if dx * dx + dy * dy < r2 and (killer[o] < 0 or size[i] > size[killer[o]]):
                            killer[o] = i
                    j = nxt[j]
    _unbin(pos, cell, head)


@njit(cache=True, nogil=True)
def food_in_reach(cores, food_pos, r):
    """Pairs (creature i, pellet f) with f within r[i] of core i, ordered by i then f."""
    n = 0
    ci = np.empty(64, dtype=np.int64)
    fi = np.empty(64, dtype=np.int64)
    for i in range(cores.shape[0]):
        x, y = cores[i, 0], cores[i, 1]
        r2 = r[i] * r[i]
        for f in range(food_pos.shape[0]):
            dx = food_pos[f, 0] - x
            if dx * dx >= r2:
                continue
            dy = food_pos[f, 1] - y
            if dx * dx + dy * dy < r2:
                if n == ci.shape[0]:
                    ci = np.concatenate((ci, np.empty(n, dtype=np.int64)))
                    fi = np.concatenate((fi, np.empty(n, dtype=np.int64)))
                ci[n] = i
                fi[n] = f
                n += 1
    return ci[:n], fi[:n]


def threads_for(n_nodes):
    """Thread count for a world of n_nodes. Sync costs about as much as the work, so small
    worlds use at most half the pool (one chiplet on a 5900X)."""
    pool = numba.config.NUMBA_NUM_THREADS
    if n_nodes >= 2000:
        return pool
    return max(1, min(pool // 2, n_nodes // 150, pool))


def grid_shape(W, H, cell):
    return int(W / cell) + 2, int(H / cell) + 2


def scratch(n):
    """Work arrays for run_substeps: force, cforce, snap, nxt."""
    return np.zeros((n, 2)), np.zeros((n, 2)), np.zeros((n, 2)), np.full(n, -1, dtype=np.int64)
