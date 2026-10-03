"""Numba vision: unsquashed RGB light at every eye cluster.

Point sources (food, other creatures' cells) add colour · cone · (1 − d/range), with cone
linear in cos(angle) from 1 on-axis to 0 at the edge. Walls are sampled by rays across the
cone, triangle-weighted; a wall filling the view at distance 0 reads exactly its colour.
"""
import math

import numpy as np

from numba import njit, prange


@njit(cache=True, nogil=True, parallel=True)
def see(cent, dirs, rng, cl_owner, src_pos, src_col, src_owner,
        cos_half, W, H, wall_col, n_rays, out):
    """out[c] = RGB at cluster c. Skips the cluster's own body (food has owner -1). A
    cluster with no facing direction sees nothing."""
    nc, ns = cent.shape[0], src_pos.shape[0]
    half = math.acos(cos_half)
    for c in prange(nc):
        out[c, 0] = out[c, 1] = out[c, 2] = 0.0
        cx, cy, ux, uy, r = cent[c, 0], cent[c, 1], dirs[c, 0], dirs[c, 1], rng[c]
        if ux * ux + uy * uy < 0.25 or r <= 0.0:
            continue
        r2 = r * r
        for m in range(ns):
            if src_owner[m] == cl_owner[c]:
                continue
            dx, dy = src_pos[m, 0] - cx, src_pos[m, 1] - cy
            d2 = dx * dx + dy * dy
            if d2 >= r2:
                continue
            d = math.sqrt(d2) + 1e-9
            cone = ((dx * ux + dy * uy) / d - cos_half) / (1.0 - cos_half)
            if cone <= 0.0:
                continue
            wgt = min(cone, 1.0) * (1.0 - d / r)
            out[c, 0] += wgt * src_col[m, 0]
            out[c, 1] += wgt * src_col[m, 1]
            out[c, 2] += wgt * src_col[m, 2]

        # Walls
        base = math.atan2(uy, ux)
        acc = tot = 0.0
        for k in range(n_rays):
            f = -1.0 + 2.0 * (k + 0.5) / n_rays
            wk = 1.0 - abs(f)
            rx, ry = math.cos(base + f * half), math.sin(base + f * half)
            t = 1e18
            if rx > 1e-9:
                t = min(t, (W - cx) / rx)
            elif rx < -1e-9:
                t = min(t, -cx / rx)
            if ry > 1e-9:
                t = min(t, (H - cy) / ry)
            elif ry < -1e-9:
                t = min(t, -cy / ry)
            acc += wk * max(0.0, 1.0 - max(t, 0.0) / r)
            tot += wk
        v = acc / tot
        out[c, 0] += v * wall_col[0]
        out[c, 1] += v * wall_col[1]
        out[c, 2] += v * wall_col[2]


@njit(cache=True, nogil=True)
def eye_geometry(pos, cln, clni, clea, cleb, clei, nc):
    """Each cluster's centre (mean node) and facing (unit sum of exposed-edge normals)."""
    cent = np.zeros((nc, 2))
    dirs = np.zeros((nc, 2))
    cnt = np.zeros(nc)
    for k in range(cln.shape[0]):
        c = clni[k]
        cent[c, 0] += pos[cln[k], 0]
        cent[c, 1] += pos[cln[k], 1]
        cnt[c] += 1.0
    for k in range(clea.shape[0]):
        c = clei[k]
        dirs[c, 0] += pos[cleb[k], 1] - pos[clea[k], 1]
        dirs[c, 1] -= pos[cleb[k], 0] - pos[clea[k], 0]
    for c in range(nc):
        cent[c, 0] /= cnt[c]
        cent[c, 1] /= cnt[c]
        dl = math.sqrt(dirs[c, 0] ** 2 + dirs[c, 1] ** 2) + 1e-9
        dirs[c, 0] /= dl
        dirs[c, 1] /= dl
    return cent, dirs


@njit(cache=True, nogil=True)
def cell_centres(pos, quads, out):
    for i in range(quads.shape[0]):
        for d in range(2):
            out[i, d] = (pos[quads[i, 0], d] + pos[quads[i, 1], d] + pos[quads[i, 2], d]
                         + pos[quads[i, 3], d]) * 0.25
