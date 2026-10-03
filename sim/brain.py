"""CTRNN brain runtime.

    tau dy/dt = -y + W·out        out = tanh(y + bias)
    sensor input = tanh(tune · Σ_c bind_c · rgb_c)
    plastic links: dw/dt = eta·(A·pre·post + B·pre + C·post + D), clipped to ±HEBB_W_MAX
    links into a modulated neuron j: elig += dt/ELIG_TAU·(hebb − elig), dw/dt = eta·m_j·elig

Sensors and motors bind eye clusters and muscles by distance in wiring-address space.
`step_all` (numba) steps every brain in a `BrainBatch` at once; `Brain.step` is the
reference version. Keep the two in sync.
"""
import math

import numpy as np
from numba import njit

from .config import HEBB_W_MAX, MOD_MAX, ELIG_TAU
from .genome import FIXED_INPUTS, DYN_KINDS


def _gauss(a, b, sigma):
    """exp(-|a_i - b_j|² / 2σ²); sigma broadcasts against the (len(a), len(b)) result."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    return np.exp(-d2 / (2 * sigma ** 2))


class Brain:
    def __init__(self, g, body, rng):
        self.sensor_ids = g.ids("sensor")
        self.in_ids = list(FIXED_INPUTS) + self.sensor_ids
        self.dyn_ids = [i for k in DYN_KINDS for i in g.ids(k)]
        self.motor_ids = g.ids("motor")
        self.all_ids = self.in_ids + self.dyn_ids
        idx = {nid: i for i, nid in enumerate(self.all_ids)}
        n_in, n_dyn = len(self.in_ids), len(self.dyn_ids)
        self.n_in = n_in

        self.W = np.zeros((n_dyn, n_in + n_dyn))
        self.edges = []  # (src_idx, dst_idx, plastic), for the viewer
        plastic = []     # (src_idx, dst_row, eta*A, eta*B, eta*C, eta*D)
        mod = []         # (src_idx, dst_row, w)
        for c in g.conns.values():
            if c.enabled and c.src in idx and c.dst in idx and idx[c.dst] >= n_in:
                self.edges.append((idx[c.src], idx[c.dst], c.eta > 0 and g.nodes[c.src].kind != "mod"))
                if g.nodes[c.src].kind == "mod":
                    mod.append((idx[c.src], idx[c.dst] - n_in, c.w))
                    continue
                self.W[idx[c.dst] - n_in, idx[c.src]] += c.w
                if c.eta > 0:
                    plastic.append((idx[c.src], idx[c.dst] - n_in, *(c.eta * h for h in c.hebb)))
        # (src, dst) is unique per genome, so each plastic edge owns exactly one W entry.
        p = np.array(plastic, dtype=float).reshape(-1, 6)
        self.p_src, self.p_dst = p[:, 0].astype(np.int64), p[:, 1].astype(np.int64)
        self.p_flat = self.p_dst * (n_in + n_dyn) + self.p_src    # index into W.ravel()
        self.p_a, self.p_b, self.p_c, self.p_d = p[:, 2].copy(), p[:, 3].copy(), p[:, 4].copy(), p[:, 5].copy()
        self._Wf = self.W.ravel()                                 # a view: writes go to W
        md = np.array(mod, dtype=float).reshape(-1, 3)
        self.md_src, self.md_dst, self.md_w = md[:, 0].astype(np.int64), md[:, 1].astype(np.int64), md[:, 2].copy()
        self.n_dyn = n_dyn
        self.m = np.ones(n_dyn)
        self.p_modded = np.isin(self.p_dst, self.md_dst)          # plastic links that use elig
        self.elig = np.zeros(len(self.p_src))
        self.bias = np.array([g.nodes[i].bias for i in self.dyn_ids])
        self.tau = np.array([g.nodes[i].tau for i in self.dyn_ids])
        self.motor_idx = np.array([self.dyn_ids.index(m) for m in self.motor_ids], dtype=np.int64)
        self.kinds = [g.nodes[i].kind for i in self.all_ids]
        self.ids = self.all_ids

        s_pos = np.array([g.nodes[i].pos for i in self.sensor_ids], dtype=float).reshape(-1, 2)
        m_pos = np.array([g.nodes[i].pos for i in self.motor_ids], dtype=float).reshape(-1, 2)
        s_reach = np.array([g.nodes[i].reach for i in self.sensor_ids], dtype=float)
        m_reach = np.array([g.nodes[i].reach for i in self.motor_ids], dtype=float)
        self.bind_s = _gauss(s_pos, body.cluster_addr, s_reach[:, None])   # sensors x clusters
        self.tune = np.array([g.nodes[i].tune for i in self.sensor_ids], dtype=float).reshape(-1, 3)
        wm = _gauss(body.mus_addr, m_pos, m_reach[None, :])                # muscles x motors
        self.bind_m = wm / (wm.sum(1, keepdims=True) + 0.05)

        self.freq = g.osc_freq
        self.phase = rng.uniform(0, 2 * math.pi)
        self.y = np.zeros(n_dyn)
        self.values = np.zeros(n_in + n_dyn)

    def step(self, energy_frac, ate, t, cluster_signals, dt):
        ph = 2 * math.pi * self.freq * t + self.phase
        inp = np.empty(self.n_in)
        inp[0], inp[1], inp[2], inp[3] = energy_frac, math.sin(ph), math.cos(ph), ate
        if len(self.sensor_ids):
            inp[len(FIXED_INPUTS):] = (np.tanh(((self.bind_s @ cluster_signals) * self.tune).sum(1))
                       if cluster_signals.size else 0.0)
        out = np.tanh(self.y + self.bias)
        full = np.concatenate([inp, out])
        self.y += (dt / self.tau) * (-self.y + self.W @ full)
        out = np.tanh(self.y + self.bias)
        self.values[:self.n_in], self.values[self.n_in:] = inp, out   # in place: may be a BrainBatch view
        if len(self.p_src):
            # flat take/put and in-place min/max: ~3x cheaper than fancy indexing + np.clip
            pre, post = self.values.take(self.p_src), out.take(self.p_dst)
            dw = (self.p_a * post + self.p_b) * pre + self.p_c * post + self.p_d
            if len(self.md_src):
                self.elig += (dt / ELIG_TAU) * (dw - self.elig)
                m = 1.0 + np.bincount(self.md_dst, self.md_w * self.values.take(self.md_src), self.n_dyn)
                np.minimum(m, MOD_MAX, out=m)
                np.maximum(m, -MOD_MAX, out=m)
                self.m[:] = m
                dw = np.where(self.p_modded, m.take(self.p_dst) * self.elig, dw)
            w = self._Wf.take(self.p_flat)
            w += dt * dw
            np.minimum(w, HEBB_W_MAX, out=w)
            np.maximum(w, -HEBB_W_MAX, out=w)
            self._Wf.put(self.p_flat, w)
        if len(self.motor_idx) == 0:
            return np.zeros(self.bind_m.shape[0])
        motor = 0.5 * (out[self.motor_idx] + 1.0)
        return self.bind_m @ motor

    def edge_weights(self):
        """Live weight of each edge in self.edges (modulatory links: their fixed weight)."""
        mw = {(int(s), int(d) + self.n_in): float(w) for s, d, w in zip(self.md_src, self.md_dst, self.md_w)}
        return [mw[(s, d)] if (s, d) in mw else float(self.W[d - self.n_in, s]) for s, d, _ in self.edges]


# Columns of BrainBatch.off, one row per brain.
(O_NIN, O_NDYN, O_V, O_Y, O_W, O_P0, O_P1, O_MD0, O_MD1, O_S, O_CL, O_NCL, O_BS,
 O_MU, O_NMU, O_MO, O_NMO, O_BM) = range(18)


class BrainBatch:
    """Every brain packed into flat arrays for `step_all`. Each Brain's W, y, values, elig
    and m are then repointed at views of the packed arrays, so they stay live."""

    def __init__(self, brains, cl_off, mu_off):
        off = []
        L = {k: [] for k in ("vals", "y", "bias", "tau", "W", "p_src", "p_dst", "p_flat", "p_a", "p_b",
                             "p_c", "p_d", "elig", "p_mod", "md_src", "md_dst", "md_w", "m", "tune",
                             "bind_s", "motor", "bind_m", "freq", "phase")}
        v = yo = wo = po = mo = so = bs = mto = bm = 0
        for i, br in enumerate(brains):
            ni, nd = br.n_in, br.n_dyn
            ns, ncl = br.bind_s.shape
            nmu, nmo = br.bind_m.shape
            npl, nmd = len(br.p_src), len(br.md_src)
            off.append((ni, nd, v, yo, wo, po, po + npl, mo, mo + nmd, so, cl_off[i], ncl, bs,
                        mu_off[i], nmu, mto, nmo, bm))
            for k, a in (("vals", br.values), ("y", br.y), ("bias", br.bias), ("tau", br.tau),
                         ("W", br.W.ravel()), ("p_src", br.p_src), ("p_dst", br.p_dst),
                         ("p_flat", br.p_flat), ("p_a", br.p_a), ("p_b", br.p_b), ("p_c", br.p_c),
                         ("p_d", br.p_d), ("elig", br.elig), ("p_mod", br.p_modded),
                         ("md_src", br.md_src), ("md_dst", br.md_dst), ("md_w", br.md_w),
                         ("m", br.m), ("tune", br.tune), ("bind_s", br.bind_s.ravel()),
                         ("motor", br.motor_idx), ("bind_m", br.bind_m.ravel())):
                L[k].append(a)
            L["freq"].append([br.freq])
            L["phase"].append([br.phase])
            v += ni + nd; yo += nd; wo += nd * (ni + nd); po += npl; mo += nmd; so += ns
            bs += ns * ncl; mto += nmo; bm += nmu * nmo

        def cat(k, dtype=float, shape=(0,)):
            return np.concatenate(L[k], dtype=dtype, casting="unsafe") if L[k] else np.zeros(shape, dtype)

        self.off = off = np.array(off, dtype=np.int64).reshape(-1, 18)
        self.vals, self.y, self.bias, self.tau = cat("vals"), cat("y"), cat("bias"), cat("tau")
        self.W = cat("W")
        self.p_src, self.p_dst, self.p_flat = cat("p_src", np.int64), cat("p_dst", np.int64), cat("p_flat", np.int64)
        self.p_a, self.p_b, self.p_c, self.p_d = cat("p_a"), cat("p_b"), cat("p_c"), cat("p_d")
        self.elig, self.p_mod = cat("elig"), cat("p_mod", np.bool_)
        self.md_src, self.md_dst, self.md_w = cat("md_src", np.int64), cat("md_dst", np.int64), cat("md_w")
        self.m = cat("m")
        self.tune = cat("tune", float, (0, 3)).reshape(-1, 3)
        self.bind_s, self.motor, self.bind_m = cat("bind_s"), cat("motor", np.int64), cat("bind_m")
        self.freq, self.phase = cat("freq"), cat("phase")
        self.scratch = np.zeros(len(self.y))
        for br, o in zip(brains, off.tolist()):
            ni, nd = o[O_NIN], o[O_NDYN]
            V, Y, Wo = o[O_V], o[O_Y], o[O_W]
            br.values = self.vals[V:V + ni + nd]
            br.y = self.y[Y:Y + nd]
            br.m = self.m[Y:Y + nd]
            br._Wf = self.W[Wo:Wo + nd * (ni + nd)]
            br.W = br._Wf.reshape(nd, ni + nd)
            br.elig = self.elig[o[O_P0]:o[O_P1]]

    def step(self, energy_frac, ate, age, signals, dt, target):
        """Step every brain one tick and write muscle targets in [0, 1]."""
        step_all(self.off, len(FIXED_INPUTS), dt, energy_frac, ate, age, self.freq, self.phase,
                 signals, self.vals, self.y, self.bias, self.tau, self.W, self.scratch,
                 self.p_src, self.p_dst, self.p_flat, self.p_a, self.p_b, self.p_c, self.p_d,
                 self.elig, self.p_mod, self.md_src, self.md_dst, self.md_w, self.m,
                 self.tune, self.bind_s, self.motor, self.bind_m,
                 HEBB_W_MAX, MOD_MAX, ELIG_TAU, target)


@njit(cache=True, nogil=True)
def step_all(off, n_fixed, dt, energy_frac, ate, age, freq, phase, signals,
             vals, y, bias, tau, W, scratch, p_src, p_dst, p_flat, p_a, p_b, p_c, p_d,
             elig, p_mod, md_src, md_dst, md_w, m, tune, bind_s, motor, bind_m,
             w_max, mod_max, elig_tau, target):
    """Brain.step for every brain in a BrainBatch."""
    for b in range(off.shape[0]):
        ni, nd = off[b, O_NIN], off[b, O_NDYN]
        V, Y, Wo = off[b, O_V], off[b, O_Y], off[b, O_W]
        nt = ni + nd

        # Inputs
        ph = 2.0 * math.pi * freq[b] * age[b] + phase[b]
        vals[V] = energy_frac[b]
        vals[V + 1] = math.sin(ph)
        vals[V + 2] = math.cos(ph)
        vals[V + 3] = ate[b]
        S, CL, ncl, BS = off[b, O_S], off[b, O_CL], off[b, O_NCL], off[b, O_BS]
        for s in range(ni - n_fixed):
            acc = 0.0
            for c in range(ncl):
                rgb = (signals[CL + c, 0] * tune[S + s, 0] + signals[CL + c, 1] * tune[S + s, 1]
                       + signals[CL + c, 2] * tune[S + s, 2])
                acc += bind_s[BS + s * ncl + c] * rgb
            vals[V + n_fixed + s] = math.tanh(acc)

        # CTRNN Euler step
        for i in range(nd):
            vals[V + ni + i] = math.tanh(y[Y + i] + bias[Y + i])
        for i in range(nd):
            acc = 0.0
            row = Wo + i * nt
            for j in range(nt):
                acc += W[row + j] * vals[V + j]
            scratch[Y + i] = acc
        for i in range(nd):
            y[Y + i] += (dt / tau[Y + i]) * (-y[Y + i] + scratch[Y + i])
            vals[V + ni + i] = math.tanh(y[Y + i] + bias[Y + i])

        # Plasticity, with neuromodulation
        P0, P1, MD0, MD1 = off[b, O_P0], off[b, O_P1], off[b, O_MD0], off[b, O_MD1]
        if P1 > P0:
            modded = MD1 > MD0
            if modded:
                for i in range(nd):
                    m[Y + i] = 1.0
                for k in range(MD0, MD1):
                    m[Y + md_dst[k]] += md_w[k] * vals[V + md_src[k]]
                for i in range(nd):
                    m[Y + i] = min(max(m[Y + i], -mod_max), mod_max)
            for k in range(P0, P1):
                pre, post = vals[V + p_src[k]], vals[V + ni + p_dst[k]]
                dw = (p_a[k] * post + p_b[k]) * pre + p_c[k] * post + p_d[k]
                if modded:
                    elig[k] += (dt / elig_tau) * (dw - elig[k])
                    if p_mod[k]:
                        dw = m[Y + p_dst[k]] * elig[k]
                w = W[Wo + p_flat[k]] + dt * dw
                W[Wo + p_flat[k]] = min(max(w, -w_max), w_max)

        # Muscles: binding-weighted blend of motor outputs mapped to [0, 1]
        MU, nmu, MO, nmo, BM = off[b, O_MU], off[b, O_NMU], off[b, O_MO], off[b, O_NMO], off[b, O_BM]
        for u in range(nmu):
            acc = 0.0
            for j in range(nmo):
                acc += bind_m[BM + u * nmo + j] * 0.5 * (vals[V + ni + motor[MO + j]] + 1.0)
            target[MU + u] = min(max(acc, 0.0), 1.0)
