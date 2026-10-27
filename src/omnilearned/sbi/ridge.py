"""Ridge-following lattice scans for long, thin, curved posteriors.

At large N the posterior can be a ridge: long along one parameter and only a fraction of
a grid step wide across it (toy v2 at N = 1000: b spans ≈ 4–10 fm while the conditional
widths are ≈ 0.1 fm in b and 0.0025 GeV in T_ref). An axis-aligned grid fine enough to
resolve the width is mostly empty. Here the parameters are written along one long axis
``s`` (a parameter, e.g. b) and whitened cross coordinates ``u``:

    θ(s, u):  θ_axis = s,   θ_cross = m(s) + L(s) u

with the ridge centre m(s) and a lower-triangular L(s) (positive diagonal) interpolated
linearly in s between knots. The Jacobian is block-triangular, so
|∂θ / ∂(s, u)| = det L(s) exactly, whatever m'(s) and L'(s) are.

The scan is :func:`~omnilearned.sbi.scan.support_scan` on an axis-aligned lattice in
(s, u), with log prior log p(θ(s, u)) + log det L(s). Lattice cell masses are therefore
exact posterior masses in θ. The map only decides where nodes go: a poorly placed map
costs nodes, not accuracy, as long as the support closes inside the lattice (checked).

* :func:`locate_ridge` places the knots from a coarse scan of θ (cheap numerics): per
  knot, quadratic fits on a 3^(d-1) stencil in the cross coordinates, marching outward
  from the coarse maximum.
* :func:`ridge_scan` runs the lattice scan and returns a :class:`RidgeResult`, whose
  summary, joint-level inputs and save/load match :class:`~omnilearned.sbi.scan.ScanResult`
  (load either with :func:`load_result`).

Summaries in θ: means, standard deviations and correlations are mass-weighted node
moments. The long-axis marginal is the lattice marginal (trapezoid in s). A cross
parameter's marginal CDF is the sum over nodes of each node's mass spread over the
projection of its cell, approximated by a normal of the cell's projected variance
((Δs ∂θ_k/∂s)² + h² Σ_j L_kj²) / 12. The highest-density ordering (joint level) uses the
θ density, not the lattice density.
"""

import json
import math

import numpy as np
from scipy.special import ndtr

from omnilearned.sbi.scan import (
    Grid,
    ScanResult,
    dilate,
    equal_tailed_interval,
    support_scan,
)


class RidgeMap:
    """θ(s, u) = (s on the long axis, m(s) + L(s) u across), linear in s between knots."""

    def __init__(self, names, axis, knots, centers, chols):
        self.names = list(names)
        self.axis = axis
        self.k = self.names.index(axis)
        self.cross = [i for i in range(len(self.names)) if i != self.k]
        self.knots = np.asarray(knots, dtype=np.float64)
        self.centers = np.asarray(centers, dtype=np.float64).reshape(len(self.knots), len(self.cross))
        self.chols = np.asarray(chols, dtype=np.float64).reshape(len(self.knots), len(self.cross),
                                                                  len(self.cross))
        if np.any(np.diff(self.knots) <= 0):
            raise ValueError("knots must be strictly increasing")
        if np.any(np.diagonal(self.chols, axis1=1, axis2=2) <= 0):
            raise ValueError("cholesky factors need a positive diagonal")
        if np.any(np.triu(self.chols, 1) != 0):
            raise ValueError("cholesky factors must be lower triangular")

    def _interp(self, s):
        s = np.asarray(s, dtype=np.float64).reshape(-1)
        if len(self.knots) == 1:
            return (np.repeat(self.centers, len(s), 0), np.repeat(self.chols, len(s), 0))
        i = np.clip(np.searchsorted(self.knots, s) - 1, 0, len(self.knots) - 2)
        t = np.clip((s - self.knots[i]) / (self.knots[i + 1] - self.knots[i]), 0.0, 1.0)
        m = (1 - t)[:, None] * self.centers[i] + t[:, None] * self.centers[i + 1]
        L = (1 - t)[:, None, None] * self.chols[i] + t[:, None, None] * self.chols[i + 1]
        return m, L

    def to_theta(self, s, u):
        s = np.asarray(s, dtype=np.float64).reshape(-1)
        u = np.asarray(u, dtype=np.float64).reshape(len(s), len(self.cross))
        m, L = self._interp(s)
        th = np.empty((len(s), len(self.names)))
        th[:, self.k] = s
        th[:, self.cross] = m + np.einsum("nij,nj->ni", L, u)
        return th

    def to_lattice(self, theta):
        """Inverse map: θ [n, d] -> (s [n], u [n, d-1])."""
        th = np.atleast_2d(np.asarray(theta, dtype=np.float64))
        s = th[:, self.k]
        m, L = self._interp(s)
        u = np.linalg.solve(L, (th[:, self.cross] - m)[..., None])[..., 0]
        return s, u

    def log_jacobian(self, s):
        _, L = self._interp(s)
        return np.log(np.diagonal(L, axis1=1, axis2=2)).sum(1)

    def cross_cell_sd(self, s, u, ds, h):
        """Standard deviation of each cross parameter over a lattice cell (uniform cell)."""
        e = 1e-6 * max(1.0, float(np.ptp(self.knots)) if len(self.knots) > 1 else 1.0)
        dth = (self.to_theta(s + e, u) - self.to_theta(s - e, u))[:, self.cross] / (2 * e)
        _, L = self._interp(s)
        var = (ds * dth) ** 2 + h ** 2 * (L ** 2).sum(2)
        return np.sqrt(var / 12.0)

    def to_dict(self):
        return {"names": self.names, "axis": self.axis, "knots": self.knots.tolist(),
                "centers": self.centers.tolist(), "chols": self.chols.tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(d["names"], d["axis"], d["knots"], d["centers"], d["chols"])


def lattice_grid(ridge_map, bounds, step, h, u_max):
    """(s, u) lattice: s over the long axis's prior range at ``step``; u in [-u_max, u_max] at ``h``."""
    lo, hi = bounds[ridge_map.k]
    n_s = int(round((hi - lo) / step)) + 1
    n_u = 2 * int(round(u_max / h)) + 1
    u_ax = np.linspace(-h * (n_u // 2), h * (n_u // 2), n_u)
    names = [ridge_map.axis] + [f"u_{ridge_map.names[i]}" for i in ridge_map.cross]
    return Grid(names, [np.linspace(lo, hi, n_s)] + [u_ax] * len(ridge_map.cross))


# ---------------------------------------------------------------------------
# locating the ridge
# ---------------------------------------------------------------------------


def _quadratic_fit(U, y):
    """Least-squares quadratic y ≈ c + gᵀu + ½ uᵀHu; returns (c, g, H)."""
    n, d = U.shape
    cols, pairs = [np.ones(n)], []
    cols += [U[:, i] for i in range(d)]
    for i in range(d):
        for j in range(i, d):
            cols.append(U[:, i] * U[:, j])
            pairs.append((i, j))
    coef = np.linalg.lstsq(np.stack(cols, 1), y, rcond=None)[0]
    g = coef[1:1 + d]
    H = np.zeros((d, d))
    for (i, j), c in zip(pairs, coef[1 + d:]):
        if i == j:
            H[i, i] = 2 * c
        else:
            H[i, j] = H[j, i] = c
    return coef[0], g, H


def refine_knot(total_fn, ridge_names, k, cross, s, center, chol, box, iterations=6,
                stencil=1.5, trust=3.0, tol=0.25):
    """Centre and cross-section metric of the ridge at long-axis value ``s``.

    ``total_fn(theta[M, d]) -> [M]`` log posterior (cheap numerics allowed). Each iteration
    evaluates a 3^(d-1) stencil at ±``stencil`` in the current whitened coordinates, fits a
    quadratic, moves to its maximum (at most ``trust`` units) and sets L from the fitted
    curvature. Stops when the move is below ``tol`` units. Points are kept inside ``box``
    (cross bounds, already widened by any margin).
    """
    dc = len(cross)
    grid1 = np.array([-1.0, 0.0, 1.0]) * stencil
    U = np.stack(np.meshgrid(*[grid1] * dc, indexing="ij"), -1).reshape(-1, dc)
    c, L = np.asarray(center, float).copy(), np.asarray(chol, float).copy()
    lo, hi = np.array([b[0] for b in box]), np.array([b[1] for b in box])
    info = {"iterations": 0, "converged": False, "evaluations": 0, "fallbacks": 0}
    best = (-np.inf, c.copy())
    for it in range(iterations):
        X = np.clip(c + U @ L.T, lo, hi)
        th = np.empty((len(X), len(ridge_names)))
        th[:, k] = s
        th[:, cross] = X
        y = np.asarray(total_fn(th), dtype=np.float64)
        info["evaluations"] += len(y)
        info["iterations"] = it + 1
        ok = np.isfinite(y)
        if ok.any() and np.nanmax(y) > best[0]:
            best = (float(np.nanmax(y)), X[np.nanargmax(y)].copy())
        Uf = np.linalg.solve(L, (X - c).T).T  # clipped stencil in whitened units
        if ok.sum() < len(y) or ok.sum() < (dc + 1) * (dc + 2) // 2:
            info["fallbacks"] += 1
            c, L = best[1].copy(), 0.5 * L
            continue
        _, g, H = _quadratic_fit(Uf, y)
        ev = np.linalg.eigvalsh(H)
        if ev.max() >= 0:
            info["fallbacks"] += 1
            c, L = X[np.argmax(y)].copy(), 0.5 * L
            continue
        du = -np.linalg.solve(H, g)
        nrm = float(np.linalg.norm(du))
        if nrm > trust:
            du *= trust / nrm
        c = np.clip(c + L @ du, lo, hi)
        L = L @ np.linalg.cholesky(np.linalg.inv(-H))
        if nrm < tol:
            info["converged"] = True
            break
    info["best_logpost"] = best[0]
    info["at_box"] = bool(np.any(np.isclose(c, lo) | np.isclose(c, hi)))
    info["converged"] = info["converged"] and not info["at_box"]
    return c, L, info


def locate_ridge(total_fn, coarse, axis, bounds, knot_step=0.5, delta_knots=15.0,
                 iterations=6, stencil=1.5, margin=0.25, chord_tol=0.25, max_knots=200, log=print):
    """Knots of a :class:`RidgeMap` from a coarse θ scan (a :class:`ScanResult`).

    The coarse scan only supplies the start: the coarse maximum, with half the coarse
    cross spacing as the first metric (a coarse grid need not connect a strongly curved
    ridge). From there knots march along the ridge itself, each started from the linear
    extrapolation of the two nearest knots and refined with :func:`refine_knot`. The step
    starts at ``knot_step / 8``, doubles after each converged knot up to ``knot_step`` and
    halves when a knot does not converge; a knot that fails at ``knot_step / 64`` ends that
    direction. A direction also ends after the first knot whose best log posterior is
    ``delta_knots`` below the best of all knots, at the long-axis prior bound, or where
    the centre reaches the cross box (the prior widened by ``margin``); it resumes if a
    later knot raises the best. Then each knot interval is bisected: the midpoint knot is
    refined from the interpolated centre and kept, and both halves are bisected again while
    the refined centre is more than ``chord_tol`` cross-section units from the interpolated
    one (the piecewise-linear centre line of a curved ridge cuts the chord). Intervals with
    an end at the cross box, or with both ends ``delta_knots`` below the best, are not
    bisected. Uses only the scan (never truth).
    """
    names = coarse.grid.names
    k = names.index(axis)
    cross = [i for i in range(len(names)) if i != k]
    lp = coarse.log_posterior().reshape(-1)
    pts = coarse.grid.points()
    i_map = int(np.nanargmax(lp))
    s_lo, s_hi = bounds[k]
    box = []
    for i in cross:
        w = bounds[i][1] - bounds[i][0]
        box.append((bounds[i][0] - margin * w, bounds[i][1] + margin * w))
    data = {}

    def run(s, c, L):
        c, L, info = refine_knot(total_fn, names, k, cross, s, c, L, box,
                                 iterations=iterations, stencil=stencil)
        log(f"  knot {axis}={s:.4f}: centre {np.round(c, 5).tolist()} "
            f"sd {np.round(np.sqrt((L ** 2).sum(1)), 5).tolist()} log post {info['best_logpost']:.2f} "
            f"({info['evaluations']} evaluations, converged={info['converged']})")
        return c, L, info

    L0 = np.diag([0.5 * float(np.mean(np.diff(coarse.grid.axes[i]))) for i in cross])
    s0 = float(pts[i_map, k])
    data[s0] = run(s0, pts[i_map, cross].astype(float), L0)
    step = {1: knot_step / 8, -1: knot_step / 8}
    ended = {1: None, -1: None}
    extended = True
    while extended and len(data) < max_knots:
        extended = False
        for dirn in (1, -1):
            while len(data) < max_knots and ended[dirn] in (None, "logpost"):
                srt = sorted(data)
                end = srt[-1] if dirn > 0 else srt[0]
                top = max(v[2]["best_logpost"] for v in data.values())
                if (dirn > 0 and end >= s_hi) or (dirn < 0 and end <= s_lo):
                    ended[dirn] = "bound"
                    break
                if data[end][2]["best_logpost"] < top - delta_knots:
                    ended[dirn] = "logpost"
                    break
                s_new = min(s_hi, end + step[dirn]) if dirn > 0 else max(s_lo, end - step[dirn])
                c, L = data[end][0], data[end][1]
                if len(srt) > 1:
                    nxt = srt[-2] if dirn > 0 else srt[1]
                    c = c + (c - data[nxt][0]) * (s_new - end) / (end - nxt)
                c, L, info = run(s_new, c, L)
                if not (info["converged"] or info["at_box"]) and step[dirn] > knot_step / 64 * (1 + 1e-9):
                    step[dirn] /= 2
                    continue
                data[s_new] = (c, L, info)
                extended = True
                if info["at_box"]:
                    ended[dirn] = "cross box"
                elif not info["converged"]:
                    ended[dirn] = "not converged"
                else:
                    step[dirn] = min(knot_step, 2 * step[dirn])
            if ended[dirn] == "logpost":
                # resume if the best has risen since this direction stopped
                srt = sorted(data)
                end = srt[-1] if dirn > 0 else srt[0]
                top = max(v[2]["best_logpost"] for v in data.values())
                if data[end][2]["best_logpost"] >= top - delta_knots:
                    ended[dirn] = None
                    extended = True
    top = max(v[2]["best_logpost"] for v in data.values())
    min_dx = knot_step / 64 * (1 + 1e-9)
    n_bisect = 0
    todo = list(zip(sorted(data)[:-1], sorted(data)[1:]))
    while todo and len(data) < max_knots:
        a, b = todo.pop()
        if (data[a][2]["at_box"] or data[b][2]["at_box"] or b - a < 2 * min_dx
                or max(data[a][2]["best_logpost"], data[b][2]["best_logpost"]) < top - delta_knots):
            continue
        sm = 0.5 * (a + b)
        cm, Lm = 0.5 * (data[a][0] + data[b][0]), 0.5 * (data[a][1] + data[b][1])
        c, L, info = run(sm, cm, Lm)
        info["chord_error"] = float(np.linalg.norm(np.linalg.solve(L, c - cm)))
        data[sm] = (c, L, info)
        n_bisect += 1
        if info["chord_error"] > chord_tol:
            todo += [(a, sm), (sm, b)]
    knots = np.array(sorted(data))
    infos = [data[s][2] for s in knots]
    rmap = RidgeMap(names, axis, knots, [data[s][0] for s in knots], [data[s][1] for s in knots])
    return rmap, {"knots": knots.tolist(), "knot_info": infos,
                  "coarse_max_logpost": float(np.nanmax(lp)), "coarse_map": pts[i_map].tolist(),
                  "delta_knots": float(delta_knots), "ended": {"up": ended[1], "down": ended[-1]},
                  "chord_tol": float(chord_tol), "bisections": n_bisect,
                  "max_chord_error": max([i.get("chord_error", 0.0) for i in infos]),
                  "evaluations": int(sum(i["evaluations"] for i in infos))}


def _uniform_log_prior_theta(theta, bounds):
    inside = np.ones(len(theta), dtype=bool)
    vol = 1.0
    for i, (lo, hi) in enumerate(bounds):
        inside &= (theta[:, i] >= lo) & (theta[:, i] <= hi)
        vol *= (hi - lo) if hi > lo else 1.0
    return np.where(inside, -math.log(vol), -np.inf)


def ridge_scan(logp_fn, n_events, ridge_map, bounds, step, h, u_max=6.0, delta_logpost=10.0,
               event_chunk=256, candidate_chunk=8, log_norm_fn=None, keep_per_event=True,
               progress=None, max_seconds=None, coarse=None):
    """Support scan on the (s, u) lattice of ``ridge_map``; returns a :class:`RidgeResult`.

    ``logp_fn(event_slice, theta[m, d])`` and ``log_norm_fn(theta[m, d])`` take θ in
    ``ridge_map.names`` order; ``bounds`` is the uniform prior box in that order. Seeds are
    the ridge centre line (u = 0). The support is closed only if the scan closed, no
    supported node lies on the u edge of the lattice, and (with ``coarse``, the locate
    scan) no coarse node within ``delta_logpost`` of the coarse maximum maps outside the
    lattice (a mode the map missed).
    """
    lat = lattice_grid(ridge_map, bounds, step, h, u_max)
    pts = lat.points()
    theta = ridge_map.to_theta(pts[:, 0], pts[:, 1:])
    log_prior = _uniform_log_prior_theta(theta, bounds) + ridge_map.log_jacobian(pts[:, 0])

    def fn(sl, cand):
        return logp_fn(sl, ridge_map.to_theta(cand[:, 0], cand[:, 1:]))

    norm = None
    if log_norm_fn is not None:
        def norm(cand):
            return log_norm_fn(ridge_map.to_theta(cand[:, 0], cand[:, 1:]))

    seed = np.zeros(lat.shape, dtype=bool)
    seed[(slice(None),) + tuple(len(a) // 2 for a in lat.axes[1:])] = True
    res = support_scan(fn, n_events, lat, None, delta_logpost=delta_logpost,
                       event_chunk=event_chunk, candidate_chunk=candidate_chunk,
                       log_prior=log_prior.reshape(lat.shape), log_norm_fn=norm,
                       keep_per_event=keep_per_event, progress=progress,
                       max_seconds=max_seconds, seed=seed, skip_outside_prior=True)
    out = RidgeResult(res, ridge_map, bounds, step, h, u_max)
    d = res.diagnostics
    lp = res.log_posterior()
    active = res.evaluated & np.isfinite(lp) & (lp >= np.nanmax(lp) - delta_logpost)
    edge = np.zeros(lat.shape, dtype=bool)
    for ax in range(1, len(lat.axes)):
        sl = [slice(None)] * len(lat.axes)
        for e in (0, -1):
            sl[ax] = e
            edge[tuple(sl)] = True
    d["supported_on_u_edge"] = int((active & edge).sum())
    d["u_edge_mass"] = float(out.posterior_mass()[edge].sum())
    d["coarse_support_outside_lattice"] = None
    if coarse is not None:
        clp = coarse.log_posterior().reshape(-1)
        csup = np.isfinite(clp) & (clp >= np.nanmax(clp) - delta_logpost)
        _, cu = ridge_map.to_lattice(coarse.grid.points()[csup])
        d["coarse_support_outside_lattice"] = int((np.abs(cu) > u_max).any(1).sum())
    d["scan_closed"] = d["support_closed"]
    d["support_closed"] = bool(d["scan_closed"] and d["supported_on_u_edge"] == 0
                               and not d["coarse_support_outside_lattice"])
    d["scan"] = "ridge"
    return out


class RidgeResult:
    """A support scan on a ridge lattice, summarized in θ (see the module docstring)."""

    def __init__(self, scan, ridge_map, bounds, step, h, u_max):
        self.scan, self.map = scan, ridge_map
        self.bounds = [tuple(map(float, b)) for b in bounds]
        self.step, self.h, self.u_max = float(step), float(h), float(u_max)

    # ScanResult-compatible attributes used by drivers and reports
    @property
    def evaluated(self):
        return self.scan.evaluated

    @property
    def n_events(self):
        return self.scan.n_events

    @property
    def diagnostics(self):
        return self.scan.diagnostics

    def parameter_names(self):
        return list(self.map.names)

    def fixed_names(self):
        return []

    def theta_points(self):
        pts = self.scan.grid.points()
        return self.map.to_theta(pts[:, 0], pts[:, 1:]).reshape(self.scan.grid.shape + (-1,))

    def log_posterior(self):
        """Unnormalized log posterior density in θ at every lattice node (nan if not evaluated)."""
        s = self.scan.grid.points()[:, 0]
        return self.scan.log_posterior() - self.map.log_jacobian(s).reshape(self.scan.grid.shape)

    def posterior_mass(self):
        return self.scan.posterior_mass()

    def _cross_interval(self, i, level, mass, th, sd):
        keep = mass > 1e-14 * mass.max()
        v, w, s = th[keep, i], mass[keep], np.maximum(sd[keep], 1e-300)
        t = np.linspace((v - 6 * s).min(), (v + 6 * s).max(), 4001)
        F = np.zeros_like(t)
        for c0 in range(0, len(v), 512):
            F += (w[c0:c0 + 512, None] * ndtr((t[None] - v[c0:c0 + 512, None]) / s[c0:c0 + 512, None])).sum(0)
        F /= F[-1]
        return [float(np.interp(0.5 * (1 - level), F, t)), float(np.interp(0.5 * (1 + level), F, t))]

    def summary(self, levels=(0.68, 0.95)):
        names = self.map.names
        d = len(names)
        th = self.theta_points().reshape(-1, d)
        mass = self.posterior_mass().reshape(-1)
        mean = mass @ th
        cov = (th - mean).T @ ((th - mean) * mass[:, None])
        std = np.sqrt(np.clip(np.diag(cov), 0, None))
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = cov / np.outer(std, std)
        ll = self.scan.loglik.reshape(-1)
        i_mle = int(np.nanargmax(ll))
        pts = self.scan.grid.points()
        csd = np.zeros((len(th), d))
        csd[:, self.map.cross] = self.map.cross_cell_sd(pts[:, 0], pts[:, 1:], self.step, self.h)
        out = {"mode": self.scan.mode, "n_events": self.scan.n_events, "parameters": names,
               "mle": {n: float(th[i_mle, k]) for k, n in enumerate(names)},
               "max_loglik": float(np.nanmax(ll)),
               "posterior_mean": dict(zip(names, mean.tolist())),
               "posterior_std": dict(zip(names, std.tolist())),
               "posterior_correlation": corr.tolist(),
               "credible_intervals": {}, "profile_intervals": {}, "boundary": {},
               "lattice": {"axis": self.map.axis, "step": self.step, "h": self.h, "u_max": self.u_max,
                           "shape": list(self.scan.grid.shape)}}
        shape = self.scan.grid.shape
        for k, n in enumerate(names):
            if k == self.map.k:
                axis, dens, _ = self.scan.marginal(self.scan.grid.names[0])
                out["credible_intervals"][n] = {str(lv): equal_tailed_interval(axis, dens, lv) for lv in levels}
            else:
                out["credible_intervals"][n] = {str(lv): self._cross_interval(k, lv, mass, th, csd[:, k])
                                                for lv in levels}
            # nodes next to the prior bound of this parameter (or at the lattice end in s)
            lo, hi = self.bounds[k]
            outside = ((th[:, k] < lo) | (th[:, k] > hi)).reshape(shape)
            near = dilate(outside) & ~outside
            if k == self.map.k:
                near = np.zeros(shape, dtype=bool)
                near[0], near[-1] = True, True
            out["boundary"][n] = {"mle_on_edge": bool(near.reshape(-1)[i_mle]),
                                  "edge_node_mass": float(mass[near.reshape(-1)].sum())}
        return out

    def save(self, path_prefix, extra=None):
        doc_extra = {"ridge": {"map": self.map.to_dict(), "bounds": self.bounds, "step": self.step,
                               "h": self.h, "u_max": self.u_max},
                     "lattice_summary": self.scan.summary(),
                     "integration_measure": "lattice cell masses in (s, u) with log det L(s): "
                                            "posterior masses in theta",
                     "summary": self.summary()}
        doc_extra.update(extra or {})
        return self.scan.save(path_prefix, extra=doc_extra)

    @classmethod
    def load(cls, path_prefix):
        with open(f"{path_prefix}.json") as f:
            doc = json.load(f)
        scan = ScanResult.load(path_prefix)
        r = doc["ridge"]
        return cls(scan, RidgeMap.from_dict(r["map"]), r["bounds"], r["step"], r["h"], r["u_max"])


def load_result(path_prefix):
    """A saved fit result: :class:`RidgeResult` if it was a ridge scan, else :class:`ScanResult`."""
    with open(f"{path_prefix}.json") as f:
        doc = json.load(f)
    return RidgeResult.load(path_prefix) if "ridge" in doc else ScanResult.load(path_prefix)


__all__ = ["RidgeMap", "RidgeResult", "lattice_grid", "locate_ridge", "load_result", "refine_knot",
           "ridge_scan"]
