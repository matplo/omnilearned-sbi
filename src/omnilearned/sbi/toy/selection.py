"""Selected-distribution normalization of the surrogate.

The surrogate is trained on selected jets, but its support is unbounded in
the transformed coordinates (the fiducial |eta|, |phi| cuts are enforced by
the atanh transform; jet-level cuts are not). The selected density is

    q_sel(G | theta) = q_raw(G | theta) 1[G in S] / Pr_q(S | theta).

Pr_q(S | theta) is estimated by sampling the surrogate with common random
numbers (identical base noise for every candidate, so differences between
candidates are smooth) and applying the same FastJet selection. It depends
only on the checkpoint, never on observations or truth, and is tabulated once
per checkpoint on a fixed grid and interpolated linearly in log Pr.
"""

import json
import math
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from omnilearned.sbi.toy import simulator, simulator_v2
from omnilearned.sbi.toy.simulator import PARAMETERS, apply_selection


def select_clouds(x_phys, sel, mask=None):
    """Apply the toy selection to physical-coordinate clouds [N, P, 3].

    toy-quenching-v1 (``sel`` a ``simulator.Selection``): fixed 32 particles.
    toy-quenching-v2 (``simulator_v2.Selection``): left-packed constituents
    with an explicit ``mask`` (default: all real), selected by
    ``simulator_v2.select_sets``, the same function the toy applies to its jets.
    """
    x_phys = np.asarray(x_phys, dtype=np.float64)
    if isinstance(sel, simulator_v2.Selection):
        m = np.ones(x_phys.shape[:2], bool) if mask is None else np.asarray(mask, bool)
        return simulator_v2.select_sets(x_phys[..., 0], x_phys[..., 1],
                                        np.where(m, np.exp(np.where(m, x_phys[..., 2], 0.0)), 0.0),
                                        m, sel)
    ev = {"eta": x_phys[..., 0], "phi": x_phys[..., 1], "pt": np.exp(x_phys[..., 2]),
          "valid": np.all(np.isfinite(x_phys), axis=(1, 2))}
    ok, reason, r = apply_selection(ev, sel)
    return ok, reason, r


def acceptance_table(surrogate, sel, axes, n_samples, seed, nsteps=32, solver="midpoint",
                     log=print):
    """Pr_q(S | theta) on the product grid ``axes`` (PARAMETERS order).

    For v2 selections the surrogate draws N from its multiplicity model.
    """
    variable = isinstance(sel, simulator_v2.Selection)
    nreason = len(simulator_v2.REJECTION_REASONS if variable else simulator.REJECTION_REASONS)
    shape = tuple(len(a) for a in axes)
    acc = np.zeros(shape)
    reasons = np.zeros(shape + (nreason,), dtype=np.int64)
    pts = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, len(axes))
    for i, th in enumerate(pts):
        if variable:
            x, mask = surrogate.sample(th, n_samples, None, seed=seed, nsteps=nsteps,
                                       solver=solver, candidate_names=list(PARAMETERS))
        else:
            x = surrogate.sample(th, n_samples, sel.fixed_constituent_count, seed=seed,
                                 nsteps=nsteps, solver=solver, candidate_names=list(PARAMETERS))
            mask = None
        ok, reason, _ = select_clouds(x.astype(np.float64), sel, mask)
        idx = np.unravel_index(i, shape)
        acc[idx] = ok.mean()
        for k in range(nreason):
            reasons[idx + (k,)] = int((reason == k).sum())
        if (i + 1) % max(1, len(pts) // 10) == 0:
            log(f"  acceptance table {i + 1}/{len(pts)}: Pr={acc[idx]:.4f}")
    return acc, reasons


def save_table(path, axes, acc, reasons, meta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, acceptance=acc, reasons=reasons,
             **{f"axis_{k}": a for k, a in enumerate(axes)})
    with open(path.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=1)


class TableLogNorm:
    """log Pr_q(S | theta) interpolated from a saved table."""

    def __init__(self, path):
        d = np.load(path)
        self.axes = [d[f"axis_{k}"] for k in range(len(PARAMETERS))]
        acc = d["acceptance"]
        if np.any(acc <= 0):
            raise ValueError("acceptance table has empty cells")
        self.log_acc = np.log(acc)
        self.interp = RegularGridInterpolator(self.axes, self.log_acc, method="linear",
                                              bounds_error=False, fill_value=None)
        self.description = f"surrogate selection acceptance table {Path(path).name}"

    def __call__(self, points, names=PARAMETERS):
        pts = np.atleast_2d(points)
        if list(names) != list(PARAMETERS):
            order = [list(names).index(p) for p in PARAMETERS]
            pts = pts[:, order]
        return self.interp(pts)


# ---------------------------------------------------------------------------
# per-N selection normalization (surrogate fidelity plan, S1)
# ---------------------------------------------------------------------------
#
# p(N | theta) is trained on the multiplicities of *selected* jets, so the consistent
# selected density is
#
#     p(N | theta) q(x | N, theta) 1[x in S] / Pr_q(S | N, theta),
#
# which keeps p(N | theta) as the selected multiplicity distribution. Dividing by
# Pr_q(S | theta) = sum_N p(N | theta) Pr_q(S | N, theta) instead (``TableLogNorm``)
# reweights N by Pr_q(S | N, theta) / Pr_q(S | theta) whenever the acceptance depends on N.
#
# Pr_q(S | N, theta) = a(N, theta) is estimated from the same common-random-number samples
# as the per-theta table (N drawn from p(N | theta)): per grid node a binomial logistic fit
# with a logit piecewise linear in log N (knots at quantiles of the sampled N, a small
# ridge on the slope changes), the coefficients interpolated linearly in theta. N outside
# the sampled range is evaluated at the nearest sampled N.

def _basis(n, knots):
    t = np.log(np.asarray(n, np.float64))
    return np.stack([np.ones_like(t), t - knots[0]] + [np.maximum(t - k, 0.0) for k in knots[1:]], -1)


def default_knots(generated, n_knots=5):
    """Knots in log N: the lowest sampled N and weighted quantiles of the sampled N."""
    g = np.asarray(generated, np.float64)
    n = np.flatnonzero(g > 0)
    n = n[n > 0]
    cdf = np.cumsum(g[n]) / g[n].sum()
    q = [n[np.searchsorted(cdf, f)] for f in np.linspace(0, 1, n_knots + 1)[1:-1]]
    return np.log(np.unique(np.concatenate([[n.min()], q]).astype(np.float64)))


def fit_logit_in_n(generated, accepted, knots=None, ridge=1e-4, iters=200, tol=1e-10):
    """Binomial logistic fit of the acceptance against N from counts per N.

    generated, accepted: [n_max + 1] counts indexed by N. Returns (coef, knots). A ridge on
    the slope changes and half a pseudo-acceptance (spread over N as the generated clouds)
    keep nodes with few distinct N, or none accepted, finite.
    """
    g, k = np.asarray(generated, np.float64), np.asarray(accepted, np.float64)
    n = np.flatnonzero(g > 0)
    n = n[n > 0]
    if n.size == 0:
        raise ValueError("no generated clouds")
    knots = default_knots(g) if knots is None else np.asarray(knots, np.float64)
    X = _basis(n, knots)
    gn, kn = g[n] + g[n] / g[n].sum(), k[n] + 0.5 * g[n] / g[n].sum()
    p0 = kn.sum() / gn.sum()
    c = np.zeros(X.shape[1])
    c[0] = math.log(p0 / (1 - p0))
    R = np.diag([0.0, 0.0] + [ridge * gn.sum()] * (X.shape[1] - 2))
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ c)))
        grad = X.T @ (kn - gn * p) - R @ c
        H = (X * (gn * p * (1 - p))[:, None]).T @ X + R + 1e-9 * gn.sum() * np.eye(X.shape[1])
        step = np.linalg.solve(H, grad)
        step *= min(1.0, 5.0 / max(np.max(np.abs(step)), 1e-300))  # damped Newton
        c = c + step
        if np.max(np.abs(step)) < tol:
            break
    return c, knots


class PerNLogNorm:
    """log Pr_q(S | N, theta) from a per-N table (``save_per_n_table``).

    ``__call__(points, counts, names)`` -> [M, len(counts)]; ``bind(counts)`` returns the
    per-candidate mean over events, the form ``fit.run_fit``'s ``log_norm_fn`` takes
    (the scan subtracts n_events times it, i.e. the sum over events of log a(N_i, theta)).
    """

    def __init__(self, path):
        d = np.load(path)
        self.axes = [d[f"axis_{k}"] for k in range(len(PARAMETERS))]
        self.coef = d["coef"]  # [*grid, n_basis]
        self.knots = d["knots"]
        self.n_range = (int(d["n_min"]), int(d["n_max"]))
        self.interp = RegularGridInterpolator(self.axes, self.coef, method="linear",
                                              bounds_error=False, fill_value=None)
        self.description = f"surrogate per-N selection acceptance table {Path(path).name}"

    def __call__(self, points, counts, names=PARAMETERS):
        pts = np.atleast_2d(points)
        if list(names) != list(PARAMETERS):
            pts = pts[:, [list(names).index(p) for p in PARAMETERS]]
        c = self.interp(pts)  # [M, n_basis]
        n = np.clip(np.asarray(counts), *self.n_range)
        eta = c @ _basis(n, self.knots).T  # [M, n_events]
        return -np.logaddexp(0.0, -eta)  # log sigmoid

    def bind(self, counts):
        counts = np.asarray(counts)
        uniq, inv = np.unique(counts, return_inverse=True)
        weights = np.bincount(inv) / counts.size

        def fn(points, names=PARAMETERS):
            return self(points, uniq, names) @ weights

        fn.description = f"{self.description}, bound to {counts.size} events"
        return fn


def save_per_n_table(path, axes, generated, accepted, meta, n_knots=5):
    """Fit and save the per-N table from per-node counts [*grid, n_max + 1]."""
    generated, accepted = np.asarray(generated), np.asarray(accepted)
    shape = generated.shape[:-1]
    tot = generated.reshape(-1, generated.shape[-1]).sum(0)
    knots = default_knots(tot, n_knots)  # common knots: coefficients interpolate in theta
    n = np.flatnonzero(tot > 0)
    n = n[n > 0]
    coef = np.zeros(shape + (len(knots) + 1,))
    for idx in np.ndindex(*shape):
        coef[idx] = fit_logit_in_n(generated[idx], accepted[idx], knots)[0]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, coef=coef, knots=knots, n_min=int(n.min()), n_max=int(n.max()),
             generated=generated, accepted=accepted, **{f"axis_{k}": a for k, a in enumerate(axes)})
    with open(path.with_suffix(".json"), "w") as f:
        json.dump({**meta, "knots_N": np.exp(knots).round(3).tolist(), "n_range": [int(n.min()), int(n.max())]},
                  f, indent=1)
    return coef
