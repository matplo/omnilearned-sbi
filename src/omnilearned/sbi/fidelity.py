"""Statistics-aware fidelity gate: direct toy vs surrogate samples.

A setting fails only if a discrepancy is both

* significant: family-wise permutation p-value (jets of the two samples
  pooled and re-split; max KS D over observables per permutation, so
  correlated observables need no independence assumption; below 5
  exceedances the tail is extrapolated with a Gumbel fit to the permutation
  maxima) below ``alpha / S`` for S settings (Bonferroni over settings), and
* relevant: some observable mean is off by more than ``rel_tol`` standard
  errors at the analysis event count N (|Δμ| √N / σ_direct), with the
  2-standard-error lower bound of |Δμ| still above that tolerance.

Statistical jitter therefore cannot fail a model: an insignificant
difference passes whatever its size, and a significant but tiny one passes
too. Across settings the family-wise p-values should look uniform; the
global check reports a KS test against uniformity and the count below 0.01
against Binomial(S, 0.01).

``equivalent_shift`` turns observable-mean shifts into a parameter shift
δθ = (GᵀC⁻¹G)⁻¹GᵀC⁻¹Δμ with direct-toy response gradients G and per-jet
covariance C, to compare with posterior widths.
"""

import math

import numpy as np

from omnilearned.sbi.toy.observables import jet_observables

JET = ["jet_pt", "width", "soft8_fraction", "outer_fraction", "ptd", "lead_fraction"]
JET_VARIABLE = JET + ["n_constituents"]  # variable-multiplicity clouds (toy v2)
PARTICLE = ["eta", "phi", "logpt", "dr", "logz"]


def particle_obs(x, mask=None):
    """Per-jet arrays [n, P] of constituent observables (NaN where padded)."""
    x = np.asarray(x, dtype=np.float64)
    m = np.ones(x.shape[:2], bool) if mask is None else np.asarray(mask, bool)
    eta, phi = x[..., 0], x[..., 1]
    pt = np.where(m, np.exp(np.where(m, x[..., 2], 0.0)), 0.0)
    px, py, pz = (pt * np.cos(phi)).sum(1), (pt * np.sin(phi)).sum(1), (pt * np.sinh(eta)).sum(1)
    jpt = np.hypot(px, py)
    jeta, jphi = np.arcsinh(pz / jpt), np.arctan2(py, px)
    dr = np.hypot(eta - jeta[:, None], np.angle(np.exp(1j * (phi - jphi[:, None]))))
    out = {"eta": eta, "phi": phi, "logpt": x[..., 2], "dr": dr,
           "logz": np.log(np.where(m, pt, 1.0) / jpt[:, None])}
    return {k: np.where(m, v, np.nan) for k, v in out.items()}


def all_obs(x, mask=None):
    keys = JET if mask is None else JET_VARIABLE
    o = {k: v for k, v in jet_observables(x, mask).items() if k in keys}
    o.update(particle_obs(x, mask))
    return o


def ks_d(a, b):
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]  # padded constituents are NaN
    a, b = np.sort(a), np.sort(b)
    grid = np.concatenate([a, b])
    return float(np.abs(np.searchsorted(a, grid, side="right") / a.size
                        - np.searchsorted(b, grid, side="right") / b.size).max())


def permutation_test(oa, ob, keys, n_perm=500, seed=0, particle_stride=4):
    """KS D per observable with permutation p-values and a family-wise p-value.

    ``oa``/``ob``: dicts of per-jet arrays ([n] jet level, [n, P] particle
    level, NaN where padded). Jets are permuted as units, so constituents
    stay together.
    Particle observables use every ``particle_stride``-th jet (speed).
    """
    rng = np.random.default_rng(seed)
    na = len(oa[keys[0]])
    pooled = {k: _concat_padded(oa[k], ob[k]) for k in keys}
    ntot = len(pooled[keys[0]])

    def stats(ia, ib):
        out = []
        for k in keys:
            v = pooled[k]
            if v.ndim == 2:
                out.append(ks_d(v[ia[::particle_stride]].ravel(), v[ib[::particle_stride]].ravel()))
            else:
                out.append(ks_d(v[ia], v[ib]))
        return np.array(out)

    idx = np.arange(ntot)
    obs = stats(idx[:na], idx[na:])
    null = np.empty((n_perm, len(keys)))
    for r in range(n_perm):
        p = rng.permutation(ntot)
        null[r] = stats(p[:na], p[na:])
    # Standardize each observable by its null median so the max is comparable.
    scale = np.median(null, 0)
    scale = np.where(scale > 0, scale, 1.0)
    p_each = (1 + (null >= obs).sum(0)) / (n_perm + 1)
    mx, m = (null / scale).max(1), float((obs / scale).max())
    exceed = int((mx >= m).sum())
    p_perm = (1 + exceed) / (n_perm + 1)
    # Fewer than 5 exceedances: extrapolate with a Gumbel fit (moments) to the
    # permutation maxima, so p-values below 1/(n_perm+1) are resolved.
    beta = float(mx.std(ddof=1)) * math.sqrt(6) / math.pi
    mu = float(mx.mean()) - 0.5772156649 * beta
    p_tail = float(-math.expm1(-math.exp(-(m - mu) / beta))) if beta > 0 else p_perm
    p_family = p_perm if exceed >= 5 else min(p_perm, p_tail)
    return {"ks_d": dict(zip(keys, obs.tolist())), "p": dict(zip(keys, p_each.tolist())),
            "p_family": float(p_family), "p_perm": float(p_perm), "p_tail": p_tail,
            "p_method": "permutation" if exceed >= 5 else "gumbel_tail", "n_perm": n_perm}


def _concat_padded(a, b):
    """Concatenate per-jet arrays; particle arrays of different widths are NaN-padded."""
    if a.ndim == 2 and a.shape[1] != b.shape[1]:
        w = max(a.shape[1], b.shape[1])
        pad = lambda v: np.pad(v, ((0, 0), (0, w - v.shape[1])), constant_values=np.nan)  # noqa: E731
        a, b = pad(a), pad(b)
    return np.concatenate([a, b])


def mean_relevance(oa, ob, keys, n_events, rel_tol=0.5):
    """Mean shift per jet observable in standard errors at ``n_events`` jets."""
    out = {}
    for k in keys:
        a, b = oa[k], ob[k]
        dmu = float(b.mean() - a.mean())
        se = float(math.hypot(a.std() / math.sqrt(a.size), b.std() / math.sqrt(b.size)))
        unit = float(a.std() / math.sqrt(n_events))
        out[k] = {"delta_mean": dmu, "se": se, "z": dmu / se if se > 0 else 0.0,
                  "relevance": abs(dmu) / unit, "relevance_lower": max(0.0, abs(dmu) - 2 * se) / unit,
                  "relevant": bool(max(0.0, abs(dmu) - 2 * se) / unit > rel_tol)}
    return out


def setting_verdict(perm, rel, alpha_setting):
    significant = perm["p_family"] < alpha_setting
    relevant = any(v["relevant"] for v in rel.values())
    return {"significant": bool(significant), "relevant": bool(relevant),
            "fail": bool(significant and relevant)}


def global_uniformity(pvals):
    """KS distance of p-values from Uniform(0,1) and the count below 0.01."""
    p = np.sort(np.asarray(pvals, dtype=np.float64))
    s = p.size
    d = float(np.max(np.maximum(np.arange(1, s + 1) / s - p, p - np.arange(s) / s)))
    lam = (math.sqrt(s) + 0.12 + 0.11 / math.sqrt(s)) * d
    k = np.arange(1, 101)
    ks_p = float(np.clip(2 * np.sum((-1) ** (k - 1) * np.exp(-2 * k ** 2 * lam ** 2)), 0, 1))
    n_low = int((p < 0.01).sum())
    binom_p = float(sum(math.comb(s, j) * 0.01 ** j * 0.99 ** (s - j) for j in range(n_low, s + 1)))
    return {"settings": s, "ks_d": d, "ks_p": ks_p, "n_below_0.01": n_low,
            "binomial_p_at_least": binom_p}


def equivalent_shift(G, C, dmu, cov_dmu):
    """δθ = (GᵀC⁻¹G)⁻¹GᵀC⁻¹Δμ and its covariance from cov(Δμ).

    G: [K, P] response gradients d mean_k / d theta_p; C: [K, K] per-jet
    covariance of the observables; dmu: [K]; cov_dmu: [K, K].
    """
    G, C = np.asarray(G, float), np.asarray(C, float)
    Ci = np.linalg.pinv(C)
    A = np.linalg.pinv(G.T @ Ci @ G) @ G.T @ Ci
    return A @ np.asarray(dmu, float), A @ np.asarray(cov_dmu, float) @ A.T


def evaluate(direct, surrogate, n_events=1000, n_perm=500, seed=0, n_settings=1,
             alpha=0.05, rel_tol=0.5, direct_mask=None, surrogate_mask=None):
    """Gate one setting. With masks (variable multiplicity) the multiplicity
    is an additional jet observable; pass both masks or neither."""
    if (direct_mask is None) != (surrogate_mask is None):
        raise ValueError("pass masks for both samples or for neither")
    oa, ob = all_obs(direct, direct_mask), all_obs(surrogate, surrogate_mask)
    jet = JET if direct_mask is None else JET_VARIABLE
    perm = permutation_test(oa, ob, jet + PARTICLE, n_perm=n_perm, seed=seed)
    rel = mean_relevance(oa, ob, jet, n_events, rel_tol)
    return {"permutation": perm, "relevance": rel, "n_events": n_events, "rel_tol": rel_tol,
            "alpha_setting": alpha / n_settings,
            "verdict": setting_verdict(perm, rel, alpha / n_settings)}
