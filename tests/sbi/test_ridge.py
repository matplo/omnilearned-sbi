"""Ridge-following lattice scans (omnilearned.sbi.ridge) on analytic curved ridges."""

import numpy as np
import pytest
import torch
from scipy import stats

from omnilearned.sbi.ridge import (
    RidgeMap,
    RidgeResult,
    lattice_grid,
    load_result,
    locate_ridge,
    ridge_scan,
)
from omnilearned.sbi.scan import Grid, joint_scan, support_scan, uniform_log_prior

# log p = -1/2 x^2 - 1/2 ((y - A x^2) / SY)^2 - 1/2 ((z - C x) / SZ)^2: a thin curved tube
# along x. (x, y, z) -> (x, (y - A x^2)/SY, (z - C x)/SZ) has a constant Jacobian, so
# x ~ N(0, 1), y = A x^2 + N(0, SY^2), z = C x + N(0, SZ^2), and the highest-density
# level of a point is the chi2(3) CDF of its squared distance in those coordinates.
A, C, SY, SZ = 0.3, 0.5, 0.01, 0.05
BOUNDS = [(-6.0, 6.0), (-0.2, 2.9), (-3.4, 3.4)]


def tube_logp(theta, dy=0.0):
    x, y, z = np.atleast_2d(theta).T
    return -0.5 * (x ** 2 + ((y - A * x ** 2 - dy) / SY) ** 2 + ((z - C * x) / SZ) ** 2)


def tube_reference(levels=(0.68, 0.95)):
    """Marginal means, sds and central intervals of the tube under the BOUNDS prior.

    The cut y <= 2.9 removes |x| > ~3.11, so x is not exactly N(0, 1) and E[y] < A. By 1D
    quadrature over x of the conditionally (truncated) normal y and z.
    """
    x = np.linspace(*BOUNDS[0], 24001)
    (ylo, yhi), (zlo, zhi) = BOUNDS[1], BOUNDS[2]
    ty = stats.truncnorm((ylo - A * x ** 2) / SY, (yhi - A * x ** 2) / SY, loc=A * x ** 2, scale=SY)
    tz = stats.truncnorm((zlo - C * x) / SZ, (zhi - C * x) / SZ, loc=C * x, scale=SZ)
    w = stats.norm.pdf(x) * (stats.norm.cdf((yhi - A * x ** 2) / SY) - stats.norm.cdf((ylo - A * x ** 2) / SY))
    w *= stats.norm.cdf((zhi - C * x) / SZ) - stats.norm.cdf((zlo - C * x) / SZ)
    w /= w.sum()
    out = {}
    for n, cond, lo, hi in (("x", None, *BOUNDS[0]), ("y", ty, ylo, yhi), ("z", tz, zlo, zhi)):
        if cond is None:
            m1, m2 = (w * x).sum(), (w * x ** 2).sum()
            cdf = lambda v: (w * (x <= v)).sum()  # noqa: E731
        else:
            m1 = (w * cond.mean()).sum()
            m2 = (w * (cond.var() + cond.mean() ** 2)).sum()
            cdf = lambda v, c=cond: (w * c.cdf(v)).sum()  # noqa: E731
        from scipy.optimize import brentq

        r = {"mean": m1, "std": np.sqrt(m2 - m1 ** 2)}
        for lv in levels:
            r[lv] = [brentq(lambda v: cdf(v) - q, lo, hi, xtol=1e-6) for q in (0.5 * (1 - lv), 0.5 * (1 + lv))]
        out[n] = r
    return out


def per_event(n_split=2, dy=0.0):
    """Per-event log densities that sum to tube_logp (split over n_split events)."""
    def fn(sl, cand):
        return np.repeat((tube_logp(cand, dy) / n_split)[:, None], sl.stop - sl.start, 1)
    return fn


def coarse_grid():
    # about 4 widths per step across the tube, as the N = 1000 closure grid
    return Grid(["x", "y", "z"], [np.linspace(*BOUNDS[0], 13), np.arange(-0.2, 2.9001, 0.04),
                                  np.arange(-3.4, 3.4001, 0.2)])


def located(dy=0.0):
    g = coarse_grid()
    lp_prior = uniform_log_prior(g, BOUNDS)
    coarse = support_scan(per_event(dy=dy), 2, g, (3, 8, 4), delta_logpost=15.0,
                          log_prior=lp_prior, keep_per_event=False)
    total = lambda th: tube_logp(th, dy)  # noqa: E731
    rmap, info = locate_ridge(total, coarse, "x", BOUNDS, knot_step=0.5, log=lambda *a: None)
    return coarse, rmap, info


def test_ridge_map_jacobian_and_inverse():
    knots = [0.0, 1.0, 3.0]
    centers = [[0.1, -0.2], [0.5, 0.3], [2.0, 1.0]]
    chols = [[[0.1, 0], [0.02, 0.3]], [[0.2, 0], [-0.05, 0.1]], [[0.05, 0], [0.0, 0.2]]]
    m = RidgeMap(["s", "a", "b"], "s", knots, centers, chols)
    rng = np.random.default_rng(1)
    s, u = rng.uniform(-0.5, 3.5, 50), rng.normal(size=(50, 2))
    th = m.to_theta(s, u)
    s2, u2 = m.to_lattice(th)
    np.testing.assert_allclose(s2, s)
    np.testing.assert_allclose(u2, u, atol=1e-10)
    e = 1e-6
    for i in range(10):
        J = np.zeros((3, 3))
        for j in range(3):
            d = np.zeros(3)
            d[j] = e
            p = np.r_[s[i], u[i]]
            J[:, j] = (m.to_theta(*_split(p + d)) - m.to_theta(*_split(p - d)))[0] / (2 * e)
        assert np.isclose(np.log(abs(np.linalg.det(J))), m.log_jacobian(s[i])[0], atol=1e-6)
    with pytest.raises(ValueError):
        RidgeMap(["s", "a"], "s", [0, 1], [[0], [1]], [[[1.0]], [[-1.0]]])


def _split(p):
    return p[:1], p[1:][None]


def test_locate_ridge_finds_centre_and_width():
    _, rmap, info = located()
    xs = np.array([-2.0, 0.0, 1.5, 2.5])
    m, L = rmap._interp(xs)
    np.testing.assert_allclose(m[:, 0], A * xs ** 2, atol=0.3 * SY + A * 0.5 ** 2 / 8)  # + chord
    np.testing.assert_allclose(m[:, 1], C * xs, atol=0.3 * SZ)
    np.testing.assert_allclose(np.sqrt((L ** 2).sum(2)), np.tile([SY, SZ], (4, 1)), rtol=0.15)
    # the coarse maximum is aliased (x = 3); the march still reaches both ends, where the
    # ridge leaves the prior through y <= 2.9 (|x| = 3.11) before its log post drops by 15
    x_exit = np.sqrt(BOUNDS[1][1] / A)
    assert rmap.knots[0] < -x_exit and rmap.knots[-1] > x_exit
    assert info["ended"] == {"up": "cross box", "down": "cross box"}
    assert all(k["converged"] for k in info["knot_info"] if not k["at_box"])


def test_ridge_scan_matches_analytic_tube():
    coarse, rmap, _ = located(dy=0.3 * SY)  # the map comes from a slightly wrong ("cheap") likelihood
    # step 0.125: in the y tail a cell of 0.25 spans ~0.75 marginal sd of y and the y interval
    # ends are off by up to 6% of an sd (0.125: 1.2%, converging)
    res = ridge_scan(per_event(), 2, rmap, BOUNDS, step=0.125, h=1.0, u_max=6.0, delta_logpost=12.0,
                     coarse=coarse, keep_per_event=False)
    d = res.diagnostics
    assert d["support_closed"] and d["scan_closed"] and d["supported_on_u_edge"] == 0
    assert d["coarse_support_outside_lattice"] == 0
    assert d["evaluated"] < 0.6 * res.scan.grid.size
    s = res.summary()
    mean, std, ci = s["posterior_mean"], s["posterior_std"], s["credible_intervals"]
    ref = tube_reference()
    for n in "xyz":
        assert abs(mean[n] - ref[n]["mean"]) < 0.01 * ref[n]["std"], (n, mean[n], ref[n]["mean"])
        assert abs(std[n] / ref[n]["std"] - 1) < 0.01, (n, std[n], ref[n]["std"])
        for lv in (0.68, 0.95):
            np.testing.assert_allclose(ci[n][str(lv)], ref[n][lv], atol=0.02 * ref[n]["std"])
    # joint level of points, against chi2(3): these HPD regions lie inside the prior, so
    # only the normalization (prior mass 0.998) changes. Counting whole cells, the level is
    # off by up to ~0.003 / 0.015 / 0.055 at h = 0.5 / 1.0 / 1.5 (cells cut by the contour)
    from omnilearned.sbi.toy.closure import joint_level

    lpost = res.log_posterior()
    norm = np.nanmax(lpost) - np.nanmax(tube_logp(res.theta_points().reshape(-1, 3)))
    for p, r2 in (((0.0, 0.0, 0.0), 0.0), ((1.0, A + 0.5 * SY, C), 1.25), ((-1.5, A * 2.25, -C * 1.5 + SZ), 3.25)):
        lev = joint_level(res, float(tube_logp(np.array(p))[0] + norm))
        assert abs(lev - stats.chi2.cdf(r2, 3)) < 0.02, (p, lev)


def test_ridge_scan_with_prior_bound_matches_dense_grid():
    """2D curved ridge cut by a prior bound: lattice against a dense fine grid."""
    a, sy, bnd = 0.3, 0.02, [(-4.0, 4.0), (-0.1, 0.9)]  # y <= 0.9 cuts |x| > ~1.73

    def lp(th):
        x, y = np.atleast_2d(th).T
        return -0.5 * (x ** 2 + ((y - a * x ** 2) / sy) ** 2)

    fn = lambda sl, c: np.repeat(lp(c)[:, None], sl.stop - sl.start, 1)  # noqa: E731
    dense_g = Grid(["x", "y"], [np.linspace(*bnd[0], 1601), np.linspace(*bnd[1], 2001)])
    dense = joint_scan(fn, 1, dense_g, candidate_chunk=400000, keep_per_event=False,
                       log_prior=uniform_log_prior(dense_g, bnd))
    cg = Grid(["x", "y"], [np.linspace(*bnd[0], 17), np.arange(-0.1, 0.9001, 0.08)])
    coarse = support_scan(fn, 1, cg, (2, 3), delta_logpost=15.0, log_prior=uniform_log_prior(cg, bnd),
                          keep_per_event=False)
    rmap, _ = locate_ridge(lp, coarse, "x", bnd, knot_step=0.25, log=lambda *a: None)
    res = ridge_scan(fn, 1, rmap, bnd, step=0.05, h=0.75, u_max=6.0, delta_logpost=14.0,
                     keep_per_event=False)
    assert res.diagnostics["support_closed"]
    a_s, b_s = dense.summary(), res.summary()
    for n in ("x", "y"):
        assert abs(a_s["posterior_mean"][n] - b_s["posterior_mean"][n]) < 0.01 * a_s["posterior_std"][n]
        assert abs(a_s["posterior_std"][n] - b_s["posterior_std"][n]) < 0.01 * a_s["posterior_std"][n]
        for lv in ("0.68", "0.95"):
            np.testing.assert_allclose(b_s["credible_intervals"][n][lv], a_s["credible_intervals"][n][lv],
                                       atol=0.02 * a_s["posterior_std"][n])
    assert b_s["boundary"]["y"]["edge_node_mass"] > 0.01  # mass piles against y = 0.9
    pm = res.posterior_mass()
    th = res.theta_points().reshape(-1, 2)
    assert np.all(pm.reshape(-1)[th[:, 1] > bnd[1][1]] == 0)
    assert np.isclose(pm.sum(), 1.0)


def test_ridge_result_save_load_and_dispatch(tmp_path):
    from omnilearned.sbi.scan import ScanResult

    coarse, rmap, _ = located()
    res = ridge_scan(per_event(), 2, rmap, BOUNDS, step=0.5, h=1.5, coarse=coarse)
    res.save(str(tmp_path / "r"), extra={"provenance": {"x": 1}})
    back = load_result(str(tmp_path / "r"))
    assert isinstance(back, RidgeResult)
    np.testing.assert_allclose(np.nan_to_num(back.log_posterior(), nan=-1), np.nan_to_num(res.log_posterior(), nan=-1))
    assert back.summary()["posterior_mean"] == pytest.approx(res.summary()["posterior_mean"])
    assert back.parameter_names() == ["x", "y", "z"] and back.fixed_names() == []
    g = Grid(["x"], [np.linspace(0, 1, 3)])
    joint_scan(lambda sl, c: np.zeros((len(c), 1)), 1, g).save(str(tmp_path / "g"))
    assert isinstance(load_result(str(tmp_path / "g")), ScanResult)


def test_lattice_grid_shape():
    m = RidgeMap(["a", "b", "c"], "b", [0.0, 1.0], [[0, 0], [1, 1]], [np.eye(2)] * 2)
    g = lattice_grid(m, [(0, 1), (0, 11), (0, 2)], 0.5, 1.5, 6.0)
    assert g.names == ["b", "u_a", "u_c"] and g.shape == (23, 9, 9)
    assert g.axes[1][4] == 0.0


def _tiny_surrogate(tmp_path):
    from omnilearned.sbi.base import GaussianBase
    from omnilearned.sbi.flow import FlowSurrogate
    from omnilearned.sbi.parameters import ConditionLayout
    from omnilearned.sbi.transforms import FeatureTransform
    from omnilearned.sbi.velocity import SetFlowVelocity

    torch.manual_seed(0)
    layout = ConditionLayout(["mu", "nu"], ["mu", "nu"], {"mu": 0.0, "nu": 0.0}, {"mu": 1.0, "nu": 1.0})
    tr = FeatureTransform(["a", "b", "c"], [None, None, None], [0, 0, 0], [1, 1, 1])
    sur = FlowSurrogate(SetFlowVelocity(3, 2, dim=16, depth=1, heads=2), GaussianBase.standard(3), tr, layout)
    ck = tmp_path / "ck.pt"
    return ck, sur.save(ck)


def test_run_fit_ridge_mode_resumes_from_caches(tmp_path):
    from omnilearned.sbi.fit import LogpCache, load_fit_config, run_fit
    from omnilearned.sbi.observations import write_observations

    ck, sha = _tiny_surrogate(tmp_path)
    rng = np.random.default_rng(0)
    write_observations(tmp_path / "obs.h5", rng.normal(size=(30, 4, 3)), np.ones((30, 4), bool),
                       np.arange(30), ["a", "b", "c"])
    cfg_path = tmp_path / "fit.toml"
    cfg_path.write_text(f"""
schema = "omnilearned-sbi-fit-v1"
checkpoint = "{ck}"
checkpoint_sha256 = "{sha}"
observations = "{tmp_path / 'obs.h5'}"
output_prefix = "{tmp_path / 'out' / 'result'}"
device = "cpu"
mode = "fixed_vector"
[grid]
mu = {{min = -1.0, max = 1.0, n = 5}}
nu = {{min = -1.0, max = 1.0, n = 5}}
[prior]
type = "uniform"
bounds = {{mu = [-1.0, 1.0], nu = [-1.0, 1.0]}}
[likelihood]
solver = "euler"
nsteps = 2
num_probes = 1
[scan]
event_chunk = 16
candidate_chunk = 8
ridge = {{axis = "mu", step = 0.25, h = 1.5, u_max = 4.5, delta_logpost = 6.0, locate = {{likelihood = {{nsteps = 1}}, seed_stride = [2, 2], delta_logpost = 50.0, knot_step = 0.5, iterations = 2}}}}
""")
    cfg = load_fit_config(cfg_path)
    ident = {"fit_config_sha256": cfg["_config_sha256"]}
    (tmp_path / "out").mkdir()
    cache = LogpCache(tmp_path / "out" / "logp_cache.npz", ident)
    doc = run_fit(cfg, logp_cache=cache, log=lambda *a: None)
    cache.flush()
    assert "ridge" in doc and doc["diagnostics"]["scan"] == "ridge"
    assert (tmp_path / "out" / "logp_cache_locate.npz").exists()
    loc = doc["diagnostics"]["locate"]
    assert loc["options"]["nsteps"] == 1 and doc["provenance"]["likelihood_options"]["nsteps"] == 2
    res = load_result(str(tmp_path / "out" / "result"))
    assert np.isclose(res.posterior_mass().sum(), 1.0)
    # a second run of the same fit reads every value from the two caches
    cfg2 = load_fit_config(cfg_path)
    cache2 = LogpCache(tmp_path / "out" / "logp_cache.npz", ident)
    doc2 = run_fit(cfg2, logp_cache=cache2, log=lambda *a: None)
    lk = doc2["diagnostics"]["likelihood"]
    assert lk["rows"] == 0 and lk["cached_rows"] > 0
    assert doc2["diagnostics"]["locate"]["likelihood"]["rows"] == 0
    assert doc2["summary"]["posterior_mean"] == pytest.approx(doc["summary"]["posterior_mean"])
