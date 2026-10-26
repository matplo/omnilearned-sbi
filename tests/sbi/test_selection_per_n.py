"""Per-N selection normalization (fidelity plan S1) and gate residual diagnostics (S0)."""

import json

import numpy as np
import pytest



def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def _accept(n):  # known acceptance, falling with N
    return _sigmoid(3.0 - 0.08 * n)


class _FakeSurrogate:
    """N from a fixed p(N); 'clouds' carry a uniform u in x[..., 0] that the fake selection
    compares with a(N): a cloud passes with probability exactly a(N)."""

    def __init__(self, pn):
        self.pn = np.asarray(pn, float) / np.sum(pn)

    def sample(self, theta, n, num_particles=None, seed=0, pad_to=None, **kw):
        rng = np.random.default_rng(seed)
        counts = rng.choice(len(self.pn), n, p=self.pn) if num_particles is None else np.asarray(num_particles)
        w = int(pad_to or counts.max())
        mask = np.arange(w)[None, :] < counts[:, None]
        x = np.zeros((n, w, 3))
        x[:, :, 0] = rng.random(n)[:, None]
        return x, mask


def _fake_select(x, sel, mask):
    n = mask.sum(1)
    ok = x[:, 0, 0] < _accept(n)
    return ok, np.where(ok, 0, 6), None


@pytest.fixture
def fake(monkeypatch):
    from omnilearned.sbi.toy import sample_select

    monkeypatch.setattr(sample_select, "select_clouds", _fake_select)
    pn = np.exp(-0.5 * ((np.arange(81) - 35) / 10.0) ** 2)
    pn[:2] = 0
    return _FakeSurrogate(pn)


def test_redraw_keeps_p_of_n_and_dropping_does_not(fake):
    from omnilearned.sbi.toy.sample_select import sample_and_select

    n = 40000
    _, m0, r0 = sample_and_select(fake, (7.5, 0.3, 0.95), n, None, seed=3)
    _, m1, r1 = sample_and_select(fake, (7.5, 0.3, 0.95), n, None, seed=3, redraw=True, max_rounds=1000)
    _, m2, r2 = sample_and_select(fake, (7.5, 0.3, 0.95), n, None, seed=3, redraw=True, max_rounds=1)
    assert r2["redraw"]["unfilled"] > 0 and len(m2) == n - r2["redraw"]["unfilled"]  # reported, dropped
    drawn = fake.sample(None, n, None, seed=3)[1].sum(1)
    # redraw: every cloud is kept at its drawn N, so the selected multiset of N is the drawn one
    np.testing.assert_array_equal(np.sort(m1.sum(1)), np.sort(drawn))
    assert r1["redraw"]["unfilled"] == 0 and r1["selected"] == n and r1["redraw"]["rounds"] > 1
    assert r1["acceptance"] == r0["acceptance"] < 1  # the report keeps the first-pass Pr(S|theta)
    # dropping rejected clouds tilts N by a(N): the selected mean N falls by the predicted amount
    pn = fake.pn
    k = np.arange(len(pn))
    pred = np.sum(k * pn * _accept(k)) / np.sum(pn * _accept(k))
    se = m0.sum(1).std() / np.sqrt(len(m0))
    assert abs(m0.sum(1).mean() - pred) < 4 * se and drawn.mean() - m0.sum(1).mean() > 10 * se
    g, a = np.array(r0["acceptance_by_n"]["generated"]), np.array(r0["acceptance_by_n"]["accepted"])
    assert g.sum() == n and a.sum() == r0["selected"]


def test_logit_fit_recovers_known_acceptance():
    from omnilearned.sbi.toy.selection import _basis, fit_logit_in_n

    rng = np.random.default_rng(0)
    n = np.arange(200)
    g = np.where(n >= 5, rng.poisson(400 * np.exp(-0.5 * ((n - 40) / 15.0) ** 2)), 0)
    k = rng.binomial(g, _accept(n))
    c, knots = fit_logit_in_n(g, k)
    used = g > 20
    est = _sigmoid(_basis(n[used], knots) @ c)
    a = _accept(n)[used]
    assert np.all(np.abs(est - a) < 3 * np.sqrt(a * (1 - a) / g[used]) + 0.01)
    w = g[used] / g[used].sum()  # where the clouds are, the fit is much closer
    assert np.sum(w * np.abs(est - a)) < 0.005
    # no accepted clouds at all: finite, small acceptance
    c0, knots = fit_logit_in_n(g, np.zeros_like(g))
    assert np.all(np.isfinite(c0)) and _sigmoid(_basis([40], knots) @ c0)[0] < 1e-3


def _per_n_table(tmp_path, fn=_accept, seed=0):
    from omnilearned.sbi.toy.selection import save_per_n_table

    axes = [np.linspace(0, 11, 3), np.linspace(0.25, 0.55, 2), np.linspace(0.2, 1.2, 2)]
    rng = np.random.default_rng(seed)
    n = np.arange(257)
    gen = np.zeros((3, 2, 2, 257), np.int64)
    acc = np.zeros_like(gen)
    for idx in np.ndindex(3, 2, 2):
        gen[idx] = np.where(n >= 2, rng.poisson(3000 * np.exp(-0.5 * ((n - 30) / 12.0) ** 2)), 0)
        acc[idx] = rng.binomial(gen[idx], fn(n))
    save_per_n_table(tmp_path / "pn.npz", axes, gen, acc, {"unit": True})
    return tmp_path / "pn.npz"


def test_per_n_lognorm_table_bind_and_unbiased_multiplicity(tmp_path):
    from omnilearned.sbi.toy.selection import PerNLogNorm

    t = PerNLogNorm(_per_n_table(tmp_path))
    pts = np.array([[7.5, 0.30, 0.95], [2.0, 0.5, 0.4]])
    counts = np.array([10, 20, 20, 35, 50])
    la = t(pts, counts)
    assert la.shape == (2, 5)
    np.testing.assert_allclose(np.exp(la), np.broadcast_to(_accept(counts), (2, 5)), atol=0.02)
    np.testing.assert_allclose(t.bind(counts)(pts), la.mean(1))
    # names in another order give the same values
    np.testing.assert_allclose(t(pts[:, [1, 0, 2]], counts, ["T_ref", "b", "tau_init"]), la)
    assert np.isfinite(t(pts, [1, 300])).all()  # outside the sampled N: nearest sampled N
    with open(tmp_path / "pn.json") as f:
        assert len(json.load(f)["knots_N"]) == len(t.knots)

    # The multiplicity part of the selected likelihood. p(N | lam) = Poisson(lam), trained on
    # selected multiplicities; data from Poisson(lam0). Per-N normalization: the a(N) term does
    # not depend on lam, so the MLE is the sample mean. Per-theta normalization
    # (divide by sum_N p(N|lam) a(N)) shifts the MLE whenever a depends on N.
    from scipy.stats import poisson

    rng = np.random.default_rng(1)
    lam0 = 30.0
    data = rng.poisson(lam0, 20000)
    lams = np.linspace(28, 34, 601)
    k = np.arange(257)
    ll_n = np.array([poisson.logpmf(data, L).sum() for L in lams]) \
        - len(data) * t.bind(data)(np.repeat(pts[:1], len(lams), 0))
    ll_t = np.array([poisson.logpmf(data, L).sum() - len(data) * np.log(np.sum(poisson.pmf(k, L) * _accept(k)))
                     for L in lams])
    se = np.sqrt(lam0 / len(data))
    assert abs(lams[np.argmax(ll_n)] - data.mean()) < 0.011
    assert lams[np.argmax(ll_t)] - data.mean() > 5 * se


def test_closure_per_n_log_norm_binds_observed_counts(tmp_path):
    from omnilearned.sbi.observations import file_sha256
    from omnilearned.sbi.toy import closure
    from omnilearned.sbi.toy.records import write_observation_file
    from omnilearned.sbi.toy.selection import PerNLogNorm

    table = _per_n_table(tmp_path)
    counts = np.array([5, 12, 30, 30, 44])
    mask = np.arange(48)[None, :] < counts[:, None]
    rec = {"x": np.where(mask[..., None], 0.1, 0.0), "mask": mask, "event_id": np.arange(5)}
    obs = tmp_path / "obs.h5"
    write_observation_file(obs, rec, attrs={})
    cfg = {"log_norm_table": str(table), "log_norm_table_sha256": file_sha256(table),
           "log_norm_mode": "per_n", "observations": str(obs), "mode": "fixed_vector"}
    fn = closure._log_norm(cfg)
    pts = np.array([[7.5, 0.3, 0.95]])
    np.testing.assert_allclose(fn(pts), PerNLogNorm(table)(pts, counts).mean(1))
    with pytest.raises(ValueError, match="fixed-vector"):
        closure._log_norm({**cfg, "mode": "local_variable"})


def test_flow_sample_with_explicit_counts():
    from test_variable_multiplicity import _variable_surrogate

    sur = _variable_surrogate()
    counts = np.array([3, 9, 1, 9, 4])
    x, m = sur.sample([0.5, 1.0], 5, counts, seed=2, nsteps=4, sort_by_length=True, pad_to=12)
    assert x.shape == (5, 12, 3)
    np.testing.assert_array_equal(m.sum(1), counts)
    assert np.all(x[~m] == 0) and np.isfinite(x).all()
    with pytest.raises(ValueError, match="positive counts"):
        sur.sample([0.5, 1.0], 2, np.array([0, 3]), nsteps=2)


def _gate_dir(path, deltas, ses, theta, p):
    """Point JSONs in the gate's format with the given signed shifts (units 1)."""
    from omnilearned.sbi.fidelity import JET_VARIABLE

    path.mkdir(parents=True, exist_ok=True)
    for i, (d, s, th, pp) in enumerate(zip(deltas, ses, theta, p)):
        rel = {k: {"delta_mean": float(d[j]), "se": float(s[j]), "relevance": float(abs(d[j])),
                   "relevant": bool(abs(d[j]) - 2 * s[j] > 0.5)} for j, k in enumerate(JET_VARIABLE)}
        doc = {"point_id": i, "theta": list(th), "permutation": {"p_family": float(pp)},
               "relevance_by_n": {"250": rel, "1000": rel}}
        (path / f"point_{i:05d}.json").write_text(json.dumps(doc))
    return path


def test_residual_map_scaling_and_seed_spread(tmp_path):
    from omnilearned.sbi.toy import residuals

    rng = np.random.default_rng(0)
    P, K = 400, 7
    lo, hi = np.array([0, 0.25, 0.2]), np.array([11, 0.55, 1.2])
    theta = lo + (hi - lo) * rng.random((P, 3))
    u = residuals.box_coordinates(theta, list(zip(lo, hi)))
    se = np.full((P, K), 0.3)
    bias = np.zeros((P, K))
    bias[:, 1] = -0.6 - 0.4 * u[:, 1]  # width: offset and a T_ref slope
    g_a = _gate_dir(tmp_path / "a", bias * 2.0 + se * rng.standard_normal((P, K)), se, theta, rng.random(P))
    g_b = _gate_dir(tmp_path / "b", bias + se * rng.standard_normal((P, K)), se, theta, rng.random(P))
    g = residuals.load_gate(g_b)
    y, s = residuals.in_units(g)
    fit = residuals.linear_map(u, y, s)[1]
    np.testing.assert_allclose(fit["coef"], [-0.6, 0, -0.4, 0], atol=4 * fit["coef_se"].max())
    b2, e = residuals.bias_squared(y[:, 1], s[:, 1])
    true_b2 = np.mean(bias[:, 1] ** 2)
    assert abs(b2 - true_b2) < 4 * e
    # B halves from n = 1000 to 4000: alpha = 0.5
    ga = residuals.load_gate(g_a)
    ya, sa = residuals.in_units(ga)
    al = residuals.scaling_exponent(residuals.bias_squared(ya[:, 1], sa[:, 1])[0], b2, 1000, 4000)
    assert abs(al - 0.5) < 0.05
    assert abs(residuals.predict_b2(b2, 4000, 0.5, 16000) - b2 / 4) < 1e-12
    # seed spread: shared direct noise cancels; model scatter tau = 0.2 recovered
    direct = se / np.sqrt(2) * rng.standard_normal((P, K))
    runs = [bias + 0.2 * rng.standard_normal((P, K)) + se / np.sqrt(2) * rng.standard_normal((P, K)) - direct
            for _ in range(3)]
    sp = residuals.seed_spread(runs, [se] * 3, common_random_numbers=False)
    np.testing.assert_allclose(sp["tau"], 0.2, atol=0.03)
    # common random numbers (the gates' seeding): the surrogate noise is shared too, the
    # independent-noise subtraction over-corrects to 0, the CRN bound recovers tau; offsets
    # 0, 0.3, 0.6 between runs give offset_std 0.3
    surr = se / np.sqrt(2) * rng.standard_normal((P, K))
    runs = [bias + c + 0.2 * rng.standard_normal((P, K)) + surr - direct for c in (0.0, 0.3, 0.6)]
    np.testing.assert_allclose(residuals.seed_spread(runs, [se] * 3)["offset_std"], 0.3, atol=0.03)
    sp = residuals.seed_spread([r - c for r, c in zip(runs, (0.0, 0.3, 0.6))], [se] * 3)
    assert np.all(sp["tau_independent"] == 0)  # tau² 0.04 < surrogate se² 0.045
    np.testing.assert_allclose(sp["tau"], 0.2, atol=0.03)
    row = residuals.gate_row(g_b)
    assert row["points"] == P and row["fail"] <= row["significant"]
    residuals.main(["scaling", "--gate-a", str(g_a), "--n-a", "1000", "--gate-b", str(g_b), "--n-b", "4000",
                    "--out", str(tmp_path / "s.json")])
    assert abs(json.loads((tmp_path / "s.json").read_text())["regions"]["all"]["per_observable"]["width"]["alpha"]
               - 0.5) < 0.05


def test_mean_shift_decomposes_into_multiplicity_mix_and_fixed_n():
    from omnilearned.sbi.toy.boundary import decompose, merged_bins

    rng = np.random.default_rng(0)
    n_d = rng.poisson(30, 50000) + 2
    n_s = rng.poisson(28, 50000) + 2  # surrogate N tilted low
    o_d = 0.01 * n_d + rng.normal(0, 0.1, n_d.size)
    o_s = 0.01 * n_s + 0.02 + rng.normal(0, 0.1, n_s.size)  # and +0.02 at fixed N
    tot, mix, fixed, nb = decompose(o_s, n_s, o_d, n_d)
    assert abs(tot - (o_s.mean() - o_d.mean())) < 1e-9  # exact identity
    assert abs(mix - 0.01 * (n_s.mean() - n_d.mean())) < 0.003 and abs(fixed - 0.02) < 0.003
    bins = merged_bins(n_s, n_d, 30)
    assert bins[0][0] == min(n_s.min(), n_d.min()) and bins[-1][1] == max(n_s.max(), n_d.max())
    assert all(b[0] == a[1] + 1 for a, b in zip(bins[:-1], bins[1:])) and nb == len(bins)


def test_dr_max_from_pt_weighted_axis():
    from omnilearned.sbi.toy.boundary import dr_max

    x = np.zeros((2, 3, 3))
    x[0, :, 0] = [0.0, 0.0, 0.3]   # eta; equal pT -> axis at eta 0.1
    x[0, :, 2] = 0.0
    x[1, :2, 1] = [0.2, -0.2]      # phi; third slot padded
    mask = np.array([[1, 1, 1], [1, 1, 0]], bool)
    np.testing.assert_allclose(dr_max(x, mask), [0.2, 0.2])
