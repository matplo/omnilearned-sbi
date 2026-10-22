"""Variable constituent multiplicity: toy v2, padded records, p(N | theta) and masks."""

import numpy as np
import pytest
import torch

from omnilearned.sbi.base import GaussianBase
from omnilearned.sbi.likelihood import LikelihoodOptions, stable_probes

pytest.importorskip("hepyy")


@pytest.fixture(scope="module")
def v2_jets():
    from omnilearned.sbi.toy.config import V2_CONFIG, sample_setting

    lo = sample_setting((0.0, 0.25, 0.2), 400, 11, path=V2_CONFIG)
    hi = sample_setting((0.0, 0.55, 0.2), 400, 11, path=V2_CONFIG)
    return lo, hi


def test_v2_records_are_padded_left_packed_and_selected(v2_jets):
    from omnilearned.sbi.toy.config import V2_CONFIG, load_toy_config
    from omnilearned.sbi.toy.selection import select_clouds

    _, _, sel = load_toy_config(V2_CONFIG)
    for rec, man in v2_jets:
        x, m, n = rec["x"], rec["mask"], rec["n_constituents"]
        assert man["backend"] == "toy-quenching-v2" and man["complete"]
        assert x.shape == (400, 256, 3) and m.shape == (400, 256)
        assert np.all(x[~m] == 0)
        np.testing.assert_array_equal(m, np.arange(256)[None, :] < n[:, None])
        assert n.min() >= sel.min_constituents and n.max() <= sel.max_constituents
        pt = np.where(m, np.exp(x[..., 2]), 0.0)
        assert np.all(np.diff(pt, axis=1)[m[:, 1:]] <= 0)  # pT-ordered
        # The selected region S is the same for toy jets and surrogate clouds.
        ok, _, _ = select_clouds(x, sel, m)
        assert ok.all()


def test_v2_multiplicity_carries_medium_information(v2_jets):
    (lo, _), (hi, _) = v2_jets
    assert hi["n_constituents"].mean() > lo["n_constituents"].mean() + 8


# Reference from the GPU node (2026-10-22). Selection, membership, masks and ids are
# identical on every host; floats agree to roundoff only, because numpy picks
# SIMD kernels per CPU (Perlmutter AMD Milan, AVX2, differed from the Intel
# AVX-512 GPU node in the last bits, <= 1.4e-15 relative). The bitwise hashes
# are recorded per instruction set.
V2_DISCRETE = "199d00848e44ebb733f0390e2fceca826a364c39e5b80df86eb8853afb82a961"
V2_FINGERPRINT = [
    [-389.8388860008742, -394.51665392117053, -395.8486688738363, -382.6167465668338, 1047.8406948297177],
    [-266.58037865893857, -251.62678592344065, -256.4102678869091, -260.9618149757064, 1486.7414014208553],
    [-3364.0737369574185, -3328.951143611859, -3351.972003290646, -3305.8611329840905, 33689.765506368196],
]
V2_BITWISE = {
    "avx512": "b50145838ae3d81b9b562f77715983f7e37fef9e46a2dca1860d122f68b23a8b",  # Intel Xeon 4210R, numpy 2.2.6
    "avx2": "86870b042a6ab1162471207f56a109c271ad1e5310e708c31c48e8b72e08e2c2",  # Perlmutter Milan; also the GPU node with NPY_DISABLE_CPU_FEATURES=AVX512*
}


def test_v2_cross_host_reproducibility():
    from omnilearned.sbi.toy.config import (V2_CONFIG, array_sha256, discrete_sha256,
                                            fingerprint_deviation, sample_setting)

    rec, man = sample_setting((6.0, 0.40, 0.65), 300, 424242, path=V2_CONFIG)
    assert man["accepted"] == 300 and man["attempted"] == 2227
    assert discrete_sha256(rec) == V2_DISCRETE
    assert fingerprint_deviation(rec, V2_FINGERPRINT) < 1e-12
    h = array_sha256(rec)
    if h not in V2_BITWISE.values():
        pytest.skip(f"bitwise hash {h} not recorded for this instruction set (tolerance check passed)")


def test_training_records_round_trip_and_trim(tmp_path, v2_jets):
    from omnilearned.sbi.toy.records import read_training_set, write_training_set

    (lo, _), (hi, _) = v2_jets
    write_training_set(tmp_path / "t.h5", [((0.0, 0.25, 0.2), lo), ((0.0, 0.55, 0.2), hi)],
                       backend="toy-quenching-v2")
    d = read_training_set(tmp_path / "t.h5")
    w = int(max(lo["n_constituents"].max(), hi["n_constituents"].max()))
    assert d["x"].shape == (800, w, 3) and d["mask"].shape == (800, w)
    np.testing.assert_array_equal(d["n_constituents"], d["mask"].sum(1))
    np.testing.assert_array_equal(d["x"][:400], lo["x"][:, :w])
    full = read_training_set(tmp_path / "t.h5", trim=False)
    assert full["x"].shape[1] == 256


# ---------------------------------------------------------------- framework


def test_stable_probes_do_not_depend_on_padding_length():
    ids = np.array([5, 77, 1 << 40])
    for kind in ("rademacher", "gaussian"):
        a = stable_probes(ids, 40, 3, 4, 9, kind, evaluation=3)
        b = stable_probes(ids, 256, 3, 4, 9, kind, evaluation=3)
        torch.testing.assert_close(a, b[:, :, :40])


def test_masked_observables_ignore_padding():
    from omnilearned.sbi.toy.observables import jet_observables

    rng = np.random.default_rng(0)
    x = np.stack([rng.normal(0, 0.1, (50, 20)), rng.normal(0, 0.1, (50, 20)),
                  rng.normal(1.0, 1.0, (50, 20))], -1)
    ref = jet_observables(x)
    pad = np.concatenate([x, np.zeros((50, 30, 3))], 1)
    got = jet_observables(pad, np.broadcast_to(np.arange(50) < 20, (50, 50)))
    for k, v in ref.items():
        np.testing.assert_allclose(got[k], v, rtol=1e-12, err_msg=k)
    assert np.all(got["n_constituents"] == 20)


def test_multiplicity_model_learns_poisson_and_is_normalized():
    from omnilearned.sbi.multiplicity import MultiplicityModel, train_multiplicity

    rng = np.random.default_rng(1)
    c = rng.uniform(-1, 1, 20000)
    n = rng.poisson(20 + 10 * c)
    vc = rng.uniform(-1, 1, 4000)
    vn = rng.poisson(20 + 10 * vc)
    m, hist = train_multiplicity(MultiplicityModel(1, 64, hidden=32, depth=2),
                                 torch.tensor(c[:, None]), n, torch.tensor(vc[:, None]), vn,
                                 steps=600, batch_size=2048, lr=3e-3, log=lambda s: None)
    probe = torch.tensor([[-0.5], [0.5]], dtype=torch.float32)
    lp = torch.log_softmax(m.logits(probe).double(), -1)
    torch.testing.assert_close(lp.exp().sum(-1), torch.ones(2, dtype=torch.float64))
    mean = (lp.exp() * torch.arange(65)).sum(-1).numpy()
    np.testing.assert_allclose(mean, [15.0, 25.0], atol=1.0)
    assert hist["best_val_nll"] < 3.2  # Poisson(20) entropy ~ 2.9 nats


def _variable_surrogate(seed=0):
    from omnilearned.sbi.flow import FlowSurrogate
    from omnilearned.sbi.multiplicity import MultiplicityModel, layout_with_multiplicity
    from omnilearned.sbi.parameters import ConditionLayout
    from omnilearned.sbi.transforms import FeatureTransform
    from omnilearned.sbi.velocity import SetFlowVelocity

    torch.manual_seed(seed)
    layout = layout_with_multiplicity(["a", "b"], [0, 0], [1, 2])
    model = SetFlowVelocity(3, layout.dim, dim=32, depth=2, heads=4)
    with torch.no_grad():  # non-trivial velocity (the output layer starts at zero)
        model.out.weight.normal_(0, 0.1)
        model.out.bias.normal_(0, 0.05)
    mult = MultiplicityModel(2, 64, hidden=16, depth=1)
    with torch.no_grad():
        mult.net[-1].bias.copy_(-0.5 * (torch.arange(65.0) - 12) ** 2 / 9)
    mult.layout = ConditionLayout.from_bounds(["a", "b"], [0, 0], [1, 2])
    tr = FeatureTransform(["eta", "phi", "lpt"], [2.5, 1.2, None], [0, 0, 1], [0.5, 0.4, 1])
    return FlowSurrogate(model.eval(), GaussianBase.standard(3), tr, layout, multiplicity=mult)


def test_variable_surrogate_sample_density_padding_and_persistence(tmp_path):
    from omnilearned.sbi.flow import FlowSurrogate

    sur = _variable_surrogate()
    x, mask = sur.sample([0.5, 1.0], 12, None, seed=4, nsteps=6, pad_to=64)
    counts = mask.sum(1)
    assert x.shape == (12, 64, 3) and np.all(x[~mask] == 0)
    np.testing.assert_array_equal(mask, np.arange(64)[None, :] < counts[:, None])
    assert counts.min() >= 1 and len(set(counts.tolist())) > 1

    opts = LikelihoodOptions(solver="midpoint", nsteps=6, num_probes=2,
                             probe_schedule="per_step")
    cands = np.array([[0.5, 1.0], [0.2, 1.7]])
    ids = np.arange(12) + 1000
    lp64, d64 = sur.log_prob(x.astype(np.float64), mask, cands, ids, opts)
    w = int(counts.max())
    x256 = np.pad(x, ((0, 0), (0, 256 - 64), (0, 0)))
    m256 = np.pad(mask, ((0, 0), (0, 256 - 64)))
    lp256, _ = sur.log_prob(x256.astype(np.float64), m256, cands, ids, opts)
    np.testing.assert_allclose(lp256, lp64, rtol=0, atol=1e-6)  # padding length is irrelevant
    assert np.isfinite(lp64).all()
    # The multiplicity term is log p(N | theta) of the multiplicity model.
    np.testing.assert_allclose(d64["log_multiplicity"], sur.log_multiplicity(counts, cands))
    paired, _ = sur.log_prob_paired(x[:, :w].astype(np.float64), mask[:, :w],
                                    np.repeat(cands[:1], 12, 0), ids, opts)
    np.testing.assert_allclose(paired, lp64[0], atol=1e-6)

    path = tmp_path / "v.pt"
    sur.save(path)
    back = FlowSurrogate.load(path)
    lp_back, _ = back.log_prob(x.astype(np.float64), mask, cands, ids, opts)
    np.testing.assert_allclose(lp_back, lp64, atol=1e-6)


def test_flow_matching_loss_ignores_padding_width():
    from omnilearned.sbi.flow import flow_matching_loss

    sur = _variable_surrogate()
    g = torch.Generator().manual_seed(0)
    u = torch.randn(16, 10, 3, generator=g)
    mask = torch.arange(10)[None, :] < torch.randint(2, 10, (16, 1), generator=g)
    u = u * mask[..., None]
    cond = torch.randn(16, 3, generator=g)
    t = torch.rand(16, generator=g)
    eps = torch.randn(16, 10, 3, generator=g)
    a = flow_matching_loss(sur.model, u, mask, cond, sur.base, t=t, eps=eps)
    pad = lambda v: torch.cat([v, torch.zeros((16, 30) + v.shape[2:], dtype=v.dtype)], 1)  # noqa: E731
    b = flow_matching_loss(sur.model, pad(u), pad(mask), cond, sur.base, t=t, eps=pad(eps))
    torch.testing.assert_close(a, b)


def test_fidelity_gate_with_masks_adds_multiplicity(v2_jets):
    from omnilearned.sbi.fidelity import evaluate

    (lo, _), _ = v2_jets
    a = {k: lo[k][:200] for k in ("x", "mask")}
    b = {k: lo[k][200:] for k in ("x", "mask")}
    w = int(lo["n_constituents"].max())
    res = evaluate(a["x"], b["x"][:, :w], n_events=250, n_perm=50,
                   direct_mask=a["mask"], surrogate_mask=b["mask"][:, :w])
    assert "n_constituents" in res["relevance"] and "n_constituents" in res["permutation"]["ks_d"]
    assert not res["verdict"]["fail"]  # two halves of the same toy sample
    with pytest.raises(ValueError):
        evaluate(a["x"], b["x"], direct_mask=a["mask"])


def test_length_bucketed_minibatches_are_narrow_and_unbiased():
    from omnilearned.sbi.flow import minibatch_sampler

    lengths = torch.as_tensor(np.random.default_rng(3).poisson(30, 4000) + 2)
    gen = torch.Generator().manual_seed(0)
    # buckets <= 1 keeps the original uniform stream
    uni = minibatch_sampler(lengths, 64, 0, gen)
    ref = torch.randint(0, 4000, (64,), generator=torch.Generator().manual_seed(0))
    assert torch.equal(uni(), ref) and uni.edges is None
    draw = minibatch_sampler(lengths, 64, 8, torch.Generator().manual_seed(1))
    assert len(draw.edges) == 8 and draw.edges == sorted(draw.edges) and draw.edges[-1] == int(lengths.max())
    counts = torch.zeros(4000)
    spans = []
    for _ in range(4000):
        idx = draw()
        counts += torch.bincount(idx, minlength=4000)
        spans.append(int(lengths[idx].max() - lengths[idx].min()))
    assert np.median(spans) <= 6 < int(lengths.max() - lengths.min())  # narrow batches
    # Every event has probability 1/n per draw: per-quantile totals are equal up to
    # the bucket choice (64 x Binomial(4000, 1/8) events per group) ...
    per_group = counts[torch.argsort(lengths, stable=True)].reshape(8, -1).sum(1)
    expected, sd = 4000 * 64 / 8, 64 * (4000 / 8 * 7 / 8) ** 0.5
    assert torch.all((per_group - expected).abs() < 4 * sd)
    # ... and uniform inside a group (multinomial spread of ~64 x 500 draws over 500 events).
    first = counts[torch.argsort(lengths, stable=True)][:500]
    assert first.std() / first.mean() < 3 * (500 / first.sum()) ** 0.5


def test_sample_sorted_by_length_keeps_draw_order_and_distribution():
    sur = _variable_surrogate()
    x0, m0 = sur.sample([0.5, 1.0], 600, None, seed=7, nsteps=4, batch=64)
    x1, m1 = sur.sample([0.5, 1.0], 600, None, seed=7, nsteps=4, batch=64, sort_by_length=True)
    np.testing.assert_array_equal(m0, m1)  # same N per draw position, returned in draw order
    assert np.all(x1[~m1] == 0) and np.isfinite(x1).all()
    # Different noise assignment, same distribution: per-feature means agree in se.
    a, b = x0[m0], x1[m1]
    se = np.sqrt(a.var(0) / len(a) + b.var(0) / len(b))
    assert np.all(np.abs(a.mean(0) - b.mean(0)) < 4 * se)
    # Clouds of equal N are exchangeable: within each N the sorted run is a valid draw.
    assert not np.allclose(x0, x1)


def test_fit_sorted_by_multiplicity_matches_file_order(tmp_path):
    from omnilearned.sbi.fit import run_fit
    from omnilearned.sbi.observations import Observations

    sur = _variable_surrogate()
    x, mask = sur.sample([0.5, 1.0], 40, None, seed=3, nsteps=4)
    assert len(set(mask.sum(1).tolist())) > 3
    obs = Observations(x.astype(np.float64), mask, np.arange(40) + 77, None, (), ("eta", "phi", "lpt"),
                       "sha", "mem")
    docs = {}
    for sort in (False, True):
        cfg = {"schema": "omnilearned-sbi-fit-v1", "device": "cpu",
               "output_prefix": str(tmp_path / f"s{int(sort)}" / "fit"),
               "grid": {"a": [0.3, 0.5, 0.7], "b": [1.0, 1.5]},
               "likelihood": {"solver": "midpoint", "nsteps": 4, "num_probes": 2,
                              "probe_schedule": "per_step"},
               "scan": {"event_chunk": 8, "candidate_chunk": 3, "sort_by_multiplicity": sort}}
        docs[sort] = run_fit(cfg, surrogate=sur, observations=obs, log=lambda m: None)
    from omnilearned.sbi.scan import ScanResult

    a, b = (ScanResult.load(str(tmp_path / f"s{i}" / "fit")) for i in (0, 1))
    np.testing.assert_allclose(b.per_event, a.per_event, rtol=0, atol=1e-5)  # file order kept
    np.testing.assert_allclose(b.loglik, a.loglik, rtol=0, atol=1e-4)
    assert docs[True]["provenance"]["evaluation_order"] == "sorted by multiplicity"
