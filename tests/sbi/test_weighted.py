"""Hard-pT sampling bias (weighted toy samples): biased draw, weights, shard format."""

import json

import h5py
import numpy as np
import pytest

from omnilearned.sbi.toy.config import V2_CONFIG, load_toy_config
from omnilearned.sbi.toy.simulator_v2 import PtBias, draw_hard_pt

CFG, C, SEL = load_toy_config(V2_CONFIG)


def _weighted_mean_se(x, w):
    m = np.sum(w * x) / w.sum()
    return m, np.sqrt(np.sum(w ** 2 * (x - m) ** 2)) / w.sum()


def _unbiased_moment(f):
    g = np.geomspace(C.hard_pt_min_GeV, C.hard_pt_max_GeV, 400001)
    p = g ** -C.power_law_exponent
    return np.trapezoid(p * f(g), g) / np.trapezoid(p, g)


def test_unbiased_hard_pt_draw_is_the_old_formula():
    u = np.random.default_rng(1).uniform(0, 1, 1000)
    n = C.power_law_exponent
    lo, hi = C.hard_pt_min_GeV ** (1 - n), C.hard_pt_max_GeV ** (1 - n)
    old = (lo + u * (hi - lo)) ** (1 / (1 - n))
    for bias in (None, PtBias(0.0), PtBias(0.0, 50.0, 250.0)):
        np.testing.assert_array_equal(draw_hard_pt(u, C, bias), old)


@pytest.mark.parametrize("bias", [PtBias(3.0), PtBias(3.0, 100.0, 250.0), PtBias(4.0, 100.0, 250.0),
                                  PtBias(4.0), PtBias(2.0, 80.0, 150.0)])
def test_biased_hard_pt_density_and_weights(bias):
    """The draw follows pT^-n b(pT) (k = 4: exponent 1, log-uniform), and 1/b reweights it to pT^-n."""
    u = np.random.default_rng(2).uniform(0, 1, 400_000)
    p0 = draw_hard_pt(u, C, bias)
    assert p0.min() >= C.hard_pt_min_GeV and p0.max() <= C.hard_pt_max_GeV
    edges = np.geomspace(C.hard_pt_min_GeV, C.hard_pt_max_GeV, 41)
    g = np.geomspace(C.hard_pt_min_GeV, C.hard_pt_max_GeV, 400001)
    dens = g ** -C.power_law_exponent * bias.factor(g)
    cdf = np.concatenate([[0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(g))])
    expect = np.diff(np.interp(edges, g, cdf / cdf[-1])) * p0.size
    obs = np.histogram(p0, edges)[0]
    pull = (obs - expect) / np.sqrt(expect)
    assert np.all(np.abs(pull) < 5), pull
    w = 1.0 / bias.factor(p0)
    for f in (lambda x: x, np.log, lambda x: (x > 150).astype(float)):
        m, se = _weighted_mean_se(f(p0), w)
        assert abs(m - _unbiased_moment(f)) < 4 * se


def test_bias_cap_below_minimum_is_rejected():
    with pytest.raises(ValueError, match="cap"):
        draw_hard_pt(np.array([0.5]), C, PtBias(3.0, 100.0, 30.0))


# ---------------------------------------------------------------- toy (FastJet)


@pytest.fixture(scope="module")
def biased_and_unbiased():
    pytest.importorskip("hepyy")
    from omnilearned.sbi.toy.config import sample_setting

    th = (3.0, 0.45, 0.6)
    unb = sample_setting(th, 4000, 5, path=V2_CONFIG)
    bia = sample_setting(th, 4000, 6, path=V2_CONFIG, bias=PtBias(3.0, 100.0, 250.0))
    return unb, bia


def test_biased_toy_weights_reproduce_unbiased_jets(biased_and_unbiased):
    (ru, mu), (rb, mb) = biased_and_unbiased
    assert "weight" not in ru and "pthat" not in ru and "sampling_bias" not in mu
    w = rb["weight"]
    assert w.shape == (4000,) and np.all(w > 0)
    np.testing.assert_allclose(w.sum(), 4000, rtol=1e-12)
    np.testing.assert_array_equal(rb["pthat"], rb["diag"]["P0"])
    wb = w * PtBias(3.0, 100.0, 250.0).factor(rb["pthat"])  # w = const / b(P0)
    np.testing.assert_allclose(wb, wb[0], rtol=1e-12)
    sb = mb["sampling_bias"]
    assert sb["exponent"] == 3.0 and sb["cap_GeV"] == 250.0 and 0 < sb["ess"] < 4000
    # Unweighted, the biased jets are harder; weighted, they follow the unbiased law.
    assert rb["jet_pt"].mean() > ru["jet_pt"].mean() + 5
    for x_b, x_u in ((rb["pthat"], ru["diag"]["P0"]), (rb["jet_pt"], ru["jet_pt"]),
                     (rb["n_constituents"].astype(float), ru["n_constituents"].astype(float)),
                     ((rb["jet_pt"] > 120).astype(float), (ru["jet_pt"] > 120).astype(float))):
        m, se = _weighted_mean_se(x_b, w)
        se_u = x_u.std() / np.sqrt(x_u.size)
        assert abs(m - x_u.mean()) < 4 * np.hypot(se, se_u), (m, x_u.mean(), se, se_u)
    # The bias raises the acceptance (fewer events below the 60 GeV cut).
    assert mb["acceptance"] > 1.5 * mu["acceptance"]


def test_zero_bias_is_the_unbiased_toy():
    pytest.importorskip("hepyy")
    from omnilearned.sbi.toy.config import array_sha256, sample_setting

    a, _ = sample_setting((6.0, 0.40, 0.65), 50, 424242, path=V2_CONFIG)
    b, mb = sample_setting((6.0, 0.40, 0.65), 50, 424242, path=V2_CONFIG, bias=PtBias(0.0))
    assert array_sha256(a) == array_sha256(b) and "weight" not in b and "sampling_bias" not in mb


def test_weighted_shards_round_trip_and_old_shards_read_unit_weights(tmp_path):
    pytest.importorskip("hepyy")
    from omnilearned.sbi.toy import datasets
    from omnilearned.sbi.toy.datasets import generate_shard, make_design, read_shards, shard_paths, write_index

    pts, _ = make_design(4, "train", CFG)
    bias = PtBias(3.0, 100.0, 250.0)
    wdir, udir = tmp_path / "w", tmp_path / "u"
    m = generate_shard(wdir / "train_00000.h5", "train", np.arange(2), pts[:2], 30, log=lambda s: None, bias=bias)
    datasets._cli(["generate", "--design", _design_file(tmp_path, pts), "--shard", "1", "--points-per-shard", "2",
                   "--jets", "30", "--out-dir", str(wdir), "--bias-k", "3", "--bias-cap", "250"])
    generate_shard(udir / "train_00000.h5", "train", np.arange(2), pts[:2], 30, log=lambda s: None)
    assert m["sampling_bias"] == bias.to_dict()
    assert all(q["toy"]["sampling_bias"]["exponent"] == 3.0 for q in m["points"])
    idx = write_index(wdir, "train")
    assert idx["sampling_bias"] == [bias.to_dict()] and idx["n_jets"] == 120
    assert write_index(udir, "train")["sampling_bias"] == [None]
    with h5py.File(wdir / "train_00000.h5") as f:
        assert json.loads(f.attrs["sampling_bias"]) == bias.to_dict()
        assert f["weight"].shape == f["pthat"].shape == (60,)
        assert "diag" not in f
    d = read_shards(shard_paths(wdir, "train"))
    for p in range(4):
        np.testing.assert_allclose(d["weight"][d["point_id"] == p].sum(), 30, rtol=1e-12)
    assert np.ptp(d["weight"]) > 0
    u = read_shards(shard_paths(udir, "train"))
    assert u["weight"].dtype == np.float64
    np.testing.assert_array_equal(u["weight"], np.ones(60))
    # Same stream, different hard-pT law: the biased jets are different jets.
    assert not np.array_equal(u["event_id"], d["event_id"][:60])
    with pytest.raises(ValueError, match="counts-only"):
        generate_shard(tmp_path / "c" / "train_counts_00000.h5", "train", np.arange(1), pts[:1], 5,
                       log=lambda s: None, counts_only=True, bias=bias)


def _design_file(tmp_path, pts):
    p = tmp_path / "design_train.json"
    p.write_text(json.dumps({"split": "train", "points": np.asarray(pts).tolist()}))
    return str(p)


@pytest.mark.parametrize("mode", ["loss", "sample"])
def test_train_driver_on_weighted_shards(tmp_path, mode):
    pytest.importorskip("hepyy")
    from omnilearned.sbi.toy import train
    from omnilearned.sbi.toy.datasets import generate_shard, make_design, write_index

    bias = PtBias(3.0, 100.0, 250.0)
    d = tmp_path / "data"
    for split, n in (("train", 2), ("validation", 1)):
        pts, _ = make_design(n, split, CFG)
        generate_shard(d / f"{split}_00000.h5", split, np.arange(n), pts, 40, log=lambda s: None, bias=bias)
        write_index(d, split)
    out = tmp_path / "run"
    train.main(["--data-dir", str(d), "--out", str(out), "--steps", "6", "--val-every", "3",
                "--warmup-steps", "2", "--dim", "16", "--depth", "1", "--heads", "2",
                "--mult-steps", "4", "--mult-hidden", "8", "--mult-depth", "1", "--batch-size", "16",
                "--device", "cpu", "--length-buckets", "2", "--weight-mode", mode])
    run = json.loads((out / "run.json").read_text())
    ws = run["data"]["weights"]["train"]
    assert not ws["unit"] and ws["n"] == 80 and 0 < ws["ess"] < 80 and len(ws["per_point_bin_median"]) == 8
    assert run["flow_history"]["weights"]["mode"] == mode and run["multiplicity_history"]["weighted"]
    assert run["meta"]["train_options"]["weight_mode"] == mode
