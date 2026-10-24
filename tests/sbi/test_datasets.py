"""Training-scale toy datasets (designs, shards, index, observations) and the
smoke drivers (training, sample-and-select)."""

import json

import h5py
import numpy as np
import pytest

from omnilearned.sbi.toy.config import V2_CONFIG, load_toy_config
from omnilearned.sbi.toy.design import reserved_settings

CFG, _, SEL = load_toy_config(V2_CONFIG)


def test_design_is_reproducible_in_box_and_respects_margin():
    from omnilearned.sbi.toy.datasets import BOX, box_arrays, make_design, margin_clashes

    a, info = make_design(200, "train", CFG)
    b, _ = make_design(200, "train", CFG)
    v, _ = make_design(200, "validation", CFG)
    np.testing.assert_array_equal(a, b)
    assert not np.any(np.all(np.isclose(a[:, None], v[None]), -1))  # independent streams
    lo, hi = box_arrays(BOX)
    assert np.all((a >= lo) & (a <= hi))
    assert info["n"] == 200 and not margin_clashes(a, reserved_settings(CFG)).any()
    # The margin applies only when every coordinate is close.
    dev = np.array(CFG["holdout"]["development_validation"])
    close = dev + 0.01 * (hi - lo)
    far_one_axis = close + np.array([0.05 * (hi - lo)[0], 0, 0])
    assert margin_clashes(np.stack([close, far_one_axis]), reserved_settings(CFG)).tolist() == [True, False]


@pytest.fixture(scope="module")
def shard_dir(tmp_path_factory):
    pytest.importorskip("hepyy")
    from omnilearned.sbi.toy.datasets import generate_shard, make_design, write_index

    d = tmp_path_factory.mktemp("shards")
    for split, n in (("train", 4), ("validation", 2)):
        pts, _ = make_design(n, split, CFG)
        half = (n + 1) // 2
        for k, ids in enumerate((np.arange(half), np.arange(half, n))):
            if ids.size:
                generate_shard(d / f"{split}_{k:05d}.h5", split, ids, pts[ids], 60, log=lambda m: None)
        write_index(d, split)
    return d


def test_shard_layout_reproducibility_and_index(shard_dir, tmp_path):
    from omnilearned.sbi.observations import file_sha256
    from omnilearned.sbi.toy.datasets import GLOBAL_NAMES, generate_shard, read_shards, shard_paths

    with open(shard_dir / "index_train.json") as f:
        idx = json.load(f)
    assert idx["n_points"] == 4 and idx["n_jets"] == 240
    with h5py.File(shard_dir / "train_00000.h5") as f:
        assert f["data/x"].shape == (120, 256, 3) and f["data/x"].dtype == np.float32
        assert list(f["global"].attrs["global_names"]) == list(GLOBAL_NAMES)
        assert "diag" not in f and not any(k.startswith("p0") for k in f)
    with h5py.File(shard_dir / "train_00000.diag.h5") as f:
        assert "event_id" in f and len(f.keys()) > 1
    # A shard depends only on its points, not on the job that made it.
    with open(shard_dir / "train_00001.json") as f:
        man = json.load(f)
    pts = np.array([q["theta"] for q in man["points"]])
    ids = np.array([q["point_id"] for q in man["points"]])
    m2 = generate_shard(tmp_path / "train_00001.h5", "train", ids, pts, 60, log=lambda m: None)
    assert m2["sha256"] == man["sha256"] == file_sha256(shard_dir / "train_00001.h5")
    assert [q["discrete_sha256"] for q in m2["points"]] == [q["discrete_sha256"] for q in man["points"]]

    d = read_shards(shard_paths(shard_dir, "train"))
    w = int(d["n_constituents"].max())
    assert d["x"].shape == (240, w, 3) and d["mask"].shape == (240, w)
    np.testing.assert_array_equal(d["mask"].sum(1), d["n_constituents"])
    np.testing.assert_array_equal(d["global"][:, 6], d["n_constituents"])
    assert np.unique(d["event_id"]).size == 240 and sorted(np.unique(d["point_id"])) == [0, 1, 2, 3]
    full = read_shards(shard_paths(shard_dir, "train"), trim=False)
    np.testing.assert_array_equal(full["x"][:, :w], d["x"])
    assert np.all(full["x"][:, w:] == 0)
    ok, _, _ = __import__("omnilearned.sbi.toy.selection", fromlist=["x"]).select_clouds(
        d["x"].astype(np.float64), SEL, d["mask"])
    assert ok.all()


def test_index_rejects_modified_shard(shard_dir, tmp_path):
    import shutil

    from omnilearned.sbi.toy.datasets import write_index

    for p in shard_dir.glob("validation_*"):
        shutil.copy(p, tmp_path / p.name)
    with h5py.File(tmp_path / "validation_00000.h5", "a") as f:
        f.attrs["edited"] = 1
    with pytest.raises(ValueError, match="hash"):
        write_index(tmp_path, "validation")


def test_observations_carry_no_truth(tmp_path):
    pytest.importorskip("hepyy")
    from omnilearned.sbi.observations import read_observations
    from omnilearned.sbi.toy.datasets import make_observations

    sha, _ = make_observations(tmp_path / "obs.h5", tmp_path / "truth" / "t.json",
                               (7.5, 0.30, 0.95), 30, "development", 3)
    obs = read_observations(tmp_path / "obs.h5")
    assert len(obs) == 30 and obs.sha256 == sha
    assert obs.x.shape[1] == int(obs.mask.sum(1).max())
    with h5py.File(tmp_path / "obs.h5") as f:
        assert not any("theta" in k or "truth" in k for k in list(f) + list(f.attrs))
    with open(tmp_path / "truth" / "t.json") as f:
        assert json.load(f)["theta"] == [7.5, 0.30, 0.95]


@pytest.fixture(scope="module")
def tiny_run(shard_dir, tmp_path_factory):
    from omnilearned.sbi.toy import train

    out = tmp_path_factory.mktemp("run")
    train.main(["--data-dir", str(shard_dir), "--out", str(out), "--steps", "6", "--val-every", "3",
                "--warmup-steps", "2", "--dim", "16", "--depth", "1", "--heads", "2",
                "--mult-steps", "4", "--mult-hidden", "8", "--mult-depth", "1",
                "--batch-size", "16", "--device", "cpu", "--length-buckets", "2"])
    return out


def test_train_and_sample_select_drivers(tiny_run):
    from omnilearned.sbi.flow import FlowSurrogate
    from omnilearned.sbi.toy import sample_select

    out = tiny_run
    with open(out / "run.json") as f:
        run = json.load(f)
    assert run["data"]["n_train"] == 240 and run["flow_history"]["best_step"] in (3, 6)
    assert run["host"]["source_revision"] is not None and run["device"]["device"] == "cpu"
    sur = FlowSurrogate.load(out / "checkpoint.pt")
    assert sur.multiplicity is not None and sur.uses_multiplicity_context
    assert sur.metadata["checkpoint_sha256"] == run["checkpoint_sha256"]
    x, mask, rep = sample_select.sample_and_select(sur, (7.5, 0.30, 0.95), 40, SEL, nsteps=4)
    assert rep["generated"] == 40 and x.shape[0] == mask.shape[0] == rep["selected"]
    assert sum(rep["rejections"].values()) == 40 - rep["selected"]


def test_source_revision_of_checkout_and_snapshot(tmp_path):
    from omnilearned.sbi.toy.datasets import source_revision

    rev, dirty = source_revision()
    assert rev is not None and len(rev) == 40 and dirty in (True, False)
    (tmp_path / "SOURCE_REVISION").write_text("a" * 40 + "\n")
    assert source_revision(tmp_path) == ("a" * 40, False)


def test_gate_driver_run_summarize_convergence(shard_dir, tiny_run, tmp_path):
    from omnilearned.sbi.toy import gate
    from omnilearned.sbi.toy.datasets import generate_shard, make_design, write_index

    pts, _ = make_design(2, "validation", CFG)
    fdir = tmp_path / "fid"
    generate_shard(fdir / "fidelity_00000.h5", "fidelity", np.arange(2), pts, 40, log=lambda m: None)
    write_index(fdir, "fidelity")
    with h5py.File(fdir / "fidelity_00000.h5") as f, h5py.File(shard_dir / "validation_00000.h5") as g:
        assert not set(f["event_id"][...]) & set(g["event_id"][...])  # own stream
    ck = str(tiny_run / "checkpoint.pt")
    out = tmp_path / "gate"
    common = ["--checkpoint", ck, "--fidelity-dir", str(fdir), "--n-generate", "30",
              "--device", "cpu"]
    gate.main(["run", *common, "--out", str(out), "--nsteps", "4", "--n-perm", "20",
               "--save-samples"])
    gate.main(["run", *common, "--out", str(out), "--nsteps", "4"])  # resumable: no rerun
    gate.main(["summarize", "--out", str(out), "--n-settings", "4"])
    with open(out / "summary.json") as f:
        assert json.load(f)["alpha_per_point"] == 0.0125
    gate.main(["summarize", "--out", str(out)])
    with open(out / "summary.json") as f:
        summ = json.load(f)
    assert summ["points"] == 2 and summ["alpha_per_point"] == 0.025 and summ["complete"]
    assert {t["point_id"] for t in summ["table"]} == {0, 1}
    with open(out / "point_00000.json") as f:
        p0 = json.load(f)
    assert set(p0["relevance_by_n"]) == {"250", "1000"} and p0["n_direct"] == 40
    with h5py.File(out / "samples_00000.h5") as f:
        assert f["x"].shape[0] == p0["n_surrogate"]
    gate.main(["convergence", *common, "--out", str(tmp_path / "conv.json"), "--points", "1",
               "--steps", "2", "4"])
    with open(tmp_path / "conv.json") as f:
        conv = json.load(f)
    assert "2->4" in conv["points"]["1"]["pairs"]


def test_counts_only_shards_match_full_shards(shard_dir, tmp_path):
    from omnilearned.sbi.toy.datasets import generate_shard, read_counts, read_shards

    with open(shard_dir / "train_00000.json") as f:
        man = json.load(f)
    pts = np.array([q["theta"] for q in man["points"]])
    ids = np.array([q["point_id"] for q in man["points"]])
    m = generate_shard(tmp_path / "train_counts_00000.h5", "train", ids, pts, 60, log=lambda m: None,
                       counts_only=True)
    assert m["schema"] == "toy-counts-v1" and m["diag_file"] is None
    assert m["n_jets"] == 120
    n, th, pid = read_counts([tmp_path / "train_counts_00000.h5"])
    full = read_shards([shard_dir / "train_00000.h5"])
    np.testing.assert_array_equal(n, full["n_constituents"])
    np.testing.assert_array_equal(th, full["theta"])
    np.testing.assert_array_equal(pid, full["point_id"])
    with pytest.raises(ValueError):
        read_counts([shard_dir / "train_00000.h5"])


# ---- closure campaigns (toy.closure) ---------------------------------------

def test_joint_level_of_gaussian_posterior():
    from omnilearned.sbi.scan import Grid, ScanResult
    from omnilearned.sbi.toy.closure import joint_level

    ax = np.linspace(-6, 6, 241)
    g = Grid(["a", "b"], [ax, ax])
    pts = g.points().reshape(241, 241, 2)
    res = ScanResult(g, -0.5 * (pts ** 2).sum(-1), np.zeros((241, 241)))
    for r in (0.5, 1.0, 1.5, 2.5):
        assert abs(joint_level(res, -0.5 * r * r) - (1 - np.exp(-0.5 * r * r))) < 5e-3  # node spacing 0.05


def test_clopper_pearson_and_toml_writer(tmp_path):
    import tomllib

    from omnilearned.sbi.toy.closure import clopper_pearson, dump_toml

    lo, hi = clopper_pearson(7, 10)
    assert abs(lo - 0.3475) < 1e-3 and abs(hi - 0.9333) < 1e-3
    assert clopper_pearson(0, 5)[0] == 0.0 and clopper_pearson(5, 5)[1] == 1.0
    doc = {"a": 1, "s": "x", "grid": {"b": [1.0, 2.0], "c": {"min": 0.0, "max": 1.0, "n": 3}},
           "scan": {"support": {"seed_stride": [2, 2], "delta_logpost": 10.0}, "flag": True}}
    assert tomllib.loads(dump_toml(doc)) == doc


def _campaign_spec(ck, source="toy", table=None, k=3):
    spec = {"schema": "toy-closure-campaign-v1", "name": "unit", "checkpoint": str(ck),
            "setting": "development_validation", "n_events": 20, "n_ensembles": k,
            "index_offset": 900, "source": source, "device": "cpu",
            "fit": {"grid": {"b": [6.5, 7.5, 8.5], "T_ref": [0.28, 0.32], "tau_init": [0.85, 1.05]},
                    "prior": {"type": "uniform", "bounds": {"b": [0.0, 11.0], "T_ref": [0.25, 0.55],
                                                           "tau_init": [0.2, 1.2]}},
                    "likelihood": {"solver": "midpoint", "nsteps": 2, "num_probes": 1,
                                   "probe_schedule": "per_step"},
                    "scan": {"event_chunk": 16, "candidate_chunk": 6}}}
    if table:
        spec["acceptance_table"] = str(table)
    return spec


def test_closure_campaign_prepare_fit_seal_report(tiny_run, tmp_path):
    from omnilearned.sbi.fit import load_fit_config
    from omnilearned.sbi.observations import read_observations
    from omnilearned.sbi.toy import closure
    from omnilearned.sbi.toy.selection import save_table

    axes = [np.linspace(0, 11, 3), np.linspace(0.25, 0.55, 3), np.linspace(0.2, 1.2, 3)]
    pts = np.stack(np.meshgrid(*axes, indexing="ij"), -1)
    acc = 0.9 - 0.1 * (pts[..., 1] - 0.25)  # smooth synthetic Pr(S | theta)
    save_table(tmp_path / "acc.npz", axes, acc, np.zeros(acc.shape + (0,), np.int64), {})
    d = tmp_path / "camp"
    man = closure.prepare(_campaign_spec(tiny_run / "checkpoint.pt", table=tmp_path / "acc.npz"), d,
                          log=lambda m: None)
    assert man["label"] == "direct-simulator closure" and len(man["tasks"]) == 3
    for k in range(3):  # fits see no truth: observations and configs are truth-free
        cfg = load_fit_config(d / "fits" / f"fit_{k:04d}.toml")
        obs = read_observations(cfg["observations"])
        assert len(obs) == 20 and np.all((obs.event_id >> 48) == 4)  # development stream
        assert "truth" not in (d / "fits" / f"fit_{k:04d}.toml").read_text()
    with pytest.raises(FileExistsError):
        closure.prepare(_campaign_spec(tiny_run / "checkpoint.pt"), d)
    # two workers split the tasks; reruns skip sealed fits
    closure.main(["fit", "--dir", str(d), "--worker", "0", "--num-workers", "2"])
    assert sorted(p.parent.name for p in d.glob("fits/*/seal.json")) == ["fit_0000", "fit_0002"]
    closure.main(["fit", "--dir", str(d), "--worker", "1", "--num-workers", "2"])
    assert closure.fit_task(d, 1) is None
    with np.load(d / "fits" / "fit_0001" / "result.npz") as z:
        assert "log_norm" in z.files
    doc = closure.report(d, log=lambda m: None)
    assert doc["n_fits"] == 3 and not doc["refused"]
    assert all(0.0 <= r["level"] <= 1.0 for r in doc["fits"])
    assert doc["fits"][0]["truth"] == {"b": 7.5, "T_ref": 0.30, "tau_init": 0.95}
    agg = doc["aggregate"]
    assert set(agg["mean_pull"]) == {"b", "T_ref", "tau_init"} and agg["levels_ks"]["n"] == 3
    # a result edited after its seal is refused
    res = d / "fits" / "fit_0002" / "result.json"
    res.write_text(res.read_text().replace('"mode"', '"mode" ', 1))
    doc = closure.report(d, log=lambda m: None)
    assert doc["n_fits"] == 2 and doc["refused"][0]["task"] == 2
    # control comparison
    (tmp_path / "ctl").mkdir()
    (tmp_path / "ctl" / "report.json").write_text(json.dumps({"aggregate": agg}))
    doc = closure.report(d, control=tmp_path / "ctl", log=lambda m: None)
    assert set(doc["versus_control"]) == {"b", "T_ref", "tau_init"}


def test_closure_surrogate_control_and_grid_order(tiny_run, tmp_path, monkeypatch):
    from omnilearned.sbi.flow import FlowSurrogate
    from omnilearned.sbi.observations import read_observations
    from omnilearned.sbi.toy import closure, sample_select

    sur = FlowSurrogate.load(tiny_run / "checkpoint.pt")
    with pytest.raises(RuntimeError, match="acceptance too low"):  # the 6-step model selects ~nothing
        closure._surrogate_ensemble(sur, SEL, (7.5, 0.30, 0.95), 5, 0, 0, nsteps=2)

    def accept_all(sur, theta, n, sel, seed=0, nsteps=128, **kw):  # stand-in for a trained model
        x, m = sur.sample(np.asarray(theta), n, None, seed=seed, nsteps=2,
                          candidate_names=["b", "T_ref", "tau_init"])
        return x, m, {"generated": n, "selected": n}

    monkeypatch.setattr(sample_select, "sample_and_select", accept_all)
    d = tmp_path / "ctl"
    man = closure.prepare(_campaign_spec(tiny_run / "checkpoint.pt", source="surrogate", k=2), d,
                          log=lambda m: None)
    assert man["label"].startswith("self-closure control")
    obs = read_observations(d / "obs" / "obs_0001.h5")
    assert len(obs) == 20 and np.all((obs.event_id >> 48) == closure.SELF_CLOSURE_STREAM)
    # a fixed (single-node) axis gets no pull: fit and report a 2D campaign
    spec2 = _campaign_spec(tiny_run / "checkpoint.pt", k=1)
    spec2["fit"]["grid"]["tau_init"] = [0.95]
    closure.prepare(spec2, tmp_path / "fixed", log=lambda m: None)
    closure.fit_task(tmp_path / "fixed", 0, log=lambda m: None)
    rep = closure.report(tmp_path / "fixed", log=lambda m: None)
    assert rep["fits"][0]["fixed"] == ["tau_init"] and set(rep["aggregate"]["mean_pull"]) == {"b", "T_ref"}
    spec = _campaign_spec(tiny_run / "checkpoint.pt", table=tmp_path / "missing.npz")
    spec["fit"]["grid"] = {"T_ref": [0.3], "b": [7.0], "tau_init": [0.9]}
    with pytest.raises(ValueError, match="ordered"):
        closure.prepare(spec, tmp_path / "bad")


def test_acceptance_shards_merge(tiny_run, tmp_path):
    from omnilearned.sbi.toy import closure

    ax = ["6:8:2", "0.28:0.32:2", "0.9:1.0:2"]
    for k in (0, 1):
        closure.main(["acceptance", "--checkpoint", str(tiny_run / "checkpoint.pt"), *sum([["--axis", a] for a in ax], []),
                      "--n-samples", "16", "--nsteps", "2", "--shard", str(k), "--num-shards", "2",
                      "--device", "cpu", "--out", str(tmp_path / f"part{k}.npz")])
    with pytest.raises(ValueError, match="missing"):
        closure.acceptance_merge([tmp_path / "part0.npz"], tmp_path / "t.npz")
    acc = closure.acceptance_merge([tmp_path / "part0.npz", tmp_path / "part1.npz"], tmp_path / "t.npz")
    assert acc.shape == (2, 2, 2) and np.all((acc >= 0) & (acc <= 1))
    with open(tmp_path / "t.json") as f:
        assert json.load(f)["common_random_numbers"] is True


def test_closure_claims_and_interleaved_workers(tiny_run, tmp_path):
    import os
    import socket
    import time

    from omnilearned.sbi.toy import closure

    d1, d2 = tmp_path / "toy", tmp_path / "ctl"
    closure.prepare(_campaign_spec(tiny_run / "checkpoint.pt", k=2), d1, log=lambda m: None)
    spec = _campaign_spec(tiny_run / "checkpoint.pt", k=2)
    spec["index_offset"] = 950
    closure.prepare(spec, d2, log=lambda m: None)
    assert closure.claim(d1, 0) and not closure.claim(d1, 0)  # held by this live process
    lock = d1 / "fits" / "fit_0000" / "claim.lock"
    for stale in ({"job": None, "host": socket.gethostname(), "pid": 2 ** 22 + 12345, "time": time.time()},
                  {"job": None, "host": "elsewhere", "pid": 1, "time": time.time() - 6 * 3600}):
        lock.write_text(json.dumps(stale))  # dead local process / too old
        assert not closure.claim(d1, 0)  # workers never reclaim
        assert closure.unlock_stale([d1], log=lambda m: None) == 1 and closure.claim(d1, 0)
    closure.release(d1, 0)
    os.environ.pop("SLURM_JOB_ID", None)
    assert closure.run_claiming([d1, d2], log=lambda m: None) == 4
    assert not list(tmp_path.glob("*/fits/*/claim.lock"))
    assert all((d / "fits" / f"fit_{k:04d}" / "logp_cache.npz").exists() for d in (d1, d2) for k in (0, 1))
    assert closure.run_claiming([d1, d2], log=lambda m: None) == 0  # all sealed


def test_chunked_transform_fit_and_prepare_match_one_shot():
    import torch

    from omnilearned.sbi.multiplicity import layout_with_multiplicity
    from omnilearned.sbi.toy.train import prepare
    from omnilearned.sbi.transforms import FeatureTransform

    rng = np.random.default_rng(2)
    n, w = 1037, 9
    cnt = rng.integers(2, w + 1, n)
    mask = np.arange(w)[None, :] < cnt[:, None]
    x = np.stack([rng.uniform(-2.4, 2.4, (n, w)), rng.uniform(-1.1, 1.1, (n, w)), rng.normal(1, 1, (n, w))], -1)
    x = np.where(mask[..., None], x, 0.0).astype(np.float32)
    one = FeatureTransform.fit(["eta", "phi", "lpt"], [2.5, 1.2, None], x, mask)
    chk = FeatureTransform.fit(["eta", "phi", "lpt"], [2.5, 1.2, None], x, mask, chunk=100)
    np.testing.assert_allclose(chk.shift, one.shift, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(chk.scale, one.scale, rtol=1e-12)
    data = {"x": x, "mask": mask, "theta": rng.uniform([0, 0.25, 0.2], [11, 0.55, 1.2], (n, 3)),
            "n_constituents": cnt}
    lay = layout_with_multiplicity(["b", "T_ref", "tau_init"], [0, 0.25, 0.2], [11, 0.55, 1.2])
    a, b = prepare(data, one, lay, chunk=100), prepare(data, one, lay, chunk=10 ** 6)
    assert a["u"].dtype == torch.float32 and torch.equal(a["u"], b["u"]) and torch.equal(a["cond"], b["cond"])
    ref = one.forward(torch.as_tensor(x, dtype=torch.float64), torch.as_tensor(mask))[0].float()
    assert torch.equal(a["u"], ref)


def _claim_racer(args):
    import time

    from omnilearned.sbi.toy import closure

    d, start = args
    while time.time() < start:
        pass
    return closure.claim(d, 0)


def test_concurrent_stale_reclaim_has_one_winner(tmp_path):
    import multiprocessing as mp
    import time

    from omnilearned.sbi.toy import closure

    stale = {"job": None, "host": "elsewhere", "pid": 1, "time": time.time() - 6 * 3600}
    ctx = mp.get_context("fork")
    for rep in range(20):
        d = tmp_path / f"c{rep}"
        lock = d / "fits" / "fit_0000" / "claim.lock"
        lock.parent.mkdir(parents=True)
        lock.write_text(json.dumps(stale))
        with ctx.Pool(8) as pool:  # stale lock in place: nobody claims (no worker reclaim)
            assert sum(pool.map(_claim_racer, [(d, time.time() + 0.3)] * 8)) == 0
        closure.unlock_stale([d], log=lambda m: None)
        with ctx.Pool(8) as pool:  # after the single cleanup: exactly one winner
            wins = pool.map(_claim_racer, [(d, time.time() + 0.3)] * 8)
        assert sum(wins) == 1, (rep, wins)
        assert json.loads(lock.read_text())["host"] != "elsewhere"
        assert not list(lock.parent.glob("claim.lock.stale.*"))


def test_unlock_stale_keeps_live_locks(tmp_path):
    import os
    import socket
    import time

    from omnilearned.sbi.toy import closure

    for k, info in enumerate([{"job": None, "host": "x", "pid": 1, "time": time.time() - 6 * 3600},
                              {"job": None, "host": socket.gethostname(), "pid": os.getpid(), "time": time.time()}]):
        p = tmp_path / "fits" / f"fit_{k:04d}" / "claim.lock"
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps(info))
    assert closure.unlock_stale([tmp_path], log=lambda m: None) == 1
    assert not (tmp_path / "fits" / "fit_0000" / "claim.lock").exists()
    assert (tmp_path / "fits" / "fit_0001" / "claim.lock").exists()
