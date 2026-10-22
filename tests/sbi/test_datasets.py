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


def test_train_and_sample_select_drivers(shard_dir, tmp_path):
    from omnilearned.sbi.flow import FlowSurrogate
    from omnilearned.sbi.toy import sample_select, train

    out = tmp_path / "run"
    train.main(["--data-dir", str(shard_dir), "--out", str(out), "--steps", "6", "--val-every", "3",
                "--warmup-steps", "2", "--dim", "16", "--depth", "1", "--heads", "2",
                "--mult-steps", "4", "--mult-hidden", "8", "--mult-depth", "1",
                "--batch-size", "16", "--device", "cpu"])
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
