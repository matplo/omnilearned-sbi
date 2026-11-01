"""toy.train on JEWEL v3 merged files: the v3-native reader, jet_pt conditioning, no
standardisation. See docs/PLAN_JEWEL_V3.md P3 and docs/FORMAT_JEWEL_V3.md.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from omnilearned.sbi.jewel import cards
from omnilearned.sbi.jewel.v3format import merge_v3, read_v3_shards, shard_paths_v3, write_jets

V3_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "jewel-2.6.0-cs-v3.toml"


def _v3_jets(n, P, seed, point_id, theta):
    """A small valid v3 payload at one theta point, all jets sharing an event."""
    rng = np.random.default_rng(seed)
    x = np.zeros((n, P, 5))
    mask = np.zeros((n, P), bool)
    for i in range(n):
        k = int(rng.integers(2, P + 1))
        pt = np.sort(rng.gamma(2.0, 3.0, k))[::-1] + 0.2
        deta = rng.normal(0, 0.12, k)
        e = np.sqrt((pt * np.cosh(deta)) ** 2)
        x[i, :k, 0] = deta
        x[i, :k, 1] = rng.normal(0, 0.12, k)
        x[i, :k, 2] = np.log(pt)
        x[i, :k, 3] = np.log(e)
        x[i, :k, 4] = rng.choice([211, -211, 22, 2112], k)
        mask[i, :k] = True
    return dict(
        x=x, mask=mask, theta=np.tile(theta, (n, 1)),
        jet_pt=rng.uniform(20, 500, n), jet_eta=rng.uniform(-2, 2, n),
        jet_phi=rng.uniform(-np.pi, np.pi, n), jet_mass=rng.uniform(1, 30, n),
        event_id=np.arange(n, dtype=np.int64) + point_id * 10_000,
        group_id=np.arange(n, dtype=np.int64) + point_id * 10_000,
        point_id=np.full(n, point_id, np.int32),
        weight=np.ones(n), pthat=rng.uniform(10, 600, n),
    )


def _merged_split(tmp_path, split, points, n_per_point=25, P=10):
    """Write per-point v3 shards, then merge_v3 them into a split's training-ready directory."""
    src = tmp_path / f"{split}_tasks"
    src.mkdir()
    recs = []
    for pid, theta in enumerate(points):
        d = _v3_jets(n_per_point, P, seed=pid, point_id=pid, theta=theta)
        name = f"{split}_p{pid:05d}_c000.h5"
        sha = write_jets(src / name, **d)
        recs.append({"file": name, "sha256": sha, "point_id": pid, "chunk": 0,
                     "n_jets": n_per_point, "events": None})
    (src / f"index_{split}.json").write_text(json.dumps(
        {"schema": "jewel-index-v1", "split": split, "shards": recs, "tasks_done": len(points)}))
    out = tmp_path / "merged" / split
    merge_v3(src, out, split, target_jets=10**9, cfg=cards.load_config(V3_CONFIG), log=lambda *a: None)
    return out


@pytest.fixture
def merged_dirs(tmp_path):
    train = _merged_split(tmp_path, "train", [[4.5, 0.45, 0.65], [8.0, 0.75, 0.3]])
    val = _merged_split(tmp_path, "validation", [[2.0, 0.2, 0.9]])
    return train, val


def test_shard_paths_v3_and_reader_round_trip(merged_dirs):
    train, _ = merged_dirs
    paths = shard_paths_v3(train, "train")
    assert paths and all(p.exists() for p in paths)
    data = read_v3_shards(paths)
    assert data["x"].shape[-1] == 4, "pid (column 4) must be dropped for the modeled features"
    assert data["mask"].sum(1).tolist() == data["n_constituents"].tolist()
    assert "jet_pt" in data["global_names"]
    assert np.unique(data["point_id"]).tolist() == [0, 1]


def test_v3_layouts_condition_on_jet_pt_and_skip_standardisation(merged_dirs, tmp_path):
    """The flow's layout carries n_constituents AND jet_pt; the multiplicity model's never
    n_constituents (it predicts that), only jet_pt -- the distinction
    test_multiplicity_with_known_context.py guards at the module level, exercised here through
    build_surrogate. The feature transform must be the identity (no affine standardisation).
    """
    import argparse

    from omnilearned.sbi.jewel.v3format import read_v3_shards, shard_paths_v3
    from omnilearned.sbi.multiplicity import MULTIPLICITY_COLUMN
    from omnilearned.sbi.toy.train import build_surrogate

    train_dir, val_dir = merged_dirs
    train = read_v3_shards(shard_paths_v3(train_dir, "train"))
    val = read_v3_shards(shard_paths_v3(val_dir, "validation"))
    args = argparse.Namespace(config=str(V3_CONFIG), arch="set", dim=16, depth=1, heads=2,
                              local_dim=8, mult_hidden=8, mult_depth=1, mult_steps=2,
                              weight_mode="none", device="cpu", tf32=False, save_after=0)
    sha, hist, mhist, meta = build_surrogate(args, train, val, tmp_path, log=lambda m: None)
    assert sha and hist["best_step"] >= 0 and mhist["best_val_nll"] == mhist["best_val_nll"]

    from omnilearned.sbi.flow import FlowSurrogate
    sur = FlowSurrogate.load(tmp_path / "checkpoint.pt")
    assert sur.layout.known == [MULTIPLICITY_COLUMN, "jet_pt"]
    assert sur.multiplicity.layout.known == ["jet_pt"], \
        "the multiplicity model must never condition on n_constituents -- it predicts that"
    t = sur.transform
    assert t.shift == [0.0] * 4 and t.scale == [1.0] * 4 and all(b is None for b in t.bounds), \
        "v3 must apply no feature standardisation (docs/FORMAT_JEWEL_V3.md)"


def test_v3_train_end_to_end_writes_a_loadable_checkpoint(merged_dirs, tmp_path):
    from omnilearned.sbi.flow import FlowSurrogate
    from omnilearned.sbi.toy import train

    train_dir, val_dir = merged_dirs
    out = tmp_path / "run"
    train.main(["--data-dir", str(train_dir), "--val-dir", str(val_dir), "--config", str(V3_CONFIG),
               "--out", str(out), "--steps", "4", "--val-every", "2", "--warmup-steps", "1",
               "--dim", "16", "--depth", "1", "--heads", "2", "--mult-steps", "4",
               "--mult-hidden", "8", "--mult-depth", "1", "--batch-size", "8", "--device", "cpu"])
    run = json.loads((out / "run.json").read_text())
    assert run["meta"]["backend"] == "jewel-2.6.0-cs-v3"
    assert run["meta"]["box"]["T_ref"] == [0.1, 0.9]  # the v3 box, not toy.datasets.BOX (0.25-0.55)
    sur = FlowSurrogate.load(out / "checkpoint.pt")
    assert sur.layout.dim == 3 + 2  # theta (3) + n_constituents + jet_pt
