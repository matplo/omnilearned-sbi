"""Drawing closure pseudo data from generated JEWEL shards (``source = "jewel"``).

JEWEL pseudo data cannot be made on the fly the way the toy's can, so a campaign draws it
from a split generated at the reserved settings. What matters here: ensembles are disjoint,
the requested theta is the one selected, an event's jets keep a shared ``event_id`` (so the
composite-likelihood correlation stays visible), and asking for more than exists fails loudly
instead of silently recycling jets.
"""

import numpy as np
import pytest

from omnilearned.sbi.toy.closure import jewel_ensembles
from omnilearned.sbi.toy.datasets import write_shard

THETAS = [(4.5, 0.45, 0.65), (1.5, 0.65, 0.35)]


def _shard(path, n_per_theta=40, jets_per_event=2, width=6):
    """A merged-style shard at two theta, with several jets sharing each event_id."""
    rng = np.random.default_rng(3)
    parts = []
    for pid, th in enumerate(THETAS):
        x = np.zeros((n_per_theta, width, 3), np.float32)
        mask = np.zeros((n_per_theta, width), bool)
        for i in range(n_per_theta):
            k = int(rng.integers(2, width + 1))
            x[i, :k] = rng.normal(size=(k, 3)).astype(np.float32)
            mask[i, :k] = True
        # jets of one event share an event_id, as the inclusive JEWEL worker writes them
        ev = np.repeat(np.arange(n_per_theta // jets_per_event, dtype=np.int64),
                       jets_per_event) + pid * 10_000
        parts.append((pid, th, {"x": x, "mask": mask, "event_id": ev,
                                "jet_pt": np.full(n_per_theta, 50.0),
                                "jet_eta": np.zeros(n_per_theta),
                                "jet_phi": np.zeros(n_per_theta),
                                "diag": {"pthat": np.full(n_per_theta, 60.0)}}))
    write_shard(path, parts, "test", "jewel-2.6.0-cs-v2", "0" * 64)


def test_jewel_ensembles_are_disjoint_and_at_the_requested_theta(tmp_path):
    _shard(tmp_path / "test_00000.h5")
    n, K = 8, 3
    recs, info = jewel_ensembles(tmp_path, "test", THETAS[0], n, K)

    assert len(recs) == K and all(r["x"].shape[0] == n for r in recs)
    assert info["jets_used"] == n * K and info["jets_available_at_theta"] == 40
    # disjoint: no jet appears in two ensembles (event_id+row identity via exact x match)
    flat = np.concatenate([r["x"].reshape(n, -1) for r in recs])
    assert np.unique(flat, axis=0).shape[0] == n * K
    # the other theta's jets must not leak in: theta 1 used event_ids >= 10000
    ids = np.concatenate([r["event_id"] for r in recs])
    assert (ids < 10_000).all()
    # inclusive sampling keeps several jets per event -- the composite-likelihood signal
    assert info["jets_per_event"] > 1.0
    assert info["distinct_events"] < info["jets_used"]


def test_jewel_ensembles_refuse_to_recycle_jets(tmp_path):
    _shard(tmp_path / "test_00000.h5")
    with pytest.raises(ValueError, match="need"):
        jewel_ensembles(tmp_path, "test", THETAS[0], 30, 2)  # 60 > 40 available


def test_jewel_ensembles_unknown_theta_is_an_error(tmp_path):
    _shard(tmp_path / "test_00000.h5")
    with pytest.raises(ValueError, match="has 0 jets"):
        jewel_ensembles(tmp_path, "test", (9.5, 0.25, 0.45), 2, 1)


def test_load_sample_config_handles_toy_and_jewel():
    from pathlib import Path

    from omnilearned.sbi.toy.closure import load_sample_config

    root = Path(__file__).resolve().parents[2] / "configs"
    cfg, sel = load_sample_config(root / "jewel-2.6.0-cs-v2.toml")
    assert cfg["backend"] == "jewel-2.6.0-cs-v2"
    assert tuple(sel.jet_pt_range_GeV) == (20.0, 500.0)
    cfg2, sel2 = load_sample_config(root / "toy-quenching-v2.toml")
    assert cfg2["backend"].startswith("toy")
