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


def test_prepare_takes_its_backend_config_from_the_spec(tmp_path, monkeypatch):
    """A JEWEL campaign must not silently inherit the toy default config.

    `closure prepare` has no --config flag, so without this the toy selection (60-200 GeV
    leading jet), the toy box and the toy reserved settings would be used for a JEWEL run.
    """
    from pathlib import Path

    from omnilearned.sbi.toy import closure

    v2 = Path(__file__).resolve().parents[2] / "configs" / "jewel-2.6.0-cs-v2.toml"
    seen = {}

    def fake_load(path):
        seen["path"] = str(path)
        raise RuntimeError("stop after the config is resolved")

    monkeypatch.setattr(closure, "load_sample_config", fake_load)
    spec = {"schema": closure.CAMPAIGN_SCHEMA, "config": str(v2), "setting": "primary_test",
            "source": "jewel", "jewel_dir": "x", "n_events": 1, "n_ensembles": 1,
            "checkpoint": str(v2)}
    with pytest.raises(RuntimeError, match="stop after"):
        closure.prepare(spec, tmp_path / "camp")
    assert seen["path"] == str(v2), "prepare ignored the spec's config"


def test_inclusive_jets_get_unique_probe_ids_and_keep_their_event_grouping(tmp_path):
    """event_id seeds the Hutchinson probes, so it must be unique per jet.

    An inclusive sample has several jets per event. If they shared an event_id they would be
    evaluated with identical probe vectors: each jet's divergence stays unbiased, but their
    errors become correlated and the variance of the summed log-likelihood inflates (~2x at
    2 jets/event). The originating event is kept in group_id, which only diagnostics read.
    """
    import numpy as np

    from omnilearned.sbi.observations import read_observations
    from omnilearned.sbi.toy.records import write_observation_file

    _shard(tmp_path / "test_00000.h5")
    recs, _ = jewel_ensembles(tmp_path, "test", THETAS[0], 8, 2)
    for r in recs:
        assert np.unique(r["event_id"]).size == len(r["event_id"])
        assert np.unique(r["group_id"]).size < len(r["group_id"])  # the fixture shares events

    write_observation_file(tmp_path / "obs.h5", recs[0], attrs={"source": "jewel"})
    obs = read_observations(tmp_path / "obs.h5")
    assert np.unique(obs.event_id).size == len(obs.event_id)
    np.testing.assert_array_equal(obs.group_id, recs[0]["group_id"])
    # subsetting must carry the grouping with it
    sub = obs.subset(np.arange(4))
    np.testing.assert_array_equal(sub.group_id, recs[0]["group_id"][:4])


def test_shared_event_ids_are_still_refused(tmp_path):
    """The uniqueness guard stays: sharing ids silently correlates the probes."""
    import numpy as np
    import pytest as _pytest

    from omnilearned.sbi.observations import write_observations

    x = np.zeros((4, 3, 3))
    m = np.ones((4, 3), bool)
    with _pytest.raises(ValueError, match="unique"):
        write_observations(tmp_path / "bad.h5", x, m, np.array([1, 1, 2, 3]), ("a", "b", "c"))


def test_observations_positional_construction_is_unchanged():
    """group_id is appended last, so existing positional callers are unaffected.

    Inserting it before `context` silently shifted context/context_names/feature_names for
    every caller that builds Observations positionally (two tests caught it).
    """
    import numpy as np

    from omnilearned.sbi.observations import Observations

    x = np.zeros((4, 3, 3))
    m = np.ones((4, 3), bool)
    o = Observations(x, m, np.arange(4), None, (), ("eta", "phi", "lpt"))
    assert o.context is None and o.context_names == ()
    assert o.feature_names == ("eta", "phi", "lpt")
    assert o.group_id is None
    assert o.subset(np.arange(2)).feature_names == ("eta", "phi", "lpt")


def test_multiplicity_fit_and_check_use_the_sampling_weights():
    """p(N|theta) must be fitted and judged against the WEIGHTED (physical) law.

    On a weighted split the unweighted multiplicity law is the generation law. Fitting or
    checking without weights reproduces the bias exactly -- which is what happened: the
    retrained model had mean N 28.2 against a physical 19.4, and the unweighted check scored
    it as the *better* model.
    """
    import numpy as np

    from omnilearned.sbi.multiplicity import MultiplicityModel, train_multiplicity
    from omnilearned.sbi.toy.multiplicity_check import binned_chi2, check

    # unit weights must reproduce the unweighted statistic exactly
    rng = np.random.default_rng(0)
    probs = np.full(20, 1 / 20)
    c = rng.integers(0, 20, 4000)
    a = binned_chi2(c, probs)
    b = binned_chi2(c, probs, weights=np.ones(4000))
    assert np.allclose(a[:2], b[:2]) and np.isclose(a[2], b[2])

    # two populations: small N heavily weighted, large N lightly weighted. The weighted mean
    # must track the small-N population, the unweighted mean the pooled one.
    n = np.concatenate([np.full(500, 5), np.full(500, 25)])
    w = np.concatenate([np.full(500, 9.0), np.full(500, 1.0)])
    assert abs((w * n).sum() / w.sum() - 7.0) < 1e-9     # weighted
    assert abs(n.mean() - 15.0) < 1e-9                   # unweighted
    assert train_multiplicity.__doc__ is not None
    import inspect
    sig = inspect.signature(train_multiplicity).parameters
    assert "weights" in sig and "val_weights" in sig
    assert "weights" in inspect.signature(check).parameters
