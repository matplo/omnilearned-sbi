"""The v3 jet format: invariants, and that OmniLearned loads it unmodified.

Normative spec: docs/FORMAT_JEWEL_V3.md. The point of these tests is that the format is
checked against OmniLearned's *actual* dataloader rather than against our reading of it.
"""

import numpy as np
import pytest

from omnilearned.sbi.jewel.v3format import (
    FEATURES,
    GLOBALS,
    read_jets,
    write_jets,
)


def _jets(n=40, P=12, seed=0, n_events=None):
    """A small valid v3 payload; several jets share an event when n_events < n."""
    rng = np.random.default_rng(seed)
    x = np.zeros((n, P, 5))
    mask = np.zeros((n, P), bool)
    for i in range(n):
        k = int(rng.integers(2, P + 1))
        pt = np.sort(rng.gamma(2.0, 3.0, k))[::-1] + 0.2
        m = rng.uniform(0.0, 0.2, k)                       # E^2 = pT^2 cosh^2(eta) + m^2
        deta = rng.normal(0, 0.12, k)
        e = np.sqrt((pt * np.cosh(deta)) ** 2 + m ** 2)
        x[i, :k, 0] = deta
        x[i, :k, 1] = rng.normal(0, 0.12, k)
        x[i, :k, 2] = np.log(pt)
        x[i, :k, 3] = np.log(e)
        x[i, :k, 4] = rng.choice([211, -211, 22, 2112], k)
        mask[i, :k] = True
    ne = n_events or n
    return dict(
        x=x, mask=mask,
        theta=np.tile([4.5, 0.45, 0.65], (n, 1)),
        jet_pt=rng.uniform(20, 500, n), jet_eta=rng.uniform(-2, 2, n),
        jet_phi=rng.uniform(-np.pi, np.pi, n), jet_mass=rng.uniform(1, 30, n),
        event_id=np.arange(n, dtype=np.int64),
        group_id=np.repeat(np.arange(ne, dtype=np.int64), int(np.ceil(n / ne)))[:n],
        point_id=np.zeros(n, np.int32),
        weight=rng.uniform(0.1, 5.0, n), pthat=rng.uniform(10, 600, n),
    )


def test_roundtrip_and_column_names(tmp_path):
    p = tmp_path / "train_00000.h5"
    d = _jets()
    write_jets(p, **d)
    got = read_jets(p)
    assert got["feature_names"] == FEATURES
    assert got["global_names"] == GLOBALS
    np.testing.assert_allclose(got["data"], d["x"].astype(np.float32), rtol=0, atol=0)
    np.testing.assert_array_equal(got["mask"], d["mask"])
    # global carries n_constituents in its last column, derived from the mask
    np.testing.assert_array_equal(got["global"][:, GLOBALS.index("n_constituents")],
                                  d["mask"].sum(1).astype(np.float32))


def test_padding_is_exactly_zero_and_logE_ge_logpT(tmp_path):
    d = _jets()
    got = read_jets(write_jets(tmp_path / "a.h5", **d) and tmp_path / "a.h5")
    x, m = got["data"], got["mask"]
    assert np.all(x[~m] == 0), "padded slots must be exactly zero in every feature"
    assert np.all(x[..., 3][m] >= x[..., 2][m] - 1e-6), "log E must be >= log pT"


def test_writer_refuses_malformed_payloads(tmp_path):
    d = _jets()
    bad = {**d, "x": d["x"].copy()}
    bad["x"][0, -1, 0] = 0.5                       # write into a padded slot
    with pytest.raises(ValueError, match="padded"):
        write_jets(tmp_path / "b.h5", **bad)

    bad = {**d, "x": d["x"].copy()}
    bad["x"][0, 0, 3] = bad["x"][0, 0, 2] - 1.0    # log E below log pT
    with pytest.raises(ValueError, match="log E"):
        write_jets(tmp_path / "c.h5", **bad)

    # event_id must be unique: it seeds the Hutchinson probes, so duplicates would hand two
    # jets identical probe vectors and correlate their divergence errors
    bad = {**d, "event_id": np.zeros(len(d["mask"]), np.int64)}
    with pytest.raises(ValueError, match="unique"):
        write_jets(tmp_path / "d.h5", **bad)


def test_inclusive_jets_share_an_event_but_not_an_id(tmp_path):
    d = _jets(n=40, n_events=10)
    got = read_jets(write_jets(tmp_path / "e.h5", **d) and tmp_path / "e.h5")
    assert np.unique(got["event_id"]).size == len(got["event_id"])
    assert np.unique(got["group_id"]).size == 10          # the correlation stays recoverable


def test_omnilearned_hepdataset_loads_a_v3_file(tmp_path):
    """The format is only useful if OmniLearned's own loader accepts it."""
    torch = pytest.importorskip("torch")
    from omnilearned.dataloader import HEPDataset

    p = tmp_path / "train_00000.h5"
    n = 40
    write_jets(p, **_jets(n=n))
    ds = HEPDataset([str(p)], file_indices=[(0, i) for i in range(n)],
                    use_cond=True, use_pid=True)
    assert len(ds) == n
    s = ds[0]
    # use_pid splits column 4 out of X, leaving the four kinematic features
    assert s["X"].shape[1] == len(FEATURES) - 1, s["X"].shape
    assert s["pid"].dtype == torch.int32
    assert s["cond"].shape[0] == len(GLOBALS)
    # OmniLearned's own validity rule must agree with our explicit mask
    got = read_jets(p)
    assert int((s["X"][:, 2] != 0).sum()) == int(got["mask"][0].sum())


def test_omnilearned_clip_inputs_assumptions_hold(tmp_path):
    """clip_inputs encodes the column semantics; our data must satisfy them."""
    pytest.importorskip("torch")
    p = tmp_path / "f.h5"
    write_jets(p, **_jets())
    got = read_jets(p)
    x, m = got["data"], got["mask"]
    # columns 0,1 are relative: every real constituent sits inside the cone clip (R < 0.8)
    dr = np.hypot(x[..., 0], x[..., 1])
    assert np.all(dr[m] < 0.8), f"max dR {dr[m].max():.3f} would be clipped by OmniLearned"
    # column 3 is already >= column 2, so clip_inputs is a no-op on our files
    np.testing.assert_array_equal(np.maximum(x[..., 3], x[..., 2])[m], x[..., 3][m])


def test_worker_end_to_end_writes_a_valid_v3_file(tmp_path):
    """The generation worker, with a fake JEWEL binary, must produce a loadable v3 file.

    This is the P1 gate: representation, format and the worker agree, and the result is
    something OmniLearned can train on.
    """
    pytest.importorskip("hepyy")
    import json
    import sys
    from pathlib import Path

    from omnilearned.sbi.jewel import cards, generate

    from test_jewel import FAKE  # the fake JEWEL used by the v1/v2 worker tests

    root = Path(__file__).resolve().parents[2]
    cfg = cards.load_config(root / "configs" / "jewel-2.6.0-cs-v3.toml")
    exe = tmp_path / "fake_jewel.py"
    exe.write_text(FAKE.format(python=sys.executable))
    exe.chmod(0o755)
    cfg["jewel"]["executable"] = str(exe)
    cfg["jewel"]["launcher"] = []
    cfg["jewel"]["env_scripts"] = []

    d = tmp_path / "probe"
    generate.plan(cfg, "probe", d, chunks=1, theta=[[4.5, 0.45, 0.65]], nevent=60,
                  keep_events=True, sampling={"wexpo": 3.0})
    generate.work(d, scratch=str(tmp_path), max_tasks=1)

    man = json.loads(next(d.glob("probe_*.json")).read_text())
    assert man["backend"] == "jewel-2.6.0-cs-v3"
    assert man["rejections"].get("too_many_constituents", 0) == 0
    assert (d / f"{man['task']['name']}.events.h5").exists(), "event stores must be written"

    got = read_jets(d / f"{man['task']['name']}.h5")
    assert got["feature_names"] == FEATURES
    x, m = got["data"], got["mask"]
    if m.any():
        # the representation invariants, on real worker output
        assert np.all(x[~m] == 0)
        assert np.all(x[..., 3][m] >= x[..., 2][m] - 1e-6)
        assert np.abs(x[..., 0][m]).max() < 1.0, "deta must be relative to the jet axis"
        assert np.unique(got["event_id"]).size == got["event_id"].size
        # several jets of one event share a group_id but never an event_id
        assert np.unique(got["group_id"]).size <= got["group_id"].size


def test_merge_keeps_points_whole_and_normalises_weights_per_point(tmp_path):
    """Merging must not split a theta point across files.

    The per-point normalisation (sum w = n_jets at that point) is only exact if every jet of
    a point is in the same file, and a reader can then hold one point from one file.
    """
    import json

    from omnilearned.sbi.jewel.v3format import merge_v3, plan_merge_groups

    # grouping: whole points, ~target per file
    shards = [{"point_id": p, "n_jets": 20000, "file": f"f{p}"} for p in range(512)]
    groups = plan_merge_groups(shards, 1_000_000)
    seen = [{s["point_id"] for s in g} for g in groups]
    assert all(not (a & b) for i, a in enumerate(seen) for b in seen[i + 1:])
    assert all(sum(s["n_jets"] for s in g) <= 1_000_000 for g in groups)

    # end to end on small files, with deliberately un-normalised weights
    src = tmp_path / "tasks"
    src.mkdir()
    rng = np.random.default_rng(3)
    recs = []
    for pid in range(4):
        d = _jets(n=10, P=8, seed=pid)
        d["point_id"] = np.full(10, pid, np.int32)
        d["event_id"] = np.arange(pid * 10, pid * 10 + 10, dtype=np.int64)
        d["weight"] = rng.uniform(0.1, 9.0, 10)          # arbitrary scale
        name = f"test_p{pid:05d}_c000.h5"
        sha = write_jets(src / name, **d)
        recs.append({"file": name, "sha256": sha, "point_id": pid, "chunk": 0,
                     "n_jets": 10, "events": None})
    (src / "index_test.json").write_text(json.dumps(
        {"schema": "jewel-index-v1", "split": "test", "shards": recs, "tasks_done": 4}))

    out = tmp_path / "merged"
    idx = merge_v3(src, out, "test", target_jets=25, log=lambda *a: None)
    assert idx["n_jets"] == 40
    got = read_jets(out / idx["files"][0]["file"])
    for q in np.unique(got["point_id"]):
        sl = got["point_id"] == q
        assert np.isclose(got["weight"][sl].sum(), sl.sum()), "sum(w) must equal n_jets per point"
    # every point lands in exactly one file
    where = {}
    for fspec in idx["files"]:
        for q in np.unique(read_jets(out / fspec["file"])["point_id"]):
            where.setdefault(int(q), []).append(fspec["file"])
    assert all(len(v) == 1 for v in where.values()), where
