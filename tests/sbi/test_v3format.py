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
