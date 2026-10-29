"""Inclusive / leading / subleading jet selection and the Sobol training design (JEWEL v2)."""

import json
import tomllib
from pathlib import Path

import numpy as np
import pytest

from omnilearned.sbi.jewel import cards, generate, jets

V2 = Path(__file__).resolve().parents[2] / "configs" / "jewel-2.6.0-cs-v2.toml"


def test_jet_indices_modes():
    assert jets.jet_indices(3, "inclusive") == [0, 1, 2]
    assert jets.jet_indices(3, "leading") == [0]
    assert jets.jet_indices(3, "subleading") == [1]
    # degenerate counts
    assert jets.jet_indices(1, "subleading") == []  # no second jet
    assert jets.jet_indices(1, "leading") == [0]
    for mode in jets.JET_MULTIPLICITY:
        assert jets.jet_indices(0, mode) == []
    with pytest.raises(ValueError):
        jets.jet_indices(2, "second-hardest")


def _two_jet_event():
    """One event with two well-separated hard jets (pT ~ 120 and ~60) plus soft junk."""
    rng = np.random.default_rng(0)
    px, py, pz, e, pid, status = [], [], [], [], [], []

    def add(pt, eta, phi, st=1, pdg=211):
        px.append(pt * np.cos(phi))
        py.append(pt * np.sin(phi))
        pz.append(pt * np.sinh(eta))
        e.append(pt * np.cosh(eta))
        pid.append(pdg)
        status.append(st)

    for frac in rng.dirichlet(np.ones(12)):  # jet A, eta 0.3 phi 0.0
        add(frac * 120.0, 0.3 + 0.03 * rng.normal(), 0.0 + 0.03 * rng.normal())
    for frac in rng.dirichlet(np.ones(10)):  # jet B, eta -0.4 phi pi (back to back)
        add(frac * 60.0, -0.4 + 0.03 * rng.normal(), np.pi + 0.03 * rng.normal())
    for _ in range(6):  # thermal
        add(0.5 + rng.random(), rng.normal(), rng.uniform(-np.pi, np.pi), st=3, pdg=21)
    return {"px": np.array(px), "py": np.array(py), "pz": np.array(pz), "e": np.array(e),
            "pid": np.array(pid), "status": np.array(status), "number": 1,
            "centrality": 10.0, "weight": 1.0, "scale": 150.0}


def _cfg(mode):
    cfg = json.loads(json.dumps(cards.load_config()))
    cfg["selection"]["jet_multiplicity"] = mode
    return cfg


def test_event_jets_inclusive_contains_leading_and_subleading():
    pytest.importorskip("hepyy")
    ev = _two_jet_event()
    sel = jets.selection_from_config(_cfg("inclusive"))
    inc, _, dinc = jets.event_jets(ev, _cfg("inclusive"), sel)
    lead, _, dlead = jets.event_jets(ev, _cfg("leading"), sel)
    sub, _, dsub = jets.event_jets(ev, _cfg("subleading"), sel)

    assert len(inc) >= 2, "the two hard jets must both be clustered"
    assert len(lead) == 1 and len(sub) == 1
    # jets are pT-ordered, so inclusive[0] is the leading one and inclusive[1] the subleading
    assert dinc[0]["jet_index"] == 0 and dlead[0]["jet_index"] == 0
    assert dsub[0]["jet_index"] == 1
    assert dinc[0]["event_jet_pt"] > dinc[1]["event_jet_pt"]
    np.testing.assert_allclose(inc[0]["pt"], lead[0]["pt"])
    np.testing.assert_allclose(inc[1]["pt"], sub[0]["pt"])
    # each jet is rotated to its OWN axis, so the rotations differ
    assert dinc[0]["phi_rotation"] != dinc[1]["phi_rotation"]
    # the leading jet's axis, recomputed from its constituents, must match the clusterer's
    # own report (reco.cluster_events gives jet_pt/eta/phi for the leading jet only)
    from omnilearned.sbi.jewel.jets import _cluster, subtracted
    s = subtracted(ev, _cfg("inclusive")["subtraction"])
    r = _cluster(s["px"], s["py"], s["pz"], s["e"], sel.jet_radius)
    assert dinc[0]["event_jet_pt"] == pytest.approx(float(r["jet_pt"][0]), rel=1e-9)
    assert dinc[0]["event_jet_eta"] == pytest.approx(float(r["jet_eta"][0]), rel=1e-9)
    assert dinc[0]["phi_rotation"] == pytest.approx(float(r["jet_phi"][0]), abs=1e-9)
    # ... and every jet's constituents sit around phi = 0 after its own rotation
    for r in (inc[0], inc[1]):
        assert np.abs(np.asarray(r["phi"])).max() < 1.0


def test_event_jet_single_form_matches_first_inclusive_jet():
    pytest.importorskip("hepyy")
    ev, cfg = _two_jet_event(), _cfg("inclusive")
    sel = jets.selection_from_config(cfg)
    recs, _, _ = jets.event_jets(ev, cfg, sel)
    one, why, _ = jets.event_jet(ev, cfg, sel)
    assert why is None
    np.testing.assert_allclose(one["pt"], recs[0]["pt"])


def test_v2_config_window_and_backend():
    cfg = tomllib.load(open(V2, "rb"))
    assert cfg["backend"] == "jewel-2.6.0-cs-v2"          # new backend id: new sample definition
    assert cfg["jewel"]["ptmin_GeV"] == 10.0 and cfg["jewel"]["ptmax_GeV"] == 600.0
    assert cfg["jewel"]["weighted"] is True
    s = cfg["selection"]
    assert s["jet_pt_range_GeV"] == [20.0, 500.0] and s["jet_abs_eta_max"] == 2.0
    assert s["jet_multiplicity"] == "inclusive"
    # the pT-hat range must bracket the jet window on both sides, or the spectrum is truncated
    assert cfg["jewel"]["ptmin_GeV"] < s["jet_pt_range_GeV"][0]
    assert cfg["jewel"]["ptmax_GeV"] > s["jet_pt_range_GeV"][1]
    # constituents must reach a jet at the eta edge plus its cone
    assert s["constituent_abs_eta_max"] >= s["jet_abs_eta_max"] + s["jet_radius"]


def test_train_sobol_design_count_and_reservation_margin():
    cfg = tomllib.load(open(V2, "rb"))
    pts = generate.design_points(cfg, "train")
    assert pts.shape == (int(cfg["design"]["train_points"]), 3)
    lo = np.array([0.0, 0.1, 0.2])
    hi = np.array([11.0, 0.9, 1.0])
    assert (pts >= lo).all() and (pts <= hi).all()
    hold = [cfg["holdout"]["development_validation"], cfg["holdout"]["primary_test"]]
    hold += list(cfg["holdout"]["additional_test_settings"])
    for h in hold:  # rule 2: no training point within the margin on EVERY axis at once
        rel = np.abs(pts - np.asarray(h, float)) / (hi - lo)
        assert not np.all(rel < 0.02, axis=1).any(), f"training design clashes with holdout {h}"


def test_cartesian_design_still_used_without_train_points():
    """Configs written before ``train_points`` keep the historical Cartesian grid."""
    cfg = tomllib.load(open(V2, "rb"))
    del cfg["design"]["train_points"]
    pts = generate.design_points(cfg, "train")
    n = int(np.prod([len(cfg["design"][p]) for p in ("b", "T_ref", "tau_init")]))
    assert pts.shape == (n, 3)
