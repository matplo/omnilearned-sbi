"""JEWEL events -> leading-jet records in the toy-v2 layout.

Per event: constituent subtraction of the raw HepMC record (:mod:`cs`), neutrinos removed
after the subtraction (the order the subtraction's authors prescribe), anti-kt clustering of
the subtracted final state, the leading jet's constituents as massless (η, φ, log pT),
pT-ordered and left-packed, with φ rotated so that the jet axis is at 0 (JEWEL's fixed
reaction plane is the only φ information). The selection is then the toy-v2 one,
``simulator_v2.select_sets`` on the records themselves (re-clustering the constituents alone
gives one jet containing all, jet pT and |η| cuts), the same function applied to surrogate
samples.
"""

import numpy as np

from omnilearned.sbi.jewel import cs
from omnilearned.sbi.toy import reco
from omnilearned.sbi.toy.simulator_v2 import REJECTION_REASONS, Selection, select_sets

NEUTRINOS = (12, 14, 16)
FINAL = (1, 4)
REASONS = ("no_jet",) + REJECTION_REASONS


def selection_from_config(cfg):
    return Selection.from_config(cfg["selection"])


def raw_final_state(ev):
    """Boolean [n]: final-state particles as stored (status 1/4, not dummies)."""
    fin = np.isin(ev["status"], FINAL)
    return fin & ~((ev["pid"] == 111) & (np.abs(ev["e"] - 1e-6) < 1e-11))


def subtracted(ev, sub):
    """Subtracted final state of one event: dict px, py, pz, e, pid, tag (neutrinos removed)."""
    out = cs.subtract(ev["status"], ev["pid"], ev["px"], ev["py"], ev["pz"], ev["e"],
                      dRmax=float(sub["dRmax"]), thermal_pid=sub["thermal_pid"])
    keep = ~np.isin(np.abs(out["pid"]), NEUTRINOS) & (out["e"] > 0)
    if sub.get("leftovers", "keep") == "drop":
        keep &= out["tag"] <= cs.TAG_PASSTHROUGH
    return {k: v[keep] for k, v in out.items()}


def _leading(px, py, pz, e, radius):
    p4 = np.stack([px, py, pz, e], -1)[None]
    r = reco.cluster_events(p4, radius)
    return r, (r["membership"][0] == 0) if r["njets"][0] else np.zeros(len(px), bool)


def event_jet(ev, cfg, sel):
    """(record or None, rejection reason or None, diagnostics) for one event."""
    s = subtracted(ev, cfg["subtraction"])
    diag = {"n_thermal": int((ev["status"] == 3).sum()),
            "thermal_pt": float(np.hypot(ev["px"], ev["py"])[ev["status"] == 3].sum())}
    if len(s["px"]) == 0:
        return None, "no_jet", diag
    r, member = _leading(s["px"], s["py"], s["pz"], s["e"], sel.jet_radius)
    if not member.any():
        return None, "no_jet", diag
    pt = np.hypot(s["px"], s["py"])[member]
    eta = np.arcsinh(s["pz"][member] / pt)
    phi = np.arctan2(s["py"][member], s["px"][member])
    tag = s["tag"][member]
    order = np.argsort(-pt, kind="stable")
    pt, eta, phi, tag = pt[order], eta[order], phi[order], tag[order]
    rot = float(r["jet_phi"][0]) if cfg["selection"].get("rotate_to_jet_axis", True) else 0.0
    phi = np.angle(np.exp(1j * (phi - rot)))
    raw = raw_final_state(ev) & ~np.isin(np.abs(ev["pid"]), NEUTRINOS)
    rr, rm = _leading(ev["px"][raw], ev["py"][raw], ev["pz"][raw], ev["e"][raw], sel.jet_radius)
    diag.update({"phi_rotation": rot, "event_jet_pt": float(r["jet_pt"][0]),
                 "event_jet_eta": float(r["jet_eta"][0]), "n_in_jet": int(member.sum()),
                 "leftover_pt_in_jet": float(pt[tag == cs.TAG_THERMAL_LEFT].sum()),
                 "n_leftover_in_jet": int((tag == cs.TAG_THERMAL_LEFT).sum()),
                 "unsubtracted_jet_pt": float(rr["jet_pt"][0]) if rr["njets"][0] else 0.0,
                 "unsubtracted_n": int(rm.sum())})
    if len(pt) > sel.max_constituents:
        return None, "too_many_constituents", diag
    return {"eta": eta, "phi": phi, "pt": pt}, None, diag


def pack(records, width):
    """Left-packed arrays [n, width] of a list of records."""
    n = len(records)
    eta, phi, pt = (np.zeros((n, width)) for _ in range(3))
    mask = np.zeros((n, width), bool)
    for i, r in enumerate(records):
        k = len(r["pt"])
        eta[i, :k], phi[i, :k], pt[i, :k] = r["eta"], r["phi"], r["pt"]
        mask[i, :k] = True
    return eta, phi, pt, mask


def select_records(records, sel):
    """Apply ``select_sets`` to candidate records: (accepted [n], reason names [n], reco)."""
    eta, phi, pt, mask = pack(records, sel.max_constituents)
    ok, reason, r = select_sets(eta, phi, pt, mask, sel)
    names = [None if k < 0 else REJECTION_REASONS[k] for k in reason]
    x = np.where(mask[..., None], np.stack([eta, phi, np.log(np.where(mask, pt, 1.0))], -1), 0.0)
    return ok, names, r, x, mask
