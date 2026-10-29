"""JEWEL events -> jet records in the toy-v2 layout.

Per event: constituent subtraction of the raw HepMC record (:mod:`cs`), neutrinos removed
after the subtraction (the order the subtraction's authors prescribe), anti-kt clustering of
the subtracted final state, then each kept jet's constituents as massless (η, φ, log pT),
pT-ordered and left-packed, with φ rotated so that *that jet's* axis is at 0 (JEWEL's fixed
reaction plane is the only φ information). The selection is then the toy-v2 one,
``simulator_v2.select_sets`` on the records themselves (re-clustering the constituents alone
gives one jet containing all, jet pT and |η| cuts), the same function applied to surrogate
samples.

Which jets are kept is ``[selection].jet_multiplicity``:

* ``inclusive`` (default): every clustered jet, as experiments measure. Jets of one event
  share an ``event_id``, so their correlation stays recoverable downstream — the summed
  likelihood over them is a composite likelihood, see the closure coverage check.
* ``leading`` / ``subleading``: the hardest / second-hardest jet only. A leading-only sample
  is biased towards jets that interacted less with the medium (in a dijet the less quenched
  one tends to lead), which is why it is not the default.
"""

import numpy as np

from omnilearned.sbi.jewel import cs
from omnilearned.sbi.toy import reco
from omnilearned.sbi.toy.simulator_v2 import REJECTION_REASONS, Selection, select_sets

NEUTRINOS = (12, 14, 16)
FINAL = (1, 4)
REASONS = ("no_jet",) + REJECTION_REASONS
JET_MULTIPLICITY = ("inclusive", "leading", "subleading")


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


def _cluster(px, py, pz, e, radius):
    """Anti-kt of one event: the reco result (``membership`` is the pT-ordered jet index)."""
    p4 = np.stack([px, py, pz, e], -1)[None]
    return reco.cluster_events(p4, radius)


def _axis(px, py, pz, e, member):
    """(pT, eta, phi) of a jet from its constituents.

    ``reco.cluster_events`` reports ``jet_pt/eta/phi`` for the leading jet only, so every
    other jet's axis is the four-momentum sum of its members. For the leading jet this
    reproduces the reported axis (asserted in the tests).
    """
    sx, sy, sz = px[member].sum(), py[member].sum(), pz[member].sum()
    pt = float(np.hypot(sx, sy))
    eta = float(np.arcsinh(sz / pt)) if pt > 0 else 0.0
    return pt, eta, float(np.arctan2(sy, sx))


def _leading(px, py, pz, e, radius):
    r = _cluster(px, py, pz, e, radius)
    return r, (r["membership"][0] == 0) if r["njets"][0] else np.zeros(len(px), bool)


def jet_indices(njets, mode):
    """Indices of the clustered jets to keep (jets are pT-ordered, 0 is the hardest)."""
    if mode not in JET_MULTIPLICITY:
        raise ValueError(f"selection.jet_multiplicity must be one of {JET_MULTIPLICITY}")
    if njets <= 0:
        return []
    if mode == "leading":
        return [0]
    if mode == "subleading":
        return [1] if njets > 1 else []
    return list(range(njets))


def event_jets(ev, cfg, sel):
    """(records, rejection reason or None, diagnostics) for one event: one entry per kept jet.

    ``records`` and ``diagnostics`` are parallel lists, empty when the event gives no jet (then
    the reason is set). Jets of one event therefore share the event's number, and so the same
    ``event_id`` once the caller builds it.
    """
    mode = cfg["selection"].get("jet_multiplicity", "inclusive")
    s = subtracted(ev, cfg["subtraction"])
    base = {"n_thermal": int((ev["status"] == 3).sum()),
            "thermal_pt": float(np.hypot(ev["px"], ev["py"])[ev["status"] == 3].sum())}
    if len(s["px"]) == 0:
        return [], "no_jet", []
    r = _cluster(s["px"], s["py"], s["pz"], s["e"], sel.jet_radius)
    idx = jet_indices(int(r["njets"][0]), mode)
    if not idx:
        return [], "no_jet", []
    # unsubtracted leading jet: event-level context, repeated on every jet (diagnostics only)
    raw = raw_final_state(ev) & ~np.isin(np.abs(ev["pid"]), NEUTRINOS)
    rr, rm = _leading(ev["px"][raw], ev["py"][raw], ev["pz"][raw], ev["e"][raw], sel.jet_radius)
    recs, diags, reason = [], [], None
    for j in idx:
        member = r["membership"][0] == j
        if not member.any():
            continue
        pt = np.hypot(s["px"], s["py"])[member]
        eta = np.arcsinh(s["pz"][member] / pt)
        phi = np.arctan2(s["py"][member], s["px"][member])
        tag = s["tag"][member]
        order = np.argsort(-pt, kind="stable")
        pt, eta, phi, tag = pt[order], eta[order], phi[order], tag[order]
        jpt, jeta, jphi = _axis(s["px"], s["py"], s["pz"], s["e"], member)
        rot = jphi if cfg["selection"].get("rotate_to_jet_axis", True) else 0.0
        phi = np.angle(np.exp(1j * (phi - rot)))
        if len(pt) > sel.max_constituents:
            reason = "too_many_constituents"  # drop this jet, keep the event's other jets
            continue
        recs.append({"eta": eta, "phi": phi, "pt": pt})
        diags.append({**base, "jet_index": int(j), "phi_rotation": rot,
                      "event_jet_pt": jpt,
                      "event_jet_eta": jeta, "n_in_jet": int(member.sum()),
                      "leftover_pt_in_jet": float(pt[tag == cs.TAG_THERMAL_LEFT].sum()),
                      "n_leftover_in_jet": int((tag == cs.TAG_THERMAL_LEFT).sum()),
                      "unsubtracted_jet_pt": float(rr["jet_pt"][0]) if rr["njets"][0] else 0.0,
                      "unsubtracted_n": int(rm.sum())})
    return recs, (reason if not recs else None), diags


def event_jet(ev, cfg, sel):
    """Single-jet form of :func:`event_jets` (first kept jet), for callers that want one."""
    recs, why, diags = event_jets(ev, cfg, sel)
    if not recs:
        return None, why or "no_jet", (diags[0] if diags else {})
    return recs[0], None, diags[0]


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
