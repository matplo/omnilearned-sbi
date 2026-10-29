"""toy-quenching-v2: the v1 toy with variable constituent multiplicity.

Same medium, geometry, hard spectrum, energy loss and broadening as
toy-quenching-v1 (``simulator``), with two changes:

* Multiplicities are Poisson. Core particles: ``1 + Poisson(mu_core - 1)``
  with ``mu_core = core_mean (P0 / pt_scale)^core_mean_pt_exponent``.
  Medium-induced soft particles: ``Poisson(mu_soft)`` with
  ``mu_soft = soft_mean_baseline + soft_mean_amplitude (1 - exp(-soft_mean_rate I1))``,
  so the multiplicity carries information on the medium.
* The observed object is the leading anti-kt jet of the event with its actual
  constituents (particles may leak out of the cone). Constituents are
  pT-ordered and left-packed into ``max_constituents`` slots (256), padded
  with zeros and an explicit mask.

The selection is defined on the constituent set alone (``select_sets``) and is
applied identically to toy jets and to surrogate samples, so the selected
region S is the same on both sides of the likelihood normalization:
real constituents finite with pT > 0, ``min_constituents <= N <=
max_constituents``, all constituents inside the fiducial |eta|, |phi| bounds,
re-clustering the constituents alone gives exactly one jet containing all of
them, jet pT in range and |eta_jet| below the cut.
"""

from dataclasses import dataclass, asdict, fields

import numpy as np

from omnilearned.sbi.toy import reco
from omnilearned.sbi.toy.simulator import (
    PARAMETERS,  # noqa: F401  (re-exported for drivers)
    UNITS,  # noqa: F401
    exit_distance,
    make_event_id,
    path_moments,
    sample_production_points,
)

BACKEND = "toy-quenching-v2"


@dataclass(frozen=True)
class ToyConstants:
    geometry_radius_fm: float = 6.5
    temperature_cut_GeV: float = 0.17
    temperature_scale_GeV: float = 0.35
    length_scale_fm: float = 1.0
    integration_steps: int = 64
    power_law_exponent: float = 5.0
    hard_pt_min_GeV: float = 40.0
    hard_pt_max_GeV: float = 500.0
    loss_coefficient: float = 0.12
    loss_lognormal_sigma: float = 0.30
    core_mean: float = 20.0
    core_mean_pt_exponent: float = 0.3
    soft_mean_baseline: float = 4.0
    soft_mean_amplitude: float = 26.0
    soft_mean_rate: float = 0.8
    max_generated_particles: int = 512
    dirichlet_core_alpha: float = 0.7
    dirichlet_soft_alpha: float = 0.9
    soft_fraction_baseline: float = 0.03
    soft_fraction_amplitude: float = 0.17
    soft_fraction_rate: float = 0.8
    vacuum_width_rad: float = 0.08
    vacuum_width_pt_exponent: float = -0.15
    broadening_width_rad: float = 0.04
    soft_width_ratio: float = 2.2
    pt_scale_GeV: float = 100.0
    center_eta_range: tuple = (-0.5, 0.5)
    center_phi_range_rad: tuple = (-0.6, 0.6)

    @classmethod
    def from_config(cls, toy):
        names = {f.name for f in fields(cls)}
        kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in toy.items()
              if k in names}
        unknown = set(toy) - names
        if unknown:
            raise ValueError(f"unknown toy constants {sorted(unknown)}")
        return cls(**kw)


@dataclass(frozen=True)
class Selection:
    jet_radius: float = 0.4
    jet_pt_range_GeV: tuple = (60.0, 200.0)
    jet_abs_eta_max: float = 1.0
    constituent_abs_eta_max: float = 2.5
    constituent_abs_phi_max_rad: float = 1.2
    constituent_pt_min_GeV: float = 0.0
    min_constituents: int = 2
    max_constituents: int = 256
    max_attempts_per_requested_event: int = 1000

    @classmethod
    def from_config(cls, sel):
        names = {f.name for f in fields(cls)}
        kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in sel.items()
              if k in names}
        if sel.get("jet_algorithm", "antikt") != "antikt":
            raise ValueError("only anti-kt is implemented")
        return cls(**kw)


REJECTION_REASONS = (
    "invalid_weights",
    "no_jet",
    "too_many_constituents",
    "too_few_constituents",
    "fiducial_constituent",
    "not_one_jet",
    "jet_pt",
    "jet_eta",
)
_R = {name: k for k, name in enumerate(REJECTION_REASONS)}


# ---------------------------------------------------------------------------
# Event generation
# ---------------------------------------------------------------------------


def _padded_dirichlet(alpha, counts, rng):
    """Dirichlet weights over ``counts[i]`` slots, zero beyond; [n, max(counts)]."""
    k = int(counts.max()) if counts.size else 0
    g = rng.standard_gamma(alpha, size=(counts.size, max(k, 1)))
    live = np.arange(g.shape[1])[None, :] < counts[:, None]
    g = np.where(live, g, 0.0)
    s = g.sum(1, keepdims=True)
    return np.divide(g, s, out=np.zeros_like(g), where=s > 0), live


@dataclass(frozen=True)
class PtBias:
    """Sampling bias of the hard pT ``P0``: a sampling option, not part of the toy law.

    ``P0`` is drawn from the generator law times b(P0) = (min(P0, cap_GeV) / ref_GeV)^exponent,
    i.e. pT^-(n - k) below the cap and pT^-n above it, and each selected jet carries the weight
    1 / b(P0), normalised so that the weights sum to the number of selected jets at the point
    (as in Pythia ``PhaseSpace:bias2Selection``). Weighted jets follow the unbiased law. It is
    kept out of ``[toy]``, so the backend id and the config hash are those of the toy.
    """

    exponent: float = 0.0
    ref_GeV: float = 100.0
    cap_GeV: float = None

    @property
    def active(self):
        return self.exponent != 0

    def factor(self, p0):
        cap = np.inf if self.cap_GeV is None else self.cap_GeV
        return (np.minimum(p0, cap) / self.ref_GeV) ** self.exponent

    def to_dict(self):
        return {"exponent": float(self.exponent), "ref_GeV": float(self.ref_GeV),
                "cap_GeV": None if self.cap_GeV is None else float(self.cap_GeV)}


def _power_law_mass(lo, hi, a):
    """Integral of x^-a over [lo, hi]."""
    return np.log(hi / lo) if a == 1 else (hi ** (1 - a) - lo ** (1 - a)) / (1 - a)


def _power_law_inverse(u, lo, hi, a):
    """Inverse CDF of x^-a on [lo, hi] at u in [0, 1)."""
    if a == 1:
        return lo * (hi / lo) ** u
    l, h = lo ** (1 - a), hi ** (1 - a)
    return (l + u * (h - l)) ** (1 / (1 - a))


def draw_hard_pt(u, c, bias=None):
    """Hard pT ``P0`` from uniforms ``u``: pT^-n on [min, max], times the bias if active."""
    nexp = c.power_law_exponent
    if bias is None or not bias.active:
        lo, hi = c.hard_pt_min_GeV ** (1 - nexp), c.hard_pt_max_GeV ** (1 - nexp)
        return (lo + u * (hi - lo)) ** (1 / (1 - nexp))
    lo, hi, k = c.hard_pt_min_GeV, c.hard_pt_max_GeV, bias.exponent
    cap = hi if bias.cap_GeV is None else min(float(bias.cap_GeV), hi)
    if cap <= lo:
        raise ValueError("the bias cap must lie above the hard-pT minimum")
    m1 = _power_law_mass(lo, cap, nexp - k)  # below the cap: pT^-(n-k) (ref cancels)
    m2 = cap ** k * _power_law_mass(cap, hi, nexp) if cap < hi else 0.0
    t = u * (m1 + m2)
    below = t < m1
    p = _power_law_inverse(np.where(below, t / m1, 0.0), lo, cap, nexp - k)
    if m2 > 0:
        p = np.where(below, p, _power_law_inverse(np.where(below, 0.0, (t - m1) / m2), cap, hi, nexp))
    return p


def generate_attempts(theta, n, rng, c, b_per_event=None, bias=None):
    """Raw (unselected) events; particle arrays padded to the largest count.

    ``bias`` (``PtBias``) changes only the hard-pT draw; the weights are set by ``sample_toy``.
    """
    theta = tuple(float(v) for v in theta)
    b, T_ref, tau = theta
    if b_per_event is not None:
        b = np.asarray(b_per_event, dtype=np.float64)
        theta = (b, T_ref, tau)
    if tau <= 0:
        raise ValueError("tau_init must be positive")
    R = c.geometry_radius_fm
    x0, y0 = sample_production_points(b, n, rng, R)
    psi = rng.uniform(0, 2 * np.pi, n)
    nx, ny = np.cos(psi), np.sin(psi)
    L = exit_distance(x0, y0, nx, ny, b, R)
    I = path_moments(x0, y0, nx, ny, L, theta, c)
    b_event = np.broadcast_to(b, (n,)).astype(np.float64)

    P0 = draw_hard_pt(rng.uniform(0, 1, n), c, bias)
    s = c.loss_lognormal_sigma
    Z = np.exp(s * rng.standard_normal(n) - s ** 2 / 2)
    P_vis = P0 * np.exp(-c.loss_coefficient * I[:, 2] * Z)

    ratio = P0 / c.pt_scale_GeV
    mu_core = c.core_mean * ratio ** c.core_mean_pt_exponent
    mu_soft = c.soft_mean_baseline + c.soft_mean_amplitude * (
        1 - np.exp(-c.soft_mean_rate * I[:, 0]))
    n_core = 1 + rng.poisson(np.maximum(mu_core - 1, 0.0))
    n_soft = rng.poisson(mu_soft)
    too_many = n_core + n_soft > c.max_generated_particles
    n_core = np.where(too_many, np.minimum(n_core, c.max_generated_particles), n_core)
    n_soft = np.where(too_many, np.maximum(c.max_generated_particles - n_core, 0), n_soft)

    w_core, live_core = _padded_dirichlet(c.dirichlet_core_alpha, n_core, rng)
    w_soft, live_soft = _padded_dirichlet(c.dirichlet_soft_alpha, n_soft, rng)
    f_soft = c.soft_fraction_baseline + c.soft_fraction_amplitude * (
        1 - np.exp(-c.soft_fraction_rate * I[:, 0]))
    f_soft = np.where(n_soft > 0, f_soft, 0.0)
    weights = np.concatenate([w_core * (1 - f_soft)[:, None], w_soft * f_soft[:, None]], 1)
    live = np.concatenate([live_core, live_soft], 1)
    pt = P_vis[:, None] * weights

    sig_vac = c.vacuum_width_rad * ratio ** c.vacuum_width_pt_exponent
    sig_core = np.sqrt(sig_vac ** 2 + c.broadening_width_rad ** 2 * I[:, 1] / ratio)
    sig_soft = c.soft_width_ratio * sig_core
    sig = np.concatenate([np.repeat(sig_core[:, None], w_core.shape[1], 1),
                          np.repeat(sig_soft[:, None], w_soft.shape[1], 1)], 1)
    deta = rng.standard_normal(sig.shape) * sig
    dphi = rng.standard_normal(sig.shape) * sig
    eta0 = rng.uniform(*c.center_eta_range, n)
    phi0 = rng.uniform(*c.center_phi_range_rad, n)
    eta = np.where(live, eta0[:, None] + deta, 0.0)
    phi = np.where(live, phi0[:, None] + dphi, 0.0)
    pt = np.where(live, pt, 0.0)
    component = np.concatenate([np.zeros(w_core.shape, np.int8), np.ones(w_soft.shape, np.int8)], 1)
    component = np.where(live, component, -1).astype(np.int8)

    real_ok = np.where(live, np.isfinite(pt) & (pt > 0), True)
    valid = np.all(real_ok, 1) & ~too_many
    return {
        "eta": eta, "phi": phi, "pt": pt, "live": live, "valid": valid,
        "diag": {
            "x0": x0, "y0": y0, "psi": psi, "L_exit": L, "I1": I[:, 0],
            "I2": I[:, 1], "I3": I[:, 2], "P0": P0, "Z_loss": Z, "P_vis": P_vis,
            "f_soft": f_soft, "sigma_core": sig_core, "eta0": eta0, "phi0": phi0,
            "n_core": n_core, "n_soft": n_soft, "b": b_event,
        },
        "component": component,
    }


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def select_sets(eta, phi, pt, mask, sel):
    """Selection on constituent sets [n, P] (left-packed, explicit mask).

    The same function is applied to toy jets and to surrogate samples.
    Returns (accepted [n], reason [n] (-1 if accepted), reco dict).
    """
    mask = np.asarray(mask, dtype=bool)
    n = mask.shape[0]
    reason = np.full(n, -1, dtype=np.int8)
    count = mask.sum(1)
    real_ok = np.where(mask, np.isfinite(eta) & np.isfinite(phi) & np.isfinite(pt) & (pt > 0), True)
    ok = np.all(real_ok, 1)
    reason[~ok] = _R["invalid_weights"]
    for cut, name in ((count > sel.max_constituents, "too_many_constituents"),
                      (count < sel.min_constituents, "too_few_constituents")):
        reason[ok & cut] = _R[name]
        ok &= ~cut
    fid = np.all(np.where(mask, (np.abs(eta) < sel.constituent_abs_eta_max)
                          & (np.abs(phi) < sel.constituent_abs_phi_max_rad)
                          & (pt > sel.constituent_pt_min_GeV), True), 1)
    reason[ok & ~fid] = _R["fiducial_constituent"]
    ok &= fid
    safe = mask & ok[:, None]
    p4 = reco.four_vectors(np.where(safe, eta, 0.0), np.where(safe, phi, 0.0),
                           np.where(safe, pt, 0.0))
    p4 = np.where(safe[..., None], p4, 0.0)
    r = reco.cluster_events(p4, sel.jet_radius)
    one = (r["njets"] == 1) & (r["jet_nconst"] == count)
    reason[ok & ~one] = _R["not_one_jet"]
    ok &= one
    ptok = (r["jet_pt"] >= sel.jet_pt_range_GeV[0]) & (r["jet_pt"] <= sel.jet_pt_range_GeV[1])
    reason[ok & ~ptok] = _R["jet_pt"]
    ok &= ptok
    etaok = np.abs(r["jet_eta"]) < sel.jet_abs_eta_max
    reason[ok & ~etaok] = _R["jet_eta"]
    ok &= etaok
    return ok, reason, r


def pack_leading_jet(ev, sel, width=None):
    """Constituents of each event's leading anti-kt jet, pT-ordered and left-packed.

    Returns dict with eta, phi, pt, mask [n, width] (width defaults to the
    largest jet in the batch, at most ``max_constituents``), count [n],
    has_jet [n] and the full-event clustering.
    """
    live = ev["live"] & ev["valid"][:, None]
    p4 = reco.four_vectors(np.where(live, ev["eta"], 0.0), np.where(live, ev["phi"], 0.0),
                           np.where(live, ev["pt"], 0.0))
    p4 = np.where(live[..., None], p4, 0.0)
    r = reco.cluster_events(p4, sel.jet_radius)
    member = (r["membership"] == 0) & live
    count = member.sum(1)
    key = np.where(member, -ev["pt"], np.inf)  # pT-descending, non-members last
    order = np.argsort(key, axis=1, kind="stable")
    w = int(width or max(1, min(int(count.max()) if count.size else 1, sel.max_constituents)))
    order = order[:, :w]
    take = lambda a: np.take_along_axis(a, order, 1)  # noqa: E731
    mask = take(member)
    return {"eta": np.where(mask, take(ev["eta"]), 0.0), "phi": np.where(mask, take(ev["phi"]), 0.0),
            "pt": np.where(mask, take(ev["pt"]), 0.0), "mask": mask, "count": count,
            "has_jet": r["njets"] > 0, "event_reco": r}


def apply_selection(ev, sel):
    """Toy selection: leading jet of the event, then ``select_sets`` on its constituents."""
    jets = pack_leading_jet(ev, sel)
    n = jets["mask"].shape[0]
    reason = np.full(n, -1, dtype=np.int8)
    ok = ev["valid"].copy()
    reason[~ok] = _R["invalid_weights"]
    reason[ok & ~jets["has_jet"]] = _R["no_jet"]
    ok &= jets["has_jet"]
    big = jets["count"] > sel.max_constituents  # cannot be packed; rejected
    reason[ok & big] = _R["too_many_constituents"]
    ok &= ~big
    s_ok, s_reason, r = select_sets(jets["eta"], jets["phi"], jets["pt"], jets["mask"], sel)
    reason[ok & ~s_ok] = s_reason[ok & ~s_ok]
    ok &= s_ok
    return ok, reason, r, jets


def sample_toy(theta, count, rng, constants, selection, stream=0, index=0,
               batch=None, attempt_cap=None, b_sampler=None, bias=None):
    """Generate ``count`` selected jets at ``theta``.

    Returns (records, manifest). Records: ``x`` [n, max_constituents, 3]
    (eta, phi, log pT; zeros where padded), ``mask``, ``n_constituents``,
    jet kinematics, event ids and the hidden ``diag`` store. With an active
    ``bias`` (``PtBias``) they also carry ``weight`` (1 / b(P0), summing to the
    number of selected jets) and ``pthat`` (= P0).
    """
    sel = selection
    W = sel.max_constituents
    cap = attempt_cap or sel.max_attempts_per_requested_event * max(count, 1)
    batch = batch or max(256, min(20000, 2 * count))
    parts, attempts, n_acc = [], 0, 0
    reasons = np.zeros(len(REJECTION_REASONS), dtype=np.int64)
    while n_acc < count and attempts < cap:
        m = int(min(batch, cap - attempts))
        bvec = None if b_sampler is None else b_sampler(m, rng)
        ev = generate_attempts(theta, m, rng, constants, b_per_event=bvec, bias=bias)
        ok, reason, r, jets = apply_selection(ev, sel)
        for k in range(len(REJECTION_REASONS)):
            reasons[k] += int((reason == k).sum())
        idx = np.flatnonzero(ok)
        pad = lambda a: np.pad(a[idx], ((0, 0), (0, W - a.shape[1])))  # noqa: E731
        msk = pad(jets["mask"])
        diag = {k: v[idx] for k, v in ev["diag"].items()}
        diag["n_in_jet"] = jets["count"][idx]
        parts.append({
            "eta": pad(jets["eta"]), "phi": pad(jets["phi"]), "pt": pad(jets["pt"]),
            "mask": msk, "n_constituents": msk.sum(1).astype(np.int32),
            "jet_pt": r["jet_pt"][idx], "jet_eta": r["jet_eta"][idx], "jet_phi": r["jet_phi"][idx],
            "event_id": make_event_id(stream, index, attempts + idx), "diag": diag,
        })
        n_acc += idx.size
        attempts += m
    rec = _concat(parts, count)
    accepted = rec["event_id"].shape[0]
    weighting = None
    if bias is not None and bias.active:
        rec["pthat"] = rec["diag"]["P0"].copy()
        w = 1.0 / bias.factor(rec["pthat"])
        rec["weight"] = w * (accepted / w.sum()) if accepted else w
        w = rec["weight"]
        weighting = {**bias.to_dict(), "weight_sum": float(w.sum()),
                     "ess": float(w.sum() ** 2 / np.sum(w ** 2)) if accepted else 0.0,
                     "weight_range": [float(w.min()), float(w.max())] if accepted else None}
    manifest = {
        "backend": BACKEND,
        "theta": dict(zip(PARAMETERS, map(float, theta))),
        "local_b": b_sampler is not None,
        "requested": int(count),
        "accepted": int(accepted),
        "attempted": int((rec["event_id"][-1] & 0xFFFFFFFF) + 1) if accepted else 0,
        "attempts_generated": int(attempts),
        "accepted_generated": int(n_acc),
        "acceptance": float(n_acc / attempts) if attempts else 0.0,
        "rejections": dict(zip(REJECTION_REASONS, reasons.tolist())),
        "max_constituents": int(W),
        "n_constituents_range": ([int(rec["n_constituents"].min()), int(rec["n_constituents"].max())]
                                 if accepted else None),
        "complete": bool(accepted >= count),
        "attempt_cap": int(cap),
        "stream": int(stream),
        "index": int(index),
    }
    if weighting is not None:
        manifest["sampling_bias"] = weighting
    return rec, manifest


def _concat(parts, count):
    out = {}
    for k in parts[0]:
        if k == "diag":
            out[k] = {d: np.concatenate([p[k][d] for p in parts])[:count] for d in parts[0][k]}
        else:
            out[k] = np.concatenate([p[k] for p in parts])[:count]
    m = out["mask"]
    logpt = np.log(np.where(m, out["pt"], 1.0))
    out["x"] = np.where(m[..., None], np.stack([out["eta"], out["phi"], logpt], -1), 0.0)
    return out


def constants_dict(c):
    return asdict(c)
