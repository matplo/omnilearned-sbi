"""JEWEL constituent subtraction: numpy port of the authors' Rivet projection
SubtractedJewelEvent (Milhano & Zapp, Eur. Phys. J. C 82 (2022) 1010, arXiv:2207.14814),
reproducing it step by step.

Ported unchanged from ``matplo/sandbox`` ``jewel/jewel_cs2.py`` (validated there against a
line-by-line C++ port of SubtractedJewelEvent on 200 edge-case events: identical rows, tags
and PDG ids, four-momenta to 4e-14). Not yet validated on real JEWEL 2.6.0 output; the JEWEL
probe compares it with JEWEL's built-in subtraction (DOSUBTRACTION T, RSUB = dRmax).
doi:10.1140/epjc/s10052-022-10954-1), reproducing it step by step.

    out = subtract(status, pid, px, py, pz, e, dRmax=0.5,     # one event, raw HepMC arrays
                   thermal_pid="reference", rng=None)
    out -> dict(px, py, pz, e, pid, tag)

What the reference does (SubtractedJewelEvent.cc) and the tag each output row gets:
  final state  = status 1 or 4, dummies skipped (fuzzyEquals(E, 1e-6));
                 E - |pz| <= 0 entries are passed through unsubtracted          -> tag 1
  thermal      = status 3; E - |pz| <= 0 entries are put into the event as-is    -> tag 3
  pairs        = all (particle, thermal) sorted by dR = sqrt(dphi^2 + dy^2), stop at dR > dRmax;
                 pT and m_delta = sqrt(m^2 + pT^2) - pT exchanged independently
  survivors    = particles with pT > 0, rebuilt from (pT, m_delta, y, phi), own PDG id -> tag 0
  leftovers    = thermal momenta with pT > 0 are ADDED to the event as particles  -> tag 2
PDG id of the thermal rows (tags 2, 3), switch thermal_pid:
  "reference"  211 always: what the shipped C++ does (rand()/RAND_MAX is integer division)
  "intended"   pi+, pi-, pi0 with probability 1/3 each (one draw per row)
  "cast"       the C++ with the division fixed: it draws twice, r1 < 1/3 -> pi+,
               else r2 < 2/3 -> pi-, else pi0, i.e. P = 1/3, 4/9, 2/9
Random modes use rng (numpy Generator); pass a seeded one for reproducible output.
Drop tags 2 and 3 to get the "leftovers dropped" convention (jewel_cs.py v1).
"""
import numpy as np

try:
    from numba import njit
except ImportError:
    njit = None

TAG_SUBTRACTED, TAG_PASSTHROUGH, TAG_THERMAL_LEFT, TAG_THERMAL_DEGENERATE = 0, 1, 2, 3
THERMAL_PID_MODES = ("reference", "intended", "cast")


def thermal_pids(n, mode="reference", rng=None):
    """PDG ids for n thermal-momentum rows (see module docstring)."""
    if mode == "reference":
        return np.full(n, 211, dtype=np.int32)
    rng = np.random.default_rng() if rng is None else rng
    if mode == "intended":
        return rng.choice(np.array([211, -211, 111], dtype=np.int32), size=n)
    if mode == "cast":
        r1, r2 = rng.random(n), rng.random(n)
        return np.where(r1 < 1. / 3., 211, np.where(r2 < 2. / 3., -211, 111)).astype(np.int32)
    raise ValueError(f"thermal_pid must be one of {THERMAL_PID_MODES}, not {mode!r}")


def _fuzzy_equals(a, b, tol=1e-5):
    """Rivet::fuzzyEquals (b scalar)."""
    both_zero = (np.abs(a) < 1e-8) & (abs(b) < 1e-8)
    return both_zero | (np.abs(a - b) < tol * (np.abs(a) + abs(b)) / 2.0)


def _kin(px, py, pz, e):
    """Rivet FourMomentum: pT, m_delta = sqrt(mass2 + pT^2) - pT, phi in [0, 2pi), rapidity."""
    pt = np.sqrt(px * px + py * py)
    md = np.sqrt(e * e - (px * px + py * py + pz * pz) + pt * pt) - pt
    phi = np.arctan2(py, px)                         # Rivet::mapAngle0To2Pi:
    phi = np.where(np.abs(phi) < 1e-8, 0.0, phi)
    phi = np.where(phi < 0.0, phi + 2.0 * np.pi, phi)
    phi = np.where(_fuzzy_equals(phi, 2.0 * np.pi), 0.0, phi)
    return pt, md, phi, 0.5 * np.log((e + pz) / (e - pz))


def _loop_py(I, K, ptp, mdp, ptt, mdt):
    ptp, mdp, ptt, mdt = ptp.tolist(), mdp.tolist(), ptt.tolist(), mdt.tolist()
    for i, k in zip(I.tolist(), K.tolist()):
        if ptp[i] > ptt[k]:
            ptp[i] -= ptt[k]
            ptt[k] = 0.0
        else:
            ptt[k] -= ptp[i]
            ptp[i] = 0.0
        if mdp[i] > mdt[k]:
            mdp[i] -= mdt[k]
            mdt[k] = 0.0
        else:
            mdt[k] -= mdp[i]
            mdp[i] = 0.0
    return np.array(ptp), np.array(mdp), np.array(ptt), np.array(mdt)


def _loop_arrays(I, K, ptp, mdp, ptt, mdt):
    for n in range(I.shape[0]):
        i, k = I[n], K[n]
        if ptp[i] > ptt[k]:
            ptp[i] -= ptt[k]
            ptt[k] = 0.0
        else:
            ptt[k] -= ptp[i]
            ptp[i] = 0.0
        if mdp[i] > mdt[k]:
            mdp[i] -= mdt[k]
            mdt[k] = 0.0
        else:
            mdt[k] -= mdp[i]
            mdp[i] = 0.0
    return ptp, mdp, ptt, mdt


_loop = njit(cache=True)(_loop_arrays) if njit is not None else _loop_py


def _p4(pt, md, phi, y):
    mt = pt + md
    return pt * np.cos(phi), pt * np.sin(phi), mt * np.sinh(y), mt * np.cosh(y)


def subtract(status, pid, px, py, pz, e, dRmax=0.5, thermal_pid="reference", rng=None):
    """Constituent-subtract one JEWEL event given its HepMC record (other statuses ignored)."""
    status, pid = np.asarray(status), np.asarray(pid)
    px, py, pz, e = (np.asarray(a, dtype=np.float64) for a in (px, py, pz, e))
    good = e - np.abs(pz) > 0.0
    fin = ((status == 1) | (status == 4)) & ~_fuzzy_equals(e, 1e-6)
    th = status == 3
    pas, deg = fin & ~good, th & ~good               # pass-through / degenerate thermal
    fin, th = fin & good, th & good

    ptp, mdp, php, yp = _kin(px[fin], py[fin], pz[fin], e[fin])
    ptt, mdt, pht, yt = _kin(px[th], py[th], pz[th], e[th])
    dphi = np.abs(php[:, None] - pht[None, :])
    dphi = np.where(dphi > np.pi, 2.0 * np.pi - dphi, dphi)
    dR = np.sqrt(dphi * dphi + (yp[:, None] - yt[None, :]) ** 2)
    I, K = np.nonzero(dR <= dRmax)                   # the reference stops at the first dR > dRmax
    order = np.argsort(dR[I, K], kind="stable")
    ptp, mdp, ptt, mdt = _loop(I[order], K[order], ptp, mdp, ptt, mdt)

    sp, lt = ptp > 0.0, ptt > 0.0
    pid_deg = thermal_pids(int(deg.sum()), thermal_pid, rng)   # same draw order as the C++
    pid_lt = thermal_pids(int(lt.sum()), thermal_pid, rng)
    blocks = [
        (px[pas], py[pas], pz[pas], e[pas], pid[pas], TAG_PASSTHROUGH),
        (px[deg], py[deg], pz[deg], e[deg], pid_deg, TAG_THERMAL_DEGENERATE),
        (*_p4(ptp[sp], mdp[sp], php[sp], yp[sp]), pid[fin][sp], TAG_SUBTRACTED),
        (*_p4(ptt[lt], mdt[lt], pht[lt], yt[lt]), pid_lt, TAG_THERMAL_LEFT),
    ]
    out = {k: np.concatenate([b[n] for b in blocks]) for n, k in enumerate(("px", "py", "pz", "e"))}
    out["pid"] = np.concatenate([b[4] for b in blocks]).astype(np.int32)
    out["tag"] = np.concatenate([np.full(len(b[0]), b[5], dtype=np.int8) for b in blocks])
    return out
