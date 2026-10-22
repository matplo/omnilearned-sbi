"""Jet observables computed from constituent features (eta, phi, log pT)."""

import numpy as np

NAMES = {
    "jet_pt": "jet pT [GeV]",
    "width": "pT-weighted <dR> to jet axis",
    "soft8_fraction": "pT fraction of the 8 softest constituents",
    "outer_fraction": "pT fraction at dR > 0.15",
    "ptd": "pT_D",
    "lead_fraction": "leading constituent pT fraction",
    "jet_eta": "jet eta",
    "n_constituents": "constituent multiplicity",
}


def jet_observables(x, mask=None):
    """x: [N, P, 3] physical features of jet constituents; optional mask [N, P].

    Padded slots (mask False) are ignored. ``soft8_fraction`` is the pT
    fraction of the 8 softest real constituents (all of them if N < 8).
    ``n_constituents`` is the real multiplicity.
    """
    x = np.asarray(x, dtype=np.float64)
    m = np.ones(x.shape[:2], bool) if mask is None else np.asarray(mask, bool)
    eta, phi = x[..., 0], x[..., 1]
    pt = np.where(m, np.exp(np.where(m, x[..., 2], 0.0)), 0.0)
    px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
    jpx, jpy, jpz = px.sum(1), py.sum(1), pz.sum(1)
    jpt = np.hypot(jpx, jpy)
    jeta = np.arcsinh(jpz / jpt)
    jphi = np.arctan2(jpy, jpx)
    dphi = np.angle(np.exp(1j * (phi - jphi[:, None])))
    dr = np.hypot(eta - jeta[:, None], dphi)
    spt = pt.sum(1)
    srt = np.sort(np.where(m, pt, np.inf), 1)  # real ascending, padded last
    soft8 = np.where(np.arange(srt.shape[1])[None, :] < np.minimum(8, m.sum(1))[:, None], srt, 0.0)
    lead = np.where(m, pt, -np.inf).max(1)
    return {
        "jet_pt": jpt,
        "jet_eta": jeta,
        "width": (pt * dr).sum(1) / spt,
        "soft8_fraction": soft8.sum(1) / spt,
        "outer_fraction": (pt * (dr > 0.15)).sum(1) / spt,
        "ptd": np.sqrt((pt ** 2).sum(1)) / spt,
        "lead_fraction": lead / spt,
        "n_constituents": m.sum(1).astype(np.float64),
    }


def ks_statistic(a, b):
    a, b = np.sort(a), np.sort(b)
    grid = np.concatenate([a, b])
    ca = np.searchsorted(a, grid, side="right") / a.size
    cb = np.searchsorted(b, grid, side="right") / b.size
    d = np.abs(ca - cb).max()
    n = a.size * b.size / (a.size + b.size)
    # Asymptotic Kolmogorov p-value.
    lam = (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) * d
    k = np.arange(1, 101)
    p = float(np.clip(2 * np.sum((-1) ** (k - 1) * np.exp(-2 * k ** 2 * lam ** 2)), 0, 1))
    return float(d), p
