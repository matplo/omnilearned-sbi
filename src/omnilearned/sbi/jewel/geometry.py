"""JEWEL's impact parameter <-> centrality mapping (medium-simple.f).

JEWEL has no fixed-b input. It samples b from the centrality interval CENTRMIN..CENTRMAX
(percent) through its geometric cross-section table, so CENTRMIN = CENTRMAX = c gives the
deterministic b(c) that :class:`JewelGeometry` reproduces (CALCTA, NTHICK, CALCXSECTION and
the b sampling of MEDNEXTEVT). Ported from the omni-sbi-mc application
(``a/src/jet_quenching_inference/jewel.py``, tag devA-2026-10-22).
"""

import numpy as np


class JewelGeometry:
    def __init__(self, A=208, n0=0.17, d=0.54, sigma_nn=6.2):
        self.A, self.n0, self.d, self.sigma_nn = A, n0, d, sigma_nn
        R = 1.12 * A ** 0.33333 - 0.86 * A ** (-0.33333)
        self.R = R
        z = np.linspace(-2 * R, 2 * R, 4001)
        bs = np.arange(100) * 2.0 * R / 100
        rho = n0 / (1 + np.exp((np.sqrt(bs[:, None] ** 2 + z[None] ** 2) - R) / d))
        self.ta_r = bs
        self.ta = np.trapezoid(rho, z, axis=1)
        self._cross()

    def thickness(self, x, y):
        r = np.sqrt(x ** 2 + y ** 2)
        rmax = self.ta_r[-1]
        line = np.clip((r * 99.0 / rmax + 1).astype(int), 1, 99) - 1
        x1, x2 = self.ta_r[line], self.ta_r[line + 1]
        y1, y2 = self.ta[line], self.ta[line + 1]
        val = y1 + (y2 - y1) / (x2 - x1) * (r - x1)
        return np.where(r > rmax, 0.0, val)

    def _cross(self):
        ix = np.arange(1, 101)
        X = -20.0 + ix * 0.4
        XX, YY = np.meshgrid(X, X, indexing="ij")
        cross = np.zeros((200, 3))
        pprev = 0.0
        for ib in range(1, 201):
            b = 0.1 * ib
            tp = self.thickness(XX + b / 2, YY)
            tm = self.thickness(XX - b / 2, YY)
            logprod = np.sum(-tp * self.sigma_nn * 0.16 * tm)
            p = (1 - np.exp(logprod)) * 8.8 / 14.0 * b
            cross[ib - 1] = [b, p, 0.0 if ib == 1 else cross[ib - 2, 2] + (p + pprev) / 2 * 0.1]
            pprev = p
        self.cross = cross
        self.cdf = cross[:, 2] / cross[-1, 2]

    def b_of_centrality(self, c_percent):
        """Impact parameter [fm] JEWEL uses for CENTRMIN = CENTRMAX = c_percent."""
        r = np.atleast_1d(np.asarray(c_percent, dtype=np.float64)) / 100.0
        out = np.empty_like(r)
        for k, rv in enumerate(r):
            i = int(np.sum(rv - self.cdf >= 0))  # Fortran count over j = 1..200
            i = min(max(i, 1), 199)
            ci, cn = self.cdf[i - 1], self.cdf[i]
            b1, b2 = (i - 1) * 0.1, i * 0.1
            out[k] = (b2 * (ci - rv) + b1 * (rv - cn)) / (ci - cn)
        return out if np.ndim(c_percent) else float(out[0])

    def centrality_for_b(self, b, tol=1e-9):
        """Centrality [%] whose JEWEL impact parameter is ``b`` (bisection)."""
        lo, hi = 0.0, 100.0
        if b <= self.b_of_centrality(lo):
            return lo
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if self.b_of_centrality(mid) < b:
                lo = mid
            else:
                hi = mid
            if hi - lo < tol:
                break
        return 0.5 * (lo + hi)
