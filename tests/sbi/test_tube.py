"""Iso-response curves and the development-scan design."""

import numpy as np

from omnilearned.sbi.toy.tube import development_scan_points, iso_curve


def test_iso_curve_recovers_analytic_degeneracy():
    # Response T^3 G(b) with G falling in b: iso-curve T(b) = T_ref (G(b_ref)/G(b))^(1/3).
    b = np.linspace(0, 11, 12)
    t = np.linspace(0.25, 0.55, 31)
    G = lambda x: 1.0 / (1.0 + 0.15 * x ** 2)  # noqa: E731
    vals = t[None, :] ** 3 * G(b)[:, None]
    curve, level = iso_curve(b, t, vals, 7.5, 0.30)
    exact = 0.30 * (G(7.5) / G(b)) ** (1 / 3)
    inside = (exact >= t[0]) & (exact <= t[-1])
    # bilinear level at the reference: small interpolation error only
    assert abs(level - 0.30 ** 3 * G(7.5)) / level < 0.02
    np.testing.assert_allclose(curve[inside], exact[inside], atol=2e-3)
    assert np.all(np.isnan(curve[~inside]))


def test_development_scan_points_lines_and_tube():
    box = [(0.0, 11.0), (0.25, 0.55), (0.2, 1.2)]
    pts, labels = development_scan_points([7.5, 0.30, 0.95], [4.0, 9.0, 10.0], [0.27, 0.34, np.nan],
                                          box, n_line=(3, 4, 5))
    assert labels.count("line_0") == 3 and labels.count("tube") == 2 and len(pts) == 14
    line = pts[[i for i, s in enumerate(labels) if s == "line_1"]]
    np.testing.assert_allclose(line[:, 0], 7.5)
    np.testing.assert_allclose(line[:, 2], 0.95)
    np.testing.assert_allclose(line[:, 1], np.linspace(0.25, 0.55, 4))


def test_response_gradients_from_lines():
    from omnilearned.sbi.toy.gate import response_gradients
    from omnilearned.sbi.toy.tube import development_scan_points

    box = [(0.0, 11.0), (0.25, 0.55), (0.2, 1.2)]
    dev = np.array([7.5, 0.30, 0.95])
    pts, labels = development_scan_points(dev, [], [], box)
    f = lambda p: np.stack([p[:, 0] ** 2 + 3 * p[:, 1], np.sin(p[:, 2]) * p[:, 1] * 10], 1)  # noqa: E731
    G = response_gradients(pts, f(pts), dev, labels)
    exact = np.array([[2 * 7.5, 3, 0], [0, 10 * np.sin(0.95), 10 * 0.30 * np.cos(0.95)]])
    np.testing.assert_allclose(G, exact, rtol=1e-2, atol=1e-6)  # cubic term over the window
