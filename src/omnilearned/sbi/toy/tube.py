"""Iso-response curves of the toy in the (b, T_ref) plane (lesson 1, toy v2).

Quenching depends on the medium roughly through T_ref³ G(b, τ_init), so jets stay
the same along a curve T_ref(b) through any point. ``response_grid`` draws
selected direct-toy jets on a (b, T_ref) grid at fixed τ_init (stream
``sensitivity``) and records the mean and standard error of the hidden path
moments I₁, I₂, I₃, the multiplicity N and the jet observables. ``iso_curve``
returns, for each b, the T_ref at which a mean response equals its value at a
reference point (linear interpolation along T_ref; NaN where the level is not
crossed). If the curves of all responses nearly coincide, jets alone cannot
separate b from T_ref along them, and N adds no new direction.

    python -m omnilearned.sbi.toy.tube --out tube.json --tau 0.95 --ref 7.5 0.30
"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from omnilearned.sbi.toy.config import V2_CONFIG, load_toy_config, sample_setting
from omnilearned.sbi.toy.datasets import host_info
from omnilearned.sbi.toy.observables import jet_observables

MOMENTS = ("I1", "I2", "I3")
JET = ("n_constituents", "width", "soft8_fraction", "outer_fraction", "ptd", "lead_fraction",
       "jet_pt")


def _responses(args):
    theta, n, seed, index, config = args
    rec, man = sample_setting(theta, n, seed, stream="sensitivity", index=index, path=config)
    out = {k: rec["diag"][k] for k in MOMENTS}
    obs = jet_observables(rec["x"], rec["mask"])
    out.update({k: obs[k] for k in JET})
    return {k: (float(np.mean(v)), float(np.std(v) / np.sqrt(v.size))) for k, v in out.items()}, man


def response_grid(b_axis, t_axis, tau, n, seed, config=V2_CONFIG, workers=16):
    """Mean and se of every response on the (b, T_ref) grid: dict name -> [nb, nt, 2]."""
    jobs = [((b, t, tau), n, seed, i * len(t_axis) + j, str(config))
            for i, b in enumerate(b_axis) for j, t in enumerate(t_axis)]
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_responses, jobs))
    names = list(res[0][0])
    grid = {k: np.array([r[0][k] for r in res]).reshape(len(b_axis), len(t_axis), 2) for k in names}
    acc = np.array([r[1]["acceptance"] for r in res]).reshape(len(b_axis), len(t_axis))
    return grid, acc


def iso_curve(b_axis, t_axis, values, ref_b, ref_t):
    """T_ref(b) where ``values`` [nb, nt] (bilinear) equals its value at (ref_b, ref_t)."""
    b_axis, t_axis, values = map(np.asarray, (b_axis, t_axis, values))
    i = np.clip(np.searchsorted(b_axis, ref_b) - 1, 0, len(b_axis) - 2)
    j = np.clip(np.searchsorted(t_axis, ref_t) - 1, 0, len(t_axis) - 2)
    u = (ref_b - b_axis[i]) / (b_axis[i + 1] - b_axis[i])
    w = (ref_t - t_axis[j]) / (t_axis[j + 1] - t_axis[j])
    level = ((1 - u) * (1 - w) * values[i, j] + u * (1 - w) * values[i + 1, j]
             + (1 - u) * w * values[i, j + 1] + u * w * values[i + 1, j + 1])
    out = np.full(len(b_axis), np.nan)
    for k, row in enumerate(values):
        d = row - level
        cross = np.flatnonzero(np.sign(d[:-1]) * np.sign(d[1:]) <= 0)
        if cross.size:
            c = cross[np.argmin(np.abs(t_axis[cross] - ref_t))]  # crossing nearest the reference
            if d[c + 1] == d[c]:
                out[k] = t_axis[c]
            else:
                out[k] = t_axis[c] + (t_axis[c + 1] - t_axis[c]) * d[c] / (d[c] - d[c + 1])
    return out, float(level)


def development_scan_points(dev, iso_b, iso_t, box, n_line=(12, 13, 11)):
    """1D lines through ``dev`` along each axis (box edges included) plus the tube points
    (iso_b, iso_t) at the development tau_init; returns (points [n, 3], labels)."""
    pts, labels = [], []
    for k, n in enumerate(n_line):
        for v in np.linspace(box[k][0], box[k][1], n):
            p = np.array(dev, dtype=np.float64)
            p[k] = v
            pts.append(p)
            labels.append(f"line_{k}")
    for b, t in zip(iso_b, iso_t):
        if np.isfinite(t) and box[1][0] <= t <= box[1][1]:
            pts.append(np.array([b, t, dev[2]]))
            labels.append("tube")
    return np.array(pts), labels


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--tau", type=float, default=0.95)
    ap.add_argument("--ref", type=float, nargs=2, default=[7.5, 0.30], help="reference (b, T_ref)")
    ap.add_argument("--b", type=float, nargs=3, default=[0.0, 11.0, 12], help="min max n")
    ap.add_argument("--t", type=float, nargs=3, default=[0.25, 0.55, 13], help="min max n")
    ap.add_argument("--n", type=int, default=2000, help="selected jets per grid setting")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--config", default=str(V2_CONFIG))
    ap.add_argument("--scan-design", default=None,
                    help="also write a development-scan design (lines + I3 tube) to this path")
    ap.add_argument("--tube-response", default="I3")
    a = ap.parse_args(argv)
    cfg, _, _ = load_toy_config(a.config)
    seed = int(cfg["randomness"]["validation_seed"])
    b_axis = np.linspace(a.b[0], a.b[1], int(a.b[2]))
    t_axis = np.linspace(a.t[0], a.t[1], int(a.t[2]))
    t0 = time.time()
    grid, acc = response_grid(b_axis, t_axis, a.tau, a.n, seed, a.config, a.workers)
    curves = {k: iso_curve(b_axis, t_axis, v[..., 0], *a.ref) for k, v in grid.items()}
    doc = {"tau_init": a.tau, "reference": a.ref, "b_axis": b_axis, "t_axis": t_axis,
           "n_per_setting": a.n, "seed": seed, "stream": "sensitivity",
           "responses": {k: v for k, v in grid.items()}, "acceptance": acc,
           "iso_curves": {k: {"T_ref": c, "level": lv} for k, (c, lv) in curves.items()},
           "seconds": time.time() - t0, "host": host_info()}
    with open(a.out, "w") as f:
        json.dump(doc, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"grid {len(b_axis)}x{len(t_axis)} at tau={a.tau}, {doc['seconds']:.0f}s")
    if a.scan_design:
        from omnilearned.sbi.toy.datasets import BOX
        from omnilearned.sbi.toy.simulator import PARAMETERS

        dev = cfg["holdout"]["development_validation"]
        box = [BOX[p] for p in PARAMETERS]
        tube_b = b_axis[(b_axis > 0) & (b_axis < box[0][1])]
        tube_t = curves[a.tube_response][0][(b_axis > 0) & (b_axis < box[0][1])]
        pts, labels = development_scan_points(dev, tube_b, tube_t, box)
        with open(a.scan_design, "w") as f:
            json.dump({"split": "development_scan", "parameters": list(PARAMETERS),
                       "points": pts.tolist(), "labels": labels, "development": dev,
                       "tube_response": a.tube_response, "iso_file": a.out,
                       "host": host_info()}, f, indent=1)
        print(f"development scan design: {len(pts)} points -> {a.scan_design}")
    print("b      " + " ".join(f"{k[:8]:>8s}" for k in curves))
    for i, b in enumerate(b_axis):
        print(f"{b:5.2f}  " + " ".join(f"{curves[k][0][i]:8.4f}" for k in curves))


if __name__ == "__main__":
    main()
