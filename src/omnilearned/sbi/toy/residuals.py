"""Gate residual diagnostics (surrogate fidelity plan, S0).

Reads the per-point gate JSONs of ``toy.gate run`` (and optionally the ``shift.json`` of
``toy.gate shift``) and measures, in relevance units at N = 1000 (mean shift divided by
the direct jets' standard error at 1000 jets, the unit of the gate's relevance):

* ``map``: where in θ the residual sits. Per jet observable, a weighted least-squares fit of
  the signed shift against [1, u_b, u_T, u_τ] (u scaled to [−1, 1] over the box), and the
  mean squared bias by T_ref band.
* ``scaling``: how the residual falls with training jets per point. The noise-corrected mean
  squared bias B² = mean(Δ² − se²) of two gates of generators trained on n_a and n_b jets per
  point, the exponent α of B ∝ n^−α, and the prediction at other n.
* ``seeds``: the training-to-training spread from gates of generators that differ only in
  the seed. The direct fidelity jets are shared, so their noise cancels in Δ_i − Δ_j, and
  the variance of the difference is 2 τ² + 2 se_s² (τ: the model-to-model scatter of the
  bias, se_s: the surrogate sample's part of the standard error, ≈ se/√2).
* ``row``: the scalar gate statistics of one run (FAILs, p < 0.01, mean largest relevance,
  constrained and marginal shift at the development point).

    python -m omnilearned.sbi.toy.residuals map --gate RUN/gate --out map.json
    python -m omnilearned.sbi.toy.residuals scaling --gate-a R1/gate --n-a 2000 \\
        --gate-b R2/gate --n-b 6000 --out scaling.json
    python -m omnilearned.sbi.toy.residuals seeds --gates S0/gate S1/gate --out seeds.json
    python -m omnilearned.sbi.toy.residuals row --gate RUN/gate --shift RUN/shift.json
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from omnilearned.sbi.fidelity import JET_VARIABLE
from omnilearned.sbi.flow import json_dumps

ANALYSIS_N = "1000"
T_BANDS = (0.25, 0.35, 0.45, 0.55)
ALPHA = 0.05
DEFAULT_BOX = [[0.0, 11.0], [0.25, 0.55], [0.2, 1.2]]  # toy v2 (design_*.json "box")


def load_gate(gate_dir, n=ANALYSIS_N, keys=JET_VARIABLE):
    """Arrays from the point JSONs: theta [P, 3], delta, se, unit [P, K] (shift, its se and
    the relevance unit at ``n`` jets, signed relevance = delta / unit), p_family [P]."""
    rows = []
    for p in sorted(Path(gate_dir).glob("point_*.json")):
        with open(p) as f:
            rows.append(json.load(f))
    if not rows:
        raise SystemExit(f"no gate points in {gate_dir}")
    out = {"point_id": np.array([r["point_id"] for r in rows]), "keys": list(keys),
           "theta": np.array([r["theta"] for r in rows], float),
           "p_family": np.array([r["permutation"]["p_family"] for r in rows], float)}
    rel = [[r["relevance_by_n"][n][k] for k in keys] for r in rows]
    out["delta"] = np.array([[v["delta_mean"] for v in row] for row in rel], float)
    out["se"] = np.array([[v["se"] for v in row] for row in rel], float)
    out["relevance"] = np.array([[v["relevance"] for v in row] for row in rel], float)
    out["relevant"] = np.array([[v["relevant"] for v in row] for row in rel], bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["unit"] = np.where(out["relevance"] > 0, np.abs(out["delta"]) / out["relevance"], np.nan)
    return out


def in_units(g):
    """Signed shift and its se in relevance units [P, K]."""
    return g["delta"] / g["unit"], g["se"] / g["unit"]


def box_coordinates(theta, box):
    lo, hi = np.asarray(box, float).T
    return 2 * (np.asarray(theta, float) - lo) / (hi - lo) - 1


def linear_map(u, y, se):
    """Weighted least squares y ≈ c0 + c·u per column: coefficients, se, z, χ²/dof [K, ...]."""
    X = np.column_stack([np.ones(len(u)), u])
    res = []
    for k in range(y.shape[1]):
        w = 1 / se[:, k] ** 2
        A = X.T @ (X * w[:, None])
        cov = np.linalg.inv(A)
        c = cov @ X.T @ (w * y[:, k])
        chi2 = float(np.sum(w * (y[:, k] - X @ c) ** 2))
        res.append({"coef": c, "coef_se": np.sqrt(np.diag(cov)), "z": c / np.sqrt(np.diag(cov)),
                    "chi2": chi2, "dof": len(u) - X.shape[1]})
    return res


def bias_squared(y, se):
    """Noise-corrected mean squared bias over points, B² = mean(y² − se²), and its se.

    For y ~ N(B, se²) the estimator is unbiased with variance (4 B² se² + 2 se⁴) per point.
    """
    y, se = np.asarray(y, float), np.asarray(se, float)
    b2 = float(np.mean(y ** 2 - se ** 2))
    var = np.mean(4 * max(b2, 0.0) * se ** 2 + 2 * se ** 4) / y.size
    return b2, float(math.sqrt(var))


def scaling_exponent(b2_a, b2_b, n_a, n_b):
    """α in B ∝ n^−α from B²(n_a), B²(n_b); nan if either is not positive."""
    if b2_a <= 0 or b2_b <= 0:
        return float("nan")
    return float(-0.5 * math.log(b2_b / b2_a) / math.log(n_b / n_a))


def predict_b2(b2_b, n_b, alpha, n):
    return float(b2_b * (n / n_b) ** (-2 * alpha))


def seed_spread(ys, ses):
    """Model-to-model scatter τ² of the signed shift (relevance units) from runs that differ
    only in the seed, using every pair; per observable [K]."""
    ys, ses = np.asarray(ys, float), np.asarray(ses, float)
    se_s2 = np.mean(ses ** 2, 0) / 2  # surrogate-sample part of the se (equal sample sizes)
    d2 = [np.mean((ys[i] - ys[j]) ** 2, 0) for i in range(len(ys)) for j in range(i + 1, len(ys))]
    tau2 = np.mean(d2, 0) / 2 - np.mean(se_s2, 0)
    return {"tau2": tau2, "tau": np.sqrt(np.maximum(tau2, 0)), "pair_msd": np.mean(d2, 0),
            "surrogate_se2": np.mean(se_s2, 0), "n_runs": len(ys)}


def gate_row(gate_dir, shift_path=None, n_settings=128, alpha=ALPHA):
    """Scalar gate statistics of one run."""
    g = load_gate(gate_dir)
    thr = alpha / n_settings
    rel_any = g["relevant"].any(1)
    row = {"gate": str(gate_dir), "points": int(len(g["p_family"])),
           "fail": int(np.sum((g["p_family"] < thr) & rel_any)),
           "significant": int(np.sum(g["p_family"] < thr)),
           "p_below_0.01": int(np.sum(g["p_family"] < 0.01)),
           "mean_max_relevance": float(np.mean(g["relevance"].max(1))),
           "relevant_counts": {k: int(g["relevant"][:, i].sum()) for i, k in enumerate(g["keys"])}}
    if shift_path and Path(shift_path).exists():
        with open(shift_path) as f:
            s = json.load(f)
        F = np.asarray(s["fisher_per_jet"], float)
        dev = s["points"][0]
        dth = np.asarray(dev["delta_theta"], float)
        row["constrained_shift"] = float(math.sqrt(1000 * dth @ F @ dth))
        row["marginal_shift"] = [float(v) for v in dev["in_sigma_marginal_N1000"]]
        row["shift_label"] = dev["label"]
    return row


def _map(a):
    g = load_gate(a.gate)
    box = DEFAULT_BOX
    if a.design:
        with open(a.design) as f:
            box = list(json.load(f)["box"].values())
    y, se = in_units(g)
    u = box_coordinates(g["theta"], box)
    fits = linear_map(u, y, se)
    bands = []
    for lo, hi in zip(T_BANDS[:-1], T_BANDS[1:]):
        m = (g["theta"][:, 1] >= lo) & (g["theta"][:, 1] < hi)
        if m.sum() < 2:
            continue
        bands.append({"T_ref": [lo, hi], "points": int(m.sum()),
                      "mean_signed": y[m].mean(0), "mean_signed_se": np.sqrt((se[m] ** 2).sum(0)) / m.sum(),
                      "b2": [bias_squared(y[m, k], se[m, k]) for k in range(y.shape[1])],
                      "fails": int(np.sum((g["p_family"][m] < ALPHA / 128) & g["relevant"][m].any(1)))})
    doc = {"gate": a.gate, "keys": g["keys"], "box": box, "coordinates": ["1", "u_b", "u_T_ref", "u_tau_init"],
           "fits": {k: f for k, f in zip(g["keys"], fits)}, "bands": bands,
           "b2_all": {k: bias_squared(y[:, i], se[:, i]) for i, k in enumerate(g["keys"])}}
    _write(a.out, doc)
    print(f"{a.gate}: signed shift (relevance units, N=1000) = c0 + c_b u_b + c_T u_T + c_tau u_tau")
    for k, f in doc["fits"].items():
        print(f"  {k:15s} " + "  ".join(f"{c:+.2f}±{s:.2f}" for c, s in zip(f["coef"], f["coef_se"]))
              + f"  chi2/dof {f['chi2']:.0f}/{f['dof']}")
    for b in bands:
        print(f"  T_ref {b['T_ref']}: {b['points']} pts, {b['fails']} FAIL, B^2 "
              + " ".join(f"{k}={v[0]:.2f}" for k, v in zip(g["keys"], b["b2"])))


def _scaling(a):
    ga, gb = load_gate(a.gate_a), load_gate(a.gate_b)
    if not np.array_equal(ga["point_id"], gb["point_id"]):
        raise ValueError("the two gates cover different points")
    ya, sa = in_units(ga)
    yb, sb = in_units(gb)
    regions = {"all": np.ones(len(ya), bool), "T_ref>0.45": ga["theta"][:, 1] > 0.45,
               "T_ref<=0.45": ga["theta"][:, 1] <= 0.45}
    doc = {"gate_a": a.gate_a, "gate_b": a.gate_b, "n_a": a.n_a, "n_b": a.n_b, "keys": ga["keys"],
           "predict_n": a.predict, "regions": {}}
    for name, m in regions.items():
        reg = {}
        for i, k in enumerate(ga["keys"]):
            b2a, ea = bias_squared(ya[m, i], sa[m, i])
            b2b, eb = bias_squared(yb[m, i], sb[m, i])
            al = scaling_exponent(b2a, b2b, a.n_a, a.n_b)
            reg[k] = {"b2_a": [b2a, ea], "b2_b": [b2b, eb], "alpha": al,
                      "predicted_b2": {str(n): predict_b2(b2b, a.n_b, al, n) if np.isfinite(al) else None
                                       for n in a.predict}}
        reg["max_relevance_mean"] = [float(ga["relevance"][m].max(1).mean()), float(gb["relevance"][m].max(1).mean())]
        doc["regions"][name] = {"points": int(m.sum()), "per_observable": reg}
    _write(a.out, doc)
    for name, r in doc["regions"].items():
        print(f"{name} ({r['points']} pts): mean max relevance {r['per_observable']['max_relevance_mean']}")
        for k in a.show:
            v = r["per_observable"][k]
            print(f"  {k:15s} B^2 {v['b2_a'][0]:.3f}±{v['b2_a'][1]:.3f} -> {v['b2_b'][0]:.3f}±{v['b2_b'][1]:.3f}"
                  f"  alpha {v['alpha']:.2f}  predicted B^2 {v['predicted_b2']}")


def _seeds(a):
    gs = [load_gate(d) for d in a.gates]
    for g in gs[1:]:
        if not np.array_equal(g["point_id"], gs[0]["point_id"]):
            raise ValueError("gates cover different points")
    ys, ses = zip(*[in_units(g) for g in gs])
    sp = seed_spread(ys, ses)
    rows = [gate_row(d, s) for d, s in zip(a.gates, a.shifts or [None] * len(a.gates))]
    scal = {}
    for key in ("fail", "p_below_0.01", "mean_max_relevance", "constrained_shift"):
        v = [r[key] for r in rows if key in r]
        if len(v) > 1:
            scal[key] = {"values": v, "mean": float(np.mean(v)), "std": float(np.std(v, ddof=1))}
    m = [r["marginal_shift"] for r in rows if "marginal_shift" in r]
    if len(m) > 1:
        scal["marginal_shift"] = {"values": m, "mean": np.mean(m, 0), "std": np.std(m, 0, ddof=1)}
    doc = {"gates": a.gates, "keys": gs[0]["keys"], "spread": sp, "rows": rows, "scalars": scal}
    _write(a.out, doc)
    print("model-to-model scatter tau (relevance units, N=1000): "
          + " ".join(f"{k}={t:.2f}" for k, t in zip(gs[0]["keys"], sp["tau"])))
    for k, v in scal.items():
        print(f"  {k}: {np.round(v['values'], 3).tolist()} std {np.round(v['std'], 3).tolist()}")


def _row(a):
    print(json_dumps(gate_row(a.gate, a.shift, a.n_settings)))


def _write(path, doc):
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json_dumps(doc))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("map")
    m.add_argument("--gate", required=True)
    m.add_argument("--design", default=None, help="design JSON with the box (default: toy v2 box)")
    m.add_argument("--out", default=None)
    s = sub.add_parser("scaling")
    s.add_argument("--gate-a", required=True)
    s.add_argument("--n-a", type=float, required=True)
    s.add_argument("--gate-b", required=True)
    s.add_argument("--n-b", type=float, required=True)
    s.add_argument("--predict", type=float, nargs="+", default=[12000, 24000])
    s.add_argument("--show", nargs="+", default=["jet_pt", "width", "outer_fraction"])
    s.add_argument("--out", default=None)
    d = sub.add_parser("seeds")
    d.add_argument("--gates", nargs="+", required=True)
    d.add_argument("--shifts", nargs="+", default=None)
    d.add_argument("--out", default=None)
    r = sub.add_parser("row")
    r.add_argument("--gate", required=True)
    r.add_argument("--shift", default=None)
    r.add_argument("--n-settings", type=int, default=128)
    a = ap.parse_args(argv)
    {"map": _map, "scaling": _scaling, "seeds": _seeds, "row": _row}[a.cmd](a)


if __name__ == "__main__":
    main()
