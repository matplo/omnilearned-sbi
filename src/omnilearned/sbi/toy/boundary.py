"""Where the surrogate's selected jets differ from the direct jets (fidelity plan S0/S1).

Two diagnostics at gate points, from one set of surrogate clouds per point (all clouds,
before selection, with their rejection reasons) and the direct fidelity jets:

* **N-mix vs fixed-N decomposition** of each jet-observable mean shift:

      Δ⟨o⟩ = Σ_N [p_s(N) − p_d(N)] o_d(N)   (multiplicity mix; what same-N redraws change)
           + Σ_N p_s(N) [o_s(N) − o_d(N)]  (shift at fixed N; flow fidelity and truncation)

  with p_s, p_d the selected-surrogate and direct N distributions and o_s(N), o_d(N) the
  per-N means (N bins with too few jets on either side are merged). Both terms are reported
  in relevance units at N = 1000, for the dropped (default) and the redrawn selected samples.

* **Selection boundaries:** densities of the selection variables near their cuts, direct vs
  selected surrogate, and the surrogate's mass beyond each cut. A smooth density fitted to
  truncated data leaks mass across the cut and is short of it just inside, which pulls the
  selected mean away from the cut (jet pT up from 60 GeV; the widest jets, close to the
  one-jet requirement, missing).
  - ``jet_pt`` of the anti-kt jet of the cloud;
  - ``dr_max``: the largest constituent distance from the pT-weighted axis.

    python -m omnilearned.sbi.toy.boundary run --checkpoint CKPT --fidelity-dir DIR \\
        --points 0 4 8 --out OUT_DIR [--n-generate 22000] [--device cuda]
    python -m omnilearned.sbi.toy.boundary summarize --dir OUT_DIR
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from omnilearned.sbi import fidelity
from omnilearned.sbi.flow import json_dumps
from omnilearned.sbi.toy.simulator_v2 import REJECTION_REASONS

KEYS = ("width", "outer_fraction", "jet_pt", "n_constituents", "ptd", "lead_fraction", "soft8_fraction")
ANALYSIS_N = 1000
PT_EDGES = np.arange(50.0, 90.01, 2.0)
DR_EDGES = np.arange(0.20, 0.701, 0.02)
NOT_ONE_JET = REJECTION_REASONS.index("not_one_jet")


def dr_max(x, mask):
    """Largest constituent ΔR from the pT-weighted (η, φ) axis [n]."""
    pt = np.where(mask, np.exp(np.where(mask, x[..., 2], 0.0)), 0.0)
    w = pt / np.maximum(pt.sum(1, keepdims=True), 1e-300)
    eta, phi = x[..., 0], x[..., 1]
    ae, ap = (w * eta).sum(1, keepdims=True), (w * phi).sum(1, keepdims=True)
    d = np.sqrt((eta - ae) ** 2 + (phi - ap) ** 2)
    return np.where(mask, d, -np.inf).max(1)


def merged_bins(n_s, n_d, min_count=30):
    """Contiguous N bins with at least ``min_count`` jets on both sides: list of (lo, hi)."""
    lo = int(min(n_s.min(), n_d.min()))
    hi = int(max(n_s.max(), n_d.max()))
    cs = np.bincount(n_s - lo, minlength=hi - lo + 1)
    cd = np.bincount(n_d - lo, minlength=hi - lo + 1)
    bins, start, a, b = [], lo, 0, 0
    for k in range(hi - lo + 1):
        a, b = a + cs[k], b + cd[k]
        if a >= min_count and b >= min_count:
            bins.append((start, lo + k))
            start, a, b = lo + k + 1, 0, 0
    if bins and start <= hi:  # remainder joins the last bin
        bins[-1] = (bins[-1][0], hi)
    elif not bins:
        bins = [(lo, hi)]
    return bins


def decompose(o_s, n_s, o_d, n_d, min_count=30):
    """(total, mix, fixed_n) mean-shift terms; total = mix + fixed_n exactly."""
    bins = merged_bins(n_s, n_d, min_count)
    ps, pd, ms, md = [], [], [], []
    for lo, hi in bins:
        s, d = (n_s >= lo) & (n_s <= hi), (n_d >= lo) & (n_d <= hi)
        ps.append(s.mean())
        pd.append(d.mean())
        ms.append(o_s[s].mean() if s.any() else 0.0)
        md.append(o_d[d].mean() if d.any() else 0.0)
    ps, pd, ms, md = map(np.asarray, (ps, pd, ms, md))
    mix = float(np.sum((ps - pd) * md))
    fixed = float(np.sum(ps * (ms - md)))
    return mix + fixed, mix, fixed, len(bins)


def point_report(xs, ms, ok, reason, jet_pt_s, xd, md, jet_pt_d, ok_redraw=None, xr=None, mr=None):
    """Decomposition and boundary densities at one point (relevance units at N = 1000)."""
    od = fidelity.jet_observables(xd, md)
    os_ = fidelity.jet_observables(xs, ms)
    n_d, n_s = md.sum(1), ms.sum(1)
    out = {"n_direct": int(len(xd)), "n_generated": int(len(xs)), "acceptance": float(ok.mean()), "observables": {}}
    samples = {"dropped": (os_, n_s, ok)}
    if xr is not None:
        samples["redrawn"] = (fidelity.jet_observables(xr, mr), mr.sum(1), np.ones(len(xr), bool))
    for k in KEYS:
        unit = float(np.std(od[k]) / np.sqrt(ANALYSIS_N))
        row = {}
        for name, (o, n, sel) in samples.items():
            tot, mix, fixed, nb = decompose(o[k][sel], n[sel], od[k], n_d)
            row[name] = {"total": tot / unit, "n_mix": mix / unit, "fixed_n": fixed / unit, "n_bins": nb}
        out["observables"][k] = row
    dd, ds = dr_max(xd, md), dr_max(xs, ms)
    out["boundary"] = {
        "jet_pt": {"edges": PT_EDGES, "direct": np.histogram(jet_pt_d, PT_EDGES)[0] / len(xd),
                   "surrogate_selected": np.histogram(jet_pt_s[ok], PT_EDGES)[0] / max(ok.sum(), 1),
                   "surrogate_all": np.histogram(jet_pt_s, PT_EDGES)[0] / len(xs),
                   "below_cut_fraction": float(np.mean(jet_pt_s < 60.0))},
        "dr_max": {"edges": DR_EDGES, "direct": np.histogram(dd, DR_EDGES)[0] / len(xd),
                   "surrogate_selected": np.histogram(ds[ok], DR_EDGES)[0] / max(ok.sum(), 1),
                   "surrogate_rejected_not_one_jet": np.histogram(ds[reason == NOT_ONE_JET], DR_EDGES)[0] / len(xs)},
        "reasons": np.bincount(reason[~ok].astype(np.int64), minlength=len(REJECTION_REASONS)).tolist()}
    return out


def run(a):
    import torch

    from omnilearned.sbi.device import resolve_device
    from omnilearned.sbi.flow import FlowSurrogate
    from omnilearned.sbi.toy.datasets import host_info
    from omnilearned.sbi.toy.gate import direct_by_point, selection
    from omnilearned.sbi.toy.sample_select import sample_and_select
    from omnilearned.sbi.toy.selection import select_clouds
    from omnilearned.sbi.toy.simulator import PARAMETERS

    dev = resolve_device(a.device)
    sur = FlowSurrogate.load(a.checkpoint, device=dev.device)
    sel = selection(a.config)
    direct = direct_by_point(a.fidelity_dir)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for pid in a.points:
        js = out / f"point_{pid:05d}.json"
        if js.exists():
            continue
        t0 = time.time()
        xd, md, theta = direct[pid]
        _, _, rd = select_clouds(xd.astype(np.float64), sel, md)
        xs, ms = sur.sample(np.asarray(theta), a.n_generate, None, seed=a.seed + pid, nsteps=a.nsteps,
                            candidate_names=list(PARAMETERS), batch=a.batch, sort_by_length=True)
        ok, reason, rs = select_clouds(xs.astype(np.float64), sel, ms)
        xr, mr, rep = sample_and_select(sur, theta, a.n_generate, sel, seed=a.seed + pid, nsteps=a.nsteps,
                                        batch=a.batch, redraw=True)
        doc = point_report(xs, ms, ok, reason, rs["jet_pt"], xd, md, rd["jet_pt"], xr=xr, mr=mr)
        doc.update({"point_id": pid, "theta": np.asarray(theta).tolist(), "checkpoint": a.checkpoint,
                    "checkpoint_sha256": sur.metadata.get("checkpoint_sha256"), "nsteps": a.nsteps,
                    "seed": a.seed + pid, "redraw_report": rep, "device": dev.to_dict(), "host": host_info(),
                    "gpu_name": torch.cuda.get_device_name() if dev.device == "cuda" else None,
                    "seconds": time.time() - t0})
        js.write_text(json_dumps(doc))
        w = doc["observables"]["width"]
        print(f"point {pid} theta={np.round(theta, 3).tolist()} acc={doc['acceptance']:.3f} width "
              f"dropped {w['dropped']['total']:+.2f} = mix {w['dropped']['n_mix']:+.2f} + fixedN "
              f"{w['dropped']['fixed_n']:+.2f}; redrawn {w['redrawn']['total']:+.2f} ({doc['seconds']:.0f}s)",
              flush=True)


def summarize(a):
    rows = [json.loads(p.read_text()) for p in sorted(Path(a.dir).glob("point_*.json"))]
    if not rows:
        raise SystemExit(f"no points in {a.dir}")
    doc = {"points": len(rows), "observables": {}}
    for k in KEYS:
        r = {}
        for name in ("dropped", "redrawn"):
            v = np.array([[p["observables"][k][name][t] for t in ("total", "n_mix", "fixed_n")] for p in rows])
            r[name] = {"mean": v.mean(0).tolist(), "se": (v.std(0, ddof=1) / np.sqrt(len(v))).tolist()}
        doc["observables"][k] = r
    for b in ("jet_pt", "dr_max"):
        doc[b] = {f: np.mean([p["boundary"][b][f] for p in rows], 0).tolist()
                  for f in rows[0]["boundary"][b] if f != "below_cut_fraction"}
    doc["jet_pt"]["below_cut_fraction"] = float(np.mean([p["boundary"]["jet_pt"]["below_cut_fraction"] for p in rows]))
    Path(a.dir, "summary.json").write_text(json_dumps(doc))
    print(f"{len(rows)} points; mean over points [total, N-mix, fixed-N] (relevance units, N=1000):")
    for k, r in doc["observables"].items():
        print(f"  {k:15s} dropped {np.round(r['dropped']['mean'], 2).tolist()}  redrawn {np.round(r['redrawn']['mean'], 2).tolist()}")


def main(argv=None):
    from omnilearned.sbi.toy.config import V2_CONFIG

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--checkpoint", required=True)
    r.add_argument("--fidelity-dir", required=True)
    r.add_argument("--points", type=int, nargs="+", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--n-generate", type=int, default=22000)
    r.add_argument("--nsteps", type=int, default=128)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--batch", type=int, default=1024)
    r.add_argument("--device", default="auto")
    r.add_argument("--config", default=str(V2_CONFIG))
    s = sub.add_parser("summarize")
    s.add_argument("--dir", required=True)
    a = ap.parse_args(argv)
    {"run": run, "summarize": summarize}[a.cmd](a)


if __name__ == "__main__":
    main()
