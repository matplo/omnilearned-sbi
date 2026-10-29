"""Surrogate fidelity as a function of jet pT (fidelity plan S0, pT-differential).

The gate and the shift integrate over the jet-pT spectrum, which falls by a factor ≈ 60
from 60 to 200 GeV (46 % of the jets in 60–70 GeV, 0.8 % in 170–200 GeV, at every θ). This
tool compares direct and selected surrogate jets in bins of the anti-kt jet pT (the selection
variable) at gate points:

* the spectrum, surrogate / direct;
* per observable and bin: the mean shift as an effect size Δ/σ_direct and as a pull
  Δ/se, the spread ratio σ_s/σ_d, and the two-sample KS statistic.

Over points: effect sizes are averaged (se from the spread over points), pulls combined as
Σ pull / √P. An effect size e in a bin holding a fraction f of the jets moves the mean of a
1000-jet analysis by about e √(1000 f) se.

    python -m omnilearned.sbi.toy.ptbins run --checkpoint CKPT --fidelity-dir DIR \\
        --points 0 8 16 --out OUT_DIR [--n-generate 22000] [--no-redraw] [--device cuda]
    python -m omnilearned.sbi.toy.ptbins plot --dirs DIR1 DIR2 --labels A B --out FIG_DIR
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from omnilearned.sbi.flow import json_dumps
from omnilearned.sbi.toy.observables import jet_observables, ks_statistic

KEYS = ("width", "outer_fraction", "n_constituents", "ptd", "lead_fraction", "soft8_fraction")
PT_EDGES = np.array([60.0, 70, 80, 90, 100, 120, 140, 170, 200])
SPECTRUM_EDGES = np.arange(60.0, 200.01, 5.0)


def pt_binned(pt_d, obs_d, pt_s, obs_s, edges=PT_EDGES, keys=KEYS, min_count=10):
    """Direct vs surrogate comparison in jet-pT bins. obs_*: dict key -> per-jet array."""
    bd, bs = np.digitize(pt_d, edges) - 1, np.digitize(pt_s, edges) - 1
    nb = len(edges) - 1
    doc = {"edges": np.asarray(edges).tolist(),
           "n_direct": np.bincount(bd[(bd >= 0) & (bd < nb)], minlength=nb).tolist(),
           "n_surrogate": np.bincount(bs[(bs >= 0) & (bs < nb)], minlength=nb).tolist(),
           "observables": {}}
    for k in keys:
        r = {f: [] for f in ("mean_d", "mean_s", "sd_d", "sd_s", "effect", "pull", "sd_ratio", "ks", "ks_p")}
        for i in range(nb):
            a, b = obs_d[k][bd == i], obs_s[k][bs == i]
            if min(a.size, b.size) < min_count:
                for f in r:
                    r[f].append(None)
                continue
            md, ms, sd, ss = a.mean(), b.mean(), a.std(ddof=1), b.std(ddof=1)
            se = np.sqrt(sd ** 2 / a.size + ss ** 2 / b.size)
            d, p = ks_statistic(a, b)
            for f, v in zip(r, (md, ms, sd, ss, (ms - md) / sd, (ms - md) / se, ss / sd, d, p)):
                r[f].append(float(v))
        doc["observables"][k] = r
    return doc


def combine(rows, keys=KEYS):
    """Average per-point ``pt_binned`` docs: effect and sd_ratio mean ± se over points,
    pulls combined as Σ pull / √P (bins missing at a point are skipped)."""
    out = {"points": len(rows), "edges": rows[0]["edges"],
           "n_direct": np.sum([r["n_direct"] for r in rows], 0).tolist(),
           "n_surrogate": np.sum([r["n_surrogate"] for r in rows], 0).tolist(), "observables": {}}
    for k in keys:
        o = {}
        for f in ("effect", "sd_ratio", "pull"):
            v = np.array([[np.nan if x is None else x for x in r["observables"][k][f]] for r in rows])
            n = np.sum(~np.isnan(v), 0)
            if f == "pull":
                o["pull_combined"] = (np.nansum(v, 0) / np.sqrt(np.maximum(n, 1))).tolist()
            else:
                o[f] = np.nanmean(v, 0).tolist()
                o[f + "_se"] = (np.nanstd(v, 0, ddof=1) / np.sqrt(np.maximum(n, 1))).tolist()
        out["observables"][k] = o
    return out


def run(a):
    import torch

    from omnilearned.sbi.device import resolve_device
    from omnilearned.sbi.flow import FlowSurrogate
    from omnilearned.sbi.toy.datasets import host_info
    from omnilearned.sbi.toy.gate import direct_by_point, selection
    from omnilearned.sbi.toy.sample_select import sample_and_select
    from omnilearned.sbi.toy.selection import select_clouds

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
        xs, ms, rep = sample_and_select(sur, theta, a.n_generate, sel, seed=a.seed + pid, nsteps=a.nsteps,
                                        batch=a.batch, redraw=a.redraw)
        _, _, rs = select_clouds(xs.astype(np.float64), sel, ms)
        od, os_ = jet_observables(xd, md), jet_observables(xs, ms)
        doc = pt_binned(rd["jet_pt"], od, rs["jet_pt"], os_)
        doc["spectrum"] = {"edges": SPECTRUM_EDGES.tolist(),
                           "direct": np.histogram(rd["jet_pt"], SPECTRUM_EDGES)[0].tolist(),
                           "surrogate": np.histogram(rs["jet_pt"], SPECTRUM_EDGES)[0].tolist()}
        doc.update({"point_id": pid, "theta": np.asarray(theta).tolist(), "checkpoint": a.checkpoint,
                    "checkpoint_sha256": sur.metadata.get("checkpoint_sha256"), "redraw": a.redraw,
                    "nsteps": a.nsteps, "seed": a.seed + pid, "sample_report": rep, "device": dev.to_dict(),
                    "host": host_info(), "gpu_name": torch.cuda.get_device_name() if dev.device == "cuda" else None,
                    "seconds": time.time() - t0})
        js.write_text(json_dumps(doc))
        w = doc["observables"]["width"]["effect"]
        print(f"point {pid} theta={np.round(theta, 3).tolist()} width effect by pT bin "
              f"{[None if v is None else round(v, 3) for v in w]} ({doc['seconds']:.0f}s)", flush=True)


def plot(a):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = a.labels or [Path(d).name for d in a.dirs]
    runs = []
    for d in a.dirs:
        rows = [json.loads(p.read_text()) for p in sorted(Path(d).glob("point_*.json"))]
        if not rows:
            raise SystemExit(f"no points in {d}")
        c = combine(rows)
        c["spectrum"] = {f: np.sum([r["spectrum"][f] for r in rows], 0) for f in ("direct", "surrogate")}
        Path(d, "ptbins_summary.json").write_text(json_dumps(c))
        runs.append(c)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    e = np.array(runs[0]["edges"])
    ctr, hw = 0.5 * (e[1:] + e[:-1]), 0.5 * np.diff(e)
    off = (np.arange(len(runs)) - (len(runs) - 1) / 2) * 1.2

    fig, ax = plt.subplots(2, 1, figsize=(7, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    se_ = np.array(SPECTRUM_EDGES)
    for j, (c, lab) in enumerate(zip(runs, labels)):
        d, s = c["spectrum"]["direct"], c["spectrum"]["surrogate"]
        fd, fs = d / d.sum(), s / s.sum()
        if j == 0:
            ax[0].stairs(fd, se_, color="k", label=f"direct ({c['points']} points)")
        ax[0].stairs(fs, se_, color=f"C{j}", label=lab)
        r = fs / np.where(fd > 0, fd, np.nan)
        err = r * np.sqrt(1 / np.maximum(d, 1) + 1 / np.maximum(s, 1))
        ax[1].errorbar(0.5 * (se_[1:] + se_[:-1]) + off[j], r, err, fmt="o", ms=3, color=f"C{j}")
    ax[0].set(yscale="log", ylabel="fraction of selected jets / 5 GeV", title="Jet-pT spectrum, pooled over points")
    ax[0].legend(fontsize=8)
    ax[1].axhline(1, color="k", lw=0.5)
    ax[1].set(xlabel="anti-kt jet pT [GeV]", ylabel="surrogate / direct", ylim=(0.7, 1.3))
    for x in ax:
        x.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "ptbins_spectrum.png", dpi=130)

    for f, ylab, ref, fn in (("effect", "Δ⟨o⟩ / σ_direct (mean over points)", 0.0, "ptbins_effect.png"),
                             ("sd_ratio", "σ_surrogate / σ_direct (mean over points)", 1.0, "ptbins_sd_ratio.png")):
        fig, axs = plt.subplots(2, 3, figsize=(13, 6.5), sharex=True)
        for x, k in zip(axs.flat, KEYS):
            for j, (c, lab) in enumerate(zip(runs, labels)):
                o = c["observables"][k]
                x.errorbar(ctr + off[j], o[f], o[f + "_se"], xerr=hw if j == 0 else None, fmt="o", ms=3,
                           color=f"C{j}", label=lab, elinewidth=0.8)
            x.axhline(ref, color="k", lw=0.5)
            x.set_title(k)
            x.grid(alpha=0.3)
        for x in axs[1]:
            x.set_xlabel("anti-kt jet pT [GeV]")
        for x in axs[:, 0]:
            x.set_ylabel(ylab)
        axs[0, 0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / fn, dpi=130)

    fig, axs = plt.subplots(1, len(runs), figsize=(4.2 * len(runs) + 1, 3.6), squeeze=False)
    for x, c, lab in zip(axs[0], runs, labels):
        m = np.array([c["observables"][k]["pull_combined"] for k in KEYS])
        im = x.imshow(m, cmap="RdBu_r", vmin=-10, vmax=10, aspect="auto")
        for (i, jj), v in np.ndenumerate(m):
            x.text(jj, i, f"{v:.0f}", ha="center", va="center", fontsize=7)
        x.set_xticks(range(len(ctr)), [f"{int(e[i])}–{int(e[i + 1])}" for i in range(len(ctr))], rotation=45, fontsize=7)
        x.set_yticks(range(len(KEYS)), KEYS, fontsize=8)
        x.set_title(f"{lab}: Σ pull/√P", fontsize=9)
    fig.colorbar(im, ax=axs[0].tolist(), shrink=0.8)
    fig.savefig(out / "ptbins_pulls.png", dpi=130, bbox_inches="tight")
    print(f"figures in {out}")


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
    r.add_argument("--no-redraw", dest="redraw", action="store_false")
    r.add_argument("--nsteps", type=int, default=128)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--batch", type=int, default=1024)
    r.add_argument("--device", default="auto")
    r.add_argument("--config", default=str(V2_CONFIG))
    p = sub.add_parser("plot")
    p.add_argument("--dirs", nargs="+", required=True)
    p.add_argument("--labels", nargs="+")
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    {"run": run, "plot": plot}[a.cmd](a)


if __name__ == "__main__":
    main()
