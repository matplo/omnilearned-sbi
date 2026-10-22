"""M3 fidelity gate at scale: surrogate vs direct toy at the validation points.

``run``: for each validation point (optionally a shard of the points, one
process per GPU) draw clouds from the surrogate (N from p(N | theta), IEEE
float32, converged sampler), apply the toy's v2 selection, and gate them
against the direct-toy ``fidelity`` jets at the same point
(``fidelity.evaluate``: permutation family-wise p-value over jet and
constituent observables, mean relevance at the analysis N). One JSON per
point (resumable) plus the selected surrogate clouds (HDF5).

``summarize``: Bonferroni verdicts with S = number of points
(significant and relevant at N = 1000 fails; relevance at 250 recorded), the
global uniformity check of the family-wise p-values, and a table.

``convergence``: lesson 5b. The same base noise and N draws at several step
counts; observable mean shifts between successive step counts in standard
errors, and the KS distance, per point.
"""

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from omnilearned.sbi import fidelity
from omnilearned.sbi.device import resolve_device
from omnilearned.sbi.flow import FlowSurrogate, json_dumps
from omnilearned.sbi.toy.config import V2_CONFIG, load_toy_config
from omnilearned.sbi.toy.datasets import host_info, read_shards, shard_paths
from omnilearned.sbi.toy.sample_select import sample_and_select

ANALYSIS_N = (250, 1000)


def _pad(v, w):
    return np.pad(v, [(0, 0), (0, w - v.shape[1])] + [(0, 0)] * (v.ndim - 2))


def direct_by_point(fidelity_dir):
    d = read_shards(shard_paths(fidelity_dir, "fidelity"))
    out = {}
    for pid in np.unique(d["point_id"]):
        sel = d["point_id"] == pid
        x, m = d["x"][sel], d["mask"][sel]
        w = max(1, int(m.sum(1).max()))
        out[int(pid)] = (x[:, :w], m[:, :w], d["theta"][sel][0])
    return out


def gate_point(sur, sel, theta, xd, md, n_generate, nsteps, seed, n_perm, solver="midpoint",
               batch=4096):
    xs, ms, rep = sample_and_select(sur, theta, n_generate, sel, seed=seed, nsteps=nsteps,
                                    solver=solver, batch=batch)
    w = max(xd.shape[1], xs.shape[1])
    t0 = time.time()
    res = fidelity.evaluate(_pad(xd, w), _pad(xs, w), n_events=max(ANALYSIS_N), n_perm=n_perm,
                            seed=seed, direct_mask=_pad(md, w), surrogate_mask=_pad(ms, w))
    oa = fidelity.all_obs(_pad(xd, w), _pad(md, w))
    ob = fidelity.all_obs(_pad(xs, w), _pad(ms, w))
    res["relevance_by_n"] = {str(n): fidelity.mean_relevance(oa, ob, fidelity.JET_VARIABLE, n)
                             for n in ANALYSIS_N}
    res["surrogate"] = rep
    res["n_direct"], res["n_surrogate"] = int(xd.shape[0]), int(xs.shape[0])
    res["gate_seconds"] = time.time() - t0
    return res, xs, ms


def run(a):
    dev = resolve_device(a.device)
    sur = FlowSurrogate.load(a.checkpoint, device=dev.device)
    _, _, sel = load_toy_config(a.config)
    direct = direct_by_point(a.fidelity_dir)
    pids = sorted(direct)
    if a.points:
        pids = [p for p in pids if p in set(a.points)]
    pids = pids[a.shard::a.num_shards]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    host = host_info()
    for pid in pids:
        js = out / f"point_{pid:05d}.json"
        if js.exists():
            continue
        xd, md, theta = direct[pid]
        if a.max_direct:
            xd, md = xd[:a.max_direct], md[:a.max_direct]
        t0 = time.time()
        res, xs, ms = gate_point(sur, sel, theta, xd, md, a.n_generate, a.nsteps, a.seed + pid,
                                 a.n_perm, batch=a.batch)
        res.update({"point_id": pid, "theta": theta.tolist(), "checkpoint": a.checkpoint,
                    "checkpoint_sha256": sur.metadata.get("checkpoint_sha256"),
                    "nsteps": a.nsteps, "device": dev.to_dict(), "host": host,
                    "gpu_name": torch.cuda.get_device_name() if dev.device == "cuda" else None,
                    "seconds": time.time() - t0})
        if a.save_samples:
            with h5py.File(out / f"samples_{pid:05d}.h5", "w") as f:
                f.create_dataset("x", data=xs.astype(np.float32), compression="gzip")
                f.create_dataset("mask", data=ms, compression="gzip")
                f.attrs["theta"] = theta
        with open(js, "w") as f:
            f.write(json_dumps(res))
        v = res["verdict"]
        print(f"point {pid} theta={np.round(theta, 3).tolist()} acc={res['surrogate']['acceptance']:.3f} "
              f"p={res['permutation']['p_family']:.3g} sig={v['significant']} rel={v['relevant']} "
              f"{res['seconds']:.0f}s", flush=True)


def summarize(a):
    rows = []
    for p in sorted(Path(a.out).glob("point_*.json")):
        with open(p) as f:
            rows.append(json.load(f))
    S = len(rows)
    alpha_s = a.alpha / max(S, 1)
    table = []
    for r in rows:
        p = r["permutation"]["p_family"]
        rel = {n: [k for k, v in r["relevance_by_n"][n].items() if v["relevant"]]
               for n in r["relevance_by_n"]}
        sig = p < alpha_s
        table.append({"point_id": r["point_id"], "theta": r["theta"], "p_family": p,
                      "significant": bool(sig), "relevant_1000": rel["1000"], "relevant_250": rel["250"],
                      "fail": bool(sig and rel["1000"]),
                      "max_relevance_1000": max(v["relevance"] for v in r["relevance_by_n"]["1000"].values()),
                      "acceptance": r["surrogate"]["acceptance"],
                      "n_direct": r["n_direct"], "n_surrogate": r["n_surrogate"]})
    glob = fidelity.global_uniformity([t["p_family"] for t in table]) if table else None
    doc = {"points": S, "alpha": a.alpha, "alpha_per_point": alpha_s,
           "fails": [t["point_id"] for t in table if t["fail"]],
           "significant": [t["point_id"] for t in table if t["significant"]],
           "global_uniformity": glob, "table": table,
           "checkpoint_sha256": sorted({r["checkpoint_sha256"] for r in rows}),
           "nsteps": sorted({r["nsteps"] for r in rows})}
    with open(Path(a.out) / "summary.json", "w") as f:
        f.write(json_dumps(doc))
    print(f"{S} points, alpha/S {alpha_s:.2e}: {len(doc['fails'])} FAIL, "
          f"{len(doc['significant'])} significant; uniformity {glob}")
    for t in table:
        flag = "FAIL" if t["fail"] else ("sig " if t["significant"] else "    ")
        print(f"  {flag} {t['point_id']:4d} {np.round(t['theta'], 3).tolist()} p={t['p_family']:.3g} "
              f"acc={t['acceptance']:.3f} maxrel1000={t['max_relevance_1000']:.2f} {t['relevant_1000']}")


def convergence(a):
    dev = resolve_device(a.device)
    sur = FlowSurrogate.load(a.checkpoint, device=dev.device)
    _, _, sel = load_toy_config(a.config)
    direct = direct_by_point(a.fidelity_dir)
    out = {}
    for pid in a.points:
        theta = direct[pid][2]
        obs = {}
        for ns in a.steps:
            t0 = time.time()
            x, m = sur.sample(theta, a.n_generate, None, seed=a.seed, nsteps=ns, solver="midpoint",
                              candidate_names=["b", "T_ref", "tau_init"], batch=a.batch,
                              sort_by_length=True)
            ok, _, _ = __import__("omnilearned.sbi.toy.selection", fromlist=["x"]).select_clouds(
                x.astype(np.float64), sel, m)
            o = fidelity.all_obs(x, m)  # all clouds (paired across step counts)
            obs[ns] = {"obs": o, "acc": float(ok.mean()), "seconds": time.time() - t0}
        rep = {}
        for lo, hi in zip(a.steps[:-1], a.steps[1:]):
            r = {}
            for k in fidelity.JET_VARIABLE:
                d = obs[hi]["obs"][k] - obs[lo]["obs"][k]  # paired: same noise and N
                good = np.isfinite(d)
                se = obs[hi]["obs"][k][good].std() / np.sqrt(good.sum())
                r[k] = {"mean_shift_se": float(d[good].mean() / se),
                        "ks": float(fidelity.ks_d(obs[lo]["obs"][k], obs[hi]["obs"][k]))}
            r["acceptance"] = [obs[lo]["acc"], obs[hi]["acc"]]
            rep[f"{lo}->{hi}"] = r
        out[str(pid)] = {"theta": theta.tolist(), "pairs": rep,
                         "seconds": {str(k): v["seconds"] for k, v in obs.items()}}
        for pair, r in rep.items():
            worst = max(fidelity.JET_VARIABLE, key=lambda k: abs(r[k]["mean_shift_se"]))
            print(f"point {pid} {pair}: worst mean shift {worst} {r[worst]['mean_shift_se']:+.2f} se "
                  f"(n={a.n_generate}), max KS {max(r[k]['ks'] for k in fidelity.JET_VARIABLE):.4f}, "
                  f"acc {r['acceptance']}", flush=True)
    doc = {"checkpoint": a.checkpoint, "checkpoint_sha256": sur.metadata.get("checkpoint_sha256"),
           "steps": a.steps, "n_generate": a.n_generate, "seed": a.seed, "points": out,
           "device": dev.to_dict(), "host": host_info()}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        f.write(json_dumps(doc))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "convergence"):
        p = sub.add_parser(name)
        p.add_argument("--checkpoint", required=True)
        p.add_argument("--fidelity-dir", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--n-generate", type=int, default=22000)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--batch", type=int, default=4096)
        p.add_argument("--device", default="auto")
        p.add_argument("--config", default=str(V2_CONFIG))
    r = sub.choices["run"]
    r.add_argument("--nsteps", type=int, default=128)
    r.add_argument("--n-perm", type=int, default=200)
    r.add_argument("--points", type=int, nargs="*", default=None)
    r.add_argument("--shard", type=int, default=0)
    r.add_argument("--num-shards", type=int, default=1)
    r.add_argument("--max-direct", type=int, default=None)
    r.add_argument("--save-samples", action="store_true")
    c = sub.choices["convergence"]
    c.add_argument("--points", type=int, nargs="+", required=True)
    c.add_argument("--steps", type=int, nargs="+", default=[64, 128, 256])
    s = sub.add_parser("summarize")
    s.add_argument("--out", required=True)
    s.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args(argv)
    {"run": run, "summarize": summarize, "convergence": convergence}[a.cmd](a)


if __name__ == "__main__":
    main()
