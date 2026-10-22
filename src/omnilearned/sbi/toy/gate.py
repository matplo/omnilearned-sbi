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


def response_gradients(points, means, dev, labels, window=5):
    """d mean_k / d theta_p at ``dev`` from quadratic fits along the three lines.

    points [n, 3], means [n, K]; ``labels`` 'line_p' marks the line along axis p.
    Uses the ``window`` line points nearest to the development value.
    """
    points, means = np.asarray(points, float), np.asarray(means, float)
    G = np.zeros((means.shape[1], 3))
    for p in range(3):
        idx = np.array([i for i, s in enumerate(labels) if s == f"line_{p}"])
        near = idx[np.argsort(np.abs(points[idx, p] - dev[p]))[:window]]
        u = points[near, p] - dev[p]
        for k in range(means.shape[1]):
            c = np.polyfit(u, means[near, k], 2)
            G[k, p] = c[1]
    return G


def shift(a):
    """Equivalent parameter shift of the surrogate at the development setting and the tube."""
    from omnilearned.sbi.toy.simulator import PARAMETERS

    dev_ = resolve_device(a.device)
    sur = FlowSurrogate.load(a.checkpoint, device=dev_.device)
    cfg, _, sel = load_toy_config(a.config)
    with open(Path(a.scan_dir) / "design_development_scan.json") as f:
        des = json.load(f)
    dev = np.asarray(des["development"], float)
    pts, labels = np.asarray(des["points"]), des["labels"]
    d = read_shards(shard_paths(a.scan_dir, "development_scan"))
    keys = fidelity.JET_VARIABLE
    means, obs_by_point = [], {}
    for pid in range(len(pts)):
        m = d["point_id"] == pid
        x, mk = d["x"][m], d["mask"][m]
        w = max(1, int(mk.sum(1).max()))
        o = fidelity.all_obs(x[:, :w], mk[:, :w])
        obs_by_point[pid] = (x[:, :w], mk[:, :w], np.column_stack([o[k] for k in keys]))
        means.append(obs_by_point[pid][2].mean(0))
    G = response_gradients(pts, np.array(means), dev, labels)
    at_dev = [i for i, p in enumerate(pts) if np.allclose(p, dev)]
    targets = at_dev[:1] + [i for i, s in enumerate(labels) if s == "tube"]
    if a.max_points:
        targets = targets[:a.max_points]
    C = np.cov(obs_by_point[targets[0]][2], rowvar=False)
    F = G.T @ np.linalg.pinv(C) @ G  # per-jet Fisher information from these observables
    out = {"parameters": list(PARAMETERS), "observables": keys, "G": G, "C_dev": C,
           "fisher_per_jet": F, "points": []}
    for n in ANALYSIS_N:
        out[f"sigma_conditional_N{n}"] = 1 / np.sqrt(np.diag(F) * n)
        out[f"sigma_marginal_N{n}"] = np.sqrt(np.diag(np.linalg.pinv(F)) / n)
    for pid in targets:
        xd, md, od = obs_by_point[pid]
        xs, ms, rep = sample_and_select(sur, pts[pid], a.n_generate, sel, seed=a.seed + pid,
                                        nsteps=a.nsteps, batch=a.batch)
        o = fidelity.all_obs(xs, ms)
        osur = np.column_stack([o[k] for k in keys])
        dmu = osur.mean(0) - od.mean(0)
        cov = np.cov(od, rowvar=False) / len(od) + np.cov(osur, rowvar=False) / len(osur)
        dth, cov_th = fidelity.equivalent_shift(G, C, dmu, cov)
        se = np.sqrt(np.diag(cov_th))
        sig = out["sigma_conditional_N1000"]
        row = {"point_id": pid, "label": labels[pid], "theta": pts[pid], "delta_mean": dmu,
               "delta_theta": dth, "delta_theta_se": se, "in_sigma_conditional_N1000": dth / sig,
               "se_in_sigma_conditional_N1000": se / sig,
               "pass_0.2_sigma": bool(np.all(np.maximum(np.abs(dth) - 2 * se, 0) <= 0.2 * sig)),
               "surrogate": rep, "n_direct": int(len(od))}
        out["points"].append(row)
        print(f"point {pid} {labels[pid]} theta={np.round(pts[pid], 3).tolist()} "
              f"dtheta={np.round(dth, 4).tolist()} +- {np.round(se, 4).tolist()} "
              f"= {np.round(dth / sig, 2).tolist()} sigma_cond(N=1000) acc={rep['acceptance']:.3f}",
              flush=True)
    out.update({"checkpoint": a.checkpoint, "checkpoint_sha256": sur.metadata.get("checkpoint_sha256"),
                "nsteps": a.nsteps, "n_generate": a.n_generate, "device": dev_.to_dict(),
                "host": host_info()})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        f.write(json_dumps(out))
    print("sigma_conditional N=1000", np.round(out["sigma_conditional_N1000"], 4).tolist(),
          "marginal", np.round(out["sigma_marginal_N1000"], 4).tolist())


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
        p.add_argument("--batch", type=int, default=1024)  # sorted by N: narrow batches are faster
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
    h = sub.add_parser("shift", help="equivalent parameter shift at the development setting and tube")
    h.add_argument("--checkpoint", required=True)
    h.add_argument("--scan-dir", required=True)
    h.add_argument("--out", required=True)
    h.add_argument("--n-generate", type=int, default=22000)
    h.add_argument("--nsteps", type=int, default=128)
    h.add_argument("--seed", type=int, default=0)
    h.add_argument("--batch", type=int, default=1024)
    h.add_argument("--max-points", type=int, default=None)
    h.add_argument("--device", default="auto")
    h.add_argument("--config", default=str(V2_CONFIG))
    s = sub.add_parser("summarize")
    s.add_argument("--out", required=True)
    s.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args(argv)
    {"run": run, "summarize": summarize, "convergence": convergence, "shift": shift}[a.cmd](a)


if __name__ == "__main__":
    main()
