"""Direct check and retraining of the multiplicity model p(N | theta) of a checkpoint.

``check``: at every point of a direct-toy split (e.g. the ``fidelity`` jets at the
validation points) compare the toy's N with p(N | theta): mean shift in standard
errors, the relevance at the analysis N (|Δmean| √N / σ_N) and a binned χ² of
the N histogram (bins with expected count ≥ 5 merged from the tails). These are
the selected jets on both sides only up to the selection acting on the flow's
clouds; p(N | theta) is trained on selected toy jets, so it should match them.

``retrain``: the flow is conditioned on N as known context, so p(N | theta) can be
replaced without touching q(x | N, theta). Trains a new ``MultiplicityModel`` on
the training shards (selected on validation NLL) and writes a new checkpoint with
the same flow (metadata records the parent checkpoint and the new options).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from omnilearned.sbi.flow import FlowSurrogate, json_dumps
from omnilearned.sbi.multiplicity import MultiplicityModel, train_multiplicity
from omnilearned.sbi.toy.datasets import host_info, read_shards, shard_paths
from omnilearned.sbi.toy.simulator import PARAMETERS


def _counts(path, split):
    """(n_constituents, theta, point_id) of a split without reading the clouds."""
    import h5py

    n, th, pid = [], [], []
    for p in shard_paths(path, split):
        with h5py.File(p, "r") as f:
            n.append(f["data/n_constituents"][...])
            th.append(f["global"][:, :len(PARAMETERS)])
            pid.append(f["point_id"][...])
    return np.concatenate(n), np.concatenate(th), np.concatenate(pid)


def binned_chi2(counts, probs, min_expected=5.0):
    """χ², dof, p of observed integer ``counts`` against category ``probs``."""
    n = counts.size
    obs = np.bincount(counts, minlength=probs.size)[:probs.size].astype(float)
    exp = probs * n
    # merge categories from both ends until each bin expects >= min_expected
    edges, acc_o, acc_e = [], 0.0, 0.0
    bins_o, bins_e = [], []
    for o, e in zip(obs, exp):
        acc_o += o
        acc_e += e
        if acc_e >= min_expected:
            bins_o.append(acc_o)
            bins_e.append(acc_e)
            acc_o = acc_e = 0.0
    if bins_e:
        bins_o[-1] += acc_o
        bins_e[-1] += acc_e
    bins_o, bins_e = np.array(bins_o), np.array(bins_e)
    chi2 = float(((bins_o - bins_e) ** 2 / bins_e).sum())
    dof = max(1, len(bins_e) - 1)
    return chi2, dof, float(stats.chi2.sf(chi2, dof))


def check(surrogate, n, theta, pid, analysis_n=1000):
    rows = []
    for p in np.unique(pid):
        m = pid == p
        th = theta[m][0]
        lp = surrogate.log_multiplicity(np.arange(surrogate.multiplicity.max_n + 1), th[None],
                                        list(PARAMETERS))[0]
        prob = np.exp(lp)
        k = np.arange(prob.size)
        mu = float((prob * k).sum())
        sd = float(np.sqrt((prob * k * k).sum() - mu * mu))
        nn = n[m]
        dmu = mu - nn.mean()
        se = nn.std() / np.sqrt(nn.size)
        chi2, dof, pv = binned_chi2(nn, prob)
        rows.append({"point_id": int(p), "theta": th.tolist(), "direct_mean": float(nn.mean()),
                     "model_mean": mu, "model_sd": sd, "delta_mean": float(dmu), "z": float(dmu / se),
                     "relevance": float(abs(dmu) * np.sqrt(analysis_n) / nn.std()),
                     "chi2": chi2, "dof": dof, "p": pv, "n": int(nn.size)})
    z = np.array([r["z"] for r in rows])
    pv = np.array([r["p"] for r in rows])
    return {"points": rows, "n_points": len(rows), "rms_z": float(np.sqrt((z ** 2).mean())),
            "max_abs_z": float(np.abs(z).max()), "n_p_below_0.01": int((pv < 0.01).sum()),
            "max_relevance": float(max(r["relevance"] for r in rows)),
            "analysis_n": analysis_n}


def _print(doc, label):
    worst = sorted(doc["points"], key=lambda r: -abs(r["z"]))[:5]
    print(f"{label}: {doc['n_points']} points, rms z {doc['rms_z']:.2f}, max |z| {doc['max_abs_z']:.1f}, "
          f"chi2 p<0.01 at {doc['n_p_below_0.01']}, max relevance(N={doc['analysis_n']}) "
          f"{doc['max_relevance']:.2f}")
    for r in worst:
        print(f"   pt {r['point_id']} {np.round(r['theta'], 3).tolist()} dN {r['delta_mean']:+.3f} "
              f"z {r['z']:+.1f} rel {r['relevance']:.2f} chi2/dof {r['chi2']:.0f}/{r['dof']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--checkpoint", required=True)
    c.add_argument("--data-dir", required=True)
    c.add_argument("--split", default="fidelity")
    c.add_argument("--out", required=True)
    r = sub.add_parser("retrain")
    r.add_argument("--checkpoint", required=True)
    r.add_argument("--train-dir", required=True)
    r.add_argument("--counts-dir", default=None,
                   help="train p(N|theta) on counts shards (train_counts_*.h5) from this directory")
    r.add_argument("--val-dir", default=None,
                   help="directory of the validation split (default: --train-dir; JEWEL keeps it apart)")
    r.add_argument("--out", required=True, help="new checkpoint path")
    r.add_argument("--hidden", type=int, default=256)
    r.add_argument("--depth", type=int, default=4)
    r.add_argument("--steps", type=int, default=30000)
    r.add_argument("--batch-size", type=int, default=8192)
    r.add_argument("--lr", type=float, default=1e-3)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--device", default="cpu")
    r.add_argument("--threads", type=int, default=8)
    t = sub.add_parser("attach", help="put the p(N|theta) of one checkpoint onto another flow")
    t.add_argument("--checkpoint", required=True, help="flow checkpoint to keep")
    t.add_argument("--multiplicity-from", required=True, help="checkpoint whose p(N|theta) is used")
    t.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    sur = FlowSurrogate.load(a.checkpoint)
    if a.cmd == "attach":
        src = FlowSurrogate.load(a.multiplicity_from)
        if src.multiplicity.layout.to_dict() != sur.multiplicity.layout.to_dict():
            raise ValueError("multiplicity condition layouts differ")
        meta = dict(sur.metadata)
        parent = meta.pop("checkpoint_sha256")
        meta["multiplicity_attached"] = {
            "parent_checkpoint": a.checkpoint, "parent_sha256": parent,
            "multiplicity_from": a.multiplicity_from,
            "multiplicity_from_sha256": src.metadata["checkpoint_sha256"],
            "multiplicity_retrained": src.metadata.get("multiplicity_retrained"), "host": host_info()}
        new = FlowSurrogate(sur.model, sur.base, sur.transform, sur.layout, meta, "cpu",
                            multiplicity=src.multiplicity)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        print(f"attached p(N|theta) of {a.multiplicity_from} -> {a.out} sha {new.save(a.out)[:12]}")
        return
    if a.cmd == "check":
        n, th, pid = _counts(a.data_dir, a.split)
        doc = check(sur, n, th, pid)
        doc.update({"checkpoint": a.checkpoint, "checkpoint_sha256": sur.metadata["checkpoint_sha256"],
                    "data": a.data_dir, "split": a.split, "host": host_info()})
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w") as f:
            f.write(json_dumps(doc))
        _print(doc, Path(a.checkpoint).name)
        return
    torch.set_num_threads(a.threads)
    start = time.time()
    if a.counts_dir:
        from omnilearned.sbi.toy.datasets import read_counts

        n, th, _ = read_counts(sorted(Path(a.counts_dir).glob("train_counts_*.h5")))
    else:
        n, th, _ = _counts(a.train_dir, "train")
    vn, vth, _ = _counts(a.val_dir or a.train_dir, "validation")
    lay = sur.multiplicity.layout
    cond = lambda t: torch.from_numpy(lay.assemble(t, list(PARAMETERS)))  # noqa: E731
    model = MultiplicityModel(len(PARAMETERS), sur.multiplicity.max_n, hidden=a.hidden, depth=a.depth)
    model.layout = lay
    model, hist = train_multiplicity(model, cond(th), n, cond(vth), vn, steps=a.steps,
                                     batch_size=a.batch_size, lr=a.lr, seed=a.seed,
                                     device=a.device, val_every=1000)
    meta = dict(sur.metadata)
    parent = meta.pop("checkpoint_sha256")
    meta.update({"multiplicity_retrained": {"parent_checkpoint": a.checkpoint, "parent_sha256": parent,
                                            "train_counts": a.counts_dir, "train_dir": a.train_dir,
                                            "val_dir": a.val_dir or a.train_dir, "n_train": int(n.size),
                                            "hidden": a.hidden, "depth": a.depth, "steps": a.steps,
                                            "batch_size": a.batch_size, "lr": a.lr, "seed": a.seed,
                                            "best_val_nll": hist["best_val_nll"],
                                            "best_step": hist["best_step"],
                                            "seconds": time.time() - start, "host": host_info()}})
    new = FlowSurrogate(sur.model, sur.base, sur.transform, sur.layout, meta, "cpu", multiplicity=model)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    sha = new.save(a.out)
    print(f"retrained p(N|theta): best val NLL {hist['best_val_nll']:.5f} at {hist['best_step']}, "
          f"{time.time() - start:.0f}s -> {a.out} sha {sha[:12]}")


if __name__ == "__main__":
    main()
