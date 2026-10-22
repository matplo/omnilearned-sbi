"""Train a toy-v2 NLE surrogate q(x, N | theta) = p(N | theta) q(x | N, theta).

Reads training and validation shards (``toy.datasets``), fits the feature
transform on the training split only, trains the categorical multiplicity
model (selected on validation NLL) and the flow-matching velocity (selected on
the validation flow-matching loss of the EMA weights), and writes one
checkpoint holding both, plus ``run.json`` (options, data hashes, history,
source revision, device, matmul precision).

The flow condition is theta mapped to [-1, 1] over the box plus N / 100 as
known context (``layout_with_multiplicity``). Training may use TF32 matmuls
(``--tf32``, recorded); evaluation always runs in IEEE float32.

    python -m omnilearned.sbi.toy.train --data-dir DATA --out RUN --steps 1000
"""

import argparse
import copy
import json
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from omnilearned.sbi.base import GaussianBase
from omnilearned.sbi.device import resolve_device
from omnilearned.sbi.flow import FlowSurrogate, TrainOptions, json_dumps, train_flow
from omnilearned.sbi.multiplicity import (
    MultiplicityModel,
    layout_with_multiplicity,
    train_multiplicity,
)
from omnilearned.sbi.parameters import ConditionLayout
from omnilearned.sbi.toy.config import V2_CONFIG, load_toy_config
from omnilearned.sbi.toy.datasets import BOX, box_arrays, host_info, read_shards, shard_paths
from omnilearned.sbi.observations import file_sha256
from omnilearned.sbi.toy.simulator import PARAMETERS
from omnilearned.sbi.transforms import FeatureTransform
from omnilearned.sbi.velocity import SetFlowVelocity
from omnilearned.sbi.velocity_pet import SmoothPETVelocity

FEATURES = ["eta", "phi", "log_pt_over_GeV"]


def prepare(data, transform, layout):
    """Model-coordinate tensors for ``train_flow``: u, mask, cond."""
    x = torch.as_tensor(data["x"], dtype=torch.float64)
    mask = torch.as_tensor(data["mask"], dtype=torch.bool)
    u, _ = transform.forward(x, mask)
    cond = layout.assemble(data["theta"], list(PARAMETERS),
                           context=data["n_constituents"][:, None].astype(np.float64))
    return {"u": u.float(), "mask": mask, "cond": torch.from_numpy(cond)}


def build_surrogate(args, train, val, out, log=print):
    _, _, sel = load_toy_config(args.config)
    lo, hi = box_arrays(BOX)
    transform = FeatureTransform.fit(
        FEATURES, [sel.constituent_abs_eta_max, sel.constituent_abs_phi_max_rad, None],
        train["x"], train["mask"])
    base = GaussianBase.standard(3)
    layout = layout_with_multiplicity(list(PARAMETERS), lo, hi)
    mlayout = ConditionLayout.from_bounds(list(PARAMETERS), lo, hi)
    dev = resolve_device(args.device)

    t0 = time.time()
    mult = MultiplicityModel(len(PARAMETERS), sel.max_constituents, hidden=args.mult_hidden,
                             depth=args.mult_depth)
    mult.layout = mlayout
    mcond = lambda d: torch.from_numpy(mlayout.assemble(d["theta"], list(PARAMETERS)))  # noqa: E731
    mult, mhist = train_multiplicity(mult, mcond(train), train["n_constituents"], mcond(val),
                                     val["n_constituents"], steps=args.mult_steps,
                                     device=dev.device, log=log)
    mhist["seconds"] = time.time() - t0

    if args.arch == "set":
        model = SetFlowVelocity(3, layout.dim, dim=args.dim, depth=args.depth, heads=args.heads,
                                cond_embed=True, cond_to_particles=True)
    else:
        model = SmoothPETVelocity(3, layout.dim, dim=args.dim, depth=args.depth, heads=args.heads,
                                  local_dim=args.local_dim)
    opts = TrainOptions(**{f.name: getattr(args, f.name) for f in fields(TrainOptions)
                           if hasattr(args, f.name)})
    tr, va = prepare(train, transform, layout), prepare(val, transform, layout)
    meta = {"backend": "toy-quenching-v2", "arch": args.arch, "parameters": list(PARAMETERS), "box": BOX,
            "train_options": opts.to_dict(), "tf32_training": bool(args.tf32),
            "n_params": int(sum(p.numel() for p in model.parameters()))}

    def save(m, step, vl):
        if step >= args.save_after:
            s = FlowSurrogate(copy.deepcopy(m).cpu(), base, transform, layout,
                              dict(meta, best_step=step, best_val_loss=vl), "cpu", multiplicity=mult)
            s.save(out / "checkpoint.pt")

    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
    torch.set_float32_matmul_precision("high" if args.tf32 else "highest")
    log(f"flow: {meta['n_params']} parameters, train {tr['u'].shape}, val {va['u'].shape}, "
        f"device {dev.device}")
    best, hist = train_flow(model, base, tr, va, opts, log=log, on_best=save)
    torch.set_float32_matmul_precision("highest")
    sur = FlowSurrogate(best, base, transform, layout,
                        dict(meta, best_step=hist["best_step"], best_val_loss=hist["best_val_loss"]),
                        "cpu", multiplicity=mult)
    sha = sur.save(out / "checkpoint.pt")
    return sha, hist, mhist, meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=str(V2_CONFIG))
    ap.add_argument("--arch", default="set", choices=["set", "pet_smooth"],
                    help="SetFlowVelocity or SmoothPETVelocity (PET2 with smooth locality)")
    ap.add_argument("--local-dim", dest="local_dim", type=int, default=64,
                    help="pet_smooth: width of the pairwise local messages")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch-size", dest="batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-steps", dest="warmup_steps", type=int, default=200)
    ap.add_argument("--ema-decay", dest="ema_decay", type=float, default=0.999)
    ap.add_argument("--val-every", dest="val_every", type=int, default=250)
    ap.add_argument("--val-batches", dest="val_batches", type=int, default=8)
    ap.add_argument("--length-buckets", dest="length_buckets", type=int, default=0,
                    help="draw each minibatch inside one of k multiplicity quantile buckets")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tf32", action="store_true", help="TF32 matmuls during training only")
    ap.add_argument("--mult-steps", type=int, default=2000)
    ap.add_argument("--mult-hidden", type=int, default=128)
    ap.add_argument("--mult-depth", type=int, default=3)
    ap.add_argument("--save-after", type=int, default=0,
                    help="first step at which an improving checkpoint is written")
    ap.add_argument("--cpu-threads", type=int, default=8,
                    help="torch CPU threads (noise is drawn on the CPU; several trainings "
                         "per node oversubscribe the cores with the default)")
    args = ap.parse_args(argv)
    torch.set_num_threads(args.cpu_threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "train.log", "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    start = time.time()
    host = host_info()  # source revision at start (the checkout may change while training)
    log(f"source {host['source_revision']} dirty={host['source_dirty']} host {host['host']}")
    tp, vp = shard_paths(args.data_dir, "train"), shard_paths(args.data_dir, "validation")
    train, val = read_shards(tp), read_shards(vp)
    log(f"data: train {train['x'].shape} from {len(tp)} shards, val {val['x'].shape} "
        f"from {len(vp)} shards, {time.time() - start:.1f}s")
    data = {"train": [{"file": str(p), "sha256": file_sha256(p)} for p in tp],
            "validation": [{"file": str(p), "sha256": file_sha256(p)} for p in vp],
            "n_train": int(train["x"].shape[0]), "n_val": int(val["x"].shape[0]),
            "train_points": int(np.unique(train["point_id"]).size),
            "val_points": int(np.unique(val["point_id"]).size)}
    sha, hist, mhist, meta = build_surrogate(args, train, val, out, log)
    dev = hist["device"]
    run = {"checkpoint": str(out / "checkpoint.pt"), "checkpoint_sha256": sha, "args": vars(args),
           "data": data, "meta": meta, "flow_history": hist, "multiplicity_history": mhist,
           "wall_seconds": time.time() - start, "host": host, "device": dev,
           "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None}
    with open(out / "run.json", "w") as f:
        f.write(json_dumps(run))
    log(f"done: best step {hist['best_step']} val {hist['best_val_loss']:.5f}, "
        f"multiplicity val NLL {mhist['best_val_nll']:.4f}, {run['wall_seconds']:.0f}s, sha {sha[:12]}")


if __name__ == "__main__":
    main()
