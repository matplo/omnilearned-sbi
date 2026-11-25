"""P6 Part B: train JetPtModel (log p(pT_J|theta)) on real v3 data, attach to a new checkpoint.

JetPtModel (src/omnilearned/sbi/jetpt.py) needs only (theta, jet_pt, weight) per event -- none
of the flow/multiplicity sub-models' weights -- so this does NOT retrain the main flow. It
reuses the exact data path (config, data-dir) and reader (toy.train.shards_for_config) the
production run used (see that checkpoint's own run.json for --data-dir/--config/--val-dir),
so JetPtModel sees the same training/validation split; trains a JetPtModel by weighted MLE
(toy.train's weight_mode="loss" convention, same hard-pT-biased-shard correction used
elsewhere); loads the EXISTING production checkpoint, sets its `.jet_pt` sub-model, and saves
to a NEW checkpoint directory -- the original checkpoint is never modified.

    PYTHONPATH=$V3_REPO/src henv -n omni-sbi -x python scripts/p6_train_jetpt.py \\
        --checkpoint CKPT --data-dir DATA --config CONFIG --out OUT_DIR \\
        [--max-train N --max-val N]   # smoke test on a subsample before the full run
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch


def read_jetpt_columns(paths):
    """theta/global/global_names/weight from v3 shard files, WITHOUT the constituent clouds.

    read_jets/read_v3_shards (jewel/v3format.py) always load the full "data"/"mask" arrays
    (the per-particle clouds) via h5py's f[k][...] -- for ~10M events x up to ~90 padded
    slots that's tens of GB loaded only to be discarded, since JetPtModel needs just
    (theta, jet_pt, weight) per event. This reads only the lightweight per-event datasets.
    """
    theta, glob, weight, names = [], [], [], None
    for p in paths:
        with h5py.File(p, "r") as f:
            theta.append(f["theta"][...])
            glob.append(f["global"][...])
            weight.append(f["weight"][...])
            gn = tuple(f["global"].attrs["global_names"])
            if names is None:
                names = gn
            elif names != gn:
                raise ValueError(f"{p} disagrees on global_names")
    return {"theta": np.concatenate(theta), "global": np.concatenate(glob),
           "global_names": names, "weight": np.concatenate(weight)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", required=True, help="existing production checkpoint to attach to")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--val-dir", default=None, help="default: --data-dir")
    ap.add_argument("--config", required=True, help="e.g. configs/jewel-2.6.0-cs-v3.toml")
    ap.add_argument("--out", required=True, help="output directory; writes checkpoint.pt + run.json")
    ap.add_argument("--n-components", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-mode", default="loss", choices=["none", "loss"],
                    help="loss: weighted NLL by the shards' own per-jet weight (hard-pT "
                         "biased sample -- matches the production flow/multiplicity training)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-train", type=int, default=None, help="subsample for a quick smoke test")
    ap.add_argument("--max-val", type=int, default=None)
    a = ap.parse_args()

    from omnilearned.sbi.device import resolve_device
    from omnilearned.sbi.flow import FlowSurrogate
    from omnilearned.sbi.jetpt import JetPtModel, layout_with_jet_pt, train_jet_pt
    from omnilearned.sbi.observations import file_sha256
    from omnilearned.sbi.jewel.v3format import shard_paths_v3
    from omnilearned.sbi.toy.datasets import box_arrays
    from omnilearned.sbi.toy.simulator import PARAMETERS
    from omnilearned.sbi.toy.train import sample_setup

    dev = resolve_device(a.device)
    backend, _, box = sample_setup(a.config)
    if not backend.endswith("-v3"):
        raise ValueError(f"JetPtModel is a v3-only sub-model; got backend {backend!r}")
    lo, hi = box_arrays(box)

    tp = shard_paths_v3(a.data_dir, "train")
    vp = shard_paths_v3(a.val_dir or a.data_dir, "validation")
    print(f"reading {len(tp)} train shards, {len(vp)} val shards (theta/global/weight only)...",
          flush=True)
    train = read_jetpt_columns(tp)
    val = read_jetpt_columns(vp)
    print(f"train {train['theta'].shape[0]} events, val {val['theta'].shape[0]} events", flush=True)

    rng = np.random.default_rng(a.seed)

    def subsample(data, k):
        if k is None or k >= data["theta"].shape[0]:
            return data
        idx = rng.choice(data["theta"].shape[0], k, replace=False)
        return {"theta": data["theta"][idx], "global": data["global"][idx],
                "global_names": data["global_names"], "weight": np.asarray(data["weight"])[idx]}

    train = subsample(train, a.max_train)
    val = subsample(val, a.max_val)
    print(f"using train {train['theta'].shape[0]}, val {val['theta'].shape[0]}", flush=True)

    layout = layout_with_jet_pt(list(PARAMETERS), lo, hi)
    cond_train = torch.from_numpy(layout.assemble(train["theta"], list(PARAMETERS))).float()
    cond_val = torch.from_numpy(layout.assemble(val["theta"], list(PARAMETERS))).float()

    def log_jet_pt(data):
        idx = list(data["global_names"]).index("jet_pt")
        return np.log(np.asarray(data["global"])[:, idx].astype(np.float64))

    log_pt_train, log_pt_val = log_jet_pt(train), log_jet_pt(val)

    w_train = w_val = None
    if a.weight_mode == "loss":
        w_train = torch.as_tensor(np.asarray(train["weight"]), dtype=torch.float32)
        w_val = torch.as_tensor(np.asarray(val["weight"]), dtype=torch.float32)

    model = JetPtModel(layout.dim, n_components=a.n_components, hidden=a.hidden, depth=a.depth)
    model, hist = train_jet_pt(model, cond_train, log_pt_train, cond_val, log_pt_val,
                               steps=a.steps, batch_size=a.batch_size, lr=a.lr, seed=a.seed,
                               device=dev.device, weights=w_train, val_weights=w_val)
    model.layout = layout
    print(f"jet_pt best_val_nll={hist['best_val_nll']:.4f} at step {hist['best_step']}", flush=True)

    sur = FlowSurrogate.load(a.checkpoint, device="cpu")
    if sur.jet_pt is not None:
        print("WARNING: base checkpoint already has a jet_pt sub-model; overwriting it", flush=True)
    sur.jet_pt = model

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "checkpoint.pt"
    sha = sur.save(ckpt_path)
    run = {
        "base_checkpoint": a.checkpoint, "base_checkpoint_sha256": file_sha256(Path(a.checkpoint)),
        "checkpoint": str(ckpt_path), "checkpoint_sha256": sha,
        "data_dir": a.data_dir, "val_dir": a.val_dir or a.data_dir, "config": a.config,
        "n_train": int(train["theta"].shape[0]), "n_val": int(val["theta"].shape[0]),
        "args": vars(a), "jet_pt_history": hist,
    }
    (out_dir / "run.json").write_text(json.dumps(run, indent=1, default=float))
    print(f"wrote {ckpt_path} sha={sha[:12]}", flush=True)


if __name__ == "__main__":
    main()
