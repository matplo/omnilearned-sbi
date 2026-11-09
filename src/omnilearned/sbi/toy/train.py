"""Train an NLE surrogate q(x, N | theta) = p(N | theta) q(x | N, theta).

Reads training and validation shards, fits the feature transform on the
training split only, trains the categorical multiplicity model (selected on
validation NLL) and the flow-matching velocity (selected on the validation
flow-matching loss of the EMA weights), and writes one checkpoint holding
both, plus ``run.json`` (options, data hashes, history, source revision,
device, matmul precision).

The flow condition is theta mapped to [-1, 1] over the box plus N / 100 as
known context (``layout_with_multiplicity``). Weighted samples (hard-pT biased
shards, per-jet ``weight``) are trained with ``--weight-mode loss|sample``; the
multiplicity NLL is then weighted too, and run.json records the effective sample
size per point and jet-pT bin. Training may use TF32 matmuls
(``--tf32``, recorded); evaluation always runs in IEEE float32.

Dispatches on ``--config``'s backend id (``sample_setup``): a toy or JEWEL v2 backend reads
``toy.datasets`` shards (3 features: absolute eta, phi, log pT) and fits an affine
standardisation; a backend ending ``-v3`` reads OmniLearned-native merged files
(``jewel.v3format``, 4 features: deta/dphi relative to the jet axis, log pT, log E), skips
standardisation (docs/FORMAT_JEWEL_V3.md: the paper's minimal set is already O(1)-scaled), and
conditions both the flow and the multiplicity model on jet_pt as known context in addition to
theta -- the multiplicity model's own layout (``layout_for_multiplicity_model``) never on
n_constituents, since that is what it predicts.

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
    MULTIPLICITY_COLUMN,
    MultiplicityModel,
    layout_for_multiplicity_model,
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
# v3 (docs/FORMAT_JEWEL_V3.md): deta, dphi are relative to the jet axis (v2's were absolute --
# the defect the v3 fork exists to fix), plus log E. The on-disk fifth column (pid, the PDG
# code) is dropped by read_v3_shards: the three-factor likelihood never conditions on it.
FEATURES_V3 = ["deta", "dphi", "log_pt", "log_e"]
PT_EDGES = (60.0, 70, 80, 90, 100, 120, 140, 170, 200)  # as toy.ptbins


def weight_summary(data, edges=PT_EDGES):
    """Effective sample size (sum w)^2 / sum w^2 of a split: total, per point, per jet-pT bin."""
    w, pid = np.asarray(data["weight"], np.float64), data["point_id"]
    pt = data["global"][:, list(data["global_names"]).index("jet_pt")]
    nb = len(edges) - 1
    _, p = np.unique(pid, return_inverse=True)
    key = p * nb + np.clip(np.digitize(pt, edges) - 1, 0, nb - 1)
    size = (p.max() + 1) * nb if p.size else nb
    s1, s2 = np.bincount(key, w, size), np.bincount(key, w * w, size)
    ess = lambda a, b: np.divide(a ** 2, b, out=np.zeros_like(a), where=b > 0)  # noqa: E731
    per_bin = ess(s1, s2).reshape(-1, nb)
    per_point = ess(s1.reshape(-1, nb).sum(1), s2.reshape(-1, nb).sum(1))
    return {"n": int(w.size), "ess": float(w.sum() ** 2 / np.sum(w * w)), "unit": bool(np.all(w == 1)),
            "range": [float(w.min()), float(w.max())],
            "per_point_median": float(np.median(per_point)), "edges": list(edges),
            "per_point_bin_median": np.median(per_bin, 0).tolist(),
            "per_point_bin_p10": np.percentile(per_bin, 10, 0).tolist()}


def context_for_layout(data, layout):
    """Known-context columns for ``layout.known``, in the layout's own column order.

    n_constituents comes from the shard's mask sum; any other known column (currently only
    ``jet_pt``, logged per the multiplicity-module docstring: "recommended with log(pT/GeV)
    scaling, centered near 0" -- here just logged, since v3 applies no further standardisation)
    is read from ``global``/``global_names``. One helper for both the flow's layout and the
    multiplicity model's layout, so a backend's known-context columns are plumbed in exactly
    one place rather than once per caller.
    """
    if not layout.known:
        return None
    cols = []
    for name in layout.known:
        if name == MULTIPLICITY_COLUMN:
            cols.append(data["n_constituents"].astype(np.float64))
        elif name == "jet_pt":
            idx = list(data["global_names"]).index("jet_pt")
            cols.append(np.log(data["global"][:, idx].astype(np.float64)))
        else:
            raise ValueError(f"no known source for context column {name!r}")
    return np.column_stack(cols)


def joint_strata(n_constituents, t_ref, n_bins=8, t_bins=5):
    """Joint (multiplicity x T_ref) stratum index per event, by quantile bins of each.

    Mitigation for the P4 finding that the surrogate's beyond-R=0.4 deficit tracks T_ref
    directly, root-caused to genuine statistical scarcity of real examples at high
    multiplicity (root_cause.html) -- stratifying by multiplicity alone would miss the
    T_ref-specific component found later (FINDINGS_JEWEL_V3.md). Quantile bins (not fixed-
    width) so every stratum gets a comparable share of events regardless of how skewed the
    underlying distribution is -- a fixed-width high-N bin would otherwise be the empty
    stratum `stratified_weights` refuses.
    """
    n_constituents = np.asarray(n_constituents)
    t_ref = np.asarray(t_ref)
    n_edges = np.quantile(n_constituents, np.linspace(0, 1, n_bins + 1)[1:-1])
    t_edges = np.quantile(t_ref, np.linspace(0, 1, t_bins + 1)[1:-1])
    n_bin = np.digitize(n_constituents, n_edges)
    t_bin = np.digitize(t_ref, t_edges)
    return n_bin * t_bins + t_bin


def prepare(data, transform, layout, chunk=200_000, weights=False, strata=None):
    """Model-coordinate tensors for ``train_flow``: u, mask, cond (and w if ``weights``).

    The transform runs in float64 on chunks of ``chunk`` events into a preallocated
    float32 array, so host memory stays near the size of the data (6 M jets × 81 slots
    would otherwise need several 12 GB float64 temporaries).

    ``strata``: per-event stratum index (e.g. from ``joint_strata``); when given, also
    computes ``sample_w`` via ``flow.stratified_weights`` for ``weight_mode='stratified'``.
    Requires ``weights=True`` (strata boosts the physical weight, not a substitute for it).
    """
    mask = torch.as_tensor(data["mask"], dtype=torch.bool)
    u = torch.empty(data["x"].shape, dtype=torch.float32)
    for s in range(0, len(u), chunk):
        x = torch.as_tensor(np.asarray(data["x"][s:s + chunk]), dtype=torch.float64)
        u[s:s + chunk] = transform.forward(x, mask[s:s + chunk])[0].float()
    cond = layout.assemble(data["theta"], list(PARAMETERS),
                           context=context_for_layout(data, layout),
                           context_names=layout.known)
    out = {"u": u, "mask": mask, "cond": torch.from_numpy(cond)}
    if weights:
        out["w"] = torch.as_tensor(np.asarray(data["weight"]), dtype=torch.float32)
    if strata is not None:
        if not weights:
            raise ValueError("strata needs weights=True (it boosts the physical weight)")
        from omnilearned.sbi.flow import stratified_weights

        sample_w, _ = stratified_weights(np.asarray(data["weight"]), strata)
        out["sample_w"] = torch.as_tensor(sample_w, dtype=torch.float32)
    return out


def sample_setup(config):
    """(backend id, selection, box) of a toy or JEWEL sample configuration.

    JEWEL configs (``configs/jewel-*.toml``) carry the parameter box under
    ``[design].box`` and a toy-v2 style ``[selection]``.
    """
    try:
        import tomllib
    except ModuleNotFoundError:  # Python < 3.11
        import tomli as tomllib
    with open(config, "rb") as f:
        backend = tomllib.load(f).get("backend", "")
    if backend.startswith("jewel"):
        from omnilearned.sbi.jewel import cards, jets

        cfg = cards.load_config(config)
        box = {p: tuple(cfg["design"]["box"][p]) for p in PARAMETERS}
        return backend, jets.selection_from_config(cfg), box
    _, _, sel = load_toy_config(config)
    return backend, sel, BOX


def shards_for_config(config):
    """(shard_paths, read_shards) functions matching ``config``'s backend.

    One place to decide this, so a caller (gate, closure, ...) does not have to repeat the
    "-v3" backend check that main() already makes. A backend ending "-v3" reads merged files
    through ``jewel.v3format`` (deta/dphi relative to the jet axis, log E, jet_pt known
    context); anything else reads the historical toy/v2 shard schema via ``toy.datasets``.
    """
    backend, _, _ = sample_setup(config)
    if backend.endswith("-v3"):
        from omnilearned.sbi.jewel.v3format import read_v3_shards, shard_paths_v3
        return shard_paths_v3, read_v3_shards
    return shard_paths, read_shards


def build_surrogate(args, train, val, out, log=print):
    backend, sel, box = sample_setup(args.config)
    is_v3 = backend.endswith("-v3")
    lo, hi = box_arrays(box)
    if is_v3:
        # docs/FORMAT_JEWEL_V3.md: the paper's minimal feature set is already centred with
        # O(1) variance, so no affine standardisation is fit -- an identity transform keeps
        # FlowSurrogate/train_flow unchanged while matching that no-op exactly.
        features = FEATURES_V3
        transform = FeatureTransform(features, [None] * len(features), [0.0] * len(features),
                                     [1.0] * len(features))
        layout = layout_with_multiplicity(list(PARAMETERS), lo, hi, known_context={"jet_pt": None})
        mlayout = layout_for_multiplicity_model(list(PARAMETERS), lo, hi, known_context={"jet_pt": None})
    else:
        features = FEATURES
        transform = FeatureTransform.fit(
            features, [sel.constituent_abs_eta_max, sel.constituent_abs_phi_max_rad, None],
            train["x"], train["mask"], chunk=200_000)
        layout = layout_with_multiplicity(list(PARAMETERS), lo, hi)
        mlayout = ConditionLayout.from_bounds(list(PARAMETERS), lo, hi)
    base = GaussianBase.standard(len(features))
    dev = resolve_device(args.device)

    t0 = time.time()
    mult = MultiplicityModel(mlayout.dim, sel.max_constituents, hidden=args.mult_hidden,
                             depth=args.mult_depth)
    mult.layout = mlayout
    mcond = lambda d: torch.from_numpy(mlayout.assemble(  # noqa: E731
        d["theta"], list(PARAMETERS), context=context_for_layout(d, mlayout), context_names=mlayout.known))
    weighted = getattr(args, "weight_mode", "none") != "none"
    stratified = getattr(args, "weight_mode", "none") == "stratified"
    t_ref_idx = list(PARAMETERS).index("T_ref")
    mult, mhist = train_multiplicity(mult, mcond(train), train["n_constituents"], mcond(val),
                                     val["n_constituents"], steps=args.mult_steps,
                                     device=dev.device, log=log,
                                     weights=train["weight"] if weighted else None,
                                     val_weights=val["weight"] if weighted else None)
    mhist["weighted"] = weighted
    mhist["seconds"] = time.time() - t0

    if args.arch == "set":
        model = SetFlowVelocity(len(features), layout.dim, dim=args.dim, depth=args.depth,
                                heads=args.heads, cond_embed=True, cond_to_particles=True)
    else:
        model = SmoothPETVelocity(len(features), layout.dim, dim=args.dim, depth=args.depth,
                                  heads=args.heads, local_dim=args.local_dim)
    opts = TrainOptions(**{f.name: getattr(args, f.name) for f in fields(TrainOptions)
                           if hasattr(args, f.name)})
    train_strata = (joint_strata(train["n_constituents"], train["theta"][:, t_ref_idx])
                   if stratified else None)
    tr = prepare(train, transform, layout, weights=weighted, strata=train_strata)
    va = prepare(val, transform, layout, weights=weighted)  # validation is never stratified
    train.pop("x", None)  # the float32 model coordinates replace the physical ones
    val.pop("x", None)
    meta = {"backend": backend, "arch": args.arch, "parameters": list(PARAMETERS), "box": box,
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
    resume_path = out / "resume.pt"
    resume_from = resume_path if getattr(args, "resume", False) and resume_path.exists() else None
    best, hist = train_flow(model, base, tr, va, opts, log=log, on_best=save,
                            resume_from=resume_from, resume_path=resume_path,
                            stop_after_step=getattr(args, "stop_after_step", None))
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
    ap.add_argument("--config", default=str(V2_CONFIG),
                    help="sample configuration: toy (configs/toy-*.toml) or JEWEL (configs/jewel-*.toml)")
    ap.add_argument("--val-dir", default=None,
                    help="validation shards from this directory (default: --data-dir)")
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
    ap.add_argument("--weight-mode", dest="weight_mode", default="none",
                    choices=["none", "loss", "sample", "stratified"],
                    help="per-jet sampling weights of the shards: weighted loss, minibatches "
                         "drawn proportional to the weight (both target the physical law "
                         "unchanged), or 'stratified' -- minibatches drawn to flatten the "
                         "joint (multiplicity, T_ref) distribution (more draws of the rare, "
                         "real tail the P4 halo-deficit finding traced to), loss-corrected "
                         "(flow.stratified_weights) so the trained objective still targets "
                         "the physical law despite the boosted draw -- see FINDINGS_JEWEL_V3.md")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tf32", action="store_true", help="TF32 matmuls during training only")
    ap.add_argument("--mult-steps", type=int, default=2000)
    ap.add_argument("--mult-hidden", type=int, default=128)
    ap.add_argument("--mult-depth", type=int, default=3)
    ap.add_argument("--save-after", type=int, default=0,
                    help="first step at which an improving checkpoint is written")
    ap.add_argument("--resume", action="store_true",
                    help="continue from --out/resume.pt if it exists (same target --steps; "
                         "a fresh run always writes one at every validation point, so a "
                         "second invocation with the same --out and this flag picks up where "
                         "a killed one left off, not just from its last best-val checkpoint)")
    ap.add_argument("--stop-after-step", dest="stop_after_step", type=int, default=None,
                    help="stop cleanly after this step (run.json/checkpoint still written) "
                         "instead of running to --steps -- for ending before a walltime "
                         "limit instead of relying on SIGKILL mid-step; --resume continues "
                         "from here to the real --steps in a later invocation")
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
    paths, reader = shards_for_config(args.config)
    tp, vp = paths(args.data_dir, "train"), paths(args.val_dir or args.data_dir, "validation")
    train, val = reader(tp), reader(vp)
    log(f"data: train {train['x'].shape} from {len(tp)} shards, val {val['x'].shape} "
        f"from {len(vp)} shards, {time.time() - start:.1f}s")
    data = {"train": [{"file": str(p), "sha256": file_sha256(p)} for p in tp],
            "validation": [{"file": str(p), "sha256": file_sha256(p)} for p in vp],
            "n_train": int(train["x"].shape[0]), "n_val": int(val["x"].shape[0]),
            "train_points": int(np.unique(train["point_id"]).size),
            "val_points": int(np.unique(val["point_id"]).size),
            "weights": {"train": weight_summary(train), "validation": weight_summary(val)}}
    for split, ws in data["weights"].items():
        log(f"weights {split}: ESS {ws['ess']:.0f} of {ws['n']} (unit={ws['unit']}), per point "
            f"{ws['per_point_median']:.0f}, per point and pT bin {np.round(ws['per_point_bin_median']).tolist()}")
    if args.weight_mode == "none" and not data["weights"]["train"]["unit"]:
        log("WARNING: the training shards carry non-unit weights but --weight-mode is none")
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
