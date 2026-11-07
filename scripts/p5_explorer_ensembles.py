"""Phase D of the true-vs-inferred theta explorer: for 5 real, withheld fidelity theta
points, fit 3 disjoint 250-jet ensembles each with a 3x3x3 candidate grid around the truth,
at nsteps=256/float64 (the setting the P5 follow-up found substantially more trustworthy
than production, though not fully converged for the highest-multiplicity jets -- see
FINDINGS_JEWEL_V3.md), using the real Pr_q(S|theta) acceptance table from Phase C.

Writes one JSON per ensemble immediately after it finishes (ensembles/<point>_<ens>.json),
so a walltime kill or crash partway through a multi-hour run never loses completed work --
scripts/p5_explorer_collect.py merges whatever exists afterward. theta_true is written only
to these per-ensemble files, never to the fit cfg dict passed to run_fit (check_fit_config
forbids truth-like keys there by design).

    PYTHONPATH=$PWD/src henv -n omni-sbi -x python scripts/p5_explorer_ensembles.py \\
        --checkpoint CKPT --config configs/jewel-2.6.0-cs-v3.toml \\
        --fidelity-dir $JEST_NLE_ROOT/data/jewel_v3/merged \\
        --acceptance-table $JEST_NLE_ROOT/runs/jewel_v3/p5_numerics/acceptance_5x5x5.npz \\
        --out-dir $JEST_NLE_ROOT/runs/jewel_v3/p5_explorer/ensembles
"""

import argparse
import json
from pathlib import Path

import numpy as np


N_POINTS = 5
N_ENSEMBLES = 3
N_JETS = 250
GRID_N = 3
GRID_HALF_WIDTH_FRAC = 0.25  # of the box's own half-range, per axis


def pick_theta_points(direct, lo, hi, n=N_POINTS):
    pids = sorted(direct)
    thetas = {pid: direct[pid][2] for pid in pids}
    order = sorted(pids, key=lambda p: thetas[p][0])
    idx = np.linspace(0, len(order) - 1, n).astype(int)
    return [order[i] for i in idx]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--fidelity-dir", required=True)
    ap.add_argument("--acceptance-table", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--nsteps", type=int, default=256)
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--num-probes", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only", type=int, nargs="+", default=None,
                    help="work-item indices (0..14) to run; default all 15")
    a = ap.parse_args()

    from omnilearned.sbi.fit import run_fit
    from omnilearned.sbi.flow import FlowSurrogate
    from omnilearned.sbi.jewel import cards
    from omnilearned.sbi.observations import Observations
    from omnilearned.sbi.toy.gate import direct_by_point
    from omnilearned.sbi.toy.selection import TableLogNorm
    from omnilearned.sbi.toy.simulator import PARAMETERS

    out_dir = Path(a.out_dir)
    (out_dir / "ensembles").mkdir(parents=True, exist_ok=True)

    cfg_box = cards.load_config(a.config)["design"]["box"]
    lo = np.array([cfg_box[p][0] for p in PARAMETERS])
    hi = np.array([cfg_box[p][1] for p in PARAMETERS])
    half_width = (hi - lo) / 2 * GRID_HALF_WIDTH_FRAC

    direct = direct_by_point(a.fidelity_dir, a.config)
    theta_pids = pick_theta_points(direct, lo, hi)
    print(f"theta points: {[(pid, direct[pid][2].tolist()) for pid in theta_pids]}")

    log_norm_fn = TableLogNorm(a.acceptance_table)
    sur = FlowSurrogate.load(a.checkpoint, device=a.device)

    work = [(pi, pid, ei) for pi, pid in enumerate(theta_pids) for ei in range(N_ENSEMBLES)]
    if a.only is not None:
        work = [work[i] for i in a.only]

    for pi, pid, ei in work:
        out_path = out_dir / "ensembles" / f"{pi}_{ei}.json"
        if out_path.exists():
            print(f"point {pi} ensemble {ei}: already done, skipping")
            continue
        x, mask, theta_true, weight, jet_pt = direct[pid]
        sl = slice(ei * N_JETS, (ei + 1) * N_JETS)
        if mask[sl].shape[0] < N_JETS:
            raise ValueError(f"point {pid} has only {mask.shape[0]} jets, "
                             f"need {(ei + 1) * N_JETS} for ensemble {ei}")
        obs = Observations(
            x=x[sl], mask=mask[sl], event_id=np.arange(N_JETS, dtype=np.int64),
            context=np.log(jet_pt[sl])[:, None], context_names=("jet_pt",),
        )
        grid_axes = {
            name: {"min": float(max(lo[k], theta_true[k] - half_width[k])),
                  "max": float(min(hi[k], theta_true[k] + half_width[k])), "n": GRID_N}
            for k, name in enumerate(PARAMETERS)
        }
        cfg = {
            "schema": "omnilearned-sbi-fit-v1",
            "checkpoint": a.checkpoint, "observations": "in_memory",
            "output_prefix": str(out_dir / "ensembles" / f"{pi}_{ei}_scan"),
            "device": a.device, "mode": "fixed_vector",
            "grid": grid_axes,
            "likelihood": {"solver": "midpoint", "nsteps": a.nsteps, "dtype": a.dtype,
                          "divergence": "hutchinson", "num_probes": a.num_probes,
                          "probe_schedule": "per_step", "probe_seed": a.seed},
            # Deliberately small: nsteps=256/float64/Hutchinson-per_step retains a much
            # bigger computation graph per row than anything run so far this session --
            # the acceptance-table OOM (Phase C) happened at a far cheaper generation-only
            # batch=4096. event_chunk x candidate_chunk here is only 9*64=576 combined rows,
            # vs. the naive 250*27=6750 an unchunked call would build at once.
            "scan": {"event_chunk": 64, "candidate_chunk": GRID_N},
        }
        print(f"point {pi} (pid={pid}, theta_true={theta_true.tolist()}) ensemble {ei}: fitting...")
        doc = run_fit(cfg, surrogate=sur, observations=obs, log_norm_fn=log_norm_fn)
        record = {"point_index": pi, "point_id": int(pid), "ensemble_index": ei,
                 "theta_true": theta_true.tolist(), "grid": grid_axes,
                 "summary": doc["summary"], "wall_seconds": doc["wall_seconds"]}
        out_path.write_text(json.dumps(record, indent=1))
        print(f"  mle={doc['summary']['mle']} std={doc['summary']['posterior_std']} "
              f"({doc['wall_seconds']:.0f}s) -> {out_path}")

    print(f"done: {len(work)} work items processed (or already present) under {out_dir}/ensembles")


if __name__ == "__main__":
    main()
