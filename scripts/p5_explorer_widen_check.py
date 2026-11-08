"""Diagnostic follow-up to Phase D: every point's T_ref MLE landed exactly at its
3x3x3 grid's edge (true + 0.1, the window's own half-width), with near-total posterior
mass there -- "still climbing at the edge of every window tested" rather than noise or the
known b-T_ref-tau_init ridge (which would pull in a theta-dependent direction, not a constant
absolute offset). This checks directly: does the real data's likelihood in T_ref actually
turn over somewhere, or does it keep climbing across most of the box?

For each of the 5 explorer points, b and tau_init are held fixed at their TRUE values (a 1D
profile, not a joint grid -- cheap: ~9 candidates vs. the original 27) while T_ref sweeps
most of the box. Reuses ensemble 0's same 250 jets, same likelihood settings
(nsteps=256/float64/8 Hutchinson probes) and the same Phase C acceptance table.

    PYTHONPATH=$PWD/src henv -n omni-sbi -x python scripts/p5_explorer_widen_check.py \\
        --checkpoint CKPT --config configs/jewel-2.6.0-cs-v3.toml \\
        --fidelity-dir $JEST_NLE_ROOT/data/jewel_v3/merged \\
        --acceptance-table $JEST_NLE_ROOT/runs/jewel_v3/p5_numerics/acceptance_5x5x5.npz \\
        --explorer-data $JEST_NLE_ROOT/runs/jewel_v3/p5_explorer/explorer_data.json \\
        --out-dir $JEST_NLE_ROOT/runs/jewel_v3/p5_explorer/widen_check
"""

import argparse
import json
from pathlib import Path

import numpy as np


N_JETS = 250
AXIS_N = 9


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--fidelity-dir", required=True)
    ap.add_argument("--acceptance-table", required=True)
    ap.add_argument("--explorer-data", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only", type=int, nargs="+", default=None, help="point indices 0..4")
    ap.add_argument("--ensemble", type=int, default=0, help="which 250-jet ensemble (0,1,2) "
                    "to use -- repeat with different ensembles to test whether a finding is "
                    "a consistent bias (same across independent jet samples) or sample noise")
    ap.add_argument("--axis", default="T_ref", choices=("b", "T_ref", "tau_init"),
                    help="which parameter to sweep (the other two stay fixed at truth)")
    ap.add_argument("--t-min", type=float, default=None, help="override the swept axis's "
                    "lower bound (default: ~2%% in from the box's own lower edge)")
    ap.add_argument("--t-max", type=float, default=None, help="override the swept axis's "
                    "upper bound (default: ~2%% in from the box's own upper edge)")
    ap.add_argument("--tag", default="", help="suffix for the output filename, so a custom "
                    "range/ensemble doesn't overwrite the default sweep's file")
    ap.add_argument("--weighted-resample", action="store_true", help="draw the N_JETS "
                    "observation jets by the fidelity sample's own per-jet weight instead of "
                    "slicing unweighted by file order -- the fidelity split is a weighted MC "
                    "sample (weight spans ~1e-4 to ~1e2), and file order is not representative "
                    "(the first N_JETS jets run ~2x the weighted-mean jet_pt at every point)")
    ap.add_argument("--resample-seed", type=int, default=0)
    a = ap.parse_args()

    from omnilearned.sbi.fit import run_fit
    from omnilearned.sbi.flow import FlowSurrogate
    from omnilearned.sbi.jewel import cards
    from omnilearned.sbi.observations import Observations
    from omnilearned.sbi.toy.gate import direct_by_point
    from omnilearned.sbi.toy.selection import TableLogNorm
    from omnilearned.sbi.toy.simulator import PARAMETERS

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_box = cards.load_config(a.config)["design"]["box"]
    t_lo, t_hi = cfg_box[a.axis]
    axis_pos = list(PARAMETERS).index(a.axis)

    explorer = json.loads(Path(a.explorer_data).read_text())
    direct = direct_by_point(a.fidelity_dir, a.config)
    log_norm_fn = TableLogNorm(a.acceptance_table)
    sur = FlowSurrogate.load(a.checkpoint, device=a.device)

    points = explorer["points"]
    indices = a.only if a.only is not None else range(len(points))
    for pi in indices:
        rec = points[pi]
        pid = rec["point_id"]
        theta_true = np.array(rec["theta_true"])
        x, mask, theta_direct, weight, jet_pt = direct[pid]
        np.testing.assert_allclose(theta_direct, theta_true, atol=1e-9)
        if a.weighted_resample:
            rng = np.random.default_rng(a.resample_seed + a.ensemble)
            p = weight / weight.sum()
            idx = rng.choice(len(weight), size=N_JETS, replace=True, p=p)
            x_sel, mask_sel, jet_pt_sel = x[idx], mask[idx], jet_pt[idx]
        else:
            sl = slice(a.ensemble * N_JETS, (a.ensemble + 1) * N_JETS)
            x_sel, mask_sel, jet_pt_sel = x[sl], mask[sl], jet_pt[sl]
        obs = Observations(
            x=x_sel, mask=mask_sel, event_id=np.arange(N_JETS, dtype=np.int64),
            context=np.log(jet_pt_sel)[:, None], context_names=("jet_pt",),
        )
        # wide sweep on the chosen axis, the other two fixed at truth (single-node axes);
        # small margin in from the box edges themselves so the sweep's own endpoints aren't
        # degenerate, unless --t-min/--t-max pushes the window further toward (or past) that
        margin = 0.02 * (t_hi - t_lo)
        t_min = a.t_min if a.t_min is not None else round(t_lo + margin, 6)
        t_max = a.t_max if a.t_max is not None else round(t_hi - margin, 6)
        grid = {name: {"min": float(theta_true[k]), "max": float(theta_true[k]), "n": 1}
               for k, name in enumerate(PARAMETERS)}
        grid[a.axis] = {"min": t_min, "max": t_max, "n": AXIS_N}
        suffix = a.tag or ("" if a.ensemble == 0 else f"_e{a.ensemble}")
        suffix += "_w" if a.weighted_resample else ""
        out_prefix = f"point{pi}_{a.axis}{suffix}"
        cfg = {
            "schema": "omnilearned-sbi-fit-v1", "checkpoint": a.checkpoint,
            "observations": "in_memory", "output_prefix": str(out_dir / out_prefix),
            "device": a.device, "mode": "fixed_vector", "grid": grid,
            "likelihood": {"solver": "midpoint", "nsteps": 256, "dtype": "float64",
                          "divergence": "hutchinson", "num_probes": 8,
                          "probe_schedule": "per_step", "probe_seed": 0},
            "scan": {"event_chunk": 64, "candidate_chunk": 3},
        }
        print(f"point {pi} (pid={pid}): theta_true={theta_true.tolist()}, "
              f"{a.axis} axis {grid[a.axis]['min']:.3f}..{grid[a.axis]['max']:.3f} (n={AXIS_N})")
        doc = run_fit(cfg, surrogate=sur, observations=obs, log_norm_fn=log_norm_fn)
        axis_vals = np.linspace(grid[a.axis]["min"], grid[a.axis]["max"], AXIS_N)
        loglik = np.load(f"{cfg['output_prefix']}.npz")["loglik"].reshape(-1)
        out_path = out_dir / f"{out_prefix}.json"
        out_path.write_text(json.dumps({
            "point_index": pi, "point_id": pid, "theta_true": theta_true.tolist(),
            "axis": a.axis, "axis_values": axis_vals.tolist(), "loglik": loglik.tolist(),
            "mle": doc["summary"]["mle"], "wall_seconds": doc["wall_seconds"],
        }, indent=1))
        peak = axis_vals[np.argmax(loglik)]
        on_edge = peak in (axis_vals[0], axis_vals[-1])
        print(f"  peak at {a.axis}={peak:.3f} (true={theta_true[axis_pos]:.3f}) "
              f"{'<<< STILL ON WIDE-GRID EDGE' if on_edge else '(turned over inside the window)'} "
              f"({doc['wall_seconds']:.0f}s) -> {out_path}")

    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
