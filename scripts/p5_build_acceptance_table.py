"""Build Pr_q(S | theta) acceptance table for a v3 checkpoint (Phase C of the
true-vs-inferred-theta explorer plan).

There is no p(pT_J | theta) model yet, so sampling at an arbitrary grid node (which may
not coincide with a real fidelity theta point) needs a stand-in jet_pT pool. This script
uses the weighted jet_pT spectrum of the NEAREST real fidelity point (normalized Euclidean
distance in the box) as that pool -- documented as an approximation in the output metadata,
same convention build_explorer_data.py already used for nearest-PASS lookups.

    PYTHONPATH=$PWD/src henv -n omni-sbi -x python scripts/p5_build_acceptance_table.py \\
        --checkpoint CKPT --config configs/jewel-2.6.0-cs-v3.toml \\
        --fidelity-dir $JEST_NLE_ROOT/data/jewel_v3/merged \\
        --n-per-axis 6 --n-samples 4096 --nsteps 128 --out table.npz
"""

import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--fidelity-dir", required=True)
    ap.add_argument("--n-per-axis", type=int, default=6)
    ap.add_argument("--n-samples", type=int, default=4096)
    ap.add_argument("--nsteps", type=int, default=128, help="sampling steps (cheap -- no "
                    "divergence computed here), independent of the fit's own nsteps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from omnilearned.sbi.flow import FlowSurrogate
    from omnilearned.sbi.jewel import cards, jets
    from omnilearned.sbi.toy.gate import direct_by_point
    from omnilearned.sbi.toy.selection import acceptance_table, save_table
    from omnilearned.sbi.toy.simulator import PARAMETERS

    cfg = cards.load_config(a.config)
    box = cfg["design"]["box"]
    lo = np.array([box[p][0] for p in PARAMETERS])
    hi = np.array([box[p][1] for p in PARAMETERS])
    axes = [np.linspace(lo[k], hi[k], a.n_per_axis) for k in range(3)]

    sel = jets.selection_from_config(cfg)
    direct = direct_by_point(a.fidelity_dir, a.config)
    pids = sorted(direct)
    thetas = np.array([direct[pid][2] for pid in pids])
    thetas_norm = (thetas - lo) / (hi - lo)
    pools = {pid: (direct[pid][4], direct[pid][3]) for pid in pids}  # (jet_pt, weight)

    def pool_fn(theta):
        tn = (np.asarray(theta, np.float64) - lo) / (hi - lo)
        nearest = pids[int(np.argmin(np.linalg.norm(thetas_norm - tn, axis=1)))]
        return pools[nearest]

    sur = FlowSurrogate.load(a.checkpoint, device=a.device)
    print(f"grid: {a.n_per_axis}^3 = {a.n_per_axis ** 3} points, box b={list(box['b'])} "
          f"T_ref={list(box['T_ref'])} tau_init={list(box['tau_init'])}")
    acc, reasons = acceptance_table(sur, sel, axes, a.n_samples, a.seed, nsteps=a.nsteps,
                                    jet_pt_pool_fn=pool_fn)
    save_table(a.out, axes, acc, reasons, {
        "checkpoint": a.checkpoint, "checkpoint_sha256": sur.metadata.get("checkpoint_sha256"),
        "config": a.config, "n_samples": a.n_samples, "nsteps": a.nsteps, "seed": a.seed,
        "jet_pt_pool": "nearest real fidelity theta point (normalized Euclidean distance); "
                       "approximation, no p(pT_J|theta) model exists yet",
        "acceptance_range": [float(acc.min()), float(acc.max())],
    })
    print(f"acceptance {acc.shape}, range [{acc.min():.4f}, {acc.max():.4f}] -> {a.out}")


if __name__ == "__main__":
    main()
