"""P5 fix candidate (a): does a graded/non-uniform time grid (LikelihoodOptions.grid_power)
shrink or stabilize the t=0 divergence-spike ensemble bias found in
p5_ensemble_spike_sensitivity.py (n=1000, 4.5-19 sigma, non-cancelling)?

Reruns the same spike/bulk/full midpoint-divergence decomposition, at primary_test theta0
and the same 4 established nearby-theta pairs, for several grid_power values (1.0 = today's
uniform grid baseline, then >1 values that concentrate steps near t=0) -- all at the SAME
nsteps=64, i.e. the SAME cost as production today. Multiplicity-sorted chunking as in the
original script (padding every jet to a batch's global max width makes the model's per-jet
cost scale as batch x width^2 -- see FINDINGS_JEWEL_V3.md).

One correctness fix over the original script: the "spike window" is now defined by a *t-value*
cutoff (t < SPIKE_T_CUTOFF) rather than a step-index cutoff, since step index no longer tracks
t proportionally once the grid is graded (grid_power != 1.0). At grid_power=1.0 this reduces
to the same cutoff step the original script used (cutoff/nsteps = 0.05).

Reports, per grid_power and per pair: mean +- SE of Delta_spike, Delta_bulk, Delta_full across
jets, and the ensemble ratio Delta_spike_sum/Delta_full_sum -- compare across grid_power values
to see whether the spike's apparent contribution shrinks/stabilizes as resolution concentrates
there, the way a convergent quadrature should, rather than just changing arbitrarily.

    PYTHONPATH=$V3_REPO/src henv -n omni-sbi -x python scripts/p5_graded_grid_validation.py \\
        [--n-jets N] [--chunk-size K] [--grid-powers 1.0 2.0 3.0 4.0]
"""

import argparse
import json
import sys
import time

import numpy as np
import torch

V3_REPO = "/global/cfs/cdirs/alice/ploskon/henv-omni-sbi/omnilearned-sbi-v3"
CKPT = "/global/cfs/cdirs/alice/ploskon/jest-nle/runs/jewel_v3/p3_train/v3_set_d256_L6_400k_s0/checkpoint.pt"
OBS = "/global/cfs/cdirs/alice/ploskon/jest-nle/runs/jewel_v3/p5_numerics/p5_obs.h5"
OUT = "/global/cfs/cdirs/alice/ploskon/jest-nle/scratch/p5_graded_grid_validation.json"

THETA0 = (4.5, 0.45, 0.65)
PAIRS = {
    "pair0_b": (0.22, 0.0, 0.0),
    "pair1_Tref": (0.0, 0.016, 0.0),
    "pair2_tau": (0.0, 0.0, 0.016),
    "pair3_diag": (0.22, 0.016, 0.016),
}
NSTEPS = 64
SPIKE_T_CUTOFF = 0.05  # same fraction of [0,1] as the original step-index cutoff (3/64)


def process_chunk(sur, x_raw, mask_raw, context_raw, thetas, param_names, nsteps, grid_power):
    """Spike/bulk/full midpoint-divergence decomposition for one multiplicity-homogeneous
    chunk of jets, padded only to THIS chunk's own max active width, using a graded grid."""
    from omnilearned.sbi.likelihood import (
        LikelihoodOptions, _graded_edges, _rhs_factory, frozen_for_inference, ieee_float32,
    )

    n = x_raw.shape[0]
    x_phys, mask = sur._trim(x_raw, mask_raw)
    context_row, context_names = sur._context(mask, context_raw, None)
    x_t = torch.as_tensor(np.asarray(x_phys), dtype=torch.float32)
    mask_t = torch.as_tensor(np.asarray(mask), dtype=torch.bool)
    u, _ = sur.transform.forward(x_t.double(), mask_t)
    u = u.to(torch.float64)
    mask_d = mask_t.to(sur.device)
    mask_f = mask_d.to(torch.float64).unsqueeze(-1)
    x0 = u.to(sur.device) * mask_f.squeeze(-1).unsqueeze(-1)

    edges = _graded_edges(nsteps, grid_power)
    cutoff_idx = int(np.searchsorted(edges, SPIKE_T_CUTOFF))  # first edge >= cutoff t

    acc_spike, acc_full = {}, {}
    for name, th in thetas.items():
        cand = np.tile(np.asarray(th, dtype=np.float64), (n, 1))
        cond_np = sur.layout.assemble(cand, param_names, context=context_row,
                                      context_names=context_names, dtype=np.float64)
        cond = torch.from_numpy(cond_np).to(sur.device, dtype=torch.float64)
        options = LikelihoodOptions(solver="midpoint", nsteps=nsteps, divergence="exact",
                                    grid_power=grid_power).validate()
        counter = [0]
        with ieee_float32(), frozen_for_inference(sur.model):
            rhs = _rhs_factory(sur.model, cond, mask_d, mask_f, options, probes=None, counter=counter)
            x = x0.clone()
            acc = torch.zeros(n, dtype=torch.float64, device=sur.device)
            acc_cut = None
            for i in range(nsteps):
                t = float(edges[i])
                h = float(edges[i + 1] - edges[i])
                v1, _ = rhs(t, x, need_div=False)
                v2, d2 = rhs(t + 0.5 * h, x + 0.5 * h * v1)
                x = x + h * v2
                acc = acc + h * d2
                if i + 1 == cutoff_idx:
                    acc_cut = acc.clone()
            if acc_cut is None:  # cutoff_idx == 0 edge case (shouldn't happen for SPIKE_T_CUTOFF=0.05)
                acc_cut = torch.zeros_like(acc)
        acc_spike[name] = acc_cut.cpu().numpy()
        acc_full[name] = acc.cpu().numpy()
    return acc_spike, acc_full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jets", type=int, default=None)
    ap.add_argument("--nsteps", type=int, default=NSTEPS)
    ap.add_argument("--chunk-size", type=int, default=50)
    ap.add_argument("--grid-powers", type=float, nargs="+", default=[1.0, 2.0, 3.0, 4.0])
    args = ap.parse_args()

    sys.path.insert(0, f"{V3_REPO}/src")
    from omnilearned.sbi.device import resolve_device
    from omnilearned.sbi.flow import FlowSurrogate
    from omnilearned.sbi.observations import read_observations
    from omnilearned.sbi.toy.simulator import PARAMETERS

    nsteps = args.nsteps
    dev = resolve_device("cuda")
    sur = FlowSurrogate.load(CKPT, device=dev.device)
    sur.model.double()
    obs = read_observations(OBS)

    n_jets = obs.x.shape[0] if args.n_jets is None else min(args.n_jets, obs.x.shape[0])
    n_real_all = obs.mask[:n_jets].sum(axis=1).astype(int)
    order = np.argsort(n_real_all)
    print(f"n_jets={n_jets} (of {obs.x.shape[0]} available), nsteps={nsteps}, "
          f"grid_powers={args.grid_powers}, chunk_size={args.chunk_size}, "
          f"mult range [{n_real_all.min()},{n_real_all.max()}]", flush=True)

    thetas = {"theta0": THETA0}
    for name, step in PAIRS.items():
        thetas[name] = tuple(a + b for a, b in zip(THETA0, step))
    param_names = list(PARAMETERS)

    all_results = {}
    for grid_power in args.grid_powers:
        print(f"\n{'='*70}\ngrid_power={grid_power}\n{'='*70}", flush=True)
        acc_spike_full = {name: np.zeros(n_jets) for name in thetas}
        acc_full_full = {name: np.zeros(n_jets) for name in thetas}

        n_chunks = int(np.ceil(n_jets / args.chunk_size))
        t_start = time.time()
        for c in range(n_chunks):
            idx = order[c * args.chunk_size:(c + 1) * args.chunk_size]
            t0 = time.time()
            acc_s, acc_f = process_chunk(
                sur, obs.x[idx], obs.mask[idx], obs.context[idx], thetas, param_names,
                nsteps, grid_power,
            )
            for name in thetas:
                acc_spike_full[name][idx] = acc_s[name]
                acc_full_full[name][idx] = acc_f[name]
            mult_lo, mult_hi = int(n_real_all[idx].min()), int(n_real_all[idx].max())
            print(f"chunk {c+1}/{n_chunks} (n={len(idx)}, mult=[{mult_lo},{mult_hi}]) "
                  f"done in {time.time()-t0:.1f}s (elapsed {time.time()-t_start:.0f}s)", flush=True)

        base_spike, base_full = acc_spike_full["theta0"], acc_full_full["theta0"]
        summary = {}
        print(f"\nPer-jet Delta (pair - theta0), mean +- SE over n_jets={n_jets}, "
              f"grid_power={grid_power}:\n", flush=True)
        for name in PAIRS:
            d_spike = acc_spike_full[name] - base_spike
            d_bulk = (acc_full_full[name] - acc_spike_full[name]) - (base_full - base_spike)
            d_full = acc_full_full[name] - base_full
            se = lambda a: float(a.std(ddof=1) / np.sqrt(len(a)))
            row = {
                "mean_spike": float(d_spike.mean()), "se_spike": se(d_spike),
                "mean_bulk": float(d_bulk.mean()), "se_bulk": se(d_bulk),
                "mean_full": float(d_full.mean()), "se_full": se(d_full),
                "ensemble_sum_spike": float(d_spike.sum()), "ensemble_sum_full": float(d_full.sum()),
                "ensemble_ratio_spike_over_full": float(d_spike.sum() / d_full.sum()) if d_full.sum() else None,
            }
            summary[name] = row
            print(f"{name}: spike={row['mean_spike']:+.5f}+-{row['se_spike']:.5f}  "
                  f"bulk={row['mean_bulk']:+.5f}+-{row['se_bulk']:.5f}  "
                  f"full={row['mean_full']:+.5f}+-{row['se_full']:.5f}  "
                  f"ratio={row['ensemble_ratio_spike_over_full']}", flush=True)
        all_results[str(grid_power)] = summary

    print(f"\n{'='*70}\nSummary: ensemble ratio Delta_spike_sum/Delta_full_sum vs grid_power\n{'='*70}",
          flush=True)
    for name in PAIRS:
        line = "  ".join(f"gp={gp}:{all_results[str(gp)][name]['ensemble_ratio_spike_over_full']:+.3f}"
                         for gp in args.grid_powers
                         if all_results[str(gp)][name]['ensemble_ratio_spike_over_full'] is not None)
        print(f"{name}: {line}", flush=True)

    with open(OUT, "w") as f:
        json.dump({
            "n_jets": n_jets, "nsteps": nsteps, "grid_powers": args.grid_powers,
            "spike_t_cutoff": SPIKE_T_CUTOFF, "theta0": THETA0, "pairs": PAIRS,
            "n_real": n_real_all.tolist(), "results_by_grid_power": all_results,
        }, f, indent=1)
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
