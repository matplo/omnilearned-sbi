"""P5 ensemble-level follow-up: does the t=0 divergence spike survive averaging over a
realistic inference ensemble of jets, or is it jet-to-jet noise that cancels?

The earlier per-jet check (p5_spike_sensitivity.py, 2 jets) found that for an individual
N=25 jet, the spike region's own pairwise theta-sensitivity exceeds the full trajectory's
and flips sign relative to the bulk's. But real inference sums log-likelihood over an
ensemble of jets, not a single one -- a per-jet effect with (near) zero mean across jets
will cancel under that sum (law of large numbers); a per-jet effect with a non-zero mean
will not, and survives at the same rate as the real signal.

Processes ALL available jets in p5_obs.h5 through the same spike (first 5% of steps) /
bulk / full midpoint-divergence decomposition, at primary_test theta0 and the same 4
established nearby-theta pairs used throughout P5, batched in multiplicity-sorted chunks
(not one global batch) -- padding every jet to the GLOBAL max multiplicity makes the
model's per-jet attention cost scale as batch x width^2, which blew up a first attempt
(batch=1000, width=93) to >90 minutes with zero progress. Chunking by multiplicity keeps
each batch's padding close to its own jets' actual size.

Reports, per pair:
  - mean +- SE of Delta_spike, Delta_bulk, Delta_full across jets (a non-zero mean is a
    real ensemble-level bias; consistent with zero means it cancels)
  - the running Delta_spike_ensemble / Delta_full_ensemble ratio vs. cumulative ensemble
    size N=1..all, to see whether/how fast it converges.

    PYTHONPATH=$V3_REPO/src henv -n omni-sbi -x python p5_ensemble_spike_sensitivity.py [--n-jets N] [--chunk-size K]
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
OUT = "/global/cfs/cdirs/alice/ploskon/jest-nle/scratch/p5_ensemble_spike_sensitivity.json"

THETA0 = (4.5, 0.45, 0.65)
PAIRS = {
    "pair0_b": (0.22, 0.0, 0.0),
    "pair1_Tref": (0.0, 0.016, 0.0),
    "pair2_tau": (0.0, 0.0, 0.016),
    "pair3_diag": (0.22, 0.016, 0.016),
}
NSTEPS = 64
SPIKE_FRAC = 0.05


def process_chunk(sur, x_raw, mask_raw, context_raw, thetas, param_names, nsteps, cutoff, h):
    """Run spike/bulk/full midpoint-divergence decomposition for one multiplicity-homogeneous
    chunk of jets, padded only to THIS chunk's own max active width."""
    from omnilearned.sbi.likelihood import LikelihoodOptions, _rhs_factory, frozen_for_inference, ieee_float32

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

    acc_spike, acc_full = {}, {}
    for name, th in thetas.items():
        cand = np.tile(np.asarray(th, dtype=np.float64), (n, 1))
        cond_np = sur.layout.assemble(cand, param_names, context=context_row,
                                      context_names=context_names, dtype=np.float64)
        cond = torch.from_numpy(cond_np).to(sur.device, dtype=torch.float64)
        options = LikelihoodOptions(solver="midpoint", nsteps=nsteps, divergence="exact").validate()
        counter = [0]
        with ieee_float32(), frozen_for_inference(sur.model):
            rhs = _rhs_factory(sur.model, cond, mask_d, mask_f, options, probes=None, counter=counter)
            x = x0.clone()
            acc = torch.zeros(n, dtype=torch.float64, device=sur.device)
            acc_cut = None
            for i in range(nsteps):
                t = i * h
                v1, _ = rhs(t, x, need_div=False)
                v2, d2 = rhs(t + 0.5 * h, x + 0.5 * h * v1)
                x = x + h * v2
                acc = acc + h * d2
                if i + 1 == cutoff:
                    acc_cut = acc.clone()
        acc_spike[name] = acc_cut.cpu().numpy()
        acc_full[name] = acc.cpu().numpy()
    return acc_spike, acc_full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jets", type=int, default=None)
    ap.add_argument("--nsteps", type=int, default=NSTEPS)
    ap.add_argument("--chunk-size", type=int, default=50)
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
    order = np.argsort(n_real_all)  # multiplicity-sorted so chunk padding stays tight
    print(f"n_jets={n_jets} (of {obs.x.shape[0]} available), nsteps={nsteps}, "
          f"chunk_size={args.chunk_size}, mult range [{n_real_all.min()},{n_real_all.max()}]", flush=True)

    cutoff = max(1, int(round(SPIKE_FRAC * nsteps)))
    h = 1.0 / nsteps
    print(f"spike cutoff at step {cutoff} (t={cutoff * h:.4f})\n", flush=True)

    thetas = {"theta0": THETA0}
    for name, step in PAIRS.items():
        thetas[name] = tuple(a + b for a, b in zip(THETA0, step))
    param_names = list(PARAMETERS)

    acc_spike_full = {name: np.zeros(n_jets) for name in thetas}
    acc_full_full = {name: np.zeros(n_jets) for name in thetas}
    n_real_sorted = n_real_all[order]

    n_chunks = int(np.ceil(n_jets / args.chunk_size))
    t_start = time.time()
    for c in range(n_chunks):
        idx = order[c * args.chunk_size:(c + 1) * args.chunk_size]
        t0 = time.time()
        acc_s, acc_f = process_chunk(
            sur, obs.x[idx], obs.mask[idx], obs.context[idx], thetas, param_names,
            nsteps, cutoff, h,
        )
        for name in thetas:
            acc_spike_full[name][idx] = acc_s[name]
            acc_full_full[name][idx] = acc_f[name]
        mult_lo, mult_hi = int(n_real_all[idx].min()), int(n_real_all[idx].max())
        print(f"chunk {c+1}/{n_chunks} (n={len(idx)}, mult=[{mult_lo},{mult_hi}]) "
              f"done in {time.time()-t0:.1f}s (elapsed {time.time()-t_start:.0f}s)", flush=True)

    base_spike, base_full = acc_spike_full["theta0"], acc_full_full["theta0"]
    summary = {}
    print("\n" + "=" * 70, flush=True)
    print(f"Per-jet Delta (pair - theta0), mean +- SE over n_jets={n_jets}:\n", flush=True)
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
        print(f"{name}:", flush=True)
        print(f"  spike: mean={row['mean_spike']:+.5f} +- {row['se_spike']:.5f}  "
              f"(|mean|/SE = {abs(row['mean_spike'])/row['se_spike']:.2f})", flush=True)
        print(f"  bulk:  mean={row['mean_bulk']:+.5f} +- {row['se_bulk']:.5f}  "
              f"(|mean|/SE = {abs(row['mean_bulk'])/row['se_bulk']:.2f})", flush=True)
        print(f"  full:  mean={row['mean_full']:+.5f} +- {row['se_full']:.5f}  "
              f"(|mean|/SE = {abs(row['mean_full'])/row['se_full']:.2f})", flush=True)
        print(f"  ensemble sums (N={n_jets}): spike={row['ensemble_sum_spike']:+.3f}  "
              f"full={row['ensemble_sum_full']:+.3f}  ratio={row['ensemble_ratio_spike_over_full']}", flush=True)
        print(flush=True)

    rng = np.random.default_rng(0)
    checkpoints = sorted(set(
        [k for k in [1, 2, 5, 10, 20, 50, 100, 200, 500, n_jets, n_jets // 2] if 0 < k <= n_jets]
    ))
    convergence = {}
    print("Cumulative ensemble ratio Delta_spike_sum/Delta_full_sum vs. ensemble size N "
          "(mean over 20 random orderings):\n", flush=True)
    for name in PAIRS:
        d_spike_full = acc_spike_full[name] - base_spike
        d_full_full_ = acc_full_full[name] - base_full
        ratios_at_k = {k: [] for k in checkpoints}
        for _ in range(20):
            perm = rng.permutation(n_jets)
            cs = np.cumsum(d_spike_full[perm])
            cf = np.cumsum(d_full_full_[perm])
            for k in checkpoints:
                if cf[k - 1] != 0:
                    ratios_at_k[k].append(cs[k - 1] / cf[k - 1])
        convergence[name] = {
            str(k): {"mean_ratio": float(np.mean(v)) if v else None,
                     "std_ratio": float(np.std(v)) if v else None}
            for k, v in ratios_at_k.items()
        }
        line = "  ".join(f"N={k}:{convergence[name][str(k)]['mean_ratio']:+.3f}" for k in checkpoints
                         if convergence[name][str(k)]["mean_ratio"] is not None)
        print(f"{name}: {line}", flush=True)

    with open(OUT, "w") as f:
        json.dump({
            "n_jets": n_jets, "nsteps": nsteps, "cutoff_step": cutoff, "theta0": THETA0, "pairs": PAIRS,
            "n_real": n_real_all.tolist(), "summary": summary, "convergence": convergence,
            "per_jet": {name: {"delta_spike": (acc_spike_full[name] - base_spike).tolist(),
                               "delta_full": (acc_full_full[name] - base_full).tolist()}
                       for name in PAIRS},
        }, f, indent=1)
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
