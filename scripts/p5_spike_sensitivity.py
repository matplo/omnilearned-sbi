"""P5 follow-up: does the t=0 divergence spike's own contribution to the accumulated
integral depend on theta, or is it (mostly) a theta-independent offset that would safely
cancel in any pairwise Delta log-L a fit actually uses?

For each of 2 real jets, at theta0=primary_test and at theta0 shifted by each of the 4
established pairs (b/T_ref/tau_init/diagonal, same +-2%-of-range steps as the original
formal P5 check), accumulates midpoint's running divergence sum and records it at the
"spike cutoff" (first 5% of steps) separately from the full trajectory. Compares the
pairwise Delta restricted to the spike region against the real pairwise Delta (what a fit
uses) and against the bulk-only Delta (everything after the cutoff).

    PYTHONPATH=$V3_REPO/src henv -n omni-sbi -x python p5_spike_sensitivity.py
"""

import json

import numpy as np
import torch

V3_REPO = "/global/cfs/cdirs/alice/ploskon/henv-omni-sbi/omnilearned-sbi-v3"
CKPT = "/global/cfs/cdirs/alice/ploskon/jest-nle/runs/jewel_v3/p3_train/v3_set_d256_L6_400k_s0/checkpoint.pt"
OBS = "/global/cfs/cdirs/alice/ploskon/jest-nle/runs/jewel_v3/p5_numerics/p5_obs.h5"
THETA0 = (4.5, 0.45, 0.65)
PAIRS = {
    "pair0_b": (0.22, 0.0, 0.0),
    "pair1_Tref": (0.0, 0.016, 0.0),
    "pair2_tau": (0.0, 0.0, 0.016),
    "pair3_diag": (0.22, 0.016, 0.016),
}
JET_INDICES = [904, 26]
NSTEPS = 64
SPIKE_FRAC = 0.05


def assemble_f64(layout, candidates, candidate_names, context, context_names):
    candidates = np.atleast_2d(np.asarray(candidates, dtype=np.float64))
    out = np.zeros((candidates.shape[0], layout.dim), dtype=np.float64)
    for j, name in enumerate(candidate_names):
        col = layout.columns.index(name)
        out[:, col] = (candidates[:, j] - layout.center[name]) / layout.scale[name]
    if layout.known:
        ctx = np.atleast_2d(np.asarray(context, dtype=np.float64))
        cnames = list(context_names or layout.known)
        for name in layout.known:
            col = layout.columns.index(name)
            src = cnames.index(name)
            out[:, col] = (ctx[:, src] - layout.center[name]) / layout.scale[name]
    return out


def main():
    import sys
    sys.path.insert(0, f"{V3_REPO}/src")
    from omnilearned.sbi.device import resolve_device
    from omnilearned.sbi.flow import FlowSurrogate
    from omnilearned.sbi.likelihood import LikelihoodOptions, _rhs_factory, frozen_for_inference, ieee_float32
    from omnilearned.sbi.observations import read_observations
    from omnilearned.sbi.toy.simulator import PARAMETERS

    dev = resolve_device("cuda")
    sur = FlowSurrogate.load(CKPT, device=dev.device)
    sur.model.double()
    obs = read_observations(OBS)

    cutoff = max(1, int(round(SPIKE_FRAC * NSTEPS)))
    h = 1.0 / NSTEPS
    print(f"nsteps={NSTEPS}, spike cutoff at step {cutoff} (t={cutoff * h:.4f}, "
          f"{SPIKE_FRAC*100:.0f}% of trajectory)\n")

    results = {}
    for idx in JET_INDICES:
        n_real = int(obs.mask[idx].sum())
        x_phys, mask = sur._trim(obs.x[idx:idx + 1], obs.mask[idx:idx + 1])
        context_row, context_names = sur._context(mask, obs.context[idx:idx + 1], None)
        x_t = torch.as_tensor(np.asarray(x_phys), dtype=torch.float32)
        mask_t = torch.as_tensor(np.asarray(mask), dtype=torch.bool)
        u, _ = sur.transform.forward(x_t.double(), mask_t)
        u = u.to(torch.float64)
        x0 = u.to(sur.device) * mask_t.to(sur.device).to(torch.float64).unsqueeze(-1)
        mask_d = mask_t.to(sur.device)
        mask_f = mask_d.to(x0.dtype).unsqueeze(-1)

        thetas = {"theta0": THETA0}
        for name, step in PAIRS.items():
            thetas[name] = tuple(a + b for a, b in zip(THETA0, step))

        row = {}
        for name, th in thetas.items():
            cond_np = assemble_f64(sur.layout, [list(th)], list(PARAMETERS), context_row, context_names)
            cond = torch.from_numpy(cond_np).to(sur.device, dtype=torch.float64)
            options = LikelihoodOptions(solver="midpoint", nsteps=NSTEPS, divergence="exact").validate()
            counter = [0]
            with ieee_float32(), frozen_for_inference(sur.model):
                rhs = _rhs_factory(sur.model, cond, mask_d, mask_f, options, probes=None, counter=counter)
                x = x0.clone()
                acc = 0.0
                acc_at_cutoff = None
                for i in range(NSTEPS):
                    t = i * h
                    v1, _ = rhs(t, x, need_div=False)
                    v2, d2 = rhs(t + 0.5 * h, x + 0.5 * h * v1)
                    x = x + h * v2
                    acc += h * float(d2[0])
                    if i + 1 == cutoff:
                        acc_at_cutoff = acc
            row[name] = {"spike": acc_at_cutoff, "full": acc, "bulk": acc - acc_at_cutoff}
            print(f"jet {idx} theta={tuple(round(v, 4) for v in th)} ({name}): "
                  f"spike_sum={acc_at_cutoff:.4f}  full_sum={acc:.4f}  bulk_sum={acc - acc_at_cutoff:.4f}")
        results[idx] = {"n_real": n_real, "by_theta": row}
        print()

    print("=" * 70)
    print("Delta (pair - theta0), comparing spike-region-only vs bulk-only vs full:\n")
    summary = {}
    for idx, data in results.items():
        base = data["by_theta"]["theta0"]
        summary[idx] = {}
        for name in PAIRS:
            r = data["by_theta"][name]
            d_spike = r["spike"] - base["spike"]
            d_bulk = r["bulk"] - base["bulk"]
            d_full = r["full"] - base["full"]
            frac = abs(d_spike) / abs(d_full) if d_full else float("nan")
            summary[idx][name] = {"delta_spike": d_spike, "delta_bulk": d_bulk,
                                  "delta_full": d_full, "frac_of_full_from_spike": frac}
            print(f"jet {idx} {name}: Delta_spike={d_spike:+.4f}  Delta_bulk={d_bulk:+.4f}  "
                  f"Delta_full={d_full:+.4f}  (spike is {frac*100:.1f}% of |Delta_full|)")
        print()

    with open("/global/cfs/cdirs/alice/ploskon/jest-nle/scratch/p5_spike_sensitivity.json", "w") as f:
        json.dump({"nsteps": NSTEPS, "cutoff_step": cutoff, "theta0": THETA0, "pairs": PAIRS,
                   "results": results, "summary": summary}, f, indent=1)
    print("wrote p5_spike_sensitivity.json")


if __name__ == "__main__":
    main()
