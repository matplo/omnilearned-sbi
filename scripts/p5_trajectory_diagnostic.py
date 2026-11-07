"""P5 follow-up: compare midpoint vs rk4 per-step divergence values along the SAME real
jet's trajectory, with exact (autograd) divergence -- no Hutchinson probe noise at all, so
any difference between the two solvers is purely the quadrature, not randomness.

Replicates the first part of FlowSurrogate.log_prob (trim, context, transform, assemble cond)
for ONE real jet and ONE candidate theta, then manually drives the same step loop
likelihood._fixed_step uses for "midpoint" and "rk4", logging every stage's divergence value
(d2 for midpoint; d1..d4 for rk4) instead of only the final accumulated sum.

Runs under ``likelihood.ieee_float32()`` (full-precision float32 matmuls, no TF32) matching
what the decorated ``log_prob`` already does -- confirmed this process's default matmul
precision is "highest" regardless (FlowSurrogate never imports omnilearned.train, which is
the only place TF32 gets turned on), so this is a belt-and-braces guard, not a fix to a bug
found in the original run.

``--dtype float64`` casts the network, base and every tensor to double precision throughout
(ConditionLayout.assemble forces float32 at its last line regardless of input, so the
condition row is rebuilt by hand here to stay in float64 end to end) -- tests whether the
wild, sign-flipping pairwise ratios in the first pass (float32, nsteps=64) are a
discretization-truncation effect (should persist in float64) or partly float32 rounding
noise in a catastrophic-cancellation setup (should shrink in float64).

    PYTHONPATH=$PWD/src henv -n omni-sbi -x python scripts/p5_trajectory_diagnostic.py \\
        --checkpoint CKPT --observations OBS.h5 --jet-index N --theta b T_ref tau_init \\
        --nsteps 64 --dtype float32 --out traj.json
"""

import argparse
import json

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--observations", required=True)
    ap.add_argument("--jet-index", type=int, nargs="+", default=None,
                    help="rows into the observations file; default picks the jet whose "
                         "multiplicity is closest to the sample median")
    ap.add_argument("--theta", type=float, nargs=3, required=True, metavar=("b", "T_ref", "tau_init"))
    ap.add_argument("--nsteps", type=int, default=64)
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from omnilearned.sbi.device import resolve_device
    from omnilearned.sbi.flow import FlowSurrogate
    from omnilearned.sbi.likelihood import (
        LikelihoodOptions, _rhs_factory, frozen_for_inference, ieee_float32,
    )
    from omnilearned.sbi.observations import read_observations
    from omnilearned.sbi.toy.simulator import PARAMETERS

    dtype = torch.float64 if a.dtype == "float64" else torch.float32
    dev = resolve_device(a.device)
    sur = FlowSurrogate.load(a.checkpoint, device=dev.device)
    if dtype == torch.float64:
        sur.model.double()
    obs = read_observations(a.observations)

    def assemble_f64(layout, candidates, candidate_names, context, context_names):
        """ConditionLayout.assemble, but kept in float64 (the library version forces
        float32 at its last line regardless of input dtype)."""
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

    indices = a.jet_index
    if indices is None:
        counts = obs.mask.sum(1)
        indices = [int(np.argmin(np.abs(counts - np.median(counts))))]

    docs = []
    for idx in indices:
        n_real = int(obs.mask[idx].sum())
        jet_pt = float(np.exp(obs.context[idx, 0]))
        print(f"jet index {idx}: {n_real} real constituents, jet_pt = {jet_pt:.1f} GeV")

        x_phys = obs.x[idx:idx + 1]
        mask = obs.mask[idx:idx + 1]
        context = obs.context[idx:idx + 1]

        # --- replicate FlowSurrogate.log_prob's setup for M=1 candidate, N=1 event ---
        x_phys, mask = sur._trim(x_phys, mask)
        context_row, context_names = sur._context(mask, context, None)
        x = torch.as_tensor(np.asarray(x_phys), dtype=torch.float32)
        mask_t = torch.as_tensor(np.asarray(mask), dtype=torch.bool)
        u, logj = sur.transform.forward(x.double(), mask_t)
        u = u.to(dtype)
        cond_np = assemble_f64(sur.layout, [a.theta], list(PARAMETERS), context_row, context_names)
        cond = torch.from_numpy(cond_np).to(sur.device, dtype=dtype)
        x0 = u.to(sur.device) * mask_t.to(sur.device).to(dtype).unsqueeze(-1)
        mask_d = mask_t.to(sur.device)
        mask_f = mask_d.to(x0.dtype).unsqueeze(-1)

        options = LikelihoodOptions(solver="midpoint", nsteps=a.nsteps, divergence="exact").validate()
        counter = [0]
        with ieee_float32(), frozen_for_inference(sur.model):
            rhs = _rhs_factory(sur.model, cond, mask_d, mask_f, options, probes=None, counter=counter)

            h = 1.0 / a.nsteps
            steps = []

            x_mid, x_rk4 = x0.clone(), x0.clone()
            acc_mid, acc_rk4 = 0.0, 0.0
            for i in range(a.nsteps):
                t = i * h
                # midpoint: one real divergence eval per step (its own quadrature point)
                v1, _ = rhs(t, x_mid, need_div=False)
                v2, d2 = rhs(t + 0.5 * h, x_mid + 0.5 * h * v1)
                x_mid = x_mid + h * v2
                acc_mid += h * float(d2[0])

                # rk4: four stage divergences per step, same step, independent trajectory state
                w1, e1 = rhs(t, x_rk4)
                w2, e2 = rhs(t + 0.5 * h, x_rk4 + 0.5 * h * w1)
                w3, e3 = rhs(t + 0.5 * h, x_rk4 + 0.5 * h * w2)
                w4, e4 = rhs(t + h, x_rk4 + h * w3)
                x_rk4 = x_rk4 + (h / 6.0) * (w1 + 2 * w2 + 2 * w3 + w4)
                acc_rk4 += (h / 6.0) * float(e1[0] + 2 * e2[0] + 2 * e3[0] + e4[0])

                steps.append({"t": t, "midpoint_d2": float(d2[0]),
                             "rk4_d1": float(e1[0]), "rk4_d2": float(e2[0]),
                             "rk4_d3": float(e3[0]), "rk4_d4": float(e4[0]),
                             "acc_midpoint": acc_mid, "acc_rk4": acc_rk4})

        doc = {"jet_index": idx, "n_real_constituents": n_real, "jet_pt": jet_pt, "theta": a.theta,
              "nsteps": a.nsteps, "nfe": counter[0], "steps": steps,
              "final_acc_midpoint": acc_mid, "final_acc_rk4": acc_rk4}
        docs.append(doc)
        print(f"  final accumulated delta: midpoint={acc_mid:.4f}  rk4={acc_rk4:.4f}  "
              f"ratio={acc_rk4 / acc_mid if acc_mid else float('nan'):.2f}")

    with open(a.out, "w") as f:
        json.dump(docs, f, indent=1)
    print(f"wrote {a.out} ({len(docs)} jets)")


if __name__ == "__main__":
    main()
