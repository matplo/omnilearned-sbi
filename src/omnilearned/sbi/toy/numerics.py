"""M4 numerics check: Δlog L between nearby candidates with the fit's solver against a reference.

Plan rule 6 and lesson 3: the log-likelihood difference between nearby candidates must
converge in solver steps and divergence probes at the N used. For pairs (θ0, θ1) of nearby
candidates, the summed Δlog L = log L(θ1) − log L(θ0) over the observed jets is computed with
the test options (the closure's, e.g. midpoint-64 with 8 per-step probes) and with reference
options (e.g. rk4-64 with 16 per-step probes), each over several probe seeds. Criteria, each
< 0.5 (in log L units) for every pair:

* single-seed shift: |Δ_test(first seed) − mean Δ_ref|;
* seed-averaged shift: |mean Δ_test − mean Δ_ref|;
* probe std: the standard deviation of Δ_test over seeds.

The selection normalization Pr_q(S|θ) is a table independent of the solver and cancels in
the comparison; it is not included.

``--dtype float64`` runs the integration (both test and reference options) in float64
instead of the default float32 -- the 2026-11-05 P5 follow-up's "never tested" question:
whether the midpoint-vs-rk4 gap is a float32 discretization-floor effect that float64
resolves, independent of nsteps.

    python -m omnilearned.sbi.toy.numerics run --checkpoint CKPT --observations OBS.h5 \\
        --pair b0 T0 tau0 b1 T1 tau1 [--pair ...] --seeds 0 1 2 --out part0.json
    python -m omnilearned.sbi.toy.numerics merge --parts part*.json --out numerics.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from omnilearned.sbi.toy.simulator import PARAMETERS

TEST = {"solver": "midpoint", "nsteps": 64, "divergence": "hutchinson", "num_probes": 8,
        "probe": "rademacher", "probe_schedule": "per_step"}
REFERENCE = {"solver": "rk4", "nsteps": 64, "divergence": "hutchinson", "num_probes": 16,
             "probe": "rademacher", "probe_schedule": "per_step"}
LIMIT = 0.5


def _dtype(name):
    """torch dtype from a "float32"/"float64" string; mirrors ``fit.py``'s ``_dtype``.

    Not a ``LikelihoodOptions`` field (that dataclass is passed through
    ``LikelihoodOptions.from_dict``, which silently drops unknown keys) -- kept as an
    explicit argument here too, so a typo raises instead of being ignored.
    """
    import torch

    if name not in ("float32", "float64"):
        raise ValueError(f"dtype must be 'float32' or 'float64', got {name!r}")
    return torch.float64 if name == "float64" else torch.float32


def summed_logl(surrogate, obs, candidates, options, event_chunk=256, names=PARAMETERS,
               dtype="float32"):
    """Σ over events of log p for each candidate row; [n_candidates].

    Forwards ``obs.context`` (e.g. jet_pt) when present -- the v3 backend's flow has
    jet_pt as known context, and ``ConditionLayout.assemble`` raises if a known-context
    layout is asked to assemble without it (see FINDINGS_JEWEL_V3.md, P5). ``dtype``: see
    the P5 follow-up's "never tested" note -- whether running the integration in float64
    changes the test-vs-reference solver gap, same as ``fit.py``'s own dtype threading."""
    total = np.zeros(len(candidates))
    for e0 in range(0, len(obs), event_chunk):
        sl = slice(e0, e0 + event_chunk)
        ctx = obs.context[sl] if obs.context is not None else None
        lp, _ = surrogate.log_prob(obs.x[sl], obs.mask[sl], candidates, obs.event_id[sl], options,
                                   candidate_names=list(names), context=ctx, dtype=_dtype(dtype))
        total += np.asarray(lp, np.float64).sum(1)
    return total


def deltas(surrogate, obs, pairs, options_dict, seeds, event_chunk=256, log=print,
          names=PARAMETERS, dtype="float32"):
    """{seed: [Δlog L per pair]} with ``options_dict`` and probe_seed = seed."""
    from omnilearned.sbi.likelihood import LikelihoodOptions

    pairs = np.asarray(pairs, np.float64).reshape(-1, 2, len(names))
    cand, index = np.unique(pairs.reshape(-1, len(names)), axis=0, return_inverse=True)
    index = np.asarray(index).reshape(-1, 2)
    out = {}
    for s in seeds:
        t0 = time.time()
        opts = LikelihoodOptions.from_dict({**options_dict, "probe_seed": int(s)})
        ll = summed_logl(surrogate, obs, cand, opts, event_chunk, names, dtype)
        out[int(s)] = (ll[index[:, 1]] - ll[index[:, 0]]).tolist()
        log(f"{options_dict['solver']}-{options_dict['nsteps']} probes {options_dict.get('num_probes')} "
            f"seed {s}: {np.round(out[int(s)], 3).tolist()} ({time.time() - t0:.0f}s)")
    return out


def evaluate(test, reference, limit=LIMIT):
    """Criteria from {seed: [Δ per pair]} of the test and reference options."""
    ts = sorted(test)
    t = np.array([test[s] for s in ts])
    r = np.array([reference[s] for s in sorted(reference)])
    ref_mean = r.mean(0)
    single = np.abs(t[0] - ref_mean)
    avg = np.abs(t.mean(0) - ref_mean)
    std = t.std(0, ddof=1) if len(t) > 1 else np.full(t.shape[1], np.nan)
    ref_se = r.std(0, ddof=1) / np.sqrt(len(r)) if len(r) > 1 else np.full(r.shape[1], np.nan)
    ok = bool(np.all(single < limit) and np.all(avg < limit) and np.all(std < limit))
    return {"pass": ok, "limit": limit, "single_seed_shift": single.tolist(), "seed_averaged_shift": avg.tolist(),
            "probe_std": std.tolist(), "reference_mean": ref_mean.tolist(), "reference_se": ref_se.tolist(),
            "test_mean": t.mean(0).tolist(), "n_test_seeds": len(t), "n_reference_seeds": len(r)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--checkpoint", required=True)
    r.add_argument("--observations", required=True)
    r.add_argument("--pair", type=float, nargs=6, action="append", required=True,
                   metavar=("b0", "T0", "tau0", "b1", "T1", "tau1"))
    r.add_argument("--seeds", type=int, nargs="+", required=True)
    r.add_argument("--which", choices=["test", "reference", "both"], default="both")
    r.add_argument("--test", default=json.dumps(TEST), help="JSON likelihood options under test")
    r.add_argument("--reference", default=json.dumps(REFERENCE), help="JSON reference options")
    r.add_argument("--event-chunk", type=int, default=256)
    r.add_argument("--dtype", choices=["float32", "float64"], default="float32",
                   help="integration dtype for both test and reference options -- the P5 "
                   "follow-up's 'never tested' question")
    r.add_argument("--device", default="auto")
    r.add_argument("--out", required=True)
    m = sub.add_parser("merge")
    m.add_argument("--parts", nargs="+", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--limit", type=float, default=LIMIT)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        from omnilearned.sbi.device import resolve_device
        from omnilearned.sbi.flow import FlowSurrogate
        from omnilearned.sbi.observations import read_observations
        from omnilearned.sbi.toy.datasets import host_info

        dev = resolve_device(a.device)
        sur = FlowSurrogate.load(a.checkpoint, device=dev.device)
        obs = read_observations(a.observations)
        order = np.argsort(obs.mask.sum(1), kind="stable")  # as the fits: chunks sorted by N
        obs = obs.subset(order)
        test, ref = json.loads(a.test), json.loads(a.reference)
        doc = {"checkpoint": a.checkpoint, "checkpoint_sha256": sur.metadata.get("checkpoint_sha256"),
               "observations": a.observations, "observations_sha256": obs.sha256, "n_events": len(obs),
               "pairs": a.pair, "test_options": test, "reference_options": ref, "dtype": a.dtype,
               "device": dev.to_dict(), "host": host_info()}
        if a.which in ("test", "both"):
            doc["test"] = deltas(sur, obs, a.pair, test, a.seeds, a.event_chunk, dtype=a.dtype)
        if a.which in ("reference", "both"):
            doc["reference"] = deltas(sur, obs, a.pair, ref, a.seeds, a.event_chunk, dtype=a.dtype)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(doc, indent=1, default=float))
        return
    parts = [json.loads(Path(p).read_text()) for p in a.parts]
    for p in parts:
        p.setdefault("dtype", "float32")  # parts from before --dtype existed were float32
    keys = ("checkpoint_sha256", "observations_sha256", "pairs", "test_options", "reference_options",
           "dtype")
    for p in parts[1:]:
        for k in keys:
            if p[k] != parts[0][k]:
                raise ValueError(f"parts differ in {k}")
    test = {int(s): v for p in parts for s, v in p.get("test", {}).items()}
    ref = {int(s): v for p in parts for s, v in p.get("reference", {}).items()}
    doc = {k: parts[0][k] for k in keys + ("checkpoint", "observations", "n_events")}
    doc.update({"parts": a.parts, "test": test, "reference": ref, "devices": [p["device"] for p in parts],
                "hosts": [p["host"] for p in parts], **evaluate(test, ref, a.limit)})
    Path(a.out).write_text(json.dumps(doc, indent=1, default=float))
    print(f"numerics {'PASS' if doc['pass'] else 'FAIL'}: single-seed {np.round(doc['single_seed_shift'], 3).tolist()}, "
          f"seed-averaged {np.round(doc['seed_averaged_shift'], 3).tolist()}, probe std {np.round(doc['probe_std'], 3).tolist()} "
          f"(limit {a.limit}; {doc['n_test_seeds']} test / {doc['n_reference_seeds']} reference seeds)")


if __name__ == "__main__":
    main()
