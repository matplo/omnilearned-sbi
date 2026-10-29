"""Sample a toy-v2 surrogate at one setting, apply the toy selection, compare.

The surrogate draws N from p(N | theta), then the cloud from q(x | N, theta)
(IEEE float32 sampling), and the clouds pass ``select_clouds`` with the toy's
own v2 selection. Direct toy jets at the same setting come from the toy (not
from the generator). The output JSON holds the surrogate acceptance, the
rejection reasons and the fidelity gate (``fidelity.evaluate``) for this
one setting, with the source revision and device.

Settings are passed explicitly; drivers that use reserved settings must take
them from the development setting only (its truth is open by design).

    python -m omnilearned.sbi.toy.sample_select --checkpoint CKPT --theta 7.5 0.30 0.95 \
        --n 2000 --out OUT.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from omnilearned.sbi import fidelity
from omnilearned.sbi.device import resolve_device
from omnilearned.sbi.flow import FlowSurrogate, json_dumps
from omnilearned.sbi.toy.config import V2_CONFIG, load_toy_config, sample_setting
from omnilearned.sbi.toy.datasets import SPLIT_SEED, host_info
from omnilearned.sbi.toy.selection import select_clouds
from omnilearned.sbi.toy.simulator import PARAMETERS
from omnilearned.sbi.toy.simulator_v2 import REJECTION_REASONS


REDRAW_SEED_STRIDE = 1 << 24  # redraw round r uses seed + r * stride (gate seeds are seed + point)


def sample_and_select(surrogate, theta, n, sel, seed=0, nsteps=128, solver="midpoint",
                      batch=4096, sort_by_length=True, redraw=False, max_rounds=50):
    """Selected surrogate clouds at ``theta``: (x, mask, report).

    Default: N from p(N | theta), the cloud from q(x | N, theta), rejected clouds dropped,
    so the selected sample follows p(N) Pr_q(S | N) / Pr_q(S). With ``redraw`` (per-N
    selection normalization, fidelity plan S1) each rejected cloud is drawn again at the same
    N until it passes (at most ``max_rounds``), so the selected sample keeps p(N | theta) and
    follows q(x | N, theta) 1_S(x) / Pr_q(S | N, theta). The report's acceptance is always the
    first-pass Pr_q(S | theta); ``acceptance_by_n`` holds the first-pass counts per N.
    """
    t0 = time.time()
    theta = np.asarray(theta, np.float64)
    x, mask = surrogate.sample(theta, n, None, seed=seed, nsteps=nsteps,
                               solver=solver, candidate_names=list(PARAMETERS), batch=batch,
                               sort_by_length=sort_by_length)
    ok, reason, _ = select_clouds(x.astype(np.float64), sel, mask)
    counts = mask.sum(1)
    rep = {"generated": int(n), "selected": int(ok.sum()), "acceptance": float(ok.mean()),
           "rejections": {r: int((reason[~ok] == k).sum()) for k, r in enumerate(REJECTION_REASONS)},
           "nsteps": nsteps, "solver": solver, "seed": seed, "sort_by_length": sort_by_length,
           "acceptance_by_n": {"generated": np.bincount(counts).tolist(),
                               "accepted": np.bincount(counts[ok], minlength=counts.max(initial=0) + 1).tolist()}}
    if redraw:
        rounds, redrawn = 0, int((~ok).sum())
        while (~ok).any() and rounds < max_rounds:
            rounds += 1
            idx = np.flatnonzero(~ok)
            xr, mr = surrogate.sample(theta, idx.size, counts[idx], seed=seed + rounds * REDRAW_SEED_STRIDE,
                                      nsteps=nsteps, solver=solver, candidate_names=list(PARAMETERS),
                                      batch=batch, sort_by_length=sort_by_length, pad_to=mask.shape[1])
            okr, _, _ = select_clouds(xr.astype(np.float64), sel, mr)
            x[idx[okr]], mask[idx[okr]], ok[idx[okr]] = xr[okr], mr[okr], True
        rep["redraw"] = {"rounds": rounds, "redrawn": redrawn, "unfilled": int((~ok).sum()),
                         "max_rounds": max_rounds, "seed_stride": REDRAW_SEED_STRIDE}
        rep["selected"] = int(ok.sum())
    rep["sample_seconds"] = time.time() - t0
    return x[ok], mask[ok], rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--theta", type=float, nargs=3, required=True)
    ap.add_argument("--n", type=int, default=2000, help="generated surrogate clouds")
    ap.add_argument("--n-direct", type=int, default=None, help="direct toy jets (default: n)")
    ap.add_argument("--direct-stream", default="development", choices=sorted(SPLIT_SEED))
    ap.add_argument("--direct-index", type=int, default=0)
    ap.add_argument("--nsteps", type=int, default=128)
    ap.add_argument("--solver", default="midpoint")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--n-events", type=int, default=250, help="analysis N for the relevance test")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--config", default=str(V2_CONFIG))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    start = time.time()
    dev = resolve_device(a.device)
    sur = FlowSurrogate.load(a.checkpoint, device=dev.device)
    cfg, _, sel = load_toy_config(a.config)
    xs, ms, rep = sample_and_select(sur, a.theta, a.n, sel, a.seed, a.nsteps, a.solver)
    print(f"surrogate: {rep['selected']}/{rep['generated']} selected (acc {rep['acceptance']:.4f}), "
          f"{rep['sample_seconds']:.1f}s sampling", flush=True)
    seed = int(cfg["randomness"][SPLIT_SEED[a.direct_stream]])
    t0 = time.time()
    rec, man = sample_setting(a.theta, a.n_direct or rep["selected"], seed, stream=a.direct_stream,
                              index=a.direct_index, path=a.config)
    t_direct = time.time() - t0
    w = max(int(rec["n_constituents"].max()), ms.shape[1])
    pad = lambda v, w: np.pad(v, [(0, 0), (0, w - v.shape[1])] + [(0, 0)] * (v.ndim - 2))  # noqa: E731
    xd, md = rec["x"][:, :w], rec["mask"][:, :w]
    gate = fidelity.evaluate(xd, pad(xs, w), n_events=a.n_events, n_perm=a.n_perm, seed=a.seed,
                             direct_mask=md, surrogate_mask=pad(ms, w))
    obs_d = fidelity.all_obs(xd, md)
    obs_s = fidelity.all_obs(pad(xs, w), pad(ms, w))
    means = {k: {"direct": float(np.mean(obs_d[k])), "surrogate": float(np.mean(obs_s[k])),
                 "direct_std": float(np.std(obs_d[k]))} for k in fidelity.JET_VARIABLE}
    doc = {"checkpoint": a.checkpoint, "checkpoint_sha256": sur.metadata.get("checkpoint_sha256"),
           "theta": dict(zip(PARAMETERS, a.theta)), "surrogate": rep,
           "direct": {"toy": man, "stream": a.direct_stream, "index": a.direct_index,
                      "seconds": t_direct},
           "means": means, "gate": gate, "device": dev.to_dict(),
           "gpu_name": torch.cuda.get_device_name() if dev.device == "cuda" else None,
           "host": host_info(), "wall_seconds": time.time() - start}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        f.write(json_dumps(doc))
    v = gate["verdict"]
    print(f"gate: p_family {gate['permutation']['p_family']:.3g} significant={v['significant']} "
          f"relevant={v['relevant']} fail={v['fail']}")
    for k, m in means.items():
        print(f"  {k:16s} direct {m['direct']:.4f} surrogate {m['surrogate']:.4f} "
              f"(rel {gate['relevance'][k]['relevance']:.2f})")
    print(f"wall {doc['wall_seconds']:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
