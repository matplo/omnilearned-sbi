"""Mitigation (3) for the P4 halo-deficit/T_ref correlation finding (FINDINGS_JEWEL_V3.md):
40 supplementary training points concentrated at high T_ref, where real high-multiplicity
jets -- and so real examples of the beyond-R=0.4 halo -- are rarer to begin with
(root_cause.html's own estimate: ~127 effective weighted examples at N>160, out of 1.44M).

b and tau_init span their full existing box ranges (only T_ref is restricted), via a
scrambled-Sobol draw on a dedicated, reproducible stream
(design.STREAMS["jewel_train_hightref"]) so this is deterministic and auditable like every
other design point in the project, not an ad hoc random sample. Points within the reserved
holdout margin are dropped, same convention as make_design.

    PYTHONPATH=$PWD/src henv -n omni-sbi -x python scripts/p4_hightref_design.py \\
        --config configs/jewel-2.6.0-cs-v3.toml --n 40 --t-ref-min 0.6 --t-ref-max 0.9 \\
        --start-point-id 512 --out $JEST_NLE_ROOT/data/jewel_v3/p4_hightref_design.json
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import qmc


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--t-ref-min", type=float, required=True)
    ap.add_argument("--t-ref-max", type=float, required=True)
    ap.add_argument("--start-point-id", type=int, required=True,
                    help="must be > the design's own train_points (no collision)")
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from omnilearned.sbi.jewel import cards
    from omnilearned.sbi.toy.datasets import box_arrays, margin_clashes
    from omnilearned.sbi.toy.design import STREAMS, reserved_settings, rng_for
    from omnilearned.sbi.toy.simulator import PARAMETERS

    cfg = cards.load_config(a.config)
    box = {p: tuple(cfg["design"]["box"][p]) for p in PARAMETERS}
    train_points = int(cfg["design"]["train_points"])
    if a.start_point_id < train_points:
        raise ValueError(f"--start-point-id {a.start_point_id} must be >= "
                         f"design.train_points ({train_points}) to avoid NJOB collision "
                         f"(point_id 0..{train_points - 1} are already the original design)")
    t_lo, t_hi = box["T_ref"]
    if not (t_lo <= a.t_ref_min < a.t_ref_max <= t_hi):
        raise ValueError(f"T_ref window [{a.t_ref_min}, {a.t_ref_max}] must sit inside the "
                         f"box's own T_ref range {box['T_ref']}")

    seed = int(cfg["randomness"]["design_seed"])
    rng = rng_for(seed, STREAMS["jewel_train_hightref"])
    sob = qmc.Sobol(d=len(PARAMETERS), scramble=True, seed=rng)
    m = max(1, math.ceil(math.log2(max(2, int(2.0 * a.n) + 8))))  # extra headroom: the T_ref
    unit = sob.random_base2(m)                                    # restriction rejects most draws
    lo, hi = box_arrays(box)
    pts = lo + unit * (hi - lo)
    t_idx = list(PARAMETERS).index("T_ref")
    in_window = (pts[:, t_idx] >= a.t_ref_min) & (pts[:, t_idx] <= a.t_ref_max)
    clash = margin_clashes(pts, reserved_settings(cfg), box, a.margin)
    kept = np.flatnonzero(in_window & ~clash)
    if kept.size < a.n:
        raise ValueError(f"only {kept.size} of {len(pts)} draws fall in the T_ref window "
                         f"after the reservation margin -- need {a.n}; raise the Sobol headroom")
    chosen = pts[kept[:a.n]]
    point_ids = list(range(a.start_point_id, a.start_point_id + a.n))

    doc = {"schema": "jewel-v3-hightref-design-v1", "config": a.config,
          "backend": cfg["backend"], "parameters": list(PARAMETERS),
          "t_ref_window": [a.t_ref_min, a.t_ref_max], "margin": a.margin,
          "design_seed": seed, "stream": STREAMS["jewel_train_hightref"],
          "train_points_in_original_design": train_points,
          "point_ids": point_ids, "theta": chosen.tolist()}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(doc, indent=1))
    print(f"{a.n} points, point_id {point_ids[0]}-{point_ids[-1]}, "
          f"T_ref in [{chosen[:, t_idx].min():.3f}, {chosen[:, t_idx].max():.3f}] -> {a.out}")


if __name__ == "__main__":
    main()
