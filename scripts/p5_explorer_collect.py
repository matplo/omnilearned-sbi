"""Merge scripts/p5_explorer_ensembles.py's per-ensemble JSON files into one consolidated
document for the true-vs-inferred theta report (reports repo).

    PYTHONPATH=$PWD/src henv -n omni-sbi -x python scripts/p5_explorer_collect.py \\
        --dir $JEST_NLE_ROOT/runs/jewel_v3/p5_explorer/ensembles --out explorer_data.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from omnilearned.sbi.toy.simulator import PARAMETERS


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    records = [json.loads(p.read_text()) for p in sorted(Path(a.dir).glob("*.json"))
              if "_scan" not in p.stem]
    by_point = {}
    for r in records:
        by_point.setdefault(r["point_index"], {"point_id": r["point_id"],
                                                "theta_true": r["theta_true"],
                                                "ensembles": []})["ensembles"].append(r)
    missing = []
    points_out = []
    for pi in sorted(by_point):
        entry = by_point[pi]
        ensembles = sorted(entry["ensembles"], key=lambda e: e["ensemble_index"])
        if len(ensembles) < 3:
            missing.append((pi, [e["ensemble_index"] for e in ensembles]))
        mle = {p: [e["summary"]["mle"][p] for e in ensembles] for p in PARAMETERS}
        pmean = {p: [e["summary"]["posterior_mean"][p] for e in ensembles] for p in PARAMETERS}
        pstd = {p: [e["summary"]["posterior_std"][p] for e in ensembles] for p in PARAMETERS}
        points_out.append({
            "point_index": pi, "point_id": entry["point_id"], "theta_true": entry["theta_true"],
            "n_ensembles": len(ensembles),
            "mle_mean": {p: float(np.mean(mle[p])) for p in PARAMETERS},
            "mle_std": {p: float(np.std(mle[p], ddof=1)) if len(mle[p]) > 1 else 0.0
                       for p in PARAMETERS},
            "posterior_mean_mean": {p: float(np.mean(pmean[p])) for p in PARAMETERS},
            "posterior_std_mean": {p: float(np.mean(pstd[p])) for p in PARAMETERS},
            "per_ensemble": {
                "mle": mle, "posterior_mean": pmean, "posterior_std": pstd,
                "boundary": [e["summary"]["boundary"] for e in ensembles],
            },
        })
    if missing:
        print(f"WARNING: incomplete points (fewer than 3 ensembles): {missing}")
    doc = {"parameters": list(PARAMETERS), "points": points_out}
    Path(a.out).write_text(json.dumps(doc, indent=1))
    print(f"wrote {a.out}: {len(points_out)} points "
          f"({sum(p['n_ensembles'] for p in points_out)} ensembles total)")


if __name__ == "__main__":
    main()
