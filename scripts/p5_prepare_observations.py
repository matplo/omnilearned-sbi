"""P5 (NLE numerics, docs/PLAN_JEWEL_V3.md): build a real-jet observations.h5 at the
``primary_test`` holdout theta, for omnilearned.sbi.toy.numerics's Delta-log-L convergence
check.

Not toy.closure.jewel_ensembles: that one is P7-scoped (K disjoint ensembles, group_id for
the composite-likelihood check, still hardcoded to the toy/v2 shard reader -- see
docs/RESTART_JEWEL_V3.md's TODO) and reserves test-split jets for the actual closure
campaign. This script draws a single N=1000 slice for a one-time numerics sanity check and
is deliberately standalone so it does not compete with P7 for the same jets or pull
closure.py's P7 concerns into P5.

    PYTHONPATH=$PWD/src henv -n omni-sbi -x python scripts/p5_prepare_observations.py \\
        --jewel-dir $JEST_NLE_ROOT/data/jewel_v3/merged --config configs/jewel-2.6.0-cs-v3.toml \\
        --split test --setting primary_test --n 1000 --out <out>/p5_obs.h5
"""

import argparse
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jewel-dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--setting", default="primary_test",
                    help="key into cards.load_config(config)['holdout'], or 'primary_test'/"
                         "'development_validation'/'additional_test_N'")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from omnilearned.sbi.jewel import cards
    from omnilearned.sbi.observations import write_observations
    from omnilearned.sbi.toy.design import reserved_settings
    from omnilearned.sbi.toy.train import shards_for_config

    cfg = cards.load_config(a.config)
    theta = reserved_settings(cfg)[a.setting]
    print(f"theta ({a.setting}) = {theta.tolist()}")

    paths_fn, reader = shards_for_config(a.config)
    d = reader(paths_fn(a.jewel_dir, a.split))
    sel = np.all(np.abs(d["theta"] - theta[None]) <= 1e-6, axis=1)
    have = int(sel.sum())
    print(f"{have} jets at this theta in {a.jewel_dir}/{a.split}")
    if have < a.n:
        raise ValueError(f"need {a.n} jets, only {have} available")

    rng = np.random.default_rng(a.seed)
    idx = np.flatnonzero(sel)
    take = rng.choice(idx, size=a.n, replace=False)
    take.sort()  # file order, for reproducible slicing / easier spot checks

    jpt_col = list(d["global_names"]).index("jet_pt")
    x, mask = d["x"][take], d["mask"][take]
    jet_pt = d["global"][take, jpt_col]
    group_id = d["event_id"][take]  # originating JEWEL event, kept as a diagnostic only
    event_id = np.arange(a.n, dtype=np.int64)  # must be unique: seeds the Hutchinson probes

    x = np.where(mask[..., None], x, 0.0)
    # theta itself is truth and must not be written here (write_observations refuses any
    # "theta*" attribute) -- the numerics check only needs real jets, not which setting they
    # came from; the setting name is recorded in FINDINGS_JEWEL_V3.md instead.
    #
    # train.py's context_for_layout logs jet_pt before it reaches the layout ("recommended
    # with log(pT/GeV) scaling, centered near 0" -- multiplicity.py's layout_with_multiplicity
    # docstring); FlowSurrogate.sample() takes raw GeV and logs it internally for convenience,
    # but log_prob() does not -- its context argument must already be in the trained units.
    # Passing raw GeV here first (both "jet_pt" conventions exist in this codebase and only
    # one matches what log_prob expects) pushed the context ~50-100x out of the trained
    # manifold and produced multi-million-magnitude, sign-flipping log L sums; see
    # FINDINGS_JEWEL_V3.md, P5.
    sha = write_observations(a.out, x, mask, event_id, feature_names=("deta", "dphi", "log_pt", "log_e"),
                             context=np.log(jet_pt)[:, None], context_names=("jet_pt",),
                             group_id=group_id,
                             attrs={"source": "jewel_v3_test_split", "n_events": int(a.n)})
    print(f"wrote {a.out} ({sha}); jet_pt range [{jet_pt.min():.1f}, {jet_pt.max():.1f}] GeV")


if __name__ == "__main__":
    main()
