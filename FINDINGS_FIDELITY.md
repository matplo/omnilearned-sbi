# Surrogate fidelity thread (branch dev/nle-fidelity)

Running log of `docs/SURROGATE_FIDELITY_PLAN.md` steps S0–S5 (toy v2), kept apart from
`FINDINGS_NLE.md` (the main thread folds these entries in). Worktree
`/global/cfs/cdirs/alice/ploskon/henv-omni-sbi/omnilearned-sbi-fidelity`; every job runs from
`$JEST_NLE_ROOT/src_snapshots/<rev>` of a commit on this branch. Criteria unchanged (rules 2
and 5): gate FAIL = family-wise p < α/128 and relevant at N = 1000; requirement |δθ| ≤ 0.2σ at
N = 1000. Choices use validation/development data only.

Reference (round 2, `set_d256_L6_200k_multC64`): 1 FAIL, 23 with p < 0.01, constrained shift
1.17 (noise ≈ 0.55), marginal +0.53, +0.48, −0.30 σ (b, T_ref, τ_init) at N = 1000.

## Job / node-hour log

| date | step | job | resources | status | what |
|---|---|---|---|---|---|
