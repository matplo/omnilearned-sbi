# JEWEL v3 track (branch `dev/nle-jewel-v3`)

Running log of `docs/PLAN_JEWEL_V3.md`. State and restart: `docs/RESTART_JEWEL_V3.md`.
Format: `docs/FORMAT_JEWEL_V3.md`. Earlier logs are not rewritten: `FINDINGS_JEWEL.md` (v2),
`FINDINGS_NLE.md` (toy + NLE machinery), `FINDINGS_FIDELITY.md`, `FINDINGS_WEIGHTED.md`.

Every entry records the source revision, dirty status and the actual device.

## Job / node-hour log

| date | phase | job | resources | status | what |
|---|---|---|---|---|---|
| 2026-10-31 | P0 | — (login) | — | done | branch, worktree, docs |

## Why v3 exists (2026-10-31)

The v2 surrogate is far from passing the gate, and the representation audit
(`reports/m3-surrogate-fidelity/2026-10-31T1231-0700_jewel-v2_what-is-trained/`) found the
reason in the inputs rather than the training:

| | OmniLearned PET v2 (arXiv:2510.24066 §II.A) | v2 track |
|---|---|---|
| coordinates | Δη, Δφ from the jet axis | φ relative, **η absolute** |
| scale | log p_T, **log E** | log p_T only |
| standardisation | deliberately none | per-run `FeatureTransform` |
| conditioning | jet kinematics + multiplicity | **N only** |

Measured: constituent feature 0 correlates with the jet axis at **r = +0.999**, σ = 1.00,
while the R = 0.4 substructure it should carry has σ = 0.16. Subtracting the jet axis collapses
it onto the φ distribution exactly. The model spent most of its input range on where the jet
sat. `jet_pt` was stored in `global` and never reached the model.

**Carried forward from v2 as settled** (measured, not assumed):

- **WEXPO 3, not 5.** At a 20 GeV jet floor the full p̂_T range contributes and WEXPO 5
  oversamples high p̂_T: ESS/n 0.0065 vs 0.130, i.e. 47 vs 855 effective jets per θ point.
- **Training length saturates on JEWEL.** 100k→200k gains ~3 seed-σ; 200k→400k only 1.7σ, with
  best checkpoints at 284–310k. The toy's dominant lever does not transfer, so more training is
  not the fix for a fidelity gap.
- **Recoils stay on, with Milhano & Zapp constituent subtraction.** Raw events carry 366
  thermal particles on average (up to 2174); without subtraction jets are contaminated. The
  prescription is fixed in the backend id (`dRmax 0.5`, `leftovers=keep`,
  `thermal_pid=reference`).
- **Always write event stores.** v2 passed `--no-events`, which locked 10.3 node-h of
  generation to one representation. Storage was never the constraint (CFS has 44 PB free).

## P0: branch and scaffolding (2026-10-31)

`dev/nle-jewel-v3` cut from `dev/nle-jewel` (858b745); worktree
`omnilearned-sbi-v3`. Docs written: `docs/PLAN_JEWEL_V3.md` (the approved plan),
`docs/FORMAT_JEWEL_V3.md` (normative format), `docs/RESTART_JEWEL_V3.md` (live state and the
eleven traps already hit), this log. Nothing is reused from `jewel_v1`/`jewel_w1` — v3
generates fresh.
