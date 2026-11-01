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

## P2: generation under way (2026-11-01) — measured acceptance ~2x worse than planned

`topup`'s self-correction is working as designed, but the numbers it is correcting are large.
Plan §2 budgeted train at 12 node-h (1024 tasks); measured per-point acceptance drove `topup`
to grow the task list to 22 809 (train) and 5855 (fidelity), from 1024 and 256 respectively.
Rough node-hour estimate from task count and the measured ~4.1 min/task at 64 workers/node:

| split | planned tasks | actual tasks (so far) | planned node-h | estimated actual node-h |
|---|---|---|---|---|
| train | 1024 | 22 809 | 12 | ~24 |
| fidelity | 256 | 5855 | 3 | ~6 |
| validation | 256 | 1487 | 1 | ~1.6 |
| test | 8 (then 90 at `--nevent 5000`) | 90 | 0.3 | ~0.5 |

Total ≈ 33 node-h against a 16 node-h plan estimate — about 2x over, still well inside
"budget pre-approved, log node-hours, do not block on questions." Validation reached its
per-point target cleanly (649 789 jets at 128 pts × 5000 = 640 000). Test's first merge ran at
62/90 tasks (142 646 of a 200 000-jet target, 71%) because the driver's `WAVES` cap was hit
before the two outstanding farm jobs finished, not because `topup` under-estimated chunks
needed (no `too_many_constituents`-style cap warning fired) — relaunched with `--waves 6` to
let it top up and re-merge once the in-flight tasks land.

**Operational note, not a code bug:** the driver loops (`jewel_v3_drive.sh`, launched via
`setsid nohup ... &`) died twice when the interactive session ended, despite `nohup` —  the
submitted Slurm jobs themselves were unaffected, but the wave→topup→merge bookkeeping stalled
each time until manually relaunched (idempotent, so safe, but costs wall-clock before someone
notices). `setsid` alone did not change the behaviour between the two session endings tested
here; worth a method change (e.g. a persistent `screen`/`tmux` session) if it recurs.

## P2 correction: the "2x node-hours" above was mostly a crash, not low acceptance

The section above took the task-count growth at face value — `topup`'s own acceptance math,
which only sees successful manifests. It does not see failed attempts, and there were far
more of those than successes: `event_jets` left its rejection reason `None` whenever every
clustered jet candidate in an event was below `min_constituents` (a common, not exotic, event
shape — nothing exotic about two sub-threshold candidates). `run_task`'s
`pre_reject[why] += 1` then raised `KeyError(None)`, and because the exception propagates out
of the *whole* per-event loop, it discarded every event already processed in that task, not
just the offending one.

Measured failure rate before the fix: **57 593 failed attempts against 9576 successful train
manifests (86%), 41 057 against 5458 fidelity manifests (88%), 12 721+ for validation, and
enough of test's attempts that it made no net progress for the final ~3 hours of its run.**
The "2x node-hours" estimate was therefore mostly CPU spent on tasks that crashed partway
through a full JEWEL run and wrote nothing — not jets selected at a worse rate than planned.
True per-event acceptance is not yet known; it can only be measured honestly after a run with
the fix in.

Fixed in `jets.py` (`event_jets` now sets `reason = "too_few_constituents"` or `"no_jet"`
before returning whenever `recs` is empty, mirroring the singular `event_jet()` helper's
existing `why or "no_jet"` guard, which this plural path should have had from the start) and
defensively in `generate.py`'s call site. Regression test in `test_jewel_inclusive.py`.

**Action taken:** cancelled every running `jv3-*` farm job and the four driver loops (all on
the pre-fix snapshot), cleared claim locks, cut a new snapshot from the fix, and relaunched
train/fidelity/validation/test from it. Already-written manifests are untouched — nothing
generated so far is discarded, only resumed. Validation's merge (649 789 jets, 125/128
points, done via the new compute-node `jewel_v3_merge.sbatch` after the login-node OOM noted
above) stands as the last pre-fix checkpoint; worth a top-up pass once post-fix capacity is
available, since 3 points have no jets at all yet.
