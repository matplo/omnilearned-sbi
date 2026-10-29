# JEWEL NLE track (branch `dev/nle-jewel`)

Running log of `docs/PLAN_JEWEL_NLE.md`. State and restart instructions:
`docs/RESTART_JEWEL.md`. Earlier logs stay where they are and are not rewritten:
`FINDINGS_NLE.md` (toy + NLE machinery), `FINDINGS_FIDELITY.md` (toy fidelity),
`FINDINGS_WEIGHTED.md` (weighted-sample study).

Every entry records the source revision, dirty status and the actual device.

## Job / node-hour log

| date | phase | job | resources | status | what |
|---|---|---|---|---|---|
| 2026-10-28 | 0b, 1 | — (login) | — | done | branch, merge, v2 config, inclusive jets, Sobol design, tests |
| 2026-10-28 | 1p | 60481641 `jv2-probe` | 1 CPU node, 8 workers, 1 h | submitted 22:5x | probe at 4 theta, 2000 events each, `data/jewel_v2/probe` |

## Phase 0b: branch and consolidation (2026-10-28)

`dev/nle-jewel` cut from `dev/nle-perlmutter` (465ec63), with `dev/nle-weighted` merged in
(merge commit `5a5e859`, **no conflicts**). This is the first branch carrying all four
capabilities the JEWEL NLE needs at once:

| capability | came from |
|---|---|
| ridge-following lattice scan (`sbi/ridge.py`) | `dev/nle-perlmutter` |
| weighted JEWEL generation (`generate plan --wexpo`) | `dev/nle-weighted` |
| weighted training loss (`flow.py --weight-mode`) | `dev/nle-weighted` |
| per-pT fidelity (`toy/ptbins.py`) | `dev/nle-fidelity` via weighted |

`dev/nle-perlmutter` is left intact as the toy-track record rather than retargeted.

- **Worktree gotcha, now documented:** a fresh git worktree has no
  `src/omnilearned/_version.py` (gitignored, generated at install), so the whole test suite
  fails collection with `ModuleNotFoundError: omnilearned._version`. Recreated as
  `snapshot.sh` does. This cost one wasted test run; the fix is in `docs/RESTART_JEWEL.md`.

## Phase 1: backend v2 and the code it needs (2026-10-28)

**New backend `jewel-2.6.0-cs-v2`** (`configs/jewel-2.6.0-cs-v2.toml`). Changing `[jewel]` or
`[selection]` defines a new backend id, so `jewel_v1` (unweighted, PTMIN 50) and `jewel_w1`
(weighted, PTMIN 50) become legacy references — they are **not** valid training or fidelity
inputs for this sample definition.

| field | v1 | **v2** | why |
|---|---|---|---|
| `ptmin_GeV` / `ptmax_GeV` | 50 / 400 | **10 / 600** | the pT-hat range must bracket the jet window on *both* sides or the jet spectrum is truncated at its ends |
| `weighted` | false | **true** (WEXPO 5 per plan) | flat-per-decade statistics across the whole jet range |
| `jet_pt_range_GeV` | [60, 200] | **[20, 500]** | what a future experiment could measure (user) |
| `jet_abs_eta_max` | 1.0 | **2.0** | ditto |
| `constituent_abs_eta_max` | 2.5 | 2.5 | unchanged; >= 2.0 + R = 2.4, so an edge jet keeps its cone |
| `jet_multiplicity` | (leading, implicit) | **inclusive** | experiments measure inclusive jets |
| training design | 12x9x9 Cartesian (972) | **500 scrambled Sobol** | smooth theta-interpolation; no axis-aligned structure |

### Decisions settled during planning, with their evidence

- **No pT-hat slicing — single weighted range.** I had proposed slicing out of concern for the
  weight dynamic range, `(ptmax/ptmin)^WEXPO` = 1.0e8 at [10,600]. That was the wrong
  statistic. For log-uniform pT-hat sampling, **ESS/n ~ 0.4 / ln(range ratio)** — logarithmic,
  not catastrophic: [50,400] -> ~0.19, [10,600] -> ~0.11, against **0.138 measured** in
  `jewel_w1`. After `merge`'s per-point renormalisation the heaviest jet is only ~20-60x the
  mean (observed max 62 in w1), which trains fine. The user's objection was correct.
- **Inclusive jets, and why not leading.** In a dijet the jet that interacted *less* with the
  medium tends to be the leading one, so a leading-only sample is biased towards vacuum-like
  jets (user, physics). `leading` and `subleading` stay selectable for later retraining;
  refining on the subleading/leading pT ratio is deliberately out of scope for now.
  - **Consequence, to be measured not assumed:** jets of one event are correlated, so the
    summed log-likelihood is a *composite* likelihood. Point estimates stay consistent;
    credible intervals may be too narrow. `event_id` is shared across an event's jets, so the
    Phase-6 coverage test measures this directly and a one-jet-per-event refit diagnoses it.
- **Correction to an earlier claim of mine.** I had asserted that PTMIN = 50 sits too close to
  the 60 GeV jet cut. That was **never a recorded finding** (no match for `pthat`/`PTMIN` in
  `FINDINGS_FIDELITY.md`), and measuring it directly in `jewel_w1/merged/train_00000.h5`:
  **2.0 %** of 60-62 GeV jets have pT-hat in [50,55], 6.4 % have pT-hat < jet pT, 10.7 % over
  all jets. A real tail effect, not the dominant artifact I implied. The separately recorded
  "60 GeV boundary leakage" is a *flow* mismodelling the density at the cut — a different
  problem. **Lowering PTMIN is justified by the new 20 GeV jet floor, not by that 2 %.**

### Code

- `jewel/jets.py`: `event_jets()` returns one record per kept jet (`jet_indices()` selects by
  mode); `event_jet()` kept as the single-jet form. Each jet is rotated to **its own** axis.
  `max_constituents` now drops the offending jet, not the whole event.
  - **Trap found by the new test:** `reco.cluster_events` returns `jet_pt/eta/phi` for the
    **leading jet only** (`[N]`, not `[N, njets]`); only `membership` is per-jet. Indexing
    `jet_phi[j]` raised `IndexError` for j > 0. Non-leading axes are now summed from their
    constituents (`jets._axis`), and a test asserts this reproduces the clusterer's own
    leading-jet axis to 1e-9.
- `jewel/generate.py`: the worker extends records per event instead of appending one, so an
  event's jets share its `event_id`; the manifest adds `jet_multiplicity` and
  `events_with_a_jet` (`acceptance` is now jets/event and may exceed 1).
- `jewel/generate.py::design_points`: `design.train_points = N` selects a Sobol training
  design; without the key the historical Cartesian grid is used, so older configs are
  unaffected.
- `tests/sbi/test_jewel_inclusive.py` (new, 6 tests, all pass): mode selection incl.
  degenerate counts, inclusive-vs-leading-vs-subleading consistency on a synthetic two-jet
  event, per-jet rotation, the leading-axis consistency check above, the v2 window invariants
  (pT-hat brackets the jet range; constituent eta >= jet eta + R), the Sobol design count and
  **the rule-2 reservation margin against all four holdout settings** (nearest training point
  is 3.3 % of the box away, limit 2 %), and Cartesian back-compatibility.

### Test suite after the merge and the phase-1 changes

`tests/sbi` on the login node, revision `e67510f` (clean), CPU: **163 passed, 7 skipped**,
8 min 28 s. The count rises from 123 on `dev/nle-perlmutter` because the weighted branch's
tests came in with the merge. The 7 skips are the MPS-only tests, as before.

## Phase 5 started early: what the JEWEL NLE actually needs (2026-10-28)

While the probe was queued, the "how much of the NLE is toy-bound?" question was settled by
reading rather than guessing. **Almost none of it.**

| component | toy-bound? | action |
|---|---|---|
| `fit.py`, `likelihood.py`, `scan.py`, `ridge.py`, `flow.py`, `observations.py` | no toy imports at all | reuse unchanged |
| `toy/numerics.py` (M4 convergence) | takes only checkpoint + observations + theta pairs; imports the shared `PARAMETERS` | **reuse unchanged** |
| `toy/gate.py` | already takes `--config` and handles a JEWEL selection | reuse |
| `closure.report` / fits / sealing | keyed on `PARAMETERS`, which JEWEL imports from the same module | reuse unchanged |
| `closure.prepare` observation-making | **yes** — the toy runs its simulator on the fly | new `source = "jewel"` branch |
| `closure.acceptance_part` | **yes** — hardcoded toy selection | `--config` added |

So no separate `jewel/closure.py` was written: duplicating ~400 lines of report/seal logic
would have been the wrong call when only observation-making differs. `(b, T_ref, tau_init)`
is literally the same `PARAMETERS` tuple for both backends.

- `closure.jewel_ensembles(dir, split, theta, n, K)`: K **disjoint** ensembles at a theta from
  merged shards, raising if the split cannot supply `n*K` jets rather than recycling them.
  Records keep the shared `event_id`, so an inclusive sample's within-event correlation stays
  visible to the coverage report (the composite-likelihood check).
- `closure.load_sample_config(path)`: toy or JEWEL config -> `(cfg, Selection)`.
- Tests: `tests/sbi/test_jewel_closure_source.py` (4), all pass.

## Phase 1p: probe at PTMIN = 10, inclusive jets, 20-500 GeV (2026-10-28)

Snapshot `e67510f`, job 60481776 (1 interactive CPU node, 16 workers, 15 min), 4 theta x 2000
events, `data/jewel_v2/probe`. **The batch queue was oversubscribed** (`jv2-probe` 60481641 sat
in `PENDING`), so the same task directory was also served by an interactive allocation, which
started in ~20 s; claims are exclusive, so the two cannot duplicate work. The batch job was
cancelled once the interactive one had the tasks. Worth reusing: **an interactive allocation
is a drop-in overflow for a congested batch queue on any claim-based task directory.**

| b | T_ref | tau | jets | jets/event | max N_const | s/event | capped |
|---|---|---|---|---|---|---|---|
| 9.5 | 0.25 | 0.45 | 3244 | 1.62 | 94 | 0.096 | 0 |
| 7.5 | 0.35 | 0.85 | 3130 | 1.57 | 138 | 0.190 | 0 |
| 4.5 | 0.45 | 0.65 | 3598 | 1.80 | 160 | 0.251 | 0 |
| 1.5 | 0.65 | 0.35 | 3925 | 1.97 | 154 | 0.319 | 0 |

- **Inclusive selection works**: 1.6-2.0 jets/event, against 0.26 for leading-only at the old
  60 GeV / |eta| < 1 window. Clustering produces ~60 candidate jets per event, nearly all
  rejected as `too_few_constituents` (single thermal particles); that costs 9 % of task time
  (17.7 s processing against 192 s of JEWEL), so it is not worth optimising.
- **~23x cheaper per jet than `jewel_w1`**: 0.059 core-s/jet at the peripheral point and
  ~0.18 averaged over the box, against ~1.36 for w1 (leading-only, PTMIN 50). Both effects
  contribute: more jets kept per event, and low-pT-hat events are individually cheap.
  Projected production cost is **~3 node-h**, against 10.9 for w1 -- statistics are not the
  binding constraint here.
- **JEWEL regenerates its xsec/PDF tables per task** in a fresh scratch dir, so they always
  match the card's pT-hat range; changing PTMIN carries no stale-table risk.

### The 256-constituent cap has to be raised (and my first extrapolation of it was wrong)

From the first three points (94 -> 138 -> 160 at T_ref 0.25/0.35/0.45) I extrapolated linearly
to ~310 at the box edge T_ref = 0.9. **That was wrong**: the fourth point (T_ref 0.65) came
back at 154, i.e. the count *saturates* rather than growing linearly. The probe was worth
running precisely because it contradicted the extrapolation.

The cap is still too small, for a different and more careful reason -- **a 3.5 k-jet probe
underestimates the maximum of a 3 M-jet production**. Fitting the tail instead of reading off
the max: the top decile decays with a scale of **~17 constituents per e-fold**, so

| jets in sample | expected max N_const |
|---|---|
| 1.4e4 (this probe) | 160 (observed) |
| 1e6 | 233 |
| 3e6 (planned train split) | **252** |

That lands on the 256 cap before the hotter corner of the box is counted at all. A capped jet
is dropped (recorded as `too_many_constituents`), and it would be dropped **preferentially in
the hot, central region where the quenching signal is strongest** -- a theta-dependent hole in
the acceptance, which is exactly the kind of defect an NLE cannot absorb. Raising the cap is
cheap: shards are zero-padded (gzip absorbs it) and `read_shards(trim=True)` trims to the
actual maximum, so training cost follows the real N, not the cap.

### Memory: inclusive clustering had to be pre-filtered before packing

The probe's rejection tally exposed a scaling problem that only appears with inclusive jets.
Per 2000-event task the clusterer yields **~198 000 jet candidates** (~99 per event), of which
**120 630 (61 %) are single thermal particles** rejected as `too_few_constituents`. All of them
were being packed into `[n, max_constituents]` arrays before the selection saw them, which is
what sets a worker's peak memory:

| cap | candidates packed | peak/worker | x128 workers/node |
|---|---|---|---|
| 256, unfiltered | 198 k | ~2.5-3 GB | ~320-380 GB (a 512 GB node, tight) |
| 512, unfiltered | 198 k | ~5-6 GB | **OOM** |
| 512, pre-filtered | 77 k | ~1.9 GB | ~240 GB |

`select_sets` rejects exactly `count < min_constituents` on the same constituent count that
`event_jets` already has, so dropping those jets before packing is **provably equivalent** --
it cannot change the accepted set, only avoid materialising rejects. The skipped jets are
still returned and added to the manifest's rejection tally, so accounting is unchanged.
`tests/sbi/test_jewel_inclusive.py` pins the equivalence.

This is why raising the constituent cap could not simply be done on its own.

### Corner probe: the cap is raised to 512 (and "it saturates" was wrong too)

Job 60482135 (second interactive CPU node, run in parallel with the main probe), 3 hot/central
theta, 1500 events each:

| b | T_ref | tau | jets | jets/event | max N_const | s/event |
|---|---|---|---|---|---|---|
| 3.0 | 0.75 | 0.60 | 3298 | 2.20 | 215 | 0.417 |
| 0.5 | 0.88 | 0.25 | 3109 | 2.08 | 187 | 0.395 |
| 0.5 | 0.88 | 0.95 | 4130 | 2.76 | **224** | 0.573 |

**Correcting myself twice on the same quantity.** First I extrapolated 94 -> 138 -> 160
linearly to ~310 at the box edge. Then the T_ref = 0.65 point came back at 154 and I read that
as saturation. Both were wrong: that point has a *short* `tau_init` (0.35), and the count
depends on `tau_init` as well as `T_ref` -- at T_ref 0.88 the long-tau point gives 224 against
187 for the short-tau one. The clean statement is that N_const grows with the amount of medium
the jet traverses, which is set by both parameters, and no two-point extrapolation along one
axis was ever going to capture it.

Pooled over all 7 probe points (24 434 jets): max 224, p99.9 = 161, tail **~19.6 constituents
per e-fold**, so projected maxima of **297 / 318 / 342** over 1e6 / 3e6 / 1e7 jets.
**`max_constituents` 256 -> 512.** No probe jet was actually capped (`too_many_constituents`
= 0 at every point), so the probe data is bit-identical to what the new cap would have
produced and stays valid as a diagnostic.

**Cost, from the full set:** s/event spans 0.096 (peripheral) to 0.573 (hot corner), jets/event
1.6-2.8; averaged over the box ~0.16 core-s/jet. Projected production:

| split | points x jets/point | jets | ~node-h |
|---|---|---|---|
| train | 500 x 6000 | 3.0 M | ~2.1 |
| fidelity | 128 x 20000 | 2.6 M | ~1.8 |
| validation | 128 x 4000 | 0.5 M | ~0.4 |
| test | 4 x 25000 | 0.1 M | ~0.1 |

~4.4 node-h in total, against 10.9 for the smaller `jewel_w1`. Statistics are not the binding
constraint, so the jets/point targets are set generously rather than minimally.
