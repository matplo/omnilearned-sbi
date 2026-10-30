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
| 2026-10-28 | 1p | 60481641 `jv2-probe` | 1 CPU node, 8 workers, 1 h | **cancelled** (batch queue congested; interactive took the tasks) | probe at 4 theta, 2000 events each, `data/jewel_v2/probe` |
| 2026-10-28 | 1p | 60481776 `jv2-probe-int` | 1 interactive CPU node, 16 workers | done, 15 min | the 4-theta probe, via interactive overflow |
| 2026-10-28 | 1p | 60482135 `jv2-corner` | 1 interactive CPU node, 8 workers | done, ~20 min | 3 hot/central theta, sizes the constituent cap |
| 2026-10-28 | 2 | 60482494/98/99 (batch waves) + 60482535/36 (interactive) | 8+6+2 nodes x 64 workers; 2 interactive | **running** from 23:24 | production train/fidelity/validation, snapshot `0917d16`, via `jewel_v2_drive.sh` |

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

### Production started cleanly (2026-10-28 23:30)

First tasks out ~4 min after launch (the lag is the per-node cppyy PCH build), 128 workers
in flight across the two interactive farms. Spot check of the first 14 task manifests:
backend `jewel-2.6.0-cs-v2`, `jet_multiplicity = inclusive`, **1.82-1.83 jets/event**
(matching the probe's 1.6-2.8 across the box), max N_const 79-84 with **0 capped** -- the
512 cap is doing its job. Targets: train 6000 jets/point x 500 Sobol points, fidelity
20000 x 128, validation 4000 x 128; the drivers `topup` from measured acceptance until
each point reaches its target.

## Test split complete and Phase-6 pseudo data verified (2026-10-29)

`--split test` had to be added to the CLI first. It was already wired everywhere else
(`cards.SPLIT_STREAM["test"]`, `randomness.njob_base.test = 500000`); only the argparse
`choices` list omitted it. **The tempting workaround would have been a correctness bug**:
planning the reserved-setting data as `probe` reuses `njob_base.probe = 800000`, which is
exactly the NJOB range the earlier probes consumed -- identical JEWEL seeds, duplicate events,
inside the very sample used to validate inference. A test now pins
`njob_base.test != njob_base.probe`.

64 tasks, 9 min on one interactive node, `index` + `merge` done:

| theta (b, T_ref, tau) | jets | jets/event | max N_const | ensembles @1000 |
|---|---|---|---|---|
| 1.5, 0.65, 0.35 | 31075 | 1.95 | 210 | 31 |
| 4.5, 0.45, 0.65 | 28478 | 1.78 | 197 | 28 |
| 7.5, 0.35, 0.85 | 25853 | 1.62 | 165 | 25 |
| 9.5, 0.25, 0.45 | 26337 | 1.65 | 93 | 26 |

`closure.jewel_ensembles` was then run against the real merged shards: **10 disjoint
ensembles of 1000 jets at every holdout theta**, disjointness verified by exact row
comparison. The plan asks for >= 10, so there is 2.5-3x headroom for larger N or more
ensembles.

### A number for the composite-likelihood question

Within an ensemble the draw gives **1.97-2.41 jets per distinct event**. That sets the scale
of the correction the Phase-6 coverage test is looking for: if an event's jets were perfectly
correlated the effective sample size would be ~N/2 and credible intervals would come out
~sqrt(2) = 1.41x too narrow; they are only partially correlated, so the true inflation lies
between 1 and ~1.41. Small enough that 10 ensembles may not resolve it cleanly, large enough
that it must be checked rather than assumed -- the one-jet-per-event refit is the
discriminator, and **the choice of remedy (effective-N correction vs one jet per event at
inference) is a methodological decision for the user, not one to make silently.**

## Phase 2 complete: `data/jewel_v2` (2026-10-29 ~03:20)

| split | tasks | jets | points | jets/point (median) | core-h | max N_const | capped |
|---|---|---|---|---|---|---|---|
| train | 2009 | 3 600 591 | 500 | 7207 | 152.8 | 255 | 0 |
| fidelity | 1569 | 2 808 129 | 128 | 21 820 | 119.4 | 243 | 0 |
| validation | 368 | 660 405 | 128 | 5181 | 28.1 | 242 | 0 |
| test | 64 | 111 743 | 4 | 27 407 | 4.5 | 210 | 0 |

**7.18 M jets for 305 core-h = 2.38 node-h**, against 2.42 M jets for 10.9 node-h in
`jewel_w1`: **~14x more cost-efficient per jet**, and about half the ~4.4 node-h projected
from the probe (the probe's theta sample was weighted toward the expensive hot corner). Every
split overshot its jets/point target, because `topup` sizes from measured acceptance with a
5 % margin and the acceptance it measures is per point.

**The constituent cap decision is now settled by the full sample: max 255 with 0 capped
across 7.18 M jets.** The old 256 cap would have begun dropping jets -- preferentially in the
hot central region, i.e. a theta-dependent acceptance hole. 512 leaves ~2x headroom.

### Operational lesson for the next production

Three times, the interactive farms cleared a wave so much faster than the congested batch
queue that the wave's batch array ended up queueing to do **nothing**, while the driver sat in
`while squeue ...` waiting for it and so could not reach `topup`/`merge`. Each time the fix
was to `scancel` the idle array, after which the driver advanced immediately. **The driver
should poll for "all tasks done" and cancel its own wave early** rather than waiting for the
array to drain; as written it serialises on the slowest scheduler path even when the work is
finished. Not changed now (production is done and the script is recorded as it ran), but it
is the first thing to fix if this is run again.

The self-correcting part worked exactly as intended: train 1000 -> 2009 -> 2020 tasks,
fidelity 256 -> 1564 -> 1569, validation 256 -> 368, every step sized from the acceptance
actually measured per point rather than from a prediction.

## WEXPO = 5 is mismatched to a 20 GeV jet floor; regenerating at WEXPO = 3 (2026-10-29)

Measured on the finished `jewel_v2` train split (`merged/train_00000.h5`, 234 438 jets):
**ESS/n = 0.006**, normalised max/mean weight 1615 -- against 0.138 in `jewel_w1` and the
~0.11 I projected. Effective training set ~21 600 of 3.6 M jets, i.e. **~43 effective jets per
theta point**, which cannot support a conditional density over 500 points.

Where it goes, by pT-hat (the weight is the physical cross-section, the jets are what was
generated):

| pT-hat | jets generated | share of physical weight | ESS/n in bin |
|---|---|---|---|
| 10-50 | 4 % | **84 %** | 0.50-0.64 |
| 50-100 | 17 % | 15 % | 0.39 |
| 100-600 | 79 % | 1.2 % | 0.42-0.65 |

and by jet pT: **72 % of the physical weight sits in jet_pT 20-30 GeV, whose in-bin ESS/n is
0.019.** Generated pT-hat has median 184 GeV; only 50 jets of 234 k came from pT-hat < 20.

**Cause.** With `WEIGHTED T`, JEWEL samples the hard process ∝ dσ/dp × p^WEXPO and weights by
p^-WEXPO. Reading the exponent straight off the generated distribution, the true spectrum here
goes as **dσ/dp ~ p^-4**, so at WEXPO = 5 the sampled density is ∝ p^+2 per unit ln p -- it
piles up at high pT-hat while the physically dominant low-pT-hat region is starved.

**Why it did not bite before.** At the old 60 GeV jet cut, low-pT-hat events were rejected
anyway, so the accepted sample spanned only ~[50, 400] and WEXPO = 5 was well matched (ESS
0.138). **The 20 GeV floor is what exposes the full pT-hat range**, and it changes which
exponent is correct. The two decisions are coupled; nothing was wrong with WEXPO = 5 for the
window it was chosen for.

| WEXPO | sampled dN/dln p | ESS/n (log-flat model) |
|---|---|---|
| 2 | p^-1 | 0.244 |
| **3** | **p^0 (log-flat)** | **0.163** |
| 5 | p^+2 | 0.098 model / **0.006 measured** |

**WEXPO = 3 gives log-flat pT-hat coverage** -- uniform statistics per decade across 20-500
GeV, which is exactly what this analysis wants -- and ~27x the effective sample size. It is
also what this project's own earlier ESS study recommended before JEWEL's default of 5 was
adopted for the 60 GeV window.

**Action.** All four splits are being regenerated into `data/jewel_v2k3` at WEXPO = 3 (the
backend id is unchanged -- WEXPO is per-task `sampling`, not part of `[jewel]`). Production is
only ~2.4 node-h, so this is cheap. In parallel, a **control training on the WEXPO = 5 data**
(`runs/w6_k5ctl`, 200k steps) is running on the otherwise-idle GPU, so the decision is settled
by comparing two trained surrogates rather than by this argument alone.

## WEXPO = 3 regeneration: measured outcome and its true cost (2026-10-29)

`data/jewel_v2k3`, snapshot `cc7914b`, all four splits generated and merged.

| | `jewel_v2` (W=5) | `jewel_v2k3` (W=3) |
|---|---|---|
| train jets | 3 600 591 | 3 298 642 |
| **train ESS/n** | **0.0065** | **0.1297** |
| **effective train jets** | 23 309 (47/point) | **427 709 (855/point)** |
| max/mean weight | 1615 | 118 |
| pT-hat median | 184 | 67 |
| jet_pT 20-30 in-bin ESS/n | 0.019 | **0.212** |
| jets/event | 1.75 | 0.40 |
| total cost | 2.38 node-h | **7.92 node-h** |

**The honest trade.** W=3 costs **3.3x more node-hours for slightly fewer stored jets**,
because log-flat pT-hat sampling spends most events at low pT-hat where many fail the 20 GeV
jet cut (jets/event 1.75 -> 0.40). Per *effective* jet, though, it is **5.6x cheaper**
(1.17e-3 vs 6.5e-3 core-h), and the effective count is what bounds what the surrogate can
learn: 855 vs 47 effective jets per theta point. Both datasets are kept; `jewel_v2` is the
W=5 reference.

Production spent so far: 2.38 + 7.92 = **10.3 node-h**, comparable to `jewel_w1`'s 10.9 for a
single smaller dataset.

**Consequence to watch:** the lower acceptance also shrank the test split to ~6200 jets per
holdout theta -- only 6 ensembles of 1000, against the >= 10 the plan requires. Topped up to a
15 000-jet target (94 tasks added). Fidelity and train were unaffected because `topup` sizes
from measured acceptance, so they simply ran more chunks; only the test split, planned with a
fixed `--chunks 16` rather than a jet target, came up short. **Plan test splits by jet target,
not by chunk count.**

### A self-inflicted failure worth recording

The WEXPO = 5 control training (job 60488240) died in 16 s: `FATAL: no .../jewel_v2k3/merged/
index_train.json`. I had passed `sbatch` the **worktree** path of `jewel_v2_train.sh`, then
edited that script (adding the `DATASET` default) while the job sat queued; it started hours
later and read the mutated file. This is the snapshot-immutability rule of
`docs/RESTART_JEWEL.md` violated from the other direction -- the snapshot exists precisely so
a queued job cannot be changed under it. Phase 3 now goes through
`runs/jv2k3_train.sbatch`, which `exec`s `$SNAP/shell/nle/jewel_v2_train.sh`, with the reason
in a comment.

The control was **not** re-run: the ESS table above is a direct measurement of effective
sample size, so re-confirming an 18x gap by training two surrogates would spend ~2.5 GPU-h to
learn nothing the measurement does not already show.

## Phase 3 running: the weighted ESS confirmed inside training (2026-10-29)

Two sets, both on `jewel_v2k3`, snapshot `a00c9b1`:

| job | runs | where |
|---|---|---|
| 60520874 | 200k x seeds 0-2 + 100k control | interactive, 4 GPUs, ~1.9 h |
| 60520873 | **400k x seeds 0-2 + 200k control** | batch, 4 GPUs, 10 h limit |

The 200k set runs first so that the gate / `ptbins` / closure path can be exercised on a real
checkpoint while the 400k set -- the one that answers whether the toy's "training length is
the lever" result transfers to JEWEL -- works through the queue.

`toy.train` logs the weighted ESS, which independently confirms the WEXPO decision from
inside the training rather than from the shards:

```
weights train:      ESS 431710 of 3298642, per point 965
weights validation: ESS  73756 of  572015, per point 648
per point and pT bin: [940, 125, 100, 80, 112, 79, 78, 112]
```

965 effective jets per theta point, against the ~47 the WEXPO = 5 data would have given.

**Per-pT-bin effective counts are 78-125 per point above the lowest bin.** This is inherent,
not a defect of the sampling choice: the weights restore the true steeply-falling spectrum, so
the high-pT bins are weight-starved *by construction* no matter how the generation is biased.
Across 500 theta points that is still ~50 k effective jets at high pT for a shared model, but
it is the number to look at first if `ptbins` shows a high-pT fidelity problem -- and it is
the quantitative case for the factorised-kinematics fork (PLAN section 9.1) should one be
needed.

### Run names now follow the step count

The run names were hardcoded `set_d256_L6_400k_*` regardless of `STEPS`, so the 200k set
would have been written into directories labelled 400k -- in the one phase whose purpose is
to compare training lengths. Fixed (`K=$((STEPS/1000))`), both sets relaunched; ~3 min of
training discarded. Third provenance-class slip of the session, after the stale worktree
script path and planning the test split by chunk count instead of a jet target.

## Inclusive jets break the observation format's id assumption (2026-10-29)

Smoke-testing the numerics path on real JEWEL observations -- rather than waiting to meet it
in Phase 6 -- hit `ValueError: event ids must be unique` from `write_observations`.

**The guard is correct and was kept.** `event_id` seeds the Hutchinson probes
(`likelihood.stable_probes`, which does `np.unique(event_ids, return_inverse=True)`), so two
rows sharing an id are evaluated with **identical probe vectors**. Each jet's divergence
estimate stays unbiased, but their errors become correlated, inflating the variance of the
summed log-likelihood -- roughly 2x at the 2 jets/event an inclusive sample gives. Relaxing
the check would have bought a silent numerical degradation in the quantity the whole fit
rests on.

**Fix:** separate the two jobs the id was doing.

| field | role | who reads it |
|---|---|---|
| `event_id` | unique per jet; seeds the probes | the likelihood |
| `group_id` (new, optional) | the originating JEWEL event; several jets share it | diagnostics only |

`observations.py` gains an optional `group_id` dataset (carried through `Observations`,
`subset()` and the reader), `records.write_observation_file` passes it, and
`closure.jewel_ensembles` now emits a unique per-jet `event_id` with the JEWEL event in
`group_id`. Verified on real merged data: 8 jets from 4 distinct events, ids unique, grouping
preserved. This is also exactly what the Phase-6 composite-likelihood check needs -- the
one-jet-per-event refit is now a `group_id` filter.

**Sixth toy-assumption found this way, and the most consequential.** The others produced
wrong numbers; this one would have blocked Phase 6 outright, and the tempting fix (drop the
uniqueness check) would have quietly corrupted the likelihood's variance instead. The others
were: the stale `sbatch` worktree path, the test split planned by chunk count, run names
hardcoded to 400k, `ptbins` binning on 60-200, and `closure prepare` defaulting to the toy
config.

**A regression I introduced doing that, caught by the suite.** `group_id` was first inserted
before `context` in the `Observations` dataclass, which silently shifted every positional
argument after it -- callers writing `Observations(x, mask, eid, None, (), feature_names)`
had `context` land in `group_id`, `context_names` in `context`, and so on. Two tests failed
immediately. Appending the field at the end instead is the non-breaking way to extend a
dataclass; a regression test now pins positional construction. Same silent-misassignment
class as the toy-assumption bugs, but self-inflicted -- the argument for running the full
suite after every change to shared code, not only at the end. Suite: **174 passed, 7
skipped.**

## The gate's reference sample is biased for a weighted backend (2026-10-29) -- Phase 4 invalid as run

First gate run (200k seed-0 surrogate, 108/128 points) came back catastrophic: every point
FAIL, `p_family` ~1e-38, and relevance **12.1** (jet_pT) / **12.4** (width) at N = 1000
against a requirement of 0.2. Numbers that extreme are a bug signal, not a fidelity result,
so the comparison was checked before the surrogate was blamed.

**`toy/gate.py` ignores the per-jet weights entirely** (`direct_by_point` reads x/mask/theta;
the word "weight" does not occur in the file). For the toy that was correct -- direct samples
were unbiased, so the stored jets *are* the physical distribution. For a weighted JEWEL
backend it is wrong: the stored direct jets follow the **biased** generation law.

Measured at fidelity point 0 (21 216 jets):

| mean jet_pT | value |
|---|---|
| unweighted -- what the gate used as "direct" | **54.81 GeV** |
| weighted -- the physical distribution | **28.68 GeV** |
| bias in the reference | **+26.1 GeV** |
| gate's reported surrogate - direct | -18.59 GeV |

So the surrogate's mean is ~36.2 GeV: much closer to the physical 28.7 than to the 54.8 it
was scored against. **The reference was biased, not (mostly) the surrogate.** Both gate jobs
were cancelled rather than spend more GPU on an invalid comparison.

**Not a clean bill of health for the surrogate.** 36.2 vs the physical 28.7 still leaves
~+7.5 GeV unexplained -- possibly genuine surrogate error, possibly p(N|theta) or the
selection applied to surrogate samples. That can only be separated once the comparison is
correct; it is explicitly an open question, not a passed test.

### Why the obvious fixes are each wrong, and what is needed

- **Resample the direct jets in proportion to weight** (free, reuses all machinery) gives a
  physically distributed reference, but only at the split's effective size -- ESS/n 0.13 on
  21 k jets/point is ~2800 effective, versus 20 k unweighted in the toy. The noise floor on
  the constrained shift scales as sqrt(20000/2800) = 2.7x worse, which cannot resolve the
  0.2 sigma requirement. Valid but too weak.
- **Generate an unweighted fidelity sample** is conceptually cleanest -- it is literally what
  an experiment sees -- but unweighted sampling at a 20 GeV jet floor puts essentially
  everything at low pT and almost nothing above 200 GeV, so the gate could no longer test
  high-pT fidelity at all. This is the same starvation that motivated weighting.
- **Weighted statistics in the gate** is therefore the right answer: keep every jet at every
  pT, and weight the observables. The relevance part (which carries the 0.2 sigma acceptance
  criterion) needs only weighted means and variances. The permutation significance needs a
  weighted ECDF in the KS statistic, permuting (value, weight) pairs jointly -- the null is
  still exchangeability, so the test stays calibrated.

**Seventh toy assumption, and the one that invalidated a real measurement rather than a
mechanism.** The pattern now has a precise statement: *every place the toy could assume its
sample was unbiased and one-jet-per-event has to be revisited for a weighted, inclusive
backend.* `ptbins` has the same defect and needs the same treatment.
