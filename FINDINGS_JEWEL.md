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
