# JEWEL NLE: infer (b, T_ref, tau_init) from jet point clouds, 20-500 GeV

**This is the standing brief for the `dev/nle-jewel` track** (user realignment, 2026-10-28).
It supersedes `PLAN_NLE_PERLMUTTER.md` M5-M8 *for JEWEL*; that document remains the record of
the toy track and of the shared ground rules (sections 1-6), which still apply in full.
`docs/SURROGATE_FIDELITY_PLAN.md` (toy S0-S6) is closed.

**If you are resuming after an interruption, read `docs/RESTART_JEWEL.md` first** — it holds
the current state and the exact next command. This file is the *why* and the *shape*; that
one is the *where we are*.


## Context

**Why this realignment.** The toy track (`toy-quenching-v2`) built and validated the NLE
machinery — ridge-lattice scan, closure campaign, numerics check, fidelity gate — but its
physics was never independently verified, so toy fidelity results have limited transfer value.
The real objective is a machine that predicts **JEWEL** parameters from an ensemble of jets
represented as constituent point clouds. **The toy track stops here** (user decision). Its
machinery is kept and reused; its open questions (M5 criterion-3 τ_init, high-pT
`n_constituents`) are retired as toy science and carried forward only as checks on the
machinery itself.

**What changes physically.** The existing JEWEL samples were generated for a narrow analysis
window (jets 60–200 GeV, |η_jet| < 1, pThat 50–400). The target is now what a future
experiment could actually measure: **jets 20–500 GeV, |η_jet| < 2.0**, constituents
|η| < 2.5 (R = 0.4, so a jet at |η| = 2.0 has its cone inside 2.4). That requires pThat
coverage well below and above the jet window, so **pThat 10–600 GeV, WEIGHTED, WEXPO = 5**
(unchanged weighting scheme).

**Settled during planning:**
- **No pThat slicing.** ESS/n ≈ 0.4 / ln(pThat range ratio) — logarithmic, not catastrophic:
  [50,400] → ~0.19, [10,600] → ~0.11, vs 0.138 measured today. The 10⁸ max/min weight ratio is
  not the relevant quantity; with per-point normalization the heaviest jet is only ~20–60× the
  mean, as today (observed max 62). A single weighted range is correct.
- **PTMAX 400 → 600**, so the 500 GeV jet bin is not truncated from above (mirror of the PTMIN
  concern).
- **Inclusive jets — all jets passing the selection, not leading-only** (user decision, physics).
  In dijet events the leading jet typically interacted *less* with the medium, so a
  leading-only sample is biased toward vacuum-like jets; experiments measure inclusive jets.
  `jewel/jets.py` currently keeps only the leading jet, so this is a required code change
  (Phase 1). Keep leading-only / subleading-only selectable for later retraining; refining on
  the subleading/leading pT ratio is explicitly **not** in scope now.
  - **Consequence to manage, not ignore:** jets from one event are correlated (same medium
    realization, recoils, momentum conservation), so the summed log-likelihood becomes a
    *composite* likelihood. Point estimates stay consistent; **credible intervals will be too
    narrow** if the correlation is material. The Phase-6 coverage test measures this directly —
    that is exactly what coverage is for. If it under-covers, the remedies are an effective-N /
    sandwich correction, or one-jet-per-event at inference while still training inclusive.
    `event_id` is already stored per jet, so the correlation structure stays recoverable.
- **Measured, correcting an earlier claim of mine:** the PTMIN = 50 wall is a ~2 % tail effect
  (2.0 % of 60–62 GeV jets have pThat ∈ [50,55]); it was never a recorded finding. The
  separately-recorded "60 GeV boundary leakage" is a *flow* mismodelling the density at the cut
  — a different problem. Lowering PTMIN is justified by the new 20 GeV jet floor, not by that 2 %.
- Changing `[jewel]`/`[selection]` defines a **new backend id** (AGENTS.md), so all splits
  (train / validation / fidelity / test) must be regenerated consistently. Existing
  `jewel_v1` / `jewel_w1` become legacy references, not inputs.
- Production is cheap: `jewel_w1` (2.42 M jets, 9.4 M events) cost **10.9 CPU node-h**.
  Acceptance rises sharply with a 20 GeV floor and |η| < 2 (≈0.26 → near 1 leading jet/event),
  so this is roughly 3× cheaper per jet.

## Phase 0a — Write this plan down (no compute)

1. **`docs/PLAN_JEWEL_NLE.md`** in the code repo — this plan as the standing brief for the
   JEWEL-only track, committed on the consolidated branch. Add a pointer at the top of
   `PLAN_NLE_PERLMUTTER.md` saying the toy track is closed and that M5–M8 are superseded by
   this document for JEWEL, so the two briefs cannot be read as contradicting each other.
   Mark `docs/SURROGATE_FIDELITY_PLAN.md` (toy S0–S6) as closed.
2. **Journal entry in the reports repo** (`omni-sbi-nle-jet-dev-reports`): a
   `journal/entries/<stamp>_main_<slug>.md` recording the realignment — toy track stopped and
   why, the JEWEL-only objective, the new cuts and their rationale, the inclusive-jet decision
   and its physics justification, and the corrections established during planning (no pThat
   slicing; the 2 % PTMIN truncation; `ESS/n ≈ 0.4/ln(range)`). Build with `tools/journal.py`,
   sync with `tools/sync.py --push`. **This repo gets an entry at the end of every phase
   below**, not only at the start — including a figure-bearing status page under
   `reports/m5-closure/` or a new `reports/jewel-nle/` once Phase 4 and Phase 6 have results.

## Phase 0b — New branch + consolidate (no compute)

**Yes, a new branch: `dev/nle-jewel`**, cut from `dev/nle-perlmutter`, with
`dev/nle-weighted` merged into it. Rationale: this is a backend change (new config, new
selection, inclusive jets) plus a scope change, so `dev/nle-perlmutter` is left intact as the
historical toy + NLE-machinery record rather than being retargeted under it. All JEWEL work
lands on `dev/nle-jewel`; `AGENTS.md` is updated to point at `docs/PLAN_JEWEL_NLE.md`.

Three branches hold pieces this plan needs simultaneously:

| branch | has | missing |
|---|---|---|
| `dev/nle-weighted` (+37) | weighted JEWEL gen (`--wexpo`), weighted training loss, `toy/ptbins.py` | ridge NLE scan |
| `dev/nle-perlmutter` (+10) | `sbi/ridge.py` ridge-lattice scan, M5 closure driver | weighted anything |
| `dev/nle-fidelity` (+19) | `ptbins` (also in weighted) | both of the above |

Merge `dev/nle-weighted` into `dev/nle-perlmutter`. Expected conflicts are small: ridge touched
`scan.py`/`fit.py`/`toy/closure.py`/`ridge.py`; weighted touched `flow.py`/`toy/train.py`/
`toy/datasets.py`/`jewel/*`. `FINDINGS_*.md` are separate files. Run full `tests/sbi` after.

## Phase 1 — New backend config + probe (≪ 1 node-h)

**New config `configs/jewel-2.6.0-cs-v2.toml`** (copy of v1, new `backend` id):

| field | v1 | v2 |
|---|---|---|
| `jewel.ptmin_GeV` / `ptmax_GeV` | 50 / 400 | **10 / 600** |
| `jewel.weighted` | false | **true** (WEXPO = 5 via `plan --wexpo 5`) |
| `selection.jet_pt_range_GeV` | [60, 200] | **[20, 500]** |
| `selection.jet_abs_eta_max` | 1.0 | **2.0** |
| `selection.constituent_abs_eta_max` | 2.5 | 2.5 (unchanged) |
| `design` train points | 12×9×9 Cartesian = 972 | **500 scrambled Sobol** |

`ptmin_GeV` is config-only (`jewel/cards.py:95`); `--extra` cannot override it safely, so a new
config file is the correct and only clean route. No code change needed for PTMIN/PTMAX.

**Code change 1 — train design.** `jewel/generate.py::design_points`: add `"train"` to the
Sobol branch, reading `cfg["design"]["train_points"]`. `make_design(n, split, cfg, box, margin)`
is already imported (`generate.py:52`) and already enforces the 0.02 reservation margin around
the holdout settings. Test the train-Sobol design and the margin.

**Code change 2 — inclusive jets.** `jewel/jets.py`: `reco.cluster_events` already returns
`njets`, `membership` and per-jet `jet_pt/jet_eta/jet_phi`, and `_leading` merely selects
`membership == 0`. Generalize `event_jet` to return a **list** of records, one per jet passing
the selection, each with its own φ rotation (`r["jet_phi"][j]`) and its own diagnostics; apply
the `max_constituents` rejection per jet rather than dropping the whole event. In
`generate.py`, repeat `event_id` per jet so the per-event grouping survives into the shards,
and let `jets_selected` count jets rather than events. Add a `selection.jet_multiplicity`
config switch (`inclusive` | `leading` | `subleading`, default `inclusive`) so the alternative
samples are a config change, not a code change. Tests: inclusive returns ≥ the leading-only
record set; `event_id` grouping is correct; `leading` reproduces the current behaviour exactly.

**Probe** (`generate plan --split probe --theta ... --wexpo 5 --config <v2>`), 4 θ spanning
central/peripheral × low/high T_ref, ~5 k events each. Measures, and these go in FINDINGS:
- s/event and **jets/event** at PTMIN = 10 — now ≥ 1 with inclusive selection, and the main
  cost-per-jet driver. Also record the **distribution** of jets/event: it sets how much
  within-event correlation the composite likelihood carries into Phase 6;
- **max `n_constituents`** — the shard schema caps at 256; a 500 GeV central PbPb jet with
  recoils may exceed it. If it does, raise the cap before production (schema change);
- realized weight range after per-point normalization, and **ESS per jet-pT bin** across 20–500;
- jet-pT spectrum shape, to confirm WEXPO = 5 populates 20–500 roughly flat per decade.

**Output: a costed production sizing, recorded in FINDINGS, then production launches
immediately** — the user has pre-approved the budget for this plan and is unavailable; rule 8
is satisfied by that standing approval plus full node-hour logging, to be revisited later. The
probe still gates production on *technical* grounds: if it shows the 256-constituent cap
overflows or s/event is far off, fix that first rather than burning the production.

## Phase 2 — Production (after budget approval)

`$JEST_NLE_ROOT/data/jewel_v2/` with splits:
- **train**: 500 Sobol points, jets/point sized from the probe;
- **validation + fidelity**: 128 Sobol points (existing `design_points` path, unchanged);
- **test**: the 4 reserved holdout θ — development `[7.5, 0.35, 0.85]`, primary
  `[4.5, 0.45, 0.65]`, plus `[1.5, 0.65, 0.35]` and `[9.5, 0.25, 0.45]` — with **≥ 10
  independent ensembles each**, via `plan --split probe --theta` (own NJOB stream) or a new
  `test` split using `njob_base.test = 500000`, already reserved in the config.

Then `generate index` + `generate merge` per split. **Lesson from the w1 incident** (2 of 12
nodes timed out, 256 stale claim locks blocked the `afterok` chain): end every farm submission
with an `unlock-stale` + short mop-up pass as a standard step, not only on failure.

## Phase 3 — Train the surrogate, optimize for fidelity

`SetFlowVelocity` d256 L6 (beat PET on the toy), `--weight-mode loss` (beat `sample` on the
toy), weighted validation on the unbiased split, `p(N|θ)` trained on JEWEL counts and attached.

**The established lever is training length, not data or capacity** — on the toy, 400k steps
clearly beat 200k across 4 seeds, while 2× data (m3a), weighting, d384 L8 and PET all landed
inside the seed spread. So: 400k steps, ≥ 2 seeds to measure the spread, and verify that
length-scaling holds for JEWEL early (200k vs 400k on one seed) rather than assuming transfer.

**Known trap, already hit twice:** the feature transform is fitted unweighted on each run's own
training split, so **validation loss is not comparable across differently-biased samples**.
Fidelity is judged by the gate and `ptbins`, never by val loss.

## Phase 4 — Fidelity, including at unseen θ

- **Gate** (`toy/gate.py`, already takes `--config` and handles a JEWEL selection) at the 128
  fidelity points: family-wise permutation p-value per point, relevance at the analysis N,
  FAIL = significant **and** relevant; redraw variant.
- **`toy/ptbins.py`** across the full 20–500 GeV range (extend `PT_EDGES`, currently
  `[60…200]`): per-bin effect size Δ/σ, pull, spread ratio, spectrum ratio.
- **At the withheld θ**: the same checks at the 4 reserved points the surrogate never saw —
  this is the step-5 test.

## Phase 5 — NLE on JEWEL

`fit.py`, `likelihood.py`, `scan.py`, `ridge.py`, `flow.py`, `observations.py` have **zero toy
imports** — the NLE core is already generator-agnostic and is reused unchanged, including the
ridge-lattice scan.

New work, `src/omnilearned/sbi/jewel/closure.py`, mirroring `toy/closure.py` but JEWEL-shaped:
- `prepare`: build observation files by **drawing N jets from the generated test shards** at a
  θ (the toy version calls the simulator on the fly; JEWEL cannot). Truth manifest written to a
  path fits never read, opened only by the report after the seal verifies (rule 3).
- acceptance table **Pr_q(S|θ)** over the θ box using the JEWEL selection, as `acceptance_part`/
  `acceptance_merge` do for the toy, consumed by `TableLogNorm`.
- `report`: pulls, Clopper–Pearson 68/95 % coverage, joint level of truth, KS of levels.
- Reuse `toy/numerics.py` for the M4-equivalent convergence check (it only imports `PARAMETERS`
  and `host_info` from toy — parameterize those).

Rule 6 holds: **the numerics check must pass before any fit.**

## Phase 6 — Test the NLE at withheld θ

Fits at the 4 reserved holdout θ, ≥ 10 ensembles each, sealed before truth is opened. Report
pulls, coverage, joint levels — i.e. JEWEL-generated jets in, θ out, compared against the
JEWEL θ that produced them. Include the **surrogate self-closure control** (ensembles drawn
from the surrogate itself) as a labelled diagnostic, never as closure evidence (rule 1).

Two checks that must happen before the coverage numbers are trusted:
- **Lattice resolution.** The open M5 criterion-3 τ_init question was plausibly an artifact of
  the cross-axis spacing (h = 1.5σ). Run the **single-fit nested check** (finer Δs/h on one
  JEWEL fit) — it validates the scan, not the toy.
- **Composite-likelihood correlation.** Because jets are inclusive, under-coverage is the
  expected failure mode if within-event correlation is material. Diagnose by refitting the same
  ensembles with one jet per event: if coverage is correct there but too narrow with all jets,
  the correlation is real and an effective-N / sandwich correction is applied (and recorded)
  rather than silently reporting narrow intervals.

## Execution model — autonomy, parallelism, compute

**Autonomy (user instruction):** do not block on questions, **including the rule-8 production
budget** — it is pre-approved for this plan and revisited later. Every job's node-hours are
still logged in FINDINGS. If something genuinely cannot proceed, record it and move to the next
independent workstream rather than idling.

**Parallelism of the work itself.** Run independent workstreams concurrently as subagents
(spawned at **low/medium effort**), partitioned so they do not touch the same files:
- code: the `jewel/` changes (config v2, Sobol train design, inclusive jets) — one agent,
  sequential internally since `generate.py` is shared;
- `toy/ptbins.py` `PT_EDGES` extension to 20–500 + tests — separate file, parallel;
- `docs/PLAN_JEWEL_NLE.md`, `AGENTS.md` pointer, journal entry — separate files, parallel.
Where two streams would touch one file, use a git worktree per stream and merge, rather than
serializing everything.

**Compute: saturate what the queues will give.**
- **Chained interactive allocations** are the lever that worked for M5: the 4 h cap is not a
  limit if a driver re-requests back to back. Reuse the `shell/nle/m5_drive.sh` pattern
  (`nohup` driver, idempotent phases, resume from cache, stops on `DONE`/`STOP`/two stalled
  chunks) for JEWEL training and fits. Up to **2 interactive nodes per user** — keep both busy.
- **Full nodes, all 4 GPUs.** Every GPU allocation is `--gpus 4` on a whole node, with 4
  concurrent workers (one per GPU), as `jewel_train_chunk.sh` and the M5 fit workers already
  do. Never hold a node at 1 GPU. Use `--gpus 4` even for 1–3 GPU steps so Slurm does not
  renumber devices.
- **CPU production in parallel with GPU work.** JEWEL generation is a CPU array job
  (`jewel_farm.sbatch`, 128 workers/node); it runs in `regular` while GPU training of the
  previous stage runs interactively. Request a wide array — the w1 run used 12 nodes and
  finished the real work in ~2 h.
- **Chain with `--dependency=afterok`**, but end every multi-node farm with an
  `unlock-stale` + mop-up pass: the w1 incident showed two `TIMEOUT` array elements leave stale
  claim locks and `afterok` then blocks the whole chain.
- Mix queues: `regular`/`shared` for long CPU and batch GPU, interactive for anything needing
  fast turnaround or chaining.

**First ~10 h, unattended, in dependency order** (each step starts as soon as its input exists;
nothing waits on me):

| # | step | where | ~wall |
|---|---|---|---|
| 1 | branch `dev/nle-jewel`, merge weighted, full `tests/sbi` | login | 0.5 h |
| 2 | config v2 + Sobol train design + inclusive jets + tests (parallel agents) | login | 1.5–2 h |
| 3 | docs + journal + reports sync (parallel with 2) | login | — |
| 4 | **probe** at 4 θ, measures acceptance / n_const / ESS | CPU batch | ~0.5 h |
| 5 | size production from the probe, launch | CPU array, wide | 2–4 h |
| 6 | `index` + `merge` per split (`afterok`) | 1 CPU node | ~0.5 h |
| 7 | surrogate training, 4 runs on 4 GPUs (200k vs 400k + seeds) | GPU, chained | 2.5 h/chunk |
| 8 | gate + `ptbins` at fidelity points as soon as a checkpoint exists | GPU | 1 h |

Steps 7–8 continue past the 10 h window under a chaining driver. Phases 5–6 (JEWEL closure
driver, numerics, fits at withheld θ) follow once a checkpoint passes the gate.

## Verification

1. **Tests:** full `tests/sbi` green after the Phase-0 merge, and again after the
   train-Sobol design change and the new `jewel/closure.py` (AGENTS.md requires a test for
   every new driver/format/numerical method).
2. **Generator:** probe numbers (acceptance, ESS/bin, max `n_constituents`) recorded in
   FINDINGS before production; merged-index `n_jets`/`events`/`sampling_bias` checked; per-point
   Σw = n_jets confirmed.
3. **Surrogate:** gate at 128 fidelity points + `ptbins` over 20–500 GeV, with the seed spread
   measured so single-seed numbers are never over-read.
4. **Numerics:** Δlog L convergence in steps and probes at the N actually used; CUDA vs CPU.
5. **End-to-end:** sealed fits at the 4 withheld θ recover truth with correct coverage.
6. **Inclusive-jet plumbing:** `leading` mode reproduces today's records bit-for-bit;
   `event_id` grouping in the merged shards matches the per-event jet counts in the manifests.
7. Every result records source revision, dirty status and device. **Reports repo updated at
   the end of every phase**: `tools/sync.py --push` plus a journal entry, and a figure-bearing
   status page (`index.html` with embedded plots, `source/make_figs.py`, `meta.json`, as in
   `reports/m5-closure/2026-10-28T2017-0700_toy-v2_ridge-scan-closure/`) once Phase 4 and
   Phase 6 produce results.

## Explicitly not in scope

Toy generation, toy training, toy closure, the toy τ_init and `n_constituents` questions;
refining the selection on the subleading/leading pT ratio (keep the hooks, defer the study);
the factorized-jet-kinematics fork (PLAN §9.1) unless Phase-4 `ptbins` shows the weighted loss
starves the high-pT end.

## Open risks, tracked

| risk | when it surfaces | response |
|---|---|---|
| 256-constituent cap too small for 500 GeV central PbPb jets | Phase 1 probe | raise the cap before production (schema change) |
| within-event jet correlation makes intervals too narrow | Phase 6 coverage | effective-N / sandwich correction, or one jet/event at inference |
| toy's "training length is the only lever" does not transfer to JEWEL | Phase 3 (200k vs 400k) | re-open capacity/data levers for JEWEL specifically |
| 20 GeV R=0.4 jets in central PbPb are background-fluctuation dominated | Phase 4 `ptbins` low bins | physically real and what an experiment sees; surfaces as a fidelity floor, not a bug — report, do not silently drop the bin |
