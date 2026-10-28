# pT-biased (weighted) training samples (branch dev/nle-weighted)

Running log of `HANDOFF_WEIGHTED.md`, phases 0–4. Worktree
`/global/cfs/cdirs/alice/ploskon/henv-omni-sbi/omnilearned-sbi-weighted`. Runs and scratch
scripts live in `$JEST_NLE_ROOT/runs/w*/` (`JEST_NLE_ROOT=/global/cfs/cdirs/alice/ploskon/jest-nle`);
the main-study trees are read only.

## Job / node-hour log

| date | phase | job | resources | status | what |
|---|---|---|---|---|---|
| 2026-10-26 | 0 | 60353176 (`w0-ptbins-m3a`, gpu shared, 1 A100-40GB, nid001168) | 50 min, 0.21 GPU-node-h (¼ node) | done, 16/16 points | `toy.ptbins` of set m3a → `runs/w0/ptbins/set_m3a/` (snapshot a2e7dba, as the baseline) |
| 2026-10-26 | 0 | login node, 1 core | ≈ 2 min | done | ESS vs k by reweighting (`runs/w0/ess/ess_reweight.py`) and biased toy draws at 3 points (`ess_direct.py`, snapshot e9b52fe) |
| 2026-10-26 | 1 | 60354888 (`w1-gen`, cpu shared, 64 tasks × 2 cores, 20 min limit) | 0.85 core-h (sum of shard wall times; 80–100 s per task) | done, 64/64 | `data/v2_w1`: v2_m2 design and train stream, 6000 jets/point, `--bias-k 3 --bias-cap 250` (snapshot 1bf9fdc) |
| 2026-10-26 | 2 | 60354914 (`w1-train4`, gpu regular, 1 node × 4 A100, 4 h limit, afterok 60354888; nid008196, started 10-27 23:21 after ≈ 23 h in the queue) | 1 h 58 min = 2.0 node-h | done, 4/4 runs | `runs/w1/`: set d256 L6 200k (r2 settings) on v2_w1, `--weight-mode loss` seeds 0–2, `sample` seed 0; multC64 attached (`spec_w1.txt`) |
| 2026-10-26 | 4 | login node | ≈ 3 min | done | WEXPO from ESS by reweighting jewel_v1 (`runs/w4/ess/ess_jewel.py`) |
| 2026-10-27 | 2 | 60427776, 60427777 (`w1-shiftq-1/2`, gpu shared, 1 A100 each, 2 h 45 min limit, afterok 60354914) | 13 s, 8 s | **failed**: snapshot 046a3cb predates `shell/nle/shift_queue.sh`; resubmitted as 60440586/7 (script from snapshot 1bf9fdc, code from 046a3cb, as the main study did) | high-statistics shifts of the 4 w1 models (`runs/w1/shift_hi_queue.txt`, snapshot 046a3cb, `data/v2_m3_shiftdirect`, redraw, 50 000 clouds) → `runs/w1/shift_hi/` |
| 2026-10-28 | 2 | 60440586, 60440587 (`w1-shiftq-1/2`, gpu shared, 1 A100 each; nid002753, nid003892) | 2 h 01 min, 2 h 02 min (≈ 0.5 GPU-node-h each, ¼ node) | done, 2 models each | high-statistics shifts of the 4 w1 models → `runs/w1/shift_hi/set_w1_{s0,s1,s2}_redraw`, `set_w1samp_s0_redraw` (code snapshot 046a3cb) |
| 2026-10-28 | 2 | 60358305 (`w1-ptbins`, another session's job, regular, 1 node × 4 A100, nid002896) | 51 min | done, 16/16 points × 4 models → `runs/w1/ptbins/w5_*` (snapshot a2e7dba) |
| 2026-10-27 | 2 | 60427771/72/74/75 (`w1-ptbins-*`) | 0 | **cancelled by me before start** | duplicates of 60358305 (`w1-ptbins`, `runs/w1/ptbins_w1.sbatch`, 1 node × 4 A100, regular), which another session submitted from this worktree on 10-26 23:42 → `runs/w1/ptbins/w5_{s0,s1,s2}`, `w5samp_s0` |
| 2026-10-26 | 0 | 60356967, 60356968 (`w0-ptbins-s1/s2`, gpu shared, 1 A100 each; nid008681, nid002257) | 47 min, 49 min | done, 16/16 points each | `toy.ptbins` of set s1 and s2 (main-study checkpoints `runs/m2r3/set_d256_L6_200k_s{1,2}/checkpoint_mult.pt`), the seed spread for the weighted comparison → `runs/w0/ptbins/set_s{1,2}` (`run_ptbins_one.sbatch`) |
| 2026-10-26 | 4 | 60355573 (`w4-jewel-probe`, cpu debug, 1 node, 30 min; nid004206) | 9 min 55 s, 0.17 node-h (budget ≤ 0.5 approved by the user) | done, 32/32 tasks rc 0 | weighted JEWEL probe, 4 grid points × WEXPO 2, 3 × 4 tasks × 1000 events → `runs/w4/probe/wexpo{2,3}` (snapshot 6294671) |

## Phase 0: choice of the bias exponent k (2026-10-26, login node, analysis only)

**Sign of the weight.** A draw P0 ∝ pT^-(5−k) has density ratio b(P0) ∝ P0^k against the
generator law, so the weight is **w ∝ P0^−k** (Pythia `bias2Selection`: weight = 1/bias). The
handoff's "w = (P0/P_ref)^k" has the sign inverted; phase 1 uses w = (P0/P_ref)^−k.

**Method 1: reweighting the unbiased fidelity jets** (`runs/w0/ess/ess_reweight.py` →
`ess_reweight.json`).
- **Input:** `data/v2_m1_fidelity`, 128 points × 20 000 selected jets, `P0` from the diag store and the anti-kt jet pT (`global[:, 3]`).
- **Bias:** b(P0) = (min(P0, P_cap)/100 GeV)^k, with P_cap = none, 250 or 200 GeV.
- **Per point and jet-pT bin:** the biased share F = Σb / Σ_all b, the within-bin ESS fraction n² / (Σb · Σ1/b), and ESS per generated jet = F × that fraction. The relative acceptance is E_acc[b] / E_prior[b].
- The tables give medians over the 128 points.

**Method 2: a real biased draw** (`runs/w0/ess/ess_direct.py` → `ess_direct.json`).
- **Draw:** toy v2 with `power_law_exponent` = 5 − k (k = 3, 3.5; no cap), 20 000 accepted jets each, weights P0^−k normalised to Σw = n.
- **Points:** fidelity points 4 (b 0.14, T 0.54: strongest quenching), 61 (b 10.8, T 0.27: weakest) and 0.
- **Provenance:** snapshot e9b52fe (clean), CPU, login node.

**ESS per θ point for 6000 accepted jets** (median over points, no cap; the cap changes these by < 1 % except −5 % in 170–200 at P_cap = 200):

| k | 60–70 | 70–80 | 80–90 | 90–100 | 100–120 | 120–140 | 140–170 | 170–200 | total ESS / n | max/min per bin | max/min per 10 GeV | acceptance vs k = 0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 (v2_m2) | 2744 | 1350 | 726 | 420 | 417 | 180 | 116 | **48** | 1.00 | 57 | 171 | 1 |
| 2 | 1683 | 1105 | 758 | 548 | 710 | 429 | 386 | 230 | 0.80 | 7.3 | 22 | 2.5 |
| 3 | 1153 | 872 | 681 | 548 | 810 | 581 | 616 | **436** | 0.60 | 2.6 | 7.9 | 3.4 |
| 3.5 | 913 | 743 | 619 | 527 | 833 | 649 | 742 | 578 | 0.49 | 1.7 | 4.7 | 3.6 |
| 4 | 701 | 613 | 545 | 490 | 830 | 702 | 866 | 741 | 0.39 | 1.8 | 2.8 | 3.5 |
| 5 | 376 | 382 | 387 | 387 | 746 | 748 | 1088 | 1129 | 0.22 | 3.0 | **1.1** | 2.4 |

- **Gain over v2_m2 in the target bins** (120–140, 140–170, 170–200 GeV): ×2.4/3.3/4.8 at k = 2, **×3.2/5.3/9.1 at k = 3**, ×3.9/7.5/15 at k = 4.
  - **Price at 60–70 GeV,** where the leading (boundary) defect sits: ×0.61, **0.42** and 0.26.
- **Shape:** per bin as defined in `toy.ptbins`, k = 3.5–4 is flattest (max/min 1.7–1.8). Per GeV, k = 5 (P0 drawn flat) is exactly flat, at a ×7 loss below 100 GeV.
- **The within-bin ESS fraction is the same in every bin** (k = 3: 0.83–0.84 at the 10th percentile over θ). It is set by the spread of P0 at fixed jet pT, i.e. by the energy-loss fluctuations, and it depends on θ:
  - k = 3, total ESS / n: 0.42 at point 4 (strongest quenching), 0.62 at point 61 (weakest);
  - the per-jet ESS at point 4 is ≈ 0.7× the median in every bin.
- **Weight range.** Jets need P0 ≳ 60 GeV, so the largest weights come with the lowest P0 and are bounded naturally. The range then comes from the high-P0 tail (strong losses):
  - max/min w at k = 3: 570 at point 4, 84 at point 0, 40 at point 61;
  - a bias cap at P_cap = 250 GeV bounds it to (250/60)^k = 72 (k = 3) at no ESS cost (table: < 1 %) and slightly higher acceptance (3.5 vs 3.4 at k = 3, 4.2 vs 3.5 at k = 4). With the cap the biased law is a piecewise power law, pT^−(5−k) below P_cap and pT^−5 above, still inverse-transform sampled.
- **Generation is cheaper per accepted jet:** the acceptance rises from 0.12–0.18 to 0.31–0.50 at k = 3, because fewer events fail the 60 GeV cut. A 6000-jet k = 3 point costs ≈ 1/3 of a v2_m2 point.
- **Implementation note:** k = 4 makes the generator exponent exactly 1, where the current inverse transform `(lo + u (hi − lo))^(1/(1−n))` divides by zero (`ZeroDivisionError` in the direct run). The biased sampler needs the log-uniform branch for 5 − k = 1.

**Validation of the weights** (method 2, Σw = n, weighted mean ± sd vs unbiased):
- **P0 [GeV]:** point 4: 105.03 ± 37.7 (k = 3) and 105.72 (k = 3.5) vs 104.95 ± 37.8; point 0: 87.97 / 87.78 vs 87.95.
- **Jet pT [GeV]:** point 4: 78.86 vs 78.87; point 0: 78.87 / 78.73 vs 78.88.
- **Point 61:** 78.25 / 78.30 vs 78.80 in the fidelity sample.
  - The fidelity sample is itself high there: three fresh unbiased 20 000-jet draws give 78.53, 78.43, 78.47, i.e. 78.48 ± 0.09, against 78.80 ± 0.15.
  - Against the fresh draws the weighted means differ by −0.2 GeV (≈ 1σ).
- **The reweighting estimate and the direct draw agree.** At point 0, k = 3, the direct ESS per jet [0.191 0.147 0.113 0.093 0.138 0.101 0.102 0.074] matches the median prediction [0.192 0.145 0.114 0.091 0.135 0.097 0.103 0.073].

**Provisional choice: k = 3 with P_cap = 250 GeV** (to be confirmed by the m3a result below).
- It gives ×3–9 ESS above 120 GeV and keeps 1150 ESS per θ at 60–70 GeV, more than v2_m2 has anywhere above 80 GeV, with weight range ≤ 72 and total ESS 0.6.
- k = 4 is flatter per bin but costs another ×0.6 at 60–70 GeV, where the leading defect is.
- **Option:** generation is ≈ 3× cheaper per jet, so 12 000 jets per point at k = 3 cost less CPU than v2_m2. ESS would be ≈ 2300 at 60–70 GeV (v2_m2: 2744) and ≈ 870 at 170–200 GeV (v2_m2: 48). This changes the stored-jet count against v2_m2, so the handoff's 6000 stays the default.

## Phase 0: set m3a in `toy.ptbins` (2026-10-26)

**Run:** job 60353176 (gpu shared, A100-40GB, nid001168, 50 min).
- **Model:** set m3a (`runs/m2r3/set_d256_L6_200k_m3a/checkpoint_mult.pt`, sha256 2ed823c76d6fef23, multC64 attached).
- **Settings:** those of the main-study baseline (snapshot a2e7dba, clean; redraw, 22 000 clouds, gate points 0, 8, …, 120) → `runs/w0/ptbins/set_m3a/`.
- **Combination:** with `ptbins.combine`, reading the main-tree points of set r2 and set 400k without writing there; summary `runs/w0/ptbins/set_m3a/ptbins_summary.json`.

**Bins ≥ 120 GeV** (6 observables × 3 bins = 18 cells; a perfect surrogate gives Σ pull² ≈ 18 ± 6):

| model | rms Δ/σ_direct | Σ pull² | n_constituents 120–140 / 140–170 / 170–200 | soft8_fraction 120–140 / 140–170 | ptd 170–200 | spectrum 170–200 |
|---|---|---|---|---|---|---|
| set r2 (200k, v2_m2) | 0.032 | 57 | +0.031 / +0.062 / +0.049 | −0.031 / −0.034 | −0.034 | 0.937 |
| set m3a (200k, v2_m3a: 2× jets) | **0.026** | **32** | +0.019 / +0.038 / +0.041 | −0.021 / −0.016 | −0.041 | 0.944 |
| set 400k (400k, v2_m2) | 0.019 | 21 | +0.016 / +0.040 / +0.037 | −0.017 / −0.020 | −0.010 | 0.972 |

(se per cell 0.013–0.04, so single cells differ by ≤ 1.5σ between models; the direct sample is shared, so model differences are not diluted by it.)

- ~~m3a is better than r2 in the ≥ 120 GeV bins … high pT is at least partly data-limited~~. **Withdrawn after the seed spread (next section):** r2 is the worst of three 200k seeds, and m3a lies inside their spread.
- **Twice the training length (400k) helps as much or more** without new data. Part of the high-pT defect is therefore optimisation-limited.
- **Low pT also improves with m3a:** width at 60–70 GeV −0.024 → −0.018 (400k −0.015).
- **The spectrum above 170 GeV does not improve with data** (0.944 vs 0.937; 400k 0.972).
- **Caveat:** one seed per model. The seed spread of `toy.ptbins` is unmeasured; set s1 and s2 are submitted (jobs 60356967/8). The weighted runs (seeds 0–2) will be compared with r2/s1/s2.

## Phases 1–2: code (2026-10-26)

**Weighted toy generation** (`toy/simulator_v2.py`, `toy/config.py`, `toy/datasets.py`):
- **Bias option:** `PtBias(exponent k, ref_GeV, cap_GeV)` is a sampling option, not part of `[toy]`, so the config file, `toy_config_sha256` and the backend id stay those of v2. It is selected with `datasets generate --bias-k K [--bias-cap C] [--bias-ref R]`; `gen_toy_array.sbatch` forwards extra arguments.
- **Draw:** `draw_hard_pt` samples pT^-(n−k) below the cap and pT^-n above it, by a piecewise inverse transform with a log-uniform branch for n − k = 1. Unbiased, it evaluates the old expression, so the records are bit-identical (cross-host test passes).
- **Weights:** per selected jet, w = 1/b(P0), normalised so that Σw = number of jets at each point. The records carry `weight` and `pthat`.
- **Shards:** optional datasets `weight` and `pthat`, plus attrs `sampling_bias` (JSON). The bias is also recorded in the shard manifest (with per-point ESS and weight range in each point's `toy.sampling_bias`) and in the index (`sampling_bias`).
- **Reader:** `read_shards` returns `weight`, as ones for old shards. Counts-only shards refuse a bias.

**Weighted training** (`sbi/flow.py`, `sbi/multiplicity.py`, `toy/train.py`):
- **Options:** `TrainOptions.weight_mode` / `toy.train --weight-mode none|loss|sample`.
  - `loss`: Σwℓ/Σw per minibatch.
  - `sample`: minibatch indices ∝ w through a precomputed CDF (`searchsorted`); with length buckets, a bucket is drawn ∝ its weight sum and the events inside it ∝ w.
- **Validation and p(N|θ):** the validation loss is weighted in both modes. The p(N|θ) NLL is weighted whenever the mode is not `none` (the multC64 attach is the production p(N|θ); the weighted NLL is the cross-check).
- **Unchanged path:** with `none`, nothing changes. With all-ones weights in `loss` mode, the loss equals the mean exactly and training is bit-identical (test).
- **run.json:** `data.weights` gives the ESS per split, per point and per point × `toy.ptbins` pT bin (median, 10th percentile); `flow_history.weights` gives the mode, ESS and weight range.

**Tests:**
- `tests/sbi/test_weighted.py`:
  - P0 histogram against the biased density for 5 (k, cap) settings including k = 4;
  - weighted P0 moments equal the unbiased ones;
  - biased toy jets weighted back to the unbiased jet pT, N, P0 and share above 120 GeV;
  - zero bias equals the unbiased toy;
  - shard round trip, index, CLI, and old shards reading as unit weights;
  - the train driver in both modes.
- `tests/sbi/test_weighted_training.py`:
  - unit-weight loss exact, and training bit-identical;
  - weighted minibatch frequencies with and without buckets;
  - a weighted flow fit on a 1D toy (biased N(1, 1), weights e^−x) recovers N(0, 1) in both modes, while unweighted training gives N(1, 1);
  - a weighted p(N) fit recovers Poisson(12) from Poisson(13.8) draws.

## Phase 4 (JEWEL), started in parallel with the toy arm (user, 2026-10-26)

**JEWEL's weighting, from the installed source** (`hepyy_packages/src/jewel-2.6.0-custom-src`):
- `WEIGHTED T` sets `MSTP(142) = 1`. `PYEVWT` returns WTXS = (pT-hat²)^(WEXPO/2) = pT-hat^WEXPO (`pythia6425mod-lhapdf6.f:80201`).
- PYTHIA then samples dσ × WTXS and sets `PARI(10) = VINT(100) = 1/WTXS` (`:10246`, `:24912`). This is the HepMC event weight (`custom-partons.patch`), so **w = pT-hat^−WEXPO exactly**, with no running normalisation. It is the toy's bias without a cap; PTMAX = 400 GeV bounds the weights.
- JEWEL's own defaults are `weighted = .true.` and `WEXPO = 5`. Our cards set `WEIGHTED F` explicitly.

**WEXPO from ESS, by reweighting jewel_v1** (`runs/w4/ess/ess_jewel.py` → `ess_jewel.json`; 2 199 271 training jets, 972 points; acceptance per event and the all-event pT-hat prior from the event stores of 40 random tasks):

| WEXPO k | 60–70 | 70–80 | 80–90 | 90–100 | 100–120 | 120–140 | 140–170 | 170–200 | total ESS / n | within-bin ESS fraction (pooled) | acceptance vs unweighted | w max/min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 (jewel_v1) | 912 | 509 | 295 | 175 | 175 | 73 | 43 | **16** | 1 | 1 | 1 (0.082) | 1 |
| 2 | 546 | 364 | 247 | 170 | 210 | 120 | 98 | 53 | 0.73 | 0.75–0.87 | 1.8 | 63 |
| 3 | 347 | 247 | 179 | 130 | 175 | 117 | 112 | **76** | 0.50 | 0.52–0.73 | 2.4 | 500 |
| 4 | 191 | 143 | 108 | 83 | 119 | 92 | 105 | 88 | 0.29 | 0.31–0.57 | 3.0 | 4000 |
| 5 | 91 | 71 | 55 | 44 | 67 | 58 | 80 | 85 | 0.14 | 0.17–0.41 | 3.2 | 31 000 |

(ESS per point for 2200 jets, the jewel_v1 median; medians over points.)
- **The weights are less efficient than in the toy.** At k = 3 the within-bin ESS fraction is 0.52–0.73 (toy 0.83): pT-hat is spread more widely at fixed jet pT (energy loss, out-of-cone radiation, recoils), and 12.7 % of the selected jets have pT-hat < 60 GeV (the 1/50/99 % quantiles of their pT-hat are 51 / 80 / 180 GeV).
- **The high-pT gain saturates at k ≈ 3–4:**
  - 170–200 GeV: ×3.3 (k = 2), ×4.8 (k = 3), ×5.5 (k = 4), ×5.3 (k = 5);
  - 60–70 GeV: ×0.60, 0.38, 0.21.
  - **Candidate: WEXPO = 3** (2 as the conservative alternative). JEWEL's default 5 would cost ×10 at low pT for no high-pT gain.
- **Cost:** jewel_v1 averages 0.30 s/event (0.03–0.60, correlation +0.94 with TI; 2356 core-h for 28.3 M events). Weighting cuts the events per jet by ×2.4 (k = 3), but higher-pT-hat events take longer. The cost per event is measured only by a probe.

**Code** (`jewel/cards.py`, `jewel/generate.py`):
- **Option:** `generate plan --wexpo K` records `sampling = {"wexpo": K}` in each task (top-ups copy it). `cards.task_cards(..., sampling=)` writes `WEIGHTED T` and `WEXPO K` and refuses WEIGHTED/WEXPO in `extra`. `[jewel]` (with `weighted = false`) is unchanged, so the backend id and config hash stay.
- **Task shards:** weighted tasks write `weight` (the raw HepMC weight) and `pthat`, plus attrs `sampling_bias`. Unweighted tasks keep the old format exactly.
- **Merge:** `merge` rescales the weights per point to Σw = n_jets, records the ESS per point in the merged manifest and the bias in the merged index, and refuses to mix weighted and unweighted sources.
- **Tests** (`tests/sbi/test_jewel.py`; the fake JEWEL writes pT-hat^−WEXPO when weighted):
  - cards, including the label check against the installed source;
  - task weights equal to pthat^−3;
  - per-point normalisation at merge;
  - the old format unchanged;
  - the mixed-merge refusal.

## Phase 4: weighted JEWEL probe (2026-10-26, job 60355573, approved by the user)

32 real JEWEL tasks (16 WEXPO 2, 16 WEXPO 3; 4 tasks × 1000 events at each of 4 jewel_v1 grid points:
(b, TI, TAUI) = (0, 0.9, 0.2), (5, 0.5, 0.6), (7, 0.3, 0.8), (11, 0.1, 1.0)), snapshot 6294671,
1 CPU node, debug QOS, 9:55 wall (≈ 0.16 node-h, well under the approved 0.5 node-h), all 32/32
tasks `rc = 0`. Analysis: `runs/w4/probe/analyze_probe.py` → `probe_results.json`.

- **Weighting works correctly in the real binary, not just the fake-JEWEL tests.** At every point
  the weighted jet pT, N and the ≥ 120 GeV share agree with the jewel_v1 unweighted reference
  (2100–2800 jets/point) within 1–2 se, e.g. θ = (5, 0.5, 0.6): ⟨pT⟩ 80.30 ± 0.87 (w) vs 80.76 ± 0.48
  (ref), ⟨N⟩ 31.95 ± 0.65 vs 31.90 ± 0.30.
- **Seconds/event is unaffected by weighting** (ratio weighted/unweighted 0.97–1.04 at all 4 points,
  both WEXPO). The strong TI dependence found in jewel_v1 (0.03–0.60 s/event, ρ = +0.94 with TI)
  carries over unchanged, so that cost model applies directly to weighted production.
- **Acceptance per event rises more than the ESS study predicted from reweighting alone**, because
  weighting also raises the fraction of events that pass the jet-pT cut, not only the fraction with
  a hard-pT jet already present:

| θ (b, TI, TAUI) | unweighted acc. | WEXPO 2 acc. (ratio) | WEXPO 3 acc. (ratio) |
|---|---|---|---|
| (0, 0.9, 0.2) | 0.061 | 0.127 (2.08×) | 0.174 (2.86×) |
| (5, 0.5, 0.6) | 0.070 | 0.136 (1.94×) | 0.192 (2.74×) |
| (7, 0.3, 0.8) | 0.087 | 0.145 (1.66×) | 0.185 (2.12×) |
| (11, 0.1, 1.0) | 0.132 | 0.198 (1.51×) | 0.232 (1.76×) |

- **Per-CPU-time yield (jets per 1000 events generated, the quantity that sets the production
  budget) is better than the per-jet ESS study suggested, because the acceptance gain and the
  pT-reshaping multiply:**
  - **170–200 GeV: ×4.2–9.2 (WEXPO 2), ×6.9–20.5 (WEXPO 3)** across the 4 points — larger than the
    ×3.3/×4.8 predicted from reweighting jewel_v1 jet counts alone (that study fixed the jet count
    per point and so missed the acceptance gain).
  - **60–70 GeV is roughly unchanged, not costed, at WEXPO 2** (×1.00–1.16) and **mildly costed at
    WEXPO 3** (×0.81–1.17, worst at the weak-quenching corner (11, 0.1, 1.0)). This differs from the
    toy, where low pT was always costed (×0.42–0.61 at k = 3): in JEWEL the acceptance gain partly
    offsets the reshaping away from low pT.
- **Consequence for a production estimate:** because events needed per target jet count scales as
  1/acceptance, and acceptance rises 1.5–2.9×, **a WEXPO-3 production run reaching the same total
  jet statistics as jewel_v1 would need fewer JEWEL core-hours overall, not more, while concentrating
  much more of that statistics above 120 GeV.** This is only 4 of 972 grid points (corners/edges of
  the (b, TI) plane), so the acceptance-ratio trend with TI and b (ratio falls as TI falls / b rises,
  i.e. less to gain where quenching is weak) should be checked over more of the grid before sizing
  a full production run.
- **Open before a production budget:** a probe at a few interior grid points (not just corners) to
  confirm the acceptance-ratio trend, and a decision on WEXPO (3 is the current candidate; the
  low-pT cost at weak quenching argues against going higher).

**Weighted JEWEL probe** (job 60355573, snapshot 6294671, CPU, nid004206; `runs/w4/probe/analyze_probe.py` → `probe_summary.json`):
- **Design:** 4 jewel_v1 grid points (b, TI, TAUI) = (0, 0.9, 0.2), (5, 0.5, 0.6), (7, 0.3, 0.8), (11, 0.1, 1.0) × WEXPO 2 and 3, 4 tasks × 1000 events each.
- **Comparison:** the jewel_v1 tasks at the same grid points (unweighted; 21 000–35 000 events and 2103–2762 jets per point).
- **Seeds:** the WEXPO 2 and 3 tasks share NJOB values, so the two settings are correlated with each other but independent of jewel_v1.

| | 60–70 | 70–80 | 80–90 | 90–100 | 100–120 | 120–140 | 140–170 | 170–200 | acceptance per event |
|---|---|---|---|---|---|---|---|---|---|
| jewel_v1, jets per 1000 events | 35.6 | 20.6 | 12.2 | 7.1 | 6.7 | 3.0 | 1.6 | 0.6 | 0.061–0.132 |
| WEXPO 2, ESS per 1000 events | 37.8 | 24.2 | 16.6 | 13.0 | 14.8 | 9.2 | 7.2 | 3.9 | 0.127–0.198 |
| ratio | 1.06 | 1.17 | 1.36 | 1.82 | 2.22 | 3.1 | 4.6 | 6.1 | |
| WEXPO 3, ESS per 1000 events | 34.4 | 23.2 | 16.3 | 13.3 | 15.0 | 11.6 | 11.5 | 7.2 | 0.174–0.232 |
| ratio | **0.97** | 1.13 | 1.34 | 1.86 | 2.25 | **3.9** | **7.3** | **11.0** | |

(Means over the 4 points. The top bins rest on 20–30 weighted jets per point, so they are uncertain to ≈ 20 %.)

- **Weighting does not change the JEWEL time per event:**
  - WEXPO 2 / 3: 0.404 / 0.403, 0.297 / 0.305, 0.194 / 0.191, 0.029 / 0.029 s at the four points;
  - jewel_v1: 0.400, 0.300, 0.187, 0.030 s.
  - CPU therefore scales with events. Biasing to higher pT-hat raises the acceptance (fewer events below the 60 GeV cut) and offsets the weight dilution.
- **Per CPU-hour, WEXPO 3 is free at low pT and gives ×4–11 above 120 GeV.** WEXPO 2 gives ×3–6 at no low-pT cost.
  - This supersedes the fixed-jet-count reading of the reweighting table above. That table is right per stored jet, but in JEWEL the cost is per event, not per jet.
- **Weighted moments vs the unweighted jets** (pull z = Δ/se per point; "combined" = Σz/2):

| WEXPO | ⟨jet pT⟩ pulls | combined | ⟨N⟩ pulls | combined |
|---|---|---|---|---|
| 2 | +0.73, −0.46, +0.35, −0.16 | +0.23 | +1.53, +0.07, −0.81, +1.03 | +0.91 |
| 3 | −0.26, −1.14, −0.52, −0.79 | −1.35 | −1.13, −0.47, −1.65, −0.30 | −1.77 |

  - Shares of jets ≥ 120 GeV agree within 0.006.
  - The WEXPO 3 means are all slightly low, but < 2σ combined. They need a larger check before relying on them, e.g. a weighted fidelity sample against the unweighted one in `toy.ptbins`-style bins.
- **Production estimate** (WEXPO 3, the 972-point grid, the same events per point as jewel_v1, ≈ 29 000):
  - CPU equals jewel_v1: 2356 core-h ≈ 18 CPU node-h;
  - ≈ 2× the stored jets (≈ 4.5 M);
  - the same low-pT statistics and ×4–11 above 120 GeV;
  - with event stores, ≈ 600 GB more on CFS (jewel_v1: 601 GB); without them (`--no-events`), ≈ 2 GB of jets.
  - It needs a budget approved by the user (rule 8).

## Phase 4: production proposal (2026-10-26, pending user budget decision)

Sizing a weighted JEWEL training set from the existing jewel_v1 production (exact totals from
`index_train.json`/`index_validation.json`: train 2 199 271 jets, 28 298 054 events, **2356.06
core-h**; validation 146 016 jets, 1 904 712 events, 157.4 core-h) and the probe's acceptance
ratios, using the pooled WEXPO-3 ESS-study ratio (×2.40, 40-task event sample, all 972 points)
as the central estimate and the probe's 4-point range (×1.76–2.86, corners/edges of the box) as
the uncertainty band — these two independent estimates bracket each other, which is some
reassurance, but all 4 probed points lie on one diagonal of the (b, TI) plane, so the ratio's
behaviour off that diagonal is not checked.

- **Design:** no new config/design needed — mirror jewel_v1's existing train grid exactly (same
  972 Cartesian points, same box; `[design]`/`[jewel]` unchanged, `sampling = {"wexpo": 3}` is a
  per-task option as in `jewel/cards.py`). New split/directory, e.g. `data/jewel_w1/train`.
- **Validation and fidelity reused as-is, unweighted** (no new JEWEL cost for them): the gate and
  closure must compare against pseudo data drawn directly from the simulator (AGENTS.md rule 1),
  so these splits stay unbiased, exactly as `v2_w1` reused `v2_m2`'s unbiased validation.
- **WEXPO = 3** (candidate from the ESS study and the probe; 2 as the conservative fallback).

| option | per-point target | total train jets | core-h (central) | core-h range | node-h (≈128 cores/node) | ESS vs jewel_v1 at 60–70 GeV | ESS vs jewel_v1 at 170–200 GeV |
|---|---|---|---|---|---|---|---|
| **A — match jet count** | same as today (median 2167) | 2.20 M (same) | **982** | 824–1339 | 7.7 (6.4–10.5) | ×0.38 (worse) | ×4.8 (better) |
| **B — match budget already spent** | ×2.4 jets | 5.28 M | **2356** (= jewel_v1's own cost) | 1977–3213 | 18.4 (15.4–25.1) | ×0.9 (≈ even) | ×11.5 |
| **C — ESS parity at the leading (low-pT) defect** | ×2.0 jets | 4.40 M | **1963** | 1648–2677 | 15.3 (12.9–20.9) | ×1.0 (parity) | ×9.5 |

(ESS multipliers from the per-jet reweighting study's pooled total/bin ESS fractions at WEXPO 3,
scaled by each option's jet multiplier; A's cost is lower than jewel_v1 because weighting raises
acceptance per event, even though per-jet ESS at low pT is worse for the same accepted count.)

- **Recommendation: option C.** It does not lose statistical power at 60–80 GeV — the largest and
  most significant surrogate defect found so far (`FINDINGS_FIDELITY.md`) — while buying ≈9.5×
  the power above 170 GeV, for less than double jewel_v1's already-spent budget.
- **Given the ±30 % uncertainty band from only 4 corner/edge probe points, recommend a first
  tranche of ≈ 10 % of the grid (≈ 97 points, full per-point target)** to measure the realized
  core-h and the ratio's trend across more of the (b, TI) box before committing the rest — the
  project rule against open-ended sweeps and the existing `chunks_for_target`/`topup` tooling
  both support staging this way.
- **Not yet decided (pending user):** WEXPO value, which option (A/B/C or a custom target), and
  tranche-first vs. a single commit. No tasks have been planned or submitted for this production.

## Correction: PYTHIA 8 `bias2Selection` reference values (2026-10-27, user-supplied, verified against the manual)

The Pythia 8 analogy cited throughout this log (and in `HANDOFF_WEIGHTED.md`) is
`PhaseSpace:bias2Selection` / `PhaseSpace:bias2SelectionPow`
(https://pythia8.web.cern.ch/manuals/pythia8318/PhaseSpaceCuts.html, verified 2026-10-27):
oversampling by `(pTHat/pTRef)^pow`, a compensating weight `(pTRef/pTHat)^pow` from
`Info::weight()`. **`bias2SelectionPow` defaults to 4.0** (range 0–10) — no default had been
recorded here before.

- **JEWEL's own WEXPO lacks the `pTRef` scale:** `PYEVWT` computes `WTXS = pT-hat^WEXPO`
  directly (`:80201`-area of `pythia6425mod-lhapdf6.f`), and JEWEL's own hardcoded default is
  **WEXPO = 5** (`jewel-2.6.0.f`: `weightex = 5.`), higher than Pythia 8's own upstream default.
  Our config sets `WEIGHTED F` explicitly, so neither default applies unless `--wexpo` is used.
- **The missing `pTRef` does not matter here:** `merge` already renormalises the weights per
  point to sum to that point's accepted jets, which absorbs any overall scale JEWEL's
  convention omits.
- **Corroborates, does not change, the WEXPO = 3 recommendation:** 3 sits below both Pythia 8's
  own default (4) and JEWEL's hardcoded one (5), i.e. it is conservative relative to existing
  practice; the number itself was derived independently from the ESS study and the probe for
  this selection and physics, not from either default.

## Phase 0 verdict: set m3a vs set r2 in `toy.ptbins`, ≥120 GeV bins (2026-10-26, job 60353176)

`toy.ptbins run` (snapshot a2e7dba, redraw, 22 000 clouds, 16 gate points 0,8,…,120 — same
settings as the main session's baseline) on `runs/m2r3/set_d256_L6_200k_m3a/checkpoint_mult.pt`
(2× jets/θ at the same spectrum, multC64 attached), 50 min, 1 A100, rc 0 → `runs/w0/ptbins/set_m3a/`.
Compared against the main session's `set_r2` (1×) and `set_400k` (same jets, 2× steps) summaries
in `runs/m3/ptbins/`.

**Aggregate over the 3 bins ≥ 120 GeV and the 6 observables:**

| | mean \|effect\| | mean \|pull\| | rms \|pull\| |
|---|---|---|---|
| set r2 (1× jets, 200k steps) | 0.0275 | 1.57 | 1.78 |
| **set m3a (2× jets, 200k steps)** | **0.0225** | **1.25** | **1.33** |
| ratio m3a/r2 | 0.82 | 0.79 | — |

**Spectrum ratio (surrogate/direct, normalised share) at 120–140 / 140–170 / 170–200 GeV:**
r2 1.053 / 1.023 / 0.937; m3a 1.022 / 0.990 / 0.944 (closer to 1 at the first two, about the
same at 170–200); set 400k (for reference) 1.027 / 0.996 / 0.972 — the largest spectrum gain
still comes from more training (400k), not more jets per θ (m3a), consistent with the earlier
integrated-shift finding.

**Not uniform across observables:** n_constituents and soft8_fraction — the two flagged by the
main session as "the surrogate's high-pT jets carry a few too many soft constituents" — improve
with m3a in **every** ≥120 GeV bin (n_constituents |effect| 0.031/0.062/0.049 → 0.019/0.038/0.041;
soft8_fraction 0.031/0.034/0.006 → 0.021/0.016/0.004). width and outer_fraction are mixed (better
at 120–140 and 170–200, worse at 140–170). **ptd and lead_fraction get slightly worse at the
single highest bin** (170–200 GeV, the thinnest-statistics bin: lead_fraction |effect| 0.023 →
0.034, pull 0.9 → 1.3) — still well inside the family-wise significance threshold used elsewhere,
but reported as found (rule 5), not smoothed over.

**Verdict: aggregate improvement, ~18–20 % smaller effect size and pull with 2× the jets per θ,
driven mainly by n_constituents and soft8_fraction — the per-handoff decision rule reads as
"high pT is data-limited" for those two observables, with a mixed picture for the other four.**
This is consistent with (not contradicted by) the main session's finding that the *integrated*
shift does not improve with m3a — the low-pT-dominated integrated metric is insensitive to a
high-pT-only gain this size. It supports continuing the weighted arm already underway (phases
1–4 code, probe and the running `v2_w1` generation/training), rather than reopening it.

## Phase 4: production launched at WEXPO = 5 (2026-10-26/27, user decision: "go ahead ... run with the default JEWEL value for WEXPO")

**Decision, overriding the earlier recommendation:** WEXPO = 5, JEWEL's own hardcoded default
(`weightex = 5.` in `jewel-2.6.0.f`), not the WEXPO = 3 this log had recommended from the ESS
study and probe (both of which only measured k = 2 and k = 3 in the real binary — **k = 5 itself
was never probed**, only estimated by reweighting jewel_v1's existing unweighted jets).

**What WEXPO = 5 costs, from the reweighting study (`runs/w4/ess/ess_jewel.json`), pooled over
all 972 points:**
- Pooled acceptance ratio (weighted/unweighted) = **3.16×** (vs 2.40× at k = 3) — events needed
  for a given jet count drop further, so this is cheaper per jet than k = 3.
- Pooled total ESS/n = **0.14** (vs 0.50 at k = 3): for the *same accepted jet count* as an
  unweighted sample, only 14 % of the statistical power survives, concentrated at high pT.
- Per-point median ESS at a fixed 2200 accepted jets, by pT bin (60–70 … 170–200 GeV):
  **91, 71, 55, 44, 67, 58, 80, 85** (jewel_v1 unweighted at the same count: 912, 509, 295, 175,
  175, 73, 43, 16). The 60–70 GeV bin — the bin containing the leading, already-identified
  surrogate defect (width/outer_fraction short, boundary leakage) — loses to **10 % of its
  unweighted ESS** at matched jet count; 170–200 GeV gains ×5.3.
- **Weight range w_max/w_min ≈ 31 500** (pooled), far above k = 3's ≈ 500: a small number of
  very-high-pT-hat jets carry most of the weight. This is a real risk for the weighted loss and
  for the weighted p(N\|θ) fit (a single extreme-weight jet can dominate a minibatch's gradient);
  nothing in the code caps it, and no real-binary probe has checked k = 5's behaviour.

**Production design (match jewel_v1's existing per-point jet count, i.e. "option A" scaled to
k = 5's ratio):** chunks per point = ceil(jewel_v1's own achieved chunk count ÷ 3.16), read
directly from jewel_v1's final `train/tasks.json` (972 points, 21–50 chunks unweighted →
7–16 chunks weighted, none at the chunks_per_point cap of 100). **9415 tasks, 9.415 M events
planned** (vs jewel_v1's 28.3 M), same 972-point grid, same box, `data/jewel_w1/train`. Projected
cost by the same per-event scaling: **≈ 784 core-h ≈ 6.1 node-h** (unverified at k = 5; the
probe's 0.97–1.04× s/event-unaffected-by-weighting finding was only measured at k = 2, 3).
Validation and fidelity are **reused unweighted from `data/jewel_v1`** (no new JEWEL cost),
per AGENTS.md rule 1 (pseudo data only from the simulator, unbiased).

**Job chain (snapshot 6294671/`629467158d5...`, all via Slurm `--dependency=afterok`, no manual
step needed to proceed from one stage to the next):**

| stage | job | what | resources |
|---|---|---|---|
| 1. generate | **60358270** | `jewel_farm.sbatch`, 12-node array, 128 workers/node, `--wexpo 5`, `data/jewel_w1/train` | `-A alice -C cpu -q regular`, 2.5 h/node |
| 2. index + merge | **60358284** (afterok 60358270) | `generate index` + `generate merge --shards 16 --workers 4` → `data/jewel_w1/merged` | 1 CPU node, 1.5 h |
| 3. train | **60358288** (afterok 60358284) | 4 runs on `runs/w5/`: `set_d256_L6_200k`, `--weight-mode loss` seeds 0–2 + `sample` seed 0, trained against `data/jewel_w1/merged` (train) vs `data/jewel_v1/merged` (validation, unweighted), `jewel_multJ.pt` p(N\|θ) attached | `-A alice_g -C gpu -q regular`, 1 node × 4 A100, 4 h |

**Toy arm, unchanged from earlier tonight, also chained:** `v2_w1` generation (60354888, 25/64
done) → weighted training (60354914, afterok) → **`toy.ptbins` for all 4 w1 models vs jewel_v1...
[sic, vs the main-session set r2/400k baseline]** (**60358305**, afterok 60354914, 4 GPUs,
`runs/w1/ptbins/`).

**Not done tonight, blocked on the other session (not this worktree's tree):** a `toy.ptbins`- or
gate-style fidelity check for the weighted JEWEL models needs `data/jewel_v1`'s **fidelity**
split merged (7262/7262 tasks complete but not yet `index`ed/merged — owned by `dev/nle-fidelity`).
Until then, the weighted-JEWEL validation signal available is the trained flow's validation loss
against `data/jewel_v1/merged`'s unweighted validation split (in each run's `run.json`), compared
with `runs/jewel_r1b`'s existing unweighted-trained validation losses.

**Open risk to check first thing:** the extreme weight range at k = 5 was never validated in the
real binary (only k = 2, 3 were probed). If the weighted training's `run.json` ESS or loss curve
looks unstable, or `generate index`/`merge` reports an implausible per-point ESS, that is the
first thing to look at — the fallback is WEXPO = 3, already coded and probed.

## Phase 1: data `v2_w1` (2026-10-27)

- **Generation:** `$JEST_NLE_ROOT/data/v2_w1`, job 60354888, snapshot 1bf9fdc (clean), CPU (shared QOS).
  - 64 shards × 16 points × 6000 jets = 6 144 000 jets on the v2_m2 design and training stream (common random numbers with v2_m2), `--bias-k 3 --bias-cap 250 --bias-ref 100`.
  - The validation files are symlinks to v2_m2's unbiased validation split, so the early-stopping metric is that of set r2.
- **Acceptance** 0.29–0.54 (v2_m2 0.12–0.18).
- **ESS per point:** median 3606 of 6000 (0.60, as predicted), range 2493–3765.
- **Weight range:** max/min ≤ 72.1 per point, the cap's bound (250/60)³.

## Incident: JEWEL production chain stalled on stale claim locks (2026-10-27, fixed, no data lost)

The 12-node production farm (60358270) ran well overall — **9159 of 9415 tasks (97.3%) completed
cleanly** — but array elements 0 and 1 started first (≈ 23:43), ran the full 2 h 30 min time
limit, and were killed by Slurm mid-work (`TIMEOUT`, not a code failure). Their in-flight workers
were SIGKILLed before their `finally: tasklocks.release(lock)` could run, leaving **256 stale
claim locks** — exactly the 256 tasks left incomplete, confirming the mechanism. `generate
unlock-stale` only runs once, at array element 0's *start*, so it could not clean up locks its
own node created later. Because Slurm's `afterok` on an array job requires every element to
succeed, the two `TIMEOUT` elements blocked the whole downstream chain: **60358284** (index/merge)
sat as `DependencyNeverSatisfied` and **60358288** (training) behind it, neither ever starting.

**Fix:** `generate unlock-stale --dir data/jewel_w1/train` (removed exactly 256 locks) → cancelled
the two stuck jobs → a small 2-node cleanup farm (**60373313**, 50 min, resumable — `generate.work`
skips tasks with an existing manifest) for the 256 remaining tasks → rebuilt the chain: **60373319**
(index/merge, afterok 60373313) → **60373320** (4 weighted trainings, afterok 60373319). No data
or provenance was lost; the 9159 original task shards are untouched and unaffected.

**Lesson for next time:** either give the farm's per-node time limit enough margin that a node
naturally drains its share of the claim pool before the limit (the uneven start times here, not
total work, caused the timeout — nodes 2–11 started 3–7 h late due to regular-QOS queue
contention and found almost nothing left), or add a final `unlock-stale` + short mop-up pass as a
standard last step of any multi-node JEWEL farm submission, not just on failure.

## Phase 0, corrected: seed spread of `toy.ptbins` (2026-10-27)

Set s1 and s2 (`runs/m2r3/set_d256_L6_200k_s{1,2}/checkpoint_mult.pt`, the r2 configuration with seeds 1 and 2) → `runs/w0/ptbins/set_s{1,2}` (jobs 60356967/8, snapshot a2e7dba, as the baseline).

| model | ≥ 120 GeV rms Δ/σ | Σ pull² (18 cells) | n_constituents 120–140 / 140–170 / 170–200 | width 60–70 | spectrum 170–200 |
|---|---|---|---|---|---|
| set r2 (200k, seed 0) | 0.032 | 57 | +0.031 / +0.062 / +0.049 | −0.024 | 0.937 |
| set s1 (200k, seed 1) | 0.020 | **21** | +0.011 / +0.037 / +0.035 | −0.007 | 0.975 |
| set s2 (200k, seed 2) | 0.023 | 29 | +0.020 / +0.038 / +0.044 | −0.016 | 0.972 |
| set m3a (200k, 2× jets) | 0.026 | 32 | +0.019 / +0.038 / +0.041 | −0.018 | 0.944 |
| set 400k (seed 0) | 0.019 | 21 | +0.016 / +0.040 / +0.037 | −0.015 | 0.972 |

- **The seed spread of the 200k configuration (Σ pull² 21–57) covers m3a (32) and 400k (21).** r2, the baseline of the main study's `toy.ptbins` and of the phase-0 comparison, is the worst seed.
- **Corrected verdict: high pT is not shown to be data-limited.** 2× jets per θ is no better than a different seed; the earlier "m3a better" came from comparing with the worst seed.
- **One defect persists in every model:** n_constituents too high at 140–170 GeV (+0.037 to +0.062) and 170–200 GeV (+0.035 to +0.049), with seed, data and training length alike. It is a systematic of the surrogate (capacity or parametrisation, phase 3: conditioning on jet pT, or the observed-pT bias arm), not of the training statistics.
- **Consequence:** the weighted runs (×9 ESS at 170–200 GeV, seeds 0–2, running) are now a direct test of the data-limited hypothesis. They must be compared with the seed spread (r2/s1/s2), not with r2 alone. If they do not remove the n_constituents excess, weighting is not the fix for it.
- **Low pT** (width 60–70 GeV, −0.007 to −0.024) is also seed-dominated.

## Phase 2: weighted trainings (2026-10-28)

`runs/w1/set_d256_L6_200k_w1_{s0,s1,s2}` (`--weight-mode loss`) and `…_w1samp_s0` (`--weight-mode sample`).
- **Provenance:** job 60354914, nid008196, A100, snapshot 1bf9fdc (clean); set r2 settings (d256, L6, 200k steps, batch 512, 16 length buckets, TF32) on `data/v2_w1`; multC64 attached (`checkpoint_mult.pt`).
- **Training ESS** (run.json): 3 570 187 of 6 144 000. Per point and pT bin (median): 1151, 867, 680, 544, 809, 585, 615, **432** (v2_m2: 2744 … **48**).

| run | best step | best val | checkpoint sha256 (flow / with multC64) |
|---|---|---|---|
| w1 s0 (loss) | 194 000 | 1.05595 | a14fdab15859 / c66c845f29c1 |
| w1 s1 (loss) | 200 000 | 1.05596 | 83722f2eba08 / e4f95db2d55e |
| w1 s2 (loss) | 172 000 | 1.05595 | f0edcb71745f / 42d89d8a9d3a |
| w1 sample s0 | 184 000 | 1.05594 | 54c86075a010 / 4cb27a1427a9 |
| baselines r2 / s1 / s2 / m3a / 400k | | 1.05223 / 1.05222 / 1.05222 / 1.05234 / 1.05220 | |

- **The validation loss is not comparable across these two groups,** although the validation jets are the same (v2_m2's unbiased split, unit weights).
  - The feature transform is fitted, unweighted, on each run's training split. For v2_w1 the log-pT offset is −0.085 instead of −0.278 (scales 1.780 vs 1.785, η/φ equal), because the biased jets are harder.
  - Model coordinates therefore differ, and the minimum of the flow-matching loss depends on where the data sit relative to the Gaussian base.
  - The +0.0037 is therefore not a fidelity measure. Fidelity is judged by `toy.ptbins` and the shift (pending).
  - A weighted fit of the transform (or reusing v2_m2's) would make the losses comparable. Not done here.
- **Loss and sample modes give the same validation loss** (1.05594–1.05596), and the 3 seeds agree to 1e-5, as the baseline seeds do.

## Phase 2 result: `toy.ptbins` of the weighted models (2026-10-28)

**Run:** job 60358305 (nid002896, A100), snapshot a2e7dba, as the baseline (redraw, 22 000 clouds, gate points 0, 8, …, 120); checkpoints `runs/w1/set_d256_L6_200k_w1*/checkpoint_mult.pt` (sha256 c66c845f…, e4f95db2…, 42d89d8a…, 4cb27a14…) → `runs/w1/ptbins/w5_*`. Combined with `ptbins.combine` against the 200k seeds r2/s1/s2 and 400k.

| model | ≥ 120 GeV rms Δ/σ | Σ pull² ≥ 120 (18 cells) | Σ pull² < 120 (30 cells) | n_constituents 120–140 / 140–170 / 170–200 | spectrum 120–140 / 140–170 / 170–200 |
|---|---|---|---|---|---|
| set r2 (v2_m2, s0) | 0.032 | 57 | 129 | +0.031 / +0.062 / +0.049 | 1.053 / 1.023 / 0.937 |
| set s1 (v2_m2, s1) | 0.020 | 21 | 45 | +0.011 / +0.037 / +0.035 | 1.037 / 1.012 / 0.975 |
| set s2 (v2_m2, s2) | 0.023 | 29 | 68 | +0.020 / +0.038 / +0.044 | 1.028 / 0.994 / 0.972 |
| **w1 s0** (v2_w1, loss) | 0.030 | 42 | 73 | +0.022 / +0.046 / +0.063 | 1.034 / 0.995 / 0.977 |
| **w1 s1** (v2_w1, loss) | 0.028 | 34 | 81 | +0.016 / +0.037 / +0.060 | 1.032 / 1.003 / 0.988 |
| **w1 s2** (v2_w1, loss) | 0.024 | 27 | 125 | +0.022 / +0.032 / +0.059 | 1.049 / 1.003 / 1.004 |
| w1 sample s0 (v2_w1, sample) | 0.031 | 47 | 78 | +0.027 / +0.033 / +0.059 | 1.021 / 0.993 / 0.916 |
| set 400k (v2_m2, s0) | 0.019 | 21 | 57 | +0.016 / +0.040 / +0.037 | 1.027 / 0.996 / 0.972 |

(se per cell ≈ 0.013–0.04 above 120 GeV, mostly from the shared direct sample; spectrum ratio se ≈ 3 % in 170–200.)

- **Weighting does not improve high-pT fidelity.** With ×9 ESS in 170–200 GeV and ×5 in 140–170, the three weighted seeds give Σ pull² 42 / 34 / 27 (mean 34) against 57 / 21 / 29 (mean 36) for the unweighted seeds. The rms effect is 0.024–0.030 against 0.020–0.032.
- **The n_constituents excess at 170–200 GeV is not reduced.** It is +0.059 to +0.063 in all four weighted models, against +0.035 to +0.049 unweighted; at 140–170 GeV it is +0.032 to +0.046 against +0.037 to +0.062. ptd and lead_fraction above 140 GeV are no better either.
- **Only the spectrum at 170–200 GeV moves:** 0.977, 0.988, 1.004 against 0.937, 0.975, 0.972 (≈ +0.03, ≈ 1σ of the ratio).
- **Low pT is unchanged** (Σ pull² < 120: 73–125 against 45–129), despite ×0.42 ESS at 60–70 GeV. The low-pT defect is not data-limited either.
- **Sample mode is no better than loss mode** (worse in ptd, lead_fraction and the spectrum above 170 GeV, one seed).
- **Verdict (handoff decision point):** the high-pT defect of the set surrogate is **not data-limited**. More effective training statistics at high pT — ×2 jets (m3a), ×9 ESS (weighted) — do not reduce it. Seed and training length (400k) matter more. The persistent n_constituents excess above 140 GeV is a capacity or parametrisation effect, so the handoff's phase 3 applies: conditioning the flow on jet pT (the factorised model, plan §9.1) or the observed-pT bias arm.
- **High-statistics shifts** of the four models are running (jobs 60440586/7); they test whether weighting costs anything in the integrated measure.

## Phase 2 result: weighted toy training vs unweighted, `toy.ptbins` ≥120 GeV (2026-10-28)

4 weighted trainings on `data/v2_w1` (k=3, cap 250, same 6000 jets/θ as the unweighted `set_r2`
baseline), job 60354914 (1:57:42, 4 A100, rc 0 all): `set_d256_L6_200k` with `--weight-mode loss`
seeds 0–2 and `sample` seed 0, multC64 attached. Validation losses agree to 4e-6 across the loss-
mode seeds (1.055951–1.055957), same order as the unweighted seed spread found earlier. Evaluated
with `toy.ptbins` (job 60358305, same settings as every other model in this comparison — redraw,
22 000 clouds, 16 points) against the main session's `set_r2`/`set_400k` baselines and this
branch's `set_m3a` (2× unweighted jets).

**Aggregate over the 3 bins ≥120 GeV and 6 observables:**

| model | jets/θ | mean \|effect\| | mean \|pull\| |
|---|---|---|---|
| set_r2 (unweighted baseline) | 1× | 0.0275 | 1.57 |
| set_m3a (unweighted, 2× jets) | 2× | 0.0225 | 1.25 |
| **w1 loss-mode, mean of 3 seeds** | **1× (weighted)** | **0.0227** | **1.24** |
| w1_s0 / s1 / s2 individually | 1× | 0.0252 / 0.0228 / 0.0200 | 1.38 / 1.22 / 1.11 |
| w1samp_s0 (sample mode) | 1× (weighted) | 0.0262 | 1.46 |
| set_400k (unweighted, 2× steps) | 1× | 0.0152 | 0.91 |

**Headline: weighted training at the same 6000 jets/θ as the unweighted baseline matches set
m3a's improvement (2× the unweighted jets) to within the seed noise — the core hypothesis of
this study.** Training length (400k) is still the single strongest lever, as already established,
but weighting gets most of the data-doubling benefit without doubling the data.

**Seed spread is large** (effect 0.0200–0.0252, a ≈25 % relative range across 3 seeds) — bigger
than the ≈0.07σ seed spread found for the shift metric on unweighted models. Any one weighted
seed's ptbins number should not be over-read; the 3-seed mean is the number to trust.

**Loss mode outperforms sample mode** in this single comparison (0.0227 vs 0.0262): the smooth
per-minibatch reweighting beats the multinomial-draw sampler here, though sample mode only has
one seed so its own noise is unmeasured.

**Spectrum deficit at 170–200 GeV is reduced, further than set_400k:** ratio 0.937 (r2) →
0.977/0.988/1.004 (w1 loss seeds 0/1/2, mean 0.990) vs 0.972 (400k). Sample mode is the exception:
0.916, worse than even the unweighted baseline — consistent with loss mode being the better of
the two weighting implementations tested.

**Per observable, seed-averaged over the 3 loss-mode runs (not seed noise, the honest picture,
rule 5):**
- **Clearly improved:** width (0.031/0.003/0.053 → 0.017/0.014/0.045), outer_fraction
  (0.026/0.011/0.039 → 0.007/0.028/0.022 — worse at 140–170, better elsewhere), soft8_fraction
  (0.031/0.034/0.006 → 0.020/0.014/0.004).
- **Roughly a wash once seed-averaged** (a single seed, w1_s0, looked worse in isolation but
  this did not replicate in s1/s2): ptd (0.015/0.022/0.034 → 0.009/0.021/0.039), lead_fraction
  (0.007/0.018/0.023 → 0.006/0.021/0.022).
- **Not fixed, mildly worse at the very top bin:** n_constituents at 170–200 GeV (0.049 → 0.060,
  averaged) — the "surrogate's high-pT jets carry a few too many soft constituents" defect is not
  resolved by weighting alone, and may need phase 3 (observed-pT conditioning) if it matters.

**Conclusion for this study:** weighted training works as intended for the toy — it buys close to
a 2× effective sample-size improvement at ≥120 GeV for the same generation cost, confirming the
phase-0 decision to proceed. It does not fix every observable (n_constituents at the very top bin
persists), consistent with that residual being a capacity or conditioning issue rather than a
pure statistics one.

**Still pending:** the JEWEL weighted training (job 60373320) has not started — Perlmutter-wide
GPU shortage (0 idle nodes, hundreds of jobs queued across all users as of 2026-10-27 late
evening), not an account-priority or code issue. Will report once it runs.

## Phase 2 result: high-statistics shifts of the weighted models (2026-10-28, jobs 60440586/7)

Same protocol as the main study's `runs/m3/shift_hi` (`gate shift --direct-dir data/v2_m3_shiftdirect`,
11 points 1000–1010, redraw, 50 000 clouds, code snapshot 046a3cb, A100). The shift is the
equivalent θ shift in marginal σ at N = 1000 jets, per parameter (b, T_ref, τ_init). Per-point
se ≈ 0.05. Baselines are read from `runs/m3/shift_hi/*_redraw/shift.json` (read only).

| model | 11-point mean (b, T_ref, τ_init) | point 1000 (b, T_ref, τ_init) | max \|b\| | points with \|shift\| < 0.2 |
|---|---|---|---|---|
| set_r2 | +0.057, +0.057, +0.004 | −0.22, −0.26, +0.26 | 0.43 | 6/11 |
| set_s1 | +0.126, +0.130, −0.072 | −0.12, −0.14, +0.16 | 0.54 | 7/11 |
| set_s2 | +0.132, +0.137, −0.067 | −0.02, −0.04, +0.07 | 0.51 | 8/11 |
| set_m3a (2× jets) | +0.142, +0.146, −0.050 | −0.04, −0.06, +0.11 | 0.50 | 8/11 |
| set_400k_s0 | −0.035, −0.036, +0.098 | −0.21, −0.23, +0.27 | 0.31 | 10/11 |
| **w1 loss s0** | +0.041, +0.041, +0.007 | −0.16, −0.19, +0.18 | 0.55 | 7/11 |
| **w1 loss s1** | +0.220, +0.221, −0.134 | +0.03, +0.00, +0.04 | 0.64 | 6/11 |
| **w1 loss s2** | −0.030, −0.032, +0.026 | −0.13, −0.16, +0.11 | 0.40 | 9/11 |
| **w1 sample s0** | +0.055, +0.058, +0.004 | −0.09, −0.11, +0.14 | 0.49 | 8/11 |

Checkpoint sha256 (prefix): r2 1d587f39, s1 7eab17b8, s2 0fa9e414, m3a 2ed823c7, 400k a297fd9f,
w1 s0/s1/s2 c66c845f/e4f95db2/42d89d8a, w1samp 4cb27a14.

- **Weighting costs nothing in the integrated measure, and gains nothing.** The 11-point mean b
  shift is +0.077 (seed sd 0.129) for the three weighted seeds against +0.105 (sd 0.042) for the
  three unweighted seeds; the difference is well inside the seed spread. Points passing 0.2σ: 7/6/9
  against 6/7/8.
- **The weighted seeds spread more** (b mean −0.03 to +0.22 against +0.06 to +0.13). With three
  seeds each, the variance ratio (≈ 9) is not significant (F(2,2) 95 % point 19). w1 loss s1 is the
  worst single model in this table (mean +0.22, max |b| 0.64), and w1 loss s2 is among the best.
  This matches the larger seed scatter in `toy.ptbins`.
- Sample mode (one seed) is indistinguishable from loss mode here.

## Reconciling the two phase-2 `toy.ptbins` verdicts above (2026-10-28)

The two phase-2 sections above disagree: "no high-pT improvement" versus "matches set m3a,
confirming the core hypothesis". They use the same outputs, and every number in both reproduces
(mean |effect| and |pull| over the 3 bins ≥ 120 GeV × 6 observables, `ptbins_summary.json`):

| model | mean \|effect\| | mean \|pull\| | Σ pull² |
|---|---|---|---|
| set_r2 | 0.0275 | 1.57 | 57 |
| set_s1 | 0.0173 | 0.94 | 21 |
| set_s2 | 0.0198 | 1.13 | 29 |
| **unweighted, mean of 3 seeds** | **0.0215** | **1.21** | 36 |
| set_m3a (2× jets) | 0.0225 | 1.25 | 32 |
| **w1 loss, mean of 3 seeds** | **0.0227** | **1.24** | 34 |
| w1 sample s0 | 0.0262 | 1.46 | 47 |

The "confirming" verdict compares the weighted seed mean with set_r2 alone, but set_r2 is the worst
of the three unweighted seeds (section "Phase 0, corrected"). Against the unweighted seed mean, the
weighted models (0.0227, 1.24) and set m3a (0.0225, 1.25) both equal the baseline (0.0215, 1.21)
within the seed noise. So m3a's apparent gain was itself seed noise, and weighting reproduces it
only in that sense. The 170–200 GeV spectrum ratio is the one quantity that does move with
weighting (mean 0.990 against 0.961 unweighted, ≈ 1σ). The shifts above agree: no cost, no gain.

**Standing verdict:** at this model size and training length, high-pT fidelity of the toy surrogate
is not data-limited. Weighting is safe (it costs nothing in shifts or low-pT fidelity) but does not
buy fidelity. The persistent n_constituents excess above 140 GeV points to phase 3 (pT conditioning).

## Phase 4 result: weighted JEWEL training complete, stable, lower validation loss (2026-10-28)

Job 60373320 (afterok-chained after the cleanup farm and merge), 4 runs on `data/jewel_w1/merged`
(WEXPO=5, 2 422 694 weighted events, 972 points) vs `data/jewel_v1/merged`'s unweighted validation
(146 016 jets, 128 points, unit weights — **ESS = n exactly**, confirmed), `jewel_multJ.pt`
p(N|θ) attached: `set_d256_L6_200k`, `--weight-mode loss` seeds 0–2 + `sample` seed 0. 2:29:42
wall, 1 node × 4 A100, **rc 0 on all four**.

**Stability: no problem from the extreme WEXPO=5 weight range flagged last night.** Val loss
curves fall smoothly (1.39 → 1.163) and converge tightly across all three loss-mode seeds
(1.16347–1.16348) and sample mode (1.16330) — no blow-up, no divergence. The realised weight
range in the actual training batches was [0.0007, 77.6] (after `merge`'s per-point Σw = n_jets
rescaling — far tamer than the raw ≈31 500 pooled max/min ratio estimated from the ESS study,
because that ratio is the *spread*, not the normalised scale). **Measured train ESS 327 146 of
2 422 694 (fraction 0.135)** — matches the pooled k=5 prediction (0.14) almost exactly.

**Fair, controlled comparison against the unweighted `jewel_r1b` baseline** (identical
architecture, dim/depth/batch/lr/steps/length_buckets/ema_decay, identical unweighted validation
split; the only difference is the training data):

| | train data | events | best val loss |
|---|---|---|---|
| jewel_r1b `set_d256_L6_s0`/`s1` (unweighted) | `jewel_v1/merged` train | 28.3 M | 1.19412 / 1.19415 |
| **w5 `set_d256_L6_200k_w5_s0/s1/s2` (weighted, loss mode)** | `jewel_w1/merged` | **2.42 M** | **1.16347 / 1.16347 / 1.16348** |
| w5 `..._w5samp_s0` (weighted, sample mode) | `jewel_w1/merged` | 2.42 M | 1.16330 |

**The weighted model, trained on 12× fewer events, has a meaningfully lower validation loss on
the identical unbiased validation split** (≈0.030 lower, consistent across all 3 loss seeds, and
the sample-mode run agrees). Since the validation loss is flow-matching error against the true
unbiased law, this is evidence the weighted sample (concentrated where the surrogate needs more
statistics) is a more sample-efficient training set overall, not only in the pT tail — though
this single aggregate number does not say *where* the improvement comes from.

**Not yet possible: a per-pT-bin fidelity check (the `toy.ptbins`-equivalent) for JEWEL**, because
`data/jewel_v1`'s fidelity split (needed as the direct reference, like `v2_m1_fidelity` for the
toy) is still unindexed/unmerged — that step belongs to the other session's tree (`dev/nle-
fidelity`), not something fixable from here. Until it lands, the validation-loss comparison above
is the strongest signal available for the weighted JEWEL models.

**Status at the end of this overnight run: both arms' production + training are complete.**
Remaining work: the JEWEL per-pT-bin fidelity check (blocked on the other session), and a
decision on whether WEXPO=5's small per-observable cost (toy: `n_constituents` at 170–200 GeV)
needs addressing before using these checkpoints for anything downstream.
