# pT-biased (weighted) training samples (branch dev/nle-weighted)

Running log of `HANDOFF_WEIGHTED.md`, phases 0–4. Worktree
`/global/cfs/cdirs/alice/ploskon/henv-omni-sbi/omnilearned-sbi-weighted`. Runs and scratch
scripts live in `$JEST_NLE_ROOT/runs/w*/` (`JEST_NLE_ROOT=/global/cfs/cdirs/alice/ploskon/jest-nle`);
the main-study trees are read only.

## Job / node-hour log

| date | phase | job | resources | status | what |
|---|---|---|---|---|---|
| 2026-10-26 | 0 | 60353176 (`w0-ptbins-m3a`, gpu shared, 1 A100, 1.5 h limit) | submitted | pending | `toy.ptbins` of set m3a → `runs/w0/ptbins/set_m3a/` (snapshot a2e7dba, as the baseline) |
| 2026-10-26 | 0 | login node, 1 core | ≈ 2 min | done | ESS vs k by reweighting (`runs/w0/ess/ess_reweight.py`) and biased toy draws at 3 points (`ess_direct.py`, snapshot e9b52fe) |
| 2026-10-26 | 1 | 60354888 (`w1-gen`, cpu shared, 64 tasks × 2 cores, 20 min limit) | ≈ 0.5 core-h expected | submitted | `data/v2_w1`: v2_m2 design and train stream, 6000 jets/point, `--bias-k 3 --bias-cap 250` (snapshot 1bf9fdc) |
| 2026-10-26 | 2 | 60354914 (`w1-train4`, gpu regular, 1 node × 4 A100, 4 h limit, afterok 60354888) | ≈ 2 node-h expected | queued | `runs/w1/`: set d256 L6 200k (r2 settings) on v2_w1, `--weight-mode loss` seeds 0–2, `sample` seed 0; multC64 attached (`spec_w1.txt`) |
| 2026-10-26 | 4 | login node | ≈ 3 min | done | WEXPO from ESS by reweighting jewel_v1 (`runs/w4/ess/ess_jewel.py`) |
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

## Phase 0: set m3a in `toy.ptbins` (pending)

Job 60353176 (gpu shared, 1 A100): set m3a (`runs/m2r3/set_d256_L6_200k_m3a/checkpoint_mult.pt`, multC64 attached), same settings and snapshot as the main-study baseline (a2e7dba: redraw, 22 000 clouds, gate points 0, 8, …, 120) → `runs/w0/ptbins/set_m3a/`. The decision rule is from the handoff: if m3a is better than set r2 in the ≥ 120 GeV bins, high pT is data-limited and the weighted arm is the fix.

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
