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
