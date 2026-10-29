> **CLOSED (2026-10-28).** Toy track stopped; superseded by `docs/PLAN_JEWEL_NLE.md`.
> Kept as the record of S0-S6 and their results.

# Surrogate fidelity: plan for the additional steps (toy v2, then JEWEL)

Status 2026-10-25. Branch `dev/nle-perlmutter`; the brief is `PLAN_NLE_PERLMUTTER.md`
(M2/M3), results are in `FINDINGS_NLE.md`. This plan covers what to do while the round-2
generators still fail M3 by a small margin. It is a to-do list with an order, a cost per
step and the measurement that decides whether the step helped. It changes no criterion.
Rule 5 holds: the gate and the requirement stay as fixed before the results.

## 1. Where we are

| | round 1 (2000 jets/pt, 156k–200k steps) | round 2 (6000 jets/pt, 200k steps) |
|---|---|---|
| gate FAIL, set / PET (of 128) | 9 / 9 | **1 / 2** |
| p < 0.01, set / PET (1.3 expected) | 104 / 60 | 23 / 55 |
| mean largest relevance at N = 1000 | 1.05 / 1.13 | 0.91 / 0.91 |
| constrained shift at the development point, N = 1000 (noise ≈ 0.55) | ≈ 1.3 | 1.17 / 1.12 |
| marginal shift at N = 1000 (b, T_ref, τ_init) | – | +0.53, +0.48, −0.30 σ (set) |

- **Requirement** (plan M3): |δθ| ≤ 0.2 posterior σ at N = 1000. Round 2 is at about
  0.5 σ (marginal) and about 2× the noise level of the constrained-direction statistic.
- **The residual has one signature:**
  - The generated jets are slightly too narrow: width and outer fraction are low, most of
    all at high T_ref. All 3 round-2 FAILs sit at T_ref ≈ 0.53.
  - Jet pT is slightly high.
  - Multiplicity is fine after the all-counts p(N|θ).
- **Tripling the jets per point removed most failures**, but the typical shift fell only
  from about 1.05 to 0.91.
- **Training length:** the validation loss was still falling slowly at 200k steps (best at
  156k and 162k under a cosine schedule).
- **Not yet known:** whether this residual biases inference at N = 1000. The development
  closure with set round 2 at N = 1000 (`runs/m5/dev_N1000_setR2`, started 2026-10-25)
  answers it; see §4.

## 2. Hypotheses

- **H1, data density in the upper box.** Broad jets at high T_ref have more constituents and
  a wider tail; 6000 jets per point may still under-sample that tail.
  - Evidence for: the residual is concentrated there, and more data helped a lot.
- **H2, optimisation, not capacity.**
  - Evidence for: the cosine schedule ended while the loss was still improving.
  - Lesson 7 (v1): capacity did not matter, data and steps did.
- **H3, selection-induced bias (per-N normalization, see §3, S1).** About 6% of generated
  clouds fail the toy selection, mostly at high N. Dropping them shifts the selected
  sample's N and width. The likelihood uses Pr_q(S|θ), not Pr_q(S|N,θ).
- **H4, objective weighting.** Flow matching weights every constituent equally. Width and
  outer fraction are carried by a few soft wide-angle constituents, so an error there costs
  little loss.
- **H5, capacity or backbone.**
  - Evidence against: set is better than PET at equal relevance, and v1 showed no capacity
    effect.
  - It is still the cheapest thing to try in parallel.

## 3. Steps (ordered by information per cost)

Each step ends with the same measurement:
- the gate at all 128 validation points (`toy.gate run` + `summarize --n-settings 128`);
- the equivalent shift at the development point and along the tube (`toy.gate shift`);
- sampler convergence (128 → 256 steps).

A step counts as an improvement if FAILs, the p < 0.01 count and the constrained shift all
go down beyond the seed spread. Measure that spread once (S0).

**S0, diagnostics on existing results.** Cost: ≈ 1 h of analysis, plus one extra training
(about 2 GPU-h).
1. Residual map from the round-2 gate points: Δmean of width, outer fraction and jet pT
   against θ (b, T_ref, τ_init), to locate the bad region and its direction.
2. Round 1 → 2 scaling per point (2000 → 6000 jets): fit relevance ∝ n^−α, to predict what
   12 000 or 24 000 jets per point would give (H1).
3. Seed spread: retrain set round 2 with seed 1 (the JEWEL pair agreed to 3e-5 in loss, but
   gate-level spread is unknown). This sets the noise level for every comparison below.

**S1, per-N selection normalization (H3; the "#4" item).** Cost: ≈ 1 day of code and tests,
≈ 1 GPU-h for the table, one re-gate (≈ 4 node-h).
1. Acceptance table Pr_q(S|N,θ) (N as an extra axis, or a smooth fit in N).
2. The likelihood uses log p(N|θ) + log q(x|N,θ) − log Pr_q(S|N,θ), with p(N|θ) the model of
   the *selected* multiplicity, consistent with how the flow is trained.
3. The gate redraws rejected clouds at the same N, so the surrogate's selected sample keeps
   p(N|θ).
4. Re-gate set round 2 unchanged, to see what this alone removes.
5. Tests: the normalization integrates correctly on a toy where the acceptance is known.

**S2, targeted training data (H1).** Cost: toy generation is CPU-cheap (≈ 1–2 CPU node-h per
6 M jets); one 200k training ≈ 1.8 GPU-h (set).
- Option a: `v2_m3` with 12 000 jets per point everywhere (12.3 M jets; host memory as in the
  round-2 fix).
- Option b: the round-2 design plus extra Sobol points at T_ref > 0.45 (density × 2 there),
  same margin and holdout rules.
- Choose a or b by the S0 scaling fit; train set (and PET only if S0 says PET is not worse
  everywhere).

**S3, longer optimisation (H2).** Cost: ≈ 3.5 GPU-h per run.
- 400k steps with the same cosine schedule, or a longer warm plateau with EMA 0.9995 →
  0.9998.
- Run alongside S2, on the same data, so the effects separate: {round-2 data, 400k} and
  {S2 data, 200k}, then {S2 data, 400k}.

**S4, capacity (H5).** Cost: ≈ 3–4 GPU-h. Set d384 L8 on the best data of S2/S3. Stop if no
gain beyond the seed spread.

**S5, objective (H4), only if S1–S4 leave the width residual.** Cost: ≈ 1–2 days of code.
- Weight the flow-matching loss by constituent pT or ΔR, or add an auxiliary per-jet loss on
  width and outer fraction (a jet-level moment matching term).
- The change must keep the exact likelihood: weights in training only, no change to q(x|N,θ)
  as a density.

**S6, JEWEL.** Once the JEWEL fidelity samples exist (`data/jewel_v1/fidelity`, started
2026-10-25), run the same gate on the JEWEL surrogates. JEWEL has no development-scan samples
yet, so the shift needs them (a small production at the JEWEL development setting and along
its lines) and a JEWEL budget. Apply S1–S4 to JEWEL in the order the JEWEL gate suggests.

## 4. How the N = 1000 closure feeds in

- **The closure passes** (joint levels, coverage, toy − control pulls within 2 se): the
  round-2 residual is below what N = 1000 inference can see at the development point.
  - S1–S4 stay on the list, at lower priority, because the test settings may sit where the
    residual is larger.
  - M5 at the test settings may proceed with set round 2, by user decision.
- **The closure fails on the pulls**, with toy − control pointing along +b, +T_ref, −τ_init,
  the sign the gate's shift predicts: S1–S3 become the critical path. Repeat the closure with
  the improved generator.
- **Coverage alone fails** (as τ_init at N = 250, with the control failing the same way):
  this is the open criterion question, a user decision, not a fidelity result.

## 5. Bookkeeping

- Every new training goes to `runs/m2r<k>/`, every gate to `runs/m3/<run>/`.
- Every result gets a FINDINGS entry with the source revision, dirty status and device.
- The reports repository is synced after each step (`tools/sync.py --push` in
  `omni-sbi-nle-jet-dev-reports`).
- No step changes the gate rule, the requirement, the holdout or the reserved settings
  (rules 2 and 5).
