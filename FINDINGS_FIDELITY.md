# Surrogate fidelity thread (branch dev/nle-fidelity)

Running log of `docs/SURROGATE_FIDELITY_PLAN.md` steps S0–S5 (toy v2), kept apart from
`FINDINGS_NLE.md` (the main thread folds these entries in). Worktree
`/global/cfs/cdirs/alice/ploskon/henv-omni-sbi/omnilearned-sbi-fidelity`; every job runs from
`$JEST_NLE_ROOT/src_snapshots/<rev>` of a commit on this branch. Criteria unchanged (rules 2
and 5): gate FAIL = family-wise p < α/128 and relevant at N = 1000; requirement |δθ| ≤ 0.2σ at
N = 1000. Choices use validation/development data only.

Reference (round 2, `set_d256_L6_200k_multC64`): 1 FAIL, 23 with p < 0.01, constrained shift
1.17 (noise ≈ 0.55), marginal +0.53, +0.48, −0.30 σ (b, T_ref, τ_init) at N = 1000.

## Job / node-hour log

| date | step | job | resources | status | what |
|---|---|---|---|---|---|
| 2026-10-25 | S0.3 / S3 / S4 | 60282682 (`nle-train4`, regular, 1 node × 4 A100, 7 h limit) | **failed after 17 s** (`nvidia-smi -L -i` invalid, every run's subshell exited; the job still COMPLETED, 0.005 node-h) → resubmitted as 60284243 | `runs/m2r3/`: set seed 1, set seed 2 (200k), set 400k, set d384 L8 200k, all on `data/v2_m2` with round-2 settings; the round-2 all-counts p(N\|θ) attached (`checkpoint_mult.pt`) |
| 2026-10-25 | S1 | 60283786 (`s1-gate-set`), 60283787 (`s1-gate-pet`) | 1 node × 4 A100 each, 3 h | set: done (2 h 27 m); PET: **timeout** at 117/128 points | redraw re-gates (shift, convergence, 128 points) of set r2 / PET r2 → `runs/m3/{set,pet}_d256_L6_200k_multC64_redraw/` (snapshot 19ac616) |
| 2026-10-25 | S1 | 60283788 (`s1-acc-set`) | 1 node × 4 A100, 1.5 h | done (48 min), acceptance 0.914–0.982 | Pr_q(S\|θ) and per-N table of set r2 → `runs/m3/acceptance/set_d256_L6_200k_multC64/` |
| 2026-10-25 | S0/S1 | 60284684, 60284685, 60284686 (`fid_boundary`, 9b648aa) | 1 node × 4 A100 each, 1.5 h | done (47–87 min) | `toy.boundary` at 32 gate points of set r2, PET r2, set r1 → `runs/m3/boundary/<run>/` |
| 2026-10-25 | S2 | 60283789 (`nle-s2-data`) | 1 CPU node, 45 min | done (9 min; 12.3 M and 8.2 M jets) | `data/v2_m3a`, `data/v2_m3b` |
| 2026-10-25 | S2 × S3 | ~~60283791~~ cancelled (same script bug) → 60284244 (after 60283789) | 1 node × 4 A100, 8 h | done (3 h 37 m; 60284243: 3 h 36 m) | `runs/m2r3/set_d256_L6_{200k,400k}_{m3a,m3b}` |
| 2026-10-25 | gates | ~~60283792–60283807~~ cancelled → 60284245–60284252 (after 60284243), 60284253–60284261 (after 60284244), snapshot 5d3e948 | 1 node each, 3 h | s1, s2, 400k done; d384, 200k_m3a running (≈ 90/128 on 2026-10-26 10:50); 400k_m3a, m3b pending | each new model with the round-2 method (`runs/m3/m2r3_<run>`) and with redraw (`..._redraw`) |

## S0: diagnostics on existing results (2026-10-25, login node, analysis only)

New module `toy.residuals` (`map`, `scaling`, `seeds`, `row`; tested in
`tests/sbi/test_selection_per_n.py`), run from the worktree at d7f16ca + uncommitted
residuals.py (committed next). Outputs `runs/m3/fidelity_s0/`. Units: signed shift of the jet
observable means divided by the direct jets' standard error at N = 1000 (the gate's relevance
unit, so |value| is the relevance). `row` reproduces the recorded statistics (set r2: 1 FAIL,
23 below 0.01, constrained 1.17, marginal +0.53, +0.48, −0.30 σ; set r1: 9, 104, 1.32).

**S0.1 residual map** (weighted linear fit over the 128 validation points, signed shift =
c0 + c_b u_b + c_T u_T + c_τ u_τ, u ∈ [−1, 1] over the box; se ≈ 0.03 on c0, 0.05 on slopes):

| generator | width c0 / c_T | outer fraction c0 / c_T | jet pT c0 / c_T | N c0 / c_T |
|---|---|---|---|---|
| set r1 (2000 jets/pt) | −0.64 / −0.67 | −0.55 / −0.55 | +0.57 / −0.01 | −0.39 / −0.27 |
| set r2 (6000 jets/pt) | −0.62 / −0.35 | −0.54 / −0.23 | +0.38 / +0.18 | −0.33 / −0.26 |
| PET r2 (6000 jets/pt) | −0.61 / −0.56 | −0.50 / −0.33 | +0.24 / +0.17 | −0.35 / −0.27 |

- **The offsets in width (≈ −0.62), outer fraction (≈ −0.53) and N (≈ −0.35) are the same for
  all three generators**, across 3× the data and two architectures. Only the T_ref slopes and
  the jet-pT offset improved from round 1 to round 2.
- All three carry the same p(N|θ) (the all-counts multC64 model) and the same selection
  step (rejected clouds dropped). A common offset in N that does not move with the flow
  points to the selection tilt of H3 (or to p(N|θ) itself), not to flow capacity or data.
  S1 separates the two: with same-N redraws the selected surrogate N follows p(N|θ) exactly.
- By T_ref band (set r2), noise-corrected mean squared bias B² of width: 0.34 (0.25–0.35),
  0.69 (0.35–0.45), 0.89 (0.45–0.55); outer fraction 0.23 / 0.48 / 0.65; N 0.10 / 0.12 /
  0.29. The one FAIL is in the top band.

**S0.2 scaling round 1 → 2** (B² = mean(Δ² − se²), relevance units², set):

| region | observable | B² (2000/pt) | B² (6000/pt) | α (B ∝ n^−α) | predicted B² at 12k / 24k |
|---|---|---|---|---|---|
| all 128 | width | 0.87 ± 0.05 | 0.64 ± 0.05 | 0.14 | 0.53 / 0.43 |
| all 128 | outer fraction | 0.61 ± 0.05 | 0.46 ± 0.04 | 0.13 | 0.38 / 0.32 |
| all 128 | jet pT | 0.42 ± 0.04 | 0.22 ± 0.03 | 0.29 | 0.15 / 0.10 |
| all 128 | N | 0.24 ± 0.03 | 0.17 ± 0.03 | 0.14 | 0.14 / 0.12 |
| T_ref > 0.45 (42) | width | 1.55 ± 0.12 | 0.89 ± 0.09 | 0.25 | 0.63 / 0.44 |
| T_ref ≤ 0.45 (86) | width | 0.54 ± 0.05 | 0.52 ± 0.05 | 0.02 | 0.50 / 0.49 |
| T_ref ≤ 0.45 (86) | outer fraction | 0.35 ± 0.04 | 0.36 ± 0.04 | −0.02 | 0.37 / 0.38 |

- Tripling the data reduced the high-T_ref part of the width residual (α ≈ 0.25) and the jet
  pT residual (α ≈ 0.3), but **did nothing for width and outer fraction at T_ref ≤ 0.45**.
  That part is a floor, not a data deficit.
- Prediction for S2: 12 000 jets per point everywhere (option a) would lower the overall width
  B² only from 0.64 to ≈ 0.53. Doubling the density at T_ref > 0.45 (option b) targets the
  only region that scales (0.89 → ≈ 0.63 there). Option b is the better buy, and neither
  removes the floor. Decision deferred until the S1 re-gate shows how much of the floor is the
  selection tilt; caveat: round 1 → 2 also changed the steps (best at 156k vs 200k) and
  includes training-to-training scatter, which S0.3 measures.
- Mean largest relevance: all 1.05 → 0.91; T_ref > 0.45: 1.26 → 1.02; T_ref ≤ 0.45:
  0.95 → 0.85.

## S1: per-N selection normalization (code, 2026-10-25)

- **Why:** p(N|θ) is trained on the multiplicities of *selected* jets, but the surrogate's
  selected sample (gate, self-closure control) and the likelihood normalization
  Pr_q(S|θ) = Σ_N p(N|θ) Pr_q(S|N,θ) treat it as the pre-selection multiplicity. When the
  flow's acceptance falls with N, dropping rejected clouds tilts the selected N (and every
  N-correlated observable, width and outer fraction among them) low. S0.1 shows exactly such a
  common offset in N, width and outer fraction for all three generators.
- **Consistent model:** p(N|θ) q(x|N,θ) 1_S(x) / Pr_q(S|N,θ). It keeps p(N|θ) as the selected
  multiplicity distribution.
- **Code (tested in `tests/sbi/test_selection_per_n.py` and `test_datasets.py`):**
  - `FlowSurrogate.sample` takes an array of per-cloud N.
  - `sample_and_select(redraw=True)` draws rejected clouds again at the same N (seed + r·2^24
    in round r, at most 50 rounds, unfilled ones reported). The report keeps the first-pass
    acceptance and the first-pass counts per N (`acceptance_by_n`).
  - `gate run/shift --redraw`, recorded in every point JSON and the summary.
  - `selection.fit_logit_in_n` / `save_per_n_table` / `PerNLogNorm`: Pr_q(S|N,θ) from the
    per-θ table's own common-random-number samples. Per grid node a binomial logistic fit
    with a logit piecewise linear in log N (5 knots at quantiles of the sampled N, small
    ridge), the coefficients interpolated linearly in θ. `bind(counts)` gives the fit's
    `log_norm_fn` (mean over the observed events of log a(N_i, θ), so the scan subtracts
    Σ_i log a(N_i, θ)).
  - `closure acceptance` stores per-N counts; `acceptance-merge --per-n-out` writes the
    per-N table. A campaign spec `selection_normalization = "per_n"` fits with it
    (fixed-vector mode only), and its self-closure control uses same-N redraws. The default
    stays per-θ, so existing campaigns are unchanged.
  - Tests: the redraw keeps the drawn N multiset exactly, while dropping shifts ⟨N⟩ by the
    predicted p(N)a(N) tilt. The logit fit recovers a known a(N). In a Poisson toy with an
    N-dependent acceptance, the per-N likelihood's MLE equals the sample mean, while the
    per-θ normalization is biased by more than 5 se.

## S2 data, plumbing (2026-10-25)

- `datasets design --from-index K --select PARAM LO HI` extends a Sobol design (the prefix is
  unchanged; tested). The design keeps `point_ids` as the design indices (independent random
  streams). `datasets generate --shard-offset` adds shards next to another design's.
- `shell/nle/gen_toy_local.sh` fix: since 5be73c6 the xargs line quoted `$PY` as a single
  word (`"henv -n omni-sbi -x python"`), which cannot run. **Main-thread note:** fix the same
  line on dev/nle-perlmutter before the script is used again.
- `shell/nle/fidelity_s2_data.sbatch` builds two training sets:
  - `data/v2_m3a`: the round-2 design with 12 000 jets per point (option a).
  - `data/v2_m3b`: the round-2 shards (symlinks) plus the 343 extra Sobol points (indices
    1024–2047 with T_ref > 0.45) at 6000 jets (option b).
  - Both link the round-2 validation split.

## Boundary diagnostic, `toy.boundary` (9b648aa, 2026-10-25)

A second explanation besides the N tilt (H3) fits the S0 offsets: **truncation leakage at
the selection boundaries (H6).**
- A smooth density fitted to truncated data puts mass beyond the cut and lacks mass just
  inside it, so the selected mean moves away from the cut.
- Jet pT ≥ 60 GeV: the selected mean pT comes out high (observed: +0.38).
- One-jet requirement: the widest clouds fail it, so jets just inside are missing and the
  width comes out low (observed: −0.62).
- The leakage does not shrink with more data, which matches the flat S0.2 scaling below
  T_ref = 0.45.
- Same-N redraws (S1) do not remove it. Redrawing at fixed N samples the same truncated
  q(x|N), so only the N-mix part changes.

`toy.boundary` measures the split directly:
- **Decomposition:** Δ⟨o⟩ = Σ_N (p_s − p_d) o_d(N) [N mix] + Σ_N p_s (o_s(N) − o_d(N))
  [fixed N], exact by construction, for the dropped and the redrawn selected samples.
- **Boundary densities:** near the jet-pT cut and of the largest constituent distance from
  the axis, plus the surrogate's mass beyond the cuts.

A CPU smoke test (1 point, 1500 clouds, 16 steps; not converged, so not a result) already
put almost all of the width shift in the fixed-N term.

If the full run confirms it, the S5 lever is not loss weighting. It is a representation that
puts the boundary into the transform, for example constituents relative to the jet axis with
a bounded ΔR. That would be a new item ("S5b"), to be discussed before any change.

## Results: S0.3 seed spread, S1 re-gates, S3 400k, boundary decomposition (2026-10-26)

Analysis on the login node, worktree at dca6224 plus the `seed_spread` fix below. Outputs in
`runs/m3/fidelity_s0/` (`map_<run>.json`, `seeds_set_r2_s1_s2.json`) and
`runs/m3/boundary/<run>/summary.{json,txt}`. Units as in S0 (relevance at N = 1000). All
m2r3 trainings ran on `data/v2_m2` with the round-2 settings (snapshot 5d3e948). The source
diff from round 2 (b4e4c64) touches sampling options only. The seed-0 400k run repeats round
2's loss curve to the 5th digit, so seed and stopping step are the only differences. The best
validation loss is 1.05222 for all of them.

**Gate rows** (`toy.residuals row`, 128 points unless noted):

| run | best step | FAIL | p < 0.01 | mean max rel. | constrained shift | marginal b, T_ref, τ_init |
|---|---|---|---|---|---|---|
| set r2 (seed 0) | 156k | 1 | 23 | 0.91 | 1.17 | +0.53 +0.48 −0.30 |
| set r2, redraw | | 1 | 27 | 0.86 | 0.96 | +0.46 +0.42 −0.25 |
| set seed 1 | 176k | 2 | 14 | 0.76 | 1.09 | +0.66 +0.62 −0.44 |
| set seed 1, redraw | | 1 | 13 | 0.73 | 0.94 | +0.61 +0.57 −0.39 |
| set seed 2 | | 0 | 4 | 0.74 | 1.10 | +0.71 +0.67 −0.46 |
| set seed 2, redraw | | 0 | 8 | 0.71 | 0.95 | +0.66 +0.63 −0.41 |
| set 400k (seed 0) | 332k | 1 | 6 | 0.72 | 0.97 | +0.56 +0.52 −0.30 |
| set 400k, redraw | | 0 | 10 | 0.71 | 0.88 | +0.50 +0.47 −0.26 |
| PET r2 | | 2 | 55 | 0.91 | 1.12 | +0.64 +0.60 −0.34 |
| PET r2, redraw (117 points) | | 1 | 55 | 0.83 | 0.95 | +0.60 +0.57 −0.33 |

**Residual map offsets** c0 / c_T (and B², all 128 points):

| run | width | outer fraction | jet pT | N | not-one-jet leakage |
|---|---|---|---|---|---|
| set r1 | −0.64 / −0.67 (0.87) | −0.55 / −0.55 (0.60) | +0.57 / −0.01 (0.42) | −0.39 / −0.27 (0.23) | 5.3 % |
| set r2 | −0.62 / −0.35 (0.64) | −0.54 / −0.23 (0.45) | +0.38 / +0.18 (0.22) | −0.33 / −0.26 (0.17) | 3.8 % |
| set r2, redraw | −0.53 / −0.25 (0.51) | −0.46 / −0.15 (0.36) | +0.45 / +0.23 (0.30) | −0.01 / −0.01 (0.03) | |
| PET r2 | −0.61 / −0.56 (0.68) | −0.50 / −0.33 (0.41) | +0.24 / +0.17 (0.11) | −0.35 / −0.27 (0.19) | 4.2 % |
| set seed 1 | −0.35 / −0.32 (0.33) | −0.29 / −0.24 (0.21) | +0.29 / +0.11 (0.16) | −0.28 / −0.23 (0.13) | 3.2 % |
| set seed 2 | −0.36 / −0.27 (0.32) | −0.29 / −0.20 (0.21) | +0.21 / +0.07 (0.10) | −0.23 / −0.22 (0.11) | 2.7 % |
| set 400k | −0.47 / −0.22 (0.32) | −0.39 / −0.15 (0.22) | +0.25 / +0.09 (0.11) | −0.19 / −0.18 (0.08) | 2.5 % |

(Leakage: the fraction of generated surrogate clouds that fail the one-jet requirement, mean
over the gate points. Redraw rows of seed 1, seed 2 and 400k move like set r2's: N to −0.01,
width and outer fraction by +0.05 to +0.09, jet pT by −0.06 to −0.09.)

**S0.3, seed spread: the S0 "common floor" was partly a coincidence.**
- Three same-configuration trainings put the global width offset at −0.62, −0.34 and −0.35 (std 0.15), and outer fraction at −0.54, −0.29, −0.29 (std 0.14). Their validation losses are equal, so the loss does not see this.
- Per-point model-to-model scatter: τ ≤ 0.20 (width), ≤ 0.19 (outer fraction), ≤ 0.12 (jet pT).
- The S0 statement that r1, set r2 and PET r2 share the same offset compared three single trainings. S0.2's scaling exponents, from one training per data size, carry the same ±0.15 scatter, so the "floor at T_ref ≤ 0.45" is not established either.
- **The parameter shift is stable even though the observable residuals are not:** constrained shift 1.17, 1.10, 1.10 (std 0.04). The marginals got worse in the seeds with smaller residuals (b +0.53 → +0.66, +0.71).
  - So the dev-point shift is not driven by the width/outer-fraction offset that dominates the gate statistics.
  - The requirement (|δθ| ≤ 0.2σ) is missed by 2.5–3.5× in every model so far.
  - Caveat: the shift is a linear Fisher projection of 7 observable means, and the main thread found that this Fisher matrix under-estimates the information about 10× (FINDINGS_NLE, N = 1000 numerics check).
- **Estimator fix:** `toy.residuals.seed_spread` subtracted independent surrogate noise. The gates share the surrogate seeds (seed + point) as well as the direct sample, so for similar models that noise mostly cancels, and the subtraction gave τ² < 0, τ = 0 for every observable. It now reports `tau_independent` (lower bound), `tau_upper` (the unsubtracted pair RMS / √2, the default with common random numbers) and the run-to-run spread of the mean offset (`offset_std`). The test covers the CRN case.

**S1, same-N redraw: it removes the N tilt and little else.**
- Every model's N offset goes to −0.01, and the constrained shift drops by 0.10–0.21 (1.17 → 0.96 for set r2).
- The width and outer-fraction offsets shrink by only 0.05–0.12, jet pT grows by 0.06–0.09, and p < 0.01 counts do not improve (23 → 27, 14 → 13, 4 → 8, 6 → 10).
- So H3 (selection tilt through N) is a minor part of the width residual. The per-N normalization stays useful for N itself and for the shift.

**Boundary decomposition (`toy.boundary`, 32 points each): H6 confirmed as the main mechanism.**

| run | width total / N-mix / fixed-N | outer fraction | jet pT | N |
|---|---|---|---|---|
| set r1 | −1.07 / −0.15 / −0.92 | −0.90 / −0.13 / −0.78 | +0.45 / −0.13 / +0.58 | −0.52 / −0.52 / 0.00 |
| set r2 | −0.84 / −0.13 / −0.71 | −0.68 / −0.11 / −0.57 | +0.44 / −0.11 / +0.55 | −0.46 / −0.45 / 0.00 |
| PET r2 | −0.98 / −0.14 / −0.84 | −0.68 / −0.12 / −0.57 | +0.25 / −0.12 / +0.37 | −0.48 / −0.48 / 0.00 |

(dropped samples; with redraw the N-mix terms go to ≈ +0.03 and the fixed-N terms do not change.)
- About 85 % of the width and outer-fraction shift is at fixed N, and all of the N shift is N-mix.
- **Boundary densities, mean over 32 points, surrogate-selected / direct:**
  - Constituent distance from the axis: 1.04–1.09 over ΔR 0.28–0.38, then **0.91–0.95 in the last bin inside R = 0.4** (0.38–0.40); 5.5–7.2 % of generated clouds leak out and fail the one-jet requirement.
  - Jet pT: **0.82–0.89 in the first bin above the 60 GeV cut**, 1.01–1.04 above it.
  - This is the truncation-leakage pattern H6 predicts: mass is missing just inside each cut and pushed inward, which shrinks the width and outer fraction and raises the jet pT.
- Across models the leakage tracks the width offset only loosely: r1 → r2 cut the leakage 5.3 → 3.8 % at an unchanged offset, while seeds 1 and 2 have 2.7–3.2 % and offset −0.35.

**Reading and consequences for the plan:**
1. The width/outer-fraction residual is a fixed-N boundary error of the flow (H6). Neither the selection normalization (S1) nor more steps (S3: 400k gives −0.47 against −0.34 to −0.62 for the 200k seeds, within the seed spread) removes it.
2. Any S2/S3/S4 comparison must be made against the seed spread (offset std 0.15, constrained-shift std 0.04), not against the single round-2 model. The m3a / m3b / d384 gates still running are single trainings and will be read that way.
3. S5b (a representation with the R = 0.4 boundary in the transform, e.g. constituents relative to the jet axis with a bounded ΔR) is the lever for the gate residual. It needs discussion before any change. It does not obviously address the parameter shift, which barely moves with the residual: it should be checked whether that shift comes from the jet-pT/N terms or from the linear-Fisher read-out itself.
