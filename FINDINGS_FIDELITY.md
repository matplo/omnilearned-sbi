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
| 2026-10-25 | S1 | 60283786 (`s1-gate-set`), 60283787 (`s1-gate-pet`) | 1 node × 4 A100 each, 3 h | pending | redraw re-gates (shift, convergence, 128 points) of set r2 / PET r2 → `runs/m3/{set,pet}_d256_L6_200k_multC64_redraw/` (snapshot 19ac616) |
| 2026-10-25 | S1 | 60283788 (`s1-acc-set`) | 1 node × 4 A100, 1.5 h | pending | Pr_q(S\|θ) and per-N table of set r2 → `runs/m3/acceptance/set_d256_L6_200k_multC64/` |
| 2026-10-25 | S0/S1 | 60284684, 60284685, 60284686 (`fid_boundary`, 9b648aa) | 1 node × 4 A100 each, 1.5 h | pending | `toy.boundary` at 32 gate points of set r2, PET r2, set r1 → `runs/m3/boundary/<run>/` |
| 2026-10-25 | S2 | 60283789 (`nle-s2-data`) | 1 CPU node, 45 min | pending | `data/v2_m3a`, `data/v2_m3b` |
| 2026-10-25 | S2 × S3 | ~~60283791~~ cancelled (same script bug) → 60284244 (after 60283789) | 1 node × 4 A100, 8 h | pending | `runs/m2r3/set_d256_L6_{200k,400k}_{m3a,m3b}` |
| 2026-10-25 | gates | ~~60283792–60283807~~ cancelled → 60284245–60284252 (after 60284243), 60284253–60284261 (after 60284244), snapshot 5d3e948 | 1 node each, 3 h | pending | each new model with the round-2 method (`runs/m3/m2r3_<run>`) and with redraw (`..._redraw`) |

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
