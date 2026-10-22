# NLE jet-quenching inference at scale: plan for the `dev/nle-perlmutter` branch

This file is the complete brief for the agent continuing this work on Perlmutter. Read it
fully before acting; it replaces the earlier hand-off documents. Branch: `dev/nle-perlmutter`
of `matplo/omnilearned-sbi`, created 2026-10-21 from `dev/agent-a` (commit `fe19020`). The
GPU-node state it builds on is tagged `devA-2026-10-22` in this repository and in the
omni-sbi-mc application.

Optional reading (rendered pages; this file stays self-contained):
- the full technical report with figures: https://claude.ai/artifact/DnBXmQboAAFabFmnbhzszC;
- the status page: https://claude.ai/artifact/4gzmAX82tW5xSAvLqQR7Gs.

They are private until the user shares them; ask for access if a link does not open.

## 1. Goal

Infer (b [fm], T_ref [GeV], τ_init [fm/c]) from sets of jets with a **neural likelihood
(NLE, "Route A")**:
- a conditional flow-matching generator q(cloud | θ) of the jet constituent cloud;
- its exact per-jet log density (ODE + divergence);
- summed over N jets and scanned jointly over θ.

The bar is **direct-simulator closure**: pseudo data drawn directly from the simulator at
reserved, off-grid settings give unbiased posteriors with correct coverage at N = 250 and
N = 1000 jets. The toy comes first, JEWEL later. The primary toy is **`toy-quenching-v2`**:
variable constituent multiplicity, up to 256 constituents per jet, padded with zeros and an
explicit mask. `toy-quenching-v1` (fixed 32 constituents) stays as the reference for
everything measured on the GPU node.

What is new on this branch:
- the toy is part of the repository, now with variable multiplicity (v2);
- the likelihood includes the multiplicity: q(x, N | θ) = p(N | θ) q(x | N, θ);
- training data are much larger, in a JEST-like HDF5 layout;
- **generator fidelity is established before any inference**;
- everything runs at Perlmutter scale (A100 GPUs, many CPU nodes).

We keep **our NLE approach**. The JEST project (github.com/sadekr/JEST) trains an OmniLearned
PET2 generator and then an amortized posterior (NPE). We borrow its data scale and
conditioning ideas, not its inference.

## 2. Ground rules (non-negotiable)

1. **Pseudo data:** only from the simulator at reserved settings, never from the generator.
   The one exception is the labelled *self-closure control* (§6 lesson 8): pseudo data drawn
   from the surrogate itself, to measure the pulls an exact likelihood gives. It is a
   diagnostic, never closure evidence.
2. **Training data:** no training θ equals a reserved setting. The main test setting lies
   strictly between design values on every axis (continuous designs: keep a margin, §5).
3. **Truth:** test truth is opened only by a report step after each fit's seal (hashes of
   outputs) verifies. Fit configurations contain observations, checkpoint, domain, prior and
   numerics only; `omnilearned.sbi.fit` rejects keys named truth/holdout/theta_star/test_settings.
4. **Choices on development or validation data only:** every model, numerical and scan
   choice is made there and recorded in config before any test truth is compared. Use
   the development setting (7.5, 0.30, 0.95) freely; its truth is open by design.
5. **Failed checks stay failed:** report them; never re-define a criterion after seeing a result.
6. **Numerics:** Δlog L between nearby candidates must converge (steps and probes) at the
   event count actually used, for each new model.
7. **Repository hygiene:**
   - do not commit data, checkpoints, run trees, figures or binaries;
   - never force-push;
   - add tests for every new generator, record format, driver and numerical method.
8. **JEWEL:** no production without a budget approved by the user.
9. **Asking:** ask the user only for decisions, budgets or external blockers. Keep a running
   log `FINDINGS_NLE.md` at the repository root: findings, numbers, commands, budgets, open
   items. Commit it with the code.

## 3. First day

- **Environment:** the user prepares an `henv` environment on Perlmutter with torch (CUDA),
  numpy, scipy, h5py, torchdiffeq and hepyy (FastJet 3.5 through cppyy). Do not create or
  modify environments.
- **Checkout and install:**
  - `git checkout dev/nle-perlmutter`, then `pip install -e .` into the prepared environment
    (confirmed by the user);
  - `python -c "import omnilearned.sbi.toy, omnilearned.sbi.likelihood"`.
- **FastJet:** cppyy's precompiled header must not be rebuilt in a shared environment. Set
  `CLING_STANDARD_PCH=$TMPDIR/cling-$USER/allDict.cxx.pch` per node. Check with
  `python -c "from omnilearned.sbi.toy import reco; print(reco.fastjet_version())"`.
- **Tests:** `pytest tests/sbi`. The cross-host check is `test_toy.py::test_cross_host_reproducibility`:
  θ = (6.0, 0.40, 0.65), 300 jets, seed 424242, array sha256 `63a9cf13…d24d4`, 7760 attempts.
  For v2: `test_variable_multiplicity.py::test_v2_cross_host_reproducibility`, with the same
  θ, n and seed, sha256 `b5014583…b23a8b` and 2227 attempts.
  **If either fails, stop and report**: a non-reproducing toy invalidates everything downstream.
- **Smoke chain** before scaling up:
  1. tiny data (e.g. 27 settings × 500 jets);
  2. a 1k-step train;
  3. one sample-and-select;
  4. one 2-candidate fit.

## 4. What exists in this branch

**Framework** (`src/omnilearned/sbi/`, documented in `docs/SBI.md`; tests in `tests/sbi`):
- `flow.py`:
  - `FlowSurrogate` (sample, `log_prob`, `log_prob_paired`, load/save);
  - `train_flow`: flow matching with AdamW, cosine schedule, EMA, and selection by
    validation loss of the EMA weights.
- `velocity.py`: `SetFlowVelocity`, the smooth permutation-equivariant set transformer (FiLM
  time and condition, optional condition MLP into every token). It passed the smoothness
  checks: reversibility, finite-difference divergence, permutation invariance (1.5e-8, 3e-14).
  Upstream PET2 with hard kNN failed them (divergence |Δ| up to 32, permutation gap 1.33);
  see `smoothness.py`.
- `likelihood.py`, `LikelihoodOptions`:
  - solvers euler/midpoint/rk4 with fixed steps;
  - divergence exact or Hutchinson (Rademacher/Gaussian probes keyed by stable event id);
  - `probe_schedule = "per_event"` or `"per_step"` (independent probes per evaluation;
    production choice);
  - float64 accumulation.
- `scan.py`:
  - `Grid`, `joint_scan` (dense), `refine_axes`;
  - `support_scan`: evaluates only the supported region of a fine grid by flood-fill from a
    seed sub-lattice;
  - `ScanResult` (trapezoid cell volumes, marginals, profiles, slices, intervals, boundary
    flags, `evaluated` mask);
  - `local_variable_loglik` for per-event b integration.
- `fit.py`: configuration-driven fit (`omnilearned sbi-fit`, `[scan].support` or `refine`),
  with truth-key rejection.
- `observations.py`: observation file format (features, mask, event ids, sha256).
- `base.py`, `transforms.py`, `parameters.py`: base distributions, the atanh feature
  transform with log-Jacobian, condition layout.
- `fidelity.py`: the statistics-aware fidelity gate (§7, M3). It is mask-aware:
  `evaluate(..., direct_mask=, surrogate_mask=)` adds the multiplicity as a jet observable.
- `multiplicity.py`, for variable multiplicity:
  - `MultiplicityModel`: a categorical p(N | θ) over 0..256 with smooth (SiLU) logits in θ;
  - `train_multiplicity` (cross entropy, best validation state);
  - `layout_with_multiplicity`: the flow's condition is θ plus the known context
    `n_constituents` / 100.
- `FlowSurrogate` with a multiplicity model, which is saved in the checkpoint:
  - `log_prob` adds log p(N | θ);
  - the N context is filled from the mask;
  - `sample(theta, n, None, ...)` draws N first and returns `(x, mask)`.
- **Padding:** all computation is cut to the active width (`flow.active_width`). Padding to 256
  costs nothing and changes no value (tested). Probes do not depend on the padding length.

**Toy** (`src/omnilearned/sbi/toy/`; configs `configs/toy-quenching-v1.toml` and
`-v2.toml`; v1 ported unchanged from the omni-sbi-mc application):
- `simulator.sample_toy(theta, count, rng, constants, selection, stream, index)` (v1) and
  `simulator_v2.sample_toy` (v2) return records plus a manifest (attempted / accepted /
  acceptance / rejections; v2 also the multiplicity range). v2 records hold `x` [n, 256, 3],
  `mask` and `n_constituents`;
- `simulator_v2.select_sets`: the v2 selection on constituent sets, shared by toy jets and
  surrogate samples;
- `config.load_toy_config(path)` and `sample_setting(..., path=)` dispatch on the config's
  `backend`;
- `reco` (FastJet anti-kt R = 0.4 through cppyy);
- `selection.select_clouds(x, sel, mask=None)` (the same selection applied to generated
  clouds; v2 with a mask);
- `selection.acceptance_table` (Pr_q(S | θ) of a surrogate; for v2 the surrogate draws N);
- `observables.jet_observables(x, mask=None)` (jet pT, η, width, soft-8 fraction, outer
  fraction, pTD, leading fraction, multiplicity; padded slots ignored);
- `design` (event-id streams, `rng_for(seed, *keys)`, reserved settings, `check_reservation`);
- `records` (HDF5 writers; `read_training_set(trim=True)` cuts the padded axis to the
  largest real multiplicity);
- `config.sample_setting` and `config.array_sha256`.

## 5. The toy and the inference problem

**Toy physics** (`toy-quenching-v1`; constants in `[toy]`):
- **Medium:** two disks of radius R_geom = 6.5 fm at (±b/2, 0).
  - F±(r) = max(0, 1 − |r − c±|²/R_geom²) and g_b = √(F₊F₋).
  - Production point drawn ∝ g_b²; direction uniform in azimuth; L_exit is the forward distance
    to leave the overlap.
  - T(ℓ) = T_ref · g_b(r(ℓ))^(1/3) · (τ_init/ℓ)^(1/3) for ℓ ≥ τ_init, and 0 before. There are
    interactions only where T > 0.17 GeV.
  - **T_ref is the initial temperature at the centre of a head-on (b = 0) collision**, not a
    per-event temperature. At the same T_ref, larger b gives a smaller, cooler overlap with
    shorter paths: central collisions are hottest.
- **Path moments:** I_m = ∫ 1[ℓ ≥ τ_init, T > T_cut] (T/0.35 GeV)^m dℓ/1 fm for m = 1, 2, 3,
  by a 64-step midpoint rule on [τ_init, L_exit]. I₃ drives energy loss, I₂ broadening, and I₁
  the soft fraction (and, in v2, the medium-induced soft multiplicity).
- **Hard spectrum:** P0 ~ P^−5 on [40, 500] GeV, lognormal loss exp(−0.12 I_3 Z).
- **Constituents:** 24 core + 8 soft, with Dirichlet momentum weights; soft fraction f_soft(I_1).
- **Widths:** σ_vac(P0), σ_core(I_2), σ_soft = 2.2 σ_core.
- **Jet position:** jet centre η0 ∈ [−0.5, 0.5], φ0 ∈ [−0.6, 0.6], random order of constituents.

**Selection** (`[selection]`):
- all 32 particles have |η| < 2.5 and |φ| < 1.2;
- exactly one anti-kt R = 0.4 jet, containing all 32;
- 60 ≤ pT_jet ≤ 200 GeV and |η_jet| < 1;
- acceptance 1–9% per setting.

The steep spectrum makes the selected jet-pT shape nearly θ-invariant. The information is in
broadening and the soft fraction.

**Toy v2** (`toy-quenching-v2`, the primary toy here). Medium, geometry, hard spectrum, loss,
broadening and the soft energy fraction are as in v1. What changes:
- **Multiplicities:**
  - core 1 + Poisson(μ_core − 1), with μ_core = 20 (P0 / 100 GeV)^0.3;
  - medium-induced soft Poisson(μ_soft), with μ_soft = 4 + 26 (1 − e^(−0.8 I_1)).
- **Observed object:** the leading anti-kt R = 0.4 jet's actual constituents. Particles may
  leak out of the cone and are then not observed. Constituents are pT-ordered and
  left-packed into 256 slots with zero padding and a mask.
- **Selection** (`select_sets`, identical for toy jets and surrogate clouds):
  - 2 ≤ N ≤ 256;
  - all constituents inside |η| < 2.5 and |φ| < 1.2;
  - re-clustering the constituents alone gives exactly one jet containing all of them;
  - 60 ≤ pT_jet ≤ 200 GeV and |η_jet| < 1.
- **Measured on 2000 jets per setting:**
  - acceptance 8–19%;
  - mean N from 25 (0, 0.25, 0.2) to 40 (0, 0.55, 0.2), and 30 at the development setting;
  - maximum N seen: 67;
  - every rejection is the jet-pT cut.

  The multiplicity carries information on the medium.

**Box and reserved settings:**
- box: b ∈ [0, 11], T_ref ∈ [0.25, 0.55], τ_init ∈ [0.2, 1.2];
- reserved (`[holdout]`): primary test (4.5, 0.40, 0.65); additional tests (1.5, 0.50, 0.35)
  and (10.0, 0.30, 1.0); development (7.5, 0.30, 0.95).

**Per-jet likelihood:**
- The selected density is q_sel(x | θ) = q(x | θ) 1_S(x) / Pr_q(S | θ). The generator is
  trained on selected jets, but its support leaks.
- −N log Pr_q(S | θ) comes from an acceptance table. It is small: Pr ≈ 0.88–0.92, spanning
  about 9 units of log L at N = 250.
- Fixed-vector mode: log L(θ) = Σ_i log q_sel(x_i | θ).
- With variable multiplicity (v2): log q(x_i, N_i | θ) = log p(N_i | θ) + log q(x_i | N_i, θ).
  The flow is conditioned on N, and the multiplicity model is part of the surrogate and of
  the acceptance computation, since Pr_q(S | θ) samples N first.
- Later, the shared-medium mode: b is event-local, integrated per event against the selected
  geometry law g_S(b | φ). Never multiply likelihoods before integrating b.

## 6. Lessons already learned (GPU node, TITAN RTX, 2026-10-21; development data unless noted)

1. **The posterior is a thin curved tube across the box.** b, T_ref and τ_init trade off.
   - Conditional widths at N = 250: σ_b ≈ 0.5 fm, σ_T ≈ 0.012 GeV, σ_τ ≈ 0.2 fm/c.
   - Marginal widths: about 1.4–2.4 fm, 0.08 GeV, 0.23 fm/c.
   - b–T correlation: 0.86.
   - **Cause: a physical degeneracy of the toy.** Quenching scales roughly as
     T_ref³ · G(b, τ_init), where the geometric factor G falls with b. So the jets stay the
     same along T_ref ∝ G^(−1/3), which rises with b.
     - On the GPU node, the mean I₁, I₂ and I₃ were computed over the (b, T_ref) plane at
       τ_init = 0.95. Their iso-curves through the development point nearly coincide, and every
       fitted posterior ridge lies on them.
     - Jets alone cannot separate (b, T_ref, τ_init) along that curve, and it is the same for
       every surrogate.
     - Expect the same for toy v2: the soft multiplicity also depends on I₁. Check it with the
       same iso-moment plot before assuming N adds a new direction.
   - **Consequences:**
     - mean marginal pulls at a fixed truth are non-zero even with an exact likelihood
       (lesson 8);
     - in experiments and in JEWEL, b comes from centrality, not from jets, so the
       event-local-b mode (M7) is the physically relevant fit and breaks the degeneracy.
2. **Coarse grids are artifacts.** A 5³ + 2 × 6³ refinement grid gave marginals 0.71σ off a fine
   reference, because the refinement box never shrinks. `support_scan` on fine grids fixes it.
   Nested-grid comparison (subsampling the fine reference every other node) is free and reliable.
   - At N = 250 the cheapest grid within 0.1σ was 23 × 14 × 11, with Δlog post 10 and seed
     stride (6, 3, 4), and ≈ 1300–1650 evaluations.
   - The error is not monotone in per-axis spacing.
   - N = 1000 needs ≈ 4× the nodes.
3. **Numerics** (convergence vs rk4-64 with 16 per-step probes over 12 probe seeds; criteria:
   single-seed shift, seed-averaged shift and probe std, each < 0.5 at the N used):
   - midpoint-64 with 8 per-step Rademacher probes passes at N = 250 and 1000 (shift 0.04 /
     0.03, std 0.17 / 0.20);
   - midpoint-32 with 4 per-event probes fails (solver bias grows as √N);
   - the CUDA vs CPU device check passes (Δlog L 1.9e-4).
4. **Cost:**
   - one N = 250 candidate takes ≈ 2.1 s on a TITAN RTX, and an N = 250 fit about 30–55 min;
   - one fit saturates a GPU: two concurrent fits gave 0.89× total throughput. Parallelize
     across GPUs and nodes, one fit per GPU.
5. **Fidelity was the real problem.** The 1 M-parameter model (80 grid settings × 2000 jets,
   20k steps) gives smooth likelihoods but biased physics:
   - at the development setting the surrogate's jet width is low by 5σ and the soft fraction
     by 5.7σ (4000 jets);
   - jet-pT χ² is 17–42 per 8 degrees of freedom at grid settings, the 60 GeV edge is smeared,
     and the jet-pT response is 1.44×;
   - constituent η and φ are tilted by ±5%.

   The resulting **development closure fails**: 7 ensembles at N = 250 (12 × 14 × 6 support grid).
   - Mean pulls: T_ref **+0.82 ± 0.17**, τ_init **−1.14 ± 0.28**, b −0.04 ± 0.22.
   - Truth inside the joint 68% (and 95%) region in only 1 of 7 ensembles; joint levels
     0.56–1.00 (uniform if calibrated).
   - The fidelity response curves explain the sign. Near the development point the
     surrogate's broadening (width, soft-8, outer fraction, pTD) lags the toy's as T_ref rises,
     so fits overshoot T_ref.
   - The gate's equivalent τ_init shift (+0.34 ± 0.13 fm/c with 128-step samples) matches the
     τ_init undershoot.
   - Validation log-density and flow-matching loss did not reveal any of this.
   - A permutation-calibrated gate on 22 settings × 4000 jets flags 5 settings, with
     7 of 22 p-values below 0.01.
   - Record of all this: the omni-sbi-mc application's FINDINGS.md.
5b. **Sample with a converged sampler.** 32-step samples hid part of the model error (128-step
   samples fail the gate at the development point; 32-step ones do not). The likelihood
   integrates the model's density, so fidelity must use samples whose distribution no longer
   changes with more sampler steps: check 64 vs 128 vs 256 and use the converged setting.
6. **Selection normalization** matters little at this toy's acceptance, but keep it.
7. **Data and training length fixed fidelity; capacity did not matter.** The GPU-node
   scale-up used toy v1, 1430 Cartesian settings × 1500 jets (2.1 M) and 100k steps. Tagged
   `devA-2026-10-22`.
   - `dense_base`, the same 1.0 M-parameter model, passes the gate: 0 of 22 settings fail, and
     the p-values are uniform (KS p 0.39, none below 0.01). The equivalent τ_init shift at
     128 steps is +0.14 ± 0.13 fm/c.
   - A 3.3 M-parameter model is no better; its validation loss agrees to 5 digits.
   - Development closure at N = 250: the truth is inside the joint 68% region in 6 of 7 (was 1)
     and the 95% region in 7 of 7; joint levels 0.17–0.77.
   - Selection acceptance of the surrogate: 0.96–0.98.
   - Cost: training 1.04 h on a TITAN RTX; fits ≈ 16 min at N = 250 (12 × 14 × 6 support grid).
   - Expect to need at least this data density per unit volume of θ with any new model.
8. **Mean marginal pulls are not zero even with an exact likelihood.** Fits to pseudo data
   drawn from `dense_base` itself (the self-closure control) gave mean pulls T_ref +0.52 ±
   0.08, τ_init −0.54 ± 0.12 and b −0.06 ± 0.14 at the development setting. With toy data
   they were +0.69 ± 0.16, −0.52 ± 0.07 and +0.40 ± 0.25. The difference (the fidelity effect)
   is consistent with zero.
   - Cause: the tube posterior spans the prior box, so marginal posterior means at a fixed
     truth are pulled toward the tube's centre.
   - **Judge closure by coverage and the joint-level distribution, and compare pulls with the
     exact-likelihood control, never with zero.** M5 is written that way.
   - The control is cheap: one extra dev-bias run with pseudo data sampled from the surrogate.

## 7. Work plan

Each milestone ends with tests, a `FINDINGS_NLE.md` entry (numbers, commands, wall time, GPU
hours) and a commit.

### M1: Data at scale (CPU)
- **Design:** continuous θ, e.g. scrambled Sobol in the box.
  - ≥ 1000 training points and ≥ 2000 selected jets per point;
  - plus ≥ 100 separate validation points at ≥ 1000 jets each;
  - plus a held-out set of development scans: 1D lines through (7.5, 0.30, 0.95) along each
    axis and points along the tube, as in lesson 1.
  - Exclude a margin around the reserved settings: no design point within 0.02 of the box size
    in every coordinate simultaneously. `check_reservation` covers exact equality; add the
    margin check.
- **Generation:** parallel over points with Slurm job arrays. Seeds via
  `rng_for(seed, STREAMS[split], point_index)`. One shard per job, then a merged index with
  sha256 per shard.
- **Toy:** `toy-quenching-v2` (`configs/toy-quenching-v2.toml`). Keep v1 only for a small
  reference set that reproduces the GPU-node numbers if a regression question comes up.
- **Layout** (HDF5, one file per shard, JEST-like; `records.write_training_set` already
  writes most of it):
  - `data`/`x` [n, 256, 3] float32: (η, φ, log pT/GeV), absolute coordinates, zero-padded,
    with `mask` [n, 256] and `n_constituents` [n]. Real constituents are pT-ordered and
    left-packed. Chunked gzip; padding compresses to almost nothing;
  - `global` [n, k]: θ (3), jet pT, jet η, jet φ, N_const, with `attrs["global_names"]`;
  - `event_id` int64 (stream-tagged); `point_id`;
  - hidden generator diagnostics in a separate file, never read by training or fitting.

  Readers cut the padded axis to the largest real N (`read_training_set(trim=True)`).
  2 M jets × 70 × 3 float32 is about 1.7 GB in memory. Write a reader that serves batches
  without loading everything if the data exceed host RAM.
- **Acceptance:** shard hashes reproducible; the reproducibility test passes on several node
  types; per-point acceptance recorded.

### M2: Generators (GPU)
Train **both** backbones at equal data and compute budget; the fidelity gate (M3) decides.
1. **Smooth set transformer** (`SetFlowVelocity`), scaled: dim 256–384, depth 6–8, heads 8,
   with the condition MLP into tokens.
2. **PET2 backbone** (OmniLearned `network.PET2` body: interaction features, class tokens)
   with the **hard kNN replaced by a smooth local attention**. For example, a Gaussian
   distance bias exp(−ΔR²/2s²) on attention logits, or soft top-k. It must pass the
   `smoothness.py` checks (reversibility, finite-difference divergence, permutation
   invariance) before any likelihood use: NLE needs an everywhere-differentiable velocity
   for the divergence.

Common to both:
- DDP across GPUs;
- bf16 only if the likelihood check in M4 passes at float32 evaluation (train mixed, evaluate fp32);
- EMA, cosine schedule, checkpoint by validation flow-matching loss of the EMA weights;
- log GPU hours.

Conditioning: θ mapped to [−1, 1] over the box, plus N / 100 as known context
(`layout_with_multiplicity`). Jet pT is **not** a condition in the baseline (see §9 for the
factorized variant).

Multiplicity:
- Train `MultiplicityModel` on (θ, N) of the training split with `train_multiplicity`, and
  select it on the validation split.
- Attach it to the `FlowSurrogate` before saving, so one checkpoint holds p(N | θ) and
  q(x | N, θ).
- Check it directly: for each validation point compare the N histogram with the direct toy
  (it is also part of M3).

Batching: the cost is set by the largest N in a batch, because batches are cut to the active
width. Bucket batches by N (sort-by-length sampling) to avoid paying for the tail.

### M3: Fidelity gate (CPU + light GPU sampling): must pass before any inference
- **Samples:**
  - at every validation point, ≥ 20k selected direct jets and ≥ 20k generated jets. Generated
    clouds draw N from p(N | θ) and are selected by `select_clouds(x, sel, mask)`;
  - generation with a converged sampler (lesson 5b): at least 128 steps, verified against 256
    at a few points.
- **Observables:**
  - jet level: jet pT, width, soft-8 fraction, outer fraction (ΔR > 0.15), pTD, leading
    fraction, and the multiplicity N (`JET_VARIABLE`);
  - constituent level: η, φ, log pT, ΔR to the jet axis, log z.
- **Significance (calibrated null):** per point, a permutation test.
  1. Pool direct and generated jets, re-split at random (jets as units), R ≥ 200 times.
  2. Statistic: the largest KS distance over observables, each standardized by its null median.
  3. This gives a family-wise p-value per point that is exact under "same distribution".
  4. Significant if p < α / S (α = 0.05, S points).
- **Relevance:** for each jet observable, |Δmean| √N / σ_direct at the analysis N (250 and 1000).
  Relevant if the 2-standard-error lower bound of |Δmean| exceeds 0.5 of a standard error at N.
- **Verdict per point:** FAIL only if significant **and** relevant, so statistical jitter never
  fails a model.
- **Global check:** the per-point family-wise p-values should be uniform. Report a KS test
  against uniformity and the number below 0.01 against Binomial(S, 0.01). It is a warning, not
  a failure, unless points also fail.
- **Equivalent parameter shift**, at the development setting and along the tube:
  δθ = (GᵀC⁻¹G)⁻¹GᵀC⁻¹Δmean, with direct-toy response gradients G (fits to the scans) and
  per-jet covariance C. Requirement: |δθ| ≤ 0.2 posterior σ at N = 1000, within 2 standard errors.
- **End-to-end, which decides:** development closure (M5).

The significance and relevance parts are implemented in `src/omnilearned/sbi/fidelity.py`
(ported from the omni-sbi-mc application, with `tests/sbi/test_fidelity.py`, including uniform
p-values under the null). Extend it with the δθ step at scale; the application's
`a/scripts/fidelity_report.py` and `fidelity_html.py` show one way to drive it and build the
page.

Produce a self-contained HTML report (the GPU-node report linked at the top is a template:
plots with "how to read" captions; the exact model, data and numerics behind each number):
- verdict per point;
- particle- and jet-level histograms with ratio panels;
- response curves along the development scans;
- tables.

### M4: Numerics for the chosen generator (development data)
- Convergence at N = 250 and 1000: candidates around the development fit maximum along the
  tube. Reference rk4-64 with 16 per-step probes, 12 probe seeds. Criteria as in lesson 3.
- Exact-divergence subset: 24 jets.
- The multiplicity term log p(N | θ) is exact (categorical). Report its share of Δlog L across
  the posterior and check that it is smooth in θ on the scan grid.
- Device check: CUDA vs CPU log p.
- Record `[likelihood]` settings in config before M5.

### M5: Development closure (decides the fidelity question)
- ≥ 10 independent development ensembles each at N = 250 and N = 1000.
- Support scans: choose the grid with the nested subsample check against a finer reference,
  as in lesson 2.
- Per ensemble: pulls, marginal 68/95% coverage, and the **joint level of truth** (posterior
  mass at higher density than at the truth; evaluate log L at the truth as one extra candidate).
- **Control (lesson 8):** the same fits on ≥ 10 ensembles drawn from the chosen surrogate
  itself at the development truth (labelled diagnostic, rule 1).
- **Pass** (criteria fixed now, before any v2 result):
  - joint levels consistent with uniform (KS p > 0.01), with marginal and joint 68/95%
    coverage within Clopper–Pearson 95% bounds;
  - toy-data mean pulls minus control mean pulls consistent with 0 within 2 combined
    standard errors for every parameter.
- If M5 fails while M3 passes, the gate missed something relevant: add it to M3 and iterate
  M2/M3. Never tune on test data.
- **Known-b diagnostic** (cheap, development data): repeat the development fits with b fixed
  at its true value, a 2D (T_ref, τ_init) scan. It shows how much the jets constrain the medium
  once centrality is known, and it previews M7. Report it next to the 3D result. It is not a
  replacement for the 3D closure.

### M6: Test closure
- Reserved settings: primary test 20 ensembles at N = 250 and ≥ 5 at N = 1000; additional
  tests 5 each.
- Fits are sealed, then the report opens truth: bias, Clopper–Pearson 68/95% coverage,
  predictive checks (direct toy at posterior draws vs observed).
- Parallelize: one fit per GPU, many nodes.

### M7: Shared medium with event-local b
`local_variable_loglik`: the φ = (T_ref, τ_init) grid needs the same resolution study; g_S from
an independent direct acceptance table.

This is the fit that matches how data and JEWEL define b: per-event geometry from centrality.
It removes the b direction of the degeneracy (lesson 1). Give it priority right after M6 rather
than treating it as optional.

### M8: JEWEL
Only with an approved budget. JEWEL 2.6.0-custom adapter from the omni-sbi-mc application.
- Record requested vs reported parameters, seeds and attempted vs selected counts.
- Peripheral limit from its actual geometry.

## 8. Compute notes (Perlmutter)

- **Toy:** v1, 1000 selected jets in about 1.4 s on one CPU core, so 2 M jets is about 45
  core-minutes. v2, 2000 selected jets in 0.6–2.3 s, depending on acceptance (8–19%).
  FastJet is called in a compiled cppyy loop.
- **Surrogate sampling:** 5200 jets with 32 steps take about 45 s on 6 CPU threads; use GPUs
  for the fidelity samples.
- **Fits:** a candidate's cost scales with N × (steps × NFE) × probes. Budget support scans
  at roughly 1–2 k candidates per fit at N = 250, more at 1000.
- **Account and queues** (settled with the user):
  - Slurm account `alice`, user `ploskon`. Check the exact account string with
    `sacctmgr show assoc user=$USER format=account,qos` before the first batch job.
  - Batch jobs: `sbatch -A alice -C gpu ...` (or `-C cpu` for toy generation).
  - Interactive development: the on-demand interactive QOS, up to 4 h on one GPU node,
    e.g. `salloc -A alice -C gpu -q interactive -N 1 -t 04:00:00 --gpus 4`. Use it for the
    smoke chain, debugging and short evaluations; use batch jobs for long training and fits.
- **Budget:** no fixed cap. Run what the milestones need, but log every job's node-hours in
  `FINDINGS_NLE.md` (milestone, job id, nodes × hours), and do not launch open-ended sweeps.
  JEWEL production still needs explicit approval (rule 8).
- **Storage:** large data, checkpoints and run trees live in the project directory
  `/global/cfs/cdirs/alice/$USER/jest-nle` (not `$PSCRATCH`, which is purged). Use it as
  the run root, e.g. `JEST_NLE_ROOT=/global/cfs/cdirs/alice/$USER/jest-nle`, with
  `data/`, `checkpoints/`, `runs/` and `manifests/` below it. CFS is slower than scratch:
  read each HDF5 split into memory once per job, trimmed to the largest N (2 M v2 jets ≈ 1.7
  GB), and write checkpoints at a modest cadence. Nothing from this directory is committed.

## 9. To-do and forks (not in the baseline)

1. **Factorized jet kinematics**, as in JEST: q(x | θ) = p(pT_J, η_J, φ_J | θ) × q(rel | pT_J, θ).
   - The first factor is an explicit 1D/3D density, e.g. a mixture density on
     log(pT_J − 60) with uniform η_J, φ_J.
   - The second factor is a flow over constituent coordinates relative to the jet.
   - The change of variables must be full rank for an exact density: the jet axis is a
     function of the constituents. Drop one constituent's relative (η, φ) (e.g. the leading
     one) or use another bijection, and include the Jacobian.
   - Fork from this branch once the baseline gate results are known.
2. **JEST NPE** as an independent comparison on the same data.
3. ~~A variable-multiplicity toy v2~~: done (`toy-quenching-v2`), and it is the primary toy.
   For JEWEL, the same padded layout and p(N | θ) apply with a larger N range. Check the
   256 cap against JEWEL jets first.
4. A **classifier two-sample test** per point as an additional fidelity diagnostic (never a
   replacement for the calibrated gate).

## 10. Settled with the user (2026-10-21/22)

- **Account and queue:** account `alice` (user `ploskon`); batch jobs plus the on-demand
  4 h interactive GPU QOS (section 8).
- **Budget:** none fixed; run what is needed and log usage (section 8).
- **Install:** `pip install -e .` into the prepared `henv` environment.
- **Data:** `/global/cfs/cdirs/alice/$USER/jest-nle` (project directory, not scratch).
- **Constituents** (2026-10-22): use the full variable list of jet constituents, up to 256,
  zero-padded with a mask. Toy v2 has Poisson core and medium-induced soft multiplicities
  and keeps the leading jet's actual constituents. The code lives on this branch only; the
  application keeps toy v1 at the tag.

Ask the user only for new decisions, JEWEL budget or external blockers.
