# FINDINGS_NLE: running log for `dev/nle-perlmutter`

Brief: `PLAN_NLE_PERLMUTTER.md`. Newest entries at the bottom of each section. Every result
records source revision, dirty status and device.

## Resume here (updated 2026-10-22 ~12:00 PDT, after interactive job 60098593 timed out)

State: job 60098593 ended by TIMEOUT at 05:21 (4 h). Nothing lost: code pushed, results on
CFS. Complete: M1 data (train, validation, fidelity, development scans), **all 64 counts
shards** (`data/v2_m1_counts`, 20.5 M (θ, N) pairs), the three trainings, convergence and
shift for all three. Gate points with the old p(N|θ): set 38/128, pet66k 4, pet100k 0.
The batch gate jobs 60101552 / 60101555 were **cancelled** (user decision): they would have
gated the old p(N|θ), which is known to fail.

Next session (plan approved by the user), in a new 4 h interactive GPU node:
```
salloc -A alice_g -C gpu -q interactive -N 1 -t 04:00:00 --gpus 4
export JEST_NLE_ROOT=/global/cfs/cdirs/alice/ploskon/jest-nle
export CLING_STANDARD_PCH=$TMPDIR/cling-$USER/allDict.cxx.pch
```
1. p(N|θ) on all counts shards (GPU, ~3 min):
   `multiplicity_check retrain --checkpoint runs/m2/set_d256_L6/checkpoint.pt --train-dir
   data/v2_m1 --counts-dir data/v2_m1_counts --out runs/m3/mult/set_d256_L6_multC64.pt
   --device cuda`, then `check` against `data/v2_m1_fidelity`; `attach` the same p(N|θ) to
   `pet_d256_L6_66k` and `pet_d256_L6_100k` (`multiplicity_check attach`).
2. Gate the swapped checkpoints into `runs/m3/<run>_multC64/` (`shift`, then `gate run`),
   set and pet100k on 2 GPUs each; finish with `shell/nle/gate_gpu.sbatch` at session end;
   `gate summarize --n-settings 128`.
3. In parallel on the CPUs: round-2 training data, 1024 × 6000 jets (`datasets generate
   --jets 6000` into `data/v2_m2`, same design and seeds, ≤ 16 processes for memory).
4. Round-2 trainings as batch jobs (`shell/nle/train_gpu.sbatch`, `-q shared --gpus 1 -c 32`).
5. Ready once a checkpoint passes M3: the closure driver (`toy.closure`, see "M5 tooling").
   In any GPU session, spend 5 min on the concurrency test: one campaign with
   `closure_local.sh SNAP DIR 1` vs `... DIR 2` (2 fits per GPU), compare fits/hour.

Always `$VIRTUAL_ENV/bin/python` (the NERSC conda python shadows it on PATH); batch jobs
from a source snapshot (`shell/nle/snapshot.sh`, `PYTHONPATH=<snapshot>/src`); submit from
inside an allocation with `SLURM_*` unset. On a login node run tests with
`CUDA_VISIBLE_DEVICES=` (the shared login GPU can be out of memory).

## Environment (Perlmutter)

- **Data locations** (toy and JEWEL samples, sizes, file layout, readers): `docs/NLE_DATA.md`.
- **Reports archive** (private): `matplo/omni-sbi-nle-jet-dev-reports`, clone at
  `/global/cfs/cdirs/alice/ploskon/henv-omni-sbi/omni-sbi-nle-jet-dev-reports`. HTML reports with
  their sources and inputs by topic and time stamp, findings snapshots (one per commit of this
  file), docs, all small run records of `$JEST_NLE_ROOT/runs`, data metadata and a standalone
  reading guide. Refresh with `python3 tools/sync.py --push` there after updating this log.
- Python: `$VIRTUAL_ENV/bin/python` (henv `omni-sbi`, `/global/cfs/cdirs/alice/ploskon/henv-omni-sbi/.venv`).
  Caution: in this shell the NERSC conda `python` comes first on `PATH`, so a bare `python`
  is **not** the henv interpreter. Always call `$VIRTUAL_ENV/bin/python` (or fix PATH order
  when activating; not changed by the agent).
- torch 2.14.1+cu130, CUDA available, 4 GPUs; numpy 2.5.4; Python 3.13.15.
- `omnilearned` already installed editable from this checkout (pip reports
  `22.post1.dev4+g6b628b7ea`, i.e. installed at 6b628b7; editable, so the source is HEAD).

## Job / node-hour log

| date | milestone | job id | nodes × hours | what |
|---|---|---|---|---|
| 2026-10-22 | first day, M1, M2, M3 prep | 60098593 (interactive, urgent_gpu, TIMEOUT at 4 h) | 1 × 4 | checks, smoke chain, M1 data (CPU of this node), 3 trainings, sampler timing |
| 2026-10-22 | M1 acceptance | 60101162 (cpu, debug) | 1 × 0.04 | shard regeneration on nid006833 + cross-host tests |
| 2026-10-23/24 | M5 diagnostic | 60183215, 60190975, 60203872, 60212077 (interactive chunks 1–4) | 1 × 4 + 1 × 4 + 1 × 4 + 1 × 1.35 ≈ 13.4 | acceptance table, 20 fits, reports |
| 2026-10-24 | M8 JEWEL probe | 60194817 (failed, rc 127), 60195690 | 1 × 0.2, 1 × 0.25 | debug CPU |
| 2026-10-24 | M8 JEWEL production | 60196918 (array 0–11), 60196919, top-ups 60199888 (array 0–3), 60199889 | ≈ 36 (12 × 1.8 + 4 × 3.1 + 1.1 + 0.9) | training grid (2.20 M jets) + validation (146 k jets) |
| 2026-10-23 | M2 round 2 | 60183701 / 60183702 (gpu shared, 1 GPU, 3.5 / 5 h, pending) | ≤ 0.25 × 3.5, 0.25 × 5 | set / PET 200k on `v2_m2`, memory fix |
| 2026-10-24 | M3 round-2 gate | 60219376 (interactive chunk 1; `gate_drive.sh`, up to 3 chunks) | running | set + PET 200k, 128 points each |
| 2026-10-24 | M8 JEWEL surrogates | 60220855 (interactive, 4 × A100-SXM4-80GB, nid200452) | running (≤ 4) | 4 trainings on `jewel_v1` |
| 2026-10-22 | M3 | 60101552 (gpu, regular) | 0 (cancelled while pending) | gate of `set_d256_L6` with the old p(N\|θ) |
| 2026-10-22 | M3 | 60101555 (gpu, regular) | 0 (cancelled while pending) | gate of `pet_d256_L6_66k` with the old p(N\|θ) |
| 2026-10-22 | M3, M2 round 2 | 60132411 (interactive, urgent_gpu_ss11, nid003940, ran to the 4 h limit) | 1 × 4 | p(N\|θ) on all counts, gates of `set` / `pet100k` with it, `v2_m2` data |
| 2026-10-22 | M2 round 2 | 60133655 (gpu, shared, 1 GPU, 3.5 h limit) | pending | `set_d256_L6_200k` on `v2_m2` |
| 2026-10-22 | M2 round 2 | 60133656 (gpu, shared, 1 GPU, 5 h limit) | pending | `pet_d256_L6_200k` on `v2_m2` |
| 2026-10-22 | M3 | 60141872 (gpu, regular, 4 GPUs, 1.5 h limit) | pending | rest of the `pet100k_multC64` gate |

Interactive job 60098593 used: 4 GPUs × ~4 h (3 trainings ≈ 1.3, 1.3, 1.6 GPU-h; M3
sampling, shifts, gate points and fit timing on the rest) and the node's CPUs for all toy
data (M1 ≈ 5 min, fidelity 2.6 min, development scans 2.2 min, counts ≈ 11 min).

- Slurm associations (`sacctmgr show assoc user=$USER`): accounts **`alice`** and
  **`alice_g`** (GPU allocation), QOS include debug, gpu_debug, gpu_interactive, regular. GPU
  batch jobs most likely need `-A alice_g` (NERSC `_g` convention); to confirm on the first
  GPU sbatch.
- `sbatch` from inside an allocation inherits `SLURM_*` variables (GPUs, nodes) and is then
  rejected ("Job request does not match any supported policy") for a CPU job. Submit with
  them unset: `env $(env | grep -E '^(SLURM|SBATCH|SALLOC)_' | cut -d= -f1 | sed 's/^/-u /') sbatch ...`.
- Node nid003988: AMD EPYC 7763 (Milan, AVX2, no AVX-512), 128 hardware threads, 251 GB RAM.
- FastJet through cppyy: `export CLING_STANDARD_PCH=$TMPDIR/cling-$USER/allDict.cxx.pch`
  (here `TMPDIR=/tmp`); the first import builds the PCH (~1 min) per node. Harmless warning
  "CPyCppyy API not found".
- Cold imports from CFS are slow: `import omnilearned.sbi.toy, omnilearned.sbi.likelihood`
  51 s; `omnilearned sbi-fit` CLI ~4 min before the first candidate (imports upstream
  training modules). Warm imports are seconds. Batch jobs should expect this.

## First day (§3), 2026-10-22, job 60098593, nid003988, 4 × A100-SXM4-40GB

Source: 1af6812 (clean) for the checks; drivers added on top (this commit).

- Imports OK. `reco.fastjet_version()` = **FastJet 3.5.1**.
- **Cross-host checks pass**: `test_toy.py::test_cross_host_reproducibility` and
  `test_variable_multiplicity.py::test_v2_cross_host_reproducibility` PASSED (not skipped),
  so the bitwise hashes equal the recorded AVX2 values v1 `ca4c590c…afe560d`, v2
  `86870b04…e08e2c2`; attempts 7760 / 2227. The toy reproduces on Perlmutter.
- **`pytest tests/sbi`**: 72 passed, 7 skipped (all MPS), 500 s, including the CUDA tests
  under `ieee_float32` (lesson 9).
- **Smoke chain** (toy v2; everything under `$JEST_NLE_ROOT`; GPU 0):
  1. Data: `data/smoke_v2`, Sobol design 27 train points × 500 jets (3 shards) and 8
     validation points × 500 (1 shard); ~9 s per 9-point shard incl. ~5 s import.
     Acceptance 0.043–0.182, N up to 71. No point dropped by the reservation margin.
     `python -m omnilearned.sbi.toy.datasets design|generate|index`.
  2. Train: `runs/smoke_v2/train_1k`, SetFlowVelocity dim 128, depth 4, heads 4,
     cond MLP into tokens (1.02 M parameters), batch 256, 1000 steps: **16 s on one A100**
     (FP32, no TF32); multiplicity model 1000 steps 2 s (val NLL 3.437). EMA val loss 1.274
     at step 1000 (still falling). `python -m omnilearned.sbi.toy.train`.
  3. Sample-and-select at the development setting, 2000 clouds, midpoint-128: 9.8 s sampling.
     Surrogate acceptance **0.04** (1898 `not_one_jet`, 22 `jet_pt`); the gate fails
     massively (width +44 se, outer fraction +43 se at N = 250). Expected for 1k steps on
     13.5k jets; this step checks plumbing only.
     `python -m omnilearned.sbi.toy.sample_select`.
  4. 2-candidate fit (b = 7.0, 8.0 at T_ref 0.30, τ_init 0.95), 250 development jets
     (`datasets observe`, stream development, index 1), midpoint-64, 8 per-step Rademacher
     probes: all 500 per-event values finite, **2.8 s per candidate** at N = 250 on the A100
     (TITAN RTX: 2.1 s for the smaller v1 clouds, lesson 4). log L −8787.6 / −8773.5.
- New drivers and tests (`tests/sbi/test_datasets.py`, 5 tests, 103 s): design
  reproducibility and the reservation margin, shard layout, shard file sha256 reproducible
  on regeneration (`track_times=False`), index rejects a modified shard, observation files
  carry no truth, training and sample-and-select drivers end to end on CPU.

## M1: data at scale (toy v2), 2026-10-22, nid003988 CPUs

- Design: scrambled Sobol (`datasets.make_design`, key (design_seed 10921, stream)), box
  b ∈ [0, 11], T_ref ∈ [0.25, 0.55], τ_init ∈ [0.2, 1.2]; train = first 1024 points of a
  2048-point sequence, validation = first 128 of its own 256-point sequence. Reservation
  margin (0.02 × box size on every axis at once around the four reserved settings): 0 points
  dropped in either split. `data/v2_m1/design_{train,validation}.json`.
- Generation: `shell/nle/gen_toy_local.sh` on the interactive node's CPUs (56 parallel
  processes), 16 points per shard. Seeds `rng_for(training_seed|validation_seed, stream,
  point_id)`.
  - train: 1024 points × 2000 jets = **2.048 M jets**, 64 shards, 1.25 GB; per point 1.3 s
    median (max 4.2 s); 1569 point-seconds; shard wall 23–28 s.
  - validation: 128 × 1000 = 128 k jets, 8 shards.
  - total wall 312 s.
  - Acceptance 0.042–0.195 per point (the plan's 8–19% came from a coarser scan; the low
    end is at high T_ref, small b). Max N = 81 (train), 73 (validation).
  - Rejections (train): jet pT 17 762 861, fiducial constituent 64, too few constituents 1;
    the jet-pT cut is not quite the only rejection at this scale.
  - Per-shard manifests: toy manifest, `array_sha256`, `discrete_sha256`, `float_fingerprint`
    per point; file sha256; merged `index_{split}.json` (hash-checked). Shard files
    reproduce bitwise on regeneration (test).
  - Provenance: source 0ed443f with **dirty = true**: uncommitted edits to `flow.py` and
    tests (length buckets) were in the tree; no toy or dataset file differed from 0ed443f.
- Fidelity direct samples (M3): new stream `fidelity` (9) at the 128 validation points,
  20 000 jets each (**2.56 M jets**), 32 shards, 157 s wall; `data/v2_m1_fidelity`, source
  3e7a697 (clean).
- **v2 iso-response curves** (lesson 1 check; `python -m omnilearned.sbi.toy.tube`,
  `runs/m1/tube/iso_tau0.95.json`): 12 b × 13 T_ref settings at τ_init = 0.95, 2000 selected
  jets each (stream `sensitivity`), 14 s. T_ref(b) through the development point (7.5, 0.30)
  where the mean response equals its development value:

  | b | I₁ | I₂ | I₃ | N | width | outer frac. | pTD |
  |---|---|---|---|---|---|---|---|
  | 3 | 0.2546 | 0.2563 | 0.2578 | 0.2529 | 0.2526 | – | 0.2573 |
  | 5 | 0.2664 | 0.2678 | 0.2691 | 0.2638 | 0.2675 | 0.2636 | 0.2689 |
  | 7 | 0.2914 | 0.2917 | 0.2919 | 0.2913 | 0.2945 | 0.2930 | 0.2886 |
  | 8 | 0.3132 | 0.3115 | 0.3103 | 0.3152 | 0.3085 | 0.3089 | 0.3274 |
  | 9 | 0.3576 | 0.3476 | 0.3430 | 0.3715 | 0.3691 | 0.3540 | 0.3798 |
  | 10 | 0.5293 | 0.4357 | 0.4045 | – | 0.5271 | 0.4602 | 0.5368 |

  (– : level not crossed inside the box / statistical noise.) For b ≤ 8 all curves,
  **including the multiplicity N**, coincide within ≈ 0.005 GeV: N follows I₁ and adds no new
  direction there, as expected. They separate only at b ≥ 9 (I₁ vs I₃ vs N), i.e. some
  b-information exists in peripheral collisions. The tube runs from (b ≲ 3, T_ref ≈ 0.25,
  the box edge) to (b ≈ 10, T_ref ≈ 0.40–0.53).
- **Development scans** (`data/v2_m1_devscan`, stream `development_scan` = 10): lines through
  (7.5, 0.30, 0.95) along b (12 points, 0–11), T_ref (13, 0.25–0.55), τ_init (11, 0.2–1.2)
  and 10 points on the I₃ tube at τ_init 0.95 (b = 1–10); 46 points × 20 000 jets = 920 k,
  132 s, acceptance 0.085–0.187. Held out from training; the development setting itself
  is the T_ref line's third point.
- **Reproducibility on a CPU node** (job 60101162, nid006833, Milan; snapshot 7fe5624):
  `train_00005` and `validation_00003` regenerated with **identical file sha256**, discrete
  and per-point bitwise hashes; both cross-host tests PASSED there. (Login-node run not done:
  needs a login shell; the GPU node and CPU node agree bitwise.)

## M2: generators (started 2026-10-22, interactive node)

Batch GPU queue estimates at submission time (`sbatch --test-only`): shared GPU ~31 h,
regular GPU ~5 days; so the first trainings run on the interactive node. Accounts `alice`
and `alice_g` both validate for `-C gpu`.

Common: data `v2_m1` (2.048 M train / 128 k validation jets, trimmed to 81 slots), batch 512,
AdamW lr 3e-4, 1000 warmup steps, cosine, EMA 0.9995, validation every 2000 steps (16
batches, fixed noise), **length buckets 16** (`TrainOptions.length_buckets`), multiplicity
model 4000 steps (128 × 3 SiLU), **TF32 during training** (evaluation and sampling stay IEEE
float32), condition θ in [−1, 1] plus N/100.

| run | backbone | params | steps | speed (alone / shared node) | source |
|---|---|---|---|---|---|
| `runs/m2/set_d256_L6` | SetFlowVelocity d256 L6 h8, cond MLP into tokens | 5.87 M | 100k | 36 / 45 ms | 648b25b |
| `runs/m2/pet_d256_L6_66k` | SmoothPETVelocity d256 L6 h8, local 64 | 3.56 M | 66k (equal GPU time) | 55–63 / 70 ms | 4e5945f |
| `runs/m2/pet_d256_L6_100k` | same | 3.56 M | 100k (equal steps) | 59 ms | 4e5945f |

- SmoothPETVelocity (`velocity_pet.py`): PET2 structure (interaction-feature attention
  biases, class/condition/time tokens, local embedding) with the hard kNN replaced by
  Gaussian-kernel weighted messages, regularized log distance, soft-min log pT, a learnable
  Gaussian locality bias per head. Passes the smoothness checks (FD divergence < 1e-5,
  permutation gap < 1e-9, also with near-coincident particles), exact padding invariance.
  At random init the 64-mode Fourier time embedding makes the field stiff in t (midpoint
  reconstruction converges at second order only from ~128 steps); trained fields must be
  checked with the sampler-convergence test (lesson 5b) anyway.
- **Finished**: `set_d256_L6` 100k steps 4670 s (1.30 GPU-h incl. data and multiplicity),
  best EMA val 1.05229 at step 74k, checkpoint sha fe95c53c1118…; `pet_d256_L6_66k` 66k steps
  4552 s (1.26 GPU-h), best 1.05245 at step 64k, sha dfb4fe0e8820…. Multiplicity val NLL
  3.4484 / 3.4485.
- EMA validation loss (fixed noise): SetFlow 1.05362 (10k), 1.05261 (20k), 1.05240 (48k);
  PET 1.05413 (10k), 1.05287 (20k), 1.05268 (26k). Close; per lesson 5 this loss does not
  measure fidelity — the gate decides.
- Speed note: the three concurrent trainings slowed each other (36 → 45 ms, 55 → 70 ms).
  Likely CPU oversubscription: the trainer draws noise on the CPU and each process uses
  torch's default thread count (128). Set `OMP_NUM_THREADS`/`torch.set_num_threads` per
  process (gate script does) or draw noise on the GPU in later runs.

## M3 preparation

- `toy.gate run|summarize|convergence` and `shell/nle/gate_gpu.sbatch` (4 GPUs, strided point
  shards, resumable). Verdict: significant (family-wise permutation p < 0.05/S) and relevant
  at N = 1000 (2-se lower bound of |Δmean| > 0.5 se at N); relevance at N = 250 recorded.
- Sampler convergence (lesson 5b), SetFlow intermediate checkpoint (~step 16k), validation
  point 0, 8192 paired clouds (same noise and N): max jet-observable mean shift 32→64
  0.54 se, 64→128 0.19 se, 128→256 0.10 se; KS ≤ 0.0016 at 128→256. **128 midpoint steps
  converged** for this checkpoint (repeat on the final ones).
- Sampling cost, 8192 clouds, 128 midpoint steps, IEEE float32, A100: 120 s in draw order,
  **68 s sorted by N with batch 1024** (`sample(..., sort_by_length=True)`), 73 s with batch
  2048. Gate cost per model ≈ 128 points × 22 k clouds ≈ 6.4 GPU-hours (+ selection and
  permutation on CPU).
- Surrogate acceptance at that early checkpoint: 0.74–0.76 at point 0 (lesson 7 had
  0.96–0.98 for the converged v1 model).
- **Sampler convergence, final `set_d256_L6`** (validation points 0 and 64, 8192 paired
  clouds): max jet-observable mean shift 64→128 0.11 / 0.10 se, 128→256 0.02 / 0.02 se;
  KS ≤ 0.0010 at 128→256. **128 midpoint steps** used for the gate. Surrogate selection
  acceptance 0.91–0.95 (lesson 7 v1: 0.96–0.98).
- Equivalent parameter shift (`gate shift`): G from quadratic fits to the 5 line points
  nearest the development value on each axis (test against an analytic response), C the
  per-jet covariance of the 7 jet observables at the development point, δθ =
  (GᵀC⁻¹G)⁻¹GᵀC⁻¹Δμ with its covariance. Widths from the same per-jet Fisher matrix at
  N = 1000: **conditional σ = (0.140 fm, 0.0034 GeV, 0.117 fm/c)**, marginal σ = (2.91, 0.072,
  0.215) — the degeneracy in numbers (cf. lesson 1 at N = 250: conditional 0.5 / 0.012 / 0.2).
  Probe on the *early* SetFlow checkpoint (~16k steps; diagnostic of the driver, not a
  verdict), 8192 clouds: development point δθ = (+3.9 ± 1.3, +0.092 ± 0.032, −0.35 ± 0.09),
  i.e. (+1.3, +1.3, −1.6) marginal σ, almost entirely along the tube; tube points b = 1, 2
  similar. The plan's criterion (|δθ| ≤ 0.2 σ) needs a stated σ: the conditional σ is the
  strict reading; both are recorded.
- **Fit cost of the scaled models** (`runs/m4/timing/timing.json`; intermediate checkpoints,
  same architectures; N = 250 development jets, midpoint-64, 8 per-step Rademacher probes,
  event chunk 256, IEEE float32, A100 GPU 3 while the other GPUs trained):
  **SetFlow d256 L6 8.4 s/candidate, SmoothPET d256 L6 11.1 s/candidate** (candidate chunk 4
  or 8 the same within 10%; sorting by N irrelevant at N = 250 = one chunk). Lesson 4 had 2.1
  s for the 1 M v1 model on the TITAN RTX. An N = 250 support scan of ~1300–1650 candidates
  is then **~3–4 GPU-hours (SetFlow)**; N = 1000 needs ~4× events and ~4× nodes (lesson 2),
  ~50 GPU-hours per fit. M5 as planned (≥ 10 + 10 ensembles at each N, plus controls) would
  be several hundred GPU-hours with these models. Lesson 7 found capacity irrelevant for
  v1 (1.0 M = 3.3 M); a smaller model, fewer probes or fewer steps must pass M4 numerics
  first. Decision point after the gate results.

## M3 interim results (2026-10-22, in-session, partial)

All gate numbers: 22 000 generated clouds per point (N from p(N|θ), midpoint-128, IEEE
float32, sorted by N), selected with the toy's v2 selection (≥ 20 000 kept), against 20 000
direct `fidelity` jets; 200 permutations; relevance at N = 1000. Source snapshot 86941c9.

- **`set_d256_L6` gate, 24 of 128 points** (`runs/m3/set_d256_L6/gate`, summary with
  `--n-settings 128`, α/S = 3.9e-4): **4 FAIL** (points 5, 7, 25, 31), 8 significant;
  **20 of 24 family-wise p-values < 0.01** (KS against uniform p = 7e-20). Relevant
  observables (2-se lower bound > 0.5 se at N = 1000): multiplicity N at 8 points, width at
  5, jet pT at 3, soft-8 at 3. Max relevance 3.98 (point 5, τ_init = 0.20 edge: N +12.5 se,
  soft-8 −5.7 se). Typical relevance of the jet-pT and width offsets 1.0–1.5 at N = 1000.
  Surrogate selection acceptance 0.90–0.96.
- **Multiplicity model is the largest single defect** (`multiplicity_check check`, all 128
  validation points, 20 000 direct jets each, `runs/m3/mult`): the trained p(N|θ)
  (128 × 3 SiLU, 4000 steps) has rms z 3.48, max |z| 17, binned χ² p < 0.01 at 40/128 points,
  max relevance 3.9 at N = 1000 (mean N off by +0.73 at point 5, +0.46 at points 69 and 72;
  worst near τ_init = 0.2). PET's independently trained p(N|θ) is the same (rms z 4.1).
  The gate's N offsets equal the p(N|θ) offsets (point 5: +0.726 vs +0.746 after
  selection), so the flow and the selection are not the cause.
  - Retrained (256 × 4, 30k steps, batch 8192, `multiplicity_check retrain`, 357 s on
    GPU; `runs/m3/mult/set_d256_L6_mult256x4.pt`, same flow): rms z 2.10, max relevance
    1.46, but χ² p < 0.01 at 75/128; validation NLL unchanged (3.4484 → 3.4486), i.e. the
    validation NLL cannot see these errors.
  - Passing relevance at N = 1000 needs |Δ mean N| ≲ 0.1 (σ_N ≈ 7), while the training
    data have 2000 jets per point (se of mean N ≈ 0.16).
  - **Fix confirmed: more (θ, N) data.** Counts-only shards (`datasets generate
    --counts-only`, schema `toy-counts-v1`, `data/v2_m1_counts`): 20 000 jets at each
    training point, same seeds (not a superset of the original 2000 per point; see the correction in M3
    session 2). 49 of 64 shards
    (784 points, 15.7 M pairs) finished in-session; 15 shards were killed by memory at 64
    parallel processes (each holds full padded records ≈ 5 GB; use ≤ 16 in parallel); a
    resumable rerun with 16 processes was started at 05:00 and may not have finished.
    Retrained p(N|θ) (256 × 4, 30k steps, batch 8192, 123 s on GPU; `runs/m3/mult/
    set_d256_L6_mult256x4_counts49.pt`, sha 3f368c19fd9b…, same SetFlow flow):
    **validation NLL 3.4478** (was 3.4484), **rms z 1.38, max |z| 4.1, χ² p < 0.01 at
    3/128 points (expected ≈ 1.3), max relevance 0.91 at N = 1000**, with 2-se lower bounds
    below 0.5 at the worst points (e.g. point 118: |ΔN| 0.27 ± 0.065). Next: all 64 shards,
    then gate the swapped checkpoint (flow unchanged).
- **Sampler convergence** (64 / 128 / 256 midpoint steps, 8192 paired clouds, points 0 and
  64): `set` max shift 0.11 → 0.02 se; `pet_66k` 0.23 → 0.03 se; `pet_100k` 0.06 → 0.01 se.
  128 steps are converged for all three.
- **Equivalent parameter shift** (`runs/m3/<run>/shift.json`; G from the development lines,
  development point and 10 tube points at τ_init 0.95):
  - `set`: development point δθ = (+1.72 ± 0.91 fm, +0.040 ± 0.022 GeV, −0.12 ± 0.07 fm/c),
    (+0.59, +0.55, −0.56) marginal σ at N = 1000; along the tube 0.0–0.8 marginal σ.
    Each component is within 2 se of 0, but **the se is 6.5 conditional σ along the
    tube**, so the per-parameter test has no power there. In the constrained direction,
    √(N δθᵀFδθ) at N = 1000 is 1.10 at the development point and 0.32–1.95 along the tube
    (noise level ≈ 0.55): a bias of ≈ 1 posterior σ of the best-measured combination at
    N = 1000. The plan requires ≤ 0.2 σ, so on the strict reading **this fails**.
  - `pet_66k`: development δθ = (+2.45 ± 0.91, +0.054 ± 0.022, −0.14 ± 0.07), (+0.84, +0.75,
    −0.67) marginal σ; tube 0.0–0.4 marginal σ (b = 1–6). Acceptance 0.87–0.92 (lowest).
  - `pet_100k`: development δθ = (+1.38 ± 0.90, +0.031 ± 0.022, −0.078 ± 0.067), (+0.47,
    +0.43, −0.36) marginal σ; acceptance 0.945. The best of the three at the development
    point (tube not finished before the job ended).
  - Proposed for the M3 report: quote the constrained-direction statistic
    √(N δθᵀFδθ) with its null level next to the per-parameter δθ, because the
    per-parameter criterion is uninformative along the degeneracy. (Not a re-definition of
    a criterion: no M3 verdict has been declared; both are recorded.)
- Last in-session summary (05:11): `set_d256_L6` 35/128 points, **5 FAIL** (5, 7, 14, 25,
  31), 9 significant, 29/35 p < 0.01; `pet_d256_L6_66k` 3/128 points (0 FAIL, 2 p < 0.01);
  `pet_d256_L6_100k` no gate points. All 64 counts shards finished at 05:08; the retrain on
  all 64 was cut by the job end (not run to completion).
- Interim conclusion (development and validation data only): **the first-round v2
  generators do not pass M3**. The largest defect is p(N|θ). The flow itself also shows
  relevant width and jet-pT offsets at several points (relevance 1.0–1.5 at N = 1000), as
  in lesson 5 before the data scale-up. The PET backbone at equal steps (100k) looks at
  least as good as SetFlow at the development point; at equal GPU time (66k) it is worse.
  Backbone choice waits for the complete gates.

## M3 session 2 (2026-10-22 16:00–20:00 PDT, interactive job 60132411, nid003940, 4 × A100-SXM4-40GB)

Batch-style processes run from snapshot d13fd04 (`src_snapshots/d13fd04f…`, clean). The
p(N|θ) retrain, `attach`, `check`, data generation and `index` ran from the checkout at
d13fd04 with **dirty = true**: another session's uncommitted closure work was in the tree
(`design.py` adds the stream `self_closure` = 11; `closure.py`, `closure_*.sh` untracked;
`tests/sbi/test_datasets.py`). None of it is imported by these drivers and no existing
stream number changed.

- **p(N|θ) on all 64 counts shards** (20.48 M (θ, N) pairs; 256 × 4, 30k steps, batch 8192,
  GPU, 124 s): best validation NLL **3.44784** at 29k; `runs/m3/mult/set_d256_L6_multC64.pt`
  (sha 25fa66a1bb5d…, flow = `set_d256_L6`). Attached unchanged to the PET flows:
  `pet_d256_L6_66k_multC64.pt` (97ded917694c…), `pet_d256_L6_100k_multC64.pt` (4e6ecd8b23ac…).
- `check` against `v2_m1_fidelity` (128 points × 20 000 jets): **rms z 1.27, max |z| 4.6,
  χ² p < 0.01 at 2/128** (≈ 1.3 expected), max relevance 1.03 at N = 1000. Same quality as
  the 49-shard model (rms z 1.38, 3/128, 0.91): more counts no longer help, the residuals
  are at the noise level except at box corners. Worst: point 98 (b 10.5, T_ref 0.54, τ_init
  0.22; ΔN −0.224 ± 0.049, relevance 1.03, 2-se lower bound ≈ 0.58), point 118 (ΔN −0.20,
  0.70), 69 (+0.11, 0.69). `runs/m3/mult/check_set_d256_L6_multC64.json`.
- Gates and shifts of the swapped checkpoints: `runs/m3/set_d256_L6_multC64/`,
  `runs/m3/pet_d256_L6_100k_multC64/` (launcher `runs/m3/launch_multC64.sh`: GPU A shift then
  gate shard 0/2, GPU B gate shard 1/2; GPUs 0–1 set, 2–3 pet100k). Results below.
- **Round-2 data `data/v2_m2`**: same train design and seeds as `v2_m1`, **6000 jets per
  point**, 1024 points = **6.144 M jets**, 64 shards, 3.6 GB, 6 min 45 s with 16 processes;
  acceptance 0.042–0.192. Validation shards and index copied from `v2_m1` (unchanged).
- **Correction**: a larger jet count is *not* a superset of a smaller one at the same seed.
  `sample_toy` uses batch = min(20000, max(256, 2 × count)) attempts per vectorized draw, so
  the map from the random stream to events depends on `count`. Checked on `train_00007`:
  of the 2000 `v2_m1` jets per point only ≈ 300 event ids recur in `v2_m2`, and with
  different content. The earlier note that the counts shards contain the original 2000 per
  point is wrong in the same way. Consequences: none for the statistics (the draws are
  effectively independent, validation/fidelity use other streams), but **event ids are
  unique only within one dataset**, not across datasets with different jet counts.
- **Round-2 trainings** submitted (snapshot d13fd04, shared GPU queue, estimated start
  ~30 h): `runs/m2r2/set_d256_L6_200k` (60133655) and `runs/m2r2/pet_d256_L6_200k`
  (60133656, `--arch pet_smooth --local-dim 64`), both on `v2_m2`, **200k steps** (equal
  steps; round 1 showed PET needs equal steps rather than equal GPU time), otherwise the M2
  settings (d256 L6 h8, batch 512, lr 3e-4, warmup 1000, cosine, EMA 0.9995, length buckets
  16, TF32 training, val every 2000 × 16 batches). 200k × 512 = 17 epochs of `v2_m2` (round
  1: 25 epochs of `v2_m1`). Their own p(N|θ) (4000 steps) will be replaced by
  `set_d256_L6_multC64` via `attach` before the gate.
- **Shifts with the new p(N|θ)** (`gate shift`, 22 000 clouds, midpoint-128; development
  point and 10 tube points). Constrained-direction statistic √(N δθᵀFδθ) at N = 1000
  (null level ≈ 0.55):

  | run | development | tube b = 1…10 | max |
  |---|---|---|---|
  | `set` old p(N) | 1.10 | 1.95 1.84 0.99 0.90 0.35 0.32 0.95 0.76 1.37 0.96 | 1.95 |
  | `set_multC64` | 1.32 | 0.78 1.01 0.30 0.58 0.60 0.60 1.18 0.79 0.66 0.79 | 1.32 |
  | `pet100k` old p(N) | 1.01 | 1.81 1.94 1.12 0.98 0.31 0.26 0.86 0.73 1.28 0.82 | 1.94 |
  | `pet100k_multC64` | 1.27 | 0.71 0.99 0.25 0.46 0.40 0.43 0.97 0.86 0.60 0.77 | 1.27 |

  The tube improves (worst 1.9 → 1.0–1.2); the development point does not. Per-parameter δθ
  at the development point: `set` (+2.05 ± 0.91 fm, +0.047 ± 0.022 GeV, −0.099 ± 0.067),
  `pet100k` (+1.86, +0.042, −0.078), i.e. 0.6–0.7 marginal σ in b and T_ref; the shifts are
  now coherent along the tube (+b, +T_ref, −τ_init at every point). Still a fail on the
  strict 0.2 σ reading.
- **What is left at the development point is the flow, not p(N|θ).** Mean offsets of the
  selected surrogate jets there: jet pT **+0.52 GeV** (`set`) / +0.21 (`pet100k`), N −0.18 /
  −0.21. p(N|θ) itself at that point is right: `multiplicity_check check` against the
  development scans (`runs/m3/mult/check_set_d256_L6_multC64_devscan.json`, 46 points ×
  20 000 jets): ΔN −0.03 ± 0.05 at (7.5, 0.30, 0.95); all 46: rms z 1.56, χ² p < 0.01 at 1,
  worst at the b = 11 box edge (+0.24, z 6.8). The N deficit after selection comes from
  the surrogate's own selection losses (acceptance 0.94: 874 `not_one_jet`, 464 `jet_pt`
  of 22 000), which remove high-N clouds preferentially. Both defects belong to the flow:
  round 2 (`v2_m2`) is the fix to test; the surrogate acceptance is the quick indicator.

- **End of session 2 (19:58, job 60132411 reached its 4 h limit):** `set_d256_L6_multC64` gate
  complete: **128/128, 9 FAIL, 22 significant, 104/128 p < 0.01 → M3 fails** (`summary.txt`).
  FAILs on width/outer fraction and jet pT (points 13 27 37 44 50 56 84 98 117; max relevance
  2.6 at point 84); N relevant only at the box corner 98. The flow, not p(N|θ), now fails. `pet100k_multC64`
  84/128; the rest submitted as **60141872** (`gate_gpu.sbatch`, regular, 4 GPUs, 1.5 h,
  snapshot d13fd04; resumable, skips finished points). The closure concurrency test was
  prepared but not run (`runs/m5/conc/{spec.toml,run.sh,per1,per2}`; `run.sh 1` then
  `run.sh 2` on GPUs 0–1, compare seal timestamps per worker).

## M5 tooling: closure campaigns (2026-10-22, login node; code be669bd + fixed-axis fix)

`python -m omnilearned.sbi.toy.closure prepare|fit|report|acceptance|acceptance-merge`
(tests in `tests/sbi/test_datasets.py`; template `configs/closure/dev_N250.example.toml`):
- `prepare`: K ensembles at a reserved setting named by its `[holdout]` key (only the driver
  reads the value), from the toy (`source = "toy"`; development stream, or test stream for
  test settings) or from the surrogate (`source = "surrogate"`, stream `self_closure` = 11,
  labelled control). Truth-free observation files, `truth/` kept apart, one fit config per
  ensemble (checked by `check_fit_config`), optional Pr_q(S|θ) table (grid must then be in
  (b, T_ref, τ_init) order).
- `fit`: worker w of W runs tasks w::W, resumable; each fit is sealed (`seal.json`: sha256
  of result json/npz, fit config, observations, checkpoint, table; completeness flags).
- `report`: refuses any fit whose seal no longer verifies (tested by editing a result), then
  opens truth: pulls, marginal 68/95% coverage, **joint level of the truth** (log posterior
  at the truth evaluated as one extra candidate with the fit's numerics, probes and
  normalization; mass of grid cells above it; checked against an analytic 2D Gaussian),
  Clopper–Pearson bounds, KS of the levels, `--control` pull differences with combined se.
  Single-node (fixed) axes get no pull.
- `acceptance`: CRN acceptance table, sharded over GPUs (`--shard/--num-shards`), merged
  into a `selection.save_table` file for `TableLogNorm`. Cost: 8192 clouds × 128 steps
  ≈ 70 s per grid point per A100, so a 6 × 4 × 4 table ≈ 30 min on 4 GPUs.
- Parallelism: `shell/nle/closure_node.sbatch SNAP DIR [PER_GPU]` runs 4 GPUs × PER_GPU
  workers per node; as a job array (`--array=0-(nodes-1)`) the workers spread over nodes.
  `closure_local.sh` does the same inside an interactive allocation. Smoke test on the
  login node (4 workers, 4 × 20 jets, CPU, `runs/m5/smoke_local`): 4 fits sealed in 6 s,
  report verified 4/4.
- Not in the driver yet: the M5 grid choice (nested subsample against a finer reference,
  lesson 2) — a template grid is given; M4 numerics for the chosen generator.

## M5 development closure, diagnostic (started 2026-10-23 18:48 PDT)

User decision (2026-10-23): run the development closure now with the round-1 generator, as
the end-to-end test of whether the gate's residuals (9/128 FAIL, 1–3% mean offsets) bias
inference. Development data only (rule 4); the generator has **not** passed M3.
- Campaign `runs/m5/dev_N250_setC64` (runbook `docs/M5_DEV_CLOSURE.md`): checkpoint
  `runs/m3/mult/set_d256_L6_multC64.pt`; 10 toy ensembles (development stream, indices
  1000–1009) and 10 self-closure control ensembles (stream 11, 2000–2009), N = 250;
  lesson 2's v1 support grid (23 × 14 × 11, stride 6/3/4, Δ 10; **no v2 nested-grid check**);
  midpoint-64, 8 per-step probes; selection normalization from a 6 × 4 × 4 acceptance table
  (8192 clouds per point). Code f3544f2/76c2c4e (snapshot 76c2c4e): resumable fits
  (`fit.LogpCache`), dynamic claiming across allocations.
- Driven unattended from login34 (screen) by `shell/nle/m5_drive.sh`, which requests 4 h
  interactive GPU nodes back to back (max 8). Chunk 1 = job **60183215** (nid200416,
  4 × A100-SXM4-80GB), started 18:48.
- Chunk 1: PCH build 5 min; acceptance table 18:53–19:35 (96 points × 8192 clouds on 4 GPUs;
  Pr_q(S|θ) 0.890–0.969); toy and control prepared by 19:46; fits from 19:46, workers claim
  interleaved (toy 0, control 0, toy 1, control 1). First cache flushes: ≈ 9 s per candidate
  per GPU (A100-SXM4-80GB, 10 GB used), i.e. ≈ 3.5–4 h per fit; fits cross chunk boundaries
  via `logp_cache.npz`. Estimate ≈ 18–19 node-hours in total, 5–6 chunks.
- Concurrency test (plan item): chunk 1 runs 1 fit per GPU, chunks ≥ 2 run 2 per GPU
  (`PER=2` in `m5.env`); compare candidates per hour from the cache growth.
- `PYTHONUNBUFFERED=1` added to `m5.env` after chunk 1 started (worker logs of chunk 1 are
  block-buffered).
- Round-2 trainings resubmitted with the memory fix (b4e4c64): **60183701** (set, 3.5 h) and
  **60183702** (PET, 5 h), shared GPU QOS, same settings as 60133655/60133656. Both started
  ≈ 07:25 (2026-10-24) after ≈ 31 h in the queue and passed data preparation (6.14 M jets
  × 83 slots; the memory fix works): set ≈ 29.5 ms/step (200k ≈ 1.6 h), PET ≈ 46 ms/step
  (≈ 2.6 h). Set EMA val 1.05364 at 10k (round 1: 1.05362 at 10k).
- **Round-2 trainings done:** set 1 h 45 min, best EMA val **1.05223** at 156k (round 1:
  1.05229), checkpoint sha 72d3198c1dec; PET 2 h 39 min, best 1.05228 at 162k, sha
  1e921f103ec3. The all-counts p(N|θ) attached (`runs/m3/mult/{set,pet}_d256_L6_200k_multC64.pt`,
  sha 1d587f3962eb / 7a3f7b5e66bd). **Round-2 gate** (convergence, shift, all 128 points, both
  models) driven by `shell/nle/gate_drive.sh` (logs `runs/m3/r2_gate`), chunk 1 = job
  60219376 from 10:16, snapshot ecab59b.
- Round-2 sampler convergence: 128 → 256 steps ≤ 0.05 se (set), ≤ 0.03 se (PET).
- **Interim round-2 gate (set, 15 points, same points as round 1):** significant at α/128
  4 → **0**, p < 0.01 12 → **3** of 15, surrogate acceptance 0.90–0.96 → **0.93–0.97**; mean
  largest shift at N = 1000 0.89 → 0.81 (the size of the residual offsets shrinks only a
  little; point 1 still has width/outer fraction relevant at 1.36 but is not significant).
- Prediction from the gate's parameter shift: constrained-direction bias ≈ 0.6 posterior σ
  at N = 250.
- **Chunk 1 (60183215) ended at its limit (22:48, rc 143) with 4/20 sealed** (toy 0–1, control
  0–1). Fits are cheaper than estimated: 824–939 of 3542 grid nodes evaluated (support
  closed after 8–9 passes), **6700–7500 s per fit** (≈ 8 s per candidate). Toy 2–3 and
  control 2–3 were 40–55% done and resumed from their caches in chunk 2 (60190975, 22:50).
  Revised total ≈ 20 × 2 h / 4 GPUs ≈ 10–11 node-hours.
- **Claim race at the start of chunk 2:** with 8 workers, several reclaimed the same stale
  chunk-1 locks at once, so toy fit 2 and control fit 2 each ran twice (cost only; the
  duplicates compute identical values). One of each pair was killed (SIGKILL, lock kept).
  Fixed in eacdb33 (workers no longer reclaim; one `unlock-stale` step before workers;
  concurrency test with 8 claimers), used from chunk 3 via `m5.env` (`SNAP`). Chunks 1–2 ran
  76c2c4e; the likelihood code is identical.
- Interim (2 + 2 fits, no conclusions): control levels 0.04, 0.47; toy levels 0.60, 0.36.
- **Concurrency test (chunk 2, A100-SXM4-80GB), from cache growth 23:01–23:58:** one fit
  alone on a GPU ≈ 6.7–7 candidates/min (toy fits 2 and 4); two fits sharing a GPU ≈ 2.8/min
  each, **5.6/min per GPU, i.e. 0.8× of one fit per GPU** (as lesson 4 on the TITAN RTX).
  Chunks ≥ 3 run 1 fit per GPU (`PER=1`).
- Chunk 2 (60190975) ended at its limit 02:50 with 10/20 sealed; chunk 3 (60203872, nid001480,
  4 × A100-SXM4-40GB) started 02:51 with the claim fix: "removed 6 stale claim locks" once, 4
  workers, 4 distinct claims.
- **Interim at 10/20 (6 toy, 4 control; too few for conclusions, criteria apply to 10 + 10):**
  toy: joint 68/95% 4/6, 6/6, levels KS p 0.96, mean pulls b −0.47 ± 0.27, T_ref −0.04 ± 0.22,
  τ_init −0.03 ± 0.10; control: joint 4/4, 4/4, KS p 0.14, pulls b −0.10 ± 0.09, T_ref
  +0.38 ± 0.05, τ_init −0.31 ± 0.06 (the tube/box pattern of lesson 8's v1 control).
- Also: the round-2 PET training 60133656 ran out of host memory (like 60133655). Fixed in
  b4e4c64 (chunked standardization and float32 preparation; peak host memory ≈ data size).

## M8 preparation: JEWEL worker (2026-10-23/24, login node; code 27f28f0)

User decisions (2026-10-23): Pb–Pb 5.02 TeV, pT-hat 50–400 GeV, **recoils on**, θ = (b via fixed
centrality, TI, TAUI), box b 0–11 fm, TI 0.1–0.9 GeV, TAUI 0.2–1.0 fm/c, toy-v2 jet selection;
PDF **EPPS21nlo_CT18Anlo_Pb208** (LHAPDF 904400; the user installed the PDF sets); no HepMC
kept, jets + full events stored; porting the private adapter and subtraction kernel into this
public repository approved. Production still needs an approved budget (rule 8).
- Installation (henv, inspected only): JEWEL 2.6.0-custom, patches partons-v1 (status-23 hard
  partons, PARI(17) as event scale) + perf-v1 (speed only). Neither touches subtraction.
  JEWEL 2.6.0 itself has an in-generator constituent subtraction (`DOSUBTRACTION`, `RSUB`
  default 1.0, `PUTONSHELL`); with it on, JEWEL writes no scattering centres. We subtract at
  analysis level instead (Rivet `SubtractedJewelEvent` port, ΔR_max 0.5), keeping the raw
  record re-subtractable.
- Card checks against the installed source: medium labels are TI, TAUI, TC, WOODSSAXON,
  CENTRMIN, CENTRMAX, NF, N0, D, SIGMANN, MDFACTOR, MDSCALEFAC; the nucleus is `MASS`/`NPROTON`
  in the main card; `PDFSET` is read as an integer (a set name is silently ignored). A setup
  suggested by another model used PROJ, TAR, BEAMENERGY, CENTRALITY, RECOIL and a PDF name:
  none of those is a JEWEL 2.6.0 key.
- EPPS21 with JEWEL: JEWEL generates pp, pn, np and nn channels (weights from NPROTON/MASS) with
  neutrons from isospin-swapped proton PDFs; consistent with bound-proton nPDFs (the EPPS
  convention). To be confirmed in the JEWEL log (set loaded, cross sections).
- Code `omnilearned.sbi.jewel` (geometry, cards, hepmc, cs, jets, generate), config
  `configs/jewel-2.6.0-cs-v1.toml`, farm `shell/nle/jewel_farm.sbatch`, probe
  `shell/nle/jewel_probe.sbatch`, tests `tests/sbi/test_jewel.py` (7, incl. the worker end to
  end with a fake JEWEL and the event-store round trip). Grid 12 × 9 × 9 = 972 training
  points; holdouts at half steps: development (7.5, 0.35, 0.85), primary test (4.5, 0.45,
  0.65), additional (1.5, 0.65, 0.35), (9.5, 0.25, 0.45); 128 Sobol validation points.
- **Probe submitted: job 60194817** (1 CPU node, debug, 30 min): `runs/jewel/probe1/main`
  (10 settings × 50 events: TI/TAUI corners at b = 0 and 11, centre (5.5, 0.5, 0.6), the
  setting that stalled before (4.5, 0.45, 0.70)), `custom_simple` (identity check at the
  centre, same NJOB) and `jewel_cs` (JEWEL's own subtraction, RSUB 0.5, same NJOB).

- **Probe attempt 1 (60194817) failed** every task with rc 127: the nested `henv` launcher was
  not on PATH, and inside henv the JEWEL binary did not find `libLHAPDF.so`. Fixed (f54a334):
  the JEWEL subprocess sources the installed `lhapdf` and `jewel` `env.sh` (nothing modified).
- **Probe (60195690, nid004171, ≈ 15 min of a debug CPU node), 50 requested events per task
  (JEWEL's pp/pn/np/nn split delivers 47–49):**

  | (b, TI, TAUI) | centrality % | s/event | selected | N range | event store kB/event |
  |---|---|---|---|---|---|
  | (0, 0.1, 0.2) | 0.0 | 0.57 | 19% | 11–34 | 3.4 |
  | (0, 0.1, 1.0) | 0.0 | 0.59 | 9% | 10–23 | 3.2 |
  | (0, 0.9, 0.2) | 0.0 | 0.91 | 10% | 26–48 | 36.1 |
  | (0, 0.9, 1.0) | 0.0 | 1.18 | 8% | 27–45 | 44.8 |
  | (11, 0.1, 0.2) | 51.1 | 0.57 | 12% | 11–26 | 3.3 |
  | (11, 0.1, 1.0) | 51.1 | 0.59 | 6% | 7–24 | 3.3 |
  | (11, 0.9, 0.2) | 51.1 | 0.82 | 13% | 17–54 | 17.2 |
  | (11, 0.9, 1.0) | 51.1 | 0.88 | 4% | 13–21 | 26.4 |
  | (5.5, 0.5, 0.6) | 13.0 | 0.86 | 12% | 17–56 | 23.7 |
  | (4.5, 0.45, 0.70) | 8.8 | 0.75 | 2% | 9 | 21.2 |

  - **No stall** at any setting, including (4.5, 0.45, 0.70), which stalled on the old GPU node.
  - Seconds per event include the one-off table setup (≈ 20–25 s per task).
  - Selected fraction over all 478 events: **9.6%**. Rejections re-derived from the event
    stores: jet pT outside 60–200 (239; median leading-jet pT 43 GeV), constituent
    fiducial (101; forward jets, median |η_jet| 2.7, ETAMAX 3.1 partons), not one jet
    (62; soft, median 47 GeV), jet η (30). The low fraction is the steep pT-hat spectrum.
  - Multiplicities well below the 256 cap (max 56).
  - **Subtraction validated on real 2.6.0 output:** custom-simple + our Python port
    (ΔR_max 0.5) and custom-perf-simple + JEWEL's built-in subtraction (RSUB 0.5), same
    NJOB, give **identical selected jets** (same event ids and multiplicities, jet pT
    equal, constituents within 5e-7 = float32 rounding). This also confirms that
    custom-perf generates the same events as custom.
  - CFS project `alice`: 738 TB of 1.9 PB used, so a few hundred GB of event stores is fine.
    **Home is 97.3% full** (38.9/40 GiB); nothing of this work writes there.
- **Production (user: "generate the jewel on CPU alloc", 2026-10-24), started from snapshot
  c4cf179:** training grid 972 points × 21 tasks × 1000 events (20.4 M events, target ≈ 2000
  jets per point at 9.6%) in `data/jewel_v1/train`, chunk-major task order; validation 128
  Sobol points × 11 tasks (1.4 M events) in `data/jewel_v1/validation`. Jobs **60196918**
  (array 0–11, regular CPU, 4 h each, 128 workers per node, 1 h task timeout) and **60196919**
  (1 node). Expected ≈ 40 CPU node-hours (request ≤ 52) and ≈ 0.4 TB of event stores.
- **Production measurements (chunk 0 of all 972 points, 1000 events per task):** the probe's
  s/event was dominated by the ≈ 25 s table setup; real cost and acceptance by TI:

  | TI [GeV] | JEWEL s/task (median / max) | acceptance (median / min) | max N | MB/task |
  |---|---|---|---|---|
  | 0.1 | 29 / 31 | 13.4% / 11.2% | 66 | 2.7 |
  | 0.3 | 176 / 210 | 9.3% / 6.4% | 72 | 8.4 |
  | 0.5 | 282 / 348 | 7.2% / 4.4% | 101 | 19.7 |
  | 0.7 | 377 / 468 | 6.5% / 3.9% | 120 | 27.9 |
  | 0.9 | 474 / 616 | 5.5% / 3.3% | 146 | 35.1 |

  Acceptance falls with TI (more jets quenched below 60 GeV); at TI 0.9 central b slightly
  lower than peripheral (4.7% vs 6.3%). Max N 146, well below 256. 21 tasks per point give
  ≈ 2800 jets at TI 0.1 but ≈ 1150 at TI 0.9.
- **Validation sample complete (01:56):** 1909 tasks, 1.90 M events, **146 016 jets**, 128
  points with 947–1557 jets (median 1104; 10 points just below 1000 from acceptance noise),
  157 JEWEL core-hours. Main training array 60196918 completed (12 × 1 h 47 min = 21.5 node-h);
  validation 60196919 1 × 1 h 06 min.
- **Training sample complete (03:40):** `data/jewel_v1/train`, 28 362 tasks, 28.30 M events,
  **2 199 271 selected jets**, 2356 JEWEL core-hours; per grid point 1891–2943 jets (median
  2167; 53 of 972 points slightly below 2000 from acceptance noise). Jet shards 1.33 GB,
  **raw event stores 601 GB** (above the 0.4 TB guess; CFS project `alice`, shared space).
  No task failed. CPU node-hours: main array 21.5, top-up array 4 × 3.13 = 12.5 (of which
  ≈ 4 × 0.45 h only re-ran the final index: every array element hashed all 28 k shards on CFS;
  cancelled at 04:08, index intact; the farm now indexes only on request), validation 1.1 +
  0.9; **≈ 36 in total** (estimate ≤ 52). `toy.train` now takes the JEWEL box and selection
  from `--config configs/jewel-2.6.0-cs-v1.toml` and validation from `--val-dir` (e6153d0).
  Not yet done: p(N|θ) and gate-style checks on JEWEL; a JEWEL training run.
- **Sanity of the JEWEL response (3 of 21 tasks per point, 243 k jets, ≈ 150–400 jets per
  point, noisy):** ⟨N⟩ 19.8 → 41.3 for TI 0.1 → 0.9 (b = 0, TAUI 0.6), width 0.089 → 0.100,
  soft-8 fraction 0.14 → 0.03 (more constituents share the pT); ⟨N⟩ 31.8 → 27.0 for b 0 → 11
  (TI 0.5); ⟨N⟩ 27.0 → 35.7 for TAUI 0.2 → 1.0 (later TAUI = longer hot phase in JEWEL's
  profile). The multiplicity (recoil medium response surviving the subtraction) is the
  strongest single response.
- **Top-up (00:58, `generate topup`, code 148d29e):** per-point chunks from the measured
  acceptance to reach ≥ 2000 jets (training, +5% margin; +7950 tasks, 28362 in total) and
  ≥ 1000 (validation, +501 tasks, 1909 in total); new NJOB streams continue each point's
  chunk numbering. Jobs **60199888** (`jewel-train-topup`, array 0–3, 4 h) and **60199889** (`jewel-val-topup`,
  2 h). Revised total for the JEWEL training sample ≈ 18 CPU node-hours.

- **JEWEL surrogate trainings, round 1 (started 2026-10-24 11:35, user request):** job
  **60220855** (interactive, nid200452, 4 × A100-SXM4-80GB), `shell/nle/jewel_train_chunk.sh`
  from snapshot bfdf3d5 (clean), output `runs/jewel_r1/`. One training per GPU on
  `data/jewel_v1/train` (2.20 M jets, 972 points), validation `data/jewel_v1/validation`
  (146 k jets, 128 points), toy round-2 settings (d256 L6 h8, 200k steps × batch 512 ≈ 47
  epochs of the JEWEL sample, lr 3e-4, warmup 1000, cosine, EMA 0.9995, length buckets 16,
  TF32, best EMA val checkpoint from 10k): `set_d256_L6_s0`, `pet_d256_L6_s0` (smooth PET,
  local dim 64), and the same two with seed 1 (the seed pair gives the training-to-training
  spread to judge later gate and shift differences against). Then p(N|θ) on all JEWEL
  training counts (256 × 4, 30k steps, `multiplicity_check retrain --val-dir`, new option),
  attached to each flow (`runs/jewel_r1/mult/<run>_multJ.pt`) and checked against the
  validation split. Not resumable across allocations; expected 2–3 h (set 1.7 h, PET 2.7 h in
  round 2). Still missing for an M3-style JEWEL gate: fidelity samples (≈ 20 000 jets per
  point; the validation split has 950–1560).

- **JEWEL training restart (12:03):** in 60220855 the four trainings spent 22 min opening
  5400 of the 28 362 per-task shards each. Opening files on CFS ran at about 4 per second per
  process, with CPUs idle and GPUs empty; loading alone (two read passes plus hashing) would
  have taken more than the 4 h allocation.
  - **Merge:** new `jewel.generate merge` (7d28866, tested) merges each split into point-major
    shards with the same layout: 64 for train and 8 for validation, in `data/jewel_v1/merged`.
    Every source is hash-checked against the split index, and each merged manifest lists its
    sources.
  - **Restart:** the old trainings were stopped with SIGSTOP; they hold the allocation's first
    job step. The new run is an overlap step from snapshot 7d28866, output `runs/jewel_r1b/`:
    merge with 32 workers, then the same four trainings. `runs/jewel_r1/` holds only the
    aborted attempt's logs.
  - **Clean-up at the end:** SIGKILL the old step's bash (pid 1433624) first, then the stopped
    trainings. The checkout copy of `jewel_train_chunk.sh` that bash is executing has changed
    since, and it must not resume reading it.
- **Event-store status 4 = the two beam protons** (pid 2212, E = 2510 GeV, pT = 0).
  - The subtraction's "final state" is status 1 or 4, the reference convention.
  - Re-deriving three tasks with and without status 4 gives identical jets.
  - Re-deriving from the float32 event store reproduces the stored jets to ≤ 6.4e-6 in
    (η, φ, log pT).
  - `docs/NLE_DATA.md` is corrected, and now has a numpy + h5py reading section.

- **Round-2 gate, set model, complete (14:00).** `set_d256_L6_200k_multC64` at all 128
  validation points; `gate summarize --n-settings 128` from snapshot ecab59b; job 60219376,
  A100. Compared with round 1 (`set_d256_L6_multC64`, the same points):

  | | round 1 | round 2 |
  |---|---|---|
  | FAIL (significant at α/128 and relevant) | 9 | **1** |
  | significant at α/128 = 3.9e-4 | 22 | 1 |
  | p < 0.01 (1.3 expected) | 104 | 23 |
  | uniformity KS p of the 128 p-values | 1e-103 | 7e-49 |
  | mean largest relevance at N = 1000 | 1.05 | 0.91 |
  | settings where width / outer fraction / jet pT is relevant | 35 / 23 / 11 | 22 / 15 / 6 |
  | surrogate acceptance | 0.90–0.96 | 0.92–0.97 |

  - **The one FAIL is point 50**, (9.77, 0.527, 1.16), a box corner (large b, high T_ref,
    late τ_init). Its p = 3.90e-4 sits just below the threshold 3.906e-4; width, outer
    fraction and jet pT are relevant there (largest 1.55). By rule 5 it stays a FAIL, so
    **the round-2 set model does not formally pass M3**.
  - **The residual is collective.** p-values are still not uniform (23 below 0.01).
  - **Same pattern as round 1:** the generated jets are slightly too narrow at high T_ref,
    and the effect is weaker.
  - **Equivalent shift at the development point** (computed earlier): constrained direction
    1.17 (round 1 ≈ 1.3; noise ≈ 0.55); marginal shifts at N = 1000 b +0.53σ, T_ref +0.48σ,
    τ_init −0.30σ.
- **PET gate** is at 88 of 128 points (14:00). Extra PET workers now also run on the set
  model's GPUs (`runs/m3/r2_gate/extra_gpus.sh`, an overlap step in 60219376 from 13:41;
  explicit `--points` blocks 103–127 odd on GPU 1 and 104–126 even on GPU 0, same snapshot
  and seeds). The rest goes to driver chunk 2.
- **JEWEL set trainings done (13:59).** `set_d256_L6_s0`: best EMA val 1.19415 at 178k, sha
  3932a3053e22. `set_d256_L6_s1`: 1.19412 at 200k, sha e60d5b8216fc. Each took 1.72 h on an
  A100-80GB, 30 ms/step. The seed pair agrees to 3e-5 in validation loss, and the curves are
  flat from about 120k (1.1943). The JEWEL flow NLL is not comparable to the toy's (different
  feature normalisation and multiplicities). The p(N|θ) retrain on all JEWEL counts follows
  in the same job.

### M5 development closure result (DONE 2026-10-24 08:14; diagnostic, round-1 generator)

Campaign `runs/m5/dev_N250_setC64`, checkpoint `set_d256_L6_multC64`, N = 250 at the
development setting (7.5, 0.30, 0.95); 4 interactive chunks (60183215, 60190975, 60203872,
60212077); all 20 seals verified, none refused. Criteria as fixed in the plan (M5):

| criterion | toy (10 fits) | verdict |
|---|---|---|
| joint levels uniform, KS p > 0.01 | levels 0.16–1.00, **KS p 0.45** | pass |
| joint 68 / 95% coverage inside Clopper–Pearson 95% | 5/10 [0.19, 0.81]; 9/10 [0.55, 1.00] | pass |
| marginal 68 / 95%, b | 8/10; 10/10 | pass |
| marginal 68 / 95%, T_ref | 8/10; 10/10 | pass |
| marginal 68 / 95%, τ_init | **10/10** (CP [0.69, 1.00] excludes 0.68); 10/10 | **fail (over-coverage)** |
| toy − control mean pulls within 2 combined se | b −0.36 ± 0.28, T_ref −0.33 ± 0.18, τ_init +0.07 ± 0.21 | pass |

- Mean pulls, toy: b −0.36 ± 0.26, T_ref +0.07 ± 0.17, τ_init −0.17 ± 0.12; control: b
  −0.00 ± 0.11, T_ref +0.41 ± 0.07, τ_init −0.24 ± 0.17 (the tube/box pattern of lesson 8).
- Control (exact likelihood, 10 fits): levels KS p 0.88; joint 7/10, 10/10; marginal 68%
  b **10/10**, T_ref **10/10**, τ_init 9/10. The exact likelihood also over-covers the
  marginals, so the τ_init failure reflects the tube posterior filling the prior box
  (conservative marginals at this truth) rather than surrogate fidelity. Per rule 5 the
  failed criterion stays failed; whether marginal coverage at one truth is the right
  criterion for this problem is a decision for the user.
- MLE on a grid edge on some axis: 7/10 toy, 9/10 control fits (the tube reaches the box).
- Power: with 10 + 10 ensembles the pull differences resolve ≈ 0.2–0.3σ; the gate's
  prediction (≈ 0.6σ in the constrained direction at N = 250) is neither confirmed nor
  excluded by the marginal pulls (largest difference T_ref, 1.8σ). N = 1000 would test it
  about twice as sharply. Grid: lesson 2's v1 grid without a v2 nested check.
- **Reading: the round-1 generator's 1–3% residuals (9/128 gate FAILs) produce no
  detectable inference bias at N = 250 with this statistics;** joint coverage and the
  joint-level distribution are as calibrated as the exact-likelihood control's.

## Open items

- **Decision (user): M5 compute.** With the scaled models a fit candidate costs 8.4 s
  (SetFlow) / 11 s (SmoothPET) at N = 250 on an A100 in IEEE float32. M5 as written is several
  hundred GPU-hours; M6 more. Options: accept the budget, or test smaller models / cheaper
  numerics under M4 first (lesson 7: capacity did not matter for v1).
- **Next technical steps (no decision needed unless the user objects):** finish the 15
  missing counts shards (`shell/nle/gen_toy_local.sh data/v2_m1_counts/design_train.json 64
  16 20000 data/v2_m1_counts 16 --counts-only`), retrain p(N|θ) on all 64, attach it to
  every flow checkpoint (`multiplicity_check retrain --counts-dir`), and gate the swapped
  checkpoints. The flow's own offsets (width, jet pT; relevance 1.0–1.5 at N = 1000) then
  need a second-round flow training with more jets per point (e.g. 1024 × 6000, as lesson 7
  needed data density), and the gate again.
- **M3 criterion reading:** the δθ ≤ 0.2 σ requirement on per-parameter σ is not informative
  along the tube; see the proposal above (both numbers recorded).

- ~~Fit cost: sort observed events by N~~: done, `[scan].sort_by_multiplicity` (default
  true); per-event outputs stay in file order (test).
- Trainer: `--cpu-threads` (default 8) caps torch CPU threads; effect on speed with
  concurrent trainings not yet measured. Drawing noise on the GPU would remove the CPU
  step entirely (changes the noise stream).
- ~~`sbi-fit` measure string~~: now states log p(N|θ) + log q(x|N,θ) when the checkpoint has
  a multiplicity model.
