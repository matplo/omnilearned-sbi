# FINDINGS_NLE: running log for `dev/nle-perlmutter`

Brief: `PLAN_NLE_PERLMUTTER.md`. Newest entries at the bottom of each section. Every result
records source revision, dirty status and device.

## Resume here (state for picking up after an interrupted job)

- Done: §3 first day (all checks pass), M1 data, M2 trainings started (see M2 below).
- In flight at the time of writing (interactive job 60098593, ends ~05:21 PDT 2026-10-22):
  three trainings on nid003988 GPUs 0–2 (`$JEST_NLE_ROOT/runs/m2/*`). Each `checkpoint.pt`
  holds the best EMA state so far (written at every validation improvement after step 10k),
  `run.json` appears only at the end, `command.txt` has the exact command.
- Next: M3 gate of each finished checkpoint (`shell/nle/gate_gpu.sbatch`, resumable per
  point), sampler convergence on the final checkpoints, then M4/M5 on the chosen generator.
- Detailed step status: `$JEST_NLE_ROOT/manifests/first_day/STATUS.md`.
- Batch jobs run from read-only source snapshots (`shell/nle/snapshot.sh` →
  `$JEST_NLE_ROOT/src_snapshots/<rev>`, `PYTHONPATH=<snapshot>/src`), so editing or switching
  this checkout cannot reach them.

## Environment (Perlmutter)

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
| 2026-10-22 | first day, M1, M2, M3 prep | 60098593 (interactive, urgent_gpu, 4 h) | 1 × 4 | checks, smoke chain, M1 data (CPU of this node), 3 trainings, sampler timing |
| 2026-10-22 | M1 acceptance | 60101162 (cpu, debug) | 1 × 0.04 | shard regeneration on nid006833 + cross-host tests |
| 2026-10-22 | M3 | 60101552 (gpu, regular, ≤ 2 h, pending) | 1 × ≤ 2 | finish gate of `set_d256_L6` (resumable) |
| 2026-10-22 | M3 | 60101555 (gpu, regular, ≤ 2 h, pending) | 1 × ≤ 2 | finish gate of `pet_d256_L6_66k` (resumable) |

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

## Open items

- ~~Fit cost: sort observed events by N~~: done, `[scan].sort_by_multiplicity` (default
  true); per-event outputs stay in file order (test).
- Trainer: `--cpu-threads` (default 8) caps torch CPU threads; effect on speed with
  concurrent trainings not yet measured. Drawing noise on the GPU would remove the CPU
  step entirely (changes the noise stream).
- ~~`sbi-fit` measure string~~: now states log p(N|θ) + log q(x|N,θ) when the checkpoint has
  a multiplicity model.
