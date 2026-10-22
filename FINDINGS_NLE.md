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

- Slurm associations (`sacctmgr show assoc user=$USER`): accounts **`alice`** and
  **`alice_g`** (GPU allocation), QOS include debug, gpu_debug, gpu_interactive, regular. GPU
  batch jobs most likely need `-A alice_g` (NERSC `_g` convention); to confirm on the first
  GPU sbatch.
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
- Not yet done in M1: development scans (1D lines through the development setting and
  points along the tube; needs the v2 iso-moment curve, lesson 1), reproducibility of a
  shard on a CPU node and a login node.

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

## Open items

- Fit cost: sort observed events by N before chunking in `log_prob`/`make_logp_fn` (as for
  sampling) before M4/M5.
- Trainer: cap CPU threads / draw training noise on the GPU (speed note in M2).
- The `sbi-fit` measure string still says "fixed multiplicity"; with a multiplicity model the
  per-event value is log p(N|θ) + log q(x|N,θ). Update the provenance text.
