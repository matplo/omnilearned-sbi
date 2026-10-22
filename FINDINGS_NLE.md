# FINDINGS_NLE: running log for `dev/nle-perlmutter`

Brief: `PLAN_NLE_PERLMUTTER.md`. Newest entries at the bottom of each section. Every result
records source revision, dirty status and device.

## Resume here (state for picking up after an interrupted job)

- Current milestone: §3 First day (Perlmutter), started 2026-10-22, interactive job 60098593
  on nid003988 (4 × A100-SXM4-40GB), 4 h QOS.
- Detailed step status: `$JEST_NLE_ROOT/manifests/first_day/STATUS.md` (and logs next to it).

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
| 2026-10-22 | first day | 60098593 (interactive, urgent_gpu) | 1 × ≤4 | first-day checks, smoke chain |

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

## Open items

- Batch by N (bucketed sampling) in `train_flow` before M2 production (plan M2); the smoke
  trainer samples uniformly and cuts to the active width.
- The `sbi-fit` measure string still says "fixed multiplicity"; with a multiplicity model the
  per-event value is log p(N|θ) + log q(x|N,θ). Update the provenance text.
