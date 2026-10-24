# Runbook: M5 development closure, toy v2 (agent-operable)

Campaign: `$JEST_NLE_ROOT/runs/m5/dev_N250_setC64` (`JEST_NLE_ROOT=/global/cfs/cdirs/alice/ploskon/jest-nle`).
Generator: `runs/m3/mult/set_d256_L6_multC64.pt` (round-1 SetFlow flow + p(N|θ) trained on all
counts shards). This is a **diagnostic on development data** (rule 4): the generator has not
passed the M3 gate (9/128 FAIL); the closure measures whether those residuals bias inference.

## What it runs

| campaign | source | ensembles | events | stream / indices |
|---|---|---|---|---|
| `toy/` | toy at the development setting (7.5, 0.30, 0.95) | 10 | 250 | development, 1000–1009 |
| `control/` | the surrogate itself, same truth (self-closure control, lesson 8) | 10 | 250 | self_closure, 2000–2009 |

Fits: grid b 0–11 (23) × T_ref 0.25–0.55 (14) × τ_init 0.2–1.2 (11), support scan (seed stride
6, 3, 4; Δlog post 10); midpoint-64, 8 per-step Rademacher probes (lesson 3); selection
normalization from the campaign's acceptance table (6 × 4 × 4 grid, 8192 clouds per point).
The grid is lesson 2's v1 choice and has **not** had the v2 nested-grid check; record that with
any result. Expected cost ≈ 3–4 GPU-hours per fit, 20 fits ≈ 70 GPU-hours ≈ 5 chunks of 4 h.

Pass criteria (plan M5, fixed before any v2 result): joint levels consistent with uniform
(KS p > 0.01); marginal and joint 68/95% coverage inside Clopper–Pearson 95% bounds; toy-data
mean pulls minus control mean pulls within 2 combined standard errors for every parameter.
With 10 ensembles each these tests are weak; report them as they are.

## Files

- `m5.env` (snapshot, checkpoint, acceptance grid, workers per GPU), `spec_toy.toml`,
  `spec_control.toml` — do not edit after `prepare` has run.
- `logs/progress.log` — one line per phase; start here. `logs/chunk_<i>.log`,
  `logs/worker_<job>_g<gpu>_<n>.log`, `logs/acc_<g>.log`, `logs/report_{toy,control}.txt`.
- `<campaign>/fits/fit_KKKK/` — `logp_cache.npz` (resume state), `claim.lock` (while a worker
  holds it), `result.{json,npz}` and `seal.json` when done. `<campaign>/truth/` is read only by
  `report`.
- `DONE` when both campaigns are sealed and reported; `STOP` (create it) stops the driver
  from requesting further allocations.

## Run it

Everything goes through `henv -n omni-sbi -x ...` inside the scripts; nothing else to set up.

- **Unattended, from a login node** (screen/tmux): requests 4 h interactive GPU nodes
  (`-A alice_g -q interactive`, 4 GPUs) back to back, up to 8:
  `nohup shell/nle/m5_drive.sh $JEST_NLE_ROOT/runs/m5/dev_N250_setC64 8 > $JEST_NLE_ROOT/runs/m5/dev_N250_setC64/logs/drive.out 2>&1 &`
- **Inside an allocation you already have** (4 GPUs):
  `shell/nle/m5_dev_closure.sh $JEST_NLE_ROOT/runs/m5/dev_N250_setC64`
- **Batch nodes** can join at any time (workers claim fits, no double work):
  `sbatch -A alice_g -C gpu -q regular -t 04:00:00 -N 1 --gpus 4 --wrap "bash shell/nle/m5_dev_closure.sh $JEST_NLE_ROOT/runs/m5/dev_N250_setC64"`
  (submit from a login node; inside an allocation unset `SLURM_*` first).

Each chunk is idempotent: acceptance table → prepare → interim report → fits (4 GPUs ×
`PER` workers, claiming unsealed fits of both campaigns, interleaved) → final report.
A fit cut by the end of an allocation resumes from `logp_cache.npz` in the next one; a
claim whose Slurm job has ended is reclaimed automatically.

## Check status

```
tail -20 $JEST_NLE_ROOT/runs/m5/dev_N250_setC64/logs/progress.log
ls $JEST_NLE_ROOT/runs/m5/dev_N250_setC64/*/fits/*/seal.json | wc -l      # of 20
squeue --me
cat $JEST_NLE_ROOT/runs/m5/dev_N250_setC64/logs/report_toy.txt            # latest report
```

## When it is done

1. `report.json` in `toy/` (with `versus_control`) and `control/`: pulls, coverage with
   Clopper–Pearson bounds, joint levels and their KS p, pull differences toy − control.
2. Log in `FINDINGS_NLE.md`: commands, job ids, node-hours (from `squeue`/`sacct`), the
   numbers above, the source snapshot, and that the grid had no v2 nested check.
3. Compare with the prediction from the gate's parameter shift: a constrained-direction
   bias of about 0.6 posterior σ at N = 250 (1.3 at N = 1000).

Rules: never edit specs, observations or fit configs after `prepare`; never delete a seal;
a fit whose seal fails verification is reported as refused, not re-fit silently.
