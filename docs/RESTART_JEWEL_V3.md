# RESTART HERE — JEWEL v3

**Read this first after any interruption.** It holds the current state and the exact next
command. Update the state block at the end of every step; it is the only part that goes stale.

Brief: `docs/PLAN_JEWEL_V3.md`. Format: `docs/FORMAT_JEWEL_V3.md`. Log: `FINDINGS_JEWEL_V3.md`.

---

## Current state

```
PHASE:        P2, P3, P4's gate.py part all done (2026-11-04). P4's gate: checkpoint s0
                (sha 7ce2e63d5504), all 128 fidelity points, --redraw. Needed two real fixes
                along the way -- FlowSurrogate.sample() had never been taught to accept
                jet_pt (v3's layouts condition on it), and a rare-large-N redraw spiked GPU
                memory (fixed with --batch 256; default 1024 OOM'd every shard in turn, each
                after producing several good points first) -- see FINDINGS. Family-wise
                result: **decisive fail**. `gate.py summarize --alpha 0.05`:
                  128 of 128 points, alpha/S 3.91e-04: 19 FAIL, 19 significant
                  uniformity ks_p=1.78e-58, n_below_0.01=52/128, binomial_p=1.2e-68
                Not a marginal miss -- the p-value distribution overwhelmingly rejects the
                null of no mismatch. `width`/`outer_fraction` recur across most FAIL points'
                flagged-observable lists; worth following before any retraining attempt.
NEXT ACTION:  ptbins.py over the 20-500 GeV window hit a real bug on its first real run
              (fixed, see FINDINGS and the new regression test in test_ptbins.py -- NOT YET
              IN A SNAPSHOT, cut one before launching): `run()`'s spectrum_edges line did a
              bare `np.allclose(edges, PT_EDGES_60_200)`, which raises ValueError (cannot
              broadcast) rather than returning False when edges has a different length --
              exactly the v3 20-500 window (10 edges vs. 9). Once a fresh snapshot exists,
              relaunch `ptbins.py run --points 0 16 32 48 64 80 96 112 --batch 256` (a
              representative 8-point subset, not the full 128 -- the gate's verdict is
              already decisive and ptbins costs the same per point) to see whether the
              mismatch concentrates in a jet-pT range. **Then: this is a decision point for
              the user, not a mechanical next phase** -- P5 onward (NLE numerics, likelihood,
              closure) should not run against a surrogate that fails fidelity this badly;
              doing so would test the surrogate's own mismatch, not the NLE methodology.
RUNNING:      nothing
DATASET:      $JEST_NLE_ROOT/data/jewel_v3/{train,fidelity,validation,test}/ (per-task shards
              + event stores), merged/ (training-ready, dataset.json/yaml sidecar covers all
              four splits under a `splits` key -- see FORMAT doc §4)
CHECKPOINTS:  $JEST_NLE_ROOT/runs/jewel_v3/p3_train/{v3_set_d256_L6_400k_s0,_s1,_s2,
              v3_set_d256_L6_200k_ctl}/checkpoint.pt (+ run.json, resume.pt)
GATE RESULTS: $JEST_NLE_ROOT/runs/jewel_v3/p4_gate/s0/point_NNNNN.json (all 128, checkpoint
              s0); $JEST_NLE_ROOT/runs/jewel_v3/p4_ptbins/ (not yet produced -- see above)
SNAPSHOT:     c9ed27f is the latest committed (FlowSurrogate.sample jet_pt fix); the
              ptbins.py fix above is uncommitted as of this entry -- commit and cut a fresh
              snapshot before relaunching ptbins
BLOCKED ON:   nothing

TODO before P7: closure.py's jewel_ensembles (toy/closure.py:121-124) has the same hardcoded
toy/v2 shard reader direct_by_point had -- needs the same shards_for_config(config) dispatch
before closure can read real v3 test-split data. Not fixed yet: P7 needs P3-P6 done first,
so fixing it now would be speculative work ahead of need. Noted here so it isn't missed.
```

| phase | what | status |
|---|---|---|
| P0 | branch, worktree, docs | **done** |
| P1 | config v3, Δη/log E/pid, v3 format + sidecar | **done** |
| P2 | generation (train/val/fidelity/test) **with event stores** | **done** |
| P3 | training (jet_pT conditioning, no standardisation) | **done** (4 checkpoints) |
| P4 | fidelity: gate + ptbins, weighted estimators | gate **done -- decisive fail**; ptbins pending a snapshot |
| P5 | NLE numerics (rule 6) | not started |
| P6 | three-factor likelihood + Jacobian + scan | not started |
| P7 | closure at the 4 withheld θ | not started |
| P8 | composite-likelihood correction | not started |
| P9 | final HTML report | not started |

Code tracks: **A** (main session) owns generation and all job submission; **B** (likelihood:
`jetpt.py`, `flow.py`, `multiplicity.py`) and **C** (weighted estimators: `fidelity.py`,
`gate.py`, `ptbins.py`) are merged. Both were verified by checking their *call sites*, not by
trusting their reports — Track C had changed `direct_by_point`'s return arity and broken
`boundary.py`, which its own report did not mention.

---

## Orientation (fixed facts, do not re-derive)

- **Worktree:** `/global/cfs/cdirs/alice/ploskon/henv-omni-sbi/omnilearned-sbi-v3`, branch
  `dev/nle-jewel-v3`. Other worktrees (`-jewel`, `-fidelity`, `-weighted`, and the main
  checkout) belong to closed tracks — **do not switch their branches**.
- **A fresh worktree has no `src/omnilearned/_version.py`** (gitignored, written at install).
  Without it every test errors with `ModuleNotFoundError: omnilearned._version`:
  ```bash
  cd <worktree> && REV=$(git rev-parse --short=9 HEAD)
  printf 'version = __version__ = "0+worktree.g%s"\nversion_tuple = __version_tuple__ = (0, "worktree", "g%s")\n' "$REV" "$REV" > src/omnilearned/_version.py
  ```
- **Always** `henv -n omni-sbi -x python` (never create or modify environments), with
  `PYTHONPATH=$PWD/src` from the worktree.
- `export JEST_NLE_ROOT=/global/cfs/cdirs/alice/ploskon/jest-nle` — data and runs live there,
  nothing from it is committed.
- Slurm: CPU `-A alice -C cpu`, GPU `-A alice_g -C gpu`. **Max 2 interactive nodes per user**,
  4 h each, shared between CPU and GPU.
- **Budget is pre-approved**; log node-hours in FINDINGS. Do not block on questions — record
  the open item and move to the next independent workstream.

## How to resume

```bash
cd /global/cfs/cdirs/alice/ploskon/henv-omni-sbi/omnilearned-sbi-v3
git status --short && git log --oneline -5
squeue -u $USER -o "%.11i %.18j %.9T %.7M %R"
cat docs/RESTART_JEWEL_V3.md          # this file: the state block above
```
Find the first phase not marked done and run its block in `docs/PLAN_JEWEL_V3.md` §5.
**Everything long-running is idempotent** — re-running a finished step is safe and cheap.

---

## Traps already hit (all silent — they cost a wrong number, not an error)

1. **Stale script path to `sbatch`.** Jobs run from `$JEST_NLE_ROOT/src_snapshots/<rev>`. Pass
   the *snapshot's* copy of a script, never the worktree path — a queued job otherwise picks up
   later edits (this killed job 60488240).
2. **Stale claim locks** block `afterok` chains. Run `generate unlock-stale` after every farm
   wave, not only on failure.
3. **`--gpus 3`** lets Slurm renumber devices. Always `--gpus 4`, whole node.
4. **Over-long walltime requests** do not get scheduled. Ask for what the job needs.
5. **Validation loss is not a fidelity metric** across differently-biased splits — the feature
   transform is refitted per run. Judge with the gate and `ptbins`.
6. **Plan splits by jet target, not chunk count**, or they come up short and cannot self-correct.
7. **Name run directories by the actual step count**, not a hardcoded one.
8. **`reco.cluster_events` reports `jet_pt/eta/phi` for the leading jet only**; `membership` is
   per jet. Non-leading axes must be summed from constituents (`jets._axis`).
9. **`event_id` seeds the Hutchinson probes** and must be unique per jet; the originating event
   goes in `group_id`. Sharing ids correlates probe errors and inflates the likelihood variance.
10. **Appending a dataclass field anywhere but last** shifts every positional caller.
11. **Components default to the toy.** `ptbins` binned on 60–200; `closure prepare` defaulted to
    the toy config; `gate` ignored per-jet weights; `multiplicity_check` fitted p(N|θ)
    unweighted. Standing question: *does this component know which backend and which weighting
    it is working on?*

Do not commit data, checkpoints, run trees, figures or binaries; never force-push.

12. **`topup` clamps at `chunks_per_point`.** A point needing more chunks than the cap stops
    short of its target, the next wave adds nothing, and the driver reads that as "complete".
    It now warns; the fix is a larger `--nevent` for that split, not a larger cap (raising the
    cap renumbers every NJOB stream).
13. **`design_points` had no `test` case.** The withheld θ now come from `[holdout]` via
    `holdout_points`, so closure data cannot be mistyped on a command line.
14. **A sub-threshold event (every candidate jet below `min_constituents`) left `event_jets`'s
    rejection reason `None`**, crashing `run_task` and discarding the whole task's output, not
    just that event's. Not exotic -- 86-88% of all P2 task attempts hit this before the fix.
    Fixed; see FINDINGS.
15. **Shards and merges were padded to `sel.max_constituents` (1024), not the real per-jet
    max (~150-230)** -- a ~5-7x memory blow-up that OOM-killed a 650k-jet merge on a login
    node and ran a 2.75M-jet merge past a 30-min compute-node walltime. Fixed at both the
    writer and `merge_v3`; the merge-side fix alone unblocks already-written wide shards.
16. **One corrupt (0-byte) manifest crashed `write_index` for the ENTIRE split**, and the
    task behind it was stuck forever (`work()`'s "already done" check is bare file
    existence). `write_index` now deletes a corrupt manifest and continues.
17. **`dataset.json` described only whichever split merged last** -- four splits share one
    `merged/` root, and the sidecar write was a flat overwrite. Per-split fields now
    accumulate under `splits.<name>`.
18. **The debug-QOS 30-min walltime was never enough for a real merge.** Train's own merge
    (22 809 shards) needed ~2 h end to end even after the width fix (~30-50 min just to hash
    every shard at index time). Merge jobs go through `-q regular -t 02:00:00`+ now.
19. **A GPU job PENDING for hours is not necessarily congestion** -- check `sshare -A <acct>`.
    Heavy recent usage (ours: the P2 generation campaign plus GPU smoke tests) tanks fairshare
    and deprioritizes new submissions regardless of job size; splitting one 4-GPU request into
    4x 1-GPU didn't help since the penalty is per-user, not per-job-size. `-q interactive`
    (salloc, not sbatch; 4h cap, 2-node budget) is a separate pool and granted in ~20s.
20. **`salloc` backgrounded without a terminal abandons a QUEUED request.** `setsid nohup
    salloc ... &` worked when the allocation was granted immediately, but when interactive
    itself had to queue once, the client gave up waiting and exited while the job stayed
    PENDING in Slurm with nothing left to run it -- `scancel` it, don't wait on it. Use the
    harness's own background mode (`run_in_background: true` on the tool call) instead; it
    holds the request properly even when it has to wait.
21. **`train_flow` moved the WHOLE dataset to GPU, not just each minibatch** -- OOM on a real
    10.7M-jet train set (~38.5 GiB as float32) even on a 40 GiB A100, regardless of
    `--batch-size`. Fixed; see FINDINGS. A smoke test on a SMALL split (e.g. validation
    standing in for train) would not have caught this -- it only breaks at real scale, which
    is exactly why nothing was confirmed on the full real train set until a live GPU job hit
    it. Test real-scale data paths on real-scale data before trusting a smaller proxy.
22. **`torch.load(..., map_location=device)` on a resume checkpoint moved a CPU-only tensor
    (the minibatch generator's RNG state) onto the GPU too**, and `Generator.set_state`
    rejects anything but a CPU ByteTensor. Cost both seeds their first `--resume` attempt,
    after ~3h of training each. `map_location="cpu"` always for a mixed CPU/GPU state dict;
    `load_state_dict` casts to the destination's actual device on its own.
23. **A CPU-only test cannot catch a CPU/GPU boundary bug.** The resume tests in
    `test_resume_training.py` all pass `device="cpu"`, where `map_location="cpu"` and
    `map_location=device` are the same thing -- trap 22 only exists on real GPU hardware and
    was found there, not in CI. Noted in the test file itself rather than left implicit.
