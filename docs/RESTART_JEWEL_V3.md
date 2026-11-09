# RESTART HERE — JEWEL v3

**Read this first after any interruption.** It holds the current state and the exact next
command. Update the state block at the end of every step; it is the only part that goes stale.

Brief: `docs/PLAN_JEWEL_V3.md`. Format: `docs/FORMAT_JEWEL_V3.md`. Log: `FINDINGS_JEWEL_V3.md`.

---

## Current state

```
PHASE:        P2, P3, P4's gate.py part, P5 all done (2026-11-05). P4's gate: checkpoint s0
                (sha 7ce2e63d5504), all 128 fidelity points, --redraw. Family-wise result:
                **decisive fail** (19/128 FAIL at alpha/S=3.91e-04). Root-caused in the
                reports repo (`reports/m3-surrogate-fidelity/.../root_cause.html`) to the
                surrogate under-producing real constituents beyond the jet's R=0.4 cone (up
                to 39x deficit near dR~0.5); `explorer_core.html` showed core-jet (dR<0.4)
                fidelity is excellent once that known halo is excluded. **User decision:
                accepted as a known, low-priority tail issue and approved proceeding to P5
                on this checkpoint** (not blocked on a retrain).
                P5 (NLE numerics, rule 6) then found its own, independent, more serious
                problem: **decisive FAIL -- the production fit solver (`midpoint`, 64 steps,
                8 probes) and a higher-order reference (`rk4`, 64 steps, 16 probes) disagree
                on Delta-log-L by a reproducible ~20-25x on every one of 4 nearby-theta
                pairs** (e.g. pair0: -14.09 vs -329.72), while each setting's own seed-to-seed
                scatter is tiny (probe_std 0.07-0.38) -- not noise, not a step-count issue
                (doubling midpoint's steps barely moved its value), not a probe-count issue
                (halving rk4's probes barely moved ITS value), not a CUDA/CPU issue (CPU vs
                CUDA agree to <0.03). Isolated to the solver method itself -- see FINDINGS,
                "P5 numerics check (2026-11-05)".
NEXT ACTION:  **This blocks P6 (likelihood/scan) and P7 (closure) pending a user decision.**
              Per rule 6, no fit should run against untrusted Delta-log-L. Follow-up
              (2026-11-05, FINDINGS "P5 follow-up"): compared midpoint vs rk4 directly on 5
              real jets with exact (non-stochastic) divergence -- the absolute accumulated
              term only disagrees 1.2-3.1x per jet (not 20-25x), but the pairwise Delta a fit
              actually uses is far smaller than either solver's own noise floor at nsteps=64,
              so the ratio of the two solvers' pairwise shifts swings wildly and even flips
              sign across jets (+2x to +53x). This reframes the likely fix: not "which
              solver," but whether nsteps needs to go well beyond 64 (for whichever solver is
              kept) or the integration needs float64 instead of float32. **Now tested
              (2026-11-08/09, FINDINGS "the formal re-test at nsteps=256/float64"): the
              formal check re-run at nsteps=256/float64 for both solvers shrinks the gap
              ~3-4x (23-26x -> 4.6-8.3x) but does NOT close it -- still a decisive FAIL by
              the same <0.5-log-L-unit criteria.** "Raise nsteps + float64" alone is not a
              sufficient fix; whether rk4 itself has converged by nsteps=256 is itself an
              open question this raised. Still blocked on a user decision: push further
              (and separate the nsteps effect from the dtype effect, conflated in this
              result), accept something on other grounds, or read `likelihood.py`'s
              accumulation code more closely. `scripts/p5_trajectory_diagnostic.py`
              reproduces the per-jet version of this on any jet/theta. Only after the
              per-jet signal is confirmed above the discretization floor (and
              `numerics.py`'s `TEST`/`REFERENCE` or `LikelihoodOptions` defaults are updated
              accordingly) should P6 proceed. ptbins.py (the P4 followup, 20-500 GeV window) was
              deprioritized in favor of P5 per the user's direction and was not run this
              session; its fix is in FINDINGS/test_ptbins.py but still uncommitted as of the
              last P4 snapshot (c9ed27f) -- revisit if/when useful again, not urgent.
RUNNING:      nothing
DATASET:      $JEST_NLE_ROOT/data/jewel_v3/{train,fidelity,validation,test}/ (per-task shards
              + event stores), merged/ (training-ready, dataset.json/yaml sidecar covers all
              four splits under a `splits` key -- see FORMAT doc §4)
CHECKPOINTS:  $JEST_NLE_ROOT/runs/jewel_v3/p3_train/{v3_set_d256_L6_400k_s0,_s1,_s2,
              v3_set_d256_L6_200k_ctl}/checkpoint.pt (+ run.json, resume.pt)
GATE RESULTS: $JEST_NLE_ROOT/runs/jewel_v3/p4_gate/s0/point_NNNNN.json (all 128, checkpoint
              s0); $JEST_NLE_ROOT/runs/jewel_v3/p4_ptbins/ (not produced -- deprioritized)
NUMERICS:     $JEST_NLE_ROOT/runs/jewel_v3/p5_numerics/ (p5_obs.h5 -- 1000 real jets at the
              primary_test holdout theta, built by scripts/p5_prepare_observations.py;
              part_test_cuda.json, part_reference_cuda.json, numerics_cuda.json -- the formal
              merge/evaluate result; diag_midpoint128.json, diag_rk4_8probes.json,
              part_test_cpu.json -- the follow-up diagnostics; part_n256_f64_test.json,
              part_n256_f64_reference.json -- the 2026-11-08/09 nsteps=256/float64 re-test
              (evaluated directly with numerics.evaluate(), not numerics.py's own `merge`
              CLI, since each part recorded only one side's options at the new settings and
              the other at the module default -- merge's consistency check correctly refuses
              to combine them; see FINDINGS)
SNAPSHOT:     c9ed27f is the latest committed (FlowSurrogate.sample jet_pt fix). P5's fixes
              (numerics.py context forwarding, tests/sbi/test_numerics_context.py,
              scripts/p5_prepare_observations.py) are committed on top of it as of this
              entry -- cut a fresh snapshot before any future batch (sbatch, not interactive
              srun) job that needs them.
BLOCKED ON:   user decision on the P5 solver discrepancy (see NEXT ACTION) before P6 can
              proceed

SIDE WORK (2026-11-07/08, does not unblock P6/P7 -- uses ad hoc nsteps=256/float64 scripts,
not the production `LikelihoodOptions` defaults, which are UNCHANGED): built the
true-vs-inferred theta explorer the user asked to see directly. Phase A (float64 now a real
`FlowSurrogate.log_prob`/`fit.py` option, tested) and Phase B (acceptance_table's v3 jet_pt
fix, tested) are committed library changes. Phase C (5x5x5 acceptance table) and Phase D (5
points x 3 ensembles x 250 jets, 3x3x3 grid) are `scripts/p5_explorer_*.py` (code repo) +
`$JEST_NLE_ROOT/runs/jewel_v3/p5_explorer/` (data, not committed). Phase D's first run had a
real bug (unweighted slice of a weighted MC split biased T_ref) -- fixed by
sampling-importance-resampling; see FINDINGS "P5 follow-up: true-vs-inferred theta explorer".
Phase D rerun with the fix: T_ref recovers cleanly (bias ~0 at 4/5 points); b and tau_init do
not (large, sign-inconsistent scatter even in a wide 1D sweep) -- a real identifiability
limit at 250 jets/ensemble, not a bug. Phase E (the HTML report, reports repo) is done:
`reports/m4-nle-numerics/2026-11-08T0000-0130_jewel-v3_true-vs-inferred-theta/index.html`,
T_ref as the clean result, b/tau_init with the caveat stated plainly. The same bug in the
pre-existing P7 helper `toy.closure.jewel_ensembles` is also fixed now (see trap 24) --
differently, by resampling whole events rather than individual jets.

DONE (2026-11-08, ahead of P7, since trap 24 flagged it): closure.py's jewel_ensembles now
takes a `config` param and dispatches via shards_for_config, same as direct_by_point already
did -- it can read real v3 test-split data now, not just the toy/v2 schema. Found and fixed
a real wrinkle while doing it: v3 shards have event_id and group_id as separate columns
(event_id already unique per jet); the toy/v2 schema has only one column playing both roles,
which jewel_ensembles's grouping logic was built against. See FINDINGS "P5 follow-up
addendum 2". Still not exercised against real JEWEL test-split data (only test fixtures),
since P7 itself has not started.
```

| phase | what | status |
|---|---|---|
| P0 | branch, worktree, docs | **done** |
| P1 | config v3, Δη/log E/pid, v3 format + sidecar | **done** |
| P2 | generation (train/val/fidelity/test) **with event stores** | **done** |
| P3 | training (jet_pT conditioning, no standardisation) | **done** (4 checkpoints) |
| P4 | fidelity: gate + ptbins, weighted estimators | gate **done -- decisive fail, user accepted (core dR<0.4 fidelity is excellent)**; ptbins not pursued further |
| P5 | NLE numerics (rule 6) | **done -- decisive FAIL: midpoint vs rk4 solver disagree ~20-25x; blocks P6** |
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
24. **The fidelity split is weighted MC (`weight` spans ~1e-4 to ~1e2), and slicing the
    first N jets in file order is not a fair sample** -- `merge_v3` groups shards by
    generation bin, so the unweighted slice's jet_pt ran ~2x the true weighted-population
    mean at every theta point tested, biasing any likelihood that conditions on jet_pt.
    Both `scripts/p5_explorer_ensembles.py` and the existing P7 helper
    `toy.closure.jewel_ensembles` had this bug; **both are now fixed** (2026-11-08).
    `jewel_ensembles` resamples whole *events* (not individual jets) weighted by event
    weight, to keep an inclusive sample's within-event jet grouping intact for the
    composite-likelihood correlation check -- a plain jet-level resample breaks that, see
    FINDINGS "P5 follow-up addendum". `jewel_ensembles` also still hardcoded the toy/v2
    reader itself (not just the weight) -- fixed the same day by giving it the same `config`
    + `shards_for_config` dispatch `direct_by_point` already had (FINDINGS "addendum 2").
    Not yet exercised against real JEWEL test-split data (only test fixtures) since P7 itself
    has not started.
25. **v3 shards split `event_id` (unique per jet) and `group_id` (the originating JEWEL
    event) into separate columns; the toy/v2 schema has only one column that plays both
    roles.** Code written against the toy/v2 schema that reads `event_id` as "the shared
    event" (as `jewel_ensembles`'s grouping did, until trap 24's addendum 2) silently groups
    v3 jets by their already-unique-per-jet id instead -- one "event" per jet, not an error,
    just a wrong number (the composite-likelihood correlation check would see no correlation
    to check). Read `group_id` when present, fall back to `event_id` only for the legacy
    schema: `d["group_id"] if "group_id" in d else d["event_id"]`.
26. **A multi-hour script with no incremental save is a real walltime-cap risk on
    `interactive` QOS (4h max).** `toy.numerics run --which both` writes its output JSON
    only once, at the very end, after every seed of both test and reference options --
    unlike the explorer scripts' one-JSON-per-ensemble pattern. A combined
    midpoint-256/rk4-256/float64 run (2026-11-08) hit a 2h requested walltime with nothing
    to show for it (no partial file, no error -- just gone). Split into separate
    `--which test` / `--which reference` jobs (already supported) before requesting a long
    run with this script, each sized to its own cost -- rk4 at the same nsteps costs
    several times more than midpoint (observed: ~3600s vs ~530s per seed at nsteps=256).
