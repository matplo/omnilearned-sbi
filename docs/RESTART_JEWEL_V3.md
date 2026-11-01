# RESTART HERE — JEWEL v3

**Read this first after any interruption.** It holds the current state and the exact next
command. Update the state block at the end of every step; it is the only part that goes stale.

Brief: `docs/PLAN_JEWEL_V3.md`. Format: `docs/FORMAT_JEWEL_V3.md`. Log: `FINDINGS_JEWEL_V3.md`.

---

## Current state

```
PHASE:        P1 done. P2 relaunched 2026-11-01 ~12:30 from snapshot 18275284 after fixing
              the event_jets KeyError(None) bug (86-88% of all task attempts were crashing).
              P3 training code (jet_pt conditioning, no standardisation) written, tested.
NEXT ACTION:  snapshot a clean commit, then one jewel_v3_drive.sh per split:
                SNAP=$(shell/nle/snapshot.sh)
                nohup shell/nle/jewel_v3_drive.sh $SNAP train      20000 8 4 > .../train.log 2>&1 &
                nohup shell/nle/jewel_v3_drive.sh $SNAP fidelity   20000 8 4 > ... &
                nohup shell/nle/jewel_v3_drive.sh $SNAP validation  5000 4 3 > ... &
                NEVENT=5000 nohup shell/nle/jewel_v3_drive.sh $SNAP test 50000 2 3 > ... &
              NEVENT=5000 on `test` is required: 50k jets/point needs 125 tasks of 1000
              events, over the chunks_per_point cap of 100 (topup now warns if it binds).
RUNNING:      full tests/sbi on a compute node
DATASET:      $JEST_NLE_ROOT/data/jewel_v3/{train,fidelity,validation,test}, merged/
SNAPSHOT:     made in P2 from a clean commit -- never hand sbatch a worktree path
BLOCKED ON:   nothing
```

| phase | what | status |
|---|---|---|
| P0 | branch, worktree, docs | **done** |
| P1 | config v3, Δη/log E/pid, v3 format + sidecar | **done** |
| P2 | generation (train/val/fidelity/test) **with event stores** | starting |
| P3 | training (jet_pT conditioning, no standardisation) | not started |
| P4 | fidelity: gate + ptbins, weighted estimators | not started |
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
