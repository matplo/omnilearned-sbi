# RESTART HERE — JEWEL NLE track

**Read this first after any interruption** (session died, agent died, node died, laptop closed).
It is the single source of truth for *where we are* and *what to do next*. Update the
**Current state** block at the end of every step — it is the only part that goes stale.

Brief: `docs/PLAN_JEWEL_NLE.md`. Running log: `FINDINGS_JEWEL.md`. Rules: `AGENTS.md`.

---

## Current state

```
PHASE:        1p (probe running) + 5 (closure source done early)
LAST DONE:    phase 0a/0b/1 pushed; JEWEL closure observation source pushed;
              probe running on an interactive node
NEXT ACTION:  read the probe out (block below), record in FINDINGS_JEWEL.md,
              then launch Phase 2 with shell/nle/jewel_v2_drive.sh
RUNNING:      60481776 jv2-probe-int  (4 theta, data/jewel_v2/probe)
              60482135 jv2-corner     (3 hot/central theta, probe_corner)
              (batch 60481641 cancelled as redundant once the interactive ran)
OPEN RISK:    max n_constituents rises steeply with T_ref and centrality
              (94 at T=0.25 -> 138 at T=0.35). The cap is 256 and the design
              box reaches T_ref 0.9, which the first probe does NOT cover --
              hence probe_corner. If the corner exceeds ~200, raise
              selection.max_constituents BEFORE production and re-snapshot.
SNAPSHOT:     $JEST_NLE_ROOT/src_snapshots/e67510ff7d0dda7a977b556d5b7fd22b98e0163a
BLOCKED ON:   nothing
```

**Tip that worked:** the batch queue was oversubscribed and `jv2-probe` sat in `PENDING`,
while an interactive allocation for the same task directory started in ~20 s. Because workers
claim tasks exclusively, running a batch job and an interactive job against one directory is
safe and races harmlessly to whichever starts first.

| phase | what | status |
|---|---|---|
| 0a | docs + journal entry | **done** (journal pushed) |
| 0b | branch `dev/nle-jewel`, merge `dev/nle-weighted` | **done** (merge `5a5e859`) |
| 1 | v2 config, Sobol train design, inclusive jets, tests | **done** (`e67510f`, pushed) |
| 1p | probe at 4 theta -> acceptance, n_const, ESS | **running** (60481641) |
| 2 | production (train 500 Sobol / validation+fidelity 128 / test 4 holdout) | not started |
| 3 | surrogate training (weighted loss, 400k steps, seeds) | not started |
| 4 | gate + ptbins fidelity, incl. withheld theta | not started |
| 5 | JEWEL closure driver + numerics | **observation source done** (`closure.jewel_ensembles`, `source = "jewel"`, `acceptance --config`); numerics check not started |
| 6 | NLE fits at withheld theta (pulls, coverage) | not started |

---

## Orientation (fixed facts, do not re-derive)

- **Worktree:** `/global/cfs/cdirs/alice/ploskon/henv-omni-sbi/omnilearned-sbi-jewel`,
  branch `dev/nle-jewel`. Other worktrees (`-fidelity`, `-weighted`, and the main
  `omnilearned-sbi` on `dev/nle-perlmutter`) belong to other tracks — **do not switch their
  branches**.
- **A fresh worktree has no `src/omnilearned/_version.py`** (gitignored, written at install).
  Without it every test errors with `ModuleNotFoundError: omnilearned._version`. Recreate:
  ```bash
  cd <worktree> && REV=$(git rev-parse --short=9 HEAD)
  printf 'version = __version__ = "0+worktree.g%s"\nversion_tuple = __version_tuple__ = (0, "worktree", "g%s")\n' "$REV" "$REV" > src/omnilearned/_version.py
  ```
- **Always** `henv -n omni-sbi -x python` (never create/modify environments), and
  `PYTHONPATH=$PWD/src` when running from the worktree.
- `export JEST_NLE_ROOT=/global/cfs/cdirs/alice/ploskon/jest-nle` — data and runs live there,
  nothing from it is committed.
- Slurm: CPU `-A alice -C cpu`, GPU `-A alice_g -C gpu`. Interactive GPU is capped at 4 h and
  **2 nodes per user**; chain allocations with a `nohup` driver
  (`shell/nle/m5_drive.sh` is the working pattern). Always allocate whole nodes with
  `--gpus 4`, 4 workers (Slurm renumbers devices if you ask for fewer).
- **Budget is pre-approved by the user for this plan** (rule 8 satisfied by standing
  approval + node-hour logging). Do not block waiting for approval; log usage in FINDINGS.
- **Do not block on questions at all** — record the open item and move to the next
  independent workstream.

## How to resume cleanly

```bash
cd /global/cfs/cdirs/alice/ploskon/henv-omni-sbi/omnilearned-sbi-jewel
git status --short && git log --oneline -5      # what is committed
squeue -u $USER -o "%.10i %.25j %.8T %.10M %R"   # what is still running
tail -40 $JEST_NLE_ROOT/runs/jewel_v2/logs/progress.log 2>/dev/null
cat docs/RESTART_JEWEL.md                        # this file: Current state block
```

Then: find the first row above that is not `done`, and run its command block below.
**Everything is idempotent** — re-running a completed step is safe and cheap (workers skip
tasks that already have a manifest; drivers stop on `DONE`).

---

## Command blocks per phase

### Phase 1p — probe (CPU, < 1 node-h)

Measures what the production sizing needs. **Technical gate only** (not a budget gate): if
`n_constituents` hits the 256 cap, raise `selection.max_constituents` before production.

```bash
cd /global/cfs/cdirs/alice/ploskon/henv-omni-sbi/omnilearned-sbi-jewel
export JEST_NLE_ROOT=/global/cfs/cdirs/alice/ploskon/jest-nle
SNAP=$(shell/nle/snapshot.sh)          # jobs run from an immutable snapshot, never the worktree
P=$JEST_NLE_ROOT/data/jewel_v2/probe
PYTHONPATH=$SNAP/src henv -n omni-sbi -x python -m omnilearned.sbi.jewel.generate plan \
  --split probe --out $P --config $SNAP/configs/jewel-2.6.0-cs-v2.toml --wexpo 5 \
  --nevent 2000 --no-events \
  --theta 1.5 0.65 0.35  4.5 0.45 0.65  7.5 0.35 0.85  9.5 0.25 0.45
sbatch shell/nle/jewel_farm.sbatch $SNAP $P       # or run `generate work --dir $P` on a CPU node
```
**Probe read-out** (run this once the job finishes; it is the input to Phase-2 sizing):
```bash
P=$JEST_NLE_ROOT/data/jewel_v2/probe
henv -n omni-sbi -x python - <<'EOF'
import json, glob, numpy as np, h5py
ms = [json.load(open(f)) for f in sorted(glob.glob(f"{__import__('os').environ['JEST_NLE_ROOT']}/data/jewel_v2/probe/probe_*.json"))]
for m in ms:
    print(m["theta"], "jets/ev %.3f" % m["acceptance"], "jets", m["jets_selected"],
          "ev_with_jet", m.get("events_with_a_jet"), "nconst", m["n_constituents_range"],
          "s/ev %.3f" % m["seconds_per_event"], "rej", {k: v for k, v in m["rejections"].items() if v})
tot = sum(m["jets_selected"] for m in ms); ev = sum(m["events_generated"] for m in ms)
print(f"TOTAL jets {tot} events {ev} jets/event {tot/max(1,ev):.3f} "
      f"core-s/jet {sum(m['seconds_jewel'] for m in ms)/max(1,tot):.3f}")
for f in sorted(glob.glob(f"{__import__('os').environ['JEST_NLE_ROOT']}/data/jewel_v2/probe/probe_*.h5")):
    with h5py.File(f) as h:
        g = h["global"][...]; w = h["weight"][...] if "weight" in h else None
        pt = g[:, 3]
        print(f.split('/')[-1], "jet_pt", np.round([pt.min(), np.median(pt), pt.max()], 1),
              "ESS/n %.3f" % ((w.sum()**2/ (w**2).sum())/len(w)) if w is not None else "")
EOF
```
Decide from it: `--target-jets` per point for Phase 2, and **whether
`selection.max_constituents` (256) must be raised** — if `n_constituents_range` touches 256
anywhere, raise it and re-snapshot *before* production.

### Phase 2 — production

**Use the driver; it plans, runs waves, un-sticks stale locks, tops up from the *measured*
per-point acceptance, and ends with index+merge.** One per split, in parallel:

```bash
cd /global/cfs/cdirs/alice/ploskon/henv-omni-sbi/omnilearned-sbi-jewel
export JEST_NLE_ROOT=/global/cfs/cdirs/alice/ploskon/jest-nle
SNAP=$JEST_NLE_ROOT/src_snapshots/<rev>          # snapshot.sh, must be a committed rev
L=$JEST_NLE_ROOT/data/jewel_v2
mkdir -p $L
nohup shell/nle/jewel_v2_drive.sh $SNAP train      2500 8 4 > $L/drive_train.out 2>&1 &
nohup shell/nle/jewel_v2_drive.sh $SNAP fidelity   8000 4 3 > $L/drive_fidelity.out 2>&1 &
nohup shell/nle/jewel_v2_drive.sh $SNAP validation 2000 2 3 > $L/drive_validation.out 2>&1 &
```
Targets are jets *per point*: train 500 points, fidelity/validation 128 points. The fidelity
split needs many jets per point because the gate compares distributions there.

**Test split** (the 4 reserved holdout theta, >= 10 ensembles each) is planned by `--theta`
rather than a design, so it is not driven by the script:
```bash
PYTHONPATH=$SNAP/src henv -n omni-sbi -x python -m omnilearned.sbi.jewel.generate plan \
  --split probe --out $L/test --config $SNAP/configs/jewel-2.6.0-cs-v2.toml --wexpo 5 \
  --chunks 40 --no-events \
  --theta 7.5 0.35 0.85  4.5 0.45 0.65  1.5 0.65 0.35  9.5 0.25 0.45
```
then farm it the same way and `index` + `merge`.

**Never rely on `--dependency=afterok` alone for a farm**: a timed-out array element is killed
mid-task and leaves claim locks, so the dependent never fires and the chain stalls silently
(this cost the weighted run a full chain, 2026-10-27). The driver runs `unlock-stale` after
every wave for exactly this reason.

**If the batch queue is oversubscribed**, start an interactive allocation against the same
task directory — claims are exclusive, so it is safe and usually starts in seconds:
```bash
nohup salloc -A alice -C cpu -q interactive -t 04:00:00 -N 1 -J jv2-int \
  srun -N1 -n1 bash shell/nle/jewel_farm.sbatch $SNAP $L/<split> 128 14000 > /dev/null 2>&1 &
```

### Phase 3 — training

`shell/nle/jewel_train_chunk.sh` pattern: 1 node, 4 A100, 4 runs in parallel, weighted loss,
`p(N|theta)` attached. 400k steps (training length is the one lever that moved the toy);
include a 200k run to confirm that transfers to JEWEL, and >= 2 seeds for the spread.
**Never judge fidelity by validation loss** — the feature transform is refitted per run on
its own biased split, so losses across differently-biased samples are not comparable (this
error was made twice, on the toy and on JEWEL). Judge by the gate and `ptbins`.

### Phases 4-6

Gate + `ptbins` (extend `PT_EDGES` to 20-500) at the 128 fidelity points and at the 4 withheld
theta; then the JEWEL closure driver (`jewel/closure.py`, to be written, mirroring
`toy/closure.py` but drawing observations from generated test shards), numerics check
(rule 6: must pass before fits), then fits at withheld theta with pulls and coverage.

---

## Traps already hit (do not rediscover)

1. **Stale snapshot.** Jobs run from `$JEST_NLE_ROOT/src_snapshots/<rev>`. A job launched
   before a commit keeps the *old* `PYTHONPATH` for its whole life — editing `m5.env`/config
   afterwards does not reach it. Commit and snapshot *before* launching, and if code changed
   mid-flight, cancel and relaunch rather than assuming it picked the change up.
2. **Stale claim locks** block `afterok` chains (see Phase 2).
3. **`--gpus 3`** lets Slurm renumber devices so `CUDA_VISIBLE_DEVICES=3` misses. Always
   `--gpus 4`.
4. **Validation loss is not a fidelity metric** across differently-biased training splits.
5. **`reco.cluster_events` reports `jet_pt/eta/phi` for the leading jet only**; `membership`
   is the per-jet index. Non-leading jet axes must be summed from constituents
   (`jets._axis`).
6. **Do not commit** data, checkpoints, run trees, figures or binaries; never force-push.
