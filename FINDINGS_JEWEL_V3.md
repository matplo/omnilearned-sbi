# JEWEL v3 track (branch `dev/nle-jewel-v3`)

Running log of `docs/PLAN_JEWEL_V3.md`. State and restart: `docs/RESTART_JEWEL_V3.md`.
Format: `docs/FORMAT_JEWEL_V3.md`. Earlier logs are not rewritten: `FINDINGS_JEWEL.md` (v2),
`FINDINGS_NLE.md` (toy + NLE machinery), `FINDINGS_FIDELITY.md`, `FINDINGS_WEIGHTED.md`.

Every entry records the source revision, dirty status and the actual device.

## Job / node-hour log

| date | phase | job | resources | status | what |
|---|---|---|---|---|---|
| 2026-10-31 | P0 | — (login) | — | done | branch, worktree, docs |

## Why v3 exists (2026-10-31)

The v2 surrogate is far from passing the gate, and the representation audit
(`reports/m3-surrogate-fidelity/2026-10-31T1231-0700_jewel-v2_what-is-trained/`) found the
reason in the inputs rather than the training:

| | OmniLearned PET v2 (arXiv:2510.24066 §II.A) | v2 track |
|---|---|---|
| coordinates | Δη, Δφ from the jet axis | φ relative, **η absolute** |
| scale | log p_T, **log E** | log p_T only |
| standardisation | deliberately none | per-run `FeatureTransform` |
| conditioning | jet kinematics + multiplicity | **N only** |

Measured: constituent feature 0 correlates with the jet axis at **r = +0.999**, σ = 1.00,
while the R = 0.4 substructure it should carry has σ = 0.16. Subtracting the jet axis collapses
it onto the φ distribution exactly. The model spent most of its input range on where the jet
sat. `jet_pt` was stored in `global` and never reached the model.

**Carried forward from v2 as settled** (measured, not assumed):

- **WEXPO 3, not 5.** At a 20 GeV jet floor the full p̂_T range contributes and WEXPO 5
  oversamples high p̂_T: ESS/n 0.0065 vs 0.130, i.e. 47 vs 855 effective jets per θ point.
- **Training length saturates on JEWEL.** 100k→200k gains ~3 seed-σ; 200k→400k only 1.7σ, with
  best checkpoints at 284–310k. The toy's dominant lever does not transfer, so more training is
  not the fix for a fidelity gap.
- **Recoils stay on, with Milhano & Zapp constituent subtraction.** Raw events carry 366
  thermal particles on average (up to 2174); without subtraction jets are contaminated. The
  prescription is fixed in the backend id (`dRmax 0.5`, `leftovers=keep`,
  `thermal_pid=reference`).
- **Always write event stores.** v2 passed `--no-events`, which locked 10.3 node-h of
  generation to one representation. Storage was never the constraint (CFS has 44 PB free).

## P0: branch and scaffolding (2026-10-31)

`dev/nle-jewel-v3` cut from `dev/nle-jewel` (858b745); worktree
`omnilearned-sbi-v3`. Docs written: `docs/PLAN_JEWEL_V3.md` (the approved plan),
`docs/FORMAT_JEWEL_V3.md` (normative format), `docs/RESTART_JEWEL_V3.md` (live state and the
eleven traps already hit), this log. Nothing is reused from `jewel_v1`/`jewel_w1` — v3
generates fresh.

## P2: generation under way (2026-11-01) — measured acceptance ~2x worse than planned

`topup`'s self-correction is working as designed, but the numbers it is correcting are large.
Plan §2 budgeted train at 12 node-h (1024 tasks); measured per-point acceptance drove `topup`
to grow the task list to 22 809 (train) and 5855 (fidelity), from 1024 and 256 respectively.
Rough node-hour estimate from task count and the measured ~4.1 min/task at 64 workers/node:

| split | planned tasks | actual tasks (so far) | planned node-h | estimated actual node-h |
|---|---|---|---|---|
| train | 1024 | 22 809 | 12 | ~24 |
| fidelity | 256 | 5855 | 3 | ~6 |
| validation | 256 | 1487 | 1 | ~1.6 |
| test | 8 (then 90 at `--nevent 5000`) | 90 | 0.3 | ~0.5 |

Total ≈ 33 node-h against a 16 node-h plan estimate — about 2x over, still well inside
"budget pre-approved, log node-hours, do not block on questions." Validation reached its
per-point target cleanly (649 789 jets at 128 pts × 5000 = 640 000). Test's first merge ran at
62/90 tasks (142 646 of a 200 000-jet target, 71%) because the driver's `WAVES` cap was hit
before the two outstanding farm jobs finished, not because `topup` under-estimated chunks
needed (no `too_many_constituents`-style cap warning fired) — relaunched with `--waves 6` to
let it top up and re-merge once the in-flight tasks land.

**Operational note, not a code bug:** the driver loops (`jewel_v3_drive.sh`, launched via
`setsid nohup ... &`) died twice when the interactive session ended, despite `nohup` —  the
submitted Slurm jobs themselves were unaffected, but the wave→topup→merge bookkeeping stalled
each time until manually relaunched (idempotent, so safe, but costs wall-clock before someone
notices). `setsid` alone did not change the behaviour between the two session endings tested
here; worth a method change (e.g. a persistent `screen`/`tmux` session) if it recurs.

## P2 correction: the "2x node-hours" above was mostly a crash, not low acceptance

The section above took the task-count growth at face value — `topup`'s own acceptance math,
which only sees successful manifests. It does not see failed attempts, and there were far
more of those than successes: `event_jets` left its rejection reason `None` whenever every
clustered jet candidate in an event was below `min_constituents` (a common, not exotic, event
shape — nothing exotic about two sub-threshold candidates). `run_task`'s
`pre_reject[why] += 1` then raised `KeyError(None)`, and because the exception propagates out
of the *whole* per-event loop, it discarded every event already processed in that task, not
just the offending one.

Measured failure rate before the fix: **57 593 failed attempts against 9576 successful train
manifests (86%), 41 057 against 5458 fidelity manifests (88%), 12 721+ for validation, and
enough of test's attempts that it made no net progress for the final ~3 hours of its run.**
The "2x node-hours" estimate was therefore mostly CPU spent on tasks that crashed partway
through a full JEWEL run and wrote nothing — not jets selected at a worse rate than planned.
True per-event acceptance is not yet known; it can only be measured honestly after a run with
the fix in.

Fixed in `jets.py` (`event_jets` now sets `reason = "too_few_constituents"` or `"no_jet"`
before returning whenever `recs` is empty, mirroring the singular `event_jet()` helper's
existing `why or "no_jet"` guard, which this plural path should have had from the start) and
defensively in `generate.py`'s call site. Regression test in `test_jewel_inclusive.py`.

**Action taken:** cancelled every running `jv3-*` farm job and the four driver loops (all on
the pre-fix snapshot), cleared claim locks, cut a new snapshot from the fix, and relaunched
train/fidelity/validation/test from it. Already-written manifests are untouched — nothing
generated so far is discarded, only resumed. Validation's merge (649 789 jets, 125/128
points, done via the new compute-node `jewel_v3_merge.sbatch` after the login-node OOM noted
above) stands as the last pre-fix checkpoint; worth a top-up pass once post-fix capacity is
available, since 3 points have no jets at all yet.

**Confirmed fixed, not just fixed-in-theory:** ~30-60k `KeyError(None)` entries kept
appearing in `failures.log` for an hour or so *after* the relaunch, which first looked like
the fix hadn't taken. It had: every one of those post-relaunch failures stopped by
~15:52 — hours before `squeue`/`pgrep` showed anything still running on the old snapshot —
consistent with at least one pre-fix farm array element (4 h walltime) that `scancel`/`pkill`
missed, running to its own natural end rather than the fix being incomplete. Confirmed
directly: pulled one exact failing task (`train_p00390_c050`, theta/njob/nevent from
`tasks.json`) and reran `generate.run_task` on it standalone against the fixed snapshot --
`SUCCESS 400 997`, no exception. Zero new failures of any kind since.

## P2: shards and merges were padded to the configured cap, not the real max constituent count

A second, independent memory bug, found while retrying train/fidelity's merges: every v3
shard was written at width `sel.max_constituents` (1024) regardless of its real constituent
count (measured per-jet max ~148, one task's actual max 75) -- `jets.select_records_v3`
passed the configured cap straight to `pack_v3` unconditionally. `merge_v3` then computed its
output width from the *stored* shard width rather than the real per-jet max, carrying the
blow-up through. Together these turned a ~150-wide jet population into a ~1024-wide array at
merge time (~7x memory), which is what OOM-killed validation's 650k-jet merge on a login node
and ran a fidelity 2.75M-jet merge past a 30-minute compute-node walltime. Fixed in both
places: the writer now pads to the task's own real max (capped at `sel.max_constituents`,
never above it), and `merge_v3` now computes and trims to the group's real max via the mask
rather than trusting the stored width -- the second fix alone unblocks every already-written
shard without regenerating anything, since it works regardless of how wide the input happens
to be stored. Regression tests in `test_jewel_inclusive.py` (writer) and `test_v3format.py`
(merge trim).

**Driver hygiene note:** the farm's `#SBATCH -o %x-%A_%a.out` default is a path relative to
`sbatch`'s own working directory, which was the worktree when these jobs were launched by
hand rather than through a job already running from a snapshot -- left ~30 stray `.out` files
in the worktree, cleaned up (not committed; nothing in `.gitignore` currently catches this
pattern, worth adding if it recurs).

## P2: two more corrupted the merge rather than the data, both found by manual inspection

1. **One empty manifest crashed `write_index` for the whole split.** A worker killed mid-
   write (the scancel/pkill cleanup above, or a farm wave hitting its walltime) can leave an
   empty `.json` manifest. `work()`'s own "already done" check is bare file existence, so
   that task was permanently stuck (never retried) AND every future `write_index` call for
   the whole split crashed with a bare `JSONDecodeError` on the first bad file it hit. Hit
   once: `train_p00110_c021.json`, 0 bytes. `write_index` now deletes a corrupt manifest and
   logs a warning instead of raising, making the task claimable again. Regression test in
   `test_jewel.py`.

2. **`dataset.json` only ever described the split that merged last.** Four splits share one
   `merged/` root; `write_dataset_spec` wrote the sidecar fresh every call, so train,
   fidelity, validation and test each clobbered the previous split's entry. Found after all
   four had genuinely finished: `dataset.json` said `"split": "validation"` with no trace the
   other three ever existed, even with their files sitting right next to it on disk. Per-split
   fields now accumulate under a `splits` dict; regenerated the live sidecar from the four
   `index_<split>.json` files once fixed (`splits in sidecar: ['fidelity', 'test', 'train',
   'validation']`, confirmed). Regression test in `test_v3format.py`.

   Also: the 30-minute debug-QOS walltime, adequate for the smaller splits, was never enough
   for train's merge (22 809 shards: ~hashing alone took most of a debug-QOS slot). Merges
   now go through `-q regular -t 02:00:00`+ rather than `-q debug -t 00:30:00`; train's own
   merge needed ~2 h wall end to end (index + 11 output groups) even after the width fix.

## P2 generation: complete

All four splits merged and verified, final counts (all exceed their plan targets):

| split | jets | files | θ points |
|---|---|---|---|
| train | 10 705 283 | 11 | 512 / 512 |
| fidelity | 2 751 253 | 3 | 128 / 128 |
| validation | 653 086 | 1 | 128 / 128 |
| test | 215 400 | 1 | 4 / 4 (the withheld holdout θ) |

P3 (training) can start: `omnilearned.sbi.toy.train` already reads these files directly via
`jewel.v3format.read_v3_shards`/`shard_paths_v3` (wired in before generation finished, see
the P3 entry above), conditions on jet_pt, and applies no feature standardisation.

## P3 launch: a fairshare-pending queue, then a real OOM once it actually ran (2026-11-03)

**The queue wait was a fairshare penalty, not congestion.** The first P3 job (4 GPUs, regular
QOS) sat `PENDING (Priority)` for 12+ hours with zero movement; splitting it into 4x 1-GPU
jobs didn't help either. `sshare -A alice_g` explained it: `ploskon`'s `EffectvUsage` was
0.893662 against a `NormShares` of 0.010989 -- the P2 generation campaign's 33 node-hours
plus this session's own GPU smoke tests had tanked fairshare, deprioritizing *any* new
submission from this account regardless of job size. Worked around with `-q interactive`
(salloc, 4h wall, 2-node budget, a separate pool) -- granted in ~20s every time it was tried.
One operational trap on the way: `setsid nohup salloc ... &` abandons a request that has to
queue (worked fine when interactive granted instantly, failed silently once it didn't) --
switched to the harness's own background-job mode, which holds the request properly.

**Once a job actually ran, `train_flow` OOM'd on the very first validation batch.** It moved
the entire training set to GPU unconditionally (`tr = {k: v.to(device) ...}`): for the real
10.7M-jet train set that is ~38.5 GiB as float32, which alone exceeds a 40 GiB A100's usable
memory with the model, optimizer and activations still to come. Neither of the two prior
checks caught this: the CPU end-to-end test used synthetic data (thousands of jets, not
millions), and the one real-data GPU check that ran (the debug-QOS smoke test) only trained
for 50 steps and happened not to need a GPU-resident validation batch large enough to notice
-- scale was the only thing missing from every check before this. Fixed: `train_flow` now
keeps the training and validation dicts on whatever device the caller passed (CPU, from
`toy.train`) and transfers only the active minibatch, or the small fixed validation subset,
to the compute device. Confirmed fixed on real GPU hardware, not just by the fix compiling:
seeds 0 and 1 are training past the exact point that crashed before (step 2000 of 400000
reached, loss 1.03/1.26 train/val, ~75s for those 2000 steps).

**Added while already under the 4h interactive clock, because the user flagged it in time:**
`train_flow(..., resume_from=, resume_path=, stop_after_step=)` -- a full resumable snapshot
(model, EMA, optimizer, scheduler, minibatch RNG stream, history) written at every validation
point, restoring a bit-exact continuation (new test confirms this against an uninterrupted
run), and a clean-stop option that leaves `options.steps` -- and so the LR schedule -- alone.
Seeds 0/1 are running with `--resume --stop-after-step 300000` (~3h7m at the measured pace,
leaving ~50min margin inside the 4h wall); seed2 and the 200k control are queued on regular
QOS with `--resume` set, to pick up cleanly whenever fairshare lets them through.

**Measured training pace** (real hardware, real data, dim 256 / depth 6 / heads 8 / batch
512): ~0.0375 s/step, matching the v2 JEWEL reference (400k steps ≈ 4h11m) closely despite
the different constituent width (v3: 184-225, pT/log-E conditioned; v2: absolute eta, no
log E) -- per-step cost is apparently dominated by batch size and model width, not cloud
width, at least in this range.

## P3: a second resume bug, then all four runs complete

The first `--resume` attempt on both seed0 and seed1 crashed within seconds of being
submitted, each after ~3h of real training: `torch.load(resume_from, map_location=device,
weights_only=False)` moves *every* tensor in the checkpoint to the compute device, including
`gen_state` -- the minibatch-stream generator's RNG state, which is a CPU-only ByteTensor by
construction (`gen = torch.Generator(device="cpu")`). `gen.set_state()` on a CUDA tensor
raised `TypeError: RNG state must be a torch.ByteTensor` immediately. Fixed with
`map_location="cpu"`; `load_state_dict` on the already-`.to(device)` model/optimizer casts
the loaded state to match on its own, same as always. The existing resume test did not catch
this because it runs on `device="cpu"`, where the two map_locations are identical -- a gap
now stated explicitly in the test's own docstring rather than quietly left.

All four runs finished after that, each resumed at most once (seed2's first segment also
stopped cleanly at 300000, same as seed0/1):

| run | steps | best_step | best val | checkpoint sha256 (12) |
|---|---|---|---|---|
| s0 | 400 000 | 400 000 | 0.88637 | `7ce2e63d5504` |
| s1 | 400 000 | 364 000 | 0.88637 | `4c059d6268fc` |
| s2 | 400 000 | 388 000 | 0.88650 | `8c2a1579a212` |
| ctl | 200 000 | 198 000 | 0.88659 | `929670be5409` |

Seed spread across the three 400k runs is ~0.00013; the 200k control is only ~0.0002 worse.
Unlike the toy/v2 pattern that motivated running three seeds plus a shorter control (400k
beating 200k clearly, outside the seed spread), val loss here looks nearly converged by
200k. **This is val loss, not fidelity** -- per the standing rule, it is not comparable
across differently-biased splits and is never the thing P4's gate judges on. Whether 400k
actually buys anything on the gate/`ptbins` metrics is P4's question to answer, not this
number's.

## P4: the gate run, a real-scale OOM, and a decisive fail (2026-11-03 -- 2026-11-04)

Checkpoint under test: `v3_set_d256_L6_400k_s0` (`7ce2e63d5504`). 128 fidelity points, sharded
4-way (`--num-shards 4`), `--n-generate 22000 --nsteps 128 --redraw`.

**`FlowSurrogate.sample()` had never been exercised with jet_pt as known context.** P3
trained the flow and the multiplicity model to condition on jet_pt (`sur.layout.known ==
["n_constituents", "jet_pt"]`), but `sample()` never accepted or threaded a jet_pt array --
the very first real run crashed immediately with `ValueError: known context ['jet_pt'] must
be supplied`. There is no `p(pT_J|theta)` model yet (that is P6), so the fix is a convention,
not a fit: `gate.py`/`ptbins.py`/`boundary.py` resample jet_pt from the **direct** sample's
own (weighted) empirical distribution at that theta, so the surrogate is always sampled at
matched jet kinematics rather than asked to invent its own jet_pt spectrum. Threaded through
`FlowSurrogate.sample(..., jet_pt=)`, `sample_and_select(..., jet_pt=)`, and a `sort_by_length`
permutation fix (jet_pt must be reordered by the same permutation as the draw, or shapes
mismatch or NaNs appear on the redraw path). Regression test added: `test_jewel_v3_train.py
::test_v3_sample_needs_jet_pt_and_uses_it_correctly`.

**A rare-large-N OOM in the attention layer, hit by every shard in turn, mitigated by
`--batch 256`.** Shard 0 crashed first (`torch.OutOfMemoryError: ... 78.19 GiB memory in
use` on an 80 GiB A100, inside `velocity.py`'s attention forward, reached through a redraw
round), then shard 1, then shard 2 at the **default** `--batch 1024`, each at a different
point and after already producing several good results -- not a first-point crash, so none
of the three real-scale smoke tests before this run (CPU unit test, debug-QOS GPU test,
the resume test) could have caught it: all used small synthetic clouds, never a redraw round
against a multiplicity model whose categorical tail occasionally draws N close to
`max_constituents = 1024`, which is what makes the batch's padded width -- and so the
attention tensor's memory -- spike. `--batch 256` (down from the default 1024, chosen for
speed, not memory) held every shard's peak usage under the 80 GiB cap for the remainder of
the campaign; no further OOMs after the switch, across ~100 more points. The cost is time
only: `gate.py run` already writes one `point_NNNNN.json` per point and skips any point whose
file exists (`if js.exists(): continue`), so every crash (OOM or the 4h interactive walltime
itself) needed nothing but a plain relaunch of the same command to resume -- exactly the
robustness the resumable-checkpoint work in P3 was asked for, except gate.py already had it
for free via its own output layout.

**Perlmutter's interactive QOS caps concurrent jobs at 2 per user, but a `salloc --gpus N`
request still reserves the whole 4-GPU node exclusively** (`AllocTRES` shows `gres/gpu=4`
even when `ReqTRES` asked for 1). That spare capacity was used to attach additional shards as
job *steps* (`srun --jobid=<existing> --gpus 1 --overlap ...`) to an already-running
allocation instead of submitting new jobs and hitting `QOSMaxSubmitJobPerUserLimit` --
tracked, Slurm-visible job steps, not raw background processes. 4 jobs and numerous crash/
relaunch cycles later, all 128 points completed; wall time across the whole campaign (several
sequential 4h interactive allocations, shards running 2-at-a-time) was about 7h.

**Family-wise verdict: decisive fail.** `gate.py summarize --alpha 0.05`:

```
128 of 128 points, alpha/S 3.91e-04: 19 FAIL, 19 significant
uniformity: ks_d=0.714, ks_p=1.78e-58, n_below_0.01=52/128, binomial_p_at_least=1.2e-68
```

19 of 128 points fail the per-point Bonferroni-corrected test (both significant at
alpha/S and `relevant`); 52 of 128 have a permutation p-value below 0.01 against a null
under which p should be uniform on [0, 1] -- the KS and binomial statistics on that are not
borderline (p ~ 1e-58, 1e-68). This is not a marginal miss: the v3 surrogate's constituent
clouds disagree with JEWEL's at a large fraction of theta points, strongly enough that no
plausible multiple-testing correction saves it. `width` and `outer_fraction` recur across
nearly every FAIL point's flagged-observable list, with `n_constituents`, `soft8_fraction`,
`ptd` and `lead_fraction` appearing intermittently -- the mismatch looks concentrated in jet
substructure/width observables rather than being uniform across all seven `JET_VARIABLE`s, a
lead worth following before any retraining attempt (the v2 representation audit that
motivated the v3 rewrite found exactly this kind of substructure-specific signature).

Not yet done: `ptbins` over the 20-500 GeV window (run on a representative 8-point subset
rather than the full 128, given the gate's verdict is already decisive and each point costs
the same redraw-heavy sampling as the gate did) to see whether the mismatch concentrates in a
jet-pT range; the four-withheld-theta repeat of this gate. **P5 onward (NLE numerics,
likelihood, closure) should not proceed on this checkpoint until the fidelity failure is
understood or a retrained/fixed surrogate is available** -- running the likelihood and
closure machinery against a surrogate that fails this gate this badly would not test the NLE
methodology, it would just reproduce the surrogate's own mismatch.
