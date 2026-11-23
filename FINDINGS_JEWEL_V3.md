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

## P5 numerics check (2026-11-05): decisive FAIL -- midpoint and rk4 disagree ~20-25x

User decision: despite the P4 caution above, the user judged the beyond-R=0.4 mismatch a
known, acceptable, low-priority tail (see the reports repo's `explorer_core.html`, which
shows excellent core-jet fidelity once that halo is excluded) and asked to proceed to P5 on
checkpoint `v3_set_d256_L6_400k_s0` anyway. P5 (rule 6: no fits before this passes) checks
that Δlog L between nearby candidates converges in solver steps and divergence probes at the
analysis N, via `toy/numerics.py` (`python -m omnilearned.sbi.toy.numerics run`/`merge`).

**Setup bug found and fixed first.** `scripts/p5_prepare_observations.py` (new; builds a
1000-real-jet `observations.h5` at the `primary_test` holdout θ=(4.5, 0.45, 0.65) from the
merged v3 test split, via `write_observations` directly rather than `toy.closure`'s
P7-scoped, still-toy-only `jewel_ensembles`) first wrote `jet_pt` context as raw GeV. But
`train.py`'s `context_for_layout` -- the function that actually built jet_pt context during
training -- applies `np.log()` first ("recommended with log(pT/GeV) scaling, centered near
0", `multiplicity.py`'s `layout_with_multiplicity` docstring). `FlowSurrogate.sample()` takes
raw GeV and logs it internally for convenience; `log_prob()` does not -- its `context` must
already be in trained units. Passing raw GeV (20-389) instead of log(GeV) (3-6) pushed the
condition ~50-100x outside the trained manifold and produced multi-million-magnitude,
sign-flipping summed log-likelihoods across probe seeds. Fixed to `context=np.log(jet_pt)`;
verified sane magnitudes (tens, not millions) on a 20-event CPU smoke test before any GPU time
was spent. **Also found and fixed in the same pass:** `numerics.py`'s `summed_logl()` never
forwarded `obs.context` to `log_prob()` at all -- it would have raised `ValueError: known
context ['jet_pt'] must be supplied` the moment a v3 checkpoint's known-context layout met a
context-less call. Patched to slice and forward `obs.context` per chunk; regression test
`tests/sbi/test_numerics_context.py` (a `_RecordingSurrogate` stub, no real checkpoint needed)
covers slicing-per-chunk, the context-less (toy/v2) backward-compatible path, and that
`deltas()` threads context through. Full `tests/sbi` suite (231 passed, 7 skipped) green
after the patch.

**Formal result, 4 pairs of nearby θ around primary_test (single-axis ±2%-of-range steps in
b/T_ref/tau_init, plus one combined diagonal step), 1000 real jets, 12 probe seeds each,
`numerics merge`:**

```
numerics FAIL (limit 0.5; 12 test / 12 reference seeds)
                        pair0 (b)   pair1 (T_ref)  pair2 (tau_init)  pair3 (diagonal)
single_seed_shift        315.58       1291.96          212.26           1179.43
seed_averaged_shift       315.63       1292.40          212.30           1179.87
probe_std                   0.079         0.358            0.071             0.376
test_mean (midpoint-64/8probes)   -14.09        52.63            13.33             51.19
reference_mean (rk4-64/16probes) -329.72      1345.04           225.62           1231.06
```

Not a marginal miss and not probe noise: `probe_std` (seed-to-seed scatter *within* each
solver setting) is tiny, 0.07-0.38, so both the production (`midpoint`, 64 steps, 8 probes --
`toy/numerics.py`'s `TEST` dict, the actual fit solver) and reference (`rk4`, 64 steps, 16
probes -- `REFERENCE` dict) settings are each internally well-converged. But the two settings
disagree with each other by a reproducible ~20-25x on every pair.

**Follow-up diagnostics to characterize the cause (3 seeds each, same 4 pairs), done before
writing this up so the finding is actionable rather than just "it failed":**

| config | pair0 | pair1 | pair2 | pair3 | seeds |
|---|---|---|---|---|---|
| midpoint-64/8probes (production) | -14.09 | 52.63 | 13.33 | 51.19 | 12 |
| midpoint-**128**/8probes (2x steps) | -21.27 | 54.71 | 15.33 | 48.28 | 3 |
| rk4-64/**16**probes (reference) | -329.72 | 1345.04 | 225.62 | 1231.06 | 12 |
| rk4-64/**8**probes (reference's probe count halved) | -330.43 | 1347.94 | 226.20 | 1234.00 | 3 |

Doubling the production solver's step count (64 -> 128) barely moved the values
(-14.09 -> -21.27 for pair0) -- nowhere near the ~23x gap to rk4's -329.72, which rules out
"production just needs more steps" as the explanation; midpoint is already essentially
step-converged at 64. Halving rk4's probe count (16 -> 8) left its values essentially
unchanged (-329.72 -> -330.43) -- ruling out probe count as the driver too. **This isolates
the cause to the solver method itself: `midpoint` and `rk4` integrating the identical flow
ODE for the identical Δlog L give answers that differ by a large, systematic, highly
reproducible factor, not a resolution or noise effect.** That is not supposed to happen for
two consistent ODE integrators at nsteps=64 each on a reasonably smooth vector field, and
looks like a genuine implementation discrepancy -- most likely in how log-Jacobian/divergence
contributions accumulate differently along the two solver code paths in
`src/omnilearned/sbi/likelihood.py` (unexamined; flagged for the user, not fixed here) --
rather than something a different choice of solver settings could paper over.

**CUDA vs CPU agreement (secondary check, orthogonal to the finding above):** `midpoint-64/
8probes`, seeds 0 and 1, same checkpoint/observations/pairs, run once on `--device cuda` and
once on `--device cpu`. Agreement is excellent -- absolute differences were all < 0.03 across
every pair and seed (e.g. pair0 seed0: CUDA -14.139 vs CPU -14.162). This rules out a CUDA/CPU
numerical-precision explanation for the midpoint-vs-rk4 gap; the discrepancy is specific to
solver choice, not device.

**This is a decision point, not a mechanical next phase.** P5 as specified in
`docs/PLAN_JEWEL_V3.md` did not pass. Per rule 6 ("no fits before this passes"), **P6
(likelihood/scan) and P7 (closure) should not proceed against the current numerics defaults**
-- the production fit solver (`midpoint`, 64 steps, 8 probes) and the higher-fidelity
reference it's supposed to agree with disagree by over an order of magnitude, meaning the
Δlog L that would drive every posterior shape and credible interval in a real fit cannot
currently be trusted. The likely next step is to read `src/omnilearned/sbi/likelihood.py`'s
`midpoint` and `rk4` solver/divergence-accumulation code paths side by side looking for where
they diverge (a sign, a missing term, a per-step vs. per-solve normalization, or similar) --
but that investigation, and any change to `numerics.py`'s `TEST`/`REFERENCE` defaults or to
`LikelihoodOptions`, is left for the user's explicit decision rather than made here.

## P5 follow-up (2026-11-05): compared directly on real jets -- the gap is a noise-floor
## effect, not simply "rk4 is right, midpoint is wrong"

Took the recommended next step above literally: `scripts/p5_trajectory_diagnostic.py` (new)
replicates `FlowSurrogate.log_prob`'s setup for one real jet and one candidate θ, then
manually drives both solvers' step loops with **exact** (autograd, non-stochastic) divergence
-- no Hutchinson probes anywhere -- logging every stage. Run on 5 real jets spanning the
sample's multiplicity (5-93 constituents) and jet-p_T (25-389 GeV) range, at θ0 = primary_test
and at θ1 = θ0 with b += 0.22 (pair0's step).

**Absolute accumulated term** (one θ, ∫div(v)dt over the trajectory) disagrees only modestly
between solvers -- 1.17x-3.13x, growing with multiplicity (N=5: 1.67x; N=25: 1.52x; N=30 but
highest p_T: 1.17x; N=43: 2.46x; N=93: 3.13x) -- consistent with higher-dimensional jets
exposing more of the fine-scale structure §3's toy-field sanity check showed a lower-order
solver under-resolving. **Nowhere near the formal check's 20-25x.**

**Pairwise Δ (θ1-θ0), what a fit actually computes**, resolves the puzzle: midpoint's shift
from this one small step in b is only 0.0009-0.12 across these jets -- tiny against the
54-870 absolute scale, because differencing cancels almost all of the shared accumulated
term. The real per-jet θ-signal is far smaller than either solver's own ~1-3x absolute
disagreement. The *ratio* of the two solvers' pairwise shifts is accordingly wild and even
sign-flipping: +4.6x (N=5), +53.1x (N=25), +2.1x (N=30), **-20.7x (N=43)**, +19.2x (N=93) --
exactly what dividing two small, uncorrelated numerical residuals looks like, not what
dividing two estimates of the same real quantity looks like.

**This refines, not reverses, the P5 FAIL above.** It is not that `rk4` is accurate and
`midpoint` is wrong at the single-jet level -- per-jet, they're close to each other, and both
are probably far from the true continuous-time integral's θ-dependent part specifically,
because at nsteps=64 that part is smaller than each solver's own discretization error. The
aggregate 20-25x gap is plausibly what happens when ~1000 such noise-dominated per-jet
residuals are summed: if each solver's residual carries even a little net consistency of sign
across jets (both see the same real b-dependence, just resolve it with different, uncontrolled
error), the sums diverge by a large, reproducible factor without either solver being "right"
in an absolute sense.

**Reframes the likely fix:** not simply "pick rk4 over midpoint," but "at nsteps=64, the
per-jet θ-dependence a fit needs may be below either solver's own discretization floor."
Before deciding on a solver, check whether raising `nsteps` well beyond 64 makes the pairwise
Δ in the table above converge to a stable, cross-solver-consistent value instead of the
current wild swings, and whether running the integration in float64 (both this diagnostic and
the production pipeline run in float32 throughout) changes anything -- neither tested yet.
Full write-up with figures: the reports repo's
`reports/m4-nle-numerics/2026-11-05T0530-0800_jewel-v3_p5-numerics-solver-fail/index.html`

## P5 follow-up: true-vs-inferred θ explorer, a weighted-resample bug, and a real
## identifiability limit for b/tau_init (2026-11-07 -- 2026-11-08)

Source revision f8cb11f (dirty: `scripts/p5_explorer_ensembles.py` modified,
`scripts/p5_explorer_widen_check.py` new -- both committed at the end of this entry).
Device: Perlmutter GPU, `alice_g`, interactive QOS throughout (`gpu_premium` remains
inaccessible -- see the standing note below; not revisited this session per the user's own
"revisit premium later").

Built the explorer the user asked for directly ("x-axis TRUE theta, y-axis INFERRED theta,
most probable value + std, per-theta-point HTML selector") at nsteps=256/float64 (§ the
follow-up above): `scripts/p5_explorer_ensembles.py` (Phase D) fits 3 independent 250-jet
ensembles at each of 5 withheld fidelity θ points, 3x3x3 candidate grid, using a real
Pr_q(S|θ) acceptance table built via `scripts/p5_build_acceptance_table.py` (6x6x6 -> reduced
to 5x5x5 for cost). Required two real library fixes first (both tested,
`tests/sbi/test_log_prob_dtype.py` and `tests/sbi/test_acceptance_jet_pt.py`): float64
threaded through `FlowSurrogate.log_prob`/`ConditionLayout.assemble`/`fit.py` (was hardcoded
float32), and `acceptance_table` given a `jet_pt_pool_fn` + `batch` chunking (v3 checkpoints
need jet_pt as known context; the OOM pattern was identical to P4's).

**First Phase D run: every point's T_ref MLE pinned to its 3x3x3 grid's own edge**, with
near-total posterior mass there, and a consistent +0.11-0.17 bias once the grid was widened
(1D profile sweep, `scripts/p5_explorer_widen_check.py`, b/tau_init fixed at truth). Ruled out
the acceptance-table correction as the cause by decomposing `loglik` into the data term and
`-n_events*log_norm` directly: the argmax is bit-identical with or without the correction at
all 5 points (correction range ~15-50 log-units vs. the data term's ~4000-8500). Verified the
`ConditionLayout`'s center/scale for every axis exactly match `design.box` in the config (no
train/inference box mismatch). The curves themselves were smooth, unimodal, and extremely
sharply peaked (hundreds to thousands of log-units between adjacent 0.096-spaced grid nodes)
-- not noise.

**Root cause: the fidelity split is a weighted MC sample (per-jet `weight` spans ~1e-4 to
~1e2, confirmed directly), and both `direct_by_point`'s callers here (and the existing P7
`toy.closure.jewel_ensembles`) sliced the first N_JETS jets unweighted, in file order.**
Because `merge_v3` groups shards by generation bin, that slice is not representative: its
jet_pt ran ~2x the properly-weighted population mean at *every* point (checked directly,
all 5). Since jet_pt is the surrogate's known context and T_ref's effect couples to it
through energy loss, feeding a systematically high-jet_pt "observed" sample biased the
likelihood's T_ref argmax in a consistent direction.

**Fix: sampling-importance-resampling by the jet's own weight** (`np.random.default_rng(...)
.choice(..., p=weight/weight.sum(), replace=True)`) instead of an unweighted file-order slice
-- added as `--weighted-resample` to the widen-check script and made the only sampling mode
in `p5_explorer_ensembles.py` (ensembles are no longer disjoint in file order; each ensemble
index gets its own resample seed instead). Verified on 2 points before committing to a full
rerun: point 0's T_ref bias dropped +0.153 -> +0.057, and point 1's edge-pinning resolved
entirely (turned over inside the window, though with a new, smaller, seed-dependent bias
-0.11 to -0.21 -- plausibly resampling variance, since drawing only 250 jets from weights
spanning 5 orders of magnitude leaves a much smaller *effective* sample than 250).

**Full Phase D rerun with the fix (15/15 ensembles, ~3h15m, no walltime kill): T_ref is
essentially fixed.** `mle_mean.T_ref` bias is now 0.0000 at 4 of 5 points (the grid's center
node, which equals theta_true exactly for this grid construction, is now the argmax) and
+0.067 at the remaining point (0, where 2 of 3 ensembles still land on the narrow grid's own
edge). T_ref is no longer edge-pinned in 13 of 15 ensembles.

**This surfaced a second, pre-existing, and larger problem: b is now edge-pinned with
near-total posterior mass (0.9-0.9999) at the *same* node across independent ensembles, at 4
of 5 points** (bias exactly +/-1.375, the grid's own half-width) -- the earlier Phase D run
had already shown an echo of this (`b`'s MLE bit-identical, mle_std=0.000, across ensembles
at 3 points) but it was not investigated at the time. `tau_init` shows the same pattern more
weakly (edge mass 0.8-0.995 at most ensembles).

**Widened 1D sweeps for b and tau_init (all 5 points, weighted-resample) show this is not
the same bug: both parameters' likelihoods do turn over well inside a window spanning most of
the box, but the peak lands far from truth with large, inconsistent-sign scatter** -- b off by
+1.43, +3.05, -2.69, +0.33, -1.47 (box width 11); tau_init off by +0.11, -0.50, +0.50, +0.15,
+0.16 (box width 0.8), i.e. up to ~60% of the box range, both directions, point to point. This
looks like a genuine identifiability limit, not a bug: T_ref directly rescales the local
temperature that sets energy loss and leaves a strong, sharply-peaked per-jet signature;
b (impact parameter / path length geometry) and tau_init (formation time) plausibly have much
weaker single-jet signatures that only average out over many more than 250 jets, or need
observables beyond the four per-constituent features (deta, dphi, log_pt, log_e) this
checkpoint conditions on. **Phase D's narrow 3x3x3 grid (half-width = 25% of the box's own
half-range) is simply too narrow for b and tau_init** -- any report built from it needs to
say so explicitly rather than present their panels as if on the same footing as T_ref's.

Next: Phase E (the HTML explorer itself, reports repo) built from the corrected Phase D data,
with T_ref as the clean headline result and b/tau_init shown with this limitation stated
plainly. The old, bug-affected Phase D output is kept for the record under
`$JEST_NLE_ROOT/runs/jewel_v3/p5_explorer/ensembles_unweighted_buggy/`, not deleted.
§6. Nothing in the production code was changed to act on this.

## P5 follow-up addendum: the same weighted-resample bug fixed in the pre-existing P7 helper
## `toy.closure.jewel_ensembles` (2026-11-08)

Source revision 78d6262 (dirty: `src/omnilearned/sbi/toy/closure.py`,
`tests/sbi/test_jewel_closure_source.py` modified -- committed at the end of this entry).

`docs/RESTART_JEWEL_V3.md` trap 24 flagged this as open: `toy.closure.jewel_ensembles` (the
P7-scoped helper that draws disjoint JEWEL ensembles for the closure campaign) had the
identical bug `scripts/p5_explorer_ensembles.py` had -- it sliced `n * n_ensembles` jets
consecutively in file order from the pool matching the requested theta, with no reference to
`weight` at all, even though the same weighted-MC-split caveat applies to any real JEWEL
split it might be pointed at.

**Fix is not identical to the explorer's**, because `jewel_ensembles` supports *inclusive*
samples (several jets per JEWEL event, sharing `event_id` -> `group_id`), which the closure
report's composite-likelihood correlation check needs intact *within one ensemble*. A plain
jet-level sampling-importance-resampling (draw individual jets by weight, ignoring event
membership) was tried first and breaks this: with 40 jets at theta0 (20 events x 2 jets/event)
in the existing test fixture, a random jet-level draw of 8 can easily split both jets of a
pair into different ensembles, where the original (buggy) file-order slice never did, simply
because the shard happens to be written in event order. `test_inclusive_jets_get_unique_probe_ids_and_keep_their_event_grouping`
caught this immediately (`np.unique(group_id).size == len(group_id)` for an ensemble that
should have had repeats).

**Actual fix: resample whole *events*, not individual jets**, weighted by each event's own
weight (averaged over its jets, via `np.bincount` on `np.unique(..., return_inverse=True)`'s
compact group index -- weight should already be uniform within one event, since it is
assigned at generation/pT-hat-bin level, not per jet). Events are drawn without replacement
in weighted-random order, their jets concatenated in that order (keeping each event's jets
adjacent, exactly as file order happened to before), then truncated to `n * n_ensembles` and
sliced into disjoint consecutive chunks of `n` -- the same "slice a reordered pool into K
chunks" shape as the original code, just reordered by weight instead of file position. A
zero-weight event can't be drawn at all (`rng.choice(..., replace=False, p=...)` requires
every requested slot to have nonzero probability), so zero-weight events are filtered out
before the draw and their absence is folded into the existing "not enough jets" `ValueError`
rather than crashing.

New `seed` parameter (default 0, threaded from `prepare()`'s `spec.get("seed", 0)`, matching
the existing convention for the `source="surrogate"` branch). New test
`test_jewel_ensembles_respects_per_jet_weight`: a fixture where half of theta0's pool has
weight 0 asserts those jets (by `group_id`) are never drawn -- a deterministic version of the
explorer's empirical "~2x jet_pt mean" check, since this fixture has no real jet_pt signal to
measure a shift in. Full suite green (`tests/sbi`, 242 passed, 7 skipped) after the fix.

P7 itself still has not started (needs P3-P6 done first, per the plan), so this fix has not
been exercised against real JEWEL test-split data yet -- only the synthetic fixture above.

## P5 follow-up addendum 2: `jewel_ensembles` also had the hardcoded toy/v2 reader
## `direct_by_point` had (2026-11-08)

Source revision febc2a3 (dirty: `src/omnilearned/sbi/toy/closure.py`,
`tests/sbi/test_jewel_closure_source.py` modified -- committed at the end of this entry).

The weight fix above (addendum 1) only touched `jewel_ensembles`'s sampling; its reader was
still the hardcoded `from omnilearned.sbi.toy.datasets import read_shards, shard_paths`
import, unconditionally -- the exact gap `docs/RESTART_JEWEL_V3.md`'s original TODO named
("needs the same `shards_for_config(config)` dispatch [`direct_by_point`] had"). Added a
`config=None` parameter and the same `shards_for_config(config) if config else (shard_paths,
read_shards)` dispatch `gate.direct_by_point` already uses; threaded from `prepare()`'s
already-resolved `config` path. `jewel_ensembles` is now capable of reading real v3
test-split data, not just the toy/v2 schema.

**One real wrinkle found while wiring this up**: v3 shards already carry `event_id` (unique
per jet) and `group_id` (the originating JEWEL event; inclusive jets share it) as *separate*
columns (`src/omnilearned/sbi/jewel/v3format.py`'s own schema). The toy/v2 schema has only
one `event_id` column that plays both roles, which is what `jewel_ensembles`'s grouping logic
(both the weighted-event resampling, addendum 1, and the returned record's `group_id`) was
built against -- unconditionally reading `d["event_id"]` as the shared-event key would have
silently grouped v3 jets by their already-unique-per-jet id (grouping by almost-nothing,
one "event" per jet) instead of the real JEWEL event. Fixed by using `d["group_id"]` when
present, falling back to `d["event_id"]` for the toy/v2 schema
(`group_source = d["group_id"] if "group_id" in d else d["event_id"]`).

New test `test_jewel_ensembles_v3_dispatch_respects_weight_and_group_id` (a v3-format fixture
via `jewel.v3format.write_jets`, config pointed at `configs/jewel-2.6.0-cs-v3.toml`) checks
both the dispatch (data genuinely comes from the v3 reader) and that the returned `group_id`
reflects the v3 file's real grouping, not a 1:1 copy of the synthetic per-record `event_id`.
Full suite green after this fix too.

`direct_by_point` itself needed no change here -- it already had this dispatch (from earlier,
pre-this-session P4 work); only `jewel_ensembles` was still missing it. `direct_by_point` is
still hardcoded to the `"fidelity"` split name specifically (not a parameter), which is a
separate, narrower gap than "hardcoded reader" -- not touched, since nothing calls it needing
a different split yet.

## P5 follow-up, the formal re-test at nsteps=256/float64: gap shrinks ~3-4x, does NOT close
## (2026-11-08/09)

Source revision c38c282 (dirty: no). Device: Perlmutter GPU, alice_g, interactive QOS, 2
sequential jobs (`--which test`, `--which reference`; split after a combined `--which both`
2h run with no incremental save hit the walltime cap and lost everything -- see
`docs/RESTART_JEWEL_V3.md` trap 26, added from this).

The formal numerics check (`toy.numerics run`/evaluated directly via `evaluate()`, same
checkpoint, same `p5_obs.h5`, same 4 pairs around `primary_test` as the original 2026-11-05
FAIL), re-run at `nsteps=256`, `dtype=float64` for both solvers (3 seeds each,
`--event-chunk 40`), using the new `--dtype` flag added this session:

```
                           pair0 (b)   pair1 (T_ref)  pair2 (tau_init)  pair3 (diagonal)
midpoint-64/float32 (orig)   -14.09       52.63            13.33            51.19
rk4-64/float32 (orig)       -329.72     1345.04           225.62          1231.06
ratio (orig)                  23.4x       25.6x            16.9x            24.1x

midpoint-256/float64         -21.44       47.88            14.70            40.64
rk4-256/float64              -99.00      370.12            67.15           335.65
ratio (tonight)                4.6x        7.7x             4.6x             8.3x
```

`probe_std` (seed-to-seed scatter within each setting) stays tiny at the new settings too
(0.03-0.50), so both are again internally well-converged -- this is a real, reproducible
discrepancy between the methods, not noise. **Raising precision helps a lot (the gap shrinks
by roughly 3-4x) but does not resolve it**: by the same FAIL criteria as 2026-11-05 (single-
seed and seed-averaged shift both < 0.5 log-L units), this is still a decisive FAIL -- shifts
are 52-323 log-L units, not <0.5.

**This run conflated two changes at once** (nsteps 64->256 *and* dtype float32->float64), so
it cannot say how much of the 3-4x improvement came from which. Separating them (e.g.
midpoint-256/float32 alone) is the natural next step to isolate the effect, not done yet --
left for the user's direction given the cost already spent on this single data point (~4h
GPU: the test/midpoint side took ~26 min for 3 seeds (~530-545s/seed); the reference/rk4 side
took ~1h for 3 seeds (~3600s/seed, right against its own 3h45m walltime budget).

**This remains a decision point, not resolved**: "raise nsteps and switch to float64" is
*not* sufficient on its own to bring the production solver into agreement with the reference
at the rigor the formal P5 check demands. Whether rk4 itself has actually converged by
nsteps=256 (its own value moved by a similarly large factor between nsteps=64 and nsteps=256,
not obviously plateaued) is itself an open question raised by this data, not settled by it.
P6/P7 remain blocked per rule 6 until the user decides how to proceed: push nsteps/precision
further (and separate the two factors), accept a reduced-rigor production default on other
grounds, or read `likelihood.py`'s accumulation code more closely (skimmed during this
session's investigation -- the midpoint/rk4 divergence-accumulation logic looked
mathematically consistent with each method's own order, no obvious implementation bug found,
but not a line-by-line audit).

## P5 follow-up: a full line-by-line audit of likelihood.py (no bug found), and rk4-512
## confirms rk4 itself is nowhere near converged at nsteps=256 (2026-11-08/09)

Requested directly ("I believe we are still looking for a bug"): read `likelihood.py`
end-to-end. Checked: `_divergence`'s Hutchinson estimator (`eps^T J^T eps`, correctly
estimates tr(J) for Rademacher eps since trace is transpose-invariant, identical code path
for both solvers); `_fixed_step`'s midpoint branch (textbook 2-stage method, the companion
log-density ODE accumulated at the same evaluation point used for `x`, which is the
mathematically correct way to integrate it alongside the state); `_fixed_step`'s rk4 branch
(classical 4-stage weights 1-2-2-1/6, matching `sample()`'s independent rk4 implementation
elsewhere in the same file -- not a one-off inconsistency); sign conventions (`acc` added
positively on both branches, matching `logp = base_lp + delta + log_jac`); probe scheduling
(midpoint draws 1 fresh probe/step, rk4 draws 4 -- matches the documented `per_step` semantics
of "independent probes for every divergence evaluation" exactly, not an asymmetry bug, and
Hutchinson's estimator is unbiased for any fixed probe draw regardless). Also confirmed
`scripts/p5_trajectory_diagnostic.py` (the earlier exact-divergence follow-up) imports and
reuses `likelihood.py`'s own `_rhs_factory`/`rhs` closure directly, not a separate
reimplementation -- that earlier finding is evidence from the real code path. **No
implementation bug found.**

Then ran the test the audit couldn't settle on its own: rk4 at nsteps=512 (1 seed, same
checkpoint/pairs/observations), to check whether rk4 has actually converged by nsteps=256.
**It has not, and not by a wide margin:**

```
rk4-256/float64 (seed 0):  [ -99.015,  370.627,   67.144,  336.148]
rk4-512/float64 (seed 0):  [ -61.354,  214.043,   41.484,  192.755]
relative change:              -38%      -42%       -38%      -43%
```

A genuine 4th-order method should cut its error by ~16x when steps double (error ~ h^4); a
~40% change is nowhere close -- rk4 is not converging at its nominal order here. For
comparison, midpoint's own change from 128 (float32, 2026-11-05) to 256 (float64, tonight) was
much smaller in relative terms (0.8%, 12.5%, 4.1%, 15.8% -- though that comparison conflates
nsteps and dtype same as everywhere else this session). **Both solvers are still far from a
converged value**, and rk4's own degraded convergence rate (far below its nominal 4th order)
points toward the trained velocity field itself having insufficient smoothness for either
fixed-step method to resolve efficiently -- flow-matching ODEs integrated from the data
endpoint are known to be able to develop sharp/near-singular behavior, which would explain
degraded convergence for *any* fixed-step solver without requiring a bug in either one.
Re-running the ratio with the better-converged reference: midpoint-256 vs rk4-512 gives
ratios of 2.9-4.8x (down from 4.6-8.3x against rk4-256) -- consistent with both solvers
slowly approaching a common limit as precision increases, not with a systematic bug that a
convergence check would not shrink.

**This changes the practical picture, not just the numbers**: if rk4 needs far more than 512
steps to actually converge, "raise the reference's precision until it agrees with midpoint"
may be prohibitively expensive for routine use (nsteps=512 here cost 6840s for a single
4-pair, 1000-jet, 1-seed check -- a real fit's scan would need many more candidate
evaluations than that). This does not change the rule-6 block status (still unresolved), but
it does suggest the next useful experiments are about the vector field's smoothness near the
ODE's endpoints (t=0/1), not further brute-force step increases, if a practical production
setting is the goal rather than a fully converged reference.

## P4 re-opened: the beyond-R=0.4 halo deficit correlates with T_ref directly, not only
## through multiplicity -- upgraded from "accepted, low-priority" (2026-11-09)

P4's gate (2026-11-03/04) found the surrogate under-produces constituents beyond the jet's
R=0.4 cone (up to 39x deficit near dR~0.5), root-caused to genuine statistical scarcity of
real high-N training examples (`root_cause.html`: ~127 effective weighted examples at
N>160 out of 1.44M). The user accepted this as a known, low-priority tail issue given
excellent core-jet (dR<0.4) fidelity, and P5 proceeded. Asked directly whether this is
accounted for anywhere in actual inference (it is not -- `fit.py`/`scan.py`/`likelihood.py`
score the complete stored cloud unconditionally, no dR-based trim anywhere; `explorer_core.html`
was a diagnostic only, explicitly stated not to change how the gate or "the eventual NLE
likelihood" works), and then asked to test directly whether the deficit correlates with
theta, not only through N as `root_cause.html`'s own narrative implied.

**It does, and the correlation survives controlling for the multiplicity pathway.** Pulled
`theta` and `relevance.{width,outer_fraction}.delta_mean` (surrogate - direct) from all 128
P4 gate point JSONs (`$JEST_NLE_ROOT/runs/jewel_v3/p4_gate/s0/point_*.json`):

```
                        width_delta vs theta          outer_fraction_delta vs theta
raw Pearson r (p):      T_ref -0.41 (<1e-4)            T_ref -0.36 (<1e-4)
                        b     +0.24 (0.006)             b     +0.23 (0.011)
                        tau   -0.09 (0.34, ns)          tau   -0.07 (0.47, ns)

partial r | n_delta:    T_ref -0.56 (<1e-4)            T_ref -0.47 (<1e-4)
                        b     +0.34 (1e-4)              b     +0.31 (4e-4)
                        tau   -0.10 (0.26, ns)          tau   -0.07 (0.42, ns)
```

(`n_delta` = `relevance.n_constituents.delta_mean`, the surrogate's own multiplicity
deficit/excess -- used as the control variable for the "is this just N" test.) The partial
correlation controlling for the multiplicity pathway is **larger in magnitude**, not
smaller, than the raw one -- the T_ref relationship is not an artifact of N, it strengthens
once N's contribution is removed. Direction is physically sensible: higher T_ref -> stronger
quenching -> more real large-angle radiated constituents in the direct sample -> a roughly
fixed surrogate deficiency misses proportionally more of them, so `delta_mean` (surrogate
minus direct) becomes more negative. `tau_init` shows no significant correlation either way.

**Why this matters more than "low-priority tail":** T_ref is the parameter this project
infers most successfully (see the true-vs-inferred explorer). A surrogate weakness that
correlates with T_ref directly does not obviously cancel between nearby candidate thetas in
a fit's Δlog L the way a theta-independent deficiency would -- it is a plausible, previously
unverified source of T_ref bias, separate from (and not yet distinguished from) the
weighted-resample bug already found and fixed, or the identifiability limits found for
b/tau_init. Not yet known whether its magnitude at the likelihood level is large enough to
matter in practice; not measured here.

**Mitigation, three strands, discussed and approved to proceed (2026-11-09):**

1. **Immediate, no retraining**: restrict inference to the core (dR<0.4) region, which
   `explorer_core.html` already showed has excellent (1-3%) direct-vs-surrogate agreement.
   Removes the demonstrated bias source at the cost of discarding whatever real quenching
   signal lives in the halo (plausible, given the deficit's own severity tracks T_ref).
2. **Addresses the root cause**: retrain with a stratified/importance-corrected sampling
   scheme that oversamples rare high-N (and, per this finding, high-T_ref) events during
   training while correcting the loss weight inversely so the trained objective still
   targets the true physical distribution -- not simply `--weight-mode sample` (which
   samples *by* physical weight, the opposite of oversampling the physically rare tail) or
   `loss` (unweighted sampling, physical-weight loss) already in `train.py`; a new scheme.
3. **Complementary**: generate additional targeted statistics concentrated at the high-T_ref
   end of the design box, where real high-N jets are rarer to begin with -- (2) cannot
   recover statistical power that was never generated in the first place.

Implementation status of each, as of this entry, below.

**(1) core-restriction, done.** `observations.restrict_to_core(obs, r_max=0.4, require_min=1)`:
zeroes and masks out constituents at dR>=r_max, by name-lookup of `deta`/`dphi` in
`feature_names` (not a hardcoded column index). Padded (already-False) slots are never
revived by a small dR coincidence; a jet left with fewer than `require_min` core
constituents raises rather than silently scoring a near-empty cloud. Wired into
`scripts/p5_explorer_ensembles.py` as an opt-in `--core-only` flag (off by default, so this
stays an A/B toggle rather than a silent behavior change) -- also fixed that script's
`Observations` construction to pass `feature_names` at all, which it never did before
(harmless until now, since nothing downstream needed it). Tested
(`tests/sbi/test_observations_core.py`, 6 cases). Not yet re-run against real Phase D data
to confirm it actually shrinks the T_ref bias/scatter empirically -- that is a GPU-hour
cost, deferred alongside deciding whether to also re-run Phase D with (2)'s checkpoint once
it exists, rather than doing two separate expensive re-runs.

**(2) stratified training, mechanism implemented and tested, full retrain not yet launched.**
New `flow.stratified_weights(weight, strata)`: for a per-event integer `strata` (e.g. a
joint multiplicity x T_ref bucket from new `toy.train.joint_strata`), returns
`(sample_weight, loss_weight)` where `sample_weight` gives every stratum equal total draw
probability regardless of its physical rarity, and `loss_weight = weight/sample_weight` is
the self-normalized-importance-sampling correction that exactly cancels that boost in
expectation -- so training draws rare strata often enough to learn from while the trained
objective still targets the true physical law, not the flattened one. New
`weight_mode='stratified'` in `train_flow`/`train.py`'s CLI, alongside the existing `none`/
`loss`/`sample`. Important design point found while implementing: `sample_w` (the boosted
draw weight) is train-only; `w` stays the plain physical weight everywhere, including for
validation, which is always drawn uniformly and therefore always needs the plain-weight
correction regardless of how training draws -- conflating the two would have corrupted the
validation metric used for model selection.

Caught and fixed one real bug in this same change before it reached any real training: the
draw-weight selection was originally written as a dict literal
(`{"sample": tr["w"], "stratified": tr["sample_w"]}.get(mode)`), which evaluates every
value in the literal eagerly regardless of which key is looked up -- `tr["sample_w"]` was
being accessed (and raising `KeyError`) even in `none`/`loss` mode, where it does not
exist. Caught by the existing `tests/sbi/test_weighted_training.py` suite failing on
unrelated, pre-existing tests after the change (6 failures, all in modes this change was
not even supposed to touch) -- exactly the value of running the *whole* suite after a
shared-code change, not just the new tests. Fixed to a plain conditional; full suite green
after.

New tests: `stratified_weights`'s IS-correction identity verified directly (flatten +
correct reproduces the plain-weighted expectation of an arbitrary function, Monte Carlo);
`joint_strata`'s quantile binning (no empty strata, monotone in each axis); `prepare(...,
strata=...)`'s wiring (`sample_w` matches `stratified_weights`, `w` stays physical,
requires `weights=True`); and the existing biased-1D-toy end-to-end recovery test
(`test_weighted_flow_fit_on_a_1d_toy_recovers_the_true_density`) extended to run
`weight_mode='stratified'` through actual `train_flow` and confirm it still recovers the
true (unbiased) density, same as `loss`/`sample`.

**Not yet done: the actual full retrain.** This mechanism has only been exercised on tiny
synthetic data (hundreds to thousands of points, CPU, seconds). Launching it on the real
~10.7M-jet v3 train split (400k steps, ~4h GPU, matching P3's original run) is the next
step, to a **new** checkpoint path (never overwriting `v3_set_d256_L6_400k_s0`, so this is
purely additive and carries no risk to the existing production checkpoint or to anything
already measured against it this session) -- not yet launched as of this entry.

**(3) targeted high-T_ref generation, designed, not executed.** Proposed: ~40 additional
Sobol-style theta points concentrated in T_ref in [0.6, 0.9] (b, tau_init spanning their
full existing ranges), ~20,000 jets/point to match the existing train split's density,
~800k new jets. Cost estimate from the plan's own measured rate (0.545 core-s/jet,
`docs/PLAN_JEWEL_V3.md` §2): ~121 core-hours =~ 1.9 node-hours at 64 workers/node -- modest.

**Why this one was not just launched tonight, unlike (1)/(2):** `jewel/generate.py`'s
`plan()`/`cards.njob()` key NJOB uniqueness off `(split, point_id, chunk)` with a
*per-split* constant base (`[randomness].njob_base.train = 100000`,
`chunks_per_point = 100`); the existing train split already occupies point_id 0-511 under
that base. Naively planning 40 new points as point_id 0-39 under `split="train"` would
silently reuse NJOB values the existing 512-point design already used -- not an error,
a wrong number, exactly the kind of silent trap this file's "traps already hit" list is
for. Point_id 512-551 avoids that collision safely (NJOB tops out at ~155199, far below
validation's base of 300000) -- but a second, harder question is still open: how to get
the new shards into the dataset `prepare()`/training actually reads. The existing merged
`train_*.h5` files under `$JEST_NLE_ROOT/data/jewel_v3/merged/` already represent the
original 512-point design; adding 40 more points needs either a full re-merge including
the new point's raw task shards (if those still exist) or a supplementary merged file plus
a change to how training reads multiple merge directories -- not resolved here, and not a
decision to make quickly at the end of an already long session, given a mistake in shared
dataset state has a materially worse failure mode (silently wrong data for everyone, not
just this run) than anything in (1)/(2). Left as the next concrete, scoped increment.

## Correction: premium QOS access was never actually blocked -- the QOS name was wrong
## (2026-11-09)

Earlier this session (see the 2026-11-07/08 entry above): every `sbatch`/`salloc` attempt
at `-q gpu_premium` failed ("Job request does not match any supported policy"), despite
`alice_g`'s entitlements and the `gpu_ss11` partition's `AllowQos=ALL` both appearing to
permit it; concluded (wrongly) that NERSC needed to provision a separate premium
allocation bucket, and deferred per the user's "revisit premium later."

**The QOS is named `premium`, not `gpu_premium` -- a name that does not exist was being
requested the whole time, which is why nothing about entitlements or partitions explained
the failure.** `sacctmgr show assoc account=alice_g user=$USER format=qos` lists
`premium` (not `gpu_premium`) among `alice_g`'s entitled QOS; `sacctmgr show qos premium`
confirms priority 73440, 2-day max walltime, `MaxSubmitPU=5` -- the exact figures the
earlier investigation found for `gpu_premium` and attributed to a QOS that, it turns out,
was never the one being requested. `sbatch --test-only -q premium -A alice_g ...` returns
a real scheduled start time, confirming it actually works. No further NERSC action needed;
this was not a provisioning gap.

## Mitigation (2) retrain: complete, checkpoint `v3_set_d256_L6_400k_s0_stratified`
## (2026-11-09)

Snapshot `86e7498` (the commit adding `weight_mode="stratified"`), submitted via
`-q premium -A alice_g` (newly unblocked, see the entry above) -- 400,000 steps, otherwise
identical hyperparameters to `v3_set_d256_L6_400k_s0` (`--arch set --dim 256 --depth 6
--heads 8 --batch-size 512 --lr 3e-4 ... --weight-mode stratified`), writing to a **new**
output path, never touching the existing production checkpoint. Completed cleanly, no
errors: 400,000/400,000 steps, 13497s (~3h45m, comparable to the original run's ~4h11m),
best step 364000, val loss 0.88611, multiplicity val NLL 3.3405 (weighted). Checkpoint sha256
`d6a161f7a985...`.

`run.json`'s `weights` record: `ess_train` (physical weight alone) 1,417,928 of 10,705,283
jets; `ess_sample_draw` (the stratified draw) **890K -- lower, not higher**. This is the
expected trade-off, not a problem: flattening draw probability *across* 40 strata while
those strata hold wildly unequal event counts necessarily concentrates probability onto
*fewer individual events* within the sparse strata (an event in a stratum of 1000 now shares
the same total stratum mass a stratum of 1,000,000 shares one event each of) -- lower global
ESS, much better coverage of the specific rare combinations (high multiplicity, high T_ref)
the P4 finding traced the halo deficit to. Validation loss is not comparable across
differently-weighted runs (standing project caveat) and was not used to judge this -- the
real test is whether the beyond-R=0.4/T_ref correlation (the `width`/`outer_fraction` vs
T_ref partial correlation measured earlier in this entry) shrinks when the P4 gate's
correlation check is re-run against *this* checkpoint instead of the original. **Not yet
done** -- the natural next step, not executed in this entry.

## Mitigation (2) validation: inconclusive, not clearly positive (2026-11-09)

Ran `gate.py run` against the stratified checkpoint on 16 points evenly spaced across the
T_ref range (point_ids 70,42,4,14,80,30,48,58,89,43,5,15,81,31,49,59 -- the same points
re-read from the original campaign for a paired comparison), reduced cost
(`--n-generate 4000 --n-perm 50`, vs. the original campaign's 22000/200) via the newly-fixed
`premium` QOS. One real bug hit and fixed en route: the default `--batch 1024` OOM'd on the
first high-multiplicity point after point 4 (21 GiB single allocation on an already
67/79 GiB-used GPU) -- `run()` has no per-point isolation, so the whole process died, not
just that point; resumed with `--batch 256` (already-done points skipped automatically) and
it completed cleanly.

**Result: no clear improvement, and if anything a very weak move in the wrong direction,
though nothing here is statistically significant either way.**

```
                              T_ref vs width_delta      T_ref vs outer_delta
original checkpoint (n=16)     r=-0.12 (p=0.67)          r=-0.10 (p=0.71)
stratified checkpoint (n=16)   r=-0.21 (p=0.44)          r=-0.21 (p=0.43)
```

mean |width_delta| unchanged (0.0024 -> 0.0024); mean |outer_delta| dropped modestly
(0.0107 -> 0.0099, ~8%). Per-point deltas show no consistent improvement pattern -- some
points better, some worse, no systematic shift.

**Caveat this isn't a clean disproof either**: the *original* checkpoint's own correlation
on just this 16-point subsample (r=-0.12, p=0.67) is far weaker than the full 128-point
result this session measured earlier (r=-0.41, p<1e-4) -- 16 points at reduced
`n_generate`/`n_perm` is simply underpowered to reliably replicate the baseline effect, let
alone detect a partial fix on top of it. This validation is inconclusive, not a clean
negative result.

**Most likely explanation if the null result holds up at full power: stratified
resampling can only make training see the *same* limited set of rare real examples more
often -- it cannot manufacture new statistical diversity.** `root_cause.html`'s own
estimate (~127 effective weighted training examples at N>160, out of 1.44M total) may be
the real ceiling; reweighting access to those same ~127 examples harder is a different
intervention than having more of them to begin with. This is exactly why mitigation (3)
(new targeted high-T_ref generation, designed but not executed this session -- see above)
was scoped as *complementary* to (2), not redundant with it, and may be the one that
actually matters if this result holds at proper statistical power.

Not done in this entry, left as the honest next step rather than a decision made here: a
properly-powered re-test (full 128 points or at least a larger, still-reduced-cost subset,
at the original `n_generate`/`n_perm`) before concluding either way whether (2) helped.

## Mitigation (3) campaign complete: 40 high-T_ref points generated and merged (2026-11-10)

Generated the 40 supplementary high-T_ref points designed in `p4_hightref_design.json`
(point_id 512-551, T_ref in [0.610, 0.893], b/tau_init spanning their full existing box
ranges, scrambled Sobol on the dedicated `jewel_train_hightref` stream). Ran via
`generate.py work` in waves (CPU interactive/premium QOS), `unlock-stale` + `index`'s own
corrupt-manifest self-healing cleaning up walltime-kill artifacts between waves (9 zero-byte
manifests from the first wave's walltime kill; the dedicated `unlock-stale` subcommand
separately freed 17 stale claim locks from the same kill). Topped up points 512/513 (the
original smoke-test points, which had only 2 chunks each -- 4,748 and 3,710 jets, both below
the ~20,000/point target) to the same 15-chunk depth as the other 38 points via a manual
`tasks.json` append mirroring `topup()`'s own per-point chunk-extension logic (NOT
`topup()` itself, which tops up every under-target point in the WHOLE split including
existing low-acceptance production points -- out of scope here and not what was wanted).

Final per-point yield, all 40 points: min 27,244, mean 32,729 jets/point, none below the
20,000 target. Full `train/` index: 23,409/23,409 tasks, 12,014,987 jets, 552 points.

Merge hit a real bug: `generate.py`'s generic `merge()` is dead code for the v3 backend
(reads an attribute name the v3 writer never writes -- see RESTART trap 29). Switched to
the correct `v3format.merge_v3`, which in turn was far too slow run serially (projected
~4.5-5h for 23,409 small-file reads, Lustre-metadata-latency-bound not CPU-bound -- see
RESTART trap 30) and blew through two walltime budgets with zero output (no incremental
save). Wrote a resumable, parallel wrapper (one worker per output group, skip-if-already-
written per group) and reran: all 13 groups of the full 552-point split merged in ~35
minutes. Verified: merged jet total (12,014,987) exactly matches the source index; per-file
point counts sum to exactly 552 (every point present, none split across files, none
duplicated).

Output: `$JEST_NLE_ROOT/data/jewel_v3/merged_hightref/` -- a **separate, non-production**
merged dataset (13 files, `index_train.json`, `dataset.json`/`.yaml`) spanning the original
512 points plus the 40 new ones. The existing production `merged/` (512 points) is
untouched; nothing here changes what the current production checkpoints were trained on.

**Not yet done, and the actual point of the campaign**: retrain on this expanded dataset
(possibly combined with mitigation (2)'s stratified weighting) and re-run the P4 gate's
T_ref correlation check against that new checkpoint, at full statistical power (all 128
fidelity points, not the underpowered 16-point subsample mitigation (2) alone was validated
against). Until that retrain happens, mitigation (3) has supplied more real high-T_ref
statistics but has not yet been shown to fix anything.

## Mitigation (2)+(3) combined retrain complete (2026-11-11)

Retrained on `merged_hightref/` (the 552-point dataset from mitigation (3)) with
`--weight-mode stratified` (mitigation (2)), same architecture/hyperparameters as the
`v3_set_d256_L6_400k_s0_stratified` checkpoint otherwise (set arch, dim 256, depth 6,
heads 8, batch 512, lr 3e-4, 400k steps, seed 0). `--val-dir` pointed at the existing
production `merged/`'s validation split (unrelated to and untouched by mitigation (3),
which only merged the train split into `merged_hightref/`) -- val loss is a monitoring
signal only, never a fidelity judgment, per the standing project caveat, so reusing the
existing validation set here is the right call, not a shortcut.

One real bug hit on the first submission: `--val-dir merged_hightref` crashed with
`ValueError: no v3 shard files given` (`v3format.read_v3_shards`), because `merged_hightref/`
only has `index_train.json` -- mitigation (3) never merged a validation split into it (there
was no reason to; mitigation (3) only adds *training* points). Fixed by pointing `--val-dir`
at the production `merged/` instead.

Checkpoint: `v3_set_d256_L6_400k_s0_hightref_stratified`, sha `3ea5328a5c86`, best step
378,000, val(ema) 0.88604, multiplicity val NLL 3.3452, 15,597s total (~4h20m). Train ESS
1,757,570 of 12,014,987 jets (vs. 1,417,928 of 10,705,283 for the stratified-only run on the
original 512-point data -- a genuine +24% increase in effective training statistics, not
just a reweighting of the same examples).

**Val loss (0.88604) is essentially unchanged from the stratified-only run's 0.88611** --
expected and not a concern: val loss is dominated by the bulk, low-multiplicity/low-T_ref
population the halo-deficit finding never implicated; it is not sensitive to a fix
targeted at a small tail. **Not yet done**: re-run the P4 gate's T_ref correlation check
against this checkpoint (ideally at full statistical power -- all 128 fidelity points, not
the underpowered 16-point subsample mitigation (2) alone was checked against) -- this is
the actual test of whether mitigation (3)'s extra real statistics, combined with (2)'s
reweighting, close the gap that (2) alone did not.

## Mitigation (2)+(3) full 128-point P4 gate re-check: fixed the target, broke something else (2026-11-13)

Ran the full-power P4 gate (`n_generate=22000`, `n_perm=200`, `--redraw`, `--batch 256`,
matching the original production campaign's own settings exactly) against the combined
checkpoint `v3_set_d256_L6_400k_s0_hightref_stratified`, all 128 fidelity points.

**Headline: worse, not better. 30/128 FAIL (alpha/S=3.91e-04), vs. the original production
checkpoint's 19/128 FAIL.** Not spin -- a real net regression in the family-wise result.

Real bug hit and fixed along the way (separate from the result above): `sample_and_select`'s
redraw loop could pass a 0-constituent cloud's count into `surrogate.sample`, which correctly
refuses `num_particles=0` (see the commit fixing it; a 0-constituent cloud can never pass
`min_constituents>=1` selection, so "redraw at the same N" is meaningless for it and it should
just count as permanently unfilled). Crashed shard 3 of the first attempt after 14/32 points;
fixed, tested, and the resumed run (gate.py's own existing-point skip) picked up cleanly from
95/128 without redoing finished points.

**But the regression is not uniform -- comparing per-point, per-observable against the
original checkpoint's own 128 point JSONs shows a real, mixed picture:**

- **3 points flip FAIL -> PASS**, all at the highest T_ref in the design (0.75, 0.78, 0.89 --
  exactly mitigation (3)'s target band). Their original failure mode was `width` and
  `outer_fraction` -- precisely the beyond-R=0.4 halo-deficit symptom `root_cause.html`
  diagnosed. **This is the campaign's targeted fix actually working** at the points it was
  built for.
- **14 points newly fail** that passed under the original checkpoint. Every single one is
  flagged by `n_constituents` (14/14); only 1 of 14 also shows `width`. Their T_ref is
  moderate-high (mean 0.62) but spans a different, broader band than the 3 fixed points, not
  concentrated at the extreme tail. This looks like a side effect of the retrain itself --
  most likely the stratified reweighting (which boosts draw probability of rare
  high-multiplicity/high-T_ref training examples) shifting the learned multiplicity
  calibration p(N|theta) somewhat broadly, not a reappearance of the halo-specific problem.
- 16 points fail under both checkpoints (an unrelated, persistent issue); 95 pass under both.
  Net: 19 (orig) + 14 (new-only) - 3 (fixed) = 30 (new) -- the arithmetic behind the headline
  number.

**Implication, not yet acted on**: mitigation (3)'s extra high-T_ref statistics, combined with
mitigation (2)'s stratified reweighting, did close the specific gap they targeted, but the
reweighting (or some other side effect of the retrain) introduced a broader, different
multiplicity-calibration cost elsewhere that outweighs it in the family-wise count. Candidates
worth checking before concluding further: (a) retrain on the expanded `merged_hightref/`
dataset WITHOUT stratified weighting (`--weight-mode loss`, isolating mitigation (3) alone
from (2)) to see whether the extra data alone fixes the target without the new
n_constituents-driven failures; (b) inspect whether the stratified multiplicity model
specifically (not just the flow) was affected, since `joint_strata`'s stratification is over
(n_constituents, T_ref) jointly and could plausibly be reweighting the multiplicity model's
own training signal more than intended. **This checkpoint should not be treated as an
improvement over production as it stands.**

## P4 gate: how badly, exactly, and the near-neighbor instability (2026-11-16)

Follow-up to the full 128-point gate re-check above: the family-wise p-value/relevance-score
table doesn't show the actual size of the predicted-vs-true mismatch, and doesn't explain a
genuinely odd pattern -- several theta points right next to each other in the design land on
opposite sides of FAIL/PASS. Investigated both with a new HTML section (reports repo,
`reports/m3-surrogate-fidelity/2026-11-16T1653-0800_jewel-v3_hightref-stratified-gate-comparison/index.html`,
"How badly, exactly?" section).

**The near-neighbor pattern is real but mostly not alarming.** Nearest-neighbor distance
over all 128 points (normalized box coordinates) has median 0.165; 44 pairs under the new
checkpoint (27 under production) have opposite FAIL/PASS verdicts at distances 0.08-0.17 --
the close end of the distribution but not pathologically so. For 3 representative pairs
(recomputed directly from the published checkpoint/seed/settings via
`gate.py run --points ... --save-samples`, not a cheaper approximation), the actual
predicted-(surrogate)-vs-true-(direct) observable gaps are **small in absolute terms** for
2 of the 3 pairs -- a few percent of the observable's own scale. The FAIL/PASS line falls
wherever the error bars happen to just barely separate or overlap at `n_generate=22000`'s
very high statistical precision, not because the two neighboring points are physically very
different. This is the direct answer to "why do neighbors disagree": tiny, real,
well-resolved shifts near the edge of significance, not large qualitative jumps.

**One pair is a genuine exception, not borderline at all**: point 52 fails by a decisive
margin on nearly every one of the 7 gate observables, under BOTH checkpoints, while its
neighbor (point 100, distance 0.124) passes cleanly on all 7. This is a real, large, locally
consistent mismatch at that specific theta, not a statistical coin-flip -- worth flagging as
a candidate for its own investigation if the surrogate's local behavior near that corner of
the box ever becomes a priority (not pursued further in this entry).

Real bug caught and fixed while building the recompute: the first version of the analysis
script used each point JSON's own `verdict.fail` field, computed by `gate.py run()` at an
**uncorrected** alpha=0.05 (its per-point default) -- not the Bonferroni-corrected
alpha/S=3.91e-04 this whole campaign (and the original `collect.py`, which explicitly flagged
this exact trap) uses everywhere else. Recomputing `fail` from `p_family < alpha_s` directly
fixed it; the underlying means/standard errors were unaffected.

## Isolation retrain result: mitigation (3) alone is not the fix -- it's (also) the cause (2026-11-17)

Retrained on `merged_hightref/` with `--weight-mode loss` (production's own setting, no
stratified reweighting) to isolate mitigation (3)'s extra high-T_ref data from mitigation
(2)'s reweighting. Checkpoint `v3_set_d256_L6_400k_s0_hightref`, sha `cb2377b4fd2c`, best
step 386,000, val(ema) 0.88629 -- essentially identical to every other run's val loss, as
expected (not sensitive to this effect). Full 128-point gate re-check (same settings
throughout: `n_generate=22000`, `n_perm=200`, `--redraw`), three resubmissions needed
(walltime-killed at 92/128, then 120/128, then 125/128 -- each resumed cleanly, no lost
work, final 3 points were simply slow).

**Result: 31/128 FAIL -- WORSE than the combined mitigation (2)+(3) checkpoint's 30/128,
and worse than production's 19/128.**

**This retracts the hypothesis in the entry above** ("most likely a side effect of
mitigation (2)'s stratified reweighting"). The n_constituents-driven regression is not
caused by stratified reweighting -- it is present, if anything slightly *more* present,
when mitigation (3)'s extra high-T_ref training data is added WITHOUT any reweighting at
all. Three-way point-by-point comparison (production vs. mit(2)+(3) vs. mit(3)-alone):

- mit(3)-alone: 16 points newly fail vs. production, flagged almost entirely by
  `n_constituents` (16/16), plus `ptd`/`soft8_fraction`/`lead_fraction` -- same signature as
  mit(2)+(3)'s newly-failing set. Their T_ref spans nearly the *entire* design range (0.15
  to 0.86), not just the high-T_ref target band.
- mit(3)-alone: 4 points fixed vs. production (vs. 3 for mit(2)+(3)), all at the highest
  T_ref (0.75-0.86) -- the targeted width/outer_fraction halo-deficit fix happens whether or
  not stratified weighting is used, and slightly more so without it.
- mit(2)+(3) vs. mit(3)-alone directly: 26 points fail under both, 4 fail only under
  mit(2)+(3), 5 fail only under mit(3)-alone -- stratified reweighting has a real but mixed,
  roughly washed-out effect on top of the data addition; it is not the main driver of either
  the fix or the regression.

**Updated understanding**: simply adding more real high-T_ref training examples (mitigation
3) is enough, by itself, both to fix the targeted beyond-R=0.4 halo-deficit symptom at the
most extreme points AND to perturb the model's overall multiplicity (n_constituents)
calibration broadly across the box. This is not obviously a training-weighting artifact;
it looks more like a genuine capacity/generalization trade-off from this particular
training-set change (552 points instead of 512, with the new 40 concentrated in a T_ref
corner but otherwise ordinary-looking individual points) -- changing *what* the model is
trained on shifted its calibration everywhere, not just near the new data. Not yet
investigated: whether this is a capacity limit (same architecture, more/different data),
a side effect of the per-point weight renormalization (`merge_v3` rescales weight so
sum(weight)=n_jets *per point*, so adding 40 points changes every point's relative
influence on the loss slightly, not just the new ones' own), or something else. Next step,
not yet decided: whether to pursue this further (e.g. retrain with the new points added but
reweighted down to their "natural" share rather than equal per-point weight) or treat the
beyond-R=0.4 halo deficit as the known, accepted limitation it was before mitigation (3)
started (per the original P4 gate's own accepted-known-issue decision).

## P4 gate: bulk shift vs. tail effect, point-by-point (2026-11-17)

Follow-up: what does the "n_constituents"-driven FAIL actually look like at the
distribution level, not just mean+-SE? Pulled full (weighted) per-jet distributions --
true (direct) vs. predicted (surrogate) -- for 3 representative points under the isolation
checkpoint (`v3_set_d256_L6_400k_s0_hightref`), using `gate.py run --points ... --save-samples`
at the same settings as the published gate (new HTML section, reports repo, same report
directory as the comparison and near-neighbor addenda above, "What does FAIL actually mean?").

**The mechanism differs by point, and both exist:**
- **Point 88** (newly fails, T_ref=0.146, outside the target band): **pure tail effect**.
  The bulk (N<35, >99.99% of the probability mass) overlaps true and predicted almost
  exactly. The mean shift that trips the relevance test (11.5 vs 11.8) comes entirely from
  the surrogate under-producing a real, rare high-multiplicity tail (N>45, true probability
  down to ~1e-7/jet) that it should reproduce but mostly doesn't.
- **Point 107** (fixed by mitigation 3, T_ref=0.824, inside the target band): good agreement
  through the bulk and most of the tail; the only visible difference is noisy over/under
  production in the sparse extreme tail (N>90, a handful of jets on each side, no consistent
  direction) -- tail statistics noise, not a systematic effect.
- **Point 52** (fails under every checkpoint tried so far, T_ref=0.198): the opposite
  picture. All 4 flagged observables (`lead_fraction`, `n_constituents`, `ptd`,
  `soft8_fraction`) show a genuine, consistent shift across the **whole bulk** distribution,
  not a thin tail -- a real, broad local miscalibration of the surrogate at this theta,
  present before mitigation (3) and unrelated to it.

**Practical takeaway for reading any future gate FAIL**: the family-wise test (permutation
test + per-observable relevance) cannot by itself distinguish "the surrogate's tail is a bit
thin" from "the surrogate is locally wrong everywhere" -- both trip the same flag. Pulling
the actual distribution (cheap: one `gate.py run --points N --save-samples` call) is the
right next step before concluding anything about a specific FAIL point's severity or cause.

## Correction to the bulk-vs-tail entry above: the tail punches above its weight, but isn't the whole story (2026-11-17)

Follow-up question challenged the "pure tail effect" claim above (the log-scale histograms
*look* like all the mismatch sits 2+ orders of magnitude below the peak). Decomposed each
mean-shift quantitatively by percentile of the true distribution instead of trusting the
plot by eye: for each point/observable, compute the mean-shift restricted to just the
bottom 90% of jets (where the histograms visually overlap) and compare to the total shift.

**Honest result: a mix, not purely tail-driven for most of these.**

| point | observable | share of total mean-shift already present in the bottom 90% of jets |
|---|---|---|
| 88 | n_constituents | 66% |
| 107 | n_constituents | 22% |
| 107 | ptd | 61% |
| 52 | n_constituents | 43% |
| 52 | lead_fraction | 89% |
| 52 | ptd | 91% |

The rarest jets are genuinely disproportionate (point 88's rarest 0.1%, N>31, alone account
for 10% of its shift despite being 1-in-1000 jets) -- the earlier intuition that the tail
matters more than its probability mass isn't wrong. But calling point 88 a "pure tail
effect" overstated it: 66% of its shift is already present in the 90% of jets where the
histograms visually overlap -- a small, real, per-bin difference too subtle to see on a log
axis, not confined to the tail at all. Point 107 is the one case that really is
tail-dominated (78% from the top 10%). Point 52's `lead_fraction` and `ptd` are
overwhelmingly bulk-driven (89-91%), confirming it as the real, broad local miscalibration
described above, not a tail artifact in any sense.

**Lesson for next time**: a log-scale overlay plot is good for spotting *where* two
distributions visibly differ, but is actively misleading for judging *how much* of a
mean-shift comes from where -- a uniform few-percent shift across the bulk is invisible by
eye on a log axis but can dominate the total. Decompose quantitatively (as above) before
describing a mismatch's character, not just by looking at the plot.

## Mitigation (3) campaign closed: shifts accepted, adopting the combined checkpoint (2026-11-18)

User reviewed the actual quantitative shifts behind every FAIL in the isolation/combined
checkpoint comparisons above (see "P4 gate: how badly, exactly" and the bulk-vs-tail
entries) and confirmed: **few-percent-level shifts in the mean of a jet observable are an
acceptable level of discrepancy.** Every shift examined was 2-7% of the true mean (e.g.
point 88's n_constituents +2.7%, point 107's +7.4%, point 52's soft8_fraction -7.4%) --
nothing was off by a factor of 2 or more. The family-wise test's enormous statistical power
(tens of thousands of jets per point) means it can and does flag real differences at this
magnitude as decisive FAILs; that sensitivity does not by itself make the magnitude
practically significant.

**This reframes the net verdict.** The 19->30/31 FAIL count increase from mitigations
(2)/(3) is not evidence of a meaningful regression at this tolerance -- it is largely the
gate's sensitivity surfacing small, acceptable shifts across more points. Mitigation (3)'s
own target fix (the beyond-R=0.4 halo deficit's width/outer_fraction symptom at the highest
T_ref points) remains a genuine, uncontested improvement at those points.

**Decision: adopt `v3_set_d256_L6_400k_s0_hightref_stratified`** (mitigation (2)+(3)
combined: 40 extra high-T_ref training points + stratified reweighting, sha
`3ea5328a5c86`) as the working checkpoint going forward, in place of production
`v3_set_d256_L6_400k_s0`. It fixes a real problem with no cost that matters at this
tolerance.

**Point 52 remains flagged separately, not closed by this decision.** Its mismatch (4
observables, 2-7% shifts but consistent across the *entire* bulk distribution, not a tail
artifact -- see the bulk-vs-tail decomposition) is present in every checkpoint tried,
including production, and predates mitigation (3). Not investigated further in this entry;
noted as a candidate for separate attention if the surrogate's local behavior near
b=3.48, T_ref=0.198, tau_init=0.636 ever becomes a priority.

**Next**: return to the standing P5 blocker (solver-precision discrepancy between
`midpoint` and `rk4`, ~20-25x in Delta-log-L, which is a different kind of problem -- a
large discrepancy, not a few percent -- and has been gating P6/P7 throughout this entire P4
mitigation detour). See FINDINGS "P5 follow-up" / "a full line-by-line audit... rk4-512" for
where that investigation left off: the trained velocity field's own smoothness (likely near
the ODE's t=0/1 endpoints) is the leading candidate for the real constraint, not a code bug
(ruled out by the full line-by-line audit) and not pure under-resolution (ruled out by
rk4-512 itself failing to converge as a true 4th-order method should).

## P5 resumed: the velocity field's divergence genuinely spikes at t=0, not t=1 (2026-11-18)

Picked up the standing P5 decision point ("investigate the vector field's smoothness
directly"). Used `scripts/p5_trajectory_diagnostic.py` (exact, non-stochastic divergence,
float64) on 2 real jets from `p5_obs.h5` at `primary_test` theta=(4.5, 0.45, 0.65), logging
every per-step divergence value (not just the final accumulated sum) -- checkpoint
`v3_set_d256_L6_400k_s0` (sha `7ce2e63d5504`, the same one all prior P5 numbers used).

**Jet 904 (N=5 constituents), nsteps=256:** midpoint's per-step divergence `d2` at t=0
exactly is **3101**, vs. a bulk value (middle of the trajectory) of ~35 -- a ~90x spike that
decays to bulk level within the first ~9 steps (~3.5% of the trajectory). The
midpoint/rk4 ratio tracks this exactly: std=0.163 in the first 5% of steps, vs. std=0.004 in
the bulk and std=0.000 (!) in the last 5% of steps, near t=1. t=1 shows essentially perfect
solver agreement throughout (e.g. d2=19.9603 vs. rk4-avg=19.9594 at the second-to-last step).

**Jet 26 (N=25 constituents), nsteps=64:** identical pattern. d2 at t=0 is **2888.8** vs. a
bulk value of ~100-143 (~20-29x spike); the midpoint/rk4 ratio at that very first step is
3.59, collapsing to 1.00-1.06 by the second step and staying at 1.000 throughout the back
half of the trajectory (t>0.9).

**Ruled out: this is not an autograd bug.** `omnilearned.sbi.smoothness.finite_difference_divergence`
(an existing, previously-unused diagnostic module -- `git log` shows it predates this
session, from the "Route A" package work) compares the exact autograd divergence against a
central finite difference at a range of t. Relative agreement is excellent everywhere,
including right next to the singularity: max_rel_diff = 0.0038 at t=1e-4, 4.6e-5 at t=1e-3,
shrinking to ~1e-10 by the bulk. The large *absolute* difference very close to t=0 (64.8 at
t=1e-4) is exactly proportional to how large the divergence value itself is there -- i.e.
autograd is computing the (large, real) value correctly, not malfunctioning.

**Conclusion: a genuine near-singularity in the trained velocity field's divergence sits
right at the t=0 (base-distribution) boundary, not t=1 (data) and not spread through the
bulk.** This explains why uniform step refinement (same spacing at every t) converges so
slowly and inconsistently between solvers: both midpoint and rk4 assume local smoothness of
the integrand within each step, and a boundary-layer singularity concentrated in roughly the
first few percent of the trajectory is exactly what a *uniform* grid under-resolves worst,
regardless of how many total steps are used elsewhere. This also explains two things from
earlier in the P5 campaign that were otherwise puzzling: (a) doubling `nsteps` uniformly
(64->128->256->512) shrank but never closed the midpoint/rk4 gap, costing ~2h GPU per
doubling with "no sign of plateauing" (RESTART trap/NEXT ACTION) -- uniform refinement
converges slowly against a boundary singularity; (b) rk4 itself was found not converged at
nsteps=512 vs. 256 (~40% change, far short of the ~16x a true 4th-order method should show)
-- a fixed-order method's formal convergence rate assumes smoothness that doesn't hold near
t=0.

**Not yet decided, left for the user**: how to fix it. Candidates, not implemented or chosen
here:
  (a) a graded/non-uniform time grid (e.g. a reparameterization concentrating many more,
      smaller steps near t=0, coarser steps in the smooth bulk) -- directly targets the
      boundary layer without the cost of uniformly refining the whole trajectory;
  (b) an adaptive-step solver (e.g. embedded RK with local error control) instead of the
      current fixed-step `midpoint`/`rk4` -- would concentrate effort near t=0 automatically;
  (c) semi-analytic handling of the first few steps near t=0, if the base distribution's own
      score/density has a closed form there (would need checking what the flow-matching
      training objective's conditional path actually is near t=0 -- not yet investigated);
  (d) accept the known cost and pick a nsteps/solver combination empirically validated to be
      "good enough" for the fit's actual precision needs, rather than chasing full numerical
      convergence.
Either way, (a)/(b) seem more promising than further brute-force uniform step increases,
which is exactly what the standing NEXT ACTION already said before this entry pinned down
*why*.

## P5 follow-up: "just ignore the t=0 spike" is ruled out, not just untested (2026-11-18)

Direct test of the obvious cheap fix for the t=0 divergence spike found above: does that
region's own contribution to the accumulated integral depend on theta, or is it a
theta-independent offset safe to truncate/skip? Measured, for the same 2 jets (N=5, N=25)
at `primary_test` theta0=(4.5, 0.45, 0.65) and the same 4 established nearby-theta pairs
(+-2%-of-range single-axis steps in b/T_ref/tau_init, plus one diagonal step -- identical to
the original formal P5 check), midpoint's accumulated divergence split into "spike" (first
5% of steps, nsteps=64) vs. "bulk" (the rest), and compared each region's own pairwise Delta
against the real, full-trajectory pairwise Delta a fit actually uses.

**Result: no, this is not safe -- ruled out, not just unconfirmed.**

For the simplest jet (N=5), the spike region's own Delta is a small fraction of the full
Delta (2-12%) -- consistent with "safe to ignore" for this one jet. But for the N=25 jet,
the picture inverts completely:

```
                      Delta_spike   Delta_bulk   Delta_full
pair0 (b)               -0.0116      +0.0107      -0.0009
pair1 (T_ref)           +0.0992      -0.0630      +0.0363
pair2 (tau_init)        +0.0205      -0.0188      +0.0017
pair3 (diagonal)        +0.1069      -0.0706      +0.0363
```

The spike region's own theta-sensitivity is **larger in magnitude than the whole
trajectory's final answer**, with the **opposite sign** from the bulk region's own
sensitivity. The small `Delta_full` values are not evidence that neither region's
theta-dependence matters -- they are a near-cancellation between two comparably-sized,
oppositely-signed terms. Truncating the spike region would not produce a cleaner version of
the real signal; it would produce a different number, in several cases with the wrong sign.

This also sharpens the original P5 follow-up's "noise floor" framing (2026-11-05): the
pairwise Delta a fit uses was already known to be tiny and sign-flipping across jets at
nsteps=64; this pins down that a meaningful part of that instability is specifically the
badly-resolved spike region's contribution fighting the bulk's, not generic diffuse
discretization error.

**Conclusion: truncating or approximating the t=0 region is not a viable fix in general --
whether it would be safe is jet-dependent, and it already fails for a jet with only 25
constituents** (real jets range up to ~100, so this is a conservative test case, not an edge
case). This rules out candidate (c) from the entry above ("accept/approximate the first few
steps") as a general solution. The remaining candidates -- a graded/non-uniform time grid
concentrating steps near t=0, or an adaptive-step solver with local error control -- actually
resolve the region rather than discard it, and are what this result points toward. Neither
implemented yet; still the open decision.

**Caveat on this result, flagged directly by a follow-up question ("you are flagging an
issue based on TWO jets?"):** the n=2 test above is valid evidence for the narrow claim it
was designed for -- a single counterexample is enough to disprove "truncating the spike is
*always* safe," the same way one black swan disproves "all swans are white." It says
nothing about how common or severe the effect is across the dataset, or (the question that
actually matters for inference) whether it survives summing over the ensemble of jets a fit
would actually use rather than being jet-to-jet noise that cancels under that sum. See the
next entry for the properly-powered version of that question.

## P5 follow-up: the t=0 spike is an ensemble-level bias, not jet-to-jet noise -- confirmed at n=1000, not n=2 (2026-11-19)

**CORRECTION (2026-11-21), see the entry at the end of this file:** the quantity measured
here is the *accumulated divergence integral only* ("delta"/"acc" in the script), not the
full log-likelihood (`log p_base(z1) + delta + log_jacobian [+ log_multiplicity]`) that
`FlowSurrogate.log_prob`/`numerics.py` actually compute. The raw observation that the
divergence *integrand* spikes at t=0 (shown directly in per-step trajectory data) still
stands, but conclusions drawn here about the *ensemble log-likelihood*'s sensitivity used an
incomplete proxy -- see the correction entry for what changes and what doesn't.

Direct test of the question above: does the spike region's contribution to the accumulated
divergence have a non-zero *mean* across jets (a real bias that survives summing over an
inference ensemble), or is it zero-mean jet-to-jet noise that cancels under that sum? Ran
the same spike (first 5% of steps, nsteps=64) / bulk / full midpoint-divergence
decomposition, at `primary_test` theta0 and the same 4 established nearby-theta pairs, on
**all 1000 jets** in `p5_obs.h5` (multiplicity 5-93) -- not 2 cherry-picked jets.

First attempt stalled: batching all 1000 jets into one pass (padding every jet, including
N=5 ones, to the batch's global max multiplicity of 93) blew the cost up by ~50x relative
to a 50-jet calibration run, because the model's per-jet attention cost scales as
`batch x width^2` -- 90 minutes with zero output before being killed. Fixed by processing
jets in multiplicity-sorted chunks of 50 (each chunk padded only to its own max width, not
the global one); full run completed in ~77 minutes.

**Result: the spike's mean contribution across jets is non-zero at very high significance,
for all 4 theta directions, and does not shrink as the ensemble grows -- it is a real,
non-cancelling bias, not noise.**

| pair | mean Delta_spike/jet +- SE (significance) | mean Delta_full/jet +- SE (significance) | ensemble ratio Delta_spike_sum/Delta_full_sum |
|---|---|---|---|
| b | -0.00607 +- 0.00063 (9.6 sigma) | +0.00667 +- 0.00097 (6.9 sigma) | **-0.91** |
| T_ref | +0.02595 +- 0.00223 (11.7 sigma) | +0.03867 +- 0.00360 (10.7 sigma) | **+0.67** |
| tau_init | +0.00807 +- 0.00042 (19.0 sigma) | +0.00490 +- 0.00110 (4.5 sigma) | **+1.65** |
| diagonal | +0.02726 +- 0.00206 (13.2 sigma) | +0.04776 +- 0.00384 (12.4 sigma) | **+0.57** |

The cumulative ensemble ratio vs. ensemble size N (mean over 20 random orderings) converges
to these same values by N~200-500 and stays flat out to N=1000 (e.g. tau_init: 1.79 at
N=200, 1.63 at N=500, 1.65 at N=1000) -- the signature of a real asymptotic bias, not
noise that keeps shrinking.

This is a materially stronger and worse finding than the n=2 per-jet test above. It isn't a
rare, jet-dependent edge case: the spike region's own theta-sensitivity is a *dominant
fraction* of the real ensemble-level signal in every direction tested, with the wrong sign
in two of four. For `b`, the spike (-0.0061/jet) is comparable in magnitude to and opposite
in sign from the real bulk signal (+0.0127/jet) -- they nearly cancel, so the measured
ensemble sensitivity to `b` (+0.0067/jet) is roughly half of what the bulk physics alone
would give. For `tau_init`, the spike (+0.0081/jet) *exceeds* the bulk (-0.0032/jet, opposite
sign) and sets the sign of the net result -- the measured sensitivity to `tau_init` is
dominated by a numerical artifact, not physics.

**Conclusion: this is not a per-jet curiosity to be worked around case-by-case -- it is a
systematic, dataset-wide corruption of the ensemble log-likelihood gradient in theta.**
Raises the priority of actually fixing the t=0 singularity (graded time grid or adaptive
solver, candidates (a)/(b) above) rather than treating P5 as a secondary concern: as
currently integrated, the surrogate's likelihood is not just "a little off near t=0," it is
giving a wrong-sign or roughly-2x-wrong answer for the parameter directions that matter most
for inference, at the ensemble scale a real fit would use.

Script: `scripts/p5_ensemble_spike_sensitivity.py` (chunked by multiplicity; batching the
full global-max-width batch in one pass is NOT safe -- see the stall above).

## Surrogate fidelity vs. centrality: no real trend, T_ref confounding (2026-11-19)

Follow-up question: how does surrogate fidelity vary with centrality (impact parameter
`b`)? Read directly from the production checkpoint's existing 128-point P4 gate
(`v3_set_d256_L6_400k_s0`, no new sampling), recomputing `fail` at the Bonferroni-corrected
threshold (`p_family < 0.05/128` AND >=1 relevant observable, as used throughout this
analysis).

**Marginally, it looks real:** central events (b in [0, 2.75] fm) fail 18.8% of the time vs.
6.2% for the most peripheral quartile (b in [8.21, 10.94] fm); mean max relevance drops
monotonically from 2.29 to 1.69 across the four b quartiles.

**But it's T_ref confounding, not centrality.** `b` and `T_ref` are sampled independently by
design (`corr(b, T_ref) = -0.000` over the 128 points), so this was checked by restricting to
the safe mid-range T_ref band [0.25, 0.75] (n=80, excludes the already-known T_ref-edge
instability from mitigation (3)'s root-cause analysis). Within that band, fail rate is flat
at ~3.7% across all three b terciles, and the partial correlation of `b` with fail/relevance
controlling for T_ref is ~0 (+0.036 / -0.129), vs. -0.40 for T_ref-edge-distance alone. This
particular 128-point design happens to have more extreme-T_ref points at low `b` by chance,
which is what produced the marginal trend.

**Conclusion: there is no separate centrality-driven fidelity problem.** All of the real
fidelity degradation already identified lives at the T_ref extremes (near 0.1 and 0.9 GeV);
fidelity vs. centrality alone is flat once that confound is removed. Full writeup, figure and
reproducible script: `omni-sbi-nle-jet-dev-reports/reports/m3-surrogate-fidelity/2026-11-19T1007-0800_jewel-v3_fidelity-vs-centrality/`.

## P5 candidate (a) validated at n=1000, confirmed across grid_power 3.0 and 4.0: production grid under-resolves T_ref/tau_init/diagonal by 1.7-2.6x, and its "b" signal is a numerical artifact (2026-11-20/21)

**CORRECTION (2026-11-21), see the entry at the end of this file: the "b's signal is a
numerical artifact" conclusion in this entry is WRONG**, and is retracted. It was based on
the same incomplete proxy (divergence-integral-only "delta", missing `base_logp(z1)` and
`log_multiplicity`'s theta-dependence) as the entry above. The corrected, complete
log-likelihood (computed via the project's own existing `numerics.py` tool) shows `b`'s
signal is real, does not cancel, and *also* grows ~3x under the graded grid, same as the
other three directions -- no anomaly. The implementation (`LikelihoodOptions.grid_power`
etc.) and the underlying t=0 divergence-integrand spike observation are unaffected by this
correction; only the per-jet-mean "delta-only" ensemble numbers in this entry (and their
interpretation for `b`) are superseded.

Candidate (b) (adaptive solver) was tested and ruled out this session: correct but
~100-150x too expensive at the tolerance needed (scalar `rtol=atol=1e-5`), and unsafe at a
cheaper tolerance (`1e-3` corrupted the *bulk* integral by 10-100x, e.g. pair0_b's bulk went
from +0.001 to -0.35); splitting tolerance per-component (loose on positions, tight only on
the accumulator) made cost *worse*, not better, confirming the expense is intrinsic to
resolving the boundary layer, not wasted precision elsewhere. This left candidate (a) -- a
graded, non-uniform **fixed** time grid -- as the leading option, since it keeps total step
count (and cost) fixed and known in advance.

**Implementation.** New `LikelihoodOptions.grid_power` field (default 1.0 = today's exact
uniform grid, bit-identical fast path preserved): step edges `t_i = (i/nsteps)**grid_power`
for `i=0..nsteps`. `grid_power>1` concentrates many small steps near `t=0` and leaves
correspondingly larger steps near `t=1`, at the *same* `nsteps` (no change to the
`NFE_PER_STEP`/`max_nfe` budget check). New helper `_graded_edges(nsteps, grid_power)`;
`_fixed_step` generalized to use it when `grid_power != 1.0`; validated against adaptive
solvers (`grid_power != 1.0` with `solver in ADAPTIVE_SOLVERS` raises -- adaptive solvers
already grade their own steps) and against `grid_power < 1.0` (meaningless for this use).
New tests `tests/sbi/test_likelihood_graded_grid.py` (grid shape, backward compatibility,
validation errors, convergence on the analytic `LinearField` fixture under grading) --
8/8 pass; full `tests/sbi` suite: **274 passed, 0 failed, 7 skipped** (zero regression).

**Real-data validation**, `scripts/p5_graded_grid_validation.py` (modeled on
`p5_ensemble_spike_sensitivity.py`'s multiplicity-chunked batching; spike window now defined
by **t-value** `t<0.05`, not step index, since step index no longer tracks `t` proportionally
once the grid is graded). First 100 jets of `p5_obs.h5`, `primary_test` theta0, the same 4
established nearby-theta pairs, `nsteps=64` (today's production cost) throughout:

| pair | grid_power=1.0 full | grid_power=3.0 full (nsteps=64) | grid_power=3.0 full (nsteps=128) |
|---|---|---|---|
| b | +0.00412 +- 0.00301 | -0.00231 +- 0.00614 | -0.00311 +- 0.00621 |
| T_ref | +0.03576 +- 0.01135 | **+0.07977 +- 0.02137** | +0.08311 +- 0.02159 |
| tau_init | -0.00015 +- 0.00292 | +0.00147 +- 0.00478 | +0.00205 +- 0.00481 |
| diagonal | +0.03741 +- 0.01171 | **+0.07527 +- 0.02089** | +0.07838 +- 0.02105 |

**For T_ref and the diagonal direction -- the two pairs with a `full` signal well clear of
zero at n=100 -- grid_power=3.0 roughly doubles the inferred theta-sensitivity relative to
today's uniform grid, at the identical step budget**, and both the spike *and* the bulk
components grow (not just the spike shrinking relative to a fixed bulk). This raised an
immediate question: is grid_power=3 actually more accurate, or just trading spike
under-resolution for bulk under-resolution (its last step near t=1 is ~3x the uniform step
size)? **Checked directly**: doubling `nsteps` 64->128 at fixed `grid_power=3.0` moves T_ref's
`full` by +0.15 sigma and diagonal's by +0.15 sigma (bulk alone: +0.05 sigma and +0.03 sigma)
-- both comfortably converged already at `nsteps=64`, ruling out the under-resolved-bulk
concern. **Conclusion: today's production uniform 64-step grid is itself not converged for
T_ref and diagonal-direction theta-sensitivity -- it underestimates the true value by roughly
2x, not just mishandling the t=0 region specifically.** The graded grid fixes this at no
extra cost and is demonstrably converged at the production step budget.

**b and tau_init were inconclusive at n=100** (same statistical-power issue the n=2-vs-n=1000
lesson already taught). Rerun at the full **n=1000**, `grid_power=3.0`, `nsteps=64` (same
cost), multiplicity-sorted chunking (20x50), ~79 min on a dedicated interactive-QOS GPU node
with incremental per-chunk checkpointing (`scripts/p5_graded_grid_validation.py`'s
`.partial.json` sidecar, added this session after a multi-hour-script-with-no-partial-save
near-miss on a different script, see trap 26):

| pair | grid_power=1.0 full (production, n=1000) | grid_power=3.0 full (n=1000) |
|---|---|---|
| b | +0.00667 +- 0.00097 (**6.9 sigma**) | **-0.00051 +- 0.00247 (0.2 sigma)** |
| T_ref | +0.03867 +- 0.00360 (10.7 sigma) | +0.10101 +- 0.00906 (**11.2 sigma, ~2.6x**) |
| tau_init | +0.00490 +- 0.00110 (4.5 sigma) | +0.00818 +- 0.00195 (**4.2 sigma, ~1.7x**) |
| diagonal | +0.04776 +- 0.00384 (12.4 sigma) | +0.10517 +- 0.00897 (**11.7 sigma, ~2.2x**) |

T_ref, tau_init and diagonal confirm the n=100 pattern at full statistics: the graded grid
yields a 1.7-2.6x *larger*, still highly significant signal, for the same cost.

**`b` is a different and more important story.** At `grid_power=3.0`, both the spike
(-0.01613 +- 0.00222, 7.3 sigma) and bulk (+0.01562 +- 0.00083, 18.8 sigma) grew *larger* in
magnitude than their production-grid counterparts (spike -0.00607, bulk +0.01274) -- but now
they nearly perfectly cancel, leaving `full` consistent with zero. **The production uniform
grid's "b" signal (6.9 sigma, reported in the entry above as a real, non-cancelling
ensemble-level bias) was very likely itself a numerical artifact**: an incompletely-resolved
spike that didn't cancel its bulk counterpart correctly, leaving a spurious residual that
looked like real theta-sensitivity. This is not a fluke of `grid_power=3.0` specifically --
the n=100 nsteps 64->128 convergence check already showed `b`'s near-zero result stable under
refinement (-0.00231 vs -0.00311, well within 1 SE), i.e. the cancellation is the *converged*
answer, not an artifact of insufficient resolution in the other direction.

**Implication: conclusions about `b` (impact parameter) sensitivity/identifiability drawn
from the production numerics so far should be treated with caution** until this is
cross-checked (e.g. a second grid_power value, or the formal `numerics.py` gate) -- the
apparent signal for this one direction may not be real physics.

**Cross-check at `grid_power=4.0`, n=1000, same cost (2026-11-21): confirmed, not a
`grid_power=3.0` coincidence.** All four pairs agree between `grid_power=3.0` and `4.0` to
within 0.18-0.20 sigma -- effectively indistinguishable:

| pair | grid_power=3.0 full | grid_power=4.0 full | difference |
|---|---|---|---|
| b | -0.00051 +- 0.00247 | -0.00118 +- 0.00250 | 0.19 sigma |
| T_ref | +0.10101 +- 0.00906 | +0.10363 +- 0.00916 | 0.20 sigma |
| tau_init | +0.00818 +- 0.00195 | +0.00868 +- 0.00197 | 0.18 sigma |
| diagonal | +0.10517 +- 0.00897 | +0.10762 +- 0.00907 | 0.19 sigma |

`b`'s spike and bulk both continue growing slightly in magnitude from `grid_power=3.0` to
`4.0` (spike -0.01613->-0.01703, bulk +0.01562->+0.01585) while their difference stays pinned
near zero -- the cancellation is a genuine, converged, grid_power-independent feature, not a
fluke of one particular grading. **This closes the open question from the previous entry: the
production grid's "b" signal is a numerical artifact, confirmed independently of grading
choice.** T_ref/tau_init/diagonal's corrected (roughly doubled) values are likewise stable
across gradings, reinforcing they are converged and trustworthy.

**Not yet done** (deliberately out of scope for this step): the formal `numerics.py`
TEST-vs-REFERENCE Delta log L gate with the graded grid as TEST; updating `numerics.py`'s
`TEST`/`REFERENCE` dicts or `LikelihoodOptions`' production defaults. Per the standing rule,
production defaults change only after a fix is chosen and reviewed, not automatically from a
validation script.

GPU: NERSC Perlmutter interactive QOS (`-A alice_g -C gpu -q interactive`), used after the
first attempt hit a CUDA OOM from unrelated users' jobs on a shared/contended node. Data
computed directly from `p5_obs.h5`, not stored; raw per-pair JSON preserved at
`$JEST_NLE_ROOT/scratch/p5_graded_grid_validation_n1000_gp3_nsteps64.json` and
`..._n1000_gp4_nsteps64.json` (the two n=1000 results above), `..._nsteps64.json` (n=100,
grid_power 1.0 vs 3.0) and the nsteps=128 convergence check's live output file.

## CORRECTION: the two entries above measured an incomplete quantity -- "b's signal is a numerical artifact" is retracted; the formal numerics.py gate shows all 4 directions improve consistently (2026-11-21)

**Root cause.** `scripts/p5_ensemble_spike_sensitivity.py` and
`scripts/p5_graded_grid_validation.py` hand-roll the midpoint/rk4 step loop (needed for
step-level access to split spike/bulk/full) by calling `_rhs_factory` directly and tracking
only the accumulated divergence integral (`acc`, called "delta" in the scripts). They never
compute `base.log_prob(z1, mask)` -- the base-density term of the real log-likelihood,
`logp = log p_base(z1) + delta + log_jacobian [+ log_multiplicity]` (see `likelihood.py`'s
`log_prob()`, and `flow.py`'s `FlowSurrogate.log_prob()`, which adds a third term,
`log_multiplicity`, when the checkpoint has a multiplicity model attached -- this one does).
`z1` (and therefore `base_logp`) depends on theta, since it's the endpoint of integrating the
theta-conditioned velocity field -- it is not a constant that cancels in a pairwise
difference. **Two full theta-dependent terms were silently missing from every "delta-only"
ensemble number in the two entries above**, for both the original n=1000 ensemble-bias study
and this session's candidate (a) validation.

**How this was caught.** Before finalizing candidate (a), the project's own existing formal
gate (`numerics.py`'s `deltas()`/`evaluate()`, which correctly calls the real
`FlowSurrogate.log_prob`) was run as a cross-check -- the step this should have been checked
against from the start. `numerics.py run --test '{"solver":"midpoint","nsteps":64,
"divergence":"hutchinson","num_probes":8,"probe":"rademacher","probe_schedule":"per_step"[,
"grid_power":3.0]}'`, n=1000, 3 seeds, both `float32` and `float64` (ruling out precision as
the cause -- both agree closely):

| pair | grid_power=1.0 (full logp, summed) | grid_power=3.0 (full logp, summed) | ratio |
|---|---|---|---|
| b | -12.559 +- 0.017 | -38.162 +- 0.044 | **+3.04x** |
| T_ref | +42.873 +- 0.090 | +121.786 +- 0.229 | **+2.84x** |
| tau_init | +11.580 +- 0.034 | +21.648 +- 0.031 | **+1.87x** |
| diagonal | +41.336 +- 0.126 | +105.629 +- 0.158 | **+2.56x** |

(SEs here are seed-to-seed Hutchinson-probe scatter at n=1000 summed events, not jet-to-jet
scatter -- a different, much tighter quantity than the per-jet-mean +- SE used in the
delta-only entries above; the two aren't directly comparable in absolute scale, only in sign
and ratio.)

**Corrected conclusion: all four directions, including `b`, show a consistent, same-sign
~1.9-3.0x increase in full log-likelihood sensitivity under the graded grid, at the same
cost.** There is no cancellation, no sign flip, and no anomaly for `b` -- the delta-only
proxy's near-perfect spike/bulk cancellation for `b` (confirmed reproducible across
grid_power=3.0 and 4.0 in the entry above) was real, but it was a cancellation *within the
incomplete proxy itself*, not evidence that the production grid's true likelihood
sensitivity for `b` was spurious. **The "b identifiability conclusions should be treated
with caution" warning in both entries above is retracted.** If anything, the corrected
result makes candidate (a)'s case stronger and cleaner than before: a uniform ~2-3x
under-resolution across every tested direction, with the production grid's own existing
numerics gate as the judge, not a custom proxy.

**What is unaffected by this correction:** the raw, directly-observed per-step divergence
*integrand* trajectories (e.g. `trajectory_diag_sweep.json`, the t0-spike-explainer report)
showing a sharp boundary-layer spike at t=0 are real data, not a derived ensemble statistic,
and stand on their own. `LikelihoodOptions.grid_power`, `_graded_edges`, and `_fixed_step`'s
generalization are unaffected (they are solver internals, not analysis). The unit tests, the
nsteps 64->128 convergence check, and the grid_power 3.0-vs-4.0 agreement all remain valid
*as checks of the integrator's own self-consistency* -- they just weren't, on their own,
checks of the *real* log-likelihood, which is why this gap went unnoticed until the formal
gate was run.

**Lesson for future P5/numerics work:** validate any custom ensemble diagnostic against
`numerics.py`'s formal gate (or at minimum spot-check it against `FlowSurrogate.log_prob`'s
own `logp` output, not just its `delta_logp` diagnostic) *before* drawing conclusions,
especially before writing a "this signal is spurious" claim into FINDINGS. A custom script
that reimplements part of the solver for instrumentation purposes can silently omit real
theta-dependent terms that the production code path includes.

**Not yet done**: updating `numerics.py`'s `TEST`/`REFERENCE` dicts or
`LikelihoodOptions`' production defaults to `grid_power=3.0` (or similar) -- this correction
if anything strengthens the case, but the decision is still open. Data:
`$JEST_NLE_ROOT/scratch/numerics_gate_test{A,B}_gp{1,3}_f64.json`.

## P5 formal gate: midpoint-128/grid_power=4.0 PASSES against a high-fidelity reference -- numerics.py's TEST/REFERENCE updated (2026-11-22)

Closing the decision point left open by the correction above: built a high-fidelity
REFERENCE (`solver=midpoint, nsteps=128, divergence=hutchinson, num_probes=32,
grid_power=4.0`) and ran the project's own formal `numerics.py` `evaluate()` (the real
`FlowSurrogate.log_prob`, `LIMIT=0.5` log-L units, n=1000, 3 seeds, float64) against it for
four candidate TEST configurations, same 4 established pairs throughout:

| config | b | T_ref | tau_init | diagonal | verdict |
|---|---|---|---|---|---|
| A: today's production (`midpoint-64`, `grid_power=1.0`) | 27.3 | 84.7 | 11.4 | 69.7 | **FAIL** |
| B: graded grid, same cost (`midpoint-64`, `grid_power=3.0`) | 1.7 | 5.8 | 1.3 | 5.4 | FAIL (15-16x better than A) |
| C: graded grid, 2x steps (`midpoint-128`, `grid_power=3.0`) | 0.27 | 1.05 | 0.27 | 1.05 | FAIL (b, tau_init now pass) |
| D: graded grid, 2x steps, more aggressive (`midpoint-128`, `grid_power=4.0`) | **0.02** | **0.24** | **0.04** | **0.25** | **PASS, all 4 pairs** |

(all values are seed-averaged shift vs. REFERENCE, in the same summed-log-L units
`numerics.py` has always used; `probe_std` was small, 0.03-0.40, throughout -- Hutchinson
probe noise was never the limiting factor, confirming this is a genuine discretization gap,
not Monte Carlo variance.)

**A/B/C isolate the two knobs cleanly.** B vs A: regrading alone (same cost) cuts the error
~15-16x but doesn't fully converge. C vs B: doubling step count at the *same* grading closes
most of the remaining gap (another ~5-6x), confirming step count, not grid shape, was now
the dominant residual error source once the spike itself was addressed. D vs C: pushing
`grid_power` to 4.0 at the same 128 steps closes the rest -- a clean, comfortable pass with
headroom (0.02-0.36, well under the 0.5 limit) against a reference most of an order of
magnitude more expensive (32 probes vs. 8, used only to build a trustworthy anchor, not as
the recommended operating point).

**Recommendation, now implemented: `solver=midpoint, nsteps=128, grid_power=4.0`** (keep
`divergence=hutchinson, num_probes=8, probe_schedule=per_step` as today) -- **2x today's step
count, not the 100-150x the ruled-out adaptive solver (candidate (b)) would have cost.**
`src/omnilearned/sbi/toy/numerics.py`'s `TEST` dict updated accordingly; `REFERENCE` bumped to
`rk4-256, grid_power=4.0, 16 probes` so it remains meaningfully more trustworthy than the new
`TEST` (not re-validated against an even-higher reference -- today's A/B/C/D progression
already gives strong, independent confidence in the `grid_power=4.0`/`nsteps=128` combination
specifically). `tests/sbi/test_numerics_context.py` still passes (7/7) -- those tests use
their own explicit option dicts, not the module-level `TEST`/`REFERENCE` constants.

**Not changed, deliberately**: `LikelihoodOptions`'s own dataclass defaults
(`solver="rk4", nsteps=64, grid_power=1.0`). No project TOML config currently sets a
`[likelihood]` section (`fit.py`'s `LikelihoodOptions.from_dict(cfg.get("likelihood", {}))`
would silently fall through to these bare defaults otherwise), and the dataclass default
solver is `rk4`, not `midpoint` -- an rk4+grading combination has not been separately
validated here, so changing the fallback silently would be an uncontrolled, broader change
than what was actually tested. **Any real fit going forward should set
`[likelihood]` explicitly** in its own TOML config:
```
[likelihood]
solver = "midpoint"
nsteps = 128
grid_power = 4.0
divergence = "hutchinson"
num_probes = 8
probe = "rademacher"
probe_schedule = "per_step"
```
P6 can proceed once a real fit config does this (see `docs/RESTART_JEWEL_V3.md`'s NEXT
ACTION). Data: `$JEST_NLE_ROOT/scratch/numerics_gate_{reference_gp4_n128_p32,testC_gp3_n128,
testD_gp4_n128}_f64.json`.
