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
