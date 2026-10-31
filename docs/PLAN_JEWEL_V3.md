# JEWEL v3: fresh generation, OmniLearned-native format, NLE end to end

**Standing brief for branch `dev/nle-jewel-v3`.** Approved 2026-10-31.

Read `docs/RESTART_JEWEL_V3.md` first for live state and the exact next command; this file is
the *why* and the *shape*. Format detail: `docs/FORMAT_JEWEL_V3.md`. Log: `FINDINGS_JEWEL_V3.md`.
Shared ground rules and lessons: `PLAN_NLE_PERLMUTTER.md` sections 1-6 (its milestones M5-M8 are
superseded here). The toy track and the v2 track are both closed.

---

# Plan: `dev/nle-jewel-v3` — fresh JEWEL, OmniLearned-native format, NLE end to end

## Context

The v2 track built the machinery and produced two real results (WEXPO 5 mismatched to a 20 GeV
jet floor: ESS/n 0.0065 → 0.130 at WEXPO 3; training length saturates on JEWEL, unlike the toy).
Its surrogate is far from passing the gate, and the representation audit
(`reports/m3-surrogate-fidelity/2026-10-31T1231-0700_jewel-v2_what-is-trained/`) showed why:
constituent η is **absolute** (r = +0.999 with the jet axis, σ = 1.00, against σ = 0.16 for the
R = 0.4 substructure it should carry), log E is missing, a per-run standardisation is applied
that OmniLearned deliberately removes, and `jet_pt` is stored but never conditioned on.

v3 generates fresh — nothing reused from `jewel_v1`/`jewel_w1` — in OmniLearned's native
representation, **always writing raw event stores** so no future representation change needs a
regeneration.

## 1. Target kinematic range for inference

| | |
|---|---|
| collision | Pb–Pb, √s_NN = 5.02 TeV |
| medium response | **`KEEPRECOILS T`** — kept |
| subtraction | **Milhano & Zapp constituent subtraction** (EPJC 82 (2022) 1010), `WRITESCATCEN T`, ΔR_max 0.5, `leftovers=keep`, `thermal_pid=reference` |
| **jets** | anti-k_T **R = 0.4**, **p_T 20–500 GeV**, **\|η_jet\| < 2.0**, **inclusive** |
| constituents | \|η\| < 2.5, `max_constituents = 1024` |
| hard process | **p̂_T 10–600 GeV**, `WEIGHTED T`, **WEXPO 3** |
| inference target | **θ = (b, T_ref, τ_init)** from an ensemble of such jets |

p̂_T brackets the jet window both sides. WEXPO 3 gives log-flat p̂_T sampling (measured ESS/n
0.130 vs 0.0065 at WEXPO 5).

## 2. The θ grid

**512 scrambled-Sobol points** for train; 128 Sobol each for validation and fidelity; the four
reserved holdouts for test: (7.5, 0.35, 0.85), (4.5, 0.45, 0.65), (1.5, 0.65, 0.35),
(9.5, 0.25, 0.45).

| θ | JEWEL card | range |
|---|---|---|
| `b` | CENTRMIN = CENTRMAX = centrality(b) | 0 – 11 fm |
| `T_ref` | `TI` | 0.1 – 0.9 GeV |
| `tau_init` | `TAUI` | 0.2 – 1.0 fm/c |

Verified for the proposed design: reaches the box edges on every axis, max/mean
nearest-neighbour gap 2.0–2.5, nearest training point to any holdout **0.0327 of the box**
(rule: ≥ 0.02).

| split | points × jets/pt | jets | CPU |
|---|---|---|---|
| train | 512 × 20 000 | 10.2 M | ~12 node-h |
| fidelity | 128 × 20 000 | 2.6 M | ~3 node-h |
| validation | 128 × 5 000 | 0.6 M | ~1 node-h |
| test | 4 × 50 000 | 0.2 M | ~0.3 node-h |

(0.545 core-s/jet and 0.40 jets/event, both measured on `jewel_v2k3`.)

## 3. The storage format

### Tier 1 — `jets/` : OmniLearned-native HDF5

HDF5 because OmniLearned's `HEPDataset` indexes `f["data"]`, `f["pid"]`, `f["global"]`
directly. Column semantics are fixed by OmniLearned, confirmed from its `clip_inputs`
(`hypot(X[:,0],X[:,1]) < 0.8` ⇒ 0,1 relative; `X[:,3] = clip(X[:,3], a_min=X[:,2])` ⇒ 3 is
log E) and `pid_idx = 4`:

```
data      [n, P, 5] float32   0: Δη = η − η_jet          relative to the jet axis
                              1: Δφ = φ − φ_jet          wrapped to (−π, π]
                              2: log(pT/GeV)             OmniLearned's validity marker
                              3: log(E/GeV)              ≥ feature 2
                              4: pid (PDG code)          split off at pid_idx=4
pid       [n, 1]   float32    class label (single class) — OmniLearned's `y`
global    [n, 8]   float32    b, T_ref, tau_init, jet_pt, jet_eta, jet_phi, jet_mass, n_const
mask      [n, P]   bool       explicit (ours)
weight    [n]      float64    p̂_T^−WEXPO, per-point normalised
pthat     [n]      float64
event_id  [n]      int64      unique per jet — seeds the Hutchinson probes
group_id  [n]      int64      originating JEWEL event; inclusive jets share it
point_id  [n]      int32
```

**~1 M jets per file**, gzip, `P` = the per-file maximum. Measured: current shards hold 212 k
jets × 512 slots in 0.08 GiB (0.4 kB/jet; padding compresses ~15×), so a 1 M-jet 5-feature file
is **≈0.4 GiB** and the train split is ~10 files. An explicit `mask` is carried because
OmniLearned's `data[:,:,2] != 0` rule would reject a genuine p_T = 1 GeV particle.

### Constituent capacity

Measured **per jet** on `jewel_v2k3` (48 589 jets). Jet p_T drives it, not centrality:

| selection | mean | p99.9 | max |
|---|---|---|---|
| all | 26.5 | 108 | 148 |
| b 0–2 fm | 26.3 | 109 | 131 |
| jet p_T 200–500 | 43.8 | 140 | 148 |
| worst: b<3, T_ref>0.7, p_T>200 | 53.6 | 131 | 131 |

Extreme-value projection: ~217 over 10⁷ jets, ~248 over 10⁸. `max_constituents = 1024` is ~7×
the observed maximum, so the cut cannot bind; stored width is the per-file maximum so it costs
nothing. **`too_many_constituents` must be 0 in every manifest, asserted at index time** — a
non-zero count is a hard error.

### Tier 2 — `events/` : raw HepMC records, always written

`jewel-events-v1`: `particles/{px,py,pz,e,pid,status}` + `offsets` + `events/{number,event_id,
centrality,scale,weight,xsec_pb}`; status 1 final state, 4 beam, 23 hard partons, **3 thermal
momenta** (what makes subtraction re-runnable). ≈24 kB/event ⇒ **≈700 GB** for this production
(CFS has 44 PB free).

### Sidecar

`dataset.yaml` at each dataset root, mirrored into the index JSON that code reads: schema
version, backend id, the full `[jewel]`/`[subtraction]`/`[selection]` blocks, column names and
units for `data` and `global`, the weighting convention, file list with jet counts and sha256,
and the θ design with its seed. A file is self-describing without the generating code.

## 4. Execution — parallel tracks

**These tracks write code; they do not run measurements.** Every measurement phase (P2–P8) is
strictly sequenced in §5 and §6 — nothing is measured before the thing it measures exists. What
parallelises is *development*: B and C are pure code with synthetic unit tests (e.g. "unit
weights must reproduce the unweighted result exactly", the check already used for
`binned_chi2`), so they need no JEWEL data and are ready by the time the pipeline reaches them.

Tracks B–D are **spawned agents** (`subagent_type: general-purpose`, `effort: low`,
`isolation: worktree` so they cannot collide on the git index), each owning a disjoint file
set. The main session keeps Track A and **all job submission**.

| track | owner | writes | ready for | tested against |
|---|---|---|---|---|
| **A** generation + all Slurm | main session | `jewel/*.py`, `configs/`, `shell/nle/` | P1, P2 | real data as it lands |
| **B** likelihood code | agent, low effort, worktree | `sbi/likelihood.py`, `sbi/flow.py`, `sbi/multiplicity.py`, `tests/sbi/test_likelihood_*` | **P6**, after P5 | analytic densities, synthetic clouds |
| **C** weighted-estimator code | agent, low effort, worktree | `sbi/fidelity.py`, `toy/gate.py`, `toy/ptbins.py`, `tests/sbi/test_gate_*` | **P4**, after P3 | synthetic weights; unit weights reproduce unweighted |
| **D** docs + HTML | agent, low effort | `docs/*.md`, reports repo | continuous | — |

Merge order into `dev/nle-jewel-v3`: A, then C, then B (B touches `flow.py`, which A also edits
for the shard layout — A merges first and B rebases).

### Compute rules

- **CPU generation:** `sbatch -A alice -C cpu -q regular -N 1 --array=0-7`, 64 workers/node
  (inclusive clustering packs ~77 k candidates/task; 128 workers risks OOM at a 1024 cap).
- **Interactive overflow, always:** the batch queue is routinely congested while interactive
  starts in ~20 s. Run `salloc -A alice -C cpu -q interactive -t 04:00:00 -N 1` against the
  *same task directory* — claims are exclusive so it cannot duplicate work. **Limit: 2
  interactive nodes total**, shared between CPU and GPU.
- **Cancel an idle wave.** If the interactive farms finish a wave while its batch array is still
  queued, `scancel` the array so the driver reaches `topup`/`merge` immediately. This happened
  three times in v2 and each time cost ~30 min.
- **GPU:** whole nodes, `--gpus 4`, 4 workers. Submit through a wrapper that `exec`s the
  **snapshot's** copy of the script, never a worktree path.
- **Walltime:** request what the job needs (400k training measured at 4 h 11 m → ask 5 h, not
  10 h; a 10 h request sat unscheduled for hours).

## 5. The chain, with commands

Each phase ends with tests, a FINDINGS entry, a journal entry and a commit.

**P0 — branch and scaffolding** (15 min, main).
`git branch dev/nle-jewel-v3 dev/nle-jewel; git worktree add ...-v3`; write
`src/omnilearned/_version.py` (gitignored, absent in a fresh worktree — without it the whole
suite fails collection); create `docs/{PLAN,FORMAT,RESTART}_JEWEL_V3.md` and
`FINDINGS_JEWEL_V3.md`; spawn tracks B, C.

**P1 — config, representation, format** (2–3 h, Track A).
`configs/jewel-2.6.0-cs-v3.toml` (backend `jewel-2.6.0-cs-v3`, `max_constituents = 1024`,
`train_points = 512`, window as §1). `jewel/jets.py`: emit **Δη** from `_axis()`, add **log E**,
keep **pid**. `jewel/generate.py`: v3 layout, **event stores by default**, assert zero
`too_many_constituents` at `index`, `merge` targets ~1 M jets/file. New
`jewel/dataset_spec.py` writes/validates `dataset.yaml`.
*Gate:* format tests pass **and** a generated file loads through
`omnilearned.dataloader.HEPDataset(use_cond=True, use_pid=True)`.

**P2 — generation** (~4–6 h wall, Track A).
Snapshot, then one driver per split (`shell/nle/jewel_v3_drive.sh`, the v2 driver with
`WEXPO`/`DATASET` parameters), 8 nodes × 64 workers, plus both interactive nodes on `train` and
`fidelity`. Test split planned by **jet target**, not chunk count. Then `index` + `merge`.
*Gate:* zero truncation; reprocessing one task's event store reproduces its shard exactly.

**P3 — training** (~4 h, Track A).
`shell/nle/jewel_v3_train.sh`, 1 node × 4 GPUs, 4 runs: 400k × seeds 0–2 + a 200k control
(JEWEL saturates near 200–300k, so the control is the measurement, not a formality). Weighted
loss; `jet_pt` as known context (Track B's layout change); **no** `FeatureTransform`
standardisation. `p(N | pT_J, θ)` fitted **with weights**.
*Gate:* ESS logged; seed spread measured; val loss compared only within this set.

**P4 — fidelity** (~3 h, Track A runs it with Track C's estimators). **Requires P3: there is
no fidelity to measure before a surrogate is trained.**
Gate at 128 fidelity points sharded over 4 GPUs with `--redraw`, then `ptbins` over the 20–500
window, both using Track C's **weighted** estimators, then repeated at the four withheld θ.
*Gate:* unit weights reproduce the unweighted numbers exactly.

**P5 — NLE numerics** (~1 h, 4 GPUs). Rule 6: no fits before this passes.
`toy/numerics.py` unchanged (checkpoint + observations + θ pairs only). Δlog L convergence in
solver steps and Hutchinson probes at the analysis N against an rk4 reference, ≥ 12 probe
seeds; single-seed shift, seed-averaged shift and probe std each < 0.5; CUDA-vs-CPU check.
Record the chosen `[likelihood]` block in the fit config.

**P6 — likelihood and scan** (Track B writes it during P2–P4; it is *exercised* only here,
after P5 has passed and a trained surrogate exists).

```
log q(jet|θ) = log p(pT_J|θ) + log p(N|pT_J,θ)
             + log q(Δη,Δφ,log pT,log E | pT_J,N,θ) + log|J| − log Pr_q(S|θ)
```

Condition the flow on `pT_J` as known context (extend `multiplicity.layout_with_multiplicity`,
which already supports known columns). Add the explicit 1-D `p(pT_J|θ)` — where much of the
quenching signal lives, and what fixes the measured high-p_T starvation (effective counts per θ
point `[940,125,100,80,112,79,78,112]` across p_T bins). Recompute the acceptance table
`Pr_q(S|θ)` for the v3 selection (`closure.acceptance_part`, already `--config`-aware).
**Jacobian:** the jet axis is a function of the constituents, so relative coordinates are not a
free change of variables — drop the leading constituent's (Δη, Δφ) or another explicit
full-rank bijection, and carry `log|J|`.
Scan: reuse `sbi/scan.py` + `sbi/ridge.py` unchanged; run the **nested check** (finer Δs/h on
one fit) before the campaign.
*Gate:* the three-factor density normalises (numerical integration on a reduced problem); the
Jacobian has a unit test; the nested check passes.

**P7 — closure at withheld θ** (~1–2 days of chained GPU, Track A).
`toy/closure.py` reused: `prepare` already supports `source = "jewel"` and draws disjoint
ensembles (`jewel_ensembles`); sealing and truth-after-seal unchanged. **≥ 10 ensembles ×
N = 1000 jets at each of the four withheld θ**, plus the labelled surrogate self-closure
control. Chain 4 h interactive allocations with the `m5_drive.sh` pattern.
**Pass criteria, fixed now (rule 5):** joint levels uniform (KS p > 0.01); marginal and joint
68/95 % coverage inside Clopper–Pearson bounds; toy-minus-control mean pulls within 2 combined
standard errors per parameter.

**P8 — composite-likelihood correction** (~half of P7).
Inclusive jets give 1.97–2.41 jets per distinct event, so the summed log-likelihood is a
composite likelihood and intervals may be too narrow. Refit the same ensembles with one jet per
event (a `group_id` filter) as the discriminator. If it bites, the remedy — effective-N /
sandwich correction, or one jet per event at inference — **goes to the user**.

**P9 — final HTML report** (Track D): θ grid and coverage figures, generation settings, format
tables, the per-event code path with the real functions named, fidelity and closure results.

## 6. ETA

| | wall | notes |
|---|---|---|
| P0 + P1 | 3 h | B, C, D running concurrently |
| P2 | 4–6 h | 8 batch nodes + 2 interactive |
| P3 | 4 h | starts when `train` merges |
| P4 + P5 | 4 h | share the GPU node |
| **first fidelity verdict** | **~15 h** | |
| P7 | 1–2 days | 20+ sealed fits, chained |
| P8 + P9 | 0.5 day | |
| **closure result** | **~3 days** | |

## 7. Verification

1. **Representation:** Δη and Δφ have equal spread (≈0.16 at R = 0.4) and neither correlates
   with the jet axis; log E ≥ log p_T for every real constituent; padding exactly zero.
2. **No truncation:** `too_many_constituents` = 0 across every manifest, asserted.
3. **OmniLearned compatibility:** a file loads through `HEPDataset(use_cond=True, use_pid=True)`
   with the expected shapes — tested.
4. **Round trip:** reprocessing a task's event store reproduces its jet shard exactly.
5. **Numerics (P5):** Δlog L converges in steps and probes; CUDA vs CPU agree.
6. **Density (P6):** the three-factor likelihood normalises; the Jacobian has a unit test; the
   nested scan check passes before the campaign.
7. **Weighted estimators:** unit weights reproduce the unweighted gate/`ptbins` numbers exactly.
8. **Closure (P7):** sealed fits, pulls, Clopper–Pearson coverage, joint levels against the
   criteria above; self-closure control reported and labelled as a diagnostic.
9. Full `tests/sbi` green after each phase; FINDINGS + journal per phase; reports repo synced.

## 8. Handoff

Built in P0 and kept current: `docs/RESTART_JEWEL_V3.md` (live state block, exact next command,
traps), `docs/PLAN_JEWEL_V3.md`, `docs/FORMAT_JEWEL_V3.md`, `FINDINGS_JEWEL_V3.md`. Every
long step is idempotent and resumable; jobs run from an immutable snapshot.

**Traps carried forward — all silent, all cost a wrong number rather than an error:** a
worktree script path handed to `sbatch` (it can change while queued — pass the snapshot's
copy); a split planned by chunk count instead of a jet target; run directories named by
hardcoded step counts; `ptbins` binning on the toy's 60–200 window; `closure prepare`
defaulting to the toy config; `gate` ignoring per-jet weights; `multiplicity_check` fitting
`p(N|θ)` unweighted; a new dataclass field inserted before existing positional ones. Standing
question for anything not yet exercised on JEWEL: *does this component know which backend and
which weighting it is working on?*
