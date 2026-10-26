# Handoff: pT-biased (weighted) training samples, branch `dev/nle-weighted`

This is a separate study, run by its own Claude Code session in this worktree
(`/global/cfs/cdirs/alice/ploskon/henv-omni-sbi/omnilearned-sbi-weighted`). The main study
(fidelity → closure → inference, branch `dev/nle-fidelity`, worktree `omnilearned-sbi-fidelity`)
continues unchanged in another session. The two studies exchange findings and nothing else.

Read first: `AGENTS.md`, `PLAN_NLE_PERLMUTTER.md` (rules, especially rule 8 on the JEWEL budget),
`FINDINGS_FIDELITY.md` sections "Training statistics vs jet pT" and "High-statistics shifts", and
`docs/NLE_DATA.md`. Keep the running log in `FINDINGS_WEIGHTED.md` (new, at the repository root).

## Why

- **The training spectrum falls steeply.** v2_m2 has 1024 θ points × 6000 jets. 46 % of the jets are in 60–70 GeV and 0.8 % in 170–200 GeV, i.e. 47 jets per θ point. The shape is almost independent of θ.
- **The surrogate is not conditioned on jet pT,** so its high-pT behaviour rests on about 1 % of the training jets.
- **The current tests can't see high pT.** The gate, the shift and the boundary decomposition average over pT, so a high-pT defect is diluted by its ≈ 6 % weight above 120 GeV.
- **The user's proposal:** oversample high pT at generation and train with event weights. This is the analogue of Pythia 8 "Reweighting of 2 → 2 processes" (`PhaseSpace:bias2Selection`, https://pythia8.web.cern.ch/manuals/pythia8318/PhaseSpaceCuts.html) and JEWEL `WEIGHTED`/`WEXPO`.

## The key point about weights

- **The weight depends on the hidden pThat** (toy: `P0`; JEWEL: HepMC `scale`), not on the jet the surrogate sees. So the weights must enter at **training**: loss = Σ w ℓ / Σ w, or a minibatch draw ∝ w.
- **Weighted training targets the true law.** The surrogate then learns p(x | N, θ), and all downstream code stays as it is, because it compares with unweighted direct samples:
  - the gate and the shift (`toy/gate.py`);
  - `toy/ptbins.py`;
  - the closure and the fits (`toy/closure.py`, `sbi/likelihood.py`).
- **What weighting buys and what it doesn't.** The expected objective is unchanged; the gain is lower Monte Carlo noise at high pT, not more emphasis on it. It helps if high-pT fidelity is **data-limited**.
- **If it is capacity-limited instead:** use phase 3. There the bias is a function a(pT) of the *observed* jet pT, reached by unweighting within jet-pT strata. The surrogate learns p·a/A(θ), and inference adds −log a(pT) + log A(θ). A(θ) must be smooth in θ, which is a new error source.

## Facts (file:line on 0b57663)

- **Toy hard pT:** `P0` is drawn by inverse transform from pT^-5 over 40–500 GeV (`toy/simulator_v2.py:151-153`, `configs/toy-quenching-v2.toml:18-20`).
  - Energy loss: `P_vis = P0·exp(-0.12·I3·Z)` (`:154-156`).
  - `P0` exists only in the diag output (`:196-200`).
  - Selection 60–200 GeV is at `:241-243`.
  - Fixed accepted count per point: `sample_toy` `:303-327`.
- **Config rule (AGENTS.md):** changing `[toy]` or `[selection]` defines a new backend id. The bias is a *sampling* option and leaves the physics law unchanged. Put it outside `[toy]`, e.g. a `[sampling]` section or a CLI option, so the backend id and `toy_config_sha256` stay those of v2. Record the bias settings in the shard index.
- **Shards:** writer `toy/datasets.py generate` (`:342-429`), schema `:9-14,55,139-181`; there is no weight field. Reader `read_shards` `:291-317`.
- **Loss:** masked flow matching, plain `.mean()` (`sbi/flow.py:81-101`).
  - Sampler: `minibatch_sampler` (`flow.py:356-382`), with length buckets (`toy/train.py:163`).
  - Early stopping uses the validation loss (`flow.py:412-416`).
- **p(N|θ):** `sbi/multiplicity.py:94-123`, and `toy/multiplicity_check.py` (`retrain` from counts shards, `attach`). The existing multC64 (`runs/m3/mult/set_d256_L6_200k_multC64.pt`) is the true law and can be attached unchanged.
- **JEWEL:** config `configs/jewel-2.6.0-cs-v1.toml:17-22` has `weighted = false`, PTMIN 50, PTMAX 400.
  - `sbi/jewel/cards.py:72-93` accepts `extra=` card lines (e.g. WEXPO).
  - `sbi/jewel/hepmc.py:36` reads the event weight and pThat into the per-jet diag (`generate.py:286-287`), not into the shard.

## Phases

**Phase 0: decision data, no new code.**
- The main session already runs the `toy.ptbins` baseline for set 400k, set r2 and PET r2 (`runs/m3/ptbins/`); ask it for the results, don't redo them.
- Here: run `toy.ptbins` for **set m3a**, trained on v2_m3a, i.e. 2× the jets at the same spectrum (`runs/m2r3/set_d256_L6_200k_m3a/checkpoint_mult.pt` or the multC64 one; check). If m3a is better at high pT, high pT is data-limited and the weighted arm is the right fix.
- Choose the bias exponent k from the effective sample size per jet-pT bin and θ, ESS = (Σw)²/Σw², from a quick CPU toy run. Aim for a roughly flat ESS per generated jet over 60–200 GeV, with weights capped. Expect k ≈ 2–4.

**Phase 1: weighted toy generation.**
- Optional bias k: draw `P0` ∝ pT^-(5−k), w = (P0/P_ref)^k.
- Normalize so that **Σw = n_jets at each θ point**.
- Write `weight` and `pthat` into the shard records. Optional fields: an absent weight reads as 1. `read_shards` returns `weight`.
- Dataset `data/v2_w1`: same design and θ points as v2_m2, 6000 accepted jets per point, CPU array job (`shell/nle/gen_toy_array.sbatch`).

**Phase 2: weighted training.**
- `flow.py`: per-jet weights in the training and validation losses, option `--weight-mode loss|sample` (sample: multinomial ∝ w in `minibatch_sampler`, compatible with length buckets).
- `train.py`: pass the weights through and record the ESS in run.json. p(N|θ): attach multC64, with a weighted NLL as a cross-check.
- Runs: set d256 L6 200k on v2_w1, seeds 0–2, against set r2/s1/s2 on v2_m2. The seed spread of the 11-point shift is ≈ 0.07σ, so 3 seeds are needed.
- Evaluation, unchanged code:
  - `toy.ptbins` (primary: the 120–200 GeV bins);
  - the gate at 128 points with `--redraw` (`shell/nle/fid_gate.sbatch`);
  - the high-statistics shift (`shell/nle/shift_queue.sh`, `data/v2_m3_shiftdirect`).

**Phase 3, only if needed:** the observed-pT bias arm, and/or conditioning the flow on jet pT.

**Phase 4: JEWEL.**
- `weighted = true` plus WEXPO through `cards.task_cards(extra=...)`, with the HepMC weight propagated into the shard `weight` field (`generate.py:241-345`).
- **No JEWEL production without a user-approved budget.** Propose a small probe first.

## Tests (`tests/sbi`; required for every new generator, format, driver, numerical method)

- A biased draw plus its weights reproduces the unbiased P0 and jet-pT moments.
- Old shards read back with unit weights.
- With all-ones weights the weighted loss equals the old loss exactly, and training is bit-identical for the same seed.
- A weighted fit on a 1D toy recovers the true density.

## Rules and resources

- **Environment:** always `henv -n omni-sbi -x python`, with `PYTHONPATH=$PWD/src` in this worktree, so the main checkout's editable install is not imported. Never create or modify environments.
- **Jobs:** run from `$JEST_NLE_ROOT/src_snapshots/<rev>` (`shell/nle/snapshot.sh`), with `JEST_NLE_ROOT=/global/cfs/cdirs/alice/ploskon/jest-nle`. Export it explicitly in submissions; it was lost after a node reboot.
- **Provenance:** record source revision, dirty status and device with every result.
- **Git:** don't commit data, checkpoints, run trees or figures; never force-push; don't switch branches in any checkout.
- **Where things go:** data in `$JEST_NLE_ROOT/data/v2_w*/`, runs in `$JEST_NLE_ROOT/runs/w*/`. Never write into `runs/m2r3`, `runs/m3` or other main-study trees, except reading.
- **Slurm:** account `alice_g`, `-C gpu`. **At most 2 interactive nodes per user, and the main study has priority.** Use regular/shared QOS batch jobs unless the user frees an interactive slot. Ask the user before large GPU/CPU spends.
- **Coordination:** `ListAgents` / `SendMessage` to the main session (worktree `omnilearned-sbi-fidelity`) for results that matter to both. Each side reads the other's FINDINGS file. Neither session edits the other's branch.
