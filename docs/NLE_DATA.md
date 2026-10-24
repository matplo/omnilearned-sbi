# NLE data on Perlmutter: where the toy and JEWEL samples are

Root: `$JEST_NLE_ROOT = /global/cfs/cdirs/alice/ploskon/jest-nle` (CFS, shared `alice`
allocation). Nothing below is in git (rule: no data in the repository). Paths are relative to
the root unless absolute. State as of 2026-10-24.

```
jest-nle/
  data/            samples (this page)
  runs/            trainings, gates, fits, closure campaigns (one directory per study)
  src_snapshots/   read-only git-archive snapshots that jobs import (shell/nle/snapshot.sh)
  manifests/       first-day reproducibility manifests
```

## Toy `toy-quenching-v2` (config `configs/toy-quenching-v2.toml`)

Box b 0–11 fm, T_ref 0.25–0.55 GeV, τ_init 0.2–1.2 fm/c. Designs are scrambled Sobol
points (design seed 10921) with the 0.02 reservation margin around the held-out settings.

| directory | split | points | jets / point | jets | shards | size | used for |
|---|---|---|---|---|---|---|---|
| `data/v2_m1` | train | 1024 | 2000 | 2.048 M | 64 | 1.3 GB (with validation) | round-1 trainings (`runs/m2/*`) |
| `data/v2_m1` | validation | 128 | 1000 | 128 k | 8 | | early stopping, round 1 |
| `data/v2_m2` | train | 1024 (same design) | 6000 | 6.144 M | 64 | 3.6 GB | round-2 trainings (`runs/m2r2/*`) |
| `data/v2_m2` | validation | 128 | 1000 | 128 k | 8 | | early stopping, round 2 |
| `data/v2_m1_counts` | train (counts only) | 1024 | 20 000 | 20.48 M (θ, N) pairs | 64 | 785 MB | p(N\|θ) "multC64" (`runs/m3/mult/`) |
| `data/v2_m1_fidelity` | fidelity | 128 (the validation points) | 20 000 | 2.56 M | 32 | 1.5 GB | M3 gate (`gate run`), sampler convergence |
| `data/v2_m1_devscan` | development_scan | 46 | 20 000 | 920 k | 23 | 510 MB | equivalent shift (`gate shift`): development point, lines, tube (`runs/m1/tube/iso_tau0.95.json`) |
| `data/smoke_v2` | train / validation | 27 / 8 | 500 | 13.5 k / 4 k | 3 / 1 | 11 MB | smoke tests only |

Files per shard (`<split>_<nnnnn>`):
- `.h5`: schema `toy-shard-v1`;
- `.diag.h5`: hidden generator diagnostics, never read by training or fits;
- `.json`: manifest with seeds, per-point acceptance, hashes and host.

Each directory also has:
- `design_<split>.json`: the θ points;
- `index_<split>.json`: shard list with sha256, read by `shard_paths`.

The counts shards (`train_counts_<nnnnn>.h5`, schema `toy-counts-v1`) hold only
`n_constituents`, θ and the ids.

## JEWEL `jewel-2.6.0-cs-v1` (config `configs/jewel-2.6.0-cs-v1.toml`)

JEWEL 2.6.0-custom (custom-perf-simple):
- Pb–Pb at 5.02 TeV, pT-hat 50–400 GeV;
- recoils on, constituent subtraction (ΔR_max 0.5) applied at analysis level;
- EPPS21nlo_CT18Anlo_Pb208 (LHAPDF 904400);
- toy-v2 jet selection, jets rotated to φ = 0.

Training uses the Cartesian grid b = 0..11 fm (12 values), TI = 0.1..0.9 GeV (9) and
TAUI = 0.2..1.0 fm/c (9): 972 points. Validation uses 128 Sobol points in the same box.
The held-out settings lie on half steps of the grid (`[holdout]` in the config).

| directory | split | points | jets / point | jets | events | tasks | jet shards | event stores |
|---|---|---|---|---|---|---|---|---|
| `data/jewel_v1/train` | train | 972 | 1891–2943 (median 2167) | 2 199 271 | 28.30 M | 28 362 | 1.33 GB (+ 0.65 GB diag) | 601 GB |
| `data/jewel_v1/validation` | validation | 128 | 947–1557 | 146 016 | 1.90 M | 1 909 | 0.09 GB (+ 0.04 GB diag) | 40 GB |

One task is 1000 JEWEL events at one point. The files per task
(`<split>_p<point>_c<chunk>`) are:

| file | schema | content |
|---|---|---|
| `.h5` | `toy-shard-v1` | selected jets in the toy layout, so `toy.train`, `read_shards` and the gate read them unchanged |
| `.diag.h5` | | per jet: requested b, centrality, pT-hat, weight, unsubtracted jet pT and N, leftover pT in the jet, thermal pT and count, φ rotation |
| `.events.h5` | `jewel-events-v1` | raw HepMC records of every event (see below) |
| `.json` | | task manifest: cards, NJOB seed, counts, rejections, timing, hashes |

The `.events.h5` store keeps:
- per event (`events/`): id, number, centrality, scale = pT-hat, weight, cross section;
- flat particle arrays (`particles/`, float32 px, py, pz, E, pid, status) with per-event
  `offsets`. Statuses: 1 and 4 (final state, dummies dropped), 3 (thermal momenta for the
  subtraction), 23 (the two hard partons).

With the raw events, the subtraction and the jet selection can be redone without rerunning
JEWEL. Read them with `omnilearned.sbi.jewel.generate.read_events_store`.

Each split directory also has:
- `index_<split>.json`: schema `jewel-index-v1`, per shard the file, sha256, point and
  number of jets;
- `tasks.json`: the plan;
- `tasks_before_topup.json`: the plan before the top-up;
- `claims/` (task locks) and `logs/`.

`train/hepmc/` keeps two gzipped HepMC files for checking the subtraction:
`train_p00000_c000.hepmc.gz` and `train_p00500_c000.hepmc.gz`.

Probe outputs from before production, kept separately: `runs/jewel/probe1/`:
- `main`: custom-perf-simple at 8 settings;
- `custom_simple` and `jewel_cs`: the event-identity checks.

## Reading

```python
from omnilearned.sbi.toy.datasets import shard_paths, read_shards
paths = shard_paths("/global/cfs/cdirs/alice/ploskon/jest-nle/data/jewel_v1/train", "train")
d = read_shards(paths)   # x, mask, n_constituents, global, theta, event_id, point_id, global_names
```

The arrays in each jet shard are:
- `data/x [n, 256, 3]` float32: (η, φ, log pT/GeV) per constituent, zero-padded and
  left-packed;
- `data/mask [n, 256]`, `data/n_constituents [n]`;
- `global [n, 7]`: columns (b, T_ref, τ_init, jet pT, jet η, jet φ, n_constituents). For
  JEWEL, T_ref = TI and τ_init = TAUI;
- `event_id [n]` int64 (stream-tagged), `point_id [n]`.

`read_shards` cuts the padded axis to the largest real multiplicity.

Training on JEWEL:

```
python -m omnilearned.sbi.toy.train --config configs/jewel-2.6.0-cs-v1.toml \
  --data-dir $JEST_NLE_ROOT/data/jewel_v1/train --val-dir $JEST_NLE_ROOT/data/jewel_v1/validation
```

The 4-GPU recipe is `shell/nle/jewel_train_chunk.sh`.

## Caveats

- Event ids are unique within a dataset, not across datasets. For example, `v2_m1` and
  `v2_m2` share stream numbers but different jet counts per point (FINDINGS, M2).
- Changing `[toy]`/`[selection]` (toy) or `[jewel]`/`[subtraction]`/`[selection]` (JEWEL)
  defines a new backend id. New samples then go to a new directory; the directories above
  are never overwritten.
- No JEWEL fidelity or development-scan samples exist yet. The validation split, with about
  1000 jets per point, is too small for an M3-style gate.
