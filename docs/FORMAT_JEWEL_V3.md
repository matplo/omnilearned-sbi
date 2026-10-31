# JEWEL v3 data format — normative specification

Backend id `jewel-2.6.0-cs-v3`. Two tiers: **`jets/`**, the training format, loadable directly
by OmniLearned; and **`events/`**, raw HepMC records so any future representation change needs
no regeneration. A `dataset.yaml` sidecar makes a dataset self-describing.

Rationale and the measurements behind the choices: `docs/PLAN_JEWEL_V3.md`.

---

## 1. Why HDF5 and not Parquet

OmniLearned's `HEPDataset` (`src/omnilearned/dataloader.py`) is h5py-based and indexes
`f["data"]`, `f["pid"]`, `f["global"]` by row. Writing HDF5 in its layout means a v3 file is
training input with no conversion layer. Parquet would buy columnar scans we do not need and
cost a converter on the critical path.

## 2. Tier 1 — `jets/` : the training format

One file per ~1 M jets, gzip-compressed.

```
data      [n, P, 5]  float32
mask      [n, P]     bool
pid       [n, 1]     float32
global    [n, 8]     float32
weight    [n]        float64
pthat     [n]        float64
event_id  [n]        int64
group_id  [n]        int64
point_id  [n]        int32
theta     [n, 3]     float64
```

### 2.1 `data` — the constituent point cloud

| idx | name | definition | units |
|---|---|---|---|
| 0 | `deta` | η_const − η_jet | — |
| 1 | `dphi` | φ_const − φ_jet, wrapped to (−π, π] | rad |
| 2 | `log_pt` | log(p_T / GeV) | — |
| 3 | `log_e` | log(E / GeV), guaranteed ≥ index 2 | — |
| 4 | `pid` | PDG code of the constituent | — |

Constituents are p_T-ordered and left-packed; padded slots are **exactly zero** in every column.

**This column order is fixed by OmniLearned, not chosen by us.** Evidence in its
`HEPDataset.__getitem__`:

```python
# clip_inputs: "Enforce particles to be inside R=0.8 and pT > 0.5 MeV"
mask_part = (torch.hypot(sample["X"][:, 0], sample["X"][:, 1]) < 0.8) & (sample["X"][:, 2] > -0.7)
sample["X"][:, 3] = np.clip(sample["X"][:, 3], a_min=sample["X"][:, 2], a_max=None)
...
sample["pid"] = sample["X"][:, self.pid_idx].int()      # pid_idx default 4
```

`hypot(X[:,0], X[:,1])` compared with a cone radius ⇒ indices 0 and 1 are **relative** to the
jet axis. `X[:,3]` clipped from below by `X[:,2]` ⇒ index 3 is **log E ≥ log p_T**. `pid_idx = 4`
⇒ index 4 is the particle id, which OmniLearned splits off into its own embedding.

It also matches the paper's minimal input set verbatim (arXiv:2510.24066 §II.A): *"we maintain
the difference in η and φ between the particle and jet axis as inputs but only add two
additional features, consisting of the logarithm of the transverse momentum log p_T and energy
log E"* — followed by the statement that the standardisation step is removed because this
representation is already centred with O(1) variance. **We therefore apply no feature
standardisation.**

### 2.2 `mask` — why it is carried explicitly

OmniLearned infers validity from `data[:,:,2] != 0`, i.e. from log p_T. That would reject a
genuine p_T = 1 GeV constituent. Our writer guarantees padding is exactly zero so the
OmniLearned rule works in practice, but **our own readers use `mask`** and the dataset carries
it.

### 2.3 `global` — jet-level block

| idx | name | note |
|---|---|---|
| 0–2 | `b`, `T_ref`, `tau_init` | θ — **inferred**, never supplied at inference |
| 3 | `jet_pt` | **known context** for the surrogate |
| 4 | `jet_eta` | removed by the Δη coordinate; carried for diagnostics |
| 5 | `jet_phi` | reaction-plane information only |
| 6 | `jet_mass` | |
| 7 | `n_constituents` | known context |

θ sits in `global` because training needs it; at inference it is scanned, not read. The
likelihood conditions on `jet_pt` and `n_constituents` only.

### 2.4 Identifiers

- **`event_id`** is **unique per jet** and seeds the Hutchinson probes
  (`likelihood.stable_probes`). Two rows sharing it get identical probe vectors, which
  correlates their divergence errors and inflates the variance of the summed log-likelihood.
- **`group_id`** is the originating JEWEL event. With inclusive jets several jets share one, so
  the within-event correlation stays recoverable — this is what the composite-likelihood
  coverage check and the one-jet-per-event refit use.

### 2.5 Weights

`weight` is the JEWEL event weight p̂_T^−WEXPO, renormalised per θ point so Σw = n_jets at that
point. WEXPO = 3 (log-flat p̂_T sampling; measured ESS/n 0.130 against 0.0065 at WEXPO 5).
**Every estimator over these jets must use the weights** — the unweighted sample is the
generation law, not the physical one.

### 2.6 Shapes and sizes

`P` is the **maximum constituent count in that file**, not a global constant. Padding gzips to
near nothing (measured: 212 k jets × 512 slots in 0.08 GiB, 0.4 kB/jet, ≈15× compression), so a
1 M-jet 5-feature file is ≈0.4 GiB.

`selection.max_constituents = 1024` is a *selection* parameter and is deliberately far above
the data (measured per-jet max 148; extreme-value projection ~248 over 10⁸ jets) so the cut
**cannot bind**. `too_many_constituents` must be 0 in every task manifest and is **asserted at
index time** — a non-zero count is a hard error, because truncation would bite preferentially
at high jet p_T where the quenching signal is strongest.

## 3. Tier 2 — `events/` : raw HepMC records

Schema `jewel-events-v1`, ragged, **always written** (v2's `--no-events` cost the ability to
re-derive and is not repeated):

```
particles/{px,py,pz,e} float32   pid int32   status int8
offsets  int64                    events/{number,event_id,centrality,scale,weight,xsec_pb}
```

`status` 1 = final state, 4 = beam, 23 = hard partons, **3 = thermal momenta**. Status 3 is what
makes the subtraction re-runnable; it is present where the medium is hot (849/event at b = 4,
T_ref = 0.9) and genuinely absent at T_ref = 0.1 or b = 11. With these, both the subtraction and
the jet finding can be redone — different R, window, inclusive/leading, or representation —
with no JEWEL CPU. ≈24 kB/event.

## 4. Sidecar — `dataset.yaml`

At each dataset root, with the same content mirrored into the index JSON that code reads:
schema version and backend id; the full `[jewel]`, `[subtraction]`, `[selection]` blocks;
column names, definitions and units for `data` and `global`; the weighting convention; the θ
design with its seed and the reserved settings; and the file list with per-file jet counts and
sha256. The intent is that a file can be understood without the generating code.

## 5. Reading a file

```python
# OmniLearned, directly
from omnilearned.dataloader import HEPDataset
ds = HEPDataset([path], file_indices=idx, use_cond=True, use_pid=True)

# ours
import h5py
with h5py.File(path) as f:
    x, m = f["data"][...], f["mask"][...]
    jet_pt = f["global"][:, 3]
    w = f["weight"][...]
```

## 6. Invariants (tested, not assumed)

1. `deta` and `dphi` have equal spread (≈0.16 for R = 0.4) and **neither correlates with the
   jet axis** — the v2 defect was `r = +0.999` between feature 0 and `jet_eta`.
2. `log_e >= log_pt` for every real constituent.
3. Padded slots are exactly zero in all five columns.
4. `event_id` is unique per row; `group_id` has repeats exactly when jets share an event.
5. Σ`weight` = n_jets within each `point_id`.
6. `too_many_constituents` = 0 in every manifest.
7. A file loads through `HEPDataset(use_cond=True, use_pid=True)` with the expected shapes.
8. Reprocessing a task's event store reproduces its jet shard exactly.
