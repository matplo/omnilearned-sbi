"""JEWEL v3 jet files: OmniLearned-native HDF5, plus the ``dataset.yaml`` sidecar.

Normative specification: ``docs/FORMAT_JEWEL_V3.md``. Summary of one file:

    data      [n, P, 5] float32   0 deta, 1 dphi, 2 log(pT/GeV), 3 log(E/GeV), 4 pid
    mask      [n, P]    bool      explicit; padded slots are exactly zero in `data`
    pid       [n, C]    float32   CLASS label (OmniLearned's `y`), not the particle id
    global    [n, 8]    float32   b, T_ref, tau_init, jet_pt, jet_eta, jet_phi, jet_mass, n_const
    weight    [n]       float64   per-jet sampling weight (1 for an unbiased split)
    pthat     [n]       float64
    event_id  [n]       int64     unique per jet -- seeds the Hutchinson probes
    group_id  [n]       int64     originating JEWEL event; inclusive jets share it
    point_id  [n]       int32
    theta     [n, 3]    float64

The column order of ``data`` is fixed by ``omnilearned.dataloader.HEPDataset``, not chosen
here: its ``clip_inputs`` does ``hypot(X[:,0], X[:,1]) < 0.8`` (so 0,1 are relative to the jet
axis) and ``X[:,3] = clip(X[:,3], a_min=X[:,2])`` (so 3 is log E >= log pT), and it splits the
particle id out at ``pid_idx = 4``. It is also the paper's minimal set (arXiv:2510.24066
II.A), which is why no feature standardisation is applied.

``pid`` is OmniLearned's class label and collides in name with the particle id in column 4 of
``data``; that is upstream's naming, kept so files load unmodified.
"""

import json
from pathlib import Path

import h5py
import numpy as np

SCHEMA = "jewel-jets-v3"
FEATURES = ("deta", "dphi", "log_pt", "log_e", "pid")
GLOBALS = ("b", "T_ref", "tau_init", "jet_pt", "jet_eta", "jet_phi", "jet_mass", "n_constituents")
FEATURE_DOC = {
    "deta": "eta_constituent - eta_jet",
    "dphi": "phi_constituent - phi_jet, wrapped to (-pi, pi]",
    "log_pt": "log(pT/GeV); OmniLearned's validity marker (data[:,:,2] != 0)",
    "log_e": "log(E/GeV), guaranteed >= log_pt",
    "pid": "PDG code of the constituent; OmniLearned splits this off at pid_idx=4",
}


def write_jets(path, *, x, mask, theta, jet_pt, jet_eta, jet_phi, jet_mass, event_id,
               group_id, point_id, weight=None, pthat=None, n_classes=1, attrs=None):
    """Write one v3 jet file. Returns its sha256."""
    from omnilearned.sbi.observations import file_sha256

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(x, np.float32)
    mask = np.asarray(mask, bool)
    n, P = mask.shape
    if x.shape != (n, P, len(FEATURES)):
        raise ValueError(f"data must be [n, P, {len(FEATURES)}], got {x.shape}")
    if np.any(x[~mask] != 0):
        raise ValueError("padded slots must be exactly zero in every feature")
    if np.any(x[..., 3][mask] < x[..., 2][mask] - 1e-6):
        raise ValueError("log E must be >= log pT for every real constituent")
    event_id = np.asarray(event_id, np.int64)
    if np.unique(event_id).size != event_id.size:
        raise ValueError("event_id must be unique per jet (it seeds the probes); "
                         "the originating event belongs in group_id")
    theta = np.asarray(theta, np.float64)
    nc = mask.sum(1).astype(np.float64)
    g = np.column_stack([theta, jet_pt, jet_eta, jet_phi, jet_mass, nc]).astype(np.float32)
    if g.shape[1] != len(GLOBALS):
        raise ValueError(f"global must have {len(GLOBALS)} columns, got {g.shape[1]}")
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=x, compression="gzip")
        f.create_dataset("mask", data=mask, compression="gzip")
        # OmniLearned's `y`: a class label. One class here, so a column of ones.
        f.create_dataset("pid", data=np.ones((n, n_classes), np.float32))
        ds = f.create_dataset("global", data=g)
        ds.attrs["global_names"] = list(GLOBALS)
        f.create_dataset("weight", data=np.ones(n) if weight is None else np.asarray(weight, np.float64))
        if pthat is not None:
            f.create_dataset("pthat", data=np.asarray(pthat, np.float64))
        f.create_dataset("event_id", data=event_id)
        f.create_dataset("group_id", data=np.asarray(group_id, np.int64))
        f.create_dataset("point_id", data=np.asarray(point_id, np.int32))
        f.create_dataset("theta", data=theta)
        f.attrs["schema"] = SCHEMA
        f.attrs["feature_names"] = list(FEATURES)
        f.attrs["n_jets"] = int(n)
        f.attrs["max_constituents_stored"] = int(P)
        for k, v in (attrs or {}).items():
            f.attrs[k] = v
    return file_sha256(path)


def read_jets(path):
    """Everything in a v3 jet file, as a dict of arrays."""
    with h5py.File(path, "r") as f:
        out = {k: f[k][...] for k in ("data", "mask", "global", "weight", "event_id",
                                      "group_id", "point_id", "theta") if k in f}
        if "pthat" in f:
            out["pthat"] = f["pthat"][...]
        out["feature_names"] = tuple(f.attrs["feature_names"])
        out["global_names"] = tuple(f["global"].attrs["global_names"])
    return out


def dataset_spec(cfg, split, files, design=None):
    """The self-describing sidecar for a dataset root.

    Written as ``dataset.yaml`` (readable, commented) and mirrored into the index JSON that
    code consumes, so a file can be understood without the generating code.
    """
    j, sub, sel = cfg["jewel"], cfg["subtraction"], cfg["selection"]
    return {
        "schema": SCHEMA,
        "backend": cfg["backend"],
        "split": split,
        "generator": {
            "sqrts_GeV": j["sqrts_GeV"], "ptmin_GeV": j["ptmin_GeV"], "ptmax_GeV": j["ptmax_GeV"],
            "weighted": j["weighted"], "keeprecoils": j["keeprecoils"],
            "writescatcen": j["writescatcen"], "pdfset": j["pdfset"],
            "mass_number": j["mass_number"], "proton_number": j["proton_number"],
        },
        "subtraction": dict(sub),
        "selection": dict(sel),
        "columns": {
            "data": [{"index": i, "name": n, "definition": FEATURE_DOC[n]}
                     for i, n in enumerate(FEATURES)],
            "global": [{"index": i, "name": n} for i, n in enumerate(GLOBALS)],
        },
        "weighting": {
            "convention": "JEWEL HepMC event weight PARI(10) = pthat^-WEXPO",
            "normalisation": "rescaled per point_id so sum(weight) = n_jets at that point",
            "note": "the UNWEIGHTED sample is the generation law; every estimator must weight",
        },
        "identifiers": {
            "event_id": "unique per jet; seeds the Hutchinson probes",
            "group_id": "originating JEWEL event; inclusive jets of one event share it",
        },
        "design": design or {},
        "files": files,
    }


def write_dataset_spec(root, spec):
    """Write ``dataset.yaml`` (and ``dataset.json``) at a dataset root."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "dataset.json").write_text(json.dumps(spec, indent=1, default=str))
    try:
        import yaml
        (root / "dataset.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    except ImportError:          # yaml is a convenience; the JSON is what code reads
        pass
    return root / "dataset.json"
