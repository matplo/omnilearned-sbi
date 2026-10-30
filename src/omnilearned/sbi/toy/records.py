"""Event record files.

* Training/validation sets (simulation only): features, mask, multiplicity,
  event ids and the generating parameter labels (plus four-vectors and
  FastJet membership for v1).
* Observation files: framework ``omnilearned.sbi.observations`` format with
  features, mask and event ids only.
* Diagnostic stores: hidden generator history (positions, hard scale, path
  moments, component labels). Never read by fitting code.
* Truth manifests: JSON under a separate ``truth/`` directory, opened only by
  the report after fit outputs are sealed.
"""

import json
from pathlib import Path

import h5py
import numpy as np

from omnilearned.sbi.observations import file_sha256, write_observations

FEATURES = ("eta", "phi", "log_pt_over_GeV")


def write_training_set(path, parts, backend=None):
    """parts: list of (theta, records) for one split.

    ``x`` [N, P, 3] and ``mask`` [N, P]: P is 32 for toy-quenching-v1 (all
    real) and ``max_constituents`` (256) for v2, with left-packed real
    constituents and zero padding. Four-vectors and FastJet membership are
    written when the records carry them (v1).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cat = lambda k: np.concatenate([r[k] for _, r in parts])  # noqa: E731
    x = cat("x")
    mask = np.concatenate([r["mask"] if "mask" in r else np.ones(r["x"].shape[:2], bool)
                           for _, r in parts])
    if np.any(x[~mask] != 0):
        raise ValueError("padded entries must be zero")
    theta = np.concatenate([np.repeat(np.asarray(t)[None], r["x"].shape[0], 0)
                            for t, r in parts])
    with h5py.File(path, "w") as f:
        chunks = (min(4096, x.shape[0]),) + x.shape[1:] if x.shape[0] else None
        f.create_dataset("x", data=x, compression="gzip", chunks=chunks)
        f.create_dataset("mask", data=mask, compression="gzip",
                         chunks=None if chunks is None else chunks[:2])
        f["n_constituents"] = mask.sum(1).astype(np.int32)
        f["event_id"] = cat("event_id")
        f["theta"] = theta
        f["setting_index"] = np.concatenate(
            [np.full(r["x"].shape[0], i) for i, (_, r) in enumerate(parts)])
        if all("p4" in r for _, r in parts):
            f["p4"] = cat("p4").astype(np.float32)
            f["membership"] = cat("membership").astype(np.int8)
        for k in ("jet_pt", "jet_eta", "jet_phi"):
            if all(k in r for _, r in parts):
                f[k] = cat(k)
        f.attrs["feature_names"] = list(FEATURES)
        f.attrs["parameter_names"] = ["b", "T_ref", "tau_init"]
        f.attrs["max_constituents"] = int(x.shape[1])
        if backend:
            f.attrs["backend"] = backend
    return file_sha256(path)


def read_training_set(path, trim=True):
    """Arrays of a training/validation file.

    With ``trim`` the padded axis is cut to the largest real multiplicity in
    the file (constituents are left-packed), which saves memory without
    changing any value.
    """
    with h5py.File(path, "r") as f:
        out = {k: f[k][...] for k in ("x", "mask", "event_id", "theta", "setting_index",
                                         "n_constituents", "jet_pt", "jet_eta", "jet_phi")
               if k in f}
    out["mask"] = out["mask"].astype(bool)
    out.setdefault("n_constituents", out["mask"].sum(1).astype(np.int32))
    if trim and out["mask"].size:
        w = max(1, int(out["n_constituents"].max()))
        out["x"], out["mask"] = out["x"][:, :w], out["mask"][:, :w]
    return out


def write_diagnostics(path, parts):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f["event_id"] = np.concatenate([r["event_id"] for _, r in parts])
        for k in parts[0][1]["diag"]:
            f[k] = np.concatenate([r["diag"][k] for _, r in parts])
        f.attrs["access"] = "diagnostic only; never an input to fitting"
    return file_sha256(path)


def write_observation_file(path, records, attrs=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = records.get("mask")
    if mask is None:
        mask = np.ones(records["x"].shape[:2], dtype=bool)
    x = np.where(mask[..., None], records["x"], 0.0).astype(np.float64)
    return write_observations(path, x, mask, records["event_id"], FEATURES, attrs=attrs,
                              group_id=records.get("group_id"))


def write_truth(path, doc):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)
    return file_sha256(path)
