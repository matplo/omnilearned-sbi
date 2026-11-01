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


def shard_paths_v3(directory, split):
    """Merged v3 file paths of a split, from its index (hash-checked) or by name."""
    directory = Path(directory)
    idx = directory / f"index_{split}.json"
    if idx.exists():
        return [directory / f["file"] for f in json.loads(idx.read_text())["files"]]
    return sorted(directory.glob(f"{split}_*.h5"))


def read_v3_shards(paths, trim=True):
    """Concatenated training arrays from merged v3 files: x, mask, n_constituents, theta,
    global, global_names, event_id, group_id, point_id, weight, pthat (if present).

    ``x`` is trimmed to the four *modeled* kinematic features (``FEATURES[:4]``: deta, dphi,
    log_pt, log_e) -- column 4, the constituent PDG code, is not part of the density the flow
    learns (the three-factor likelihood in docs/PLAN_JEWEL_V3.md never conditions on it) and is
    dropped here so callers never have to know the on-disk layout carries a fifth column.
    """
    paths = [Path(p) for p in paths]
    parts = [read_jets(p) for p in paths]
    if not parts:
        raise ValueError("no v3 shard files given")
    names = parts[0]["feature_names"]
    if any(p["feature_names"] != names for p in parts):
        raise ValueError("shards disagree on feature_names")
    kin = [names.index(n) for n in FEATURES[:4]]  # deta, dphi, log_pt, log_e; pid excluded
    width = max(1, max(int(p["mask"].sum(1).max()) if p["mask"].size else 1 for p in parts)) \
        if trim else max(p["data"].shape[1] for p in parts)

    def pad_or_trim(a, w):
        if a.shape[1] == w:
            return a
        if a.shape[1] > w:
            return a[:, :w]
        z = [(0, 0), (0, w - a.shape[1])] + [(0, 0)] * (a.ndim - 2)
        return np.pad(a, z)

    out = {k: [] for k in ("x", "mask", "n_constituents", "global", "event_id",
                          "group_id", "point_id", "weight")}
    has_pthat = all("pthat" in p for p in parts)
    if has_pthat:
        out["pthat"] = []
    global_names = None
    for p in parts:
        out["x"].append(pad_or_trim(p["data"][..., kin], width))
        out["mask"].append(pad_or_trim(p["mask"], width))
        out["n_constituents"].append(p["mask"].sum(1).astype(np.int64))
        out["global"].append(p["global"])
        global_names = p["global_names"]
        out["event_id"].append(p["event_id"])
        out["group_id"].append(p["group_id"])
        out["point_id"].append(p["point_id"])
        out["weight"].append(p["weight"])
        if has_pthat:
            out["pthat"].append(p["pthat"])
    out = {k: np.concatenate(v) for k, v in out.items()}
    out["global_names"] = global_names
    out["theta"] = np.concatenate([p["theta"] for p in parts])
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


def plan_merge_groups(shards, target_jets=1_000_000):
    """Group per-task shards into output files of ~``target_jets``, keeping points whole.

    A theta point is never split across output files: the per-point weight normalisation
    (sum w = n_jets at that point) must be computed over all of a point's jets, and keeping
    points whole also means a reader can hold one point in memory from one file.
    """
    by_point = {}
    for s in shards:
        by_point.setdefault(int(s["point_id"]), []).append(s)
    groups, cur, cur_n = [], [], 0
    for pid in sorted(by_point):
        n = sum(int(s["n_jets"]) for s in by_point[pid])
        if cur and cur_n + n > target_jets:
            groups.append(cur)
            cur, cur_n = [], 0
        cur.extend(by_point[pid])
        cur_n += n
    if cur:
        groups.append(cur)
    return groups


def merge_v3(directory, out, split, target_jets=1_000_000, cfg=None, design=None, log=print):
    """Merge per-task v3 files into ~``target_jets``-jet files, with the sidecar.

    Weights are renormalised per point so ``sum(weight) = n_jets`` at that point; because
    points are never split across files this is exact.
    """
    from omnilearned.sbi.observations import file_sha256

    directory, out = Path(directory), Path(out)
    index = json.loads((directory / f"index_{split}.json").read_text())
    groups = plan_merge_groups(index["shards"], target_jets)
    out.mkdir(parents=True, exist_ok=True)
    files, total = [], 0
    for gi, grp in enumerate(groups):
        parts = [read_jets(directory / s["file"]) for s in grp]
        P = max(p["data"].shape[1] for p in parts)

        def pad(a, width=P):
            if a.shape[1] == width:
                return a
            z = [(0, 0), (0, width - a.shape[1])] + [(0, 0)] * (a.ndim - 2)
            return np.pad(a, z)

        d = {k: np.concatenate([p[k] for p in parts]) for k in
             ("global", "weight", "event_id", "group_id", "point_id", "theta")}
        x = np.concatenate([pad(p["data"]) for p in parts])
        mask = np.concatenate([pad(p["mask"]) for p in parts])
        pthat = (np.concatenate([p["pthat"] for p in parts])
                 if all("pthat" in p for p in parts) else None)
        w, pid_ = d["weight"].astype(np.float64), d["point_id"]
        for q in np.unique(pid_):
            sl = pid_ == q
            tot = w[sl].sum()
            if tot > 0:
                w[sl] *= sl.sum() / tot
        gn = list(GLOBALS)
        name = f"{split}_{gi:05d}.h5"
        sha = write_jets(
            out / name, x=x, mask=mask, theta=d["theta"],
            jet_pt=d["global"][:, gn.index("jet_pt")], jet_eta=d["global"][:, gn.index("jet_eta")],
            jet_phi=d["global"][:, gn.index("jet_phi")], jet_mass=d["global"][:, gn.index("jet_mass")],
            event_id=d["event_id"], group_id=d["group_id"], point_id=pid_, weight=w, pthat=pthat,
            attrs={"split": split, "merged_from_tasks": len(grp)})
        files.append({"file": name, "sha256": sha, "n_jets": int(x.shape[0]),
                      "n_points": int(np.unique(pid_).size), "max_constituents": int(P)})
        total += int(x.shape[0])
        log(f"  {name}: {x.shape[0]} jets, {np.unique(pid_).size} points, P={P}")
    idx = {"schema": SCHEMA, "split": split, "n_jets": total, "n_files": len(files),
           "source_index": str(directory / f"index_{split}.json"),
           "source_index_sha256": file_sha256(directory / f"index_{split}.json"),
           "files": files}
    (out / f"index_{split}.json").write_text(json.dumps(idx, indent=1))
    if cfg is not None:
        write_dataset_spec(out, dataset_spec(cfg, split, files, design))
    log(f"merged {index['tasks_done']} tasks -> {len(files)} files, {total} jets")
    return idx
