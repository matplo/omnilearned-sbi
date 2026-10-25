"""Training-scale toy datasets: continuous designs, shards, manifests, readers.

Design (``make_design``): scrambled Sobol points in the box, keyed by
``(design_seed, stream of the split)``. Points closer to a reserved setting
than ``margin`` of the box size in every coordinate simultaneously are dropped
(``margin_clashes``); exact equality is the special case ``check_reservation``
covers for Cartesian designs.

Shards (``generate_shard``): one HDF5 file per job, JEST-like layout

    data/x [n, 256, 3] float32   (eta, phi, log pT/GeV), zero-padded, left-packed
    data/mask [n, 256] bool, data/n_constituents [n] int32
    global [n, 7] float64        attrs["global_names"] = GLOBAL_NAMES
    event_id [n] int64 (stream-tagged), point_id [n] int32

with the hidden generator diagnostics in a separate ``*.diag.h5`` file that
training and fitting never read. Jets at point ``p`` of split ``s`` come from
``rng_for(seed_s, STREAMS[s], p)``, so a shard does not depend on how points
are grouped into jobs. Each shard has a JSON manifest: per point the toy
manifest (attempted, accepted, acceptance, rejections), ``array_sha256``
(bitwise, per instruction set), ``discrete_sha256`` and ``float_fingerprint``
(cross-host), and the file sha256. ``write_index`` merges shard manifests.

Readers (``read_shards``) cut the padded axis to the largest real
multiplicity; ``x`` chunks are 32 slots wide, so the padding is not read.
"""

import json
import math
import os
import platform
import socket
import subprocess
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import qmc

from omnilearned.sbi.observations import file_sha256
from omnilearned.sbi.toy.config import (
    V2_CONFIG,
    array_sha256,
    discrete_sha256,
    float_fingerprint,
    load_toy_config,
    sample_setting,
)
from omnilearned.sbi.toy.design import STREAMS, reserved_settings, rng_for
from omnilearned.sbi.toy.simulator import PARAMETERS

SHARD_SCHEMA = "toy-shard-v1"
BOX = {"b": (0.0, 11.0), "T_ref": (0.25, 0.55), "tau_init": (0.2, 1.2)}
GLOBAL_NAMES = ("b", "T_ref", "tau_init", "jet_pt", "jet_eta", "jet_phi", "n_constituents")
SPLIT_SEED = {"train": "training_seed", "validation": "validation_seed",
              "development": "validation_seed", "test": "test_seed",
              "fidelity": "validation_seed",  # direct-toy gate samples at validation points
              "development_scan": "validation_seed"}  # lines and tube through the development setting
X_CHUNK = (2048, 32, 3)


def box_arrays(box=BOX):
    lo = np.array([box[p][0] for p in PARAMETERS], dtype=np.float64)
    hi = np.array([box[p][1] for p in PARAMETERS], dtype=np.float64)
    return lo, hi


def margin_clashes(points, reserved, box=BOX, margin=0.02):
    """Boolean [n]: point within ``margin`` x box size of a reserved setting on every axis."""
    lo, hi = box_arrays(box)
    pts = np.atleast_2d(points)
    clash = np.zeros(pts.shape[0], bool)
    for th in reserved.values():
        clash |= np.all(np.abs(pts - np.asarray(th)) < margin * (hi - lo), axis=1)
    return clash


def make_design(n, split, cfg, box=BOX, margin=0.02):
    """First ``n`` scrambled-Sobol points of the split that respect the margin.

    Returns (points [n, 3], info) with the number of dropped points.
    """
    seed = int(cfg["randomness"]["design_seed"])
    rng = rng_for(seed, STREAMS[split])
    sob = qmc.Sobol(d=len(PARAMETERS), scramble=True, seed=rng)
    m = max(1, math.ceil(math.log2(max(2, int(1.1 * n) + 8))))
    unit = sob.random_base2(m)
    lo, hi = box_arrays(box)
    pts = lo + unit * (hi - lo)
    clash = margin_clashes(pts, reserved_settings(cfg), box, margin)
    kept = np.flatnonzero(~clash)
    if kept.size < n:
        raise ValueError("not enough design points after the reservation margin")
    info = {"split": split, "n": int(n), "sobol_log2": int(m), "design_seed": seed,
            "stream": STREAMS[split], "dropped_by_margin": int(clash[:kept[n - 1] + 1].sum()),
            "margin": margin, "box": {k: list(v) for k, v in box.items()}}
    return pts[kept[:n]], info


def source_revision(repo=None):
    """(git revision, dirty flag) of the checkout this module is imported from.

    A source snapshot (``git archive`` for batch jobs) has no ``.git``; it carries
    a ``SOURCE_REVISION`` file with the archived commit and is clean by construction.
    """
    repo = Path(repo or Path(__file__).resolve().parents[4])
    stamp = repo / "SOURCE_REVISION"
    if stamp.exists() and not (repo / ".git").exists():
        return stamp.read_text().strip(), False
    try:
        rev = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=True).stdout.strip()
        st = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
                            capture_output=True, text=True, check=True).stdout.strip()
        return rev, bool(st)
    except (OSError, subprocess.CalledProcessError):
        return None, None


def host_info():
    try:
        from numpy._core._multiarray_umath import __cpu_features__ as feats
    except ImportError:
        feats = {}
    rev, dirty = source_revision()
    return {"host": socket.gethostname(), "machine": platform.machine(),
            "processor": platform.processor(), "numpy": np.__version__,
            "avx512f": bool(feats.get("AVX512F", False)), "avx2": bool(feats.get("AVX2", False)),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "source_revision": rev, "source_dirty": dirty}


def _dataset(f, name, data, **kw):
    return f.create_dataset(name, data=data, track_times=False, **kw)


def write_shard(path, parts, split, backend, toy_config_sha256):
    """parts: list of (point_id, theta, records). Returns (file sha256, diag sha256)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cat = lambda k: np.concatenate([r[k] for _, _, r in parts])  # noqa: E731
    x = cat("x").astype(np.float32)
    mask = cat("mask").astype(bool)
    if np.any(x[~mask] != 0):
        raise ValueError("padded entries must be zero")
    n = x.shape[0]
    theta = np.concatenate([np.repeat(np.asarray(t, np.float64)[None], r["x"].shape[0], 0)
                            for _, t, r in parts]) if n else np.zeros((0, 3))
    nconst = mask.sum(1).astype(np.int32)
    glob = np.column_stack([theta, cat("jet_pt"), cat("jet_eta"), cat("jet_phi"),
                            nconst.astype(np.float64)]) if n else np.zeros((0, len(GLOBAL_NAMES)))
    pid = np.concatenate([np.full(r["x"].shape[0], p, np.int32) for p, _, r in parts])
    chunk = lambda c: tuple(min(a, b) for a, b in zip(c, x.shape)) if n else None  # noqa: E731
    with h5py.File(path, "w", track_order=False) as f:
        _dataset(f, "data/x", x, compression="gzip", chunks=chunk(X_CHUNK))
        _dataset(f, "data/mask", mask, compression="gzip", chunks=chunk(X_CHUNK)[:2] if n else None)
        _dataset(f, "data/n_constituents", nconst)
        g = _dataset(f, "global", glob)
        g.attrs["global_names"] = list(GLOBAL_NAMES)
        _dataset(f, "event_id", cat("event_id").astype(np.int64))
        _dataset(f, "point_id", pid)
        f.attrs["schema"] = SHARD_SCHEMA
        f.attrs["backend"] = backend
        f.attrs["split"] = split
        f.attrs["toy_config_sha256"] = toy_config_sha256
        f.attrs["feature_names"] = ["eta", "phi", "log_pt_over_GeV"]
        f.attrs["parameter_names"] = list(PARAMETERS)
        f.attrs["max_constituents"] = int(x.shape[1])
    dpath = path.with_suffix(".diag.h5")
    with h5py.File(dpath, "w") as f:
        _dataset(f, "event_id", cat("event_id").astype(np.int64))
        for k in parts[0][2]["diag"]:
            _dataset(f, k, np.concatenate([r["diag"][k] for _, _, r in parts]))
        f.attrs["access"] = "diagnostic only; never an input to training or fitting"
    return file_sha256(path), file_sha256(dpath)


COUNTS_SCHEMA = "toy-counts-v1"


def write_counts_shard(path, parts, split, backend, toy_config_sha256):
    """Multiplicities only (for p(N | theta)): n_constituents, theta, event_id, point_id."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        _dataset(f, "n_constituents", np.concatenate([r["n_constituents"] for _, _, r in parts]).astype(np.int32))
        _dataset(f, "theta", np.concatenate([np.repeat(np.asarray(t, np.float64)[None], len(r["n_constituents"]), 0)
                                             for _, t, r in parts]))
        _dataset(f, "event_id", np.concatenate([r["event_id"] for _, _, r in parts]).astype(np.int64))
        _dataset(f, "point_id", np.concatenate([np.full(len(r["n_constituents"]), p, np.int32) for p, _, r in parts]))
        f.attrs.update({"schema": COUNTS_SCHEMA, "backend": backend, "split": split,
                        "toy_config_sha256": toy_config_sha256,
                        "parameter_names": list(PARAMETERS)})
    return file_sha256(path)


def read_counts(paths):
    """(n_constituents, theta, point_id) from counts shards."""
    n, th, pid = [], [], []
    for p in paths:
        with h5py.File(p, "r") as f:
            if f.attrs.get("schema") != COUNTS_SCHEMA:
                raise ValueError(f"{p} is not a {COUNTS_SCHEMA} file")
            n.append(f["n_constituents"][...])
            th.append(f["theta"][...])
            pid.append(f["point_id"][...])
    return np.concatenate(n), np.concatenate(th), np.concatenate(pid)


def generate_shard(out, split, point_ids, points, jets_per_point, config=V2_CONFIG, log=print,
                   counts_only=False):
    """Draw ``jets_per_point`` selected jets at each point and write one shard + manifest.

    ``counts_only`` writes only the multiplicities (``write_counts_shard``)."""
    cfg, _, _ = load_toy_config(config)
    seed = int(cfg["randomness"][SPLIT_SEED[split]])
    start = time.time()
    parts, per_point = [], []
    for pid, th in zip(point_ids, points):
        t0 = time.time()
        rec, man = sample_setting(th, jets_per_point, seed, stream=split, index=int(pid), path=config)
        if not man["complete"]:
            raise RuntimeError(f"point {pid} {th} incomplete: {man}")
        per_point.append({"point_id": int(pid), "theta": [float(v) for v in th], "toy": man,
                          "array_sha256": array_sha256(rec), "discrete_sha256": discrete_sha256(rec),
                          "float_fingerprint": float_fingerprint(rec).tolist(),
                          "seconds": time.time() - t0})
        if counts_only:  # keep the multiplicities only (padded records are ~GB per shard)
            rec = {"n_constituents": rec["n_constituents"].copy(), "event_id": rec["event_id"].copy()}
        parts.append((int(pid), np.asarray(th, np.float64), rec))
        log(f"  {split} point {pid} theta={np.round(th, 4).tolist()} acc={man['acceptance']:.4f} "
            f"N=[{man['n_constituents_range'][0]},{man['n_constituents_range'][1]}] {time.time() - t0:.1f}s")
    cfg_sha = file_sha256(config)
    if counts_only:
        sha, dsha = write_counts_shard(out, parts, split, cfg["backend"], cfg_sha), None
    else:
        sha, dsha = write_shard(out, parts, split, cfg["backend"], cfg_sha)
    manifest = {"schema": COUNTS_SCHEMA if counts_only else SHARD_SCHEMA,
                "file": Path(out).name, "sha256": sha,
                "diag_file": None if counts_only else Path(out).with_suffix(".diag.h5").name,
                "diag_sha256": dsha,
                "split": split, "backend": cfg["backend"], "toy_config": str(config),
                "toy_config_sha256": cfg_sha, "seed": seed, "seed_key": SPLIT_SEED[split],
                "stream": STREAMS[split], "jets_per_point": int(jets_per_point),
                "n_jets": int(sum(len(r["event_id"]) for _, _, r in parts)),
                "points": per_point, "wall_seconds": time.time() - start, "host": host_info()}
    with open(Path(out).with_suffix(".json"), "w") as f:
        json.dump(manifest, f, indent=1)
    return manifest


def write_index(directory, split, pattern=None):
    """Merge the shard manifests of a split into ``index_<split>.json``."""
    directory = Path(directory)
    mans = []
    for p in sorted(directory.glob(pattern or f"{split}_*.json")):
        with open(p) as f:
            m = json.load(f)
        if m.get("schema") == SHARD_SCHEMA and m["split"] == split:
            if file_sha256(directory / m["file"]) != m["sha256"]:
                raise ValueError(f"{m['file']} does not match its manifest hash")
            mans.append(m)
    pids = [q["point_id"] for m in mans for q in m["points"]]
    if len(set(pids)) != len(pids):
        raise ValueError("a point appears in more than one shard")
    acc = np.array([q["toy"]["acceptance"] for m in mans for q in m["points"]])
    index = {"schema": SHARD_SCHEMA + "-index", "split": split,
             "shards": [{"file": m["file"], "sha256": m["sha256"], "n_jets": m["n_jets"],
                         "point_ids": [q["point_id"] for q in m["points"]]} for m in mans],
             "n_points": len(pids), "n_jets": int(sum(m["n_jets"] for m in mans)),
             "acceptance_range": [float(acc.min()), float(acc.max())] if acc.size else None,
             "backend": sorted({m["backend"] for m in mans}),
             "toy_config_sha256": sorted({m["toy_config_sha256"] for m in mans})}
    with open(directory / f"index_{split}.json", "w") as f:
        json.dump(index, f, indent=1)
    return index


def shard_paths(directory, split):
    """Shard files of a split, from its index (hash-checked) or by name."""
    directory = Path(directory)
    idx = directory / f"index_{split}.json"
    if idx.exists():
        with open(idx) as f:
            return [directory / s["file"] for s in json.load(f)["shards"]]
    return sorted(p for p in directory.glob(f"{split}_*.h5") if not p.name.endswith(".diag.h5"))


def read_shards(paths, trim=True):
    """Concatenated arrays of shard files: x, mask, n_constituents, theta, global, ids."""
    paths = [Path(p) for p in paths]
    width = 1
    for p in paths:
        with h5py.File(p, "r") as f:
            n = f["data/n_constituents"][...]
            width = max(width, int(n.max()) if n.size else 1)
            full = int(f.attrs["max_constituents"])
    w = width if trim else full
    out = {k: [] for k in ("x", "mask", "n_constituents", "global", "event_id", "point_id")}
    names = None
    for p in paths:
        with h5py.File(p, "r") as f:
            if f.attrs.get("schema") != SHARD_SCHEMA:
                raise ValueError(f"{p} is not a {SHARD_SCHEMA} file")
            out["x"].append(f["data/x"][:, :w])
            out["mask"].append(f["data/mask"][:, :w].astype(bool))
            out["n_constituents"].append(f["data/n_constituents"][...])
            out["global"].append(f["global"][...])
            names = list(f["global"].attrs["global_names"])
            out["event_id"].append(f["event_id"][...])
            out["point_id"].append(f["point_id"][...])
    out = {k: np.concatenate(v) for k, v in out.items()}
    out["global_names"] = names
    out["theta"] = out["global"][:, :len(PARAMETERS)]
    return out


def make_observations(out, truth_out, theta, n, split, index, config=V2_CONFIG):
    """Pseudo data from the toy at ``theta``: observation file (no truth) + truth manifest.

    The truth manifest goes to a separate path that fits never read; for test
    settings it is opened only by the report after the fit outputs are sealed.
    """
    from omnilearned.sbi.toy.records import write_observation_file, write_truth

    cfg, _, _ = load_toy_config(config)
    seed = int(cfg["randomness"][SPLIT_SEED[split]])
    rec, man = sample_setting(theta, n, seed, stream=split, index=index, path=config)
    w = max(1, int(rec["n_constituents"].max()))
    sha = write_observation_file(out, {"x": rec["x"][:, :w], "mask": rec["mask"][:, :w],
                                       "event_id": rec["event_id"]},
                                 attrs={"backend": cfg["backend"], "n_events": int(n)})
    tsha = write_truth(truth_out, {"observations": str(out), "observations_sha256": sha,
                                   "theta": [float(v) for v in theta], "toy": man,
                                   "seed": seed, "stream": split, "index": int(index),
                                   "host": host_info()})
    return sha, tsha


def _cli(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("design", help="write a Sobol design JSON")
    d.add_argument("--split", required=True, choices=sorted(SPLIT_SEED))
    d.add_argument("--n", type=int, required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--config", default=str(V2_CONFIG))
    d.add_argument("--from-index", type=int, default=0,
                   help="keep design points from this index on (extend an existing design: the "
                        "Sobol prefix is unchanged, so points [0, from) are the smaller design)")
    d.add_argument("--select", nargs=3, action="append", default=[], metavar=("PARAM", "LO", "HI"),
                   help="keep only points with LO < PARAM <= HI (repeatable); point ids stay the "
                        "design indices, so their random streams are distinct from other points")
    g = sub.add_parser("generate", help="generate one shard of a design")
    g.add_argument("--design", required=True)
    g.add_argument("--shard", type=int, default=None, help="default: $SLURM_ARRAY_TASK_ID")
    g.add_argument("--points-per-shard", type=int, required=True)
    g.add_argument("--jets", type=int, required=True)
    g.add_argument("--out-dir", required=True)
    g.add_argument("--config", default=str(V2_CONFIG))
    g.add_argument("--skip-existing", action="store_true")
    g.add_argument("--shard-offset", type=int, default=0,
                   help="file number = shard + offset (add shards next to another design's)")
    g.add_argument("--counts-only", action="store_true",
                   help="store only multiplicities (training data for p(N | theta))")
    g.add_argument("--split", default=None, choices=sorted(SPLIT_SEED),
                   help="draw with this split's stream and seed instead of the design's "
                        "(e.g. fidelity samples at validation points)")
    i = sub.add_parser("index", help="merge shard manifests of a split")
    i.add_argument("--dir", required=True)
    i.add_argument("--split", required=True)
    o = sub.add_parser("observe", help="pseudo-data observation file from the toy")
    o.add_argument("--theta", type=float, nargs=3, required=True)
    o.add_argument("--n", type=int, required=True)
    o.add_argument("--split", default="development", choices=sorted(SPLIT_SEED))
    o.add_argument("--index", type=int, required=True)
    o.add_argument("--out", required=True)
    o.add_argument("--truth-out", required=True)
    o.add_argument("--config", default=str(V2_CONFIG))
    a = ap.parse_args(argv)
    if a.cmd == "observe":
        sha, _ = make_observations(a.out, a.truth_out, a.theta, a.n, a.split, a.index, a.config)
        print(f"observations {a.n} jets -> {a.out} sha256 {sha}")
    elif a.cmd == "design":
        cfg, _, _ = load_toy_config(a.config)
        pts, info = make_design(a.n, a.split, cfg)
        ids = np.arange(len(pts))
        keep = ids >= a.from_index
        for name, lo, hi in a.select:
            col = pts[:, list(PARAMETERS).index(name)]
            keep &= (col > float(lo)) & (col <= float(hi))
        if a.from_index or a.select:
            info.update({"from_index": a.from_index, "select": a.select, "point_ids": ids[keep].tolist()})
        pts = pts[keep]
        info.update({"parameters": list(PARAMETERS), "points": pts.tolist(),
                     "toy_config": a.config, "toy_config_sha256": file_sha256(a.config),
                     "host": host_info()})
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(info, f, indent=1)
        print(f"design {a.split}: {len(pts)} of {a.n} points, dropped {info['dropped_by_margin']} by margin -> {a.out}")
    elif a.cmd == "generate":
        with open(a.design) as f:
            des = json.load(f)
        k = a.shard if a.shard is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])
        pts = np.asarray(des["points"])
        rows = np.arange(k * a.points_per_shard, min(len(pts), (k + 1) * a.points_per_shard))
        if rows.size == 0:
            print(f"shard {k}: no points")
            return
        pid = np.asarray(des.get("point_ids", np.arange(len(pts))))
        split = a.split or des["split"]
        out = Path(a.out_dir) / f"{split}{'_counts' if a.counts_only else ''}_{k + a.shard_offset:05d}.h5"
        if a.skip_existing and out.with_suffix(".json").exists():
            print(f"shard {k}: exists, skipped")
            return
        m = generate_shard(out, split, pid[rows], pts[rows], a.jets, a.config, counts_only=a.counts_only)
        print(f"shard {k}: {m['n_jets']} jets, {len(rows)} points, {m['wall_seconds']:.1f}s -> {out}")
    else:
        idx = write_index(a.dir, a.split)
        print(f"index {a.split}: {idx['n_points']} points, {idx['n_jets']} jets, "
              f"acceptance {idx['acceptance_range']}")


if __name__ == "__main__":
    _cli()
