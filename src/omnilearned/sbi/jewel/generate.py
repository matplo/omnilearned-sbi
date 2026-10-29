"""JEWEL generation worker: tasks on a parameter design -> jet shards + raw event stores.

A task is (split, point, chunk): ``events_per_task`` JEWEL events at one θ = (b, TI, TAUI)
with its own random stream NJOB. The worker runs JEWEL in node-local scratch, reads the
HepMC file event by event, and writes next to each other in the output directory:

* ``<task>.h5`` + ``<task>.diag.h5``: selected jets in the toy-v2 shard layout
  (``toy.datasets.write_shard``: data/x [n, 256, 3], mask, global, event_id, point_id), so
  training, the gate and the readers work unchanged; hidden per-jet diagnostics apart;
* ``<task>.events.h5``: the raw event store, every event's status 1/3/4 records (dummies
  dropped) and status-23 hard partons as float32 px, py, pz, E, int32 pid, int8 status with
  per-event offsets, plus an event table. Raw, so the subtraction (another ΔR_max or leftover
  convention) and the jets (another R or cut) can be redone without rerunning JEWEL;
* ``<task>.json``: the manifest (cards, NJOB, θ, centrality, counts, rejections, timing,
  JEWEL log tail, file hashes, host, source revision).

HepMC files are deleted after the task unless the task is marked ``keep_hepmc`` (gzipped
copy under ``hepmc/``). Tasks are listed in ``tasks.json`` (``plan``); workers claim them
with exclusive lock files (``omnilearned.sbi.tasks``) and skip tasks with a manifest, so any
number of workers on any number of nodes can share one directory, and reruns resume.

    python -m omnilearned.sbi.jewel.generate plan --split train --out DIR [--chunks 3]
    python -m omnilearned.sbi.jewel.generate work --dir DIR
    python -m omnilearned.sbi.jewel.generate merge --dir DIR --out MERGED --shards 64

Weighted samples (``plan --wexpo K``: WEIGHTED T, WEXPO K, a sampling option recorded per task)
add the per-jet ``weight`` (the raw HepMC event weight pT-hat^-K) and ``pthat`` to the task
shards; ``merge`` rescales the weights per point so that they sum to the point's jets.

``merge`` concatenates the per-task jet shards of a split (tens of thousands of small files,
slow to open on CFS) into a few point-major shards with the same layout, plus diagnostics, a
manifest per merged shard listing its source tasks with their hashes, and an
``index_<split>.json`` that ``toy.datasets.shard_paths`` reads; train and validation can share
one merged directory. The event stores stay with the tasks.
"""

import argparse
import gzip
import json
import os
import shutil
import signal
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np

from omnilearned.sbi import tasks as tasklocks
from omnilearned.sbi.jewel import BACKEND, cards, hepmc, jets
from omnilearned.sbi.jewel.geometry import JewelGeometry
from omnilearned.sbi.observations import file_sha256
from omnilearned.sbi.toy.datasets import GLOBAL_NAMES, X_CHUNK, _dataset, host_info, make_design, write_shard
from omnilearned.sbi.toy.design import STREAMS
from omnilearned.sbi.toy.simulator import PARAMETERS

EVENTS_SCHEMA = "jewel-events-v1"
MERGED_SCHEMA = "jewel-merged-v1"
TASKS_SCHEMA = "jewel-tasks-v1"


def task_name(split, point_id, chunk):
    return f"{split}_p{int(point_id):05d}_c{int(chunk):03d}"


def box(cfg):
    return {p: tuple(cfg["design"]["box"][p]) for p in PARAMETERS}


def design_points(cfg, split):
    """Points of a split: Sobol if ``design.train_points`` is set, else the Cartesian grid.

    ``design.train_points = N`` selects a continuous scrambled-Sobol training design (the NLE
    needs the surrogate to interpolate smoothly in theta, and a Cartesian grid leaves
    axis-aligned structure); without it the historical Cartesian grid is used, so configs
    written before this option behave exactly as they did.
    """
    if split == "train":
        n = cfg["design"].get("train_points")
        if n:
            pts, _ = make_design(int(n), "train", cfg, box=box(cfg))
            return pts
        axes = [np.asarray(cfg["design"][p], float) for p in PARAMETERS]
        return np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, len(PARAMETERS))
    if split in ("validation", "fidelity"):
        pts, _ = make_design(int(cfg["design"]["validation_points"]), "validation", cfg, box=box(cfg))
        return pts
    raise ValueError(f"no design for split {split}")


def plan(cfg, split, out, chunks=1, points=None, theta=None, keep_hepmc=(), extra=None,
         nevent=None, keep_events=True, sampling=None):
    """Write ``tasks.json``: every (point, chunk) of the split (or explicit ``theta`` rows),
    chunk-major. ``chunks`` is a number for every point or a dict point -> number of chunks.
    ``keep_events=False`` writes jets and diagnostics but no raw event store.
    ``sampling = {"wexpo": k}``: weighted events (``cards.task_cards``)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    pts = np.atleast_2d(np.asarray(theta, float)) if theta is not None else design_points(cfg, split)
    ids = list(range(len(pts))) if points is None else list(points)
    nev = int(nevent or cfg["design"]["events_per_task"])
    tl = []
    per = {pid: int(chunks[pid] if isinstance(chunks, dict) else chunks) for pid in ids}
    # chunk-major: every point's chunk 0 first, so a partial run covers the design evenly
    for c in range(max(per.values(), default=0)):
        for pid in ids:
            if c >= per[pid]:
                continue
            tl.append({"name": task_name(split, pid, c), "split": split, "point_id": int(pid),
                       "chunk": c, "theta": [float(v) for v in pts[pid]], "nevent": nev,
                       "njob": cards.njob(cfg, split, pid, c),
                       "keep_hepmc": task_name(split, pid, c) in set(keep_hepmc),
                       "keep_events": bool(keep_events), "extra_cards": extra or {},
                       **({"sampling": dict(sampling)} if sampling else {})})
    doc = {"schema": TASKS_SCHEMA, "backend": cfg["backend"], "split": split,
           "parameters": list(PARAMETERS), "tasks": tl, "config": cfg, "host": host_info()}
    with open(out / "tasks.json", "w") as f:
        json.dump(doc, f, indent=1)
    return doc


def chunks_for_target(index_dir, target_jets, nevent, margin=1.05):
    """Chunks per point to reach ``target_jets`` at the acceptance measured in another split
    at the same points (``index_<split>.json`` + ``tasks.json`` of ``index_dir``)."""
    index_dir = Path(index_dir)
    idx = json.loads(next(index_dir.glob("index_*.json")).read_text())
    events = {}
    for t in load(index_dir)["tasks"]:
        events[t["point_id"]] = events.get(t["point_id"], 0) + int(t["nevent"])
    jets_ = {}
    for s in idx["shards"]:
        jets_[s["point_id"]] = jets_.get(s["point_id"], 0) + int(s["n_jets"])
    return {pid: int(np.ceil(target_jets * margin / max(jets_.get(pid, 0) / ev * nevent, 1e-9)))
            for pid, ev in sorted(events.items())}


def topup(directory, target_jets, margin=1.05, max_chunks=None, log=print):
    """Append tasks so that every point reaches ``target_jets`` selected jets.

    The chunks a point needs follow from its measured acceptance (finished tasks of the
    point); points without a finished task are skipped. Chunk indices continue after the
    existing ones (new NJOB streams), chunk-major. Rewrites ``tasks.json`` atomically;
    running workers keep their own list and claims prevent double work.
    """
    directory = Path(directory)
    doc = load(directory)
    cfg = doc["config"]
    cap = int(max_chunks or cfg["randomness"]["chunks_per_point"])
    by_point, done = {}, {}
    for t in doc["tasks"]:
        by_point.setdefault(t["point_id"], []).append(t)
        p = directory / f"{t['name']}.json"
        if p.exists():
            with open(p) as f:
                m = json.load(f)
            done.setdefault(t["point_id"], []).append((m["jets_selected"], m["events_generated"]))
    new = []
    for pid, tl in by_point.items():
        if pid not in done:
            continue
        jets_, ev = (sum(v) for v in zip(*done[pid]))
        acc = jets_ / max(1, ev)
        want = min(cap, int(np.ceil(target_jets * margin / max(acc * tl[0]["nevent"], 1e-9))))
        have = max(t["chunk"] for t in tl) + 1
        for c in range(have, want):
            t0 = tl[0]
            new.append({**t0, "name": task_name(doc["split"], pid, c), "chunk": c,
                        "njob": cards.njob(cfg, doc["split"], pid, c), "keep_hepmc": False})
    new.sort(key=lambda t: (t["chunk"], t["point_id"]))
    doc["tasks"] = doc["tasks"] + new
    doc.setdefault("topups", []).append({"target_jets": target_jets, "margin": margin,
                                         "added": len(new), "time": time.time()})
    tmp = directory / f"tasks.json.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=1)
    os.replace(tmp, directory / "tasks.json")
    log(f"top-up to {target_jets} jets: {len(new)} tasks added ({len(doc['tasks'])} in total)")
    return new


# ---------------------------------------------------------------------------
# one task
# ---------------------------------------------------------------------------

def _write_events(path, events, stream, point_id, chunk, nevent):
    keys = ("px", "py", "pz", "e")
    sel = [np.isin(ev["status"], (1, 3, 4, 23)) & ~((ev["pid"] == 111) & (np.abs(ev["e"] - 1e-6) < 1e-11))
           for ev in events]
    counts = np.array([int(s.sum()) for s in sel], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    cat = lambda k, dt: (np.concatenate([ev[k][s] for ev, s in zip(events, sel)]).astype(dt)  # noqa: E731
                         if events else np.zeros(0, dt))
    with h5py.File(path, "w") as f:
        g = f.create_group("particles")
        for k in keys:
            g.create_dataset(k, data=cat(k, np.float32), compression="gzip", shuffle=True,
                             chunks=True if counts.sum() else None, track_times=False)
        g.create_dataset("pid", data=cat("pid", np.int32), compression="gzip", track_times=False)
        g.create_dataset("status", data=cat("status", np.int8), compression="gzip", track_times=False)
        f.create_dataset("offsets", data=offsets, track_times=False)
        t = f.create_group("events")
        num = np.array([ev["number"] for ev in events], dtype=np.int64)
        t.create_dataset("event_id", data=_event_ids(stream, point_id, chunk, nevent, num), track_times=False)
        t.create_dataset("number", data=num, track_times=False)
        for k in ("weight", "scale", "xsec_pb", "centrality"):
            t.create_dataset(k, data=np.array([ev[k] for ev in events], dtype=np.float64), track_times=False)
        f.attrs["schema"] = EVENTS_SCHEMA
        f.attrs["content"] = ("raw HepMC records: status 1 and 4 (final state, dummies dropped), "
                              "3 (thermal momenta), 23 (hard partons); float32 px, py, pz, E [GeV]")
    return file_sha256(path)


def _event_ids(stream, point_id, chunk, nevent, number):
    return ((np.int64(stream) << 48) | (np.int64(point_id) << 32)
            | (np.int64(chunk) * np.int64(nevent) + np.asarray(number, np.int64)))


def read_events_store(path):
    """Events of an event store as dicts (pid, px, py, pz, e, status, number, ...)."""
    out = []
    with h5py.File(path, "r") as f:
        off = f["offsets"][...]
        p = {k: f["particles"][k][...] for k in ("px", "py", "pz", "e", "pid", "status")}
        t = {k: f["events"][k][...] for k in f["events"]}
    for i in range(len(off) - 1):
        sl = slice(off[i], off[i + 1])
        ev = {k: (v[sl].astype(np.float64) if v.dtype.kind == "f" else v[sl]) for k, v in p.items()}
        ev.update({k: v[i] for k, v in t.items()})
        out.append(ev)
    return out


def jewel_command(cfg, exe):
    """argv that runs ``exe params.dat`` with the package env scripts sourced (if any)."""
    j = cfg["jewel"]
    scripts = j.get("env_scripts", [])
    launcher = list(j.get("launcher", []))
    if not scripts:
        return launcher + [exe, "params.dat"]
    src = "; ".join(f'source "{s}"' for s in scripts)
    data = j.get("lhapdf_data")
    exp = f'; export LHAPDF_DATA_PATH="{data}"' if data else ""
    return launcher + ["bash", "-c", f'{src}{exp}; exec "$0" "$@"', exe, "params.dat"]


def _keep_failure(work, out_dir, name):
    d = Path(out_dir) / "failed" / name
    d.mkdir(parents=True, exist_ok=True)
    for f in ("stdout.txt", "jewel.log", "params.dat", "medium.dat"):
        if (work / f).exists():
            shutil.copy(work / f, d / f)


def run_task(cfg, task, out_dir, scratch=None, timeout=None, log=print):
    """Run one task; returns its manifest (written to ``<task>.json``)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = task["name"]
    geom = JewelGeometry(A=int(cfg["jewel"]["mass_number"]))
    stream = STREAMS[cards.SPLIT_STREAM[task["split"]]]
    t0 = time.time()
    work = Path(tempfile.mkdtemp(prefix=f"jewel_{name}_", dir=scratch))
    try:
        params, medium, cent = cards.task_cards(cfg, task["theta"], task["nevent"], task["njob"],
                                                geom, tables_dir=".", extra=task.get("extra_cards"),
                                                sampling=task.get("sampling"))
        (work / "params.dat").write_text(params)
        (work / "medium.dat").write_text(medium)
        exe = cards.inspect_installation(cfg)["executable"] if not cfg["jewel"].get("executable") \
            else cfg["jewel"]["executable"]
        cmd = jewel_command(cfg, exe)
        with open(work / "stdout.txt", "w") as so:
            proc = subprocess.Popen(cmd, cwd=work, stdout=so, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            try:
                rc = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                rc = "timeout"
        t_gen = time.time() - t0
        hep = work / "events.hepmc"
        logtail = (work / "jewel.log").read_text(errors="replace").splitlines()[-40:] \
            if (work / "jewel.log").exists() else []
        if rc != 0 and (not hep.exists() or hep.stat().st_size == 0):
            _keep_failure(work, out_dir, name)
            raise RuntimeError(f"JEWEL {name}: rc={rc}, no HepMC; log tail: {logtail[-5:]}")
        sel = jets.selection_from_config(cfg)
        t1 = time.time()
        events, recs, diags, pre_reject = [], [], [], {r: 0 for r in jets.REASONS}
        all_events = list(hepmc.read_events(hep)) if hep.exists() else []
        if rc != 0 and all_events:
            all_events = all_events[:-1]  # the last event of an interrupted run may be incomplete
        for ev in all_events:
            events.append(ev)
            evrecs, why, evdiags, pre = jets.event_jets(ev, cfg, sel)
            for reason_, count_ in pre.items():  # dropped before packing, still accounted for
                pre_reject[reason_] += count_
            if not evrecs:
                pre_reject[why] += 1
                continue
            for rec, d in zip(evrecs, evdiags):  # jets of one event share its number -> event_id
                d.update({"number": ev["number"], "centrality_record": ev["centrality"],
                          "weight": ev["weight"], "pthat": ev["scale"]})
                recs.append(rec)
                diags.append(d)
        if recs:
            ok, why, r, x, mask = jets.select_records(recs, sel)
        else:
            ok, why, r = np.zeros(0, bool), [], {"jet_pt": np.zeros(0), "jet_eta": np.zeros(0),
                                                  "jet_phi": np.zeros(0)}
            x, mask = np.zeros((0, sel.max_constituents, 3)), np.zeros((0, sel.max_constituents), bool)
        reject = dict(pre_reject)
        for w in why:
            if w is not None:
                reject[w] += 1
        keep = np.flatnonzero(ok)
        num = np.array([diags[i]["number"] for i in keep], dtype=np.int64)
        dkeys = [k for k in (diags[0] if diags else {}) if k != "number"]
        rec = {"x": x[keep], "mask": mask[keep], "jet_pt": r["jet_pt"][keep],
               "jet_eta": r["jet_eta"][keep], "jet_phi": r["jet_phi"][keep],
               "event_id": _event_ids(stream, task["point_id"], task["chunk"], task["nevent"], num),
               "diag": {**{k: np.array([diags[i][k] for i in keep], dtype=np.float64) for k in dkeys},
                        "number": num, "b_requested": np.full(len(keep), task["theta"][0]),
                        "centrality": np.full(len(keep), cent)}}
        bias = None
        if task.get("sampling"):  # raw HepMC weights; merge normalises them per point
            rec["weight"], rec["pthat"] = rec["diag"]["weight"].copy(), rec["diag"]["pthat"].copy()
            bias = sampling_record(task["sampling"])
        cfg_sha = cfg.get("_sha256", "unknown")  # of the config the task list was planned from
        sha, dsha = write_shard(out_dir / f"{name}.h5", [(task["point_id"], task["theta"], rec)],
                                task["split"], cfg["backend"], cfg_sha, sampling_bias=bias)
        keep_ev = task.get("keep_events", True)
        esha = _write_events(out_dir / f"{name}.events.h5", events, stream, task["point_id"],
                             task["chunk"], task["nevent"]) if keep_ev else None
        t_proc = time.time() - t1
        if task.get("keep_hepmc"):
            (out_dir / "hepmc").mkdir(exist_ok=True)
            with open(hep, "rb") as fi, gzip.open(out_dir / "hepmc" / f"{name}.hepmc.gz", "wb") as fo:
                shutil.copyfileobj(fi, fo)
        nc = mask[keep].sum(1) if len(keep) else np.zeros(0, int)
        man = {"schema": "jewel-task-v1", "backend": cfg["backend"], "task": task,
               "theta": dict(zip(PARAMETERS, task["theta"])), "centrality_percent": cent,
               "b_of_centrality": geom.b_of_centrality(cent), "jewel_rc": rc,
               "events_generated": len(events), "jets_selected": int(len(keep)),
               # jets per event: > 1 is normal for jet_multiplicity = inclusive
               "acceptance": len(keep) / max(1, len(events)), "rejections": reject,
               "jet_multiplicity": cfg["selection"].get("jet_multiplicity", "inclusive"),
               "events_with_a_jet": int(len(np.unique(num))) if len(keep) else 0,
               "n_constituents_range": [int(nc.min()), int(nc.max())] if len(nc) else None,
               "seconds_jewel": t_gen, "seconds_processing": t_proc,
               "seconds_per_event": t_gen / max(1, len(events)),
               "files": {"jets": f"{name}.h5", "jets_sha256": sha, "diag": f"{name}.diag.h5",
                         "diag_sha256": dsha, "events": f"{name}.events.h5" if keep_ev else None,
                         "events_sha256": esha},
               "events_bytes": os.path.getsize(out_dir / f"{name}.events.h5") if keep_ev else 0,
               "cards": {"params": params, "medium": medium}, "jewel_log_tail": logtail,
               "installation": {k: v for k, v in cards.inspect_installation(cfg).items() if k != "patches"},
               "host": host_info(), "wall_seconds": time.time() - t0}
        with open(out_dir / f"{name}.json", "w") as f:
            json.dump(man, f, indent=1, default=float)
        log(f"{name}: {len(events)} events, {len(keep)} jets ({man['acceptance']:.3f}), "
            f"JEWEL {t_gen:.0f}s ({man['seconds_per_event']:.2f} s/ev), processing {t_proc:.0f}s, rc={rc}")
        return man
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ---------------------------------------------------------------------------
# farm
# ---------------------------------------------------------------------------

def load(directory):
    with open(Path(directory) / "tasks.json") as f:
        doc = json.load(f)
    if doc.get("schema") != TASKS_SCHEMA:
        raise ValueError("not a JEWEL task list")
    return doc


def work(directory, scratch=None, timeout=None, log=print, max_tasks=None):
    """Claim and run unfinished tasks until none is left; returns the number attempted here
    (failed attempts are listed in ``failures.log``, their JEWEL output under ``failed/``)."""
    directory = Path(directory)
    doc = load(directory)
    cfg = dict(doc["config"])
    done = 0
    for t in doc["tasks"]:
        if max_tasks is not None and done >= max_tasks:
            break
        if (directory / f"{t['name']}.json").exists():
            continue
        lock = directory / "claims" / f"{t['name']}.lock"
        if not tasklocks.claim(lock):
            continue
        try:
            if (directory / f"{t['name']}.json").exists():
                continue
            done += 1  # attempts count towards max_tasks
            try:
                run_task(cfg, t, directory, scratch=scratch, timeout=timeout, log=log)
            except Exception as e:  # recorded, the farm continues with other tasks
                with open(directory / "failures.log", "a") as f:
                    f.write(json.dumps({"task": t["name"], "error": repr(e), "host": socket.gethostname(),
                                        "time": time.time()}) + "\n")
                log(f"{t['name']}: FAILED {e!r}")
        finally:
            tasklocks.release(lock)
    return done


def write_index(directory):
    """Merge task manifests into ``index_<split>.json`` (several chunks per point allowed)."""
    directory = Path(directory)
    doc = load(directory)
    mans = []
    for t in doc["tasks"]:
        p = directory / f"{t['name']}.json"
        if not p.exists():
            continue
        with open(p) as f:
            m = json.load(f)
        if file_sha256(directory / m["files"]["jets"]) != m["files"]["jets_sha256"]:
            raise ValueError(f"{m['files']['jets']} does not match its manifest hash")
        mans.append(m)
    index = {"schema": "jewel-index-v1", "split": doc["split"], "backend": doc["backend"],
             "shards": [{"file": m["files"]["jets"], "sha256": m["files"]["jets_sha256"],
                         "events": m["files"]["events"], "point_id": m["task"]["point_id"],
                         "chunk": m["task"]["chunk"], "n_jets": m["jets_selected"]} for m in mans],
             "tasks_done": len(mans), "tasks_total": len(doc["tasks"]),
             "n_jets": int(sum(m["jets_selected"] for m in mans)),
             "events": int(sum(m["events_generated"] for m in mans)),
             "seconds_jewel": float(sum(m["seconds_jewel"] for m in mans))}
    with open(directory / f"index_{doc['split']}.json", "w") as f:
        json.dump(index, f, indent=1)
    return index


def sampling_record(sampling):
    """The shard's ``sampling_bias`` attribute of a weighted JEWEL task."""
    return {"generator": "jewel", "weighted": True, "wexpo": float(sampling["wexpo"]),
            "weight": "HepMC event weight PARI(10) = pthat^-wexpo"}


def _merge_one(job):
    """Write one merged shard from its source task shards (worker of ``merge``)."""
    src_dir, out, sources = Path(job["src"]), Path(job["out"]), job["sources"]
    parts, diags, attrs = [], [], None
    for s in sources:
        path = src_dir / s["file"]
        if file_sha256(path) != s["sha256"]:
            raise ValueError(f"{path} does not match the source index")
        with h5py.File(path, "r") as f:
            a = {k: f.attrs[k] for k in ("schema", "backend", "split", "toy_config_sha256")}
            a = {k: (v.item() if hasattr(v, "item") else v) for k, v in a.items()}
            a["sampling_bias"] = f.attrs.get("sampling_bias")
            if attrs is None:
                attrs = a
                feat, width = list(f.attrs["feature_names"]), int(f.attrs["max_constituents"])
            elif a != attrs or int(f.attrs["max_constituents"]) != width:
                raise ValueError(f"{path}: attributes differ from the first source")
            parts.append({k: f[k][...] for k in ("data/x", "data/mask", "data/n_constituents", "global",
                                                 "event_id", "point_id", "weight", "pthat") if k in f})
        with h5py.File(path.with_name(path.name[:-3] + ".diag.h5"), "r") as f:
            diags.append({k: f[k][...] for k in f.keys()})
    cat = lambda k: np.concatenate([q[k] for q in parts])  # noqa: E731
    x = cat("data/x")
    n = x.shape[0]
    chunk = lambda c: tuple(min(a, b) for a, b in zip(c, x.shape)) if n else None  # noqa: E731
    with h5py.File(out, "w", track_order=False) as f:
        _dataset(f, "data/x", x, compression="gzip", chunks=chunk(X_CHUNK))
        _dataset(f, "data/mask", cat("data/mask"), compression="gzip", chunks=chunk(X_CHUNK)[:2] if n else None)
        _dataset(f, "data/n_constituents", cat("data/n_constituents"))
        g = _dataset(f, "global", cat("global"))
        g.attrs["global_names"] = list(GLOBAL_NAMES)
        _dataset(f, "event_id", cat("event_id"))
        _dataset(f, "point_id", cat("point_id"))
        if attrs["sampling_bias"] is not None:  # weights rescaled to sum to each point's jets
            w, pid_ = cat("weight").astype(np.float64), cat("point_id")
            for p in np.unique(pid_):
                s = pid_ == p
                w[s] *= s.sum() / w[s].sum()
            _dataset(f, "weight", w)
            _dataset(f, "pthat", cat("pthat"))
        for k, v in attrs.items():
            if v is not None:
                f.attrs[k] = v
        f.attrs["feature_names"] = feat
        f.attrs["parameter_names"] = list(PARAMETERS)
        f.attrs["max_constituents"] = width
        f.attrs["merged_from"] = MERGED_SCHEMA
    dpath = out.with_suffix(".diag.h5")
    with h5py.File(dpath, "w") as f:
        for k in diags[0]:
            _dataset(f, k, np.concatenate([d[k] for d in diags]))
        f.attrs["access"] = "diagnostic only; never an input to training or fitting"
    pid = cat("point_id")
    theta = cat("global")[:, :len(PARAMETERS)]
    points = [{"point_id": int(p), "theta": theta[pid == p][0].tolist(), "n_jets": int((pid == p).sum()),
               "n_tasks": sum(1 for s in sources if s["point_id"] == p)} for p in sorted({s["point_id"] for s in sources})]
    if attrs["sampling_bias"] is not None:
        w = cat("weight")
        for q in points:
            ws = w[pid == q["point_id"]]
            q["ess"] = float(ws.sum() ** 2 / np.sum(ws ** 2)) if ws.size else 0.0
    man = {"schema": MERGED_SCHEMA, "file": out.name, "sha256": file_sha256(out), "diag_file": dpath.name,
           "diag_sha256": file_sha256(dpath), "split": attrs["split"], "backend": attrs["backend"],
           "toy_config_sha256": attrs["toy_config_sha256"], "n_jets": int(n), "points": points,
           "sampling_bias": None if attrs["sampling_bias"] is None else json.loads(attrs["sampling_bias"]),
           "sources": sources, "host": host_info()}
    with open(out.with_suffix(".json"), "w") as f:
        json.dump(man, f, indent=1)
    return man


def merge(directory, out, shards, workers=1, log=print):
    """Merge the task shards of a split into ``shards`` point-major files under ``out``.

    Points are kept whole and in order (all chunks of a point in one file, chunk order),
    split into ``shards`` contiguous blocks of points. Every source is hash-checked against
    the split's ``index_<split>.json``. Returns the merged index.
    """
    directory, out = Path(directory), Path(out)
    src_index = next(directory.glob("index_*.json"))
    with open(src_index) as f:
        idx = json.load(f)
    split = idx["split"]
    by_point = {}
    for s in idx["shards"]:
        by_point.setdefault(s["point_id"], []).append(s)
    pids = sorted(by_point)
    blocks = [b for b in np.array_split(np.array(pids), min(shards, len(pids))) if b.size]
    out.mkdir(parents=True, exist_ok=True)
    jobs = [{"src": str(directory), "out": str(out / f"{split}_{k:05d}.h5"),
             "sources": [s for p in b for s in sorted(by_point[int(p)], key=lambda s: s["chunk"])]}
            for k, b in enumerate(blocks)]
    t0 = time.time()
    if workers > 1:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(workers) as pool:
            mans = pool.map(_merge_one, jobs, chunksize=1)
    else:
        mans = [_merge_one(j) for j in jobs]
    index = {"schema": MERGED_SCHEMA + "-index", "split": split, "backend": idx["backend"],
             "source_dir": str(directory), "source_index": src_index.name,
             "source_index_sha256": file_sha256(src_index),
             "shards": [{"file": m["file"], "sha256": m["sha256"], "n_jets": m["n_jets"],
                         "point_ids": [q["point_id"] for q in m["points"]]} for m in mans],
             "n_points": len(pids), "n_jets": int(sum(m["n_jets"] for m in mans)),
             "n_tasks": len(idx["shards"]), "events": idx.get("events"),
             "sampling_bias": [json.loads(b) for b in sorted({json.dumps(m["sampling_bias"], sort_keys=True)
                                                              for m in mans})],
             "host": host_info()}
    if index["n_jets"] != idx["n_jets"]:
        raise ValueError(f"merged {index['n_jets']} jets, source index has {idx['n_jets']}")
    with open(out / f"index_{split}.json", "w") as f:
        json.dump(index, f, indent=1)
    log(f"merged {len(idx['shards'])} {split} tasks -> {len(mans)} shards, {index['n_jets']} jets, "
        f"{time.time() - t0:.0f}s")
    return index


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    # "test" has its own NJOB base and stream (cards.SPLIT_STREAM, randomness.njob_base) and is
    # planned from explicit --theta (the reserved settings), not from a design. Planning it as
    # "probe" instead would reuse the probe NJOB base and hand it the probes' random seeds.
    p.add_argument("--split", required=True,
                   choices=["train", "validation", "fidelity", "test", "probe"])
    p.add_argument("--out", required=True)
    p.add_argument("--config", default=str(cards.CONFIG))
    p.add_argument("--chunks", type=int, default=1)
    p.add_argument("--points", type=int, nargs="*", default=None)
    p.add_argument("--theta", type=float, nargs="+", default=None, help="explicit rows b TI TAUI ... (probe)")
    p.add_argument("--nevent", type=int, default=None)
    p.add_argument("--target-jets", type=int, default=None,
                   help="chunks per point from this jet target and the acceptance in --acceptance-from")
    p.add_argument("--acceptance-from", default=None, help="indexed split directory at the same points")
    p.add_argument("--margin", type=float, default=1.05)
    p.add_argument("--no-events", action="store_true", help="no raw event stores (jets and diagnostics only)")
    p.add_argument("--keep-hepmc", nargs="*", default=[])
    p.add_argument("--extra", default=None, help="JSON object of extra main-card lines")
    p.add_argument("--wexpo", type=float, default=None,
                   help="weighted events: WEIGHTED T, WEXPO K (hard process sampled with pT-hat^K, "
                        "event weight pT-hat^-K; per-jet weights in the shards)")
    w = sub.add_parser("work")
    w.add_argument("--dir", required=True)
    w.add_argument("--scratch", default=None)
    w.add_argument("--timeout", type=float, default=None, help="seconds per task")
    w.add_argument("--max-tasks", type=int, default=None)
    u = sub.add_parser("unlock-stale")
    u.add_argument("--dir", required=True)
    t = sub.add_parser("topup", help="add tasks so every point reaches a jet target")
    t.add_argument("--dir", required=True)
    t.add_argument("--target-jets", type=int, required=True)
    t.add_argument("--margin", type=float, default=1.05)
    i = sub.add_parser("index")
    i.add_argument("--dir", required=True)
    m = sub.add_parser("merge", help="merge a split's task shards into a few point-major shards")
    m.add_argument("--dir", required=True, help="split directory with index_<split>.json")
    m.add_argument("--out", required=True, help="merged directory (may hold several splits)")
    m.add_argument("--shards", type=int, default=64)
    m.add_argument("--workers", type=int, default=1)
    s = sub.add_parser("inspect")
    s.add_argument("--config", default=str(cards.CONFIG))
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        cfg = cards.load_config(a.config)
        theta = None if a.theta is None else np.asarray(a.theta).reshape(-1, 3)
        chunks = a.chunks
        if a.target_jets:
            chunks = chunks_for_target(a.acceptance_from, a.target_jets,
                                       int(a.nevent or cfg["design"]["events_per_task"]), a.margin)
            if a.points is not None:
                chunks = {p: chunks[p] for p in a.points}
            a.points = sorted(chunks)
        doc = plan(cfg, a.split, a.out, chunks, a.points, theta, a.keep_hepmc,
                   json.loads(a.extra) if a.extra else None, a.nevent, keep_events=not a.no_events,
                   sampling=None if a.wexpo is None else {"wexpo": a.wexpo})
        print(f"{len(doc['tasks'])} tasks -> {a.out}/tasks.json")
    elif a.cmd == "work":
        print(f"worker ran {work(a.dir, a.scratch, a.timeout, max_tasks=a.max_tasks)} tasks")
    elif a.cmd == "unlock-stale":
        tasklocks.unlock_stale(sorted(Path(a.dir).glob("claims/*.lock")))
    elif a.cmd == "topup":
        topup(a.dir, a.target_jets, a.margin)
    elif a.cmd == "merge":
        merge(a.dir, a.out, a.shards, a.workers)
    elif a.cmd == "index":
        idx = write_index(a.dir)
        print(f"{idx['tasks_done']}/{idx['tasks_total']} tasks, {idx['n_jets']} jets, {idx['events']} events")
    else:
        print(json.dumps({k: v for k, v in cards.inspect_installation(cards.load_config(a.config)).items()
                          if k != "patches"}, indent=1))


if __name__ == "__main__":
    main()
