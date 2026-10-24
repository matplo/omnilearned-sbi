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
from omnilearned.sbi.toy.datasets import host_info, make_design, write_shard
from omnilearned.sbi.toy.design import STREAMS
from omnilearned.sbi.toy.simulator import PARAMETERS

EVENTS_SCHEMA = "jewel-events-v1"
TASKS_SCHEMA = "jewel-tasks-v1"


def task_name(split, point_id, chunk):
    return f"{split}_p{int(point_id):05d}_c{int(chunk):03d}"


def box(cfg):
    return {p: tuple(cfg["design"]["box"][p]) for p in PARAMETERS}


def design_points(cfg, split):
    """Points of a split: the Cartesian grid (train), Sobol (validation)."""
    if split == "train":
        axes = [np.asarray(cfg["design"][p], float) for p in PARAMETERS]
        return np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, len(PARAMETERS))
    if split == "validation":
        pts, _ = make_design(int(cfg["design"]["validation_points"]), "validation", cfg, box=box(cfg))
        return pts
    raise ValueError(f"no design for split {split}")


def plan(cfg, split, out, chunks=1, points=None, theta=None, keep_hepmc=(), extra=None,
         nevent=None):
    """Write ``tasks.json``: every (point, chunk) of the split (or explicit ``theta`` rows)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    pts = np.atleast_2d(np.asarray(theta, float)) if theta is not None else design_points(cfg, split)
    ids = list(range(len(pts))) if points is None else list(points)
    nev = int(nevent or cfg["design"]["events_per_task"])
    tl = []
    for pid in ids:
        for c in range(int(chunks)):
            tl.append({"name": task_name(split, pid, c), "split": split, "point_id": int(pid),
                       "chunk": c, "theta": [float(v) for v in pts[pid]], "nevent": nev,
                       "njob": cards.njob(cfg, split, pid, c),
                       "keep_hepmc": task_name(split, pid, c) in set(keep_hepmc),
                       "extra_cards": extra or {}})
    doc = {"schema": TASKS_SCHEMA, "backend": cfg["backend"], "split": split,
           "parameters": list(PARAMETERS), "tasks": tl, "config": cfg, "host": host_info()}
    with open(out / "tasks.json", "w") as f:
        json.dump(doc, f, indent=1)
    return doc


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
                                                geom, tables_dir=".", extra=task.get("extra_cards"))
        (work / "params.dat").write_text(params)
        (work / "medium.dat").write_text(medium)
        exe = cards.inspect_installation(cfg)["executable"] if not cfg["jewel"].get("executable") \
            else cfg["jewel"]["executable"]
        cmd = list(cfg["jewel"].get("launcher", [])) + [exe, "params.dat"]
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
            raise RuntimeError(f"JEWEL {name}: rc={rc}, no HepMC; log tail: {logtail[-5:]}")
        sel = jets.selection_from_config(cfg)
        t1 = time.time()
        events, recs, diags, pre_reject = [], [], [], {r: 0 for r in jets.REASONS}
        all_events = list(hepmc.read_events(hep)) if hep.exists() else []
        if rc != 0 and all_events:
            all_events = all_events[:-1]  # the last event of an interrupted run may be incomplete
        for ev in all_events:
            events.append(ev)
            rec, why, d = jets.event_jet(ev, cfg, sel)
            if rec is None:
                pre_reject[why] += 1
                continue
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
        cfg_sha = cfg.get("_sha256", "unknown")  # of the config the task list was planned from
        sha, dsha = write_shard(out_dir / f"{name}.h5", [(task["point_id"], task["theta"], rec)],
                                task["split"], cfg["backend"], cfg_sha)
        esha = _write_events(out_dir / f"{name}.events.h5", events, stream, task["point_id"],
                             task["chunk"], task["nevent"])
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
               "acceptance": len(keep) / max(1, len(events)), "rejections": reject,
               "n_constituents_range": [int(nc.min()), int(nc.max())] if len(nc) else None,
               "seconds_jewel": t_gen, "seconds_processing": t_proc,
               "seconds_per_event": t_gen / max(1, len(events)),
               "files": {"jets": f"{name}.h5", "jets_sha256": sha, "diag": f"{name}.diag.h5",
                         "diag_sha256": dsha, "events": f"{name}.events.h5", "events_sha256": esha},
               "events_bytes": os.path.getsize(out_dir / f"{name}.events.h5"),
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
    """Claim and run unfinished tasks until none is left; returns the number run here."""
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
            try:
                run_task(cfg, t, directory, scratch=scratch, timeout=timeout, log=log)
                done += 1
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--split", required=True, choices=["train", "validation", "probe"])
    p.add_argument("--out", required=True)
    p.add_argument("--config", default=str(cards.CONFIG))
    p.add_argument("--chunks", type=int, default=1)
    p.add_argument("--points", type=int, nargs="*", default=None)
    p.add_argument("--theta", type=float, nargs="+", default=None, help="explicit rows b TI TAUI ... (probe)")
    p.add_argument("--nevent", type=int, default=None)
    p.add_argument("--keep-hepmc", nargs="*", default=[])
    p.add_argument("--extra", default=None, help="JSON object of extra main-card lines")
    w = sub.add_parser("work")
    w.add_argument("--dir", required=True)
    w.add_argument("--scratch", default=None)
    w.add_argument("--timeout", type=float, default=None, help="seconds per task")
    w.add_argument("--max-tasks", type=int, default=None)
    u = sub.add_parser("unlock-stale")
    u.add_argument("--dir", required=True)
    i = sub.add_parser("index")
    i.add_argument("--dir", required=True)
    s = sub.add_parser("inspect")
    s.add_argument("--config", default=str(cards.CONFIG))
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        cfg = cards.load_config(a.config)
        theta = None if a.theta is None else np.asarray(a.theta).reshape(-1, 3)
        doc = plan(cfg, a.split, a.out, a.chunks, a.points, theta, a.keep_hepmc,
                   json.loads(a.extra) if a.extra else None, a.nevent)
        print(f"{len(doc['tasks'])} tasks -> {a.out}/tasks.json")
    elif a.cmd == "work":
        print(f"worker ran {work(a.dir, a.scratch, a.timeout, max_tasks=a.max_tasks)} tasks")
    elif a.cmd == "unlock-stale":
        tasklocks.unlock_stale(sorted(Path(a.dir).glob("claims/*.lock")))
    elif a.cmd == "index":
        idx = write_index(a.dir)
        print(f"{idx['tasks_done']}/{idx['tasks_total']} tasks, {idx['n_jets']} jets, {idx['events']} events")
    else:
        print(json.dumps({k: v for k, v in cards.inspect_installation(cards.load_config(a.config)).items()
                          if k != "patches"}, indent=1))


if __name__ == "__main__":
    main()
