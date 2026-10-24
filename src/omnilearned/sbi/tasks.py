"""Exclusive task claims with lock files, shared by closure campaigns and generation farms.

A worker claims a task by creating its lock file exclusively (``O_CREAT | O_EXCL``), so of
several concurrent workers exactly one wins. Workers never reclaim stale locks themselves:
concurrent reclaimers race (measured: two fits ran twice at the start of an allocation).
``unlock_stale`` removes locks whose Slurm job has ended, whose local process is gone or that
are older than ``max_age``; run it once, from a single process, before workers start.
"""

import json
import os
import socket
import subprocess
import time
from pathlib import Path


def _job_alive(job):
    try:
        r = subprocess.run(["squeue", "-h", "-j", str(job), "-o", "%T"], capture_output=True,
                           text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return True  # cannot tell: treat as alive
    return r.returncode == 0 and r.stdout.strip() in ("RUNNING", "COMPLETING", "PENDING", "CONFIGURING")


def lock_is_stale(info, max_age=5 * 3600.0):
    """A claim is stale when its Slurm job is gone, its local process is gone, or it is too old."""
    if time.time() - info.get("time", 0) > max_age:
        return True
    if info.get("job"):
        return not _job_alive(info["job"])
    if info.get("host") == socket.gethostname():
        try:
            os.kill(int(info["pid"]), 0)
            return False
        except (OSError, ValueError):
            return True
    return False


def read_lock(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        return {"time": 0}  # unreadable or half-written: treat as old


def claim(lock, extra=None):
    """Create ``lock`` exclusively. Returns True if this process now holds it."""
    lock = Path(lock)
    lock.parent.mkdir(parents=True, exist_ok=True)
    info = {"job": os.environ.get("SLURM_JOB_ID"), "host": socket.gethostname(), "pid": os.getpid(),
            "time": time.time(), "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), **(extra or {})}
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as f:
        json.dump(info, f)
    return True


def release(lock):
    Path(lock).unlink(missing_ok=True)


def unlock_stale(locks, log=print, max_age=5 * 3600.0):
    """Remove the stale ones among ``locks`` (paths). Rename first, so a lock re-created
    meanwhile is never deleted by mistake. Returns the number removed."""
    n = 0
    for lock in locks:
        lock = Path(lock)
        info = read_lock(lock)
        if info is None or not lock_is_stale(info, max_age):
            continue
        grave = lock.with_name(f"{lock.name}.stale.{os.getpid()}.{time.time_ns()}")
        try:
            os.rename(lock, grave)
        except FileNotFoundError:
            continue
        if read_lock(grave) != info:
            os.rename(grave, lock)
            continue
        grave.unlink(missing_ok=True)
        n += 1
    log(f"removed {n} stale claim locks")
    return n
