"""Closure campaigns: pseudo-data ensembles, parallel sealed fits, truth-blind report.

A campaign is one setting × one event count × K ensembles, from one source:

* ``source = "toy"``: pseudo data drawn directly from the simulator at a reserved
  setting (closure evidence; rule 1);
* ``source = "surrogate"``: the labelled self-closure control (lesson 8), drawn from
  the surrogate itself at the same truth and selected with the toy's selection. A
  diagnostic, never closure evidence.

Campaign directory::

    campaign.json          spec, tasks, observation and fit-config hashes
    obs/obs_KKKK.h5        truth-free observation files
    fits/fit_KKKK.toml     fit configs (observations, checkpoint, grid, prior, numerics)
    fits/fit_KKKK/result.{json,npz}, fits/fit_KKKK/seal.json
    truth/truth_KKKK.json  read only by ``report``, after the seal verifies

Commands (``python -m omnilearned.sbi.toy.closure``):

* ``prepare``: draw the ensembles, write observations, truth and fit configs. The
  setting is named by its ``[holdout]`` key; only this driver reads its value.
* ``fit``: run the fits of worker ``w`` of ``W`` (tasks ``w::W``), one process per GPU
  (or several per GPU), resumable; each fit is sealed (sha256 of its outputs, config,
  observations and checkpoint). Fits never read ``truth/``.
* ``report``: verify every seal, then open truth: pulls, marginal 68/95% coverage, and
  the joint level of the truth (posterior mass at higher density than at the truth,
  with log L at the truth evaluated as one extra candidate with the fit's numerics and
  probes). Aggregates with Clopper–Pearson bounds and a KS test of the levels; with
  ``--control`` the pull differences against a self-closure control campaign.
* ``acceptance`` / ``acceptance-merge``: the surrogate's selection acceptance table
  Pr_q(S | theta) (common random numbers), sharded over GPUs; used as ``log_norm_fn``.
  ``acceptance-merge --per-n-out`` also writes the per-N table Pr_q(S | N, theta); a spec
  with ``selection_normalization = "per_n"`` and that table as ``acceptance_table`` fits
  with sum_i log Pr_q(S | N_i, theta) and draws the control with same-N redraws (plan S1).
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from omnilearned.sbi.observations import file_sha256
from omnilearned.sbi.toy.config import V2_CONFIG, load_toy_config
from omnilearned.sbi.toy.datasets import host_info, make_observations
from omnilearned.sbi.toy.design import STREAMS, reserved_settings
from omnilearned.sbi.toy.simulator import PARAMETERS

CAMPAIGN_SCHEMA = "toy-closure-campaign-v1"
SEAL_SCHEMA = "toy-closure-seal-v1"
SELF_CLOSURE_STREAM = STREAMS["self_closure"]
SETTING_SPLIT = {"development_validation": "development"}  # all other reserved settings: test


def _toml_value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k} = {_toml_value(x)}" for k, x in v.items()) + "}"
    raise TypeError(f"cannot write {type(v)} to TOML")


def dump_toml(doc):
    """Minimal TOML writer: top-level scalars, then one table per dict value."""
    lines = [f"{k} = {_toml_value(v)}" for k, v in doc.items() if not isinstance(v, dict)]
    for k, v in doc.items():
        if isinstance(v, dict):
            lines += ["", f"[{k}]"] + [f"{kk} = {_toml_value(vv)}" for kk, vv in v.items()]
    return "\n".join(lines) + "\n"


def _load_toml(path):
    try:
        import tomllib
    except ModuleNotFoundError:  # Python < 3.11
        import tomli as tomllib
    with open(path, "rb") as f:
        return tomllib.load(f)


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------

def _surrogate_ensemble(sur, sel, theta, n, seed, index, nsteps=128, redraw=False):
    """Self-closure control: n selected clouds drawn from the surrogate at theta.

    ``redraw``: rejected clouds are drawn again at the same N (per-N normalization), so the
    control follows the per-N selected density."""
    from omnilearned.sbi.toy.sample_select import sample_and_select
    from omnilearned.sbi.toy.simulator import make_event_id

    xs, ms, draws, k = [], [], 0, 0
    while sum(len(x) for x in xs) < n:
        if draws > 100 * n:  # acceptance below 1%: not a usable surrogate here
            raise RuntimeError(f"surrogate acceptance too low at {theta}: "
                               f"{sum(len(x) for x in xs)} of {draws} draws selected")
        x, m, rep = sample_and_select(sur, theta, int(1.25 * n) + 16, sel, seed=seed + 7919 * k,
                                      nsteps=nsteps, redraw=redraw)
        xs.append(x)
        ms.append(m)
        draws += rep["generated"]
        k += 1
    w = max(x.shape[1] for x in xs)
    pad = lambda v: np.pad(v, [(0, 0), (0, w - v.shape[1])] + [(0, 0)] * (v.ndim - 2))  # noqa: E731
    x, m = np.concatenate([pad(v) for v in xs])[:n], np.concatenate([pad(v) for v in ms])[:n]
    ids = make_event_id(SELF_CLOSURE_STREAM, index, np.arange(n))
    return {"x": x.astype(np.float64), "mask": m, "event_id": ids}, {"generated": draws}


def prepare(spec, out, config=V2_CONFIG, log=print):
    """Write observations, truth and fit configs of a campaign; returns the manifest."""
    from omnilearned.sbi.fit import check_fit_config
    from omnilearned.sbi.toy.records import write_observation_file, write_truth

    out = Path(out)
    if (out / "campaign.json").exists():
        raise FileExistsError(f"{out} already holds a campaign")
    for d in ("obs", "fits", "truth"):
        (out / d).mkdir(parents=True, exist_ok=True)
    cfg, _, sel = load_toy_config(config)
    theta = reserved_settings(cfg)[spec["setting"]]
    source = spec.get("source", "toy")
    if source not in ("toy", "surrogate"):
        raise ValueError("source must be 'toy' or 'surrogate'")
    n, K, off = int(spec["n_events"]), int(spec["n_ensembles"]), int(spec.get("index_offset", 0))
    ck = str(Path(spec["checkpoint"]).resolve())
    ck_sha = file_sha256(ck)
    sur = None
    if source == "surrogate":
        from omnilearned.sbi.flow import FlowSurrogate

        sur = FlowSurrogate.load(ck, device=spec.get("device", "cpu"))
    table = spec.get("acceptance_table")
    per_n = spec.get("selection_normalization", "per_theta") == "per_n"
    if spec.get("selection_normalization", "per_theta") not in ("per_theta", "per_n"):
        raise ValueError("selection_normalization must be 'per_theta' or 'per_n'")
    if per_n and not table:
        raise ValueError("per-N selection normalization needs an acceptance_table (per-N table)")
    if table and list(spec["fit"]["grid"]) != list(PARAMETERS):
        raise ValueError(f"with an acceptance table the grid must be ordered {list(PARAMETERS)}")
    tasks = []
    for k in range(K):
        obs = out / "obs" / f"obs_{k:04d}.h5"
        truth = out / "truth" / f"truth_{k:04d}.json"
        if source == "toy":
            split = SETTING_SPLIT.get(spec["setting"], "test")
            sha, _ = make_observations(obs, truth, theta, n, split, off + k, config)
        else:
            rec, info = _surrogate_ensemble(sur, sel, theta, n, int(spec.get("seed", 0)) + off + k, off + k,
                                            redraw=per_n)
            sha = write_observation_file(obs, rec, attrs={"source": "surrogate self-closure control"})
            write_truth(truth, {"observations": str(obs), "observations_sha256": sha,
                                "theta": [float(v) for v in theta], "source": "surrogate",
                                "index": off + k, **info, "host": host_info()})
        fit_cfg = {"schema": "omnilearned-sbi-fit-v1", "checkpoint": ck, "checkpoint_sha256": ck_sha,
                   "observations": str(obs.resolve()), "observations_sha256": sha,
                   "output_prefix": str((out / "fits" / f"fit_{k:04d}" / "result").resolve()),
                   "device": spec.get("device", "cuda"), "mode": "fixed_vector"}
        if table:
            fit_cfg["log_norm_table"] = str(Path(table).resolve())
            fit_cfg["log_norm_table_sha256"] = file_sha256(table)
            fit_cfg["log_norm_mode"] = "per_n" if per_n else "per_theta"
        fit_cfg.update({k2: v for k2, v in spec["fit"].items()})
        check_fit_config(fit_cfg)
        path = out / "fits" / f"fit_{k:04d}.toml"
        path.write_text(dump_toml(fit_cfg))
        tasks.append({"task": k, "observations": obs.name, "observations_sha256": sha,
                      "fit_config": path.name, "fit_config_sha256": file_sha256(path)})
        log(f"  ensemble {k}: {n} events ({source}) -> {obs.name}")
    manifest = {"schema": CAMPAIGN_SCHEMA, "spec": spec, "source": source, "n_events": n,
                "n_ensembles": K, "checkpoint_sha256": ck_sha, "tasks": tasks,
                "toy_config_sha256": file_sha256(config), "host": host_info(),
                "label": "self-closure control (diagnostic, not closure evidence)"
                if source == "surrogate" else "direct-simulator closure"}
    with open(out / "campaign.json", "w") as f:
        json.dump(manifest, f, indent=1, default=str)
    return manifest


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------

def _log_norm(cfg):
    if not cfg.get("log_norm_table"):
        return None
    from omnilearned.sbi.toy.selection import PerNLogNorm, TableLogNorm

    if file_sha256(cfg["log_norm_table"]) != cfg.get("log_norm_table_sha256"):
        raise ValueError("acceptance table does not match the fit configuration")
    if cfg.get("log_norm_mode", "per_theta") == "per_n":
        if cfg.get("mode", "fixed_vector") != "fixed_vector":
            raise ValueError("per-N selection normalization is implemented for fixed-vector fits")
        from omnilearned.sbi.observations import read_observations

        obs = read_observations(cfg["observations"])
        return PerNLogNorm(cfg["log_norm_table"]).bind(np.asarray(obs.mask).sum(1))
    return TableLogNorm(cfg["log_norm_table"])


def _seal_files(prefix):
    return [Path(f"{prefix}.json"), Path(f"{prefix}.npz")]


def fit_task(out, k, surrogate=None, log=print, cache_every=300.0):
    """Run and seal fit ``k`` of a campaign (skipped when already sealed).

    Per-event log densities are cached in ``fits/fit_KKKK/logp_cache.npz``
    (flushed every ``cache_every`` s, on SIGTERM and at the end), so a fit cut by
    the end of an allocation resumes where it stopped.
    """
    import signal

    from omnilearned.sbi.fit import LogpCache, load_fit_config, run_fit

    out = Path(out)
    seal = out / "fits" / f"fit_{k:04d}" / "seal.json"
    if seal.exists():
        return None
    cfg = load_fit_config(out / "fits" / f"fit_{k:04d}.toml")
    seal.parent.mkdir(parents=True, exist_ok=True)
    cache = LogpCache(seal.parent / "logp_cache.npz",
                      {"fit_config_sha256": cfg["_config_sha256"],
                       "observations_sha256": cfg["observations_sha256"],
                       "checkpoint_sha256": cfg["checkpoint_sha256"]}, every=cache_every)
    if cache.data:
        log(f"fit {k}: resuming with {len(cache.data)} cached (candidate, event chunk) entries")

    def on_term(signum, frame):  # end of allocation: keep what was computed
        raise SystemExit(f"fit {k}: signal {signum}")

    prev = signal.signal(signal.SIGTERM, on_term)
    t0 = time.time()
    try:
        doc = run_fit(cfg, log_norm_fn=_log_norm(cfg), log=log, surrogate=surrogate,
                      logp_cache=cache)
    finally:
        cache.flush()
        signal.signal(signal.SIGTERM, prev)
    files = _seal_files(cfg["output_prefix"])
    rec = {"schema": SEAL_SCHEMA, "task": k, "fit_config_sha256": cfg["_config_sha256"],
           "observations_sha256": cfg["observations_sha256"],
           "checkpoint_sha256": cfg["checkpoint_sha256"],
           "log_norm_table_sha256": cfg.get("log_norm_table_sha256"),
           "outputs": {p.name: file_sha256(p) for p in files},
           "complete": bool(doc.get("diagnostics", {}).get("complete", True)
                            and doc.get("diagnostics", {}).get("support_closed", True)),
           "seconds": time.time() - t0, "sealed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "host": host_info()}
    with open(seal, "w") as f:
        json.dump(rec, f, indent=1)
    log(f"fit {k} sealed ({rec['seconds']:.0f}s, complete={rec['complete']})")
    return rec


def worker_tasks(n_tasks, worker, n_workers):
    return list(range(worker, n_tasks, n_workers))


def _lock(out, k):
    return Path(out) / "fits" / f"fit_{k:04d}" / "claim.lock"


def lock_is_stale(info, max_age=5 * 3600.0):
    from omnilearned.sbi import tasks

    return tasks.lock_is_stale(info, max_age)


def claim(out, k):
    """Atomically claim fit k (exclusive lock file, ``omnilearned.sbi.tasks``). True if ours.

    Workers never reclaim stale locks themselves; ``unlock_stale`` does, once, before
    workers start.
    """
    from omnilearned.sbi import tasks

    return tasks.claim(_lock(out, k))


def release(out, k):
    from omnilearned.sbi import tasks

    tasks.release(_lock(out, k))


def unlock_stale(dirs, log=print):
    """Remove stale claim locks of the given campaigns (run once, before workers start)."""
    from omnilearned.sbi import tasks

    return tasks.unlock_stale([p for d in dirs for p in sorted(Path(d).glob("fits/fit_*/claim.lock"))], log=log)


def run_claiming(dirs, log=print):
    """Fit unsealed tasks of one or more campaigns, interleaved, claiming each first.

    Returns the number of fits sealed by this worker. Stops when nothing is left to
    claim (other workers may still be running theirs).
    """
    queue = []
    for d in dirs:
        with open(Path(d) / "campaign.json") as f:
            queue.append([(d, t["task"]) for t in json.load(f)["tasks"]])
    order = [x for grp in zip(*queue) for x in grp] + [x for grp in queue for x in grp[min(map(len, queue)):]]
    done = 0
    for d, k in order:
        if (Path(d) / "fits" / f"fit_{k:04d}" / "seal.json").exists() or not claim(d, k):
            continue
        try:
            if fit_task(d, k, log=log) is not None:
                done += 1
        finally:
            release(d, k)
    return done


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def verify_seal(out, k):
    """(ok, reason) for fit k: seal present and every hash still matching."""
    from omnilearned.sbi.fit import load_fit_config

    out = Path(out)
    seal = out / "fits" / f"fit_{k:04d}" / "seal.json"
    if not seal.exists():
        return False, "not sealed"
    with open(seal) as f:
        s = json.load(f)
    cfg = load_fit_config(out / "fits" / f"fit_{k:04d}.toml")
    if cfg["_config_sha256"] != s["fit_config_sha256"]:
        return False, "fit config changed after the seal"
    if file_sha256(cfg["observations"]) != s["observations_sha256"]:
        return False, "observations changed after the seal"
    for p in _seal_files(cfg["output_prefix"]):
        if not p.exists() or file_sha256(p) != s["outputs"].get(p.name):
            return False, f"{p.name} changed after the seal"
    return True, "ok"


def joint_level(res, logpost_truth):
    """Posterior mass on the grid at higher posterior density than at the truth."""
    lp = res.log_posterior()
    mass = res.posterior_mass()
    return float(mass[np.isfinite(lp) & (lp > logpost_truth)].sum())


def log_posterior_at(cfg, theta_by_name, surrogate, log_norm_fn=None, event_chunk=256):
    """Unnormalized log posterior at one point with the fit's numerics (same grid units)."""
    from omnilearned.sbi.likelihood import LikelihoodOptions
    from omnilearned.sbi.observations import read_observations

    names = list(cfg["grid"])
    cand = np.array([[theta_by_name[n] for n in names]])
    obs = read_observations(cfg["observations"])
    opts = LikelihoodOptions.from_dict(cfg.get("likelihood", {}))
    ll = 0.0
    for e0 in range(0, len(obs), event_chunk):
        sl = slice(e0, e0 + event_chunk)
        lp, _ = surrogate.log_prob(obs.x[sl], obs.mask[sl], cand, obs.event_id[sl], opts,
                                   candidate_names=names)
        ll += float(lp.sum())
    if log_norm_fn is not None:
        ll -= len(obs) * float(np.asarray(log_norm_fn(cand, names))[0])
    bounds = cfg.get("prior", {}).get("bounds")
    from omnilearned.sbi.scan import Grid

    grid = Grid.from_spec(cfg["grid"])
    vol, inside = 1.0, True
    for k, n in enumerate(names):
        lo, hi = (grid.axes[k][0], grid.axes[k][-1]) if bounds is None else bounds[n]
        vol *= (hi - lo) if hi > lo else 1.0
        inside &= lo <= theta_by_name[n] <= hi
    return ll - math.log(vol) if inside else -math.inf, ll


def clopper_pearson(k, n, cl=0.95):
    from scipy import stats

    a = (1 - cl) / 2
    lo = 0.0 if k == 0 else float(stats.beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - a, k + 1, n - k))
    return [lo, hi]


def ks_uniform(values):
    from scipy import stats

    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return {"n": 0, "ks_d": None, "ks_p": None}
    r = stats.kstest(v, "uniform")
    return {"n": int(v.size), "ks_d": float(r.statistic), "ks_p": float(r.pvalue)}


def report(out, device="cpu", control=None, log=print):
    """Verify seals, open truth, and summarize the campaign (writes report.json)."""
    from omnilearned.sbi.fit import load_fit_config
    from omnilearned.sbi.flow import FlowSurrogate
    from omnilearned.sbi.scan import ScanResult

    out = Path(out)
    with open(out / "campaign.json") as f:
        man = json.load(f)
    rows, refused = [], []
    sur = None
    for t in man["tasks"]:
        k = t["task"]
        ok, why = verify_seal(out, k)
        if not ok:
            refused.append({"task": k, "reason": why})
            continue
        cfg = load_fit_config(out / "fits" / f"fit_{k:04d}.toml")
        if sur is None:
            sur = FlowSurrogate.load(cfg["checkpoint"], device=device)
        res = ScanResult.load(cfg["output_prefix"])
        summ = res.summary()
        with open(out / "truth" / f"truth_{k:04d}.json") as f:  # opened only after the seal
            truth = dict(zip(PARAMETERS, json.load(f)["theta"]))
        lpt, llt = log_posterior_at(cfg, truth, sur, _log_norm(cfg))
        row = {"task": k, "truth": truth, "level": joint_level(res, lpt),
               "loglik_truth": llt, "max_loglik": summ["max_loglik"], "pull": {},
               "mean": summ["posterior_mean"], "std": summ["posterior_std"], "covered": {},
               "boundary": summ["boundary"],
               "evaluated": None if res.evaluated is None else int(np.sum(res.evaluated))}
        row["fixed"] = [n for n, ax in zip(res.grid.names, res.grid.axes) if len(ax) < 2]
        for n in res.grid.names:
            if n in row["fixed"]:  # a single-node axis is held fixed: no pull or coverage
                continue
            row["pull"][n] = (summ["posterior_mean"][n] - truth[n]) / summ["posterior_std"][n]
            row["covered"][n] = {lv: bool(iv[0] <= truth[n] <= iv[1])
                                 for lv, iv in summ["credible_intervals"][n].items()}
        rows.append(row)
        log(f"  fit {k}: level {row['level']:.3f} pulls "
            + " ".join(f"{n} {v:+.2f}" for n, v in row["pull"].items()))
    agg = _aggregate(rows)
    doc = {"campaign": man["spec"].get("name"), "label": man["label"], "n_fits": len(rows),
           "refused": refused, "fits": rows, "aggregate": agg, "host": host_info()}
    if control is not None:
        with open(Path(control) / "report.json") as f:
            ctl = json.load(f)["aggregate"]
        doc["versus_control"] = {
            n: {"difference": agg["mean_pull"][n] - ctl["mean_pull"][n],
                "se": math.hypot(agg["mean_pull_se"][n], ctl["mean_pull_se"][n]),
                "within_2se": abs(agg["mean_pull"][n] - ctl["mean_pull"][n])
                <= 2 * math.hypot(agg["mean_pull_se"][n], ctl["mean_pull_se"][n])}
            for n in agg["mean_pull"]}
    with open(out / "report.json", "w") as f:
        json.dump(doc, f, indent=1)
    return doc


def _aggregate(rows):
    if not rows:
        return {}
    names = list(rows[0]["pull"])
    n = len(rows)
    agg = {"mean_pull": {}, "mean_pull_se": {}, "coverage": {}, "levels": [r["level"] for r in rows]}
    for p in names:
        v = np.array([r["pull"][p] for r in rows])
        agg["mean_pull"][p] = float(v.mean())
        agg["mean_pull_se"][p] = float(v.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
        agg["coverage"][p] = {}
        for lv in rows[0]["covered"][p]:
            c = sum(r["covered"][p][lv] for r in rows)
            cp = clopper_pearson(c, n)
            agg["coverage"][p][lv] = {"covered": c, "n": n, "clopper_pearson_95": cp,
                                      "nominal_inside": bool(cp[0] <= float(lv) <= cp[1])}
    lv = np.array(agg["levels"])
    agg["joint"] = {}
    for lab in ("0.68", "0.95"):  # truth inside the joint credible region of that mass
        c = int((lv <= float(lab)).sum())
        cp = clopper_pearson(c, n)
        agg["joint"][lab] = {"covered": c, "n": n, "clopper_pearson_95": cp,
                             "nominal_inside": bool(cp[0] <= float(lab) <= cp[1])}
    agg["levels_ks"] = ks_uniform(lv)
    return agg


# ---------------------------------------------------------------------------
# acceptance table
# ---------------------------------------------------------------------------

def _axes(spec):
    out = []
    for s in spec:
        lo, hi, n = s.split(":")
        out.append(np.linspace(float(lo), float(hi), int(n)))
    return out


def acceptance_part(checkpoint, axes, n_samples, seed, nsteps, shard, num_shards, out, device="cuda",
                    log=print):
    """Acceptance at the grid points ``shard::num_shards`` (same base seed everywhere: CRN)."""
    from omnilearned.sbi.flow import FlowSurrogate
    from omnilearned.sbi.toy.selection import select_clouds

    sur = FlowSurrogate.load(checkpoint, device=device)
    _, _, sel = load_toy_config(V2_CONFIG)
    pts = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, len(axes))
    idx = np.arange(len(pts))[shard::num_shards]
    acc = np.full(len(pts), np.nan)
    width = sur.multiplicity.max_n + 1
    gen_n = np.full((len(pts), width), -1, np.int64)  # per-N counts for the per-N table (S1)
    acc_n = np.full((len(pts), width), -1, np.int64)
    for i in idx:
        x, m = sur.sample(pts[i], n_samples, None, seed=seed, nsteps=nsteps,
                          candidate_names=list(PARAMETERS), batch=1024)
        ok, _, _ = select_clouds(x.astype(np.float64), sel, m)
        acc[i] = ok.mean()
        cnt = m.sum(1)
        gen_n[i], acc_n[i] = np.bincount(cnt, minlength=width), np.bincount(cnt[ok], minlength=width)
        log(f"  acceptance {np.round(pts[i], 3).tolist()}: {acc[i]:.4f}")
    np.savez(out, acceptance=acc, generated_by_n=gen_n, accepted_by_n=acc_n,
             **{f"axis_{k}": a for k, a in enumerate(axes)},
             checkpoint_sha256=sur.metadata["checkpoint_sha256"], n_samples=n_samples, seed=seed,
             nsteps=nsteps)


def acceptance_merge(parts, out, per_n_out=None, n_knots=5):
    """Combine shard files into a ``selection.save_table`` table (all cells required);
    with ``per_n_out`` also the per-N table (``selection.save_per_n_table``)."""
    from omnilearned.sbi.toy.selection import save_per_n_table, save_table

    ds = [np.load(p) for p in parts]
    axes = [ds[0][f"axis_{k}"] for k in range(len(PARAMETERS))]
    acc = np.full(ds[0]["acceptance"].shape, np.nan)
    for d in ds:
        if str(d["checkpoint_sha256"]) != str(ds[0]["checkpoint_sha256"]):
            raise ValueError("acceptance shards from different checkpoints")
        have = np.isfinite(d["acceptance"])
        acc[have] = d["acceptance"][have]
    if np.isnan(acc).any():
        raise ValueError(f"{int(np.isnan(acc).sum())} acceptance cells missing")
    shape = tuple(len(a) for a in axes)
    save_table(out, axes, acc.reshape(shape), np.zeros(shape + (0,), np.int64),
               {"checkpoint_sha256": str(ds[0]["checkpoint_sha256"]), "n_samples": int(ds[0]["n_samples"]),
                "seed": int(ds[0]["seed"]), "nsteps": int(ds[0]["nsteps"]),
                "common_random_numbers": True, "parts": [str(p) for p in parts],
                "acceptance_range": [float(acc.min()), float(acc.max())], "host": host_info()})
    if per_n_out:
        if any("generated_by_n" not in d for d in ds):
            raise ValueError("acceptance shards without per-N counts (older code)")
        gen = np.full(ds[0]["generated_by_n"].shape, -1, np.int64)
        accn = gen.copy()
        for d in ds:
            have = d["generated_by_n"][:, 0] >= 0
            gen[have], accn[have] = d["generated_by_n"][have], d["accepted_by_n"][have]
        if (gen < 0).any():
            raise ValueError("per-N counts missing for some cells")
        w = gen.shape[-1]
        save_per_n_table(per_n_out, axes, gen.reshape(shape + (w,)), accn.reshape(shape + (w,)),
                         {"checkpoint_sha256": str(ds[0]["checkpoint_sha256"]),
                          "n_samples": int(ds[0]["n_samples"]), "seed": int(ds[0]["seed"]),
                          "nsteps": int(ds[0]["nsteps"]), "common_random_numbers": True,
                          "parts": [str(p) for p in parts], "host": host_info()}, n_knots=n_knots)
    return acc.reshape(shape)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--spec", required=True, help="campaign TOML (see module docstring)")
    p.add_argument("--out", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--dir", required=True, action="append",
                   help="campaign directory; repeat with --claim to interleave campaigns")
    f.add_argument("--claim", action="store_true",
                   help="claim unsealed fits dynamically (lock files) instead of a static split")
    f.add_argument("--worker", type=int, default=None, help="default: from SLURM_PROCID or 0")
    f.add_argument("--num-workers", type=int, default=1)
    f.add_argument("--tasks", type=int, nargs="*", default=None)
    u = sub.add_parser("unlock-stale", help="remove stale claim locks (before starting workers)")
    u.add_argument("--dir", required=True, action="append")
    r = sub.add_parser("report")
    r.add_argument("--dir", required=True)
    r.add_argument("--device", default="cpu")
    r.add_argument("--control", default=None, help="self-closure control campaign directory")
    a1 = sub.add_parser("acceptance")
    a1.add_argument("--checkpoint", required=True)
    a1.add_argument("--axis", action="append", required=True, help="min:max:n, in b, T_ref, tau_init order")
    a1.add_argument("--n-samples", type=int, default=8192)
    a1.add_argument("--seed", type=int, default=0)
    a1.add_argument("--nsteps", type=int, default=128)
    a1.add_argument("--shard", type=int, default=0)
    a1.add_argument("--num-shards", type=int, default=1)
    a1.add_argument("--device", default="cuda")
    a1.add_argument("--out", required=True)
    a2 = sub.add_parser("acceptance-merge")
    a2.add_argument("--parts", nargs="+", required=True)
    a2.add_argument("--out", required=True)
    a2.add_argument("--per-n-out", default=None, help="also write the per-N table (plan S1)")
    a2.add_argument("--n-knots", type=int, default=5, help="per-N table: knots of the logit in log N")
    a = ap.parse_args(argv)
    if a.cmd == "prepare":
        spec = _load_toml(a.spec)
        if spec.get("schema") != CAMPAIGN_SCHEMA:
            raise ValueError(f"campaign spec schema must be {CAMPAIGN_SCHEMA}")
        m = prepare(spec, a.out)
        print(f"campaign {spec.get('name')}: {m['n_ensembles']} x {m['n_events']} events ({m['source']}) -> {a.out}")
    elif a.cmd == "fit":
        if a.claim:
            print(f"worker sealed {run_claiming(a.dir)} fits")
            return
        if len(a.dir) != 1:
            raise SystemExit("several --dir need --claim")
        with open(Path(a.dir[0]) / "campaign.json") as fh:
            n = len(json.load(fh)["tasks"])
        w = a.worker if a.worker is not None else int(os.environ.get("SLURM_PROCID", 0))
        for k in (a.tasks if a.tasks else worker_tasks(n, w, a.num_workers)):
            fit_task(a.dir[0], k)
    elif a.cmd == "unlock-stale":
        unlock_stale(a.dir)
    elif a.cmd == "report":
        doc = report(a.dir, a.device, a.control)
        agg = doc["aggregate"]
        print(f"{doc['n_fits']} fits verified, {len(doc['refused'])} refused {doc['refused']}")
        if agg:
            print("mean pulls", {k: f"{v:+.2f}±{agg['mean_pull_se'][k]:.2f}" for k, v in agg["mean_pull"].items()})
            print("joint", {k: (v["covered"], v["n"], v["nominal_inside"]) for k, v in agg["joint"].items()},
                  "levels KS p", agg["levels_ks"]["ks_p"])
    elif a.cmd == "acceptance":
        acceptance_part(a.checkpoint, _axes(a.axis), a.n_samples, a.seed, a.nsteps, a.shard,
                        a.num_shards, a.out, a.device)
    else:
        acc = acceptance_merge(a.parts, a.out, a.per_n_out, a.n_knots)
        print(f"acceptance table {acc.shape}, range [{acc.min():.4f}, {acc.max():.4f}] -> {a.out}")


if __name__ == "__main__":
    main()
