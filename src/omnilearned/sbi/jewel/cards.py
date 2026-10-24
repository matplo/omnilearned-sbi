"""JEWEL configuration, installation inspection and the cards of one generation task.

Parameter mapping (installed medium-simple.f; medium-card labels TI, TAUI, TC, WOODSSAXON,
CENTRMIN, CENTRMAX, NF, N0, D, SIGMANN, MDFACTOR, MDSCALEFAC; the nucleus is MASS/NPROTON in
the main card): ``T_ref`` -> TI (initial temperature at the
centre of a b = 0 collision; local T scales with n_part^(1/4)), ``tau_init`` -> TAUI, ``b`` ->
CENTRMIN = CENTRMAX = c(b) (:mod:`geometry`). Random stream: PYTHIA's MRPY(1) = NJOB * 1000,
so NJOB identifies a task's events. Ported from the omni-sbi-mc application
(``a/src/jet_quenching_inference/jewel.py``, tag devA-2026-10-22).
"""

import hashlib
import os
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib

CONFIG = Path(__file__).resolve().parents[4] / "configs" / "jewel-2.6.0-cs-v1.toml"
SPLIT_STREAM = {"train": "jewel_train", "validation": "jewel_validation", "test": "jewel_test",
                "probe": "jewel_test"}
MAX_NJOB = 900000


def load_config(path=CONFIG):
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    cfg["_path"] = str(Path(path).resolve())
    cfg["_sha256"] = _sha256(path)
    return cfg


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def inspect_installation(cfg):
    """Executables, LHAPDF set and patch hashes of the installed JEWEL; never installs."""
    j = cfg["jewel"]
    prefix = Path(os.path.expandvars(j["prefix"]))
    exe = prefix / "bin" / f"jewel-2.6.0-{j['variant']}"
    lhapdf = Path(os.path.expandvars(j["lhapdf_data"]))
    pdf = lhapdf / j["pdfset_name"]
    blocking = []
    if not os.access(exe, os.X_OK):
        blocking.append(f"JEWEL executable {exe} missing")
    if not (pdf / f"{j['pdfset_name']}.info").is_file():
        blocking.append(f"LHAPDF set {j['pdfset_name']} not installed in {lhapdf} (user installs it)")
    info = prefix / "info"
    return {"executable": str(exe), "lhapdf_data": str(lhapdf), "pdfset": j["pdfset_name"],
            "patches": (info / "CUSTOM-PATCHES.txt").read_text() if (info / "CUSTOM-PATCHES.txt").is_file() else None,
            "patch_sha256": {p.name: _sha256(p) for p in sorted(info.glob("*.patch"))},
            "executable_sha256": _sha256(exe) if exe.is_file() else None,
            "blocking": blocking, "ready": not blocking}


def njob(cfg, split, point_id, chunk):
    """NJOB of a task: base[split] + chunks_per_point * point_id + chunk (unique, < 900000)."""
    r = cfg["randomness"]
    base = r["njob_base"][split]
    n = base + int(r["chunks_per_point"]) * int(point_id) + int(chunk)
    if not (0 <= int(chunk) < int(r["chunks_per_point"])) or not 0 <= n < MAX_NJOB:
        raise ValueError(f"NJOB {n} out of range for {split} point {point_id} chunk {chunk}")
    return n


def task_cards(cfg, theta, nevent, njob_value, geometry, tables_dir=".", extra=None):
    """(params.dat text, medium.dat text, centrality %) for one task at theta = (b, TI, TAUI).

    ``extra`` adds main-card lines (e.g. ``{"DOSUBTRACTION": "T", "RSUB": 0.5}`` for the
    probe's cross-check against JEWEL's own subtraction, which suppresses WRITESCATCEN).
    """
    j = cfg["jewel"]
    b, ti, taui = (float(v) for v in theta)
    c = geometry.centrality_for_b(b)
    tdir = Path(tables_dir)
    flag = lambda v: "T" if v else "F"  # noqa: E731
    params = [
        f"NEVENT {int(nevent)}", f"NJOB {int(njob_value)}",
        "LOGFILE jewel.log", "HEPMCFILE events.hepmc",
        f"SPLITINTFILE {tdir / 'splitint.dat'}", f"PDFFILE {tdir / 'pdfs.dat'}",
        f"XSECFILE {tdir / 'xsecs.dat'}", "MEDIUMPARAMS medium.dat",
        f"SQRTS {j['sqrts_GeV']}", f"PTMIN {j['ptmin_GeV']}", f"PTMAX {j['ptmax_GeV']}",
        f"ETAMAX {j['etamax']}", f"PDFSET {j['pdfset']}", f"WEIGHTED {flag(j['weighted'])}",
        f"KEEPRECOILS {flag(j['keeprecoils'])}", f"WRITESCATCEN {flag(j['writescatcen'])}",
        f"WRITEDUMMIES {flag(j['writedummies'])}",
        f"MASS {int(j['mass_number'])}", f"NPROTON {int(j['proton_number'])}",
    ] + [f"{k} {v}" for k, v in (extra or {}).items()]
    medium = [f"TI {ti}", f"TAUI {taui}", f"CENTRMIN {c:.10f}", f"CENTRMAX {c:.10f}"]
    return "\n".join(params) + "\n", "\n".join(medium) + "\n", c
