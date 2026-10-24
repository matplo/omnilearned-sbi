"""JEWEL package: geometry, cards, HepMC reader, constituent subtraction, jet records and
the generation worker end to end with a fake JEWEL executable (no JEWEL run needed)."""

import json
import os
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from omnilearned.sbi.jewel import cards, cs, hepmc
from omnilearned.sbi.jewel.geometry import JewelGeometry

CFG = cards.load_config()
SRC = Path(CFG["jewel"]["prefix"]).parents[1] / "src" / "jewel-2.6.0-custom-src"


def test_geometry_monotone_and_inverse():
    g = JewelGeometry()
    c = np.linspace(1, 90, 30)
    b = g.b_of_centrality(c)
    assert np.all(np.diff(b) > 0) and 1.0 < g.b_of_centrality(5.0) < 4.5 and 9 < g.b_of_centrality(50.0) < 13
    for bv in (0.5, 4.5, 7.5, 11.0):
        assert abs(g.b_of_centrality(g.centrality_for_b(bv)) - bv) < 1e-6


def _labels(path, pattern):
    import re

    return set(re.findall(pattern, path.read_text(errors="replace")))


def test_cards_use_only_labels_jewel_reads():
    params, medium, c = cards.task_cards(CFG, (4.5, 0.45, 0.65), 1000, 800001, JewelGeometry())
    main_keys = {line.split()[0] for line in params.splitlines()}
    med_keys = {line.split()[0] for line in medium.splitlines()}
    assert {"KEEPRECOILS", "WRITESCATCEN", "PDFSET", "MASS", "NPROTON", "SQRTS"} <= main_keys
    assert med_keys == {"TI", "TAUI", "CENTRMIN", "CENTRMAX"}
    assert "PDFSET 904400" in params and "KEEPRECOILS T" in params and "WRITESCATCEN T" in params
    if (SRC / "jewel-2.6.0.f").exists():  # check against the installed source when present
        assert main_keys <= _labels(SRC / "jewel-2.6.0.f", r'label\.eq\."([A-Z0-9]+)"')
        assert med_keys <= _labels(SRC / "medium-simple.f", r'LABEL=="([A-Z0-9]+)"')


def test_njob_unique_in_range_and_design_respects_holdout():
    from omnilearned.sbi.jewel.generate import box, design_points
    from omnilearned.sbi.toy.datasets import margin_clashes
    from omnilearned.sbi.toy.design import reserved_settings

    grid = design_points(CFG, "train")
    assert grid.shape == (972, 3)
    res = reserved_settings(CFG)
    assert not margin_clashes(grid, res, box(CFG)).any()
    for th in res.values():  # strictly between grid nodes on every axis
        for k, p in enumerate(("b", "T_ref", "tau_init")):
            ax = np.asarray(CFG["design"][p])
            assert ax[0] < th[k] < ax[-1] and np.min(np.abs(ax - th[k])) > 1e-6
    val = design_points(CFG, "validation")
    lo = np.array([v[0] for v in box(CFG).values()])
    hi = np.array([v[1] for v in box(CFG).values()])
    assert val.shape == (128, 3) and np.all((val >= lo) & (val <= hi))
    nj = [cards.njob(CFG, s, p, c) for s, n in (("train", 972), ("validation", 128))
          for p in range(n) for c in (0, 99)]
    assert len(set(nj)) == len(nj) and max(nj) < cards.MAX_NJOB
    with pytest.raises(ValueError):
        cards.njob(CFG, "train", 0, 100)


def test_constituent_subtraction_cases():
    # particle with a collinear, softer thermal momentum: pT reduced by the thermal pT
    st = np.array([1, 3, 3, 1, 1])
    pid = np.array([211, 0, 0, 111, 12])
    px = np.array([10.0, 2.0, 0.0, 1e-6 * 0, 3.0])
    py = np.array([0.0, 0.0, 5.0, 0.0, 0.0])
    pz = np.zeros(5)
    m = np.array([0.1396, 0.0, 0.0, 0.0, 0.0])
    e = np.sqrt(px ** 2 + py ** 2 + pz ** 2 + m ** 2)
    e[3] = 1e-6  # dummy: skipped
    out = cs.subtract(st, pid, px, py, pz, e, dRmax=0.5)
    pt = np.hypot(out["px"], out["py"])
    sub = out["tag"] == cs.TAG_SUBTRACTED
    assert np.isclose(pt[sub & (out["pid"] == 211)], 8.0).all()  # 10 - 2
    left = out["tag"] == cs.TAG_THERMAL_LEFT  # the thermal at phi = pi/2 is beyond dRmax: kept
    assert left.sum() == 1 and np.isclose(pt[left], 5.0).all() and (out["pid"][left] == 211).all()
    assert not np.any((out["pid"] == 111) & (out["e"] < 1e-5))  # dummy gone
    assert (out["pid"] == 12).sum() == 1  # neutrinos are part of the subtraction; removed later


FAKE = textwrap.dedent('''\
    #!{python}
    """Fake JEWEL: writes NEVENT synthetic events in JEWEL's HepMC2 layout."""
    import sys, time
    import numpy as np
    card = dict(l.split(None, 1) for l in open(sys.argv[1]) if l.strip())
    n, njob = int(card["NEVENT"]), int(card["NJOB"])
    if "SLEEP" in card:
        time.sleep(float(card["SLEEP"]))
    rng = np.random.default_rng(njob)
    with open("events.hepmc", "w") as f, open("jewel.log", "w") as log:
        f.write("HepMC::Version 2.06.09\\nHepMC::IO_GenEvent-START_EVENT_LISTING\\n")
        for ev in range(1, n + 1):
            pthat = 80 + 40 * rng.random()
            f.write(f"E {{ev}} -1 {{pthat:.4f}} 0 0 0 0 1 1 2 0 1 1\\nU GEV MM\\nC 12.5 0.1\\n")
            f.write("H 0 0 0 0 0 0 0 0 0 0.0876 0 0 0\\n")
            rows = [(2212, 0, 0, 2.5e3, 2.5e3, 0.938, 4), (2212, 0, 0, -2.5e3, 2.5e3, 0.938, 4)]
            rows += [(21, pthat, 0, 0, pthat, 0, 23), (21, -pthat, 0, 0, pthat, 0, 23)]
            k = 18 + int(rng.integers(0, 8))
            z = rng.dirichlet(np.ones(k)) * (pthat + 10)
            for zi in z:  # a jet near eta 0.2, phi 1.0
                eta, phi = 0.2 + 0.05 * rng.normal(), 1.0 + 0.05 * rng.normal()
                rows.append((211, zi * np.cos(phi), zi * np.sin(phi), zi * np.sinh(eta), zi * np.cosh(eta), 0, 1))
            for _ in range(6):  # thermal momenta, some inside the jet
                eta, phi, pt = 0.2 + 0.3 * rng.normal(), 1.0 + 0.3 * rng.normal(), 0.5 + rng.random()
                rows.append((21, pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), pt * np.cosh(eta), 0, 3))
            rows.append((111, 0, 0, 0, 1e-6, 0, 1))  # dummy
            rows.append((12, 1.0, 0.5, 0.2, float(np.sqrt(1.29)), 0, 1))  # neutrino
            for b, (pdg, x, y, zz, e, mass, st) in enumerate(rows, start=10001):
                f.write(f"P {{b}} {{pdg}} {{x:.6e}} {{y:.6e}} {{zz:.6e}} {{e:.6e}} {{mass:.6e}} {{st}} 0 0 0 0\\n")
        f.write("HepMC::IO_GenEvent-END_EVENT_LISTING\\n")
        log.write(f"fake JEWEL done {{n}} events\\n")
    ''')


@pytest.fixture
def fake_cfg(tmp_path):
    exe = tmp_path / "fake_jewel.py"
    exe.write_text(FAKE.format(python=sys.executable))
    exe.chmod(0o755)
    cfg = json.loads(json.dumps(CFG))
    cfg["jewel"]["executable"] = str(exe)
    cfg["jewel"]["launcher"] = []
    cfg["jewel"]["env_scripts"] = []
    return cfg


def test_worker_end_to_end_with_fake_jewel(fake_cfg, tmp_path):
    pytest.importorskip("hepyy")
    from omnilearned.sbi.jewel import generate, jets
    from omnilearned.sbi.toy.datasets import read_shards
    from omnilearned.sbi.toy.simulator_v2 import select_sets

    d = tmp_path / "farm"
    doc = generate.plan(fake_cfg, "probe", d, chunks=2, theta=[[4.5, 0.45, 0.65]], nevent=40,
                        keep_hepmc=["probe_p00000_c000"])
    assert [t["njob"] for t in doc["tasks"]] == [800000, 800001]
    assert generate.work(d, scratch=str(tmp_path), log=lambda m: None) == 2
    assert generate.work(d, scratch=str(tmp_path), log=lambda m: None) == 0  # resumable: all done
    idx = generate.write_index(d)
    assert idx["tasks_done"] == 2 and idx["events"] == 80
    with open(d / "probe_p00000_c000.json") as f:
        man = json.load(f)
    assert man["events_generated"] == 40 and 0 < man["jets_selected"] <= 40
    assert man["files"]["events"] == "probe_p00000_c000.events.h5"
    assert (d / "hepmc" / "probe_p00000_c000.hepmc.gz").exists()
    assert not (d / "hepmc" / "probe_p00000_c001.hepmc.gz").exists()
    data = read_shards([d / s["file"] for s in idx["shards"]])
    n = idx["n_jets"]
    assert data["x"].shape[0] == n and np.unique(data["event_id"]).size == n
    assert np.all((data["event_id"] >> 48) == 18)  # jewel_test stream for probes
    np.testing.assert_allclose(data["theta"], np.repeat([[4.5, 0.45, 0.65]], n, 0))
    m = data["mask"]
    eta, phi, pt = data["x"][..., 0], data["x"][..., 1], np.exp(np.where(m, data["x"][..., 2], 0))
    ok, _, r = select_sets(eta, phi, np.where(m, pt, 0), m, jets.selection_from_config(fake_cfg))
    assert ok.all() and np.all(np.abs(r["jet_phi"]) < 0.05)  # rotated to the jet axis
    # the raw event store reproduces the shard's jets (float32 storage of 6-digit HepMC values)
    sel = jets.selection_from_config(fake_cfg)
    evs = generate.read_events_store(d / "probe_p00000_c000.events.h5")
    assert len(evs) == 40
    recs = [jets.event_jet(ev, fake_cfg, sel)[0] for ev in evs]
    recs = [r for r in recs if r is not None]
    ok2, _, _, x2, m2 = jets.select_records(recs, sel)
    first = read_shards([d / "probe_p00000_c000.h5"])
    w = first["x"].shape[1]
    np.testing.assert_allclose(x2[ok2][:, :w], first["x"], rtol=1e-4, atol=1e-4)
    assert sum(1 for _ in hepmc.read_events(d / "hepmc" / "probe_p00000_c000.hepmc.gz")) == 40


def test_worker_records_timeouts_and_continues(fake_cfg, tmp_path):
    from omnilearned.sbi.jewel import generate

    d = tmp_path / "farm"
    generate.plan(fake_cfg, "probe", d, chunks=1, theta=[[4.5, 0.45, 0.65]], nevent=5,
                  extra={"SLEEP": 30})
    assert generate.work(d, scratch=str(tmp_path), timeout=2, log=lambda m: None) == 1  # attempted
    assert not (d / "probe_p00000_c000.json").exists()
    fails = [json.loads(line) for line in open(d / "failures.log")]
    assert fails[0]["task"] == "probe_p00000_c000" and "timeout" in fails[0]["error"]
    assert not list((d / "claims").glob("*.lock"))


def test_reader_ignores_truncated_last_line(tmp_path):
    p = tmp_path / "t.hepmc"
    p.write_text("E 1 -1 90.0 0 0 0 0 1 1 2 0 1 1\nP 1 211 1.0 0.0 0.0 1.0 0.0 1 0 0 0 0\n"
                 "E 2 -1 95.0 0 0 0 0 1 1 2 0 1 1\nP 2 211 2.0 0.0 0.0 2.0 0.0 1 0 0 0 0\nP 3 21 1.5e+00 0.0")
    evs = list(hepmc.read_events(p))
    assert [e["number"] for e in evs] == [1, 2] and len(evs[1]["pid"]) == 1


def test_jewel_command_sources_env_scripts():
    from omnilearned.sbi.jewel.generate import jewel_command

    cmd = jewel_command(CFG, "/x/jewel")
    assert cmd[0] == "bash" and cmd[-2:] == ["/x/jewel", "params.dat"]
    assert all(f'source "{s}"' in cmd[2] for s in CFG["jewel"]["env_scripts"])
    assert "LHAPDF_DATA_PATH" in cmd[2]


def test_plan_is_chunk_major(tmp_path):
    from omnilearned.sbi.jewel import generate

    doc = generate.plan(CFG, "probe", tmp_path, chunks=2, theta=[[1, 0.2, 0.3], [2, 0.3, 0.4]], nevent=5)
    assert [(t["chunk"], t["point_id"]) for t in doc["tasks"]] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert len({t["njob"] for t in doc["tasks"]}) == 4
