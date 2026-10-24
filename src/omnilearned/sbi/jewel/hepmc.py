"""Streaming reader for the HepMC2 IO_GenEvent text files JEWEL writes.

Per event: number, scale (PARI(17), the hard-process pT with the partons-v1 patch), weight
(last field of the E line), cross section (C line, pb), centrality (H line impact-parameter
field, which JEWEL fills with the centrality fraction) and every P record as arrays: pid,
px, py, pz, e, mass, status. Statuses: 1 final state (dummies are pid 111 with E = 1e-6 when
WRITEDUMMIES), 3 thermal momenta of scattering centres (WRITESCATCEN), 4 beam particles,
23 the two outgoing hard partons (partons-v1 patch).
"""

import gzip

import numpy as np


def _open(path):
    path = str(path)
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def read_events(path):
    """Yield one dict per event."""
    ev = None
    rows = []
    with _open(path) as f:
        for line in f:
            if not line or line[0] not in "EPCH":
                continue
            tok = line.split()
            if not line.endswith("\n"):
                break  # truncated last line (JEWEL stopped by a timeout): ignore it
            tag = tok[0]
            if tag == "E":
                if ev is not None:
                    yield _finish(ev, rows)
                ev = {"number": int(tok[1]), "scale": float(tok[3]), "weight": float(tok[-1]),
                      "xsec_pb": np.nan, "centrality": np.nan}
                rows = []
            elif ev is None:
                continue
            elif tag == "P":
                rows.append((int(tok[2]), float(tok[3]), float(tok[4]), float(tok[5]),
                             float(tok[6]), float(tok[7]), int(tok[8])))
            elif tag == "C":
                ev["xsec_pb"] = float(tok[1])
            elif tag == "H":
                ev["centrality"] = float(tok[10])
    if ev is not None:
        yield _finish(ev, rows)


def _finish(ev, rows):
    a = np.array(rows, dtype=np.float64).reshape(-1, 7)
    ev["pid"] = a[:, 0].astype(np.int32)
    ev["px"], ev["py"], ev["pz"], ev["e"], ev["mass"] = (a[:, k] for k in range(1, 6))
    ev["status"] = a[:, 6].astype(np.int8)
    return ev
