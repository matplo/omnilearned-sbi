"""Toy configuration loading and reproducible single-setting draws.

The config's ``backend`` selects the generator module: ``toy-quenching-v1``
(fixed 32 constituents, ``simulator``) or ``toy-quenching-v2`` (variable
multiplicity, padded to ``max_constituents`` with a mask, ``simulator_v2``).
"""

import hashlib
from pathlib import Path

import numpy as np

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib

from omnilearned.sbi.toy import simulator, simulator_v2
from omnilearned.sbi.toy.design import STREAMS, rng_for

CONFIG_DIR = Path(__file__).resolve().parents[4] / "configs"
DEFAULT_CONFIG = CONFIG_DIR / "toy-quenching-v1.toml"
V2_CONFIG = CONFIG_DIR / "toy-quenching-v2.toml"
BACKENDS = {simulator.BACKEND: simulator, simulator_v2.BACKEND: simulator_v2}


def simulator_for(backend):
    """Generator module of a backend id."""
    if backend not in BACKENDS:
        raise ValueError(f"unknown toy backend {backend!r}; known: {sorted(BACKENDS)}")
    return BACKENDS[backend]


def load_toy_config(path=DEFAULT_CONFIG):
    """(config dict, ToyConstants, Selection) from a toy TOML file (any backend)."""
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    mod = simulator_for(cfg.get("backend", simulator.BACKEND))
    return (cfg, mod.ToyConstants.from_config(cfg["toy"]),
            mod.Selection.from_config(cfg["selection"]))


def sample_setting(theta, n, seed, stream="predictive", index=0, path=DEFAULT_CONFIG):
    """Selected direct toy jets at one setting; generator (seed, stream, index)."""
    cfg, c, s = load_toy_config(path)
    mod = simulator_for(cfg.get("backend", simulator.BACKEND))
    rng = rng_for(seed, STREAMS[stream], index)
    return mod.sample_toy(np.asarray(theta, dtype=np.float64), int(n), rng, c, s,
                          stream=STREAMS[stream], index=index)


def array_sha256(rec):
    """Platform-independent hash of the observed records (features, mask if any, ids)."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(rec["x"], dtype="<f8").tobytes())
    if "mask" in rec:
        h.update(np.ascontiguousarray(rec["mask"], dtype="u1").tobytes())
    h.update(np.ascontiguousarray(rec["event_id"], dtype="<i8").tobytes())
    return h.hexdigest()
