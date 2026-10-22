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
    """Bitwise hash of the observed records (features, mask if any, ids).

    Identical across runs on one CPU type, not across instruction sets: numpy
    picks SIMD kernels per CPU (AVX-512 vs AVX2), and transcendental results
    then differ in the last bits (measured <= 1.4e-15 relative, 2026-10-22).
    Across hosts compare :func:`discrete_sha256` and :func:`float_fingerprint`.
    """
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(rec["x"], dtype="<f8").tobytes())
    if "mask" in rec:
        h.update(np.ascontiguousarray(rec["mask"], dtype="u1").tobytes())
    h.update(np.ascontiguousarray(rec["event_id"], dtype="<i8").tobytes())
    return h.hexdigest()


def discrete_sha256(rec):
    """Hash of the discrete record content: shape, mask (v2) and event ids.

    The event ids record which attempts were selected and the mask the
    constituent multiplicities; both must be identical on every host, and any
    difference means the toy itself changed.
    """
    h = hashlib.sha256()
    x = np.asarray(rec["x"])
    h.update(np.asarray(x.shape, dtype="<i8").tobytes())
    if "mask" in rec:
        h.update(np.ascontiguousarray(rec["mask"], dtype="u1").tobytes())
    h.update(np.ascontiguousarray(rec["event_id"], dtype="<i8").tobytes())
    return h.hexdigest()


def _fingerprint_terms(rec, k):
    x = np.asarray(rec["x"], dtype=np.float64)
    x = x.reshape(-1, x.shape[-1])
    w = np.random.default_rng(0).uniform(0.5, 1.5, size=(k, x.shape[0]))
    return x, w


def float_fingerprint(rec, k=4):
    """Float features projected on fixed weights: [F, k + 1] float64.

    Columns are k weighted sums over all entries (weights from a fixed seed)
    and the sum of squares. Compare across hosts with
    :func:`fingerprint_deviation`.
    """
    x, w = _fingerprint_terms(rec, k)
    return np.concatenate([w @ x, (x * x).sum(0, keepdims=True)]).T


def fingerprint_deviation(rec, reference, k=4):
    """Largest fingerprint difference, relative to the sum of |terms| per entry.

    Roundoff gives ~1e-15; one changed entry is visible above ~1e-9.
    """
    x, w = _fingerprint_terms(rec, k)
    scale = np.concatenate([w @ np.abs(x), (x * x).sum(0, keepdims=True)]).T
    diff = np.abs(float_fingerprint(rec, k) - np.asarray(reference, dtype=np.float64))
    return float(np.max(diff / np.maximum(scale, 1e-300)))
