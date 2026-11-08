"""P5: numerics.summed_logl must forward obs.context (jet_pt) to log_prob.

The v3 backend's flow has jet_pt as known context; ConditionLayout.assemble raises if a
known-context layout is asked to assemble without it. A toy/v2 checkpoint has no known
context, so obs.context is None there and must stay None (log_prob's own default),
not an accidental zero array.
"""

import numpy as np

from omnilearned.sbi.observations import Observations
from omnilearned.sbi.toy.numerics import deltas, summed_logl


class _RecordingSurrogate:
    """Fake surrogate: log_prob returns zeros but records every context it was given."""

    def __init__(self):
        self.contexts = []

    def log_prob(self, x_phys, mask, candidates, event_ids, options, candidate_names=None,
                 context=None, dtype=None):
        self.contexts.append(context)
        self.dtype = dtype
        candidates = np.atleast_2d(candidates)
        return np.zeros((candidates.shape[0], x_phys.shape[0])), {}


def _obs(n=10, p=3, f=3, context=None):
    x = np.zeros((n, p, f))
    mask = np.ones((n, p), bool)
    event_id = np.arange(n, dtype=np.int64)
    return Observations(x, mask, event_id, context=context)


def test_summed_logl_forwards_context_when_present():
    ctx = np.arange(10, dtype=np.float64).reshape(-1, 1)  # stand-in jet_pt column
    obs = _obs(context=ctx)
    sur = _RecordingSurrogate()
    summed_logl(sur, obs, candidates=np.zeros((2, 3)), options={}, event_chunk=4)
    assert len(sur.contexts) == 3  # 10 events in chunks of 4 -> 3 calls
    seen = np.concatenate([c for c in sur.contexts], axis=0)
    np.testing.assert_array_equal(seen, ctx)


def test_summed_logl_passes_none_when_no_context():
    obs = _obs(context=None)
    sur = _RecordingSurrogate()
    summed_logl(sur, obs, candidates=np.zeros((1, 3)), options={}, event_chunk=4)
    assert all(c is None for c in sur.contexts)


def test_summed_logl_context_sliced_per_chunk():
    """Each chunk must see only its own rows of context, not the whole array."""
    ctx = (np.arange(7, dtype=np.float64) * 10.0).reshape(-1, 1)
    obs = _obs(n=7, context=ctx)
    sur = _RecordingSurrogate()
    summed_logl(sur, obs, candidates=np.zeros((1, 3)), options={}, event_chunk=3)
    assert [len(c) for c in sur.contexts] == [3, 3, 1]
    np.testing.assert_array_equal(sur.contexts[0], ctx[:3])
    np.testing.assert_array_equal(sur.contexts[-1], ctx[6:])


def test_deltas_forwards_context_through_to_summed_logl():
    ctx = np.ones((5, 1))
    obs = _obs(n=5, context=ctx)
    sur = _RecordingSurrogate()
    pairs = [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]]
    opts = {"solver": "midpoint", "nsteps": 4, "divergence": "hutchinson", "num_probes": 2,
            "probe": "rademacher", "probe_schedule": "per_step"}
    out = deltas(sur, obs, pairs, opts, seeds=[0, 1], event_chunk=5, log=lambda *a, **k: None)
    assert set(out) == {0, 1}
    assert all(c is not None for c in sur.contexts)


def test_summed_logl_dtype_defaults_float32_and_threads_float64():
    """The P5 follow-up's 'never tested' question needs dtype actually reaching log_prob."""
    import torch

    obs = _obs()
    sur = _RecordingSurrogate()
    summed_logl(sur, obs, candidates=np.zeros((1, 3)), options={}, event_chunk=4)
    assert sur.dtype == torch.float32  # unchanged default, matching every caller before this

    sur2 = _RecordingSurrogate()
    summed_logl(sur2, obs, candidates=np.zeros((1, 3)), options={}, event_chunk=4, dtype="float64")
    assert sur2.dtype == torch.float64


def test_deltas_threads_dtype_through_to_summed_logl():
    import torch

    obs = _obs(n=5)
    sur = _RecordingSurrogate()
    pairs = [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]]
    opts = {"solver": "midpoint", "nsteps": 4, "divergence": "hutchinson", "num_probes": 2,
            "probe": "rademacher", "probe_schedule": "per_step"}
    deltas(sur, obs, pairs, opts, seeds=[0], event_chunk=5, log=lambda *a, **k: None,
          dtype="float64")
    assert sur.dtype == torch.float64


def test_summed_logl_rejects_unknown_dtype():
    import pytest

    obs = _obs()
    sur = _RecordingSurrogate()
    with pytest.raises(ValueError, match="dtype"):
        summed_logl(sur, obs, candidates=np.zeros((1, 3)), options={}, dtype="float16")
