"""selection.acceptance_table must pass jet_pt to surrogate.sample() when the layout needs
it (the v3 backend) -- it never did before, which raised for any v3 checkpoint. Mirrors the
_RecordingSurrogate stub pattern from tests/sbi/test_numerics_context.py so this is a cheap,
no-GPU, no-checkpoint unit test of the threading, not a re-test of selection physics.
"""

import numpy as np
import pytest

from omnilearned.sbi.toy import selection
from omnilearned.sbi.toy.simulator_v2 import Selection


class _Layout:
    def __init__(self, known):
        self.known = known


class _RecordingSurrogate:
    """Fake surrogate: sample() returns an all-accepted cloud and records every
    jet_pt it was given."""

    def __init__(self, known=()):
        self.layout = _Layout(known)
        self.calls = []

    def sample(self, theta, n, fixed_count, seed=0, nsteps=32, solver="midpoint",
              candidate_names=None, jet_pt=None):
        self.calls.append(jet_pt)
        n_particles = 4
        x = np.zeros((n, n_particles, 3))
        x[..., 0] = 0.0   # eta
        x[..., 1] = 0.0   # phi
        x[..., 2] = np.log(20.0)  # pt
        mask = np.ones((n, n_particles), bool)
        return x, mask


SEL = Selection(jet_radius=0.4, jet_pt_range_GeV=(0.0, 1e6), jet_abs_eta_max=10.0,
                constituent_abs_eta_max=10.0, constituent_abs_phi_max_rad=10.0,
                min_constituents=1)
AXES = [np.array([1.0, 2.0]), np.array([0.3]), np.array([0.5])]


def test_no_known_context_never_calls_with_jet_pt():
    sur = _RecordingSurrogate(known=())
    acc, _ = selection.acceptance_table(sur, SEL, AXES, n_samples=5, seed=0)
    assert acc.shape == (2, 1, 1)
    assert all(c is None for c in sur.calls)


def test_jet_pt_known_context_requires_pool_fn():
    sur = _RecordingSurrogate(known=("jet_pt",))
    with pytest.raises(ValueError, match="jet_pt_pool_fn"):
        selection.acceptance_table(sur, SEL, AXES, n_samples=5, seed=0)


def test_jet_pt_known_context_resamples_from_the_given_pool():
    sur = _RecordingSurrogate(known=("jet_pt",))
    pool = np.array([10.0, 20.0, 30.0, 40.0])
    weight = np.array([1.0, 1.0, 1.0, 1.0])

    def pool_fn(theta):
        return pool, weight

    acc, _ = selection.acceptance_table(sur, SEL, AXES, n_samples=5, seed=0,
                                        jet_pt_pool_fn=pool_fn)
    assert acc.shape == (2, 1, 1)
    assert len(sur.calls) == 2  # one per grid point (2x1x1)
    for jet_pt in sur.calls:
        assert jet_pt is not None
        assert jet_pt.shape == (5,)
        assert set(np.unique(jet_pt)).issubset(set(pool))
