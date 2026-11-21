"""P5 candidate (a): graded/non-uniform time grid (LikelihoodOptions.grid_power).

grid_power=1.0 must reproduce today's exact uniform-grid behavior bit-for-bit (no
regression for any existing caller, none of which pass the new field). grid_power>1.0
concentrates steps near t=0 and must still converge to the exact analytic density as
nsteps grows, and must raise for invalid combinations (grid_power<1.0, or grid_power!=1.0
with an adaptive solver).
"""

import numpy as np
import pytest
import torch

from omnilearned.sbi.base import GaussianBase
from omnilearned.sbi.likelihood import LikelihoodOptions, _graded_edges, log_prob

from fields import LinearField, linear_exact_logp

MEANS = [0.3, -1.0, 2.0]
STDS = [0.5, 2.0, 1.5]
BASE = GaussianBase(MEANS, STDS)

torch.set_default_dtype(torch.float32)


def make_cloud(n=6, p=5, f=3, n_real=None, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, p, f, generator=g, dtype=dtype)
    mask = torch.ones(n, p, dtype=torch.bool)
    if n_real is not None:
        for i, r in enumerate(n_real):
            mask[i, r:] = False
    return x * mask.unsqueeze(-1), mask


def opts(**kw):
    d = dict(solver="rk4", nsteps=32, divergence="exact", max_nfe=100000)
    d.update(kw)
    return LikelihoodOptions(**d)


def test_graded_edges_shape():
    for nsteps in (4, 16, 64):
        edges = _graded_edges(nsteps, 1.0)
        np.testing.assert_allclose(edges, np.arange(nsteps + 1) / nsteps)

        g = _graded_edges(nsteps, 2.5)
        assert g[0] == 0.0
        assert g[-1] == 1.0
        assert np.all(np.diff(g) > 0)  # strictly increasing
        steps = np.diff(g)
        uniform = 1.0 / nsteps
        assert steps[0] < uniform  # first step much smaller than uniform
        assert steps[-1] > uniform  # last step larger than uniform


def test_grid_power_default_is_one():
    assert LikelihoodOptions().grid_power == 1.0


@pytest.mark.parametrize("solver", ["euler", "midpoint", "rk4"])
def test_grid_power_one_matches_omitting_field(solver):
    A = np.diag([0.4, -0.3, 0.8])
    x, mask = make_cloud(n_real=[5, 3, 1, 5, 2, 4])
    field = LinearField(A, alpha=1.0, beta=0.5).double()
    lp_default, _ = log_prob(field, x, mask, BASE, options=opts(solver=solver, nsteps=16))
    lp_explicit, _ = log_prob(field, x, mask, BASE,
                              options=opts(solver=solver, nsteps=16, grid_power=1.0))
    assert torch.equal(lp_default, lp_explicit)


def test_grid_power_below_one_raises():
    with pytest.raises(ValueError, match="grid_power"):
        LikelihoodOptions(grid_power=0.5).validate()


def test_grid_power_with_adaptive_solver_raises():
    with pytest.raises(ValueError, match="grid_power"):
        LikelihoodOptions(solver="dopri5", grid_power=2.0).validate()
    # grid_power=1.0 (default) with an adaptive solver is fine.
    LikelihoodOptions(solver="dopri5").validate()


def test_graded_grid_converges_on_analytic_field():
    A = np.array([[0.2, 0.5, -0.1], [-0.3, 0.1, 0.4], [0.25, -0.2, -0.3]])
    x, mask = make_cloud(n_real=[5, 4, 3, 2, 1, 5])
    exact = linear_exact_logp(x.numpy(), mask.numpy(), A, 0.8, 0.6, MEANS, STDS)
    field = LinearField(A, alpha=0.8, beta=0.6).double()
    errs = []
    for n in [8, 16, 32, 64]:
        lp, _ = log_prob(field, x, mask, BASE,
                         options=opts(solver="midpoint", nsteps=n, grid_power=2.0))
        errs.append(np.abs(lp.numpy() - exact).max())
    errs = np.array(errs)
    assert np.all(np.diff(errs) < 0), errs  # error shrinks monotonically with refinement
    assert errs[-1] < 2e-3
