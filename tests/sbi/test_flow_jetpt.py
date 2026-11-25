"""P6 Part A: wiring JetPtModel (log p(pT_J | theta)) into FlowSurrogate.

Mirrors test_variable_multiplicity.py's `_variable_surrogate`/persistence pattern, with a
JetPtModel attached alongside the multiplicity model -- the v3 backend's actual shape
(``jet_pt`` is known context for both the constituent flow and the multiplicity model, and
is additionally the target of its own p(pT_J | theta) density).
"""

import numpy as np
import pytest
import torch

from omnilearned.sbi.base import GaussianBase
from omnilearned.sbi.likelihood import LikelihoodOptions


def _surrogate_with_jetpt(seed=0):
    from omnilearned.sbi.flow import FlowSurrogate
    from omnilearned.sbi.jetpt import JetPtModel, layout_with_jet_pt
    from omnilearned.sbi.multiplicity import MultiplicityModel, layout_with_multiplicity
    from omnilearned.sbi.parameters import ConditionLayout
    from omnilearned.sbi.transforms import FeatureTransform
    from omnilearned.sbi.velocity import SetFlowVelocity

    torch.manual_seed(seed)
    layout = layout_with_multiplicity(["a", "b"], [0, 0], [1, 2], known_context={"jet_pt": None})
    model = SetFlowVelocity(3, layout.dim, dim=32, depth=2, heads=4)
    with torch.no_grad():
        model.out.weight.normal_(0, 0.1)
        model.out.bias.normal_(0, 0.05)

    mult = MultiplicityModel(3, 64, hidden=16, depth=1)
    with torch.no_grad():
        mult.net[-1].bias.copy_(-0.5 * (torch.arange(65.0) - 12) ** 2 / 9)
    mult.layout = ConditionLayout.from_bounds(["a", "b"], [0, 0], [1, 2], known=["jet_pt"])

    jp = JetPtModel(2, n_components=2, hidden=16, depth=1)
    jp.layout = layout_with_jet_pt(["a", "b"], [0, 0], [1, 2])

    tr = FeatureTransform(["eta", "phi", "lpt"], [2.5, 1.2, None], [0, 0, 1], [0.5, 0.4, 1])
    return FlowSurrogate(model.eval(), GaussianBase.standard(3), tr, layout,
                        multiplicity=mult, jet_pt=jp)


def test_log_jet_pt_without_model_is_zero():
    from omnilearned.sbi.flow import FlowSurrogate
    from omnilearned.sbi.multiplicity import layout_with_multiplicity
    from omnilearned.sbi.transforms import FeatureTransform
    from omnilearned.sbi.velocity import SetFlowVelocity

    torch.manual_seed(0)
    layout = layout_with_multiplicity(["a", "b"], [0, 0], [1, 2])
    model = SetFlowVelocity(3, layout.dim, dim=16, depth=1, heads=2)
    tr = FeatureTransform(["eta", "phi", "lpt"], [2.5, 1.2, None], [0, 0, 1], [0.5, 0.4, 1])
    sur = FlowSurrogate(model.eval(), GaussianBase.standard(3), tr, layout)
    assert sur.jet_pt is None
    out = sur.log_jet_pt(np.array([4.0, 4.5, 5.0]), np.array([[0.5, 1.0]]))
    np.testing.assert_array_equal(out, np.zeros((1, 3)))


def test_log_prob_includes_log_jet_pt_and_matches_direct_call():
    sur = _surrogate_with_jetpt()
    rng = np.random.default_rng(0)
    n = 10
    x, mask = sur.sample([0.5, 1.0], n, None, seed=4, nsteps=6, pad_to=64,
                        jet_pt=np.full(n, 120.0))
    counts = mask.sum(1)

    log_pt_obs = rng.normal(np.log(120.0), 0.2, n)
    context = log_pt_obs[:, None]
    opts = LikelihoodOptions(solver="midpoint", nsteps=6, num_probes=2, probe_schedule="per_step")
    cands = np.array([[0.5, 1.0], [0.2, 1.7]])
    ids = np.arange(n) + 1000
    lp, diag = sur.log_prob(x.astype(np.float64), mask, cands, ids, opts, context=context)
    assert "log_jet_pt" in diag
    np.testing.assert_allclose(diag["log_jet_pt"], sur.log_jet_pt(log_pt_obs, cands))
    assert np.isfinite(lp).all()

    # Without the jet_pt term, log_prob would be smaller by exactly diag["log_jet_pt"].
    sur_no_jp = _surrogate_with_jetpt()
    sur_no_jp.jet_pt = None
    lp_no_jp, diag_no_jp = sur_no_jp.log_prob(x.astype(np.float64), mask, cands, ids, opts,
                                              context=context)
    assert "log_jet_pt" not in diag_no_jp
    np.testing.assert_allclose(lp - lp_no_jp, diag["log_jet_pt"], atol=1e-6)


def test_log_prob_paired_includes_log_jet_pt():
    sur = _surrogate_with_jetpt()
    n = 8
    x, mask = sur.sample([0.5, 1.0], n, None, seed=5, nsteps=6, pad_to=32,
                        jet_pt=np.full(n, 100.0))
    log_pt_obs = np.full(n, np.log(100.0))
    context = log_pt_obs[:, None]
    opts = LikelihoodOptions(solver="midpoint", nsteps=6, num_probes=2, probe_schedule="per_step")
    rows = np.repeat([[0.5, 1.0]], n, 0)
    ids = np.arange(n) + 2000
    paired, diag = sur.log_prob_paired(x.astype(np.float64), mask, rows, ids, opts,
                                       context=context)
    assert "log_jet_pt" in diag
    assert np.isfinite(paired).all()
    expected = sur.log_jet_pt(log_pt_obs, np.array([[0.5, 1.0]]))[0]
    np.testing.assert_allclose(diag["log_jet_pt"], expected)


def test_sample_generates_jet_pt_when_model_present_and_not_supplied():
    sur = _surrogate_with_jetpt()
    # No jet_pt kwarg: should not raise, should draw it from the JetPtModel.
    x, mask = sur.sample([0.5, 1.0], 20, None, seed=7, nsteps=4, pad_to=32)
    assert x.shape == (20, 32, 3)
    counts = mask.sum(1)
    assert counts.min() >= 1


def test_sample_explicit_jet_pt_still_overrides():
    sur = _surrogate_with_jetpt()
    x1 = sur.sample([0.5, 1.0], 10, 5, seed=1, nsteps=4, jet_pt=150.0)
    x2 = sur.sample([0.5, 1.0], 10, 5, seed=1, nsteps=4, jet_pt=150.0)
    np.testing.assert_array_equal(x1, x2)


def test_sample_without_jet_pt_model_still_requires_explicit_jet_pt():
    """Backward compatibility: a surrogate with jet_pt=None behaves exactly as before."""
    sur = _surrogate_with_jetpt()
    sur.jet_pt = None
    with pytest.raises(ValueError, match="jet_pt"):
        sur.sample([0.5, 1.0], 5, 5, seed=1, nsteps=4)


def test_jet_pt_round_trips_through_save_load(tmp_path):
    from omnilearned.sbi.flow import FlowSurrogate

    sur = _surrogate_with_jetpt()
    path = tmp_path / "v.pt"
    sur.save(path)
    back = FlowSurrogate.load(path)
    assert back.jet_pt is not None

    log_pt = np.array([4.0, 4.5, 5.0])
    cands = np.array([[0.5, 1.0], [0.2, 1.7]])
    np.testing.assert_allclose(back.log_jet_pt(log_pt, cands), sur.log_jet_pt(log_pt, cands))


def test_checkpoint_without_jet_pt_is_unaffected(tmp_path):
    """A checkpoint saved before jet_pt existed (or without a jet_pt model) loads cleanly."""
    from omnilearned.sbi.flow import FlowSurrogate
    from omnilearned.sbi.multiplicity import layout_with_multiplicity
    from omnilearned.sbi.transforms import FeatureTransform
    from omnilearned.sbi.velocity import SetFlowVelocity

    torch.manual_seed(0)
    layout = layout_with_multiplicity(["a", "b"], [0, 0], [1, 2])
    model = SetFlowVelocity(3, layout.dim, dim=16, depth=1, heads=2)
    tr = FeatureTransform(["eta", "phi", "lpt"], [2.5, 1.2, None], [0, 0, 1], [0.5, 0.4, 1])
    sur = FlowSurrogate(model.eval(), GaussianBase.standard(3), tr, layout)
    path = tmp_path / "v.pt"
    sur.save(path)
    back = FlowSurrogate.load(path)
    assert back.jet_pt is None
