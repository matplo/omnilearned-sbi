"""FlowSurrogate.log_prob's dtype parameter (added for the P5 nsteps=256/float64 follow-up):
default float32 behavior is unchanged; float64 is a real, reusable option rather than the
one-off bypass scripts/p5_trajectory_diagnostic.py originally used by hand.
"""

import numpy as np
import pytest
import torch

from omnilearned.sbi.base import GaussianBase
from omnilearned.sbi.flow import FlowSurrogate
from omnilearned.sbi.likelihood import LikelihoodOptions
from omnilearned.sbi.parameters import ConditionLayout
from omnilearned.sbi.transforms import FeatureTransform

from fields import TranslationField


def _surrogate(known_context=False):
    if known_context:
        layout = ConditionLayout(["mu", "jet_pt"], ["mu"], {"mu": 0.0, "jet_pt": 0.0},
                                 {"mu": 1.0, "jet_pt": 1.0})
    else:
        layout = ConditionLayout(["mu"], ["mu"], {"mu": 0.0}, {"mu": 1.0})
    model = TranslationField([0.0, 0.0, 0.0], use_cond=True)
    tr = FeatureTransform(["a", "b", "c"], [None, None, None], [0, 0, 0], [1, 1, 1])
    return FlowSurrogate(model, GaussianBase.standard(3), tr, layout)


def _surrogate_with_params():
    """A model with real learnable parameters, to test dtype-cast restoration
    (TranslationField above is a fixed field with no parameters at all)."""
    from omnilearned.sbi.velocity import SetFlowVelocity

    torch.manual_seed(0)
    layout = ConditionLayout(["mu"], ["mu"], {"mu": 0.0}, {"mu": 1.0})
    tr = FeatureTransform(["a", "b", "c"], [None, None, None], [0, 0, 0], [1, 1, 1])
    model = SetFlowVelocity(3, 1, dim=16, depth=1, heads=2)
    return FlowSurrogate(model, GaussianBase.standard(3), tr, layout)


def _batch(n=5, p=4, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, p, 3))
    mask = np.ones((n, p), bool)
    return x, mask


OPTS = LikelihoodOptions(solver="euler", nsteps=4, divergence="hutchinson", num_probes=1)


def test_default_dtype_matches_pre_existing_behavior():
    """No dtype argument -> identical to calling with dtype=torch.float32 explicitly."""
    sur = _surrogate()
    x, mask = _batch()
    lp_default, _ = sur.log_prob(x, mask, [[0.3]], np.arange(5), OPTS)
    lp_f32, _ = sur.log_prob(x, mask, [[0.3]], np.arange(5), OPTS, dtype=torch.float32)
    np.testing.assert_array_equal(lp_default, lp_f32)


def test_float64_runs_and_restores_model_dtype():
    sur = _surrogate_with_params()
    x, mask = _batch()
    before = next(sur.model.parameters()).dtype
    lp64, _ = sur.log_prob(x, mask, [[0.3]], np.arange(5), OPTS, dtype=torch.float64)
    after = next(sur.model.parameters()).dtype
    assert before == after == torch.float32, "model must not be left mutated after a float64 call"
    assert np.isfinite(lp64).all()


def test_float64_matches_float32_within_float32_precision():
    sur = _surrogate()
    x, mask = _batch(n=8)
    lp32, _ = sur.log_prob(x, mask, [[0.3]], np.arange(8), OPTS, dtype=torch.float32)
    lp64, _ = sur.log_prob(x, mask, [[0.3]], np.arange(8), OPTS, dtype=torch.float64)
    np.testing.assert_allclose(lp64, lp32, atol=1e-3)


def test_float64_survives_model_mutation_error(monkeypatch):
    """The model is restored to its original dtype even if the call raises partway through."""
    sur = _surrogate_with_params()
    x, mask = _batch()
    before = next(sur.model.parameters()).dtype

    class Boom(RuntimeError):
        pass

    def broken_forward(*a, **k):
        raise Boom("synthetic failure mid-call")

    monkeypatch.setattr(sur.model, "forward", broken_forward)
    with pytest.raises(Boom):
        sur.log_prob(x, mask, [[0.3]], np.arange(5), OPTS, dtype=torch.float64)
    assert next(sur.model.parameters()).dtype == before


def test_known_context_layout_dtype_threading():
    """A known-context (jet_pt-style) layout assembles correctly in both dtypes."""
    sur = _surrogate(known_context=True)
    x, mask = _batch()
    context = np.log(np.array([30.0, 50.0, 80.0, 20.0, 60.0]))[:, None]
    lp32, _ = sur.log_prob(x, mask, [[0.3]], np.arange(5), OPTS, context=context,
                           dtype=torch.float32)
    lp64, _ = sur.log_prob(x, mask, [[0.3]], np.arange(5), OPTS, context=context,
                           dtype=torch.float64)
    assert np.isfinite(lp32).all() and np.isfinite(lp64).all()
    np.testing.assert_allclose(lp64, lp32, atol=1e-3)


def test_condition_layout_assemble_dtype():
    layout = ConditionLayout(["mu"], ["mu"], {"mu": 0.0}, {"mu": 1.0})
    default = layout.assemble([[0.5]])
    assert default.dtype == np.float32
    f64 = layout.assemble([[0.5]], dtype=np.float64)
    assert f64.dtype == np.float64
    np.testing.assert_allclose(f64, default.astype(np.float64))


def test_run_fit_likelihood_dtype_field(tmp_path):
    """[likelihood].dtype = "float64" in a fit config reaches FlowSurrogate.log_prob;
    an unknown dtype string raises clearly instead of silently falling back."""
    from omnilearned.sbi.fit import _dtype, run_fit
    from omnilearned.sbi.observations import write_observations

    assert _dtype({}) == torch.float32
    assert _dtype({"likelihood": {"dtype": "float64"}}) == torch.float64
    with pytest.raises(ValueError, match="dtype"):
        _dtype({"likelihood": {"dtype": "complex128"}})

    sur = _surrogate_with_params()
    rng = np.random.default_rng(1)
    obs_path = tmp_path / "obs.h5"
    write_observations(obs_path, rng.normal(size=(6, 4, 3)), np.ones((6, 4), bool),
                       np.arange(6), ["a", "b", "c"])
    cfg = {
        "schema": "omnilearned-sbi-fit-v1", "checkpoint": "unused", "observations": str(obs_path),
        "output_prefix": str(tmp_path / "out" / "fit"), "device": "cpu", "mode": "fixed_vector",
        "grid": {"mu": {"min": -0.5, "max": 0.5, "n": 3}},
        "likelihood": {"solver": "euler", "nsteps": 2, "num_probes": 1, "dtype": "float64"},
    }
    doc = run_fit(cfg, surrogate=sur)
    assert np.isfinite(doc["summary"]["mle"]["mu"])

    # cross-check against the same grid/data at the default (float32) dtype
    cfg32 = {**cfg, "likelihood": {**cfg["likelihood"], "dtype": "float32"}}
    doc32 = run_fit({**cfg32, "output_prefix": str(tmp_path / "out32" / "fit")}, surrogate=sur)
    np.testing.assert_allclose(doc["summary"]["mle"]["mu"], doc32["summary"]["mle"]["mu"],
                               atol=1e-2)
