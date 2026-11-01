"""Multiplicity model extensions: support for additional known context columns."""

import numpy as np
import pytest
import torch

from omnilearned.sbi.multiplicity import (
    MULTIPLICITY_COLUMN,
    MultiplicityModel,
    layout_for_multiplicity_model,
    layout_with_multiplicity,
    multiplicity_context,
    train_multiplicity,
)
from omnilearned.sbi.parameters import ConditionLayout


def test_layout_with_multiplicity_backward_compat():
    """layout_with_multiplicity single-argument call unchanged for backward compat."""
    # Original call: only scale parameter
    lay = layout_with_multiplicity(["a", "b"], [0, 0], [1, 2])

    # Should produce same layout as before
    assert lay.dim == 3  # 2 unknowns + 1 known (n_constituents)
    assert lay.columns == ["a", "b", MULTIPLICITY_COLUMN]
    assert lay.unknown == ["a", "b"]
    assert lay.known == [MULTIPLICITY_COLUMN]
    assert lay.scale[MULTIPLICITY_COLUMN] == 100.0  # default


def test_layout_with_multiplicity_custom_scale():
    """Custom scale for n_constituents works."""
    lay = layout_with_multiplicity(["a", "b"], [0, 0], [1, 2], scale=50.0)
    assert lay.scale[MULTIPLICITY_COLUMN] == 50.0


def test_layout_with_multiplicity_with_jet_pt():
    """layout_with_multiplicity adds jet_pt as known context."""
    lay = layout_with_multiplicity(
        ["b", "T_ref", "tau_init"],
        [0, 0.1, 0.2],
        [11, 0.9, 1.0],
        scale=100.0,
        known_context={"jet_pt": None},
        known_scales={"jet_pt": 5.0}
    )

    assert lay.dim == 5  # 3 unknowns + 2 known
    assert set(lay.columns) == {"b", "T_ref", "tau_init", "jet_pt", MULTIPLICITY_COLUMN}
    assert set(lay.unknown) == {"b", "T_ref", "tau_init"}
    assert set(lay.known) == {"jet_pt", MULTIPLICITY_COLUMN}
    assert lay.scale[MULTIPLICITY_COLUMN] == 100.0
    assert lay.scale["jet_pt"] == 5.0


def test_layout_with_multiplicity_known_context_as_list():
    """known_context can be a list or tuple."""
    # As list
    lay1 = layout_with_multiplicity(
        ["x"], [0], [1],
        known_context=["jet_pt"]
    )

    # As tuple
    lay2 = layout_with_multiplicity(
        ["x"], [0], [1],
        known_context=("jet_pt",)
    )

    assert lay1.known == lay2.known
    assert set(lay1.known) == {"jet_pt", MULTIPLICITY_COLUMN}


def test_layout_with_multiplicity_assembles_correctly():
    """ConditionLayout.assemble works with jet_pt alongside the forced n_constituents column.

    layout_with_multiplicity is the constituent FLOW's layout, which always carries
    n_constituents as known context (the flow generates exactly N constituents) -- so an
    assemble call here must supply context for both known columns, not jet_pt alone.
    """
    lay = layout_with_multiplicity(
        ["b", "T_ref"],
        [0, 0.1],
        [11, 0.9],
        scale=100.0,
        known_context={"jet_pt": None},
        known_scales={"jet_pt": 5.0}
    )
    assert lay.known == [MULTIPLICITY_COLUMN, "jet_pt"]

    # Test assembling: unknowns are normalized, knowns are not (assembly does it)
    candidates = np.array([[5.5, 0.5]])  # midpoint of bounds
    context = np.array([[30.0, 200.0]])  # n_constituents, jet_pt (GeV)

    cond = lay.assemble(candidates, context=context, context_names=[MULTIPLICITY_COLUMN, "jet_pt"])
    assert cond.shape == (1, 4)
    # Unknowns should be normalized to [-1, 1] (midpoint should be ~0)
    np.testing.assert_allclose(cond[0, :2], [0, 0], atol=0.1)
    # Known context should also be normalized
    assert np.isfinite(cond[0, 2:]).all()  # n_constituents, jet_pt both normalized


def test_multiplicity_model_conditioned_on_single_unknown():
    """MultiplicityModel conditions on theta only (original behavior)."""
    model = MultiplicityModel(2, max_n=50, hidden=32, depth=2)
    model.layout = ConditionLayout.from_bounds(["a", "b"], [0, 0], [1, 1])

    cond = torch.randn(10, 2, dtype=torch.float32)
    lp = model.log_prob(torch.randint(0, 50, (10,)), cond)
    assert lp.shape == (10,)
    assert torch.isfinite(lp).all()


def test_multiplicity_model_conditioned_on_theta_and_jet_pt():
    """MultiplicityModel conditions on theta and jet_pt -- never on n_constituents.

    Uses layout_for_multiplicity_model, not layout_with_multiplicity: the latter is the
    constituent flow's layout and always forces n_constituents into known context, which
    would require this count model to be told N before predicting N.
    """
    lay = layout_for_multiplicity_model(
        ["b", "T_ref"],
        [0, 0.1],
        [11, 0.9],
        known_context={"jet_pt": None},
        known_scales={"jet_pt": 5.0}
    )
    assert lay.known == ["jet_pt"]

    # Model takes cond_dim = 3 (2 unknowns + jet_pt)
    model = MultiplicityModel(lay.dim, max_n=64, hidden=32, depth=2)
    model.layout = lay

    # Assemble condition with both unknowns and context
    candidates = np.array([[5.5, 0.5], [3.0, 0.3]])
    context = np.array([[100.0], [150.0]])
    cond = lay.assemble_torch(candidates, "cpu", context=context, context_names=["jet_pt"])

    n = np.array([20, 30])
    lp = model.log_prob(torch.from_numpy(n).long(), cond)
    assert lp.shape == (2,)
    assert torch.isfinite(lp).all()


def test_multiplicity_weighted_training_with_jet_pt():
    """Weighted training works with jet_pt conditioning."""
    rng = np.random.default_rng(111)
    n_data = 5000

    # Synthetic data: theta-dependent multiplicity
    theta1 = rng.uniform(-1, 1, n_data)
    theta2 = rng.uniform(-1, 1, n_data)
    jet_pt = rng.uniform(20, 500, n_data)

    # Mean multiplicity increases with theta1 and jet_pt
    mean_n = 20 + 8 * (theta1 + 1) + 0.02 * (jet_pt - 250)
    n = rng.poisson(mean_n)

    # Create weights: hard-pT biased, for example
    weights = np.where(jet_pt > 250, 1.5, 0.5)
    weights /= weights.mean()

    # Split data
    n_train = int(0.8 * n_data)
    train_idx = rng.permutation(n_data)[:n_train]
    val_idx = np.setdiff1d(np.arange(n_data), train_idx)

    # Create layout for the count model itself -- never n_constituents as known context
    lay = layout_for_multiplicity_model(
        ["a", "b"],
        [-1, -1],
        [1, 1],
        known_context={"jet_pt": None},
        known_scales={"jet_pt": 5.0}
    )

    # Assemble conditions
    cand_train = np.column_stack([theta1[train_idx], theta2[train_idx]])
    ctx_train = jet_pt[train_idx, None]
    cond_train = lay.assemble_torch(cand_train, "cpu", context=ctx_train,
                                   context_names=["jet_pt"])

    cand_val = np.column_stack([theta1[val_idx], theta2[val_idx]])
    ctx_val = jet_pt[val_idx, None]
    cond_val = lay.assemble_torch(cand_val, "cpu", context=ctx_val,
                                 context_names=["jet_pt"])

    # Train with weights
    model, hist = train_multiplicity(
        MultiplicityModel(lay.dim, max_n=80, hidden=32, depth=2),
        cond_train, n[train_idx],
        cond_val, n[val_idx],
        steps=600, batch_size=512, lr=1e-3, log=lambda s: None,
        weights=torch.from_numpy(weights[train_idx]).float(),
        val_weights=torch.from_numpy(weights[val_idx]).float(),
        seed=0
    )
    model.layout = lay

    # Model should learn the distribution
    assert hist["best_val_nll"] < 4.0  # entropy of Poisson(30) ~ 3.4 nats
    assert torch.isfinite(torch.tensor(hist["val_nll"])).all()


def test_multiplicity_weighted_vs_unweighted_distribution():
    """Weighted training recovers a different distribution than unweighted."""
    rng = np.random.default_rng(112)
    n_data = 6000

    theta = rng.uniform(-1, 1, n_data)

    # Generate two distributions:
    # A: Poisson(15)
    n_a = rng.poisson(15, n_data)

    # B: Poisson(25)
    n_b = rng.poisson(25, n_data)

    # Combine A with reweighting to B
    from scipy import stats
    weights = stats.poisson.pmf(n_a, 25) / np.maximum(stats.poisson.pmf(n_a, 15), 1e-10)
    weights /= weights.mean()

    # Split
    n_train = int(0.8 * n_data)
    train_idx = rng.permutation(n_data)[:n_train]
    val_idx = np.setdiff1d(np.arange(n_data), train_idx)

    cond_train = torch.zeros(n_train, 1, dtype=torch.float32)
    cond_val = torch.zeros(len(val_idx), 1, dtype=torch.float32)

    # Train WITH weights
    model_weighted, _ = train_multiplicity(
        MultiplicityModel(1, max_n=60, hidden=32, depth=2),
        cond_train, n_a[train_idx],
        cond_val, n_a[val_idx],
        steps=500, batch_size=512, lr=1e-3, log=lambda s: None,
        weights=torch.from_numpy(weights[train_idx]).float(),
        val_weights=torch.from_numpy(weights[val_idx]).float(),
        seed=0
    )

    # Train WITHOUT weights
    model_unweighted, _ = train_multiplicity(
        MultiplicityModel(1, max_n=60, hidden=32, depth=2),
        cond_train, n_a[train_idx],
        cond_val, n_a[val_idx],
        steps=500, batch_size=512, lr=1e-3, log=lambda s: None,
        weights=None,
        seed=0
    )

    # Sample from both and check means
    with torch.no_grad():
        test_cond = torch.zeros(2000, 1, dtype=torch.float32)
        samples_weighted = model_weighted.sample(test_cond).numpy()
        samples_unweighted = model_unweighted.sample(test_cond).numpy()

    # Weighted model should have higher mean (closer to 25)
    mean_weighted = samples_weighted.mean()
    mean_unweighted = samples_unweighted.mean()

    assert mean_weighted > 17, f"weighted mean {mean_weighted} too low"
    assert mean_unweighted < 23, f"unweighted mean {mean_unweighted} too high"
    assert mean_weighted > mean_unweighted + 2, \
        f"weighted {mean_weighted} not sufficiently > unweighted {mean_unweighted}"


def test_multiplicity_unit_weights_reproduce_unweighted():
    """Training with unit weights produces same result as unweighted."""
    rng = np.random.default_rng(113)
    n_data = 4000

    theta = rng.uniform(-1, 1, n_data)
    n = rng.poisson(20 + 5 * (theta + 1))

    n_train = int(0.8 * n_data)
    train_idx = rng.permutation(n_data)[:n_train]
    val_idx = np.setdiff1d(np.arange(n_data), train_idx)

    cond_train = torch.zeros(n_train, 1, dtype=torch.float32)
    cond_val = torch.zeros(len(val_idx), 1, dtype=torch.float32)

    # Train unweighted with seed 55
    torch.manual_seed(55)
    model_unweighted, hist_unweighted = train_multiplicity(
        MultiplicityModel(1, max_n=64, hidden=32, depth=2),
        cond_train, n[train_idx],
        cond_val, n[val_idx],
        steps=400, batch_size=512, lr=1e-3, log=lambda s: None,
        weights=None, seed=55
    )

    # Train with unit weights and same seed
    unit_weights = torch.ones(n_train, dtype=torch.float32)
    torch.manual_seed(55)
    model_unit, hist_unit = train_multiplicity(
        MultiplicityModel(1, max_n=64, hidden=32, depth=2),
        cond_train, n[train_idx],
        cond_val, n[val_idx],
        steps=400, batch_size=512, lr=1e-3, log=lambda s: None,
        weights=unit_weights, seed=55
    )

    # Should produce identical results
    with torch.no_grad():
        test_cond = torch.randn(10, 1, dtype=torch.float32)
        test_n = torch.randint(0, 60, (10,))
        lp_unweighted = model_unweighted.log_prob(test_n, test_cond)
        lp_unit = model_unit.log_prob(test_n, test_cond)

    torch.testing.assert_close(lp_unweighted, lp_unit, rtol=1e-5, atol=1e-6)


def test_multiplicity_model_layout_cannot_require_n_constituents():
    """layout_for_multiplicity_model refuses n_constituents as known context, loudly.

    A count model that must be told N before predicting it is not a model. This guards the
    distinction from layout_with_multiplicity (the constituent flow's layout), which must
    always carry n_constituents for the opposite reason -- it generates exactly N particles.
    """
    with pytest.raises(ValueError, match=MULTIPLICITY_COLUMN):
        layout_for_multiplicity_model(["a", "b"], [0, 0], [1, 1],
                                      known_context={MULTIPLICITY_COLUMN: None})
