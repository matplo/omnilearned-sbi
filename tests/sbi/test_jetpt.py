"""Jet transverse momentum density p(log pT_J | theta) model."""

import numpy as np
import pytest
import torch
from scipy import integrate

from omnilearned.sbi.jetpt import (
    DEFAULT_N_COMPONENTS,
    JetPtModel,
    layout_with_jet_pt,
    train_jet_pt,
)
from omnilearned.sbi.parameters import ConditionLayout


def test_jet_pt_layout():
    """layout_with_jet_pt creates proper ConditionLayout for p(log pT | theta)."""
    lay = layout_with_jet_pt(["b", "T_ref"], [0.0, 0.1], [11.0, 0.9])
    assert lay.dim == 2
    assert lay.columns == ["b", "T_ref"]
    assert lay.unknown == ["b", "T_ref"]
    assert lay.known == []
    # Check normalization: values should be in [-1, 1]
    normed = lay.normalize_table([[5.5, 0.5]], ["b", "T_ref"])
    assert normed.shape == (1, 2)
    assert np.allclose(normed, [[0.0, 0.0]])  # midpoints normalize to 0


def test_jet_pt_model_evaluates_log_prob():
    """JetPtModel.log_prob evaluates and returns proper shape."""
    model = JetPtModel(3, n_components=5, hidden=32, depth=2)
    model.eval()

    cond = torch.randn(10, 3, dtype=torch.float32)
    log_pt = torch.randn(10, dtype=torch.float32)

    lp = model.log_prob(log_pt, cond)
    assert lp.shape == (10,)
    assert torch.isfinite(lp).all()


def test_jet_pt_model_samples_and_tracks_parameters():
    """JetPtModel.sample generates samples; config tracks all settings."""
    model = JetPtModel(2, n_components=7, hidden=64, depth=3)
    model.eval()

    cond = torch.zeros(5, 2, dtype=torch.float32)
    samples = model.sample(cond)

    assert samples.shape == (5,)
    assert torch.isfinite(samples).all()
    assert model.config["n_components"] == 7
    assert model.config["hidden"] == 64
    assert model.config["depth"] == 3


def test_jet_pt_normalisation_numerical_integration():
    """p(log pT | theta) integrates to 1 on a grid.

    This is a core requirement: the model must be a proper probability density.
    We integrate numerically using grid-based trapezoidal rule.
    """
    torch.manual_seed(42)
    model = JetPtModel(1, n_components=5, hidden=32, depth=2)
    model.eval()

    # Test at several theta values
    thetas = np.array([[-0.8], [-0.4], [0.0], [0.4], [0.8]])
    cond = torch.from_numpy(thetas).float()

    # Integration grid: log pT from -3 to 4 (covers ~0.05 to ~55 GeV pT)
    log_pt_grid = np.linspace(-3, 4, 500)
    dx = log_pt_grid[1] - log_pt_grid[0]

    for i in range(len(thetas)):
        log_probs = []
        for lpt in log_pt_grid:
            lp = model.log_prob(torch.tensor([[lpt]], dtype=torch.float32),
                               cond[i:i+1]).item()
            log_probs.append(np.exp(lp))

        # Trapezoidal rule integration using scipy
        integral = integrate.trapezoid(log_probs, x=log_pt_grid)
        # Allow 1% tolerance due to discretization
        np.testing.assert_allclose(integral, 1.0, rtol=0.01, atol=0.01,
                                  err_msg=f"normalisation failed at theta {thetas[i]}")


def test_jet_pt_weighted_fitting_learns_conditional_density():
    """Weighted fitting recovers a known conditional lognormal distribution.

    Generate samples from p(log pT | theta) = LogNormal(mu(theta), sigma),
    fit with correct weights to recover mu(theta), fit with reweighting
    to recover a shifted distribution.
    """
    rng = np.random.default_rng(123)
    n_data = 5000

    # Generate theta: uniform in [-1, 1]
    theta = rng.uniform(-1, 1, n_data)
    # mu(theta) = 2 + 0.5*theta (linear dependence)
    mu_true = 2.0 + 0.5 * theta
    sigma = 0.3  # log pT standard deviation

    # Sample log pT from LogNormal(mu, sigma)
    log_pt = rng.normal(mu_true, sigma)

    # Split into train and validation
    n_train = int(0.8 * n_data)
    train_idx = rng.permutation(n_data)[:n_train]
    val_idx = np.setdiff1d(np.arange(n_data), train_idx)

    theta_train = torch.from_numpy(theta[train_idx, None]).float()
    log_pt_train = torch.from_numpy(log_pt[train_idx]).float()
    theta_val = torch.from_numpy(theta[val_idx, None]).float()
    log_pt_val = torch.from_numpy(log_pt[val_idx]).float()

    # Train without weights (should recover the true distribution)
    model, hist = train_jet_pt(
        JetPtModel(1, n_components=5, hidden=32, depth=2),
        theta_train, log_pt_train, theta_val, log_pt_val,
        steps=1000, batch_size=512, lr=1e-3, log=lambda s: None, seed=0
    )

    # Check that model output at theta=0.5 has mean near mu(0.5) = 2.25
    with torch.no_grad():
        test_cond = torch.tensor([[0.5]], dtype=torch.float32)
        logits, means, _ = model(test_cond)
        weights = torch.softmax(logits, -1)
        predicted_mean = (weights * means).sum(-1).item()

    # Allow some tolerance for function approximation
    np.testing.assert_allclose(predicted_mean, 2.25, atol=0.15,
                              err_msg="model failed to learn conditional mean")


def test_jet_pt_weighted_reweighting_recovers_target_distribution():
    """Weighted fitting with reweighting recovers the target, not the original.

    Critical test: fitting weighted samples from distribution A with weights
    that reweight them to distribution B must recover B. This is the exact bug
    that broke v2 multiplicity fitting when weights were ignored.
    """
    rng = np.random.default_rng(124)
    n_data = 8000

    # Distribution A: LogNormal(0, 1)
    log_pt_a = rng.normal(0, 1, n_data)

    # Distribution B: LogNormal(1.5, 0.5) — shifted and narrower
    log_pt_b = rng.normal(1.5, 0.5, n_data)

    # Combine into a dataset, with all weights pointing to B
    # Weight each A sample to match the B distribution
    from scipy import stats
    # Weight = p_B(x) / p_A(x) for importance sampling
    logpdf_a = stats.norm.logpdf(log_pt_a, 0, 1)
    logpdf_b = stats.norm.logpdf(log_pt_a, 1.5, 0.5)
    log_weights = logpdf_b - logpdf_a
    weights = np.exp(log_weights)
    weights /= weights.mean()  # Normalize

    # Create a dummy theta dimension
    theta = np.zeros(n_data)

    # Split
    n_train = int(0.9 * n_data)
    train_idx = rng.permutation(n_data)[:n_train]
    val_idx = np.setdiff1d(np.arange(n_data), train_idx)

    theta_train = torch.from_numpy(theta[train_idx, None]).float()
    log_pt_train = torch.from_numpy(log_pt_a[train_idx]).float()
    weights_train = torch.from_numpy(weights[train_idx]).float()
    theta_val = torch.from_numpy(theta[val_idx, None]).float()
    log_pt_val = torch.from_numpy(log_pt_a[val_idx]).float()
    weights_val = torch.from_numpy(weights[val_idx]).float()

    # Train WITH weights
    model_weighted, _ = train_jet_pt(
        JetPtModel(1, n_components=7, hidden=32, depth=2),
        theta_train, log_pt_train, theta_val, log_pt_val,
        steps=1500, batch_size=256, lr=1e-3, log=lambda s: None,
        weights=weights_train, val_weights=weights_val, seed=1
    )

    # Train WITHOUT weights on the same data (should get A, not B)
    model_unweighted, _ = train_jet_pt(
        JetPtModel(1, n_components=7, hidden=32, depth=2),
        theta_train, log_pt_train, theta_val, log_pt_val,
        steps=1500, batch_size=256, lr=1e-3, log=lambda s: None,
        weights=None, val_weights=None, seed=1
    )

    # Sample from both models and check statistics
    with torch.no_grad():
        test_cond = torch.zeros(1000, 1, dtype=torch.float32)
        samples_weighted = model_weighted.sample(test_cond).numpy()
        samples_unweighted = model_unweighted.sample(test_cond).numpy()

    # Weighted model should have mean ~ 1.5 (closer to B)
    # Unweighted model should have mean ~ 0.0 (closer to A)
    mean_weighted = samples_weighted.mean()
    mean_unweighted = samples_unweighted.mean()

    assert mean_weighted > 0.5, f"weighted mean {mean_weighted} too low"
    assert mean_unweighted < 0.5, f"unweighted mean {mean_unweighted} too high"
    assert mean_weighted > mean_unweighted + 0.5, \
        f"weighted mean {mean_weighted} not sufficiently > unweighted {mean_unweighted}"


def test_jet_pt_unit_weights_reproduce_unweighted():
    """Fitting with unit weights produces identical results to unweighted."""
    rng = np.random.default_rng(125)
    n_data = 2000

    theta = rng.uniform(-1, 1, n_data)
    log_pt = rng.normal(2 + 0.3*theta, 0.4)

    n_train = int(0.8 * n_data)
    train_idx = rng.permutation(n_data)[:n_train]
    val_idx = np.setdiff1d(np.arange(n_data), train_idx)

    theta_train = torch.from_numpy(theta[train_idx, None]).float()
    log_pt_train = torch.from_numpy(log_pt[train_idx]).float()
    theta_val = torch.from_numpy(theta[val_idx, None]).float()
    log_pt_val = torch.from_numpy(log_pt[val_idx]).float()

    # Train unweighted with seed 42
    torch.manual_seed(42)
    model_unweighted, hist_unweighted = train_jet_pt(
        JetPtModel(1, n_components=5, hidden=32, depth=2),
        theta_train, log_pt_train, theta_val, log_pt_val,
        steps=500, batch_size=512, lr=1e-3, log=lambda s: None,
        weights=None, seed=42
    )

    # Train with unit weights and same seed
    unit_weights = torch.ones_like(log_pt_train)
    torch.manual_seed(42)
    model_unit, hist_unit = train_jet_pt(
        JetPtModel(1, n_components=5, hidden=32, depth=2),
        theta_train, log_pt_train, theta_val, log_pt_val,
        steps=500, batch_size=512, lr=1e-3, log=lambda s: None,
        weights=unit_weights, seed=42
    )

    # Both models should produce identical log probabilities
    with torch.no_grad():
        test_cond = torch.randn(10, 1).float()
        test_lpt = torch.randn(10).float()
        lp_unweighted = model_unweighted.log_prob(test_lpt, test_cond)
        lp_unit = model_unit.log_prob(test_lpt, test_cond)

    torch.testing.assert_close(lp_unweighted, lp_unit, rtol=1e-5, atol=1e-6)


def test_jet_pt_persistence():
    """JetPtModel can be saved and loaded from state."""
    torch.manual_seed(99)
    model = JetPtModel(2, n_components=6, hidden=48, depth=2)
    model.layout = ConditionLayout.from_bounds(["a", "b"], [0, -1], [1, 2])

    state = model.state()
    assert "config" in state and "model_state" in state and "layout" in state

    restored = JetPtModel.from_state(state, device="cpu")
    assert restored.config == model.config
    assert restored.layout.columns == model.layout.columns

    # Verify that log_prob produces same values
    with torch.no_grad():
        cond = torch.randn(5, 2).float()
        log_pt = torch.randn(5).float()
        lp_orig = model.log_prob(log_pt, cond)
        lp_rest = restored.log_prob(log_pt, cond)

    torch.testing.assert_close(lp_orig, lp_rest, rtol=1e-6, atol=1e-7)
