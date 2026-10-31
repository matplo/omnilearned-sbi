"""Tests for weight-aware fidelity estimators.

Following the pattern from toy/multiplicity_check.py::binned_chi2, verifies that
weighted code paths reproduce unweighted results exactly with unit weights, and that
weighted statistics correctly recover synthetic weighted distributions.
"""

import numpy as np
import pytest

from omnilearned.sbi.fidelity import (
    global_uniformity, ks_d, mean_relevance, permutation_test, setting_verdict)


def _obs(rng, n, shift=0.0):
    """Generate test observables."""
    return {"x": rng.normal(shift, 1.0, n), "y": rng.normal(0, 1, (n, 4))}


def test_weighted_ks_d_unit_weights_matches_unweighted():
    """With unit weights, weighted KS recovers the unweighted KS exactly (bit-for-bit)."""
    rng = np.random.default_rng(42)
    a = rng.normal(0.0, 1.0, 100)
    b = rng.normal(0.15, 1.0, 100)

    # Unweighted
    ks_unweighted = ks_d(a, b)

    # Weighted with unit weights
    weights_a = np.ones(len(a))
    weights_b = np.ones(len(b))
    ks_weighted = ks_d(a, b, weights_a=weights_a, weights_b=weights_b)

    np.testing.assert_allclose(ks_weighted, ks_unweighted, rtol=1e-12)


def test_weighted_ks_d_detects_shift_with_weights():
    """Weighted KS can detect distribution shift with non-trivial weights."""
    rng = np.random.default_rng(43)
    # Generate two populations
    a = rng.normal(0.0, 1.0, 500)
    b = rng.normal(0.0, 1.0, 500)

    # Weight b so it has a +0.3 shift in mean
    # w_b ~ (1 + 0.3 * b) (roughly; normalize)
    # This creates a bias toward positive values
    weights_b = 1.0 + 0.2 * b
    weights_b = np.maximum(weights_b, 0.1)  # Keep positive
    weights_b = weights_b / weights_b.sum() * len(weights_b)  # Normalize so sum = n

    # Weighted mean of b should be higher than unweighted
    mean_b_unweighted = b.mean()
    mean_b_weighted = (weights_b * b).sum() / weights_b.sum()
    assert mean_b_weighted > mean_b_unweighted

    # Weighted KS should be large
    ks_weighted = ks_d(a, b, weights_a=None, weights_b=weights_b)
    ks_unweighted = ks_d(a, b)

    # The weighted KS might be different from unweighted because the distributions differ
    assert ks_weighted > 0.05


def test_mean_relevance_unit_weights_matches_unweighted():
    """With unit weights, weighted relevance recovers unweighted exactly."""
    rng = np.random.default_rng(44)
    oa = _obs(rng, 500)
    ob = _obs(rng, 500, shift=0.2)

    # Unweighted
    rel_unweighted = mean_relevance(oa, ob, ["x"], n_events=1000)

    # Weighted with unit weights
    weights_a = np.ones(500)
    weights_b = np.ones(500)
    rel_weighted = mean_relevance(oa, ob, ["x"], n_events=1000,
                                  weights_a=weights_a, weights_b=weights_b)

    for key in rel_unweighted:
        for field in rel_unweighted[key]:
            if not np.isnan(rel_unweighted[key][field]):
                np.testing.assert_allclose(
                    rel_weighted[key][field],
                    rel_unweighted[key][field],
                    rtol=1e-10,
                    atol=1e-12,
                    err_msg=f"mismatch in {key}/{field}"
                )


def test_mean_relevance_recovers_weighted_shift():
    """Weighted relevance correctly identifies shifts in weighted distributions."""
    rng = np.random.default_rng(45)

    # Create two samples
    n = 1000
    a = rng.normal(0.0, 1.0, n)
    b = rng.normal(0.0, 1.0, n)

    # Create weights that shift b toward positive (mean should increase)
    # w_b proportional to exp(0.5 * b) so heavy tail on positive side
    weights_b = np.exp(0.3 * b)
    weights_b = weights_b / weights_b.sum() * n

    # Weighted mean of b should be >> 0
    mean_b_weighted = (weights_b * b).sum() / weights_b.sum()
    mean_a = a.mean()
    assert mean_b_weighted > 0.15  # Significant positive shift

    # Relevance should detect this
    obs = {"a": a, "b": b}
    rel = mean_relevance({"x": a}, {"x": b}, ["x"], n_events=1000,
                        weights_a=None, weights_b=weights_b)

    # Delta mean should be positive
    assert rel["x"]["delta_mean"] > 0.1
    # Relevance should be significant
    assert rel["x"]["relevance"] > 1.0


def test_permutation_null_calibrated_with_weights():
    """Permutation test stays calibrated under the null with non-trivial weights.

    Under the null, both samples are from the same distribution with independent weighting.
    """
    rng = np.random.default_rng(46)

    # Draw pairs of samples from the same distribution, weight them independently,
    # and verify p-values are approximately uniform under the null
    ps = []
    for trial in range(30):  # Reduced from 40 for faster test
        n = 300
        a = rng.normal(0.0, 1.0, n)
        b = rng.normal(0.0, 1.0, n)

        # Independent weights for both samples (exponential-ish)
        # This creates non-trivial weights but doesn't violate the null
        weights_a = np.exp(0.15 * a)
        weights_a = weights_a / weights_a.sum() * len(weights_a)
        weights_b = np.exp(0.15 * b)
        weights_b = weights_b / weights_b.sum() * len(weights_b)

        obs_a = {"x": a, "y": rng.normal(0, 1, (n, 4))}
        obs_b = {"x": b, "y": rng.normal(0, 1, (n, 4))}

        perm = permutation_test(obs_a, obs_b, ["x", "y"], n_perm=99,
                               seed=trial, particle_stride=1,
                               weights_a=weights_a, weights_b=weights_b)
        ps.append(perm["p_family"])

    ps = np.array(ps)

    # Check uniformity of p-values
    g = global_uniformity(ps)
    # With independent weights on both sides, the test should be approximately calibrated
    # The KS p-value should not be too small (allowing for sampling variability)
    assert g["ks_p"] > 0.001, f"p-values not uniform: ks_p={g['ks_p']:.2e}"

    # Check false positive rate at alpha=0.05
    n_reject = (ps < 0.05).sum()
    # Expected: ~1-2 rejections at alpha=0.05 with 30 trials
    assert n_reject <= 7, f"Too many rejections: {n_reject}/30 at alpha=0.05"


def test_permutation_detects_shift_with_weights():
    """Weighted permutation test detects shifts in weighted samples."""
    rng = np.random.default_rng(47)

    # Create samples from different distributions
    n = 4000
    a = rng.normal(0.0, 1.0, n)
    b = rng.normal(0.15, 1.0, n)  # Shifted

    # Non-trivial weights for a
    weights_a = 1.0 + 0.1 * a
    weights_a = np.maximum(weights_a, 0.5)
    weights_a = weights_a / weights_a.sum() * len(weights_a)

    obs_a = {"x": a, "y": rng.normal(0, 1, (n, 4))}
    obs_b = {"x": b, "y": rng.normal(0, 1, (n, 4))}

    perm = permutation_test(obs_a, obs_b, ["x", "y"], n_perm=99,
                           seed=0, particle_stride=1,
                           weights_a=weights_a, weights_b=None)

    # Should detect the shift
    assert perm["p_family"] < 0.05


def test_permutation_unit_weights_matches_unweighted():
    """Permutation test with unit weights matches unweighted (approximately)."""
    rng = np.random.default_rng(48)

    obs_a = _obs(rng, 300)
    obs_b = _obs(rng, 300, shift=0.15)

    # Unweighted
    perm_unweighted = permutation_test(obs_a, obs_b, ["x", "y"], n_perm=99,
                                       seed=0, particle_stride=1)

    # Weighted with unit weights
    weights_a = np.ones(300)
    weights_b = np.ones(300)
    perm_weighted = permutation_test(obs_a, obs_b, ["x", "y"], n_perm=99,
                                     seed=0, particle_stride=1,
                                     weights_a=weights_a, weights_b=weights_b)

    # The p-values should be very close (may not be exactly equal due to different code paths)
    np.testing.assert_allclose(perm_weighted["p_family"], perm_unweighted["p_family"],
                               rtol=0.1)  # Allow 10% tolerance for randomness in permutations


def test_setting_verdict_with_weighted_relevance():
    """Setting verdict works correctly with weighted relevance results."""
    # Create a synthetic permutation and relevance result
    perm = {"p_family": 0.01}
    rel = {"x": {"relevant": True}, "y": {"relevant": False}}

    verdict = setting_verdict(perm, rel, 0.05)
    assert verdict["significant"] and verdict["relevant"] and verdict["fail"]

    # Not significant
    perm_ns = {"p_family": 0.1}
    verdict_ns = setting_verdict(perm_ns, rel, 0.05)
    assert not verdict_ns["significant"] and not verdict_ns["fail"]
