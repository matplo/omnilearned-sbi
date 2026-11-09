"""toy.train.joint_strata + prepare(..., strata=...): the P4 halo-deficit mitigation's
training-side wiring (FINDINGS_JEWEL_V3.md, "the beyond-R=0.4 halo deficit correlates with
T_ref directly"). flow.stratified_weights itself is tested in test_weighted_training.py.
"""

import numpy as np
import pytest
import torch

from omnilearned.sbi.parameters import ConditionLayout
from omnilearned.sbi.toy.train import joint_strata, prepare
from omnilearned.sbi.transforms import FeatureTransform


def test_joint_strata_every_event_gets_a_bin_no_empties():
    rng = np.random.default_rng(0)
    n = rng.integers(2, 150, 2000)
    t_ref = rng.uniform(0.1, 0.9, 2000)
    strata = joint_strata(n, t_ref, n_bins=6, t_bins=4)
    uniq, counts = np.unique(strata, return_counts=True)
    assert np.all(counts > 0)
    assert len(uniq) <= 6 * 4


def test_joint_strata_is_monotone_in_each_axis_within_the_other_fixed():
    # same T_ref bin throughout (all zeros) -- n_bin should increase with n
    n = np.array([1, 1, 50, 50, 149, 149])
    t_ref = np.zeros(6)
    strata = joint_strata(n, t_ref, n_bins=3, t_bins=1)
    assert strata[0] == strata[1] < strata[2] == strata[3] < strata[4] == strata[5]


def test_prepare_requires_weights_for_strata():
    identity = FeatureTransform(["a", "b"], [None, None], [0.0, 0.0], [1.0, 1.0])
    layout = ConditionLayout.from_bounds(["b", "T_ref", "tau_init"], [0, 0, 0], [1, 1, 1])
    data = {"x": np.zeros((4, 2, 2)), "mask": np.ones((4, 2), bool),
            "theta": np.tile([0.5, 0.5, 0.5], (4, 1)), "weight": np.ones(4),
            "n_constituents": np.array([1, 1, 2, 2])}
    with pytest.raises(ValueError, match="weights=True"):
        prepare(data, identity, layout, weights=False, strata=np.array([0, 0, 1, 1]))


def test_prepare_attaches_sample_w_matching_stratified_weights():
    from omnilearned.sbi.flow import stratified_weights

    identity = FeatureTransform(["a", "b"], [None, None], [0.0, 0.0], [1.0, 1.0])
    layout = ConditionLayout.from_bounds(["b", "T_ref", "tau_init"], [0, 0, 0], [1, 1, 1])
    weight = np.array([1.0, 2.0, 3.0, 4.0])
    data = {"x": np.zeros((4, 2, 2)), "mask": np.ones((4, 2), bool),
            "theta": np.tile([0.5, 0.5, 0.5], (4, 1)), "weight": weight,
            "n_constituents": np.array([1, 1, 2, 2])}
    strata = np.array([0, 0, 1, 1])
    out = prepare(data, identity, layout, weights=True, strata=strata)
    expect, _ = stratified_weights(weight, strata)
    torch.testing.assert_close(out["sample_w"], torch.as_tensor(expect, dtype=torch.float32))
    np.testing.assert_allclose(out["w"].numpy(), weight, rtol=1e-6)  # 'w' stays physical
