"""restrict_to_core: the P4 halo-deficit mitigation (drop dR>=r_max constituents before
scoring, since the surrogate is known to be wrong there and that error correlates with
T_ref -- see FINDINGS_JEWEL_V3.md, "the beyond-R=0.4 halo deficit correlates with T_ref
directly").
"""

import numpy as np
import pytest

from omnilearned.sbi.observations import Observations, restrict_to_core

FEATURES = ("deta", "dphi", "log_pt", "log_e")


def _obs(deta, dphi, mask=None):
    deta = np.asarray(deta, np.float64)
    dphi = np.asarray(dphi, np.float64)
    n, p = deta.shape
    if mask is None:
        mask = np.ones((n, p), bool)
    x = np.zeros((n, p, 4))
    x[..., 0], x[..., 1] = deta, dphi
    x[..., 2], x[..., 3] = 1.0, 1.0  # log_pt, log_e: nonzero so a bug can't hide as "all zero"
    x[~mask] = 0.0
    return Observations(x, mask, np.arange(n, dtype=np.int64), feature_names=FEATURES)


def test_drops_constituents_beyond_r_max():
    # one jet: constituents at dR = 0.1, 0.3 (core) and 0.45, 0.49 (halo)
    obs = _obs(deta=[[0.1, 0.3, 0.45, 0.0]], dphi=[[0.0, 0.0, 0.0, 0.49]])
    out = restrict_to_core(obs, r_max=0.4)
    np.testing.assert_array_equal(out.mask, [[True, True, False, False]])
    assert np.all(out.x[0, 2:] == 0.0), "cut constituents must be zeroed, not just masked"
    np.testing.assert_array_equal(out.x[0, :2], obs.x[0, :2])  # survivors untouched


def test_already_masked_padding_stays_out():
    """A padded (already-False) slot must not be revived just because dR happens to be small."""
    mask = np.array([[True, False]])
    obs = _obs(deta=[[0.1, 0.0]], dphi=[[0.0, 0.0]], mask=mask)
    out = restrict_to_core(obs, r_max=0.4)
    np.testing.assert_array_equal(out.mask, mask)


def test_raises_without_deta_dphi():
    obs = Observations(np.zeros((2, 3, 3)), np.ones((2, 3), bool), np.arange(2),
                       feature_names=("eta", "phi", "log_pt"))
    with pytest.raises(ValueError, match="deta/dphi"):
        restrict_to_core(obs)


def test_raises_on_jets_left_with_too_few_core_constituents():
    # one jet entirely in the halo -- zero core survivors
    obs = _obs(deta=[[0.45, 0.46]], dphi=[[0.0, 0.0]])
    with pytest.raises(ValueError, match="core"):
        restrict_to_core(obs, r_max=0.4, require_min=1)


def test_require_min_zero_allows_empty_jets():
    obs = _obs(deta=[[0.45, 0.46]], dphi=[[0.0, 0.0]])
    out = restrict_to_core(obs, r_max=0.4, require_min=0)
    np.testing.assert_array_equal(out.mask, [[False, False]])


def test_context_and_group_id_carried_through_unchanged():
    obs = _obs(deta=[[0.1, 0.45]], dphi=[[0.0, 0.0]])
    obs.context = np.array([[3.5]])
    obs.context_names = ("jet_pt",)
    obs.group_id = np.array([7], dtype=np.int64)
    out = restrict_to_core(obs, r_max=0.4)
    np.testing.assert_array_equal(out.context, obs.context)
    assert out.context_names == obs.context_names
    np.testing.assert_array_equal(out.group_id, obs.group_id)
