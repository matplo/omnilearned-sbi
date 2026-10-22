"""SmoothPETVelocity: PET2 structure with smooth locality passes the smoothness checks."""

import numpy as np
import torch
import torch.nn as nn

from omnilearned.sbi.base import GaussianBase
from omnilearned.sbi.smoothness import (
    finite_difference_divergence,
    logp_refinement,
    permutation_gap,
    reconstruction_error,
)
from omnilearned.sbi.velocity import build_velocity, velocity_config
from omnilearned.sbi.velocity_pet import SmoothPETVelocity


def model(cond_dim=2, seed=0, time_max_freq=4.0):
    # Random weights on all Fourier time modes make a field that oscillates fast in t
    # (stiff, not rough); solver-convergence checks use slow modes, see below.
    torch.manual_seed(seed)
    m = SmoothPETVelocity(3, cond_dim, dim=16, depth=2, heads=2, local_dim=8, time_dim=8,
                          time_max_freq=time_max_freq).double()
    nn.init.normal_(m.out.weight, std=0.3)
    for b in m.blocks:  # LayerScale starts at 1e-3; make the blocks matter in the test
        nn.init.constant_(b.gamma1, 0.5)
        nn.init.constant_(b.gamma2, 0.5)
    return m.eval()


def inputs(B=6, P=5, seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, P, 3, generator=g, dtype=torch.float64)
    mask = torch.ones(B, P, dtype=torch.bool)
    mask[0, 3:] = False
    x = x * mask[..., None]
    cond = torch.randn(B, 2, generator=g, dtype=torch.float64)
    return x, mask, cond


def test_smooth_pet_passes_smoothness_diagnostics():
    m = model()
    x, mask, cond = inputs()
    base = GaussianBase.standard(3)
    rec = reconstruction_error(m, x, mask, cond)
    assert rec[64] < rec[8] and rec[64] < 1e-4
    # Default time modes (up to 64 cycles): stiff at random init but still converging
    # at (at least) second order once the steps resolve them.
    stiff = reconstruction_error(model(time_max_freq=64.0), x, mask, cond, steps=(128, 256, 512))
    assert stiff[256] < stiff[128] / 4 and stiff[512] < stiff[256] / 4
    ref = logp_refinement(m, x, mask, cond, base, steps=(8, 16, 32, 128))
    assert ref[32] < ref[8] / 8
    fd = finite_difference_divergence(m, x, mask, cond, h=1e-5)
    assert fd["max_abs_diff"] < 1e-5
    assert permutation_gap(m, x[1:], mask[1:], cond[1:], base) < 1e-9
    # Near-coincident particles: the log-distance feature is regularized, still smooth.
    xc = x.clone()
    xc[:, 1, :2] = xc[:, 2, :2] + 1e-6
    fd = finite_difference_divergence(m, xc, mask, cond, h=1e-7)
    assert fd["max_abs_diff"] < 1e-4


def test_smooth_pet_is_equivariant_and_ignores_padding():
    m = model()
    x, mask, cond = inputs()
    t = torch.full((x.shape[0],), 0.3, dtype=torch.float64)
    v = m(x, t, cond, mask)
    assert torch.all(v[~mask] == 0)
    perm = torch.tensor([2, 0, 1, 4, 3])
    vp = m(x[1:, perm], t[1:], cond[1:], mask[1:, perm])
    torch.testing.assert_close(vp, v[1:, perm], rtol=0, atol=1e-12)
    # Extra padded slots (any width) change no real value.
    xp = torch.cat([x, torch.zeros(x.shape[0], 7, 3, dtype=x.dtype)], 1)
    mp = torch.cat([mask, torch.zeros(x.shape[0], 7, dtype=torch.bool)], 1)
    vpad = m(xp, t, cond, mp)
    torch.testing.assert_close(vpad[:, :5], v, rtol=0, atol=1e-12)
    assert torch.all(vpad[:, 5:] == 0)
    # Values placed in padded slots are ignored too.
    xg = x.clone()
    xg[0, 3:] = 5.0
    torch.testing.assert_close(m(xg, t, cond, mask)[mask], v[mask], rtol=0, atol=1e-12)


def test_smooth_pet_config_round_trip_and_flow_checkpoint(tmp_path):
    from omnilearned.sbi.flow import FlowSurrogate
    from omnilearned.sbi.multiplicity import layout_with_multiplicity
    from omnilearned.sbi.transforms import FeatureTransform

    layout = layout_with_multiplicity(["a"], [0], [1])
    m = SmoothPETVelocity(3, layout.dim, dim=16, depth=1, heads=2, local_dim=8, time_dim=8)
    cfg = velocity_config(m)
    assert cfg["architecture"] == "pet_smooth"
    m2 = build_velocity(cfg)
    m2.load_state_dict(m.state_dict())
    tr = FeatureTransform(["eta", "phi", "lpt"], [2.5, 1.2, None], [0, 0, 1], [0.5, 0.4, 1])
    s = FlowSurrogate(m.eval(), GaussianBase.standard(3), tr, layout)
    s.save(tmp_path / "c.pt")
    s2 = FlowSurrogate.load(tmp_path / "c.pt")
    assert isinstance(s2.model, SmoothPETVelocity)
    x = np.random.default_rng(0).normal(0, 0.3, (4, 6, 3)).astype(np.float32)
    x[..., 2] += 2.0
    mask = np.ones((4, 6), bool)
    from omnilearned.sbi.likelihood import LikelihoodOptions

    opt = LikelihoodOptions(solver="midpoint", nsteps=8, divergence="exact")
    a, _ = s.log_prob(x, mask, [[0.3]], np.arange(4), opt, context=None)
    b, _ = s2.log_prob(x, mask, [[0.3]], np.arange(4), opt, context=None)
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-10)
    assert np.all(np.isfinite(a))
