"""Weighted training: weighted flow-matching loss, weighted minibatch draws, weighted p(N) NLL."""

import math

import numpy as np
import pytest
import torch
from torch import nn

from omnilearned.sbi.base import GaussianBase
from omnilearned.sbi.flow import TrainOptions, flow_matching_loss, minibatch_sampler, train_flow
from omnilearned.sbi.multiplicity import MultiplicityModel, train_multiplicity


class TinyVelocity(nn.Module):
    """MLP velocity for clouds of F features: (x, t, cond, mask) -> v, per particle."""

    def __init__(self, f=1, c=1, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(f + 1 + c, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, f))

    def forward(self, x, t, cond, mask):
        b, p, _ = x.shape
        z = torch.cat([x, t.view(-1, 1, 1).expand(b, p, 1), cond[:, None, :].expand(b, p, cond.shape[-1])], -1)
        return self.net(z) * mask[..., None].to(x.dtype)


def _clouds(n, p=4, f=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, p, f, generator=g)
    mask = torch.arange(p)[None] < torch.randint(1, p + 1, (n,), generator=g)[:, None]
    return x * mask[..., None], mask, torch.randn(n, 2, generator=g)


def test_weighted_loss_with_unit_weights_is_the_mean_exactly():
    x, mask, cond = _clouds(64)
    torch.manual_seed(0)
    model, base = TinyVelocity(3, 2), GaussianBase.standard(3)
    g = torch.Generator().manual_seed(1)
    t, eps = torch.rand(64, generator=g), base.sample_like(x, mask, g)
    plain = flow_matching_loss(model, x, mask, cond, base, t=t, eps=eps)
    ones = flow_matching_loss(model, x, mask, cond, base, t=t, eps=eps, weight=torch.ones(64))
    assert torch.equal(plain, ones)
    w = torch.rand(64, generator=g) + 0.1
    per = torch.stack([flow_matching_loss(model, x[i:i + 1], mask[i:i + 1], cond[i:i + 1], base,
                                          t=t[i:i + 1], eps=eps[i:i + 1]) for i in range(64)])
    torch.testing.assert_close(flow_matching_loss(model, x, mask, cond, base, t=t, eps=eps, weight=w),
                               (per * w).sum() / w.sum(), rtol=1e-5, atol=1e-7)


def _train(data, mode, steps=40, buckets=0):
    torch.manual_seed(3)
    model = TinyVelocity(3, 2)
    opts = TrainOptions(steps=steps, batch_size=16, val_every=20, val_batches=2, warmup_steps=5,
                        device="cpu", weight_mode=mode, length_buckets=buckets)
    best, hist = train_flow(model, GaussianBase.standard(3), data, data, opts, log=lambda m: None)
    return best.state_dict(), hist


def test_unit_weights_train_bit_identically():
    x, mask, cond = _clouds(200)
    plain = {"u": x, "mask": mask, "cond": cond}
    ref, href = _train(plain, "none")
    for mode in ("none", "loss"):
        st, h = _train(dict(plain, w=torch.ones(200)), mode)
        assert h["train_loss"] == href["train_loss"] and h["val_loss"] == href["val_loss"]
        assert all(torch.equal(ref[k], st[k]) for k in ref)
    _, h = _train(dict(plain, w=torch.ones(200)), "loss")
    assert h["weights"]["ess_train"] == pytest.approx(200) and h["weights"]["mode"] == "loss"
    with pytest.raises(ValueError, match="weights"):
        _train(plain, "loss")
    with pytest.raises(ValueError, match="weight_mode"):
        _train(dict(plain, w=torch.ones(200)), "bogus")


@pytest.mark.parametrize("buckets", [0, 3])
def test_weighted_minibatch_draws_follow_the_weights(buckets):
    g = torch.Generator().manual_seed(0)
    lengths = torch.randint(1, 30, (50,), generator=g)
    w = torch.rand(50, generator=g) ** 2
    w[7] = 0.0
    draw = minibatch_sampler(lengths, 256, buckets, torch.Generator().manual_seed(1), weights=w)
    counts = torch.bincount(torch.cat([draw() for _ in range(800)]), minlength=50).double()
    expect = w.double() / w.sum() * counts.sum()
    assert counts[7] == 0
    live = expect > 0
    assert float(((counts - expect)[live] ** 2 / expect[live]).sum()) < 49 + 5 * math.sqrt(2 * 49)
    if buckets:
        assert draw.edges is not None and len(draw.edges) == buckets
    with pytest.raises(ValueError, match="non-negative"):
        minibatch_sampler(lengths, 8, buckets, weights=-w)


def _flow_mean(model, n=4000, steps=64):
    """Mean of x from the learned velocity (Euler from the noise at t = 1 to t = 0)."""
    g = torch.Generator().manual_seed(9)
    z = torch.randn(n, 1, 1, generator=g)
    mask, cond = torch.ones(n, 1, dtype=torch.bool), torch.zeros(n, 1)
    with torch.no_grad():
        for k in range(steps, 0, -1):
            t = torch.full((n,), k / steps)
            z = z - model(z, t, cond, mask) / steps
    return float(z.mean()), float(z.std())


@pytest.mark.parametrize("mode", ["loss", "sample"])
def test_weighted_flow_fit_on_a_1d_toy_recovers_the_true_density(mode):
    """True law N(0, 1); training draws from the biased N(a, 1) = N(0, 1) e^{a x} / Z with weights
    e^{-a x}. Weighted training learns N(0, 1); unweighted training learns N(a, 1)."""
    a, n = 1.0, 20000
    g = torch.Generator().manual_seed(4)
    x = (torch.randn(n, generator=g) + a)[:, None, None]
    w = torch.exp(-a * x[:, 0, 0])
    w = w * n / w.sum()
    data = {"u": x, "mask": torch.ones(n, 1, dtype=torch.bool), "cond": torch.zeros(n, 1), "w": w}
    out = {}
    for m in (mode, "none"):
        torch.manual_seed(5)
        opts = TrainOptions(steps=1500, batch_size=512, lr=2e-3, val_every=500, val_batches=2, warmup_steps=50,
                            device="cpu", weight_mode=m, ema_decay=0.99)
        best, _ = train_flow(TinyVelocity(1, 1), GaussianBase.standard(1), data, data, opts, log=lambda s: None)
        out[m] = _flow_mean(best)
    assert abs(out[mode][0]) < 0.12 and abs(out[mode][1] - 1) < 0.12, out
    assert abs(out["none"][0] - a) < 0.12, out


def test_weighted_multiplicity_nll_recovers_the_true_pmf():
    """p(N) = Poisson(12); training N from Poisson(12) x 1.15^N, i.e. Poisson(13.8), with weights 1.15^-N."""
    rng = np.random.default_rng(0)
    n = rng.poisson(12 * 1.15, 40000)
    w = 1.15 ** -n.astype(float)
    w *= n.size / w.sum()
    cond = torch.zeros(n.size, 1)
    out = {}
    for name, wt in (("weighted", w), ("plain", None)):
        m, _ = train_multiplicity(MultiplicityModel(1, 40, hidden=16, depth=1), cond, n, cond[:2000], n[:2000],
                                  steps=600, batch_size=4096, lr=1e-2, log=lambda s: None, weights=wt,
                                  val_weights=None if wt is None else w[:2000])
        p = torch.exp(m.log_prob(torch.arange(41), torch.zeros(41, 1))).numpy()
        out[name] = float(np.sum(np.arange(41) * p))
    assert abs(out["weighted"] - 12) < 0.25, out
    assert abs(out["plain"] - 13.8) < 0.25, out
