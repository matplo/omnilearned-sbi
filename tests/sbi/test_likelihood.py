import math

import numpy as np
import pytest
import torch
from torch import nn

from omnilearned.sbi import (
    GaussianBase,
    LikelihoodOptions,
    log_prob,
    sample,
    SmoothVelocity,
)


class Linear(nn.Module):
    def __init__(self, a, shift=0.0):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(a, dtype=torch.float64))
        self.shift = shift
        self.child = nn.Dropout(0.3)

    def forward(self, x, time, condition, mask):
        return self.a * x + self.shift


@pytest.mark.parametrize("a,shift", [(0.0, 0.0), (0.0, 0.7), (0.4, 0.0), (-0.7, 0.2)])
@pytest.mark.parametrize("mode", ["exact", "hutchinson"])
def test_analytic_density_sign_base_mask(a, shift, mode):
    x = torch.tensor(
        [
            [[0.1, -0.2], [0.3, 0.8], [999.0, 999.0]],
            [[0.4, 0.2], [999.0, 999.0], [999.0, 999.0]],
        ],
        dtype=torch.float64,
    )
    mask = torch.tensor([[True, True, False], [True, False, False]])
    base = GaussianBase((0.3, -0.4), (1.3, 0.7))
    model = Linear(a, shift)
    value, diagnostic = log_prob(
        model,
        x,
        [],
        mask=mask,
        event_ids=["a", "b"],
        base=base,
        options=LikelihoodOptions(steps=16, divergence=mode),
    )
    z = torch.where(
        mask[..., None],
        x * math.exp(a) + (shift * math.expm1(a) / a if a else shift),
        0.0,
    )
    exact = base.log_prob(z, mask) + a * mask.sum(dim=1) * 2
    torch.testing.assert_close(value, exact, atol=3e-7, rtol=1e-7)
    assert (
        value.shape == (2,)
        and value.dtype == torch.float64
        and value.device.type == "cpu"
    )
    assert diagnostic["finite"]


def test_model_state_restored_including_on_failure():
    model = Linear(0.3)
    model.train()
    model.child.eval()
    model.a.requires_grad_(False)
    saved = model.a.detach().clone()
    x = torch.ones(1, 2, 1, dtype=torch.float64)
    kwargs = dict(
        mask=torch.ones(1, 2, dtype=torch.bool),
        event_ids=["event"],
        base=GaussianBase((0.0,), (1.0,)),
        options=LikelihoodOptions(steps=2, divergence="exact"),
    )
    with torch.no_grad():
        log_prob(model, x, [], **kwargs)
    assert model.training and not model.child.training and not model.a.requires_grad
    torch.testing.assert_close(saved, model.a)
    model.shift = float("nan")
    with pytest.raises(FloatingPointError):
        log_prob(model, x, [], **kwargs)
    assert model.training and not model.child.training and not model.a.requires_grad


def test_batch_and_padding_probes_are_stable():
    class Coupled(nn.Module):
        def forward(self, x, t, c, mask):
            return 0.1 * x + 0.2 * x.flip(-1)

    x = torch.arange(24, dtype=torch.float64).reshape(4, 3, 2) / 20
    mask = torch.ones(4, 3, dtype=torch.bool)
    ids = ["1", "2", "3", "4"]
    opts = LikelihoodOptions(steps=4, probes=3)
    base = GaussianBase((0.0, 0.0), (1.0, 1.0))
    full = log_prob(
        Coupled(), x, [], mask=mask, event_ids=ids, base=base, options=opts
    )[0]
    chunks = torch.cat(
        [
            log_prob(
                Coupled(),
                x[i : i + 1],
                [],
                mask=mask[i : i + 1],
                event_ids=ids[i : i + 1],
                base=base,
                options=opts,
            )[0]
            for i in range(4)
        ]
    )
    torch.testing.assert_close(full, chunks, atol=1e-13, rtol=1e-13)
    padded = torch.cat([x, torch.full((4, 2, 2), float("nan"), dtype=x.dtype)], dim=1)
    padded_mask = torch.cat([mask, torch.zeros(4, 2, dtype=torch.bool)], dim=1)
    value = log_prob(
        Coupled(), padded, [], mask=padded_mask, event_ids=ids, base=base, options=opts
    )[0]
    torch.testing.assert_close(value, full, atol=1e-13, rtol=1e-13)


def test_refinement_tightens_error_and_normalizes_density():
    base = GaussianBase((0.0,), (1.0,))
    x = torch.linspace(-5, 5, 1001, dtype=torch.float64).view(-1, 1, 1)
    mask = torch.ones(len(x), 1, dtype=torch.bool)
    ids = list(map(str, range(len(x))))
    values = []
    for steps in (2, 8):
        values.append(
            log_prob(
                Linear(0.8),
                x,
                [],
                mask=mask,
                event_ids=ids,
                base=base,
                options=LikelihoodOptions(steps=steps, divergence="exact"),
            )[0]
        )
    exact = base.log_prob(x * math.exp(0.8), mask) + 0.8
    assert (values[1] - exact).abs().max() < (values[0] - exact).abs().max() / 100
    assert np.trapezoid(values[1].exp().numpy(), x.flatten().numpy()) == pytest.approx(
        1.0, abs=2e-6
    )


def test_sampling_uses_the_same_base_and_reverse_sign():
    base = GaussianBase((0.2,), (1.4,))
    mask = torch.ones(3, 2, dtype=torch.bool)
    cond = torch.empty(3, 0, dtype=torch.float64)
    z = base.sample(
        (3, 2, 1),
        mask,
        device="cpu",
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(9),
    )
    data = sample(
        Linear(0.5),
        cond,
        mask,
        base=base,
        steps=16,
        generator=torch.Generator().manual_seed(9),
    )
    torch.testing.assert_close(data, z * math.exp(-0.5), atol=1e-8, rtol=1e-8)


def test_smooth_attention_fixed_mask_permutation_derivatives():
    torch.manual_seed(63)
    model = SmoothVelocity(features=2, conditions=1, width=8, depth=1, heads=2).double()
    x = torch.randn(1, 3, 2, dtype=torch.float64)
    mask = torch.tensor([[True, True, False]])
    cond = torch.ones(1, 1, dtype=torch.float64)
    base = GaussianBase((0.0, 0.0), (1.0, 1.0))
    opts = LikelihoodOptions(steps=2, divergence="exact")
    a = log_prob(model, x, cond, mask=mask, event_ids=["p"], base=base, options=opts)[0]
    perm = torch.tensor([1, 2, 0])
    b = log_prob(
        model,
        x[:, perm],
        cond,
        mask=mask[:, perm],
        event_ids=["p"],
        base=base,
        options=opts,
    )[0]
    torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
    x.requires_grad_(True)
    velocity = model(x, torch.tensor([0.5], dtype=x.dtype), cond, mask)
    exact = torch.autograd.grad(velocity[0, 0, 0], x)[0][0, 0, 0]
    plus, minus = x.detach().clone(), x.detach().clone()
    plus[0, 0, 0] += 1e-6
    minus[0, 0, 0] -= 1e-6
    finite = (
        model(plus, x.new_tensor([0.5]), cond, mask)[0, 0, 0]
        - model(minus, x.new_tensor([0.5]), cond, mask)[0, 0, 0]
    ) / 2e-6
    torch.testing.assert_close(exact, finite, atol=1e-8, rtol=1e-6)


def test_budget_is_enforced():
    with pytest.raises(ValueError, match="budget"):
        LikelihoodOptions(steps=100, max_forward_evaluations=10)
