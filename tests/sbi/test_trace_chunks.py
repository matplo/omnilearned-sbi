import math
import pytest
import torch
from omnilearned.sbi import GaussianBase, LikelihoodOptions, SmoothVelocity, log_prob, transport


@pytest.mark.parametrize("divergence", ["exact", "hutchinson"])
def test_vectorized_trace_matches_sequential(divergence):
    torch.manual_seed(379)
    model = SmoothVelocity(features=2, conditions=1, width=8, depth=1, heads=2).double()
    x = torch.randn(2, 3, 2, dtype=torch.float64)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    condition = torch.ones(2, 1, dtype=x.dtype)
    base = GaussianBase((.1, -.2), (1.3, .7))
    values = [log_prob(model, x, condition, mask=mask, event_ids=["a", "b"], base=base,
                       options=LikelihoodOptions(steps=2, probes=5, divergence=divergence, trace_chunk_size=k))[0]
              for k in (1, 2, 5)]
    for value in values[1:]:
        torch.testing.assert_close(value, values[0], atol=1e-12, rtol=1e-12)


def test_transport_reverse_and_forward_use_same_field():
    class Linear(torch.nn.Module):
        def forward(self, x, time, condition, mask):
            return .4*x
    x = torch.tensor([[[.2], [-.3]]], dtype=torch.float64)
    c = torch.zeros(1, 0, dtype=x.dtype)
    mask = torch.ones(1, 2, dtype=torch.bool)
    generated = transport(Linear(), x, c, mask, steps=16)
    torch.testing.assert_close(generated, x*math.exp(-.4), atol=1e-9, rtol=1e-9)
    recovered = transport(Linear(), generated, c, mask, start=0., end=1., steps=16)
    torch.testing.assert_close(recovered, x, atol=1e-9, rtol=1e-9)
