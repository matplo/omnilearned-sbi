import copy
import pytest
import torch

from omnilearned.sbi import (
    GaussianBase,
    SmoothVelocity,
    LikelihoodOptions,
    log_prob,
    resolve_device,
)
from omnilearned.sbi.training import flow_matching_step


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS unavailable in this execution context",
)
def test_mps_smooth_model_training_exact_divergence_cpu_agreement():
    device = resolve_device("mps")
    torch.manual_seed(615)
    cpu = SmoothVelocity(features=2, conditions=1, width=8, depth=1, heads=2).float()
    gpu = copy.deepcopy(cpu).to(device)
    x = torch.randn(2, 3, 2)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    condition = torch.tensor([[0.2], [-0.1]])
    base = GaussianBase((0.1, -0.2), (1.1, 0.8))
    for model, selected in ((cpu, "cpu"), (gpu, device)):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        flow_matching_step(
            model,
            x.to(selected),
            condition.to(selected),
            mask.to(selected),
            base=base,
            optimizer=optimizer,
            generator=torch.Generator().manual_seed(615),
        )
    kwargs = dict(
        event_ids=["gpu-a", "gpu-b"],
        base=base,
        options=LikelihoodOptions(steps=4, divergence="exact"),
    )
    reference = log_prob(cpu, x, condition, mask=mask, **kwargs)[0]
    actual, diagnostics = log_prob(
        gpu, x.to(device), condition.to(device), mask=mask.to(device), **kwargs
    )
    torch.testing.assert_close(actual, reference, atol=2e-4, rtol=1e-4)
    assert diagnostics["device"].startswith("mps")
    assert actual.device.type == "cpu" and actual.dtype == torch.float64
    corrected, _ = log_prob(
        gpu,
        x.to(device),
        condition.to(device),
        mask=mask.to(device),
        log_abs_det_jacobian=torch.tensor([0.3, -0.2], device=device),
        **kwargs,
    )
    torch.testing.assert_close(
        corrected - actual, torch.tensor([0.3, -0.2], dtype=torch.float64)
    )
