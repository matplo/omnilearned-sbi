"""Single-process float32 conditional flow matching with the shared base."""

import torch


def flow_matching_step(
    model, observations, condition, mask, *, base, optimizer, generator
):
    if observations.device.type == "mps" and observations.dtype != torch.float32:
        raise ValueError("MPS training requires float32 model coordinates")
    model.train()
    observations = torch.where(mask[..., None], observations, 0.0)
    noise = base.sample(
        observations.shape,
        mask,
        device=observations.device,
        dtype=observations.dtype,
        generator=generator,
    )
    time = torch.rand(
        len(observations), generator=generator, device="cpu", dtype=observations.dtype
    ).to(observations.device)
    z = ((1 - time[:, None, None]) * observations + time[:, None, None] * noise) * mask[
        ..., None
    ]
    target = (noise - observations) * mask[..., None]
    prediction = model(z, time, condition, mask)
    loss = ((prediction - target).square() * mask[..., None]).sum() / (
        mask.sum() * observations.shape[-1]
    )
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite flow-matching loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if any(
        p.grad is not None and not torch.isfinite(p.grad).all()
        for p in model.parameters()
    ):
        raise FloatingPointError("Nonfinite model gradients")
    optimizer.step()
    return float(loss.detach().cpu())
