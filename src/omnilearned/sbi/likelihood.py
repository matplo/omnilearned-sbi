"""Fixed-mask probability-flow densities, data time 0 to base time 1.

The velocity contract is ``model(x, time_vector, condition, fixed_mask)`` and
must be batch-separable and deterministic in evaluation mode. Returned values
are CPU float64, one per event; no dataset reduction is implicit.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import math
import time

import torch


@dataclass(frozen=True)
class LikelihoodOptions:
    steps: int = 64
    solver: str = "rk4"
    divergence: str = "hutchinson"
    probes: int = 4
    probe_distribution: str = "rademacher"
    seed: int = 49193
    max_forward_evaluations: int = 4096
    trace_chunk_size: int = 1

    def __post_init__(self):
        if self.solver not in ("rk4", "midpoint") or self.divergence not in (
            "exact",
            "hutchinson",
        ):
            raise ValueError("Unsupported solver/divergence")
        if self.probe_distribution not in ("rademacher", "normal"):
            raise ValueError("Unsupported probe distribution")
        if not isinstance(self.trace_chunk_size, int) or self.trace_chunk_size <= 0:
            raise ValueError("Positive integer trace_chunk_size required")
        if (
            not isinstance(self.steps, int)
            or not isinstance(self.probes, int)
            or self.steps <= 0
            or self.probes <= 0
        ):
            raise ValueError("Positive integer steps/probes required")
        if (
            self.steps * (4 if self.solver == "rk4" else 2)
            > self.max_forward_evaluations
        ):
            raise ValueError("ODE evaluation budget exceeded before integration")


@contextmanager
def evaluation_state(model):
    states = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        yield
    finally:
        # Restore mixed submodule train/eval states exactly. Parameter flags and
        # existing parameter gradients are never changed by this evaluator.
        for module, training in states:
            module.training = training


def validate_inputs(x, mask, event_ids):
    if x.ndim != 3 or x.shape[0] == 0 or x.shape[-1] == 0 or not x.is_floating_point():
        raise ValueError(
            "Expected a nonempty floating [events, particles, features] tensor"
        )
    mask = torch.as_tensor(mask, device=x.device)
    if (
        mask.dtype != torch.bool
        or mask.shape != x.shape[:2]
        or not mask.any(dim=1).all()
    ):
        raise ValueError(
            "A fixed boolean [events, particles] mask with nonempty events is required"
        )
    ids = [str(v) for v in event_ids]
    if len(ids) != x.shape[0] or len(set(ids)) != len(ids):
        raise ValueError("Unique stable event IDs required")
    x = torch.where(mask[..., None], x.detach(), 0.0)
    if not torch.isfinite(x).all():
        raise ValueError("Nonfinite observed coordinates")
    if x.device.type == "mps" and x.dtype != torch.float32:
        raise ValueError("MPS model coordinates must be float32")
    return x, mask, ids


def make_probes(x, mask, event_ids, options):
    probes = []
    for probe in range(options.probes):
        events = []
        for event_id in event_ids:
            key = f"{options.seed}\0{probe}\0{event_id}".encode()
            seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little") % (
                2**63 - 1
            )
            generator = torch.Generator(device="cpu").manual_seed(seed)
            u = torch.rand(x.shape[1:], generator=generator, dtype=torch.float64)
            if options.probe_distribution == "rademacher":
                noise = torch.where(u < 0.5, -1.0, 1.0)
            else:
                noise = math.sqrt(2) * torch.erfinv(2 * u.clamp(1e-12, 1 - 1e-12) - 1)
            events.append(noise)
        probes.append(
            torch.stack(events).to(device=x.device, dtype=x.dtype) * mask[..., None]
        )
    return probes


def _step(rhs, t, x, h, solver):
    """Return an updated state and integrated per-event divergence."""
    v1, d1 = rhs(t, x)
    if solver == "midpoint":
        v2, d2 = rhs(t + h / 2, x + h * v1 / 2)
        return x + h * v2, h * d2
    v2, d2 = rhs(t + h / 2, x + h * v1 / 2)
    v3, d3 = rhs(t + h / 2, x + h * v2 / 2)
    v4, d4 = rhs(t + h, x + h * v3)
    return x + h * (v1 + 2 * v2 + 2 * v3 + v4) / 6, h * (d1 + 2 * d2 + 2 * d3 + d4) / 6


def log_prob(
    model,
    observations,
    candidate_parameters,
    *,
    mask,
    event_ids,
    base,
    options=None,
    condition_spec=None,
    context=None,
    log_abs_det_jacobian=None,
):
    options = options or LikelihoodOptions()
    start = time.perf_counter()
    x, mask, ids = validate_inputs(observations, mask, event_ids)
    if options.trace_chunk_size > 1 and x.device.type != "cpu":
        raise ValueError("Vectorized trace chunks currently require explicit CPU execution")
    if condition_spec is not None:
        condition = condition_spec.assemble(
            candidate_parameters, context, len(x), device=x.device, dtype=x.dtype
        )
    else:
        condition = torch.as_tensor(
            candidate_parameters, device=x.device, dtype=x.dtype
        )
        if condition.ndim == 1:
            condition = condition[None].expand(len(x), -1)
        if (
            condition.ndim != 2
            or condition.shape[0] != len(x)
            or not torch.isfinite(condition).all()
        ):
            raise ValueError("Condition tensor must have shape [events, conditions]")
    probes = (
        make_probes(x, mask, ids, options) if options.divergence == "hutchinson" else []
    )
    forward_calls = 0
    derivative_calls = 0

    def rhs(t, state):
        nonlocal forward_calls, derivative_calls
        forward_calls += 1
        if forward_calls > options.max_forward_evaluations:
            raise RuntimeError("ODE forward evaluation limit reached")
        with torch.enable_grad():
            coordinates = state.detach().requires_grad_(True)
            velocity = model(
                coordinates, coordinates.new_full((len(x),), t), condition, mask
            )
            if velocity.shape != coordinates.shape:
                raise ValueError("Velocity shape mismatch")
            velocity = torch.where(mask[..., None], velocity, 0.0)
            divergence = coordinates.new_zeros(len(x))
            if velocity.requires_grad:
                if options.trace_chunk_size > 1:
                    dimensions = x.shape[1]*x.shape[2]
                    vectors = (
                        torch.eye(dimensions, device=x.device, dtype=x.dtype)
                        .view(dimensions, 1, x.shape[1], x.shape[2])
                        .expand(-1, len(x), -1, -1)
                        if options.divergence == "exact" else torch.stack(probes)
                    )
                    for offset in range(0, len(vectors), options.trace_chunk_size):
                        basis = vectors[offset:offset+options.trace_chunk_size]
                        derivative_calls += len(basis)
                        derivative = torch.autograd.grad(
                            velocity, coordinates, grad_outputs=basis,
                            is_grads_batched=True, retain_graph=True, allow_unused=True,
                        )[0]
                        if derivative is None:
                            continue
                        if options.divergence == "exact":
                            index = torch.arange(offset, offset+len(basis), device=x.device)
                            diagonal = derivative.reshape(len(basis), len(x), dimensions)[
                                torch.arange(len(basis), device=x.device), :, index
                            ]
                            divergence = divergence+diagonal.sum(dim=0)
                        else:
                            divergence = divergence+(derivative*basis).sum(dim=(0, 2, 3))/len(probes)
                elif options.divergence == "exact":
                    for particle in range(x.shape[1]):
                        if not mask[:, particle].any():
                            continue
                        for feature in range(x.shape[2]):
                            derivative_calls += 1
                            derivative = torch.autograd.grad(
                                velocity[:, particle, feature].sum(),
                                coordinates,
                                retain_graph=True,
                                allow_unused=True,
                            )[0]
                            if derivative is not None:
                                divergence = (
                                    divergence
                                    + derivative[:, particle, feature]
                                    * mask[:, particle]
                                )
                else:
                    for noise in probes:
                        derivative_calls += 1
                        derivative = torch.autograd.grad(
                            (velocity * noise).sum(),
                            coordinates,
                            retain_graph=True,
                            allow_unused=True,
                        )[0]
                        if derivative is not None:
                            divergence = divergence + (derivative * noise).sum(
                                dim=(1, 2)
                            ) / len(probes)
        if not torch.isfinite(velocity).all() or not torch.isfinite(divergence).all():
            raise FloatingPointError(f"Nonfinite velocity/divergence at t={t}")
        return velocity.detach(), divergence.detach()

    integral = torch.zeros(len(x), dtype=torch.float64, device="cpu")
    with evaluation_state(model):
        for index in range(options.steps):
            x, correction = _step(
                rhs, index / options.steps, x, 1 / options.steps, options.solver
            )
            x = torch.where(mask[..., None], x.detach(), 0.0)
            integral += correction.cpu().to(torch.float64)
            if not torch.isfinite(x).all():
                raise FloatingPointError("Nonfinite integration trajectory")
    base_values = base.log_prob(x, mask)
    values = base_values + integral
    if log_abs_det_jacobian is not None:
        jacobian = (
            log_abs_det_jacobian.detach().cpu().to(torch.float64)
            if torch.is_tensor(log_abs_det_jacobian)
            else torch.as_tensor(
                log_abs_det_jacobian, dtype=torch.float64, device="cpu"
            )
        )
        if jacobian.shape != values.shape or not torch.isfinite(jacobian).all():
            raise ValueError("Finite per-event transform Jacobian required")
        values += jacobian
    if not torch.isfinite(values).all():
        raise FloatingPointError("Nonfinite per-event density")
    return values, {
        "solver": options.solver,
        "steps": options.steps,
        "divergence": options.divergence,
        "probes": options.probes if probes else 0,
        "probe_distribution": options.probe_distribution,
        "probe_seed": options.seed,
        "probe_indexing": "stable_event_id",
        "trace_chunk_size": options.trace_chunk_size,
        "device": str(observations.device),
        "model_dtype": str(observations.dtype),
        "accumulation_device": "cpu",
        "accumulation_dtype": "float64",
        "forward_evaluations": forward_calls,
        "derivative_evaluations": derivative_calls,
        "seconds": time.perf_counter() - start,
        "finite": True,
        "base": base.metadata(),
        "base_log_prob": base_values.tolist(),
        "integrated_divergence": integral.tolist(),
    }


def transport(model, coordinates, condition, mask, *, start=1., end=0., steps=64, solver="rk4"):
    """Transport supplied coordinates through the unchanged velocity field.

    Callers supplying base coordinates are responsible for their checkpoint base
    measure. This enables paired Monte Carlo, trajectory tests and quadrature.
    """
    if solver not in ("rk4", "midpoint") or not isinstance(steps, int) or steps <= 0:
        raise ValueError("Invalid sampling solver/steps")
    mask = mask.to(device=condition.device, dtype=torch.bool)
    if not math.isfinite(start) or not math.isfinite(end):
        raise ValueError("Finite transport times required")
    x = torch.where(mask[..., None], coordinates.detach(), 0.)
    if x.shape[:2] != mask.shape or len(x) != len(condition) or not torch.isfinite(x).all():
        raise ValueError("Invalid transport coordinates, conditions or mask")

    def rhs(t, state):
        v = model(state, state.new_full((len(state),), t), condition, mask)
        v = torch.where(mask[..., None], v, 0.0)
        if not torch.isfinite(v).all():
            raise FloatingPointError("Nonfinite sampling velocity")
        return v, state.new_zeros(len(state))

    with evaluation_state(model), torch.no_grad():
        for index in range(steps):
            h = (end-start)/steps
            x, _ = _step(rhs, start+index*h, x, h, solver)
            x = torch.where(mask[..., None], x, 0.0)
            if not torch.isfinite(x).all():
                raise FloatingPointError("Nonfinite transport trajectory")
    return x


def sample(model, condition, mask, *, base, steps=64, solver="rk4", generator=None):
    """Same field and base, integrated from base time 1 to data time 0."""
    mask = mask.to(device=condition.device, dtype=torch.bool)
    x = base.sample((*mask.shape, len(base.mean)), mask, device=condition.device,
                    dtype=condition.dtype, generator=generator)
    return transport(model, x, condition, mask, steps=steps, solver=solver)
