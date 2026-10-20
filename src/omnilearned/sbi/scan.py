"""Whole-ensemble joint grids and per-event local-variable integration."""

from dataclasses import dataclass
import itertools
import math
import time

import numpy as np
import torch


@dataclass(frozen=True)
class ScanOptions:
    event_chunk_size: int = 16
    max_seconds: float = 900.0

    def __post_init__(self):
        if (
            not isinstance(self.event_chunk_size, int)
            or self.event_chunk_size <= 0
            or self.max_seconds <= 0
        ):
            raise ValueError("Positive scan chunk size and wall budget required")


class ScanBudgetExceeded(RuntimeError):
    def __init__(self, grid, completed):
        self.partial_grid, self.completed_candidates = grid, completed
        super().__init__(f"Scan wall budget exhausted after {completed} candidates")


def axis_weights(axis):
    axis = np.asarray(axis, dtype=np.float64)
    if (
        axis.ndim != 1
        or len(axis) < 2
        or not np.isfinite(axis).all()
        or np.any(np.diff(axis) <= 0)
    ):
        raise ValueError(
            "Finite strictly increasing axes with at least two nodes required"
        )
    delta = np.diff(axis)
    return np.r_[delta[0] / 2, (delta[:-1] + delta[1:]) / 2, delta[-1] / 2]


def cell_volumes(axes):
    weights = [axis_weights(v) for v in axes.values()]
    grids = np.meshgrid(*weights, indexing="ij")
    return np.prod(np.stack(grids), axis=0)


def normalize_posterior(log_likelihood, axes, log_prior=None):
    volumes = cell_volumes(axes)
    ll = np.asarray(log_likelihood, dtype=np.float64)
    prior = (
        np.zeros_like(ll)
        if log_prior is None
        else np.asarray(log_prior, dtype=np.float64)
    )
    if (
        ll.shape != volumes.shape
        or prior.shape != ll.shape
        or not np.isfinite(ll).all()
        or np.isnan(prior).any()
        or np.isposinf(prior).any()
    ):
        raise ValueError("Finite likelihood grid and matching valid prior required")
    log_mass = ll + prior + np.log(volumes)
    normalizer = np.logaddexp.reduce(log_mass.ravel())
    if not np.isfinite(normalizer):
        raise ValueError("Prior has no grid support")
    mass = np.exp(log_mass - normalizer)
    density = mass / volumes
    marginals, profiles, slices = {}, {}, {}
    maximum = np.unravel_index(np.argmax(ll), ll.shape)
    for d, (name, axis) in enumerate(axes.items()):
        other = tuple(i for i in range(ll.ndim) if i != d)
        marginals[name] = (
            mass.sum(axis=other) / axis_weights(axis) if other else density.copy()
        )
        profiles[name] = ll.max(axis=other) if other else ll.copy()
        index = list(maximum)
        index[d] = slice(None)
        slices[name] = ll[tuple(index)]
    return {
        "posterior_density": density,
        "posterior_cell_mass": mass,
        "cell_volumes": volumes,
        "log_normalizer": float(normalizer),
        "marginal_densities": marginals,
        "profile_log_likelihood": profiles,
        "conditional_slices_at_mle": slices,
        "mle": {
            name: float(np.asarray(axis)[maximum[d]])
            for d, (name, axis) in enumerate(axes.items())
        },
        "mle_on_boundary": any(
            i in (0, ll.shape[d] - 1) for d, i in enumerate(maximum)
        ),
    }


def per_event_mixture(log_density, integration_weights):
    values = np.asarray(log_density, dtype=np.float64)
    weights = np.asarray(integration_weights, dtype=np.float64)
    if (
        values.ndim != 2
        or weights.shape != (values.shape[1],)
        or np.isnan(values).any()
        or np.isposinf(values).any()
    ):
        raise ValueError(
            "Expected [events, local_nodes] log densities and node weights"
        )
    if (
        not np.isfinite(weights).all()
        or np.any(weights < 0)
        or not np.isclose(weights.sum(), 1.0, atol=1e-10, rtol=1e-10)
    ):
        raise ValueError(
            "Geometry integration weights must be normalized and nonnegative"
        )
    log_weights = np.full_like(weights, -np.inf)
    np.log(weights, out=log_weights, where=weights > 0)
    result = np.logaddexp.reduce(values + log_weights, axis=1)
    if not np.isfinite(result).all():
        raise FloatingPointError(
            "Event has no finite local-variable likelihood support"
        )
    return result


def scan(
    evaluator, observation_source, parameter_spec, *, options=None, log_prior=None
):
    """evaluator(candidate_dict, observation_chunk) -> per-event values.

    Batching is computational only. A scientific prior is applied exactly once,
    after the entire dataset likelihood has been accumulated.
    """
    options = options or ScanOptions()
    axes = {
        name: np.asarray(axis, dtype=np.float64)
        for name, axis in parameter_spec.items()
    }
    if not axes or len(observation_source) == 0:
        raise ValueError("Nonempty parameter specification and observations required")
    shape = cell_volumes(axes).shape
    grid, prior = np.full(shape, np.nan), np.zeros(shape)
    started, completed = time.perf_counter(), 0
    for index in np.ndindex(shape):
        candidate = {
            name: float(axis[index[d]]) for d, (name, axis) in enumerate(axes.items())
        }
        terms = []
        for start in range(0, len(observation_source), options.event_chunk_size):
            if time.perf_counter() - started > options.max_seconds:
                raise ScanBudgetExceeded(grid.copy(), completed)
            chunk = observation_source.subset(
                start, min(start + options.event_chunk_size, len(observation_source))
            )
            values = evaluator(candidate, chunk)
            if isinstance(values, tuple):
                values = values[0]
            if torch.is_tensor(values):
                values = values.detach().cpu().to(torch.float64).numpy()
            values = np.asarray(values, dtype=np.float64)
            if values.shape != (len(chunk),) or not np.isfinite(values).all():
                raise ValueError("Evaluator must return finite per-event values")
            terms.extend(values.tolist())
        grid[index] = math.fsum(terms)
        prior[index] = float(log_prior(candidate)) if log_prior else 0.0
        completed += 1
    posterior = normalize_posterior(grid, axes, prior)
    return {
        "axes": axes,
        "joint_log_likelihood": grid,
        "log_scientific_prior": prior,
        **posterior,
        "diagnostics": {
            "events": len(observation_source),
            "candidates": completed,
            "seconds": time.perf_counter() - started,
            "accumulation": "CPU float64 math.fsum",
            "event_chunk_size": options.event_chunk_size,
        },
    }


def scan_local(
    evaluator,
    observations,
    shared_axes,
    *,
    local_name,
    local_nodes,
    geometry_weights,
    options=None,
    log_prior=None,
):
    nodes = np.asarray(local_nodes, dtype=np.float64)
    axis_weights(nodes)  # Validate the quadrature nodes even when weights are supplied.
    if local_name in shared_axes:
        raise ValueError("Local parameter must not be a shared coordinate")

    def marginalized(candidate, chunk):
        logs = []
        for node in nodes:
            value = evaluator({**candidate, local_name: float(node)}, chunk)
            if isinstance(value, tuple):
                value = value[0]
            if torch.is_tensor(value):
                value = value.detach().cpu().numpy()
            logs.append(value)
        return per_event_mixture(np.stack(logs, axis=1), geometry_weights(candidate))

    result = scan(
        marginalized, observations, shared_axes, options=options, log_prior=log_prior
    )
    result["diagnostics"][
        "fit_mode"
    ] = "shared parameters, independent per-event local-variable integrals"
    return result
