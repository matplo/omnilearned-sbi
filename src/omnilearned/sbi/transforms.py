"""Invertible bounded-coordinate and affine transforms with physical Jacobians."""

import torch


def _collective_options(specification, dimensions):
    options = specification.get("collective_log_vector_norm")
    if options is None:
        return None
    magnitude, angle = options["log_magnitude_feature"], options["angle_feature"]
    lower, upper = map(float, options["bounds"])
    if (not isinstance(magnitude, int) or not isinstance(angle, int)
            or not 0 <= magnitude < dimensions or not 0 <= angle < dimensions or magnitude == angle
            or not torch.isfinite(torch.tensor([lower, upper])).all() or lower >= upper):
        raise ValueError("Invalid collective log-vector-norm transform")
    return magnitude, angle, (lower+upper)/2, (upper-lower)/2


def _log_vector_norm(x, mask, magnitude, angle):
    logs = x[..., magnitude]
    maximum = torch.where(mask, logs, -torch.inf).max(dim=1).values
    weights = torch.where(mask, torch.exp(logs-maximum[:, None]), 0.)
    real = (weights*torch.cos(x[..., angle])).sum(dim=1)
    imaginary = (weights*torch.sin(x[..., angle])).sum(dim=1)
    result = maximum+.5*torch.log(real.square()+imaginary.square())
    if not torch.isfinite(result).all():
        raise ValueError("Collective vector magnitude must be positive and finite")
    return result


def transform_features(features, specification, mask=None):
    x = features.detach().cpu().to(torch.float64)
    if mask is None:
        mask = torch.ones(x.shape[:2], dtype=torch.bool)
    else:
        mask = mask.detach().cpu()
        if mask.dtype != torch.bool or mask.shape != x.shape[:2]:
            raise ValueError("Transform requires a fixed boolean validity mask")
    x = torch.where(mask[..., None], x, 0.0)
    bounds = specification["bounds"]
    mean, std = x.new_tensor(specification["mean"]), x.new_tensor(specification["std"])
    if (
        len(bounds) != x.shape[-1]
        or mean.shape != (x.shape[-1],)
        or std.shape != mean.shape
        or not (std > 0).all()
    ):
        raise ValueError("Invalid checkpoint coordinate transform")
    if (
        not torch.isfinite(x).all()
        or not torch.isfinite(mean).all()
        or not torch.isfinite(std).all()
    ):
        raise ValueError("Nonfinite coordinate transform")
    u = x.clone()
    log_det = x.new_zeros(x.shape[:2])
    collective_log_det = x.new_zeros(len(x))
    collective = _collective_options(specification, x.shape[-1])
    if collective is not None:
        magnitude, angle, center, half_range = collective
        log_norm = _log_vector_norm(x, mask, magnitude, angle)
        normalized = (log_norm-center)/half_range
        if (normalized.abs() >= 1).any():
            raise ValueError("Collective vector norm outside open checkpoint interval")
        shifted = torch.atanh(normalized)
        u[..., magnitude] += torch.where(mask, (shifted-log_norm)[:, None], 0.)
        # Rank-one determinant: sum_i d log(norm)/d log(magnitude_i) = 1.
        collective_log_det = -torch.log(x.new_tensor(half_range))-torch.log1p(-normalized.square())
    for index, bound in enumerate(bounds):
        if bound is not None:
            if bound <= 0 or (u[..., index].abs() >= bound).any():
                raise ValueError("Coordinate outside open checkpoint fiducial interval")
            coordinate = u[..., index].clone()
            u[..., index] = torch.atanh(coordinate / bound)
            log_det += -torch.log(x.new_tensor(bound)) - torch.log1p(
                -(coordinate / bound).square()
            )
    log_det -= std.log().sum()
    return torch.where(mask[..., None], (u - mean) / std, 0.0), torch.where(
        mask, log_det, 0.0
    ).sum(dim=1)+collective_log_det


def inverse_features(model_coordinates, specification, mask=None):
    z = model_coordinates.detach().cpu().to(torch.float64)
    x = z * z.new_tensor(specification["std"]) + z.new_tensor(specification["mean"])
    for index, bound in enumerate(specification["bounds"]):
        if bound is not None:
            x[..., index] = bound * torch.tanh(x[..., index])
    if mask is None:
        mask = torch.ones(x.shape[:2], dtype=torch.bool)
    else:
        mask = mask.detach().cpu()
        if mask.dtype != torch.bool or mask.shape != x.shape[:2]:
            raise ValueError("Transform requires a fixed boolean validity mask")
    collective = _collective_options(specification, x.shape[-1])
    if collective is not None:
        magnitude, angle, center, half_range = collective
        transformed_norm = _log_vector_norm(x, mask, magnitude, angle)
        log_norm = center+half_range*torch.tanh(transformed_norm)
        x[..., magnitude] += (log_norm-transformed_norm)[:, None]
    return torch.where(mask[..., None], x, 0.)
