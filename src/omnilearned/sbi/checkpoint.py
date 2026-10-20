"""Density checkpoint loading rejects legacy and runtime-smoke artifacts."""

from dataclasses import asdict
import json
from pathlib import Path
import torch

from .base import GaussianBase
from .conditions import ConditionSpec
from .device import resolve_device
from .model import SmoothVelocity


def save_checkpoint(path, model, *, base, condition_spec, metadata):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    metadata_payload = {
        "schema": "omnilearned-density-v1",
        "architecture": model.architecture,
        "base": base.metadata(),
        "conditions": asdict(condition_spec),
        "mask_policy": "fixed_external",
        "metadata": metadata,
    }
    # Keep metadata plain JSON types so loading never needs numpy pickle globals.
    payload = json.loads(json.dumps(metadata_payload, allow_nan=False))
    payload["state_dict"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(payload, path)


def load_checkpoint(path, *, device):
    device = resolve_device(device)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload.get("schema") != "omnilearned-density-v1"
        or payload.get("mask_policy") != "fixed_external"
    ):
        raise ValueError(
            "Requires a separately trained fixed-mask density checkpoint; legacy/runtime checkpoints are incompatible"
        )
    model = SmoothVelocity(**payload["architecture"]).to(
        device=device, dtype=torch.float32
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    conditions = dict(payload["conditions"])
    for key in ("parameters", "known_context", "center", "scale"):
        conditions[key] = tuple(conditions[key])
    return (
        model,
        GaussianBase.from_metadata(payload["base"]),
        ConditionSpec(**conditions),
        payload["metadata"],
    )
