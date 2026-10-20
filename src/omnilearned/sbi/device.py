"""Explicit single-process device selection, with no implicit fallback."""

import os
import torch


def resolve_device(requested):
    device = torch.device(requested)
    if device.type == "mps":
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
            raise RuntimeError(
                "Disable PYTORCH_ENABLE_MPS_FALLBACK for validated MPS execution"
            )
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS is unavailable in this execution context; select CPU explicitly for CPU work"
            )
    elif device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device is unavailable")
    elif device.type != "cpu":
        raise ValueError(f"Unsupported device: {device}")
    return device
