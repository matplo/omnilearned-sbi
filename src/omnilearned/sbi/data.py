"""Measured-event input with no required class labels or simulation truth."""

from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class Observations:
    features: torch.Tensor
    mask: torch.Tensor
    event_ids: tuple
    context: dict | None = None

    def __post_init__(self):
        if (
            self.features.ndim != 3
            or self.mask.shape != self.features.shape[:2]
            or self.mask.dtype != torch.bool
        ):
            raise ValueError("Expected features [N,P,F] and boolean mask [N,P]")
        if len(self.event_ids) != len(self.features) or len(
            set(map(str, self.event_ids))
        ) != len(self.event_ids):
            raise ValueError("Unique event IDs required")
        self.context = self.context or {}
        if any(len(v) != len(self.features) for v in self.context.values()):
            raise ValueError("Known context event count mismatch")

    def __len__(self):
        return len(self.features)

    def subset(self, start, stop):
        return Observations(
            self.features[start:stop],
            self.mask[start:stop],
            self.event_ids[start:stop],
            {k: v[start:stop] for k, v in self.context.items()},
        )

    @classmethod
    def from_npz(cls, path, *, known_context=(), dtype=torch.float32):
        allowed = {
            "features",
            "mask",
            "event_id",
            "four_vectors",
            "jet_kinematics",
            "membership",
        } | set(known_context)
        with np.load(path, allow_pickle=False) as data:
            if set(data) - allowed:
                raise ValueError(
                    f"Undeclared observation fields: {sorted(set(data)-allowed)}"
                )
            required = {"features", "mask", "event_id"} | set(known_context)
            if not required <= set(data):
                raise ValueError("Missing measured input fields")
            if data["mask"].dtype != np.bool_:
                raise ValueError("Mask must be stored as bool")
            return cls(
                torch.as_tensor(data["features"], dtype=dtype),
                torch.as_tensor(data["mask"]),
                tuple(map(str, data["event_id"])),
                {k: torch.as_tensor(data[k], dtype=dtype) for k in known_context},
            )
