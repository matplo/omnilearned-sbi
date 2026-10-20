"""Candidate parameters replace unknowns; observations supply known context only."""

from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class ConditionSpec:
    parameters: tuple
    known_context: tuple = ()
    center: tuple = ()
    scale: tuple = ()

    def __post_init__(self):
        names = self.parameters + self.known_context
        if not self.parameters or len(set(names)) != len(names):
            raise ValueError("Condition names must be unique and parameters nonempty")
        n = len(names)
        if self.center and len(self.center) != n or self.scale and len(self.scale) != n:
            raise ValueError("Condition normalization length mismatch")
        if any(not math.isfinite(v) for v in (*self.center, *self.scale)) or any(
            v <= 0 for v in self.scale
        ):
            raise ValueError("Invalid condition normalization")

    def assemble(self, candidate, context, count, *, device, dtype):
        context = context or {}
        if set(candidate) != set(self.parameters) or set(context) != set(
            self.known_context
        ):
            raise ValueError(
                "Candidate and known-context names must match the checkpoint specification"
            )
        columns = []
        for name in self.parameters + self.known_context:
            value = candidate[name] if name in candidate else context[name]
            column = torch.as_tensor(value, device=device, dtype=dtype)
            if column.ndim == 0:
                column = column.expand(count)
            if column.shape != (count,) or not torch.isfinite(column).all():
                raise ValueError(f"Invalid condition column {name}")
            columns.append(column)
        result = torch.stack(columns, dim=-1)
        if self.center:
            result = result - result.new_tensor(self.center)
        if self.scale:
            result = result / result.new_tensor(self.scale)
        return result
