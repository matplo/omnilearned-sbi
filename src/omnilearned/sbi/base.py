"""One serializable base specification for perturbation, sampling and density."""

from dataclasses import asdict, dataclass
import math
import torch


@dataclass(frozen=True)
class GaussianBase:
    mean: tuple
    std: tuple

    def __post_init__(self):
        if not self.mean or len(self.mean) != len(self.std):
            raise ValueError("Base mean/std feature lengths differ")
        if any(not math.isfinite(v) for v in (*self.mean, *self.std)) or any(
            v <= 0 for v in self.std
        ):
            raise ValueError("Finite base parameters and positive scales required")

    def metadata(self):
        return {"family": "diagonal_gaussian", **asdict(self)}

    @classmethod
    def from_metadata(cls, data):
        if data["family"] != "diagonal_gaussian":
            raise ValueError("Unsupported checkpoint base family")
        return cls(tuple(data["mean"]), tuple(data["std"]))

    def _parameters(self, x):
        if x.shape[-1] != len(self.mean):
            raise ValueError("Checkpoint base feature count differs from data")
        return x.new_tensor(self.mean), x.new_tensor(self.std)

    def sample(self, shape, mask, *, device, dtype=torch.float32, generator=None):
        # CPU randomness also works for MPS without a device-specific generator.
        x = torch.randn(shape, generator=generator, dtype=dtype, device="cpu").to(
            device
        )
        mean, std = self._parameters(x)
        return torch.where(mask[..., None], x * std + mean, 0.0)

    def log_prob(self, x, mask):
        # Per-event density and all ensemble sums are evaluated on CPU float64.
        # Transfer first, then cast: a combined MPS->CPU float64 conversion can
        # silently zero values in the installed torch 2.14.1 execution path.
        x, mask = x.detach().cpu().to(torch.float64), mask.detach().cpu()
        mean, std = self._parameters(x)
        terms = (
            -0.5 * ((x - mean) / std).square() - std.log() - 0.5 * math.log(2 * math.pi)
        )
        return torch.where(mask[..., None], terms, 0.0).sum(dim=(1, 2))
