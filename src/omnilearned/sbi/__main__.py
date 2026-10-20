"""Lightweight numerical/SBI entry point without distributed training imports."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np

from .checkpoint import load_checkpoint
from .data import Observations
from .likelihood import LikelihoodOptions, log_prob
from .scan import ScanBudgetExceeded, ScanOptions, scan
from .transforms import transform_features


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "fit_config",
        help="JSON with observations, checkpoint, candidate_axes and numerical_options",
    )
    parser.add_argument("--device", required=True, choices=["cpu", "mps", "cuda"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    config = json.loads(Path(args.fit_config).read_text())
    allowed = {
        "schema",
        "observations",
        "observation_sha256",
        "checkpoint",
        "checkpoint_sha256",
        "candidate_axes",
        "scientific_prior",
        "fit_mode",
        "numerical_options",
        "scan_options",
    }
    if set(config) - allowed or config.get("fit_mode") != "fixed_vector":
        raise ValueError("Observation-only fixed-vector fit configuration required")
    import hashlib

    for key in ("observations", "checkpoint"):
        expected = config.get(f"{key[:-1] if key == 'observations' else key}_sha256")
        # observations uses the singular observation_sha256 convention.
        if (
            expected
            and hashlib.sha256(Path(config[key]).read_bytes()).hexdigest() != expected
        ):
            raise ValueError(f"{key} checksum mismatch")
    if config.get("scientific_prior") != "uniform_in_declared_physical_domain":
        raise ValueError(
            "CLI currently supports an explicit uniform scientific prior only; API accepts named nonuniform priors"
        )
    model, base, conditions, metadata = load_checkpoint(
        config["checkpoint"], device=args.device
    )
    import torch

    obs = Observations.from_npz(
        config["observations"],
        known_context=conditions.known_context,
        dtype=torch.float64,
    )
    if metadata.get("requires_selection_normalization"):
        raise ValueError(
            "Checkpoint requires application selection normalization; use its validated application fit driver"
        )
    options = LikelihoodOptions(**config["numerical_options"])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)

    def evaluator(candidate, chunk):
        if metadata.get("observation_space") == "physical_coordinates":
            x, jacobian = transform_features(
                chunk.features, metadata["transform"], mask=chunk.mask
            )
        elif metadata.get("observation_space") == "model_coordinates":
            x, jacobian = chunk.features, None
        else:
            raise ValueError("Checkpoint must declare observation_space")
        return log_prob(
            model,
            x.to(args.device, dtype=torch.float32),
            candidate,
            mask=chunk.mask.to(args.device),
            event_ids=chunk.event_ids,
            base=base,
            condition_spec=conditions,
            context=chunk.context,
            options=options,
            log_abs_det_jacobian=jacobian,
        )

    try:
        result = scan(
            evaluator,
            obs,
            config["candidate_axes"],
            options=ScanOptions(**config.get("scan_options", {})),
        )
    except ScanBudgetExceeded as exc:
        np.savez_compressed(
            output / "partial-grid.npz", joint_log_likelihood=exc.partial_grid
        )
        (output / "status.json").write_text(
            json.dumps(
                {
                    "status": "budget_exhausted",
                    "completed_candidates": exc.completed_candidates,
                }
            )
        )
        raise SystemExit(2)
    np.savez_compressed(
        output / "joint-grid.npz",
        joint_log_likelihood=result["joint_log_likelihood"],
        posterior_density=result["posterior_density"],
        posterior_cell_mass=result["posterior_cell_mass"],
        cell_volumes=result["cell_volumes"],
        **{f"axis_{k}": v for k, v in result["axes"].items()},
    )
    (output / "result.json").write_text(
        json.dumps(
            {
                "mle": result["mle"],
                "mle_on_boundary": result["mle_on_boundary"],
                "diagnostics": result["diagnostics"],
                "numerical_options": asdict(options),
                "checkpoint_metadata": metadata,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
