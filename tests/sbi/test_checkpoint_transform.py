import numpy as np
import pytest
import torch

from omnilearned.sbi import SmoothVelocity, ConditionSpec, GaussianBase
from omnilearned.sbi.checkpoint import load_checkpoint, save_checkpoint
from omnilearned.sbi.transforms import inverse_features, transform_features


def test_transform_physical_measure_and_checkpoint_roundtrip(tmp_path):
    transform = {"bounds": [2.0, None], "mean": [0.1, -0.2], "std": [0.5, 1.3]}
    x = torch.tensor([[[0.4, 1.1], [-0.3, -0.4]]], dtype=torch.float64)
    z, jacobian = transform_features(x, transform)
    torch.testing.assert_close(inverse_features(z, transform), x)
    expected = sum(
        -np.log(2) - np.log1p(-((v / 2) ** 2)) - np.log(0.5 * 1.3) for v in [0.4, -0.3]
    )
    assert jacobian.item() == pytest.approx(expected)
    model = SmoothVelocity(features=2, conditions=1, width=8, depth=1, heads=2)
    path = tmp_path / "density.pt"
    save_checkpoint(
        path,
        model,
        base=GaussianBase((0.0, 0.0), (1.0, 1.0)),
        condition_spec=ConditionSpec(("theta",)),
        metadata={"observation_space": "physical_coordinates", "transform": transform},
    )
    loaded, base, spec, metadata = load_checkpoint(path, device="cpu")
    for a, b in zip(model.parameters(), loaded.parameters()):
        torch.testing.assert_close(a, b)
    assert spec.parameters == ("theta",)
    assert metadata["transform"] == transform
    with pytest.raises(FileExistsError):
        save_checkpoint(path, model, base=base, condition_spec=spec, metadata=metadata)


def test_runtime_checkpoint_is_not_a_density_checkpoint(tmp_path):
    path = tmp_path / "runtime.pt"
    torch.save({"kind": "runtime-smoke-only-not-a-density-checkpoint"}, path)
    with pytest.raises(ValueError, match="incompatible"):
        load_checkpoint(path, device="cpu")


def test_padding_is_excluded_from_transform_jacobian():
    transform = {"bounds": [2.0, None], "mean": [0.1, -0.2], "std": [0.5, 1.3]}
    x = torch.tensor([[[0.4, 1.1], [float("nan"), float("nan")]]])
    z, jac = transform_features(x, transform, mask=torch.tensor([[True, False]]))
    one, one_jac = transform_features(x[:, :1], transform)
    torch.testing.assert_close(z[:, :1], one)
    torch.testing.assert_close(jac, one_jac)
    torch.testing.assert_close(z[:, 1], torch.zeros_like(z[:, 1]))


def test_lightweight_cli_measured_path(tmp_path):
    import json
    from omnilearned.sbi.__main__ import main

    model = SmoothVelocity(features=2, conditions=1, width=8, depth=1, heads=2)
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(
        checkpoint,
        model,
        base=GaussianBase((0.0, 0.0), (1.0, 1.0)),
        condition_spec=ConditionSpec(("theta",)),
        metadata={"observation_space": "model_coordinates"},
    )
    observed = tmp_path / "measured.npz"
    np.savez(
        observed,
        features=np.ones((2, 2, 2)),
        mask=np.ones((2, 2), dtype=bool),
        event_id=np.array(["a", "b"]),
    )
    fit = tmp_path / "fit.json"
    fit.write_text(
        json.dumps(
            {
                "observations": str(observed),
                "checkpoint": str(checkpoint),
                "fit_mode": "fixed_vector",
                "scientific_prior": "uniform_in_declared_physical_domain",
                "candidate_axes": {"theta": [0.0, 1.0]},
                "numerical_options": {"steps": 1, "divergence": "exact"},
            }
        )
    )
    output = tmp_path / "result"
    main([str(fit), "--device", "cpu", "--output", str(output)])
    with np.load(output / "joint-grid.npz") as grid:
        assert grid["posterior_cell_mass"].sum() == pytest.approx(1.0)
