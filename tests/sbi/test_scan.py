import numpy as np
import pytest
import torch

from omnilearned.sbi import (
    ConditionSpec,
    Observations,
    ScanOptions,
    scan,
    scan_local,
    per_event_mixture,
)
from omnilearned.sbi.scan import axis_weights, normalize_posterior


def data(n=7):
    return Observations(
        torch.zeros(n, 1, 1),
        torch.ones(n, 1, dtype=torch.bool),
        tuple(map(str, range(n))),
    )


def test_joint_correlated_maximum_and_batch_partition():
    def evaluator(c, batch):
        return np.full(
            len(batch),
            -((c["x"] + c["y"] - 2) ** 2 / 0.02 + (c["x"] - c["y"]) ** 2 / 2) / 2,
        )

    axes = {"x": np.linspace(0, 2, 5), "y": np.linspace(0, 2, 5)}
    a = scan(evaluator, data(), axes, options=ScanOptions(event_chunk_size=1))
    b = scan(evaluator, data(), axes, options=ScanOptions(event_chunk_size=4))
    np.testing.assert_array_equal(a["joint_log_likelihood"], b["joint_log_likelihood"])
    assert a["mle"] == {"x": 1.0, "y": 1.0}
    # Coordinate slices through (0,0) incorrectly choose (2,2).
    ll = a["joint_log_likelihood"]
    assert axes["x"][ll[:, 0].argmax()] == axes["y"][ll[0, :].argmax()] == 2.0
    assert not a["mle_on_boundary"]
    assert a["posterior_cell_mass"].sum() == pytest.approx(1.0)


def test_nonuniform_grid_volumes_and_prior_once():
    axes = {"x": np.array([0.0, 0.1, 0.4, 1.0]), "y": np.array([0.0, 0.3, 1.0])}
    result = scan(
        lambda c, b: np.zeros(len(b)),
        data(),
        axes,
        log_prior=lambda c: np.log(1 + c["x"]),
    )
    expected = np.broadcast_to((1 + axes["x"][:, None]) / 1.5, (4, 3))
    np.testing.assert_allclose(result["posterior_density"], expected)
    assert np.sum(
        result["marginal_densities"]["x"] * axis_weights(axes["x"])
    ) == pytest.approx(1.0)


def test_local_integral_precedes_event_log_sum():
    logs = np.array([[0.0, -100.0], [-100.0, 0.0]])
    correct = per_event_mixture(logs, [0.5, 0.5]).sum()
    wrong = np.logaddexp.reduce(logs.sum(axis=0) + np.log(0.5))
    assert correct == pytest.approx(-2 * np.log(2))
    assert wrong == pytest.approx(-100.0)
    with pytest.raises(ValueError, match="normalized"):
        per_event_mixture(logs, [1.0, 1.0])


def test_local_scan_known_shared_solution():
    obs = data(2)
    obs.features[:, 0, 0] = torch.tensor([-1.0, 1.0])

    def evaluator(candidate, batch):
        return (
            -0.5
            * (
                (batch.features[:, 0, 0].numpy() - candidate["phi"] - candidate["b"])
                / 0.2
            )
            ** 2
        )

    result = scan_local(
        evaluator,
        obs,
        {"phi": [-1.0, 0.0, 1.0]},
        local_name="b",
        local_nodes=[-1.0, 1.0],
        geometry_weights=lambda _: np.array([0.5, 0.5]),
    )
    assert result["mle"]["phi"] == 0.0


def test_named_candidates_have_no_truth_dependencies():
    spec = ConditionSpec(
        ("b", "T"), ("known",), center=(1.0, 0.3, 0.0), scale=(2.0, 0.1, 1.0)
    )
    condition = spec.assemble(
        {"b": 3.0, "T": 0.4},
        {"known": [7.0, 8.0]},
        2,
        device="cpu",
        dtype=torch.float64,
    )
    torch.testing.assert_close(
        condition, torch.tensor([[1.0, 1.0, 7.0], [1.0, 1.0, 8.0]], dtype=torch.float64)
    )
    with pytest.raises(ValueError, match="names"):
        spec.assemble(
            {"b": 3.0, "T": 0.4},
            {"known": [7.0, 8.0], "truth": 1.0},
            2,
            device="cpu",
            dtype=torch.float64,
        )


def test_measured_file_without_labels_fits(tmp_path):
    path = tmp_path / "measured.npz"
    np.savez(
        path,
        features=np.ones((3, 2, 1)),
        mask=np.ones((3, 2), dtype=bool),
        event_id=np.array(["a", "b", "c"]),
    )
    observations = Observations.from_npz(path)
    result = scan(
        lambda candidate, batch: -((batch.features[:, 0, 0] - candidate["mean"]) ** 2),
        observations,
        {"mean": [0.0, 1.0, 2.0]},
    )
    assert result["mle"]["mean"] == 1.0
