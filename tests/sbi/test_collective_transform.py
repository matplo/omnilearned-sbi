import math

import pytest
import torch

from omnilearned.sbi.transforms import inverse_features, transform_features


SPEC = {"bounds": [2.5, 1.2, None], "mean": [.1, -.1, .3], "std": [.5, .6, 1.1],
        "collective_log_vector_norm": {"log_magnitude_feature": 2, "angle_feature": 1,
                                        "bounds": [math.log(3.), math.log(9.)]}}


def cloud():
    return torch.tensor([[[.12, .11, math.log(2.)], [-.23, -.12, math.log(1.5)],
                           [.31, .22, math.log(2.5)]]], dtype=torch.float64)


def test_collective_inverse_permutation_and_mask():
    x = cloud()
    z, log_det = transform_features(x, SPEC)
    torch.testing.assert_close(inverse_features(z, SPEC), x, atol=1e-12, rtol=1e-12)
    permuted, permuted_det = transform_features(x[:, [2, 0, 1]], SPEC)
    torch.testing.assert_close(permuted, z[:, [2, 0, 1]])
    torch.testing.assert_close(permuted_det, log_det)
    padded = torch.cat([x, torch.full((1, 2, 3), float('nan'))], dim=1)
    mask = torch.tensor([[True, True, True, False, False]])
    transformed, padded_det = transform_features(padded, SPEC, mask)
    torch.testing.assert_close(transformed[:, :3], z)
    torch.testing.assert_close(padded_det, log_det)
    recovered = inverse_features(transformed, SPEC, mask)
    torch.testing.assert_close(recovered[:, :3], x)
    assert (recovered[:, 3:] == 0).all()


def test_collective_full_coordinate_jacobian():
    x = cloud()
    z, log_det = transform_features(x, SPEC)
    derivative = torch.empty(x.numel(), x.numel(), dtype=torch.float64)
    for coordinate in range(x.numel()):
        perturbation = torch.zeros_like(x).reshape(-1)
        perturbation[coordinate] = 1e-6
        positive, _ = transform_features(x+perturbation.reshape_as(x), SPEC)
        negative, _ = transform_features(x-perturbation.reshape_as(x), SPEC)
        derivative[:, coordinate] = ((positive-negative)/2e-6).reshape(-1)
    sign, numerical_log_det = torch.linalg.slogdet(derivative)
    assert sign == 1
    torch.testing.assert_close(numerical_log_det, log_det[0], atol=2e-8, rtol=2e-8)


def test_generated_collective_support_and_invalid_observation():
    z = torch.randn((100, 7, 3), generator=torch.Generator().manual_seed(57), dtype=torch.float64)
    x = inverse_features(z, SPEC)
    real = (x[..., 2].exp()*x[..., 1].cos()).sum(dim=1)
    imag = (x[..., 2].exp()*x[..., 1].sin()).sum(dim=1)
    norm = (real.square()+imag.square()).sqrt()
    assert ((norm > 3.) & (norm < 9.)).all()
    bad = cloud()
    bad[..., 2] += math.log(10.)
    with pytest.raises(ValueError, match="outside open"):
        transform_features(bad, SPEC)
