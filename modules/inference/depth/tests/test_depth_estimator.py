import pytest
import torch

from modules.inference.depth import DepthEstimator, TinyDepthNet
from modules.inference.depth.src.loss import depth_loss


def test_depth_output_is_aligned_and_normalized() -> None:
    estimator = DepthEstimator(TinyDepthNet(base_channels=8))
    image = torch.rand(2, 3, 33, 47)

    result = estimator.predict(image)

    assert result.relative_inverse_depth.shape == (2, 1, 33, 47)
    assert result.confidence.shape == (2, 1, 33, 47)
    assert torch.isfinite(result.relative_inverse_depth).all()
    assert torch.all((0 <= result.relative_inverse_depth) & (result.relative_inverse_depth <= 1))
    assert torch.all((0 <= result.confidence) & (result.confidence <= 1))


def test_depth_rejects_unnormalized_input() -> None:
    estimator = DepthEstimator(TinyDepthNet(base_channels=8))

    with pytest.raises(ValueError, match="normalized"):
        estimator.predict(torch.full((3, 16, 16), 255.0))


def test_depth_loss_backpropagates() -> None:
    prediction = torch.rand(2, 1, 8, 8, requires_grad=True)
    confidence = torch.rand(2, 1, 8, 8, requires_grad=True)
    target = torch.rand(2, 1, 8, 8)
    validity = torch.ones(2, 1, 8, 8)

    loss, components = depth_loss(prediction, confidence, target, validity)
    loss.backward()

    assert torch.isfinite(loss)
    assert set(components) == {"depth_l1", "gradient", "confidence"}
    assert prediction.grad is not None
    assert confidence.grad is not None
