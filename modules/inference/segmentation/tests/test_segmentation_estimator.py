import pytest
import torch

from modules.inference.segmentation import SegmentationEstimator, TinySegmentationNet
from modules.inference.segmentation.src.loss import segmentation_loss


def test_segmentation_output_is_aligned_and_normalized() -> None:
    estimator = SegmentationEstimator(TinySegmentationNet(base_channels=8))
    image = torch.rand(2, 3, 33, 47)

    result = estimator.predict(image)

    assert result.mask.shape == (2, 1, 33, 47)
    assert result.mask.dtype == torch.bool
    assert result.foreground_probability.shape == (2, 1, 33, 47)
    assert result.confidence.shape == (2, 1, 33, 47)
    assert torch.all((0 <= result.foreground_probability) & (result.foreground_probability <= 1))
    assert torch.all((0 <= result.confidence) & (result.confidence <= 1))


def test_segmentation_rejects_invalid_threshold() -> None:
    with pytest.raises(ValueError, match="threshold"):
        SegmentationEstimator(TinySegmentationNet(base_channels=8), threshold=1.1)


def test_segmentation_loss_backpropagates() -> None:
    logits = torch.randn(2, 1, 8, 8, requires_grad=True)
    target = torch.randint(0, 2, (2, 1, 8, 8)).float()

    loss, components = segmentation_loss(logits, target)
    loss.backward()

    assert torch.isfinite(loss)
    assert set(components) == {"bce", "dice"}
    assert logits.grad is not None
