"""Stable inference boundary for the foreground segmentation CNN."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from modules.inference.checkpoint import load_checkpoint

from .model import TinySegmentationNet


@dataclass(frozen=True)
class SegmentationEstimate:
    """Foreground output aligned with the input image."""

    mask: Tensor
    foreground_probability: Tensor
    confidence: Tensor


class SegmentationEstimator:
    """Validate image tensors and execute a binary segmentation model."""

    def __init__(
        self,
        model: nn.Module | None = None,
        device: str = "cpu",
        threshold: float = 0.5,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        self.device = torch.device(device)
        self.threshold = threshold
        self.model = (model or TinySegmentationNet()).to(self.device).eval()

    @classmethod
    def from_checkpoint(
        cls, path: str, device: str = "cpu", threshold: float = 0.5
    ) -> "SegmentationEstimator":
        payload = load_checkpoint(path, expected_task="segmentation", map_location=device)
        model = TinySegmentationNet(**payload["model_config"])
        model.load_state_dict(payload["model_state_dict"])
        return cls(model=model, device=device, threshold=threshold)

    @torch.inference_mode()
    def predict(self, image: Tensor) -> SegmentationEstimate:
        """Segment RGB tensors in ``[0, 1]``.

        ``image`` may have shape ``(3, H, W)`` or ``(B, 3, H, W)``. Outputs
        always have shape ``(B, 1, H, W)``.
        """
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("image must have shape (3,H,W) or (B,3,H,W)")
        if not image.is_floating_point():
            raise TypeError("image must be a floating-point tensor normalized to [0,1]")
        if not torch.isfinite(image).all():
            raise ValueError("image contains non-finite values")
        if image.numel() and (image.min() < 0 or image.max() > 1):
            raise ValueError("image values must be normalized to [0,1]")

        logits = self.model(image.to(self.device))
        probability = torch.sigmoid(logits)
        mask = probability >= self.threshold
        confidence = (2.0 * (probability - 0.5).abs()).clamp(0.0, 1.0)
        return SegmentationEstimate(
            mask=mask.cpu(),
            foreground_probability=probability.cpu(),
            confidence=confidence.cpu(),
        )
