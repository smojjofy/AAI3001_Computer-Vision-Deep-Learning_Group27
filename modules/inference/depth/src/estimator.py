"""Stable inference boundary for the depth CNN."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from modules.inference.checkpoint import load_checkpoint

from .model import TinyDepthNet


@dataclass(frozen=True)
class DepthEstimate:
    """Depth output aligned with the input image."""

    relative_inverse_depth: Tensor
    confidence: Tensor


class DepthEstimator:
    """Validate image tensors and execute a depth model in inference mode."""

    def __init__(self, model: nn.Module | None = None, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.model = (model or TinyDepthNet()).to(self.device).eval()

    @classmethod
    def from_checkpoint(
        cls, path: str, device: str = "cpu"
    ) -> "DepthEstimator":
        payload = load_checkpoint(path, expected_task="depth", map_location=device)
        model = TinyDepthNet(**payload["model_config"])
        model.load_state_dict(payload["model_state_dict"])
        return cls(model=model, device=device)

    @torch.inference_mode()
    def predict(self, image: Tensor) -> DepthEstimate:
        """Estimate relative inverse depth from RGB tensors in ``[0, 1]``.

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

        inverse_depth, confidence = self.model(image.to(self.device))
        return DepthEstimate(
            relative_inverse_depth=inverse_depth.cpu(),
            confidence=confidence.cpu(),
        )
