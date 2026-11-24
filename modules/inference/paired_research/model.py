"""Small shared encoder with separate person-mask and motion heads."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from modules.inference.segmentation.src.model import ConvBlock


class PairedMaskMotionNet(nn.Module):
    """Return one mask logit and two normalized residual-flow channels."""

    def __init__(
        self, in_channels: int = 15, base_channels: int = 12,
        mask_mode: str = "absolute", gate_radius_px: int = 12,
        residual_scale: float = 4.0,
    ) -> None:
        super().__init__()
        if in_channels != 15 or base_channels < 4:
            raise ValueError("paired model needs 15 input channels and at least 4 base channels")
        if mask_mode not in ("absolute", "bounded_residual"):
            raise ValueError("mask_mode must be absolute or bounded_residual")
        if gate_radius_px < 0 or residual_scale <= 0:
            raise ValueError("gate radius and residual scale must be valid")
        self.mask_mode = mask_mode
        self.gate_radius_px = gate_radius_px
        self.residual_scale = residual_scale
        self.encoder_1 = ConvBlock(in_channels, base_channels)
        self.encoder_2 = ConvBlock(base_channels, base_channels * 2)
        self.bottleneck = ConvBlock(base_channels * 2, base_channels * 4)
        self.decoder_2 = ConvBlock(base_channels * 6, base_channels * 2)
        self.decoder_1 = ConvBlock(base_channels * 3, base_channels)
        self.mask_head = nn.Conv2d(base_channels, 1, kernel_size=1)
        self.motion_head = nn.Conv2d(base_channels, 2, kernel_size=1)
        if mask_mode == "bounded_residual":
            nn.init.zeros_(self.mask_head.weight)
            nn.init.zeros_(self.mask_head.bias)

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        if inputs.ndim != 4 or inputs.shape[1] != 15:
            raise ValueError("inputs must have shape [batch, 15, height, width]")
        level_1 = self.encoder_1(inputs)
        level_2 = self.encoder_2(F.max_pool2d(level_1, 2))
        bottleneck = self.bottleneck(F.max_pool2d(level_2, 2))
        decoded_2 = self.decoder_2(torch.cat((
            F.interpolate(bottleneck, size=level_2.shape[-2:], mode="bilinear", align_corners=False),
            level_2,
        ), dim=1))
        decoded_1 = self.decoder_1(torch.cat((
            F.interpolate(decoded_2, size=level_1.shape[-2:], mode="bilinear", align_corners=False),
            level_1,
        ), dim=1))
        mask_logits = self.mask_head(decoded_1)
        if self.mask_mode == "bounded_residual":
            baseline = (inputs[:, 14:15] >= 0.5).to(mask_logits.dtype)
            radius = self.gate_radius_px
            permitted = F.max_pool2d(baseline, kernel_size=2 * radius + 1, stride=1, padding=radius) > 0
            baseline_logits = (baseline - 0.5) * 4.0
            correction = self.residual_scale * torch.tanh(mask_logits)
            mask_logits = torch.where(permitted, baseline_logits + correction, torch.full_like(mask_logits, -8.0))
        return mask_logits, self.motion_head(decoded_1)
