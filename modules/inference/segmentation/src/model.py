"""Small encoder-decoder CNN for binary foreground segmentation.

This is an architectural baseline only. It requires supervised training before
its masks have semantic meaning.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = _group_count(out_channels)
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs)


class TinySegmentationNet(nn.Module):
    """Compact U-Net-style CNN returning one foreground-logit channel."""

    def __init__(self, in_channels: int = 3, base_channels: int = 24) -> None:
        super().__init__()
        self.encoder_1 = ConvBlock(in_channels, base_channels)
        self.encoder_2 = ConvBlock(base_channels, base_channels * 2)
        self.bottleneck = ConvBlock(base_channels * 2, base_channels * 4)
        self.decoder_2 = ConvBlock(base_channels * 6, base_channels * 2)
        self.decoder_1 = ConvBlock(base_channels * 3, base_channels)
        self.output_head = nn.Conv2d(base_channels, 1, kernel_size=1)

    def encode(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        level_1 = self.encoder_1(inputs)
        level_2 = self.encoder_2(F.max_pool2d(level_1, 2))
        bottleneck = self.bottleneck(F.max_pool2d(level_2, 2))
        return level_1, level_2, bottleneck

    def forward(self, inputs: Tensor) -> Tensor:
        level_1, level_2, bottleneck = self.encode(inputs)

        up_2 = F.interpolate(
            bottleneck, size=level_2.shape[-2:], mode="bilinear", align_corners=False
        )
        decoded_2 = self.decoder_2(torch.cat((up_2, level_2), dim=1))
        up_1 = F.interpolate(
            decoded_2, size=level_1.shape[-2:], mode="bilinear", align_corners=False
        )
        decoded_1 = self.decoder_1(torch.cat((up_1, level_1), dim=1))
        return self.output_head(decoded_1)
