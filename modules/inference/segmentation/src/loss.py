"""Losses for the temporary foreground segmentation baseline."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def segmentation_loss(
    logits: Tensor,
    target: Tensor,
    *,
    dice_weight: float = 0.5,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Combine pixelwise BCE with soft Dice overlap."""
    target = target.float()
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probability = torch.sigmoid(logits)
    dimensions = tuple(range(1, probability.ndim))
    intersection = (probability * target).sum(dim=dimensions)
    denominator = probability.sum(dim=dimensions) + target.sum(dim=dimensions)
    dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    total = bce + dice_weight * dice
    return total, {"bce": bce, "dice": dice}
