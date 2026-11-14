"""Losses for the temporary relative-depth baseline."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def depth_loss(
    predicted_depth: Tensor,
    predicted_confidence: Tensor,
    target_depth: Tensor,
    valid_mask: Tensor,
    *,
    gradient_weight: float = 0.25,
    confidence_weight: float = 0.1,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Combine masked L1, edge consistency, and validity confidence losses."""
    valid = valid_mask >= 0.5
    depth_l1 = _masked_mean((predicted_depth - target_depth).abs(), valid)

    pred_dx = predicted_depth[..., :, 1:] - predicted_depth[..., :, :-1]
    true_dx = target_depth[..., :, 1:] - target_depth[..., :, :-1]
    valid_dx = valid[..., :, 1:] & valid[..., :, :-1]
    pred_dy = predicted_depth[..., 1:, :] - predicted_depth[..., :-1, :]
    true_dy = target_depth[..., 1:, :] - target_depth[..., :-1, :]
    valid_dy = valid[..., 1:, :] & valid[..., :-1, :]
    gradient = 0.5 * (
        _masked_mean((pred_dx - true_dx).abs(), valid_dx)
        + _masked_mean((pred_dy - true_dy).abs(), valid_dy)
    )

    confidence = F.binary_cross_entropy(
        predicted_confidence.clamp(1e-6, 1.0 - 1e-6), valid_mask.float()
    )
    total = depth_l1 + gradient_weight * gradient + confidence_weight * confidence
    return total, {"depth_l1": depth_l1, "gradient": gradient, "confidence": confidence}
