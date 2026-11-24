"""Apply one spatial transform to every member of a checked frame pair."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def augment_pairs(
    inputs: Tensor, target: Tensor, *, generator: torch.Generator,
    scale_range: tuple[float, float] = (0.75, 1.15),
    shift_fraction: tuple[float, float] = (0.15, 0.22),
    flip_probability: float = 0.5,
    colour_gain_range: tuple[float, float] = (0.75, 1.25),
) -> tuple[Tensor, Tensor]:
    """Transform RGB, depth, flow, pose, masks, and the checked target together.

    Inputs use [batch, 15, height, width]. Flow is normalized by image axes.
    The same colour change applies to the reference and selected RGB frames.
    """
    if inputs.ndim != 4 or inputs.shape[1] != 15 or target.shape != inputs.shape[:1] + (1,) + inputs.shape[2:]:
        raise ValueError("paired tensors have incompatible shapes")
    if not (0 < scale_range[0] <= scale_range[1] and 0 <= flip_probability <= 1):
        raise ValueError("invalid augmentation range")
    batch = inputs.shape[0]
    inputs = inputs.clone()
    target = target.clone()
    flip = torch.rand(batch, generator=generator) < flip_probability
    if flip.any():
        flipped = torch.flip(inputs, dims=(-1,))
        flipped[:, 10] *= -1.0
        inputs[flip] = flipped[flip]
        target[flip] = torch.flip(target[flip], dims=(-1,))

    scale = scale_range[0] + torch.rand(batch, generator=generator) * (scale_range[1] - scale_range[0])
    shift_x = (torch.rand(batch, generator=generator) * 2 - 1) * 2 * shift_fraction[0]
    shift_y = (torch.rand(batch, generator=generator) * 2 - 1) * 2 * shift_fraction[1]
    theta = torch.zeros((batch, 2, 3), dtype=inputs.dtype)
    theta[:, 0, 0] = 1.0 / scale
    theta[:, 1, 1] = 1.0 / scale
    theta[:, 0, 2] = -shift_x / scale
    theta[:, 1, 2] = -shift_y / scale
    grid = F.affine_grid(theta, inputs.shape, align_corners=False)
    rgb = F.grid_sample(inputs[:, :6], grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    signals = F.grid_sample(inputs[:, 6:14], grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    continuous = torch.cat((rgb, signals), dim=1)
    baseline = F.grid_sample(inputs[:, 14:15], grid, mode="nearest", padding_mode="zeros", align_corners=False)
    target = F.grid_sample(target, grid, mode="nearest", padding_mode="zeros", align_corners=False)
    continuous[:, 10:12] *= scale[:, None, None, None]
    continuous[:, 8:10] = (continuous[:, 8:10] >= 0.5).to(continuous.dtype)
    continuous[:, 6:8] *= continuous[:, 8:10]
    inputs = torch.cat((continuous, baseline), dim=1)

    gain = colour_gain_range[0] + torch.rand(batch, generator=generator) * (colour_gain_range[1] - colour_gain_range[0])
    contrast = 0.8 + torch.rand(batch, generator=generator) * 0.4
    channel_gain = 0.9 + torch.rand((batch, 3), generator=generator) * 0.2
    for start in (0, 3):
        colour = inputs[:, start:start + 3]
        colour = ((colour - 0.5) * contrast[:, None, None, None] + 0.5)
        colour = colour * gain[:, None, None, None] * channel_gain[:, :, None, None]
        inputs[:, start:start + 3] = colour.clamp(0, 1)
    return inputs, target
