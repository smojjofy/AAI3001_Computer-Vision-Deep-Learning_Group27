"""Dataset adapter for Princeton's small APC RGB-D training sample."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset


@dataclass(frozen=True)
class APCSamplePaths:
    color: Path
    depth: Path
    mask: Path


def decode_apc_depth(encoded: np.ndarray) -> np.ndarray:
    """Undo the dataset's three-bit circular shift and return metres."""
    values = encoded.astype(np.uint32, copy=False)
    decoded = ((values << 3) & 0xFFFF) | (values >> 13)
    return decoded.astype(np.float32) * 0.0001


def _relative_inverse_depth(depth_metres: Tensor, valid: Tensor) -> Tensor:
    """Normalize valid inverse depth robustly into ``[0, 1]`` per image."""
    output = torch.zeros_like(depth_metres)
    if not valid.any():
        return output

    inverse = depth_metres[valid].reciprocal()
    low = torch.quantile(inverse, 0.02)
    high = torch.quantile(inverse, 0.98)
    scale = (high - low).clamp_min(1e-6)
    output[valid] = ((inverse - low) / scale).clamp(0.0, 1.0)
    return output


class APCTrainingSample(Dataset[dict[str, Tensor]]):
    """Load aligned RGB, relative inverse depth, and foreground masks.

    The expected root is the extracted ``training-sample`` directory. The
    source data is metric, but this temporary model intentionally learns the
    repository's unitless relative inverse-depth contract.
    """

    def __init__(
        self,
        root: str | Path,
        image_size: tuple[int, int] = (128, 160),
        augment: bool = False,
        max_samples: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.image_size = image_size
        self.augment = augment
        self.samples = self._discover_samples()
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        if not self.samples:
            raise FileNotFoundError(
                f"no aligned APC samples found under {self.root}; expected *.color.png files"
            )

    def _discover_samples(self) -> list[APCSamplePaths]:
        samples: list[APCSamplePaths] = []
        for color in sorted(self.root.rglob("*.color.png")):
            stem = color.name.removesuffix(".color.png")
            depth = color.with_name(f"{stem}.depth.png")
            mask = color.parent / "masks" / f"{stem}.mask.png"
            if depth.is_file() and mask.is_file():
                samples.append(APCSamplePaths(color=color, depth=depth, mask=mask))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        paths = self.samples[index]
        with Image.open(paths.color) as color_image:
            color_array = np.array(color_image.convert("RGB"), dtype=np.uint8, copy=True)
        with Image.open(paths.depth) as depth_image:
            encoded_depth = np.array(depth_image, dtype=np.uint16, copy=True)
        with Image.open(paths.mask) as mask_image:
            mask_array = np.array(mask_image.convert("L"), dtype=np.uint8, copy=True)

        image = torch.from_numpy(color_array).permute(2, 0, 1).float() / 255.0
        depth = torch.from_numpy(decode_apc_depth(encoded_depth)).unsqueeze(0)
        depth_validity = torch.isfinite(depth) & (depth > 0)
        relative_depth = _relative_inverse_depth(depth, depth_validity)
        foreground = torch.from_numpy(mask_array).unsqueeze(0).float() / 255.0
        foreground = (foreground >= 0.5).float()

        image = F.interpolate(
            image.unsqueeze(0), size=self.image_size, mode="bilinear", align_corners=False
        ).squeeze(0)
        relative_depth = F.interpolate(
            relative_depth.unsqueeze(0),
            size=self.image_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        depth_validity = F.interpolate(
            depth_validity.float().unsqueeze(0), size=self.image_size, mode="nearest"
        ).squeeze(0)
        foreground = F.interpolate(
            foreground.unsqueeze(0), size=self.image_size, mode="nearest"
        ).squeeze(0)

        if self.augment and torch.rand(()) < 0.5:
            image = image.flip(-1)
            relative_depth = relative_depth.flip(-1)
            depth_validity = depth_validity.flip(-1)
            foreground = foreground.flip(-1)

        return {
            "image": image.contiguous(),
            "relative_inverse_depth": relative_depth.contiguous(),
            "depth_validity": depth_validity.contiguous(),
            "foreground_mask": foreground.contiguous(),
        }
