"""Convert aligned inference outputs into the Splat Constructor contract."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from modules.inference.depth import DepthEstimate
from modules.inference.segmentation import SegmentationEstimate
from modules.splat_constructor import ConstructionFrame, DepthSemantics, DepthUnits


@dataclass(frozen=True)
class ConstructionFrameMetadata:
    """Source metadata which inference must preserve rather than synthesize."""

    frame_id: str
    timestamp_ns: int
    source_id: str
    intrinsics: tuple[float, float, float, float]
    camera_to_world: NDArray[np.float32] = field(
        default_factory=lambda: np.eye(4, dtype=np.float32)
    )


def _single_plane(values: Tensor, name: str) -> NDArray[np.float32]:
    values = values.detach().cpu()
    if values.ndim == 4 and values.shape[:2] == (1, 1):
        values = values[0, 0]
    elif values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 2:
        raise ValueError(f"{name} must describe exactly one H x W frame")
    if not torch.is_floating_point(values) or not torch.isfinite(values).all():
        raise ValueError(f"{name} must be finite floating-point data")
    return np.ascontiguousarray(values.numpy(), dtype=np.float32)


def construction_frame_from_estimates(
    rgb: NDArray[np.uint8],
    depth: DepthEstimate,
    segmentation: SegmentationEstimate,
    metadata: ConstructionFrameMetadata,
    *,
    foreground_weight: NDArray[np.float32] | None = None,
    depth_semantics: DepthSemantics = DepthSemantics.RELATIVE_INVERSE,
    depth_units: DepthUnits = DepthUnits.UNITLESS,
) -> ConstructionFrame:
    """Build a validated constructor input from one aligned inference result.

    ``depth.confidence`` is transported as ``depth_validity`` because the
    temporary depth model predicts depth-label availability. Segmentation
    confidence is deliberately not used as a generic quality value; an explicit
    ``foreground_weight`` may be supplied by a future calibrated producer.
    """

    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must be a uint8 H x W x 3 array")
    rgb = np.ascontiguousarray(rgb)
    height, width, _ = rgb.shape
    relative_depth = _single_plane(depth.relative_inverse_depth, "relative_inverse_depth")
    depth_validity = _single_plane(depth.confidence, "depth_validity")
    mask = _single_plane(segmentation.mask.float(), "foreground_mask")
    if relative_depth.shape != (height, width) or depth_validity.shape != (height, width):
        raise ValueError("depth outputs must be aligned to RGB before construction handoff")
    if mask.shape != (height, width):
        raise ValueError("segmentation mask must be aligned to RGB before construction handoff")
    if depth_semantics != DepthSemantics.METRIC_CAMERA_Z and (
        (relative_depth < 0.0).any() or (relative_depth > 1.0).any()
    ):
        raise ValueError("relative inverse depth must be normalized to [0, 1]")
    if foreground_weight is not None:
        foreground_weight = np.asarray(foreground_weight)
        if foreground_weight.dtype != np.float32 or foreground_weight.shape != (height, width):
            raise ValueError("foreground_weight must be a float32 H x W array")
        foreground_weight = np.ascontiguousarray(foreground_weight)

    return ConstructionFrame(
        frame_id=metadata.frame_id,
        timestamp_ns=metadata.timestamp_ns,
        source_id=metadata.source_id,
        rgb=rgb,
        depth=relative_depth,
        foreground_mask=np.ascontiguousarray(mask >= 0.5, dtype=np.uint8),
        intrinsics=metadata.intrinsics,
        depth_semantics=depth_semantics,
        depth_units=depth_units,
        depth_validity=depth_validity,
        foreground_weight=foreground_weight,
        camera_to_world=np.ascontiguousarray(metadata.camera_to_world, dtype=np.float32),
    )
