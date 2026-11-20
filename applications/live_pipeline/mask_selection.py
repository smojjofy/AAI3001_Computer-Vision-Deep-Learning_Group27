"""Choose the person-mask source for one refresh.

When a background baseline exists for the current source, generation, and image
size, the mask compares the depth frame with that baseline. If the comparison
fails (no foreground, or the region fails the area guard), the depth-threshold
band runs as a fallback and the reason is recorded. Without a usable baseline the
depth-threshold band is used directly, exactly as before.

This module has no model or camera dependency, so the selection boundaries are
unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .adapters import DepthThresholdConfig, DepthThresholdMask, depth_threshold_mask
from .pose_gradient_mask import pose_watershed_mask
from .background_calibration import (
    BackgroundBaseline,
    nearer_than_background,
    person_vs_background_mask,
)

MIN_JOINT_VISIBILITY = 0.3


def pose_region_mask(
    joints: list[dict],
    *,
    width: int,
    height: int,
    margin_fraction: float = 0.15,
) -> np.ndarray | None:
    """Dilated bounding region around the visible pose joints.

    The region is a spatial prior for the background comparison, so depth noise
    outside the person cannot become part of the candidate mask. Returns None
    when no joint is usable.
    """
    xs: list[float] = []
    ys: list[float] = []
    for joint in joints:
        x = joint.get("x")
        y = joint.get("y")
        visibility = joint.get("visibility")
        if x is None or y is None:
            continue
        try:
            x = float(x)
            y = float(y)
        except (TypeError, ValueError):
            continue
        if visibility is not None and float(visibility) < MIN_JOINT_VISIBILITY:
            continue
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        xs.append(x)
        ys.append(y)
    if not xs:
        return None
    margin_x = max(2, int(round(margin_fraction * width)))
    margin_y = max(2, int(round(margin_fraction * height)))
    left = max(0, int(min(xs) * width) - margin_x)
    right = min(width, int(max(xs) * width) + margin_x)
    top = max(0, int(min(ys) * height) - margin_y)
    bottom = min(height, int(max(ys) * height) + margin_y)
    if right <= left or bottom <= top:
        return None
    region = np.zeros((height, width), dtype=np.uint8)
    region[top:bottom, left:right] = 1
    return region


@dataclass(frozen=True)
class MaskSelection:
    """The mask used for construction and how it was chosen."""

    info: DepthThresholdMask
    source: str
    fallback_reason: str | None = None
    background_area_fraction: float | None = None

    @property
    def calibrated(self) -> bool:
        return self.source == "background_calibrated"


def _background_threshold_mask(
    depth_m: np.ndarray,
    background_mask,
    validity: np.ndarray,
) -> DepthThresholdMask:
    """Wrap a background comparison result in the existing mask record."""
    mask = np.ascontiguousarray(background_mask, dtype=np.uint8)
    height, width = mask.shape
    area_fraction = float(mask.mean()) if mask.size else 0.0
    inside = depth_m[mask.astype(bool)]
    if inside.size:
        near_m = float(np.percentile(inside, 2.0))
        far_m = float(np.max(inside))
        valid_inside = float(np.isfinite(inside).mean())
    else:
        near_m = float("nan")
        far_m = float("nan")
        valid_inside = 0.0
    confidence = float(np.clip(min(area_fraction / 0.15, 1.0) * valid_inside, 0.0, 1.0))
    return DepthThresholdMask(
        mask=mask,
        validity=np.ascontiguousarray(validity, dtype=np.float32),
        near_m=near_m,
        far_m=far_m,
        valid_fraction=float(np.isfinite(depth_m).mean()) if depth_m.size else 0.0,
        area_fraction=area_fraction,
        confidence=confidence,
    )


def select_person_mask(
    depth_m: np.ndarray,
    *,
    baseline: BackgroundBaseline | None,
    source_id: str,
    generation: int,
    config: DepthThresholdConfig | None = None,
    near_margin_m: float = 0.30,
    max_span_m: float = 1.5,
    pose_joints: list[dict] | None = None,
) -> MaskSelection:
    """Select a person mask for this depth frame.

    A baseline is usable only when its image size, source, and capture run match
    this frame. A size change or a source restart must not apply an old scene's
    background to a new frame.
    """
    config = config or DepthThresholdConfig()
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError("depth_m must be a 2-D array")
    height, width = depth.shape

    if baseline is not None and baseline.matches(width, height):
        if baseline.source_id != str(source_id) or baseline.generation != int(generation):
            return MaskSelection(
                depth_threshold_mask(depth, config),
                "depth_threshold",
                "baseline_identity_mismatch",
            )
        region = None if pose_joints is None else pose_region_mask(
            pose_joints, width=width, height=height
        )
        if region is not None:
            raw_mask, raw_valid = nearer_than_background(
                depth,
                baseline,
                near_margin_m=near_margin_m,
                max_span_m=max_span_m,
            )
            gated = np.logical_and(raw_mask.astype(bool), region.astype(bool))
            gated_area = float(gated.mean()) if gated.size else 0.0
            if config.min_area_fraction <= gated_area <= config.max_area_fraction:
                return MaskSelection(
                    _background_threshold_mask(
                        depth,
                        gated.astype(np.uint8),
                        np.logical_and(raw_valid > 0, gated).astype(np.float32),
                    ),
                    "background_calibrated",
                    None,
                )
            comparison = person_vs_background_mask(
                depth,
                baseline,
                near_margin_m=near_margin_m,
                max_span_m=max_span_m,
                min_area_fraction=config.min_area_fraction,
                max_area_fraction=config.max_area_fraction,
            )
            return MaskSelection(
                depth_threshold_mask(depth, config),
                "depth_threshold_fallback",
                comparison.reason if not comparison.detected else "pose_gate_too_small",
                comparison.area_fraction,
            )
        comparison = person_vs_background_mask(
            depth,
            baseline,
            near_margin_m=near_margin_m,
            max_span_m=max_span_m,
            min_area_fraction=config.min_area_fraction,
            max_area_fraction=config.max_area_fraction,
        )
        if comparison.detected:
            return MaskSelection(
                _background_threshold_mask(depth, comparison.mask, comparison.validity),
                "background_calibrated",
                None,
            )
        # The calibrated comparison failed. The depth-threshold band is the
        # documented fallback, and the reason is kept for diagnostics.
        return MaskSelection(
            depth_threshold_mask(depth, config),
            "depth_threshold_fallback",
            comparison.reason,
            comparison.area_fraction,
        )

    reason = None
    if baseline is not None:
        reason = "baseline_identity_mismatch"
    return MaskSelection(depth_threshold_mask(depth, config), "depth_threshold", reason)


def select_pose_gradient_mask(
    depth_m: np.ndarray,
    *,
    pose_joints: list[dict] | None,
    config: DepthThresholdConfig | None = None,
    pose_margin: float = 0.12,
    gradient_stride: int = 4,
) -> MaskSelection:
    """Use the pose-seeded watershed gradient mask, with the band as fallback.

    The gradient mask needs a depth discontinuity around the pose. When it is not
    available (no pose, flat depth, or area guard failure) the depth-threshold
    band runs and the reason is recorded. ``gradient_stride`` is passed through
    unchanged; the live path keeps the default of 4.
    """
    config = config or DepthThresholdConfig()
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError("depth_m must be a 2-D array")
    mask, reason = pose_watershed_mask(
        depth,
        pose_joints,
        pose_margin=pose_margin,
        min_area_fraction=config.min_area_fraction,
        max_area_fraction=config.max_area_fraction,
        gradient_stride=gradient_stride,
    )
    if reason == "ok":
        return MaskSelection(
            _background_threshold_mask(
                depth,
                mask,
                np.isfinite(depth).astype(np.float32),
            ),
            "pose_gradient",
            None,
        )
    return MaskSelection(
        depth_threshold_mask(depth, config),
        "depth_threshold_fallback",
        reason,
    )
