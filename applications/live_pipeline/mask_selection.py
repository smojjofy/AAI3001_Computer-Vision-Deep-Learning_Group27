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
from .background_calibration import BackgroundBaseline, person_vs_background_mask


@dataclass(frozen=True)
class MaskSelection:
    """The mask used for construction and how it was chosen."""

    info: DepthThresholdMask
    source: str
    fallback_reason: str | None = None

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
        )

    reason = None
    if baseline is not None:
        reason = "baseline_identity_mismatch"
    return MaskSelection(depth_threshold_mask(depth, config), "depth_threshold", reason)
