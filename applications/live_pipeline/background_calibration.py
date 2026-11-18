"""Empty-scene background calibration for the depth-threshold person mask.

The person mask compares each frame's depth with a background baseline captured
while nobody is in view. A pixel is a person pixel when it is nearer than that
baseline by a margin. This replaces the fixed band above the nearest depth, so a
different room and a different person do not need a new constant.

The calibration is manual on purpose. The operator starts it when the scene is
empty. It freezes after a short capture and does not keep learning, so the
person can never be absorbed into the baseline.

This module has no model or camera dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import warnings

import numpy as np


class CalibrationError(RuntimeError):
    """The baseline cannot be captured or used for this frame."""


@dataclass(frozen=True)
class BackgroundBaseline:
    """A frozen background depth map and its measured statistics.

    ``depth_m`` holds the median depth of the capture for each pixel and NaN
    where the capture had no valid depth at least ``valid_ratio`` of the time.
    """

    depth_m: np.ndarray
    width: int
    height: int
    source_id: str
    generation: int
    frames_used: int
    valid_fraction: float
    median_depth_m: float
    p05_depth_m: float
    p95_depth_m: float

    def matches(self, width: int, height: int) -> bool:
        """True when this baseline applies to a frame of this size."""
        return self.width == int(width) and self.height == int(height)

    def stats(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "generation": self.generation,
            "width": self.width,
            "height": self.height,
            "frames_used": self.frames_used,
            "valid_fraction": round(self.valid_fraction, 4),
            "median_depth_m": round(self.median_depth_m, 3),
            "p05_depth_m": round(self.p05_depth_m, 3),
            "p95_depth_m": round(self.p95_depth_m, 3),
        }


@dataclass
class BackgroundCalibrator:
    """Collect empty-scene depth frames, then freeze a baseline.

    The caller decides which frames belong to the capture. This class checks
    that they are usable and that the result is plausible for a background.
    """

    min_valid_fraction: float = 0.5
    max_spread_m: float = 3.0
    frames: list[np.ndarray] = field(default_factory=list)

    def reset(self) -> None:
        self.frames = []

    def add(self, depth_m: np.ndarray) -> None:
        depth = np.asarray(depth_m, dtype=np.float32)
        if depth.ndim != 2:
            raise CalibrationError("depth must be a 2-D array")
        if self.frames and depth.shape != self.frames[0].shape:
            raise CalibrationError("depth size changed during calibration")
        self.frames.append(depth)

    @property
    def count(self) -> int:
        return len(self.frames)

    def build(
        self,
        *,
        source_id: str,
        generation: int,
        valid_ratio: float = 0.5,
    ) -> BackgroundBaseline:
        """Combine the captured frames into one baseline.

        Raises ``CalibrationError`` when the capture is empty, when too little
        of the image had usable depth, or when the depth range looks like a
        failed capture rather than a scene.
        """
        if not self.frames:
            raise CalibrationError("no frames were captured")
        stack = np.stack(self.frames, axis=0)
        valid = np.isfinite(stack) & (stack > 0.0)
        per_pixel_valid = valid.mean(axis=0)
        usable = per_pixel_valid >= float(valid_ratio)
        valid_fraction = float(usable.mean()) if usable.size else 0.0
        if valid_fraction < self.min_valid_fraction:
            raise CalibrationError(
                "only {:.1f}% of the image had usable depth; point the camera at the scene".format(
                    100.0 * valid_fraction
                )
            )
        with warnings.catch_warnings():
            # A column with no usable capture produces an all-NaN slice warning.
            # That result is expected and is excluded by the ``usable`` mask.
            warnings.simplefilter("ignore", RuntimeWarning)
            column_median = np.nanmedian(np.where(valid, stack, np.nan), axis=0)
        median = np.where(usable, column_median, np.nan).astype(np.float32)
        values = median[usable]
        p05 = float(np.percentile(values, 5))
        p95 = float(np.percentile(values, 95))
        if p95 - p05 > self.max_spread_m:
            raise CalibrationError(
                "background depth spread {:.2f} m is too large; check lighting and the camera".format(p95 - p05)
            )
        height, width = median.shape
        return BackgroundBaseline(
            depth_m=median,
            width=int(width),
            height=int(height),
            source_id=str(source_id),
            generation=int(generation),
            frames_used=len(self.frames),
            valid_fraction=valid_fraction,
            median_depth_m=float(np.median(values)),
            p05_depth_m=p05,
            p95_depth_m=p95,
        )


@dataclass
class CalibrationState:
    """The current baseline, or why there is none."""

    baseline: BackgroundBaseline | None = None
    last_error: str | None = None
    calibration_count: int = 0


def nearer_than_background(
    depth_m: np.ndarray,
    baseline: BackgroundBaseline,
    *,
    near_margin_m: float = 0.30,
    max_span_m: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Raw nearer-than-background mask, before any area guard.

    The caller applies a pose region or an area limit. Keeping the raw mask
    separate lets a pose-gated selection recover a candidate whose whole-image
    area would exceed the global guard.
    """
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.shape != baseline.depth_m.shape:
        raise CalibrationError(
            "depth size {} does not match the baseline size {}".format(depth.shape, baseline.depth_m.shape)
        )
    valid = np.isfinite(depth) & (depth > 0.0)
    reference = baseline.depth_m
    usable = valid & np.isfinite(reference)
    difference = np.where(usable, reference - depth, 0.0).astype(np.float32)
    nearer = usable & (difference >= float(near_margin_m)) & (difference <= float(max_span_m))
    return nearer.astype(np.uint8), (valid & nearer).astype(np.float32)


@dataclass(frozen=True)
class BackgroundMask:
    """Result of comparing one depth frame with the background baseline."""

    mask: np.ndarray
    validity: np.ndarray
    area_fraction: float
    reason: str = "ok"

    @property
    def detected(self) -> bool:
        return self.reason == "ok" and bool(self.mask.any())


def person_vs_background_mask(
    depth_m: np.ndarray,
    baseline: BackgroundBaseline,
    *,
    near_margin_m: float = 0.30,
    max_span_m: float = 1.5,
    min_area_fraction: float = 0.005,
    max_area_fraction: float = 0.60,
) -> BackgroundMask:
    """Pixels that are nearer than the background baseline.

    A pixel is a person pixel when its depth is at least ``near_margin_m`` in
    front of the baseline. Where the baseline has no value the pixel cannot be
    classified and is excluded. A span limit drops pixels that differ from the
    baseline by an implausible amount, which is how a broken depth estimate is
    separated from a person.

    Returns a ``BackgroundMask``. ``reason`` is ``ok`` when the region passes
    the area guard, ``too_small`` or ``too_large`` when it fails it, and
    ``no_foreground`` when nothing is nearer than the background.
    """
    mask, validity = nearer_than_background(
        depth_m,
        baseline,
        near_margin_m=near_margin_m,
        max_span_m=max_span_m,
    )
    nearer = mask.astype(bool)
    area_fraction = float(mask.mean()) if mask.size else 0.0
    empty = BackgroundMask(np.zeros_like(mask), np.zeros(mask.shape, dtype=np.float32), area_fraction)
    if not nearer.any():
        return BackgroundMask(empty.mask, empty.validity, area_fraction, "no_foreground")
    if area_fraction < min_area_fraction:
        return BackgroundMask(empty.mask, empty.validity, area_fraction, "too_small")
    if area_fraction > max_area_fraction:
        return BackgroundMask(empty.mask, empty.validity, area_fraction, "too_large")
    return BackgroundMask(mask, validity, area_fraction, "ok")
