"""Rectified stereo depth from one left-right pair, using OpenCV StereoSGBM.

This module implements the matching stage of
``docs/stereo-research-record.md``. It is a pure function: it takes two
rectified images and one calibration record, and it returns metric depth in the
shared project convention. It does not open a camera, load a model, or read a
file, so the matching stage is unit-testable without hardware.

Depth convention: metres along the camera optical axis, larger is farther, and
``NaN`` where the match is invalid. Invalid depth is never 0 and never infinite.

The formula is ``depth = fx * baseline_m / disparity_px``. OpenCV StereoSGBM
returns fixed-point disparity, so the raw value is divided by 16 before use. A
non-positive disparity means no match, which this module reports as invalid
rather than as an infinite distance.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np

from .contracts import DepthEstimation

#: OpenCV StereoSGBM stores disparity as a 4-bit fixed-point value.
DISPARITY_FIXED_POINT_SCALE = 16.0

#: Label used for a depth plane that came from this stereo matcher.
STEREO_DEPTH_SOURCE = "stereo_sgbm"


@dataclass(frozen=True)
class StereoMatcherConfig:
    """Parameters for one StereoSGBM matching run.

    ``num_disparities`` must be a positive multiple of 16, which is an OpenCV
    requirement. ``block_size`` must be an odd number greater than or equal to 5.

    ``resolution_divisor`` of 1 matches at the native pair size. A divisor of 2
    halves both images first, which the plan lists as the fallback mode; the
    disparity and the intrinsics then scale with the image, so the returned
    depth stays aligned with the full-size left image.
    """

    num_disparities: int = 128
    block_size: int = 5
    min_disparity: int = 0
    uniqueness_ratio: int = 10
    speckle_window_size: int = 100
    speckle_range: int = 2
    disp12_max_diff: int = 1
    pre_filter_cap: int = 63
    mode: str = "sgbm_3way"
    resolution_divisor: int = 1

    def __post_init__(self) -> None:
        if self.num_disparities <= 0 or self.num_disparities % 16 != 0:
            raise ValueError("num_disparities must be a positive multiple of 16")
        if self.block_size < 5 or self.block_size % 2 == 0:
            raise ValueError("block_size must be an odd number of 5 or more")
        if self.resolution_divisor not in (1, 2, 4):
            raise ValueError("resolution_divisor must be 1, 2, or 4")
        if self.mode not in ("sgbm_3way", "hh4", "sgbm"):
            raise ValueError("mode must be sgbm_3way, hh4, or sgbm")


def _opencv_mode(name: str) -> int:
    return {
        "sgbm_3way": cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        "hh4": cv2.STEREO_SGBM_MODE_HH4,
        "sgbm": cv2.STEREO_SGBM_MODE_SGBM,
    }[name]


def _to_gray(image: np.ndarray, name: str) -> np.ndarray:
    """Return a single-channel uint8 view for matching.

    A colour image is converted from RGB, which is the project's in-process
    order. An already-gray image is used unchanged.
    """
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[2] == 3:
        if array.dtype != np.uint8:
            raise ValueError(f"{name} must be uint8, got {array.dtype}")
        return cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    if array.ndim == 2:
        if array.dtype != np.uint8:
            raise ValueError(f"{name} must be uint8, got {array.dtype}")
        return array
    raise ValueError(f"{name} must be [H, W] or [H, W, 3], got shape {array.shape}")


def disparity_to_depth(
    disparity_px: np.ndarray,
    *,
    fx: float,
    baseline_m: float,
) -> np.ndarray:
    """Convert disparity in pixels to metric depth, with NaN where invalid.

    A disparity that is not finite or not strictly positive gives ``NaN``. The
    returned plane is float32.
    """
    disparity = np.asarray(disparity_px, dtype=np.float32)
    if disparity.ndim != 2:
        raise ValueError(f"disparity must be [H, W], got shape {disparity.shape}")
    if not np.isfinite(fx) or fx <= 0.0:
        raise ValueError("fx must be positive and finite")
    if not np.isfinite(baseline_m) or baseline_m <= 0.0:
        raise ValueError("baseline_m must be positive and finite")
    valid = np.isfinite(disparity) & (disparity > 0.0)
    depth = np.full(disparity.shape, np.nan, dtype=np.float32)
    np.divide(
        np.float32(fx * baseline_m),
        disparity,
        out=depth,
        where=valid,
    )
    return depth


def stereo_depth(
    left_image: np.ndarray,
    right_image: np.ndarray,
    *,
    fx: float,
    baseline_m: float,
    config: StereoMatcherConfig | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Match one rectified pair and return metric depth plus an info record.

    The returned plane is at the left image size, which the plan requires: the
    output must align with the left RGB image that feeds pose, masking, and
    construction. When ``resolution_divisor`` is greater than one, the match
    runs smaller and the depth is resized back with nearest-neighbour, so no
    invented depth appears between foreground and background.
    """
    settings = config or StereoMatcherConfig()
    left = _to_gray(left_image, "left_image")
    right = _to_gray(right_image, "right_image")
    if left.shape != right.shape:
        raise ValueError(
            f"left and right must have the same shape, got {left.shape} and {right.shape}"
        )
    height, width = left.shape
    divisor = settings.resolution_divisor
    # The baseline is a physical distance and does not change with image size.
    # The focal length is a pixel count, so the depth is always computed with
    # the FULL-resolution fx: the matched disparity is converted back to
    # full-resolution pixels below, and that keeps one formula for both modes.
    if divisor == 1:
        match_left, match_right = left, right
    else:
        matched_size = (max(1, width // divisor), max(1, height // divisor))
        match_left = cv2.resize(left, matched_size, interpolation=cv2.INTER_AREA)
        match_right = cv2.resize(right, matched_size, interpolation=cv2.INTER_AREA)

    matcher = cv2.StereoSGBM_create(
        minDisparity=int(settings.min_disparity),
        numDisparities=int(settings.num_disparities),
        blockSize=int(settings.block_size),
        P1=8 * settings.block_size * settings.block_size,
        P2=32 * settings.block_size * settings.block_size,
        disp12MaxDiff=int(settings.disp12_max_diff),
        uniquenessRatio=int(settings.uniqueness_ratio),
        speckleWindowSize=int(settings.speckle_window_size),
        speckleRange=int(settings.speckle_range),
        preFilterCap=int(settings.pre_filter_cap),
        mode=_opencv_mode(settings.mode),
    )
    started = time.perf_counter()
    raw = matcher.compute(match_left, match_right)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    disparity_px = np.asarray(raw, dtype=np.float32) / np.float32(DISPARITY_FIXED_POINT_SCALE)
    valid_matched = np.isfinite(disparity_px) & (disparity_px > 0.0)

    if divisor == 1:
        disparity_full = disparity_px
        valid_full = valid_matched
    else:
        # A disparity is a pixel count measured at the matched size. Resizing
        # the image up multiplies pixel distance by the divisor, so the
        # disparity value must be multiplied too. Nearest-neighbour copies the
        # value without changing it, which would halve the implied depth and is
        # why the value is scaled here before it is stretched.
        disparity_full = cv2.resize(
            disparity_px * np.float32(divisor),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
        valid_full = cv2.resize(
            valid_matched.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        disparity_full = np.where(valid_full, disparity_full, np.float32(0.0))

    depth = disparity_to_depth(
        np.where(valid_full, disparity_full, np.float32(0.0)),
        fx=float(fx),
        baseline_m=float(baseline_m),
    )
    valid = np.isfinite(depth) & (depth > 0.0)

    info: dict[str, Any] = {
        "source": STEREO_DEPTH_SOURCE,
        "fx": float(fx),
        "baseline_m": float(baseline_m),
        "image_size": [int(width), int(height)],
        "matched_size": [int(match_left.shape[1]), int(match_left.shape[0])],
        "resolution_divisor": int(divisor),
        "match_ms": round(float(elapsed_ms), 3),
        "valid_fraction": round(float(valid.mean()) if valid.size else 0.0, 4),
        "disparity_fixed_point_scale": DISPARITY_FIXED_POINT_SCALE,
        "config": asdict(settings),
    }
    if valid.any():
        values = depth[valid]
        info["depth_m"] = {
            "p05": round(float(np.percentile(values, 5.0)), 4),
            "p50": round(float(np.percentile(values, 50.0)), 4),
            "p95": round(float(np.percentile(values, 95.0)), 4),
            "max": round(float(values.max()), 4),
        }
    return depth, info


def stereo_depth_estimation(
    left_image: np.ndarray,
    right_image: np.ndarray,
    *,
    fx: float,
    baseline_m: float,
    config: StereoMatcherConfig | None = None,
) -> tuple[DepthEstimation, dict[str, Any]]:
    """Return :func:`stereo_depth` output as the shared depth contract.

    The estimation is metric, its validity plane equals ``isfinite(depth)`` as
    the contract requires, and the raw disparity is kept for comparison.
    """
    depth, info = stereo_depth(
        left_image, right_image, fx=fx, baseline_m=baseline_m, config=config
    )
    valid = np.isfinite(depth) & (depth > 0.0)
    estimation = DepthEstimation(
        depth=np.where(valid, depth, np.nan).astype(np.float32),
        kind="metric_depth",
        model_name=STEREO_DEPTH_SOURCE,
        valid=valid,
        confidence=valid.astype(np.float32),
    )
    return estimation, info
