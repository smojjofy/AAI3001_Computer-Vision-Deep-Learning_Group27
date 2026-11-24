"""Check the StereoSGBM matching stage against known disparity.

A synthetic pair with a known pixel shift has a known disparity everywhere, so
the expected depth is exact from ``fx * baseline / disparity``. These tests need
no camera, no SDK, and no model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
DEPTH_SRC = ROOT / "modules" / "inference" / "depth"
for _path in (str(ROOT), str(DEPTH_SRC)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from src.stereo_depth import (  # noqa: E402
    DISPARITY_FIXED_POINT_SCALE,
    STEREO_DEPTH_SOURCE,
    StereoMatcherConfig,
    disparity_to_depth,
    stereo_depth,
    stereo_depth_estimation,
)

#: Texture with fine detail that survives a half-resolution downscale, so the
#: matcher sees the same shift at both resolutions.
_TEXTURE = np.random.default_rng(11).integers(0, 256, (240, 320), dtype=np.uint8)
_TEXTURE = cv2.GaussianBlur(_TEXTURE, (3, 3), 0)


def _shifted_pair(disparity_px: int, *, bottom_rows_extra: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A rectified pair where the right image is the left image shifted left.

    A point at left column ``u`` appears at right column ``u - disparity_px``,
    which is the rectified stereo convention. An optional extra shift on the
    bottom rows creates a second depth plane for a discontinuity test.
    """
    height, width = _TEXTURE.shape
    right = np.zeros_like(_TEXTURE)
    # The scene is shifted so both images have content across the whole width.
    left = _TEXTURE.copy()
    right[:, : width - disparity_px] = left[:, disparity_px:]
    right[:, width - disparity_px :] = left[:, :disparity_px]
    if bottom_rows_extra:
        split = height // 2
        right[split:, : width - bottom_rows_extra] = left[split:, bottom_rows_extra:]
        right[split:, width - bottom_rows_extra :] = left[split:, :bottom_rows_extra]
    return left, right


# ------------------------------------------------------------ the formula


def test_depth_formula_is_exact():
    disparity = np.array([[32.0, 8.0, 16.0]], np.float32)
    depth = disparity_to_depth(disparity, fx=500.0, baseline_m=0.12)
    assert depth[0, 0] == pytest.approx(500.0 * 0.12 / 32.0)
    assert depth[0, 1] == pytest.approx(500.0 * 0.12 / 8.0)
    assert depth[0, 2] == pytest.approx(500.0 * 0.12 / 16.0)


def test_non_positive_disparity_is_nan_never_infinite():
    disparity = np.array([[0.0, -2.0, np.nan, np.inf]], np.float32)
    depth = disparity_to_depth(disparity, fx=500.0, baseline_m=0.12)
    assert np.isnan(depth).all()
    assert not np.isinf(depth).any()


def test_fixed_point_scale_is_sixteen():
    """OpenCV returns 16x disparity; the module must divide by it."""
    assert DISPARITY_FIXED_POINT_SCALE == 16.0


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_bad_geometry_is_rejected(bad):
    disparity = np.ones((2, 2), np.float32) * 8.0
    with pytest.raises(ValueError):
        disparity_to_depth(disparity, fx=bad, baseline_m=0.12)
    with pytest.raises(ValueError):
        disparity_to_depth(disparity, fx=500.0, baseline_m=bad)


# ---------------------------------------------- known disparity on a pair


def test_synthetic_pair_recovers_the_known_depth():
    """A 16-pixel shift with fx 500 and a 120 mm baseline gives 3.75 m."""
    left, right = _shifted_pair(16)
    depth, info = stereo_depth(
        left, right, fx=500.0, baseline_m=0.12, config=StereoMatcherConfig(num_disparities=64)
    )
    assert depth.shape == left.shape
    assert info["valid_fraction"] > 0.3
    trusted = depth[np.isfinite(depth)]
    # The middle of the frame carries the majority of matches, so the median
    # must land on the known plane.
    assert float(np.median(trusted)) == pytest.approx(3.75, rel=0.05)
    assert info["source"] == STEREO_DEPTH_SOURCE


def test_larger_disparity_gives_nearer_depth():
    near_left, near_right = _shifted_pair(32)
    far_left, far_right = _shifted_pair(16)
    config = StereoMatcherConfig(num_disparities=64)
    near, _ = stereo_depth(near_left, near_right, fx=500.0, baseline_m=0.12, config=config)
    far, _ = stereo_depth(far_left, far_right, fx=500.0, baseline_m=0.12, config=config)
    near_median = float(np.nanmedian(near))
    far_median = float(np.nanmedian(far))
    assert near_median < far_median
    assert near_median == pytest.approx(1.875, rel=0.10)


def test_output_is_nan_not_zero_where_unmatched():
    left, right = _shifted_pair(16)
    depth, _ = stereo_depth(left, right, fx=500.0, baseline_m=0.12)
    invalid = ~np.isfinite(depth)
    assert invalid.any(), "the test pair should leave some rows unmatched"
    assert not np.any(np.asarray(depth)[invalid] == 0.0)


def test_depth_estimation_matches_the_shared_contract():
    left, right = _shifted_pair(16)
    estimation, info = stereo_depth_estimation(left, right, fx=500.0, baseline_m=0.12)
    assert estimation.kind == "metric_depth"
    assert estimation.model_name == STEREO_DEPTH_SOURCE
    # The contract requires valid to equal isfinite(depth).
    assert np.array_equal(estimation.valid, np.isfinite(estimation.depth))
    assert (estimation.depth[estimation.valid] > 0).all()
    assert estimation.confidence.shape == estimation.depth.shape


# ------------------------------------------------------------ half resolution


def test_half_resolution_keeps_the_full_image_size():
    left, right = _shifted_pair(16)
    native, native_info = stereo_depth(
        left, right, fx=500.0, baseline_m=0.12, config=StereoMatcherConfig(num_disparities=64)
    )
    half, half_info = stereo_depth(
        left,
        right,
        fx=500.0,
        baseline_m=0.12,
        config=StereoMatcherConfig(num_disparities=64, resolution_divisor=2),
    )
    assert half_info["resolution_divisor"] == 2
    assert half_info["matched_size"] == [160, 120]
    # Output stays aligned with the left image, as the plan requires.
    assert half.shape == left.shape
    # The known plane must survive the downscale and the resize back.
    half_median = float(np.nanmedian(half))
    assert half_median == pytest.approx(3.75, rel=0.15)
    assert half_info["match_ms"] <= native_info["match_ms"] * 1.5


def test_half_resolution_never_invents_intermediate_depth():
    """Nearest-neighbour resize must not create a depth between two planes."""
    left, right = _shifted_pair(16, bottom_rows_extra=8)
    depth, _ = stereo_depth(
        left,
        right,
        fx=500.0,
        baseline_m=0.12,
        config=StereoMatcherConfig(num_disparities=64, resolution_divisor=2),
    )
    values = depth[np.isfinite(depth)]
    if values.size < 100:
        pytest.skip("not enough matches to test the resize")
    # Every valid value must equal one of the two planes the scene contains,
    # within matching error. A blend would sit between them.
    near_plane, far_plane = 3.75, 7.5
    between = (values > near_plane * 1.15) & (values < far_plane * 0.87)
    assert between.mean() < 0.10


# ---------------------------------------------------------------- misuse


def test_mismatched_pair_is_rejected():
    left, right = _shifted_pair(16)
    with pytest.raises(ValueError, match="same shape"):
        stereo_depth(left, right[:-4], fx=500.0, baseline_m=0.12)


def test_float_images_are_rejected():
    left, right = _shifted_pair(16)
    with pytest.raises(ValueError, match="uint8"):
        stereo_depth(left.astype(np.float32), right, fx=500.0, baseline_m=0.12)


def test_colour_pair_is_converted_from_rgb():
    left, right = _shifted_pair(16)
    left_rgb = np.stack([left, left, left], axis=2)
    right_rgb = np.stack([right, right, right], axis=2)
    depth, info = stereo_depth(left_rgb, right_rgb, fx=500.0, baseline_m=0.12)
    assert np.isfinite(depth).any()
    assert info["image_size"] == [left.shape[1], left.shape[0]]


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"num_disparities": 100}, "multiple of 16"),
        ({"num_disparities": 0}, "multiple of 16"),
        ({"block_size": 4}, "odd number"),
        ({"block_size": 3}, "odd number"),
        ({"resolution_divisor": 3}, "resolution_divisor"),
        ({"mode": "bogus"}, "mode must be"),
    ],
)
def test_bad_config_is_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        StereoMatcherConfig(**overrides)


def test_info_records_the_match_conditions():
    left, right = _shifted_pair(16)
    _, info = stereo_depth(left, right, fx=500.0, baseline_m=0.12)
    assert info["fx"] == 500.0
    assert info["baseline_m"] == 0.12
    assert info["match_ms"] >= 0.0
    assert 0.0 <= info["valid_fraction"] <= 1.0
    assert info["config"]["num_disparities"] == 128

def test_half_resolution_agrees_with_native_on_the_same_pair():
    """Regression: resizing disparity up must scale the value by the divisor.

    An earlier version resized the disparity image without multiplying the
    value, so a half-resolution match reported exactly half the true depth.
    """
    for shift in (16, 32):
        left, right = _shifted_pair(shift)
        native, _ = stereo_depth(
            left, right, fx=500.0, baseline_m=0.12,
            config=StereoMatcherConfig(num_disparities=96),
        )
        half, _ = stereo_depth(
            left, right, fx=500.0, baseline_m=0.12,
            config=StereoMatcherConfig(num_disparities=96, resolution_divisor=2),
        )
        native_median = float(np.nanmedian(native))
        half_median = float(np.nanmedian(half))
        expected = 500.0 * 0.12 / shift
        assert native_median == pytest.approx(expected, rel=0.05)
        # Both modes must describe the same plane, not a factor of two apart.
        assert half_median == pytest.approx(native_median, rel=0.08)
