"""Check the stereo fallback decision and the depth-source label.

The plan requires a whole-frame choice between stereo depth and YOLO depth,
with the reason recorded. These tests cover that decision and the label fix
that stopped every external depth plane being published as ``zed``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
DEPTH_SRC = ROOT / "modules" / "inference" / "depth"
for _path in (str(ROOT), str(DEPTH_SRC)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from src.pipeline import (  # noqa: E402
    StereoFallbackConfig,
    apply_stereo_depth_cap,
    stereo_frame_usable,
)


def _plane(fraction_valid: float, *, size: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """A depth plane where the first ``fraction_valid`` of each row carries 2 m.

    Filling row-wise keeps a contiguous block of valid pixels across the top of
    the plane, so a test can mask a region and know whether it is matched.
    """
    valid_columns = int(round(fraction_valid * size))
    flat_depth = np.full((size, size), np.nan, np.float32)
    flat_depth[:, :valid_columns] = 2.0
    return flat_depth, np.isfinite(flat_depth) & (flat_depth > 0.0)


# --------------------------------------------------------------- the decision


def test_a_healthy_frame_keeps_stereo_depth():
    depth, valid = _plane(0.6)
    usable, reason, detail = stereo_frame_usable(depth, valid)
    assert usable is True
    assert reason == "ok"
    assert detail["valid_fraction"] == pytest.approx(0.6)


def test_a_frame_with_no_match_falls_back():
    depth, valid = _plane(0.0)
    usable, reason, _detail = stereo_frame_usable(depth, valid)
    assert usable is False
    assert reason == "no_valid_pixels"


def test_a_sparse_frame_falls_back():
    depth, valid = _plane(0.05)
    usable, reason, detail = stereo_frame_usable(depth, valid)
    assert usable is False
    assert reason == "valid_fraction_below_minimum"
    assert detail["valid_fraction"] == pytest.approx(0.05)


def test_a_person_mask_with_few_matches_falls_back():
    """A healthy frame is not enough when the person itself is mostly unmatched."""
    depth, valid = _plane(0.6)
    mask = np.ones_like(valid)  # the whole frame counts as person
    strict = StereoFallbackConfig(min_person_fraction=0.8)
    _usable, reason, detail = stereo_frame_usable(depth, valid, person_mask=mask, config=strict)
    assert reason == "person_fraction_below_minimum"
    assert detail["person_valid_fraction"] == pytest.approx(0.6)


def test_a_person_mask_with_enough_matches_keeps_stereo():
    depth, valid = _plane(0.8)
    mask = np.zeros_like(valid)
    mask[:, :60] = True  # inside the matched columns, so every masked pixel has depth
    usable, reason, detail = stereo_frame_usable(depth, valid, person_mask=mask)
    assert reason == "ok"
    assert detail["person_valid_fraction"] == pytest.approx(1.0)


def test_an_empty_person_mask_is_reported_separately():
    depth, valid = _plane(0.6)
    mask = np.zeros_like(valid)
    _usable, reason, _detail = stereo_frame_usable(depth, valid, person_mask=mask)
    assert reason == "empty_person_mask"


def test_disagreeing_validity_and_nan_are_treated_as_invalid():
    """A pixel marked valid but holding NaN must not count as usable."""
    depth, valid = _plane(0.8)
    depth[0:40, :] = np.nan  # claim valid, hold no value
    _usable, reason, detail = stereo_frame_usable(depth, valid)
    # 80 valid columns over 100 rows is 8000 claimed pixels. Clearing all
    # columns in the top 40 rows removes 40 * 80 = 3200 of them.
    assert detail["valid_pixels"] == 4800
    assert reason == "ok"


def test_configured_thresholds_are_respected():
    depth, valid = _plane(0.3)
    strict = StereoFallbackConfig(min_valid_fraction=0.5)
    lenient = StereoFallbackConfig(min_valid_fraction=0.2)
    assert stereo_frame_usable(depth, valid, config=strict)[0] is False
    assert stereo_frame_usable(depth, valid, config=lenient)[0] is True


# ------------------------------------------------------------------- depth cap


def test_the_cap_is_off_by_default():
    assert StereoFallbackConfig().max_depth_m is None


def test_the_cap_invalidates_only_far_pixels_and_counts_them():
    depth = np.array([[1.0, 20.0], [50.0, np.nan]], np.float32)
    valid = np.array([[True, True], [True, False]])
    capped, flags, count = apply_stereo_depth_cap(depth, valid, 10.0)
    assert count == 2
    assert flags.tolist() == [[True, False], [False, False]]
    assert capped[0, 0] == pytest.approx(1.0)
    assert np.isnan(capped[0, 1]) and np.isnan(capped[1, 0])


def test_a_capped_frame_can_still_pass_when_enough_depth_remains():
    depth, valid = _plane(0.8)
    depth[:, 50:] = 40.0  # half the frame is implausibly far
    config = StereoFallbackConfig(max_depth_m=10.0, min_valid_fraction=0.2)
    usable, reason, detail = stereo_frame_usable(depth, valid, config=config)
    assert usable is True
    # Only the matched columns 0 to 79 carry depth, so the far half covers 30 of
    # those columns across every row.
    assert detail["pixels_beyond_max_depth"] == 3000


def test_a_frame_entirely_beyond_the_cap_falls_back():
    depth = np.full((10, 10), 50.0, np.float32)
    valid = np.ones((10, 10), bool)
    config = StereoFallbackConfig(max_depth_m=10.0)
    usable, reason, _detail = stereo_frame_usable(depth, valid, config=config)
    assert usable is False
    assert reason == "all_depths_beyond_maximum"


# --------------------------------------------------------------- validation


def test_mismatched_shapes_are_rejected():
    depth, _valid = _plane(0.5)
    with pytest.raises(ValueError, match="validity must have the shape"):
        stereo_frame_usable(depth, np.ones((3, 3), bool))


def test_a_bad_person_mask_shape_is_rejected():
    depth, valid = _plane(0.5)
    with pytest.raises(ValueError, match="person_mask must have the shape"):
        stereo_frame_usable(depth, valid, person_mask=np.ones((3, 3), bool))


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"min_valid_fraction": 1.5}, "min_valid_fraction"),
        ({"min_person_fraction": -0.1}, "min_person_fraction"),
        ({"max_depth_m": 0.0}, "max_depth_m"),
        ({"max_depth_m": float("nan")}, "max_depth_m"),
    ],
)
def test_bad_config_is_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        StereoFallbackConfig(**overrides)


def test_the_cap_value_is_validated():
    depth = np.ones((2, 2), np.float32)
    valid = np.ones((2, 2), bool)
    with pytest.raises(ValueError, match="max_depth_m"):
        apply_stereo_depth_cap(depth, valid, 0.0)
