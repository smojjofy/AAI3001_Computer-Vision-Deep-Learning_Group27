"""Tests for empty-scene background calibration and person segmentation.

Boundaries covered: empty capture, wrong array shape, a size change mid
capture, too little valid depth, an implausible depth spread, a person nearer
than the background, a person farther than the background, pixels with no
baseline, the span limit, and the area guard on both sides.
"""

from __future__ import annotations

import numpy as np
import pytest

from applications.live_pipeline.background_calibration import (
    BackgroundCalibrator,
    CalibrationError,
    person_vs_background_mask,
)


def background_frames(value: float = 4.0, count: int = 5, shape=(24, 32)) -> BackgroundCalibrator:
    calibrator = BackgroundCalibrator()
    for _ in range(count):
        calibrator.add(np.full(shape, value, dtype=np.float32))
    return calibrator


def build(calibrator: BackgroundCalibrator, shape=(24, 32)):
    height, width = shape
    return calibrator.build(source_id="cam", generation=3)


def test_baseline_freezes_the_median_background() -> None:
    baseline = build(background_frames(4.0))
    assert baseline.depth_m.shape == (24, 32)
    assert baseline.median_depth_m == pytest.approx(4.0)
    assert baseline.frames_used == 5
    assert baseline.valid_fraction == pytest.approx(1.0)
    assert baseline.source_id == "cam"
    assert baseline.generation == 3
    assert baseline.matches(32, 24)
    assert not baseline.matches(64, 48)


def test_calibration_keeps_one_depth_source() -> None:
    calibration = BackgroundCalibrator()
    depth = np.full((24, 32), 4.0, dtype=np.float32)
    calibration.add(depth, depth_source="stereo_sgbm")
    with pytest.raises(CalibrationError, match="depth source changed"):
        calibration.add(depth, depth_source="yolo")
    with pytest.raises(CalibrationError, match="does not match"):
        calibration.build(source_id="cam", generation=3, depth_source="yolo")
    baseline = calibration.build(source_id="cam", generation=3, depth_source="stereo_sgbm")
    assert baseline.depth_source == "stereo_sgbm"
    calibration.reset()
    assert calibration.depth_source is None


def test_invalid_pixels_are_ignored_in_the_median() -> None:
    calibrator = BackgroundCalibrator()
    for index in range(4):
        frame = np.full((8, 8), 4.0, dtype=np.float32)
        if index == 0:
            frame[0, 0] = np.nan
        calibrator.add(frame)
    baseline = calibrator.build(source_id="cam", generation=0)
    assert baseline.depth_m[0, 0] == pytest.approx(4.0)


def test_empty_capture_is_rejected() -> None:
    with pytest.raises(CalibrationError, match="no frames"):
        BackgroundCalibrator().build(source_id="cam", generation=0)


def test_non_2d_depth_is_rejected() -> None:
    calibrator = BackgroundCalibrator()
    with pytest.raises(CalibrationError, match="2-D"):
        calibrator.add(np.zeros((4, 4, 3), dtype=np.float32))


def test_size_change_during_capture_is_rejected() -> None:
    calibrator = BackgroundCalibrator()
    calibrator.add(np.full((8, 8), 4.0, dtype=np.float32))
    with pytest.raises(CalibrationError, match="size changed"):
        calibrator.add(np.full((8, 10), 4.0, dtype=np.float32))


def test_capture_with_too_little_valid_depth_is_rejected() -> None:
    calibrator = BackgroundCalibrator()
    frame = np.full((10, 10), np.nan, dtype=np.float32)
    frame[:2, :2] = 4.0
    calibrator.add(frame)
    with pytest.raises(CalibrationError, match="usable depth"):
        calibrator.build(source_id="cam", generation=0)


def test_implausible_depth_spread_is_rejected() -> None:
    calibrator = BackgroundCalibrator(max_spread_m=1.0)
    scale = np.linspace(1.0, 20.0, 100, dtype=np.float32).reshape(10, 10)
    calibrator.add(scale)
    with pytest.raises(CalibrationError, match="spread"):
        calibrator.build(source_id="cam", generation=0)


def test_person_nearer_than_background_is_masked() -> None:
    baseline = build(background_frames(4.0))
    depth = np.full((24, 32), 4.0, dtype=np.float32)
    depth[8:16, 10:20] = 2.5
    result = person_vs_background_mask(depth, baseline)
    mask, validity = result.mask, result.validity
    assert mask.dtype == np.uint8
    assert mask[10, 12] == 1
    assert mask[0, 0] == 0
    assert validity[10, 12] == pytest.approx(1.0)
    assert validity[0, 0] == pytest.approx(0.0)


def test_person_farther_than_background_is_not_masked() -> None:
    # A nearer background (wall in front of the person) must not be learned as a
    # person, and someone behind the background is out of the scene.
    baseline = build(background_frames(2.0))
    depth = np.full((24, 32), 4.0, dtype=np.float32)
    result = person_vs_background_mask(depth, baseline)
    mask, validity = result.mask, result.validity
    assert not bool(mask.any())
    assert not validity.any()


def test_equal_depth_is_not_a_person() -> None:
    baseline = build(background_frames(4.0))
    depth = np.full((24, 32), 4.0, dtype=np.float32)
    result = person_vs_background_mask(depth, baseline)
    mask = result.mask
    assert not mask.any()


def test_pixels_without_a_baseline_are_excluded() -> None:
    calibrator = BackgroundCalibrator()
    frame = np.full((10, 10), 4.0, dtype=np.float32)
    frame[:, :5] = np.nan
    calibrator.add(frame)
    baseline = calibrator.build(source_id="cam", generation=0, valid_ratio=0.5)
    # A person region large enough for the area guard, with usable baseline
    # pixels on the right half only.
    depth = np.full((10, 10), 4.0, dtype=np.float32)
    # 1.0 m in front of the 4.0 m background, inside the 1.5 m span limit.
    depth[2:9, 5:] = 3.0
    result = person_vs_background_mask(depth, baseline)
    mask = result.mask
    # The left half has no baseline, so only the right half can be a person.
    assert not bool(mask[:, :5].any())
    assert bool(mask[:, 5:].any())


def test_implausible_span_limit_drops_pixels() -> None:
    baseline = build(background_frames(4.0))
    depth = np.full((24, 32), 4.0, dtype=np.float32)
    # 3.5 m nearer than the background exceeds the default 1.5 m span limit.
    depth[8:16, 10:20] = 0.5
    result = person_vs_background_mask(depth, baseline)
    mask = result.mask
    assert not mask.any()


def test_area_guard_rejects_tiny_and_huge_regions() -> None:
    baseline = build(background_frames(4.0))
    tiny = np.full((24, 32), 4.0, dtype=np.float32)
    # 0.5 m nearer than the background: inside the span limit, tiny area.
    tiny[0, 0] = 3.5
    tiny_result = person_vs_background_mask(tiny, baseline, min_area_fraction=0.005)
    mask = tiny_result.mask
    assert tiny_result.reason == "too_small"
    assert not mask.any()

    # 1.0 m nearer everywhere: inside the span limit, whole image.
    huge = np.full((24, 32), 3.0, dtype=np.float32)
    huge_result = person_vs_background_mask(huge, baseline, max_area_fraction=0.60)
    mask = huge_result.mask
    assert huge_result.reason == "too_large"
    assert not mask.any()


def test_size_mismatch_against_baseline_is_rejected() -> None:
    baseline = build(background_frames(4.0))
    with pytest.raises(CalibrationError, match="does not match"):
        person_vs_background_mask(np.full((10, 10), 3.0, dtype=np.float32), baseline)


def test_reset_clears_captured_frames() -> None:
    calibrator = background_frames(4.0)
    assert calibrator.count == 5
    calibrator.reset()
    assert calibrator.count == 0
    with pytest.raises(CalibrationError, match="no frames"):
        calibrator.build(source_id="cam", generation=0)


def test_reason_reports_no_foreground_and_guard_failures() -> None:
    baseline = build(background_frames(4.0))
    same = np.full((24, 32), 4.0, dtype=np.float32)
    assert person_vs_background_mask(same, baseline).reason == "no_foreground"
    tiny = np.full((24, 32), 4.0, dtype=np.float32)
    tiny[0, 0] = 3.5
    assert person_vs_background_mask(tiny, baseline).reason == "too_small"
    huge = np.full((24, 32), 3.0, dtype=np.float32)
    assert person_vs_background_mask(huge, baseline).reason == "too_large"
    person = np.full((24, 32), 4.0, dtype=np.float32)
    person[4:20, 6:26] = 3.2
    good = person_vs_background_mask(person, baseline)
    assert good.reason == "ok"
    assert good.detected
