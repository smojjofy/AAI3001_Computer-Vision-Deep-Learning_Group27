"""Boundary tests for mask source selection.

Covers a calibrated region, each fallback reason, an uncalibrated run, and every
identity mismatch that must reject an old baseline: image size, source id, and
capture generation.
"""

from __future__ import annotations

import numpy as np

from applications.live_pipeline.adapters import DepthThresholdConfig
from applications.live_pipeline.background_calibration import BackgroundBaseline
from applications.live_pipeline.mask_selection import select_person_mask


def baseline_for(
    background_m: float = 4.0,
    shape: tuple[int, int] = (24, 32),
    source_id: str = "cam",
    generation: int = 2,
) -> BackgroundBaseline:
    height, width = shape
    return BackgroundBaseline(
        depth_m=np.full(shape, background_m, dtype=np.float32),
        width=width,
        height=height,
        source_id=source_id,
        generation=generation,
        frames_used=5,
        valid_fraction=1.0,
        median_depth_m=background_m,
        p05_depth_m=background_m,
        p95_depth_m=background_m,
    )


def depth_with_person(background_m: float = 4.0, person_m: float = 3.2, shape=(24, 32)) -> np.ndarray:
    depth = np.full(shape, background_m, dtype=np.float32)
    depth[4:20, 6:26] = person_m
    return depth


def test_calibrated_baseline_supplies_the_mask() -> None:
    selection = select_person_mask(
        depth_with_person(), baseline=baseline_for(), source_id="cam", generation=2
    )
    assert selection.source == "background_calibrated"
    assert selection.calibrated is True
    assert selection.fallback_reason is None
    assert selection.info.mask[10, 12] == 1
    assert selection.info.mask[0, 0] == 0
    assert selection.info.area_fraction > 0.0
    assert selection.info.confidence > 0.0


def test_missing_baseline_uses_the_depth_threshold() -> None:
    selection = select_person_mask(
        depth_with_person(), baseline=None, source_id="cam", generation=0
    )
    assert selection.source == "depth_threshold"
    assert selection.calibrated is False
    assert selection.fallback_reason is None


def test_no_foreground_falls_back_with_its_reason() -> None:
    selection = select_person_mask(
        np.full((24, 32), 4.0, dtype=np.float32),
        baseline=baseline_for(),
        source_id="cam",
        generation=2,
    )
    assert selection.source == "depth_threshold_fallback"
    assert selection.fallback_reason == "no_foreground"


def test_area_guard_failure_falls_back() -> None:
    small = np.full((24, 32), 4.0, dtype=np.float32)
    small[0, 0] = 3.5
    selection = select_person_mask(small, baseline=baseline_for(), source_id="cam", generation=2)
    assert selection.source == "depth_threshold_fallback"
    assert selection.fallback_reason == "too_small"


def test_size_mismatch_rejects_the_baseline() -> None:
    selection = select_person_mask(
        np.full((10, 10), 3.5, dtype=np.float32),
        baseline=baseline_for(),
        source_id="cam",
        generation=2,
    )
    assert selection.source == "depth_threshold"
    assert selection.fallback_reason == "baseline_identity_mismatch"


def test_source_change_rejects_the_baseline() -> None:
    selection = select_person_mask(
        depth_with_person(), baseline=baseline_for(), source_id="other-camera", generation=2
    )
    assert selection.source == "depth_threshold"
    assert selection.fallback_reason == "baseline_identity_mismatch"


def test_generation_change_rejects_the_baseline() -> None:
    selection = select_person_mask(
        depth_with_person(), baseline=baseline_for(), source_id="cam", generation=3
    )
    assert selection.source == "depth_threshold"
    assert selection.fallback_reason == "baseline_identity_mismatch"


def test_config_area_limits_reach_the_background_comparison() -> None:
    config = DepthThresholdConfig(min_area_fraction=0.0, max_area_fraction=0.05)
    selection = select_person_mask(
        depth_with_person(),
        baseline=baseline_for(),
        source_id="cam",
        generation=2,
        config=config,
    )
    # The person region is larger than 5% of the image, so the calibrated mask is
    # rejected and the same limit is used by the fallback.
    assert selection.source == "depth_threshold_fallback"
    assert selection.fallback_reason == "too_large"
