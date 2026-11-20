"""Boundary tests for mask source selection.

Covers a calibrated region, each fallback reason, an uncalibrated run, and every
identity mismatch that must reject an old baseline: image size, source id, and
capture generation.
"""

from __future__ import annotations

import numpy as np
import pytest

from applications.live_pipeline.adapters import DepthThresholdConfig
from applications.live_pipeline.background_calibration import BackgroundBaseline
from applications.live_pipeline import mask_selection as mask_selection_module
from applications.live_pipeline.mask_selection import (
    pose_region_mask,
    select_person_mask,
    select_pose_gradient_mask,
)


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


def pose_joints_box() -> list[dict]:
    # A torso box roughly in the middle of a 24 x 32 image.
    return [
        {"x": 0.35, "y": 0.3, "visibility": 0.9},
        {"x": 0.65, "y": 0.3, "visibility": 0.9},
        {"x": 0.35, "y": 0.75, "visibility": 0.8},
        {"x": 0.65, "y": 0.75, "visibility": 0.8},
    ]


def test_pose_region_requires_a_visible_joint() -> None:
    assert pose_region_mask([], width=32, height=24) is None
    assert pose_region_mask(
        [{"x": 0.5, "y": 0.5, "visibility": 0.1}], width=32, height=24
    ) is None
    assert pose_region_mask(
        [{"x": None, "y": 0.5, "visibility": 1.0}], width=32, height=24
    ) is None
    region = pose_region_mask(pose_joints_box(), width=32, height=24)
    assert region is not None
    assert region[12, 16] == 1
    assert region[0, 0] == 0


def test_pose_gate_removes_background_noise_from_the_candidate() -> None:
    # A noisy frame where a large part of the room is "nearer" than the
    # baseline. Without the pose gate the candidate is too large; with the gate
    # only the person region inside the pose box is kept.
    depth = np.full((24, 32), 4.0, dtype=np.float32)
    depth[2:22, 2:30] = 3.2  # room noise, far too large
    depth[8:18, 12:20] = 2.8  # person inside the pose box
    no_pose = select_person_mask(depth, baseline=baseline_for(), source_id="cam", generation=2)
    assert no_pose.source == "depth_threshold_fallback"
    gated = select_person_mask(
        depth,
        baseline=baseline_for(),
        source_id="cam",
        generation=2,
        pose_joints=pose_joints_box(),
    )
    assert gated.source == "background_calibrated"
    assert gated.info.mask[12, 16] == 1
    assert gated.info.mask[3, 3] == 0
    assert gated.info.area_fraction < 0.6


def pose_gradient_depth() -> np.ndarray:
    # A person block at 1.5 m in front of a 4.0 m wall.
    depth = np.full((48, 64), 4.0, dtype=np.float32)
    depth[14:38, 18:46] = 1.5
    return depth


def pose_gradient_joints() -> list[dict]:
    return [
        {"x": 0.35, "y": 0.38, "visibility": 0.9, "depth_m": 1.5},
        {"x": 0.65, "y": 0.38, "visibility": 0.9, "depth_m": 1.5},
        {"x": 0.35, "y": 0.72, "visibility": 0.8, "depth_m": 1.5},
        {"x": 0.65, "y": 0.72, "visibility": 0.8, "depth_m": 1.5},
    ]


def test_pose_gradient_selection_uses_the_gradient_mask() -> None:
    selection = select_pose_gradient_mask(
        pose_gradient_depth(), pose_joints=pose_gradient_joints()
    )
    assert selection.source == "pose_gradient"
    assert selection.fallback_reason is None
    assert selection.info.mask[26, 32] == 1
    assert selection.info.mask[1, 1] == 0


def test_pose_gradient_selection_falls_back_without_pose() -> None:
    selection = select_pose_gradient_mask(pose_gradient_depth(), pose_joints=None)
    assert selection.source == "depth_threshold_fallback"
    assert selection.fallback_reason == "no_pose"


def test_pose_gradient_selection_passes_the_stride_through(monkeypatch) -> None:
    seen: dict = {}

    def fake_mask(depth_m, joints, **kwargs):
        seen.update(kwargs)
        return np.zeros(np.shape(depth_m), np.uint8), "no_pose"

    monkeypatch.setattr(mask_selection_module, "pose_watershed_mask", fake_mask)
    depth = np.zeros((4, 4), dtype=np.float32)

    select_pose_gradient_mask(depth, pose_joints=None)
    assert seen["gradient_stride"] == 4

    select_pose_gradient_mask(depth, pose_joints=None, gradient_stride=1)
    assert seen["gradient_stride"] == 1


def test_pose_gradient_stride_is_validated() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        select_pose_gradient_mask(
            pose_gradient_depth(), pose_joints=pose_gradient_joints(), gradient_stride=0
        )
