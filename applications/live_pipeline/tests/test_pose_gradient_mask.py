"""Boundary tests for the pose-seeded watershed gradient mask.

Boundaries: no pose, all joints below the visibility floor, joints outside the
image, a flat depth plane with no person, a person block bounded by a depth step,
noisy depth, the area guard on both sides, and a non-2-D input.
"""

from __future__ import annotations

import numpy as np
import pytest

from applications.live_pipeline.pose_gradient_mask import (
    gate_validity_to_joint_depth,
    joint_depth_range,
    pose_seed_mask,
    pose_watershed_mask,
)


def torso_joints(x0=0.35, x1=0.65, y0=0.30, y1=0.75) -> list[dict]:
    return [
        {"x": x0, "y": y0, "visibility": 0.9},
        {"x": x1, "y": y0, "visibility": 0.9},
        {"x": x0, "y": y1, "visibility": 0.9},
        {"x": x1, "y": y1, "visibility": 0.9},
        {"x": (x0 + x1) / 2, "y": (y0 + y1) / 2, "visibility": 0.8},
    ]


def depth_with_person(shape=(48, 64), background=4.0, person=2.6) -> np.ndarray:
    depth = np.full(shape, background, dtype=np.float32)
    h, w = shape
    depth[int(0.28 * h):int(0.78 * h), int(0.32 * w):int(0.68 * w)] = person
    return depth


def test_no_pose_yields_no_region() -> None:
    depth = depth_with_person()
    mask, reason = pose_watershed_mask(depth, None)
    assert reason == "no_pose"
    assert not mask.any()
    mask, reason = pose_watershed_mask(depth, [])
    assert reason == "no_pose"


def test_low_visibility_and_missing_coordinates_are_not_seeds() -> None:
    hidden = [{"x": 0.5, "y": 0.5, "visibility": 0.1}]
    seeds, bounds = pose_seed_mask(hidden, width=64, height=48)
    assert bounds is None
    assert not seeds.any()
    missing = [{"x": None, "y": 0.5, "visibility": 1.0}]
    seeds, bounds = pose_seed_mask(missing, width=64, height=48)
    assert bounds is None
    outside = [{"x": 2.0, "y": 2.0, "visibility": 1.0}]
    seeds, bounds = pose_seed_mask(outside, width=64, height=48)
    assert bounds is not None
    assert not seeds.any()


def test_person_block_is_selected() -> None:
    depth = depth_with_person()
    mask, reason = pose_watershed_mask(depth, torso_joints())
    assert reason == "ok"
    assert mask.dtype == np.uint8
    assert mask[24, 32] == 1
    assert mask[1, 1] == 0
    area = float(mask.mean())
    assert 0.05 < area < 0.60


def test_flat_depth_with_no_person_is_rejected_or_tiny() -> None:
    flat = np.full((48, 64), 4.0, dtype=np.float32)
    mask, reason = pose_watershed_mask(flat, torso_joints())
    assert reason != "ok"
    assert not mask.any()


def test_noisy_depth_does_not_swallow_the_room() -> None:
    rng = np.random.default_rng(7)
    depth = depth_with_person() + rng.normal(0, 0.12, (48, 64)).astype(np.float32)
    mask, reason = pose_watershed_mask(depth, torso_joints())
    if reason == "ok":
        assert float(mask.mean()) <= 0.60
        # The room corners stay out of the person mask.
        assert mask[2, 2] == 0
        assert mask[-3, -3] == 0


def test_area_guard_rejects_a_whole_frame_region() -> None:
    depth = depth_with_person()
    mask, reason = pose_watershed_mask(depth, torso_joints(), max_area_fraction=0.05)
    assert reason == "too_large"
    assert not mask.any()


def test_non_2d_depth_is_rejected() -> None:
    with pytest.raises(ValueError, match="2-D"):
        pose_watershed_mask(np.zeros((4, 4, 3), dtype=np.float32), torso_joints())


def test_joint_depth_range_uses_visible_joint_depths() -> None:
    joints = [
        {"x": 0.4, "y": 0.4, "visibility": 0.9, "depth_m": 1.2},
        {"x": 0.6, "y": 0.4, "visibility": 0.9, "depth_m": 1.6},
        {"x": 0.5, "y": 0.5, "visibility": 0.1, "depth_m": 5.0},
        {"x": 0.5, "y": 0.6, "visibility": 0.9, "depth_m": None},
    ]
    low, high = joint_depth_range(joints, margin_m=0.1)
    # Nearest-rank percentiles of [1.2, 1.6] are 1.24 and 1.56.
    assert low == pytest.approx(1.14)
    assert high == pytest.approx(1.66)
    assert joint_depth_range([]) is None
    assert joint_depth_range([{"x": 0.5, "y": 0.5, "visibility": 0.9, "depth_m": None}]) is None


def test_background_behind_the_hand_is_gated_out() -> None:
    # Person depth ~1.5 m; the wall behind is 3.5 m. The mask must not keep it.
    depth = np.full((48, 64), 3.5, dtype=np.float32)
    depth[16:36, 16:48] = 1.5
    joints = [
        {"x": 0.35, "y": 0.4, "visibility": 0.9, "depth_m": 1.5},
        {"x": 0.6, "y": 0.4, "visibility": 0.9, "depth_m": 1.5},
        {"x": 0.4, "y": 0.65, "visibility": 0.9, "depth_m": 1.5},
    ]
    mask, reason = pose_watershed_mask(depth, joints)
    assert reason == "ok"
    # Far background pixels are excluded by the depth gate.
    assert mask[2, 2] == 0
    assert mask[24, 32] == 1


def test_hands_with_low_visibility_are_still_seeded() -> None:
    depth = np.full((48, 64), 4.0, dtype=np.float32)
    depth[10:40, 10:54] = 1.5
    joints = [
        {"x": 0.2, "y": 0.3, "visibility": 0.2, "depth_m": 1.5},
        {"x": 0.8, "y": 0.3, "visibility": 0.2, "depth_m": 1.5},
        {"x": 0.5, "y": 0.7, "visibility": 0.9, "depth_m": 1.5},
    ]
    seeds, bounds = pose_seed_mask(joints, width=64, height=48)
    assert seeds.any()
    assert bounds is not None


def test_validity_gate_removes_background_depth_inside_the_mask() -> None:
    depth = np.full((24, 32), 1.5, dtype=np.float32)
    depth[0:6, 0:8] = 2.5  # background depth inside the mask footprint
    mask = np.ones((24, 32), dtype=np.uint8)
    joints = [
        {"x": 0.5, "y": 0.5, "visibility": 0.9, "depth_m": 1.5},
        {"x": 0.6, "y": 0.5, "visibility": 0.9, "depth_m": 1.5},
    ]
    validity = gate_validity_to_joint_depth(mask, depth, joints)
    assert validity[12, 16] == 1.0
    assert validity[2, 2] == 0.0


def test_validity_gate_is_a_noop_without_joint_depths() -> None:
    depth = np.full((8, 8), 1.5, dtype=np.float32)
    mask = np.ones((8, 8), dtype=np.uint8)
    validity = gate_validity_to_joint_depth(mask, depth, [{"x": 0.5, "y": 0.5}])
    assert float(validity.mean()) == 1.0
