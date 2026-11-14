"""Pose + depth fusion with hand-made poses. No MediaPipe or model needed."""

from types import SimpleNamespace

import numpy as np
import pytest

from src import DepthFrameResult, fuse_pose_depth, normalize_prediction
from src.pose_depth import LANDMARK_NAMES, draw_pose_depth, joint_table


def landmark(x, y, visibility=0.99):
    return SimpleNamespace(x=x, y=y, z=0.0, visibility=visibility, presence=0.99)


def pose_frame(landmarks, frame_id=3, source_id="cam", ts=1000, detected=True):
    """Same fields as pose_estimator.PoseFrameResult."""
    pose = SimpleNamespace(detected=detected, landmarks=tuple(landmarks) if detected else ())
    return SimpleNamespace(frame_id=frame_id, source_id=source_id, capture_timestamp_ns=ts, pose=pose)


def depth_frame(metres, frame_id=3, source_id="cam", ts=1000):
    return DepthFrameResult(frame_id, source_id, ts, normalize_prediction(metres, "metric_depth"))


def two_halves(h=100, w=100):
    """Left half 1 m away, right half 4 m away."""
    metres = np.full((h, w), 4.0, np.float32)
    metres[:, : w // 2] = 1.0
    return metres


def test_each_joint_reads_the_depth_under_it():
    lms = [landmark(0.2, 0.5)] * 33
    lms[1] = landmark(0.8, 0.5)
    result = fuse_pose_depth(pose_frame(lms), depth_frame(two_halves()))
    assert result.joint_depth.shape == (33,)
    assert result.joint_depth[0] == pytest.approx(1.0)
    assert result.joint_depth[1] == pytest.approx(4.0)
    assert result.body_depth == pytest.approx(1.0)


def test_low_visibility_offscreen_and_invalid_joints_are_nan():
    metres = two_halves()
    metres[40:60, 10:30] = np.nan                    # a hole in the depth
    lms = [landmark(0.2, 0.2)] * 33
    lms[0] = landmark(0.2, 0.2, visibility=0.1)       # MediaPipe not sure
    lms[1] = landmark(1.5, 0.5)                       # outside the image
    lms[2] = landmark(0.2, 0.5)                       # inside the hole
    jd = fuse_pose_depth(pose_frame(lms), depth_frame(metres)).joint_depth
    assert np.isnan(jd[[0, 1, 2]]).all()
    assert np.isfinite(jd[3:]).all()


def test_no_person_gives_no_joints():
    result = fuse_pose_depth(pose_frame([], detected=False), depth_frame(two_halves()))
    assert not result.detected
    assert result.joint_depth.size == 0
    assert np.isnan(result.body_depth)
    assert joint_table(result) == []


def test_different_frames_are_refused():
    with pytest.raises(ValueError, match="different frames"):
        fuse_pose_depth(pose_frame([landmark(0.5, 0.5)] * 33, frame_id=1), depth_frame(two_halves(), frame_id=2))


def test_table_and_drawing():
    lms = [landmark(0.2 + i * 0.015, 0.5) for i in range(33)]
    result = fuse_pose_depth(pose_frame(lms), depth_frame(two_halves()))
    rows = joint_table(result)
    assert [r[1] for r in rows] == list(LANDMARK_NAMES)
    image = np.zeros((100, 100, 3), np.uint8)
    drawn = draw_pose_depth(image, result)
    assert drawn.shape == image.shape and drawn.any() and not image.any()   # draws on a copy
