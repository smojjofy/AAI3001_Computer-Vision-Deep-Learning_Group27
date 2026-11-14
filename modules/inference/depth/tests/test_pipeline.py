"""PoseDepthPipeline with stand-in pose and depth estimators. No MediaPipe, YOLO or ZED needed."""

from types import SimpleNamespace

import numpy as np
import pytest

from src import (CameraIntrinsics, DepthFrameResult, PoseDepthPipeline, external_depth,
                 normalize_prediction, zed_to_rgb)

H, W = 60, 80


def landmark(x, y, visibility=0.99):
    return SimpleNamespace(x=x, y=y, z=0.0, visibility=visibility, presence=0.99)


class FakePose:
    """Same interface as pose_estimator.PoseEstimator: every joint at the same spot."""

    def __init__(self, x=0.25, y=0.5, detected=True):
        self.x, self.y, self.detected, self.calls, self.closed = x, y, detected, 0, False

    def estimate_frame(self, rgb, frame_id, source_id, capture_timestamp_ns):
        self.calls += 1
        lms = tuple(landmark(self.x, self.y) for _ in range(33)) if self.detected else ()
        pose = SimpleNamespace(detected=self.detected, landmarks=lms)
        return SimpleNamespace(frame_id=frame_id, source_id=source_id,
                               capture_timestamp_ns=capture_timestamp_ns, pose=pose)

    def close(self):
        self.closed = True


class FakeYolo:
    """Same interface as YoloDepthEstimator: everything 2 m away."""

    def __init__(self):
        self.calls, self.closed = 0, False

    def estimate_frame(self, rgb, frame_id, source_id, capture_timestamp_ns):
        self.calls += 1
        depth = normalize_prediction(np.full(rgb.shape[:2], 2.0, np.float32), "metric_depth", "fake-yolo")
        return DepthFrameResult(frame_id, source_id, capture_timestamp_ns, depth)

    def close(self):
        self.closed = True


def frame():
    return np.zeros((H, W, 3), np.uint8)


def halves():
    """ZED-like depth: left half 1 m, right half 4 m."""
    d = np.full((H, W), 4.0, np.float32)
    d[:, : W // 2] = 1.0
    return d


@pytest.fixture
def pipe():
    return PoseDepthPipeline(pose_estimator=FakePose(), depth_estimator=FakeYolo())


def test_without_depth_uses_yolo(pipe):
    out = pipe.process_frame(frame(), 7, "zed-left", 123)
    assert out.depth_source == "yolo" and pipe._depth.calls == 1
    assert out.joint_depth.shape == (33,) and np.allclose(out.joint_depth, 2.0)
    assert out.body_depth == pytest.approx(2.0)
    assert (out.frame_id, out.source_id, out.capture_timestamp_ns) == (7, "zed-left", 123)
    assert set(out.timings_ms) == {"pose", "depth", "fuse", "total"}


def test_zed_depth_skips_yolo():
    yolo = FakeYolo()
    pipe = PoseDepthPipeline(pose_estimator=FakePose(x=0.25), depth_estimator=yolo)
    out = pipe.process_frame(frame(), 0, "zed", 0, depth=halves())
    assert out.depth_source == "zed" and yolo.calls == 0
    assert out.depth.model_name == "zed"
    assert np.allclose(out.joint_depth, 1.0)                     # joints are on the 1 m half


def test_yolo_is_only_loaded_when_needed(monkeypatch):
    pipe = PoseDepthPipeline(pose_estimator=FakePose())          # no depth estimator given
    monkeypatch.setattr("src.yolo_depth.YoloDepthEstimator",
                        lambda *a, **k: pytest.fail("YOLO must not load when ZED depth is given"))
    pipe.process_frame(frame(), 0, "zed", 0, depth=halves())


def test_zed_unknown_values_become_invalid():
    d = halves()
    d[0, 0], d[0, 1], d[0, 2], d[0, 3] = np.nan, np.inf, -np.inf, 0.0
    est = external_depth(d, (H, W))
    assert not est.valid[0, :4].any() and np.isnan(est.depth[0, :4]).all()
    assert est.valid[1:].all()


def test_zed_millimetres_and_smaller_depth_map():
    mm = halves()[::2, ::2] * 1000                               # half resolution, millimetres
    est = external_depth(mm, (H, W), depth_scale=0.001)
    assert est.shape == (H, W)
    assert set(np.unique(est.depth)) == {1.0, 4.0}               # nearest-neighbour: no blended depths


def test_mismatched_depth_estimation_is_refused(pipe):
    small = normalize_prediction(np.ones((10, 10), np.float32), "metric_depth")
    with pytest.raises(ValueError, match="frame is"):
        pipe.process_frame(frame(), 0, "zed", 0, depth=small)


def test_joint_xyz_from_intrinsics():
    k = CameraIntrinsics(fx=50.0, fy=50.0, cx=(W - 1) / 2, cy=(H - 1) / 2)
    pipe = PoseDepthPipeline(pose_estimator=FakePose(x=0.5, y=0.5), depth_estimator=FakeYolo(), intrinsics=k)
    xyz = pipe.process_frame(frame(), 0, "zed", 0).joint_xyz
    assert xyz.shape == (33, 3)
    np.testing.assert_allclose(xyz[0], [0.0, 0.0, 2.0], atol=1e-5)   # image centre -> straight ahead


def test_no_person(pipe):
    pipe._pose.detected = False
    out = pipe.process_frame(frame(), 0, "zed", 0)
    assert not out.detected and out.joint_depth.size == 0 and np.isnan(out.body_depth)


def test_zed_bgra_to_rgb():
    bgra = np.zeros((2, 2, 4), np.uint8)
    bgra[..., 0], bgra[..., 2], bgra[..., 3] = 255, 10, 255      # blue, a bit of red, alpha
    rgb = zed_to_rgb(bgra)
    assert rgb.shape == (2, 2, 3) and tuple(rgb[0, 0]) == (10, 0, 255)


def test_bad_frame_and_close(pipe):
    with pytest.raises(ValueError, match="zed_to_rgb"):
        pipe.process_frame(np.zeros((H, W, 4), np.uint8), 0, "zed", 0)
    pose, yolo = pipe._pose, pipe._depth
    pipe.close()
    assert pose.closed and yolo.closed
    with pytest.raises(RuntimeError, match="closed"):
        pipe.process_frame(frame(), 0, "zed", 0)
