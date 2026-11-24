"""Tests for refresh-pipeline contract details that do not need model weights.

These tests cover the intrinsics selection rule, the invalid-person signal that
starts the hold-then-fade timer, and the exact joints that reach mask selection.
They do not load YOLO or MediaPipe.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from applications.live_pipeline import pipeline as pipeline_module
from applications.live_pipeline.adapters import DepthThresholdMask, webcam_intrinsics
from applications.live_pipeline.camera_config import CameraConfig
from applications.live_pipeline.pipeline import RefreshError, RefreshPipeline


class FakePacket:
    def __init__(self, width: int, height: int, intrinsics=None) -> None:
        self.rgb = np.zeros((height, width, 3), np.uint8)
        self.intrinsics = intrinsics
        self.depth = None
        self.depth_units = None
        self.frame_id = 1
        self.source_id = "cam"
        self.capture_timestamp_ns = 10

    @property
    def width(self) -> int:
        return int(self.rgb.shape[1])

    @property
    def height(self) -> int:
        return int(self.rgb.shape[0])


def test_intrinsics_follow_resolution_changes() -> None:
    pipeline = RefreshPipeline(device="cpu")
    first = pipeline._ensure_intrinsics(FakePacket(640, 480))
    assert first == webcam_intrinsics(640, 480, pipeline.horizontal_fov_deg)
    resized = pipeline._ensure_intrinsics(FakePacket(1280, 720))
    assert resized == webcam_intrinsics(1280, 720, pipeline.horizontal_fov_deg)
    assert resized != first


def test_intrinsics_follow_capture_calibration() -> None:
    pipeline = RefreshPipeline(device="cpu")
    estimated = pipeline._ensure_intrinsics(FakePacket(640, 480))
    calibrated = pipeline._ensure_intrinsics(FakePacket(640, 480, (700.0, 700.0, 320.0, 240.0)))
    assert calibrated == (700.0, 700.0, 320.0, 240.0)
    assert calibrated != estimated
    # A change back to no calibration re-derives the estimate.
    assert pipeline._ensure_intrinsics(FakePacket(640, 480)) == webcam_intrinsics(
        640, 480, pipeline.horizontal_fov_deg
    )


def test_refresh_error_separates_person_loss_from_other_failures() -> None:
    person = RefreshError("pose not detected", person_missing=True)
    assert person.person_missing is True
    other = RefreshError("native construction failed")
    assert other.person_missing is False


def test_pipeline_generation_defaults_and_sets() -> None:
    pipeline = RefreshPipeline(device="cpu", generation=5)
    assert pipeline.generation == 5
    with pytest.raises(ValueError):
        pipeline._ensure_intrinsics(FakePacket(0, 0))


def test_camera_config_supplies_intrinsics_only_for_its_resolution() -> None:
    config = CameraConfig(width=640, height=480, intrinsics=(700.0, 710.0, 320.0, 240.0))
    pipeline = RefreshPipeline(device="cpu", camera_config=config)
    assert pipeline._ensure_intrinsics(FakePacket(640, 480)) == (700.0, 710.0, 320.0, 240.0)
    # A different resolution must not keep calibration values from another size.
    other = pipeline._ensure_intrinsics(FakePacket(1280, 720))
    assert other == webcam_intrinsics(1280, 720, pipeline.horizontal_fov_deg)
    # Capture intrinsics take precedence over the file.
    assert pipeline._ensure_intrinsics(
        FakePacket(640, 480, (500.0, 500.0, 300.0, 200.0))
    ) == (500.0, 500.0, 300.0, 200.0)


def test_refresh_outcome_keeps_the_exact_mask_input_joints(monkeypatch) -> None:
    # The rounded preview joints are not suitable for mask measurement. The
    # outcome must carry the same unrounded joints that mask selection received,
    # taken from the one pose result; no second inference may run.
    landmarks = [
        SimpleNamespace(x=0.123456789, y=0.987654321, visibility=0.8166),
        SimpleNamespace(x=0.654321987, y=0.123456789, visibility=0.5012),
    ]
    result = SimpleNamespace(
        detected=True,
        depth=SimpleNamespace(depth=np.full((8, 8), 2.0, np.float32)),
        joint_depth=[1.234567, float("nan")],
        pose=SimpleNamespace(landmarks=landmarks),
        depth_source="yolo",
    )

    calls: list[int] = []

    class FakeEstimator:
        def process_frame(self, rgb, frame_id, source_id, timestamp_ns, depth=None):
            calls.append(1)
            return result

        def close(self) -> None:
            pass

    selected: dict = {}

    def fake_selector(depth_m, *, pose_joints, config=None):
        selected["joints"] = pose_joints
        return SimpleNamespace(
            info=DepthThresholdMask(
                mask=np.ones((8, 8), np.uint8),
                validity=np.ones((8, 8), np.float32),
                near_m=2.0,
                far_m=2.0,
                valid_fraction=1.0,
                area_fraction=0.5,
                confidence=0.5,
            ),
            source="pose_gradient",
            fallback_reason=None,
            background_area_fraction=None,
        )

    preview_joints = [{"index": 0, "x": 0.1235, "y": 0.9877, "visibility": 0.82, "depth_m": 1.235}]
    monkeypatch.setattr(pipeline_module, "select_pose_gradient_mask", fake_selector)
    monkeypatch.setattr(
        pipeline_module,
        "_import_depth_stack",
        lambda: (None, None, lambda rgb, value: {"joints": preview_joints}),
    )
    monkeypatch.setattr(pipeline_module, "splat_state_payload", lambda *a, **k: ({"count": 5}, b""))
    import modules.splat_constructor as splat_constructor

    monkeypatch.setattr(
        splat_constructor, "construct_splat_state", lambda frame, config: SimpleNamespace()
    )

    pipeline = RefreshPipeline(device="cpu")
    pipeline._pipeline = FakeEstimator()
    outcome = pipeline.process(FakePacket(8, 8), generation=0)

    assert len(calls) == 1
    assert outcome.mask_pose_joints is selected["joints"]
    assert outcome.mask_pose_joints[0]["x"] == 0.123456789
    assert outcome.mask_pose_joints[1]["depth_m"] is None
    # The display field keeps its rounded preview values.
    assert outcome.pose_joints == preview_joints


def test_stereo_person_gap_retries_yolo_without_a_second_pose(monkeypatch) -> None:
    packet = FakePacket(20, 20, (40.0, 40.0, 10.0, 10.0))
    stereo_depth = np.full((20, 20), np.nan, np.float32)
    stereo_depth[:, :10] = 2.0  # The background passes the whole-image check.
    yolo_depth = np.full((20, 20), 2.0, np.float32)
    pose = SimpleNamespace(landmarks=[SimpleNamespace(x=0.75, y=0.5, visibility=0.9)])

    class FakeEstimator:
        pose_calls = 0
        fallback_calls = 0

        def process_frame(self, rgb, frame_id, source_id, timestamp_ns, *, depth, depth_source):
            self.pose_calls += 1
            return SimpleNamespace(
                detected=True, depth=SimpleNamespace(depth=depth), pose=pose,
                joint_depth=[float("nan")], depth_source=depth_source,
            )

        def replace_depth_with_yolo(self, rgb, previous):
            self.fallback_calls += 1
            assert previous.pose is pose
            return SimpleNamespace(
                detected=True, depth=SimpleNamespace(depth=yolo_depth), pose=pose,
                joint_depth=[2.0], depth_source="yolo",
            )

    def mask_selector(depth_m, *, pose_joints, config):
        mask = np.zeros((20, 20), np.uint8)
        mask[5:15, 12:18] = 1
        return SimpleNamespace(
            info=DepthThresholdMask(mask, np.isfinite(depth_m).astype(np.float32),
                                    2.0, 2.0, 0.5, 0.15, 0.5),
            source="depth_threshold_fallback", fallback_reason=None,
            background_area_fraction=None,
        )

    monkeypatch.setattr(pipeline_module, "select_pose_gradient_mask", mask_selector)
    monkeypatch.setattr(pipeline_module, "splat_state_payload", lambda *a, **k: ({"count": 5}, b""))
    import modules.splat_constructor as splat_constructor
    monkeypatch.setattr(splat_constructor, "construct_splat_state", lambda *a, **k: SimpleNamespace())

    pipeline = RefreshPipeline(device="cpu", stereo_enabled=True)
    estimator = FakeEstimator()
    pipeline._pipeline = estimator
    pipeline._stereo_depth_for = lambda _packet: (
        stereo_depth, "stereo_sgbm", {"status": "used", "match_ms": 1.0}
    )
    outcome = pipeline.process(packet, generation=0)
    assert estimator.pose_calls == 1
    assert estimator.fallback_calls == 1
    assert outcome.depth_source == "yolo"
    assert outcome.stereo_info["fallback_reason"] == "person_fraction_below_minimum"


def test_stereo_background_capture_uses_the_pair() -> None:
    packet = FakePacket(8, 8)
    depth = np.full((8, 8), 3.0, np.float32)

    class FakeEstimator:
        def process_frame(self, rgb, frame_id, source_id, timestamp_ns, *, depth, depth_source):
            assert depth_source == "stereo_sgbm"
            return SimpleNamespace(detected=False, depth=SimpleNamespace(depth=depth),
                                   depth_source=depth_source)

    pipeline = RefreshPipeline(device="cpu", stereo_enabled=True)
    pipeline._pipeline = FakeEstimator()
    pipeline._stereo_depth_for = lambda _packet: (depth, "stereo_sgbm", {"status": "used"})
    captured, person_detected, source = pipeline.capture_background_depth(packet)
    np.testing.assert_array_equal(captured, depth)
    assert person_detected is False
    assert source == "stereo_sgbm"


def test_stereo_mask_ignores_a_baseline_from_another_depth_source(monkeypatch) -> None:
    packet = FakePacket(8, 8)
    depth = np.full((8, 8), 2.0, np.float32)
    result = SimpleNamespace(
        pose=SimpleNamespace(landmarks=[]), joint_depth=[], depth_source="stereo_sgbm"
    )
    selected = SimpleNamespace(info=DepthThresholdMask(
        np.ones((8, 8), np.uint8), np.ones((8, 8), np.float32),
        2.0, 2.0, 1.0, 1.0, 1.0,
    ), source="depth_threshold_fallback")
    calls = []
    monkeypatch.setattr(pipeline_module, "select_pose_gradient_mask", lambda *a, **k: calls.append("pose") or selected)
    monkeypatch.setattr(pipeline_module, "select_person_mask", lambda *a, **k: calls.append("baseline") or selected)
    monkeypatch.setattr(pipeline_module, "depth_threshold_mask", lambda *a, **k: calls.append("depth_band") or selected.info)
    pipeline = RefreshPipeline(device="cpu", stereo_enabled=True)
    pipeline.background_baseline = SimpleNamespace(depth_source="yolo")
    _, mismatch_selection, _ = pipeline._mask_for_result(result, depth, packet, 0)
    assert calls == ["depth_band"]
    assert mismatch_selection.fallback_reason == "baseline_depth_source_mismatch"
    pipeline.background_baseline = SimpleNamespace(depth_source="stereo_sgbm")
    pipeline._mask_for_result(result, depth, packet, 0)
    assert calls == ["depth_band", "baseline"]
