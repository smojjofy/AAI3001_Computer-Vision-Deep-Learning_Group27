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
