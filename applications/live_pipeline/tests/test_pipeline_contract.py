"""Tests for refresh-pipeline contract details that do not need model weights.

These tests cover the intrinsics selection rule and the invalid-person signal
that starts the hold-then-fade timer. They do not load YOLO or MediaPipe.
"""

from __future__ import annotations

import numpy as np
import pytest

from applications.live_pipeline.adapters import webcam_intrinsics
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
