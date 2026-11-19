"""Boundary tests for the optional capture size on the camera sources.

The capture size is an opt-in keyword. These tests pin the two properties that
matter: omitting it keeps the previous behaviour, and an invalid value is
rejected before any device is opened.
"""

from __future__ import annotations

import pytest

from stream_handler.sources import (
    CameraSource,
    OBSVirtualCameraSource,
    OpenCVFrameSource,
    RecordedVideoSource,
)


def test_camera_source_defaults_to_the_device_size() -> None:
    # Existing callers pass no capture size and must be unaffected.
    source = CameraSource(0, backend="dshow")
    assert source.capture_size is None
    assert source.camera_backend == "dshow"
    assert source.location == 0
    assert source.source_id == "camera:0"


def test_camera_source_accepts_an_explicit_capture_size() -> None:
    source = CameraSource(1, backend="dshow", capture_size=(1280, 720))
    assert source.capture_size == (1280, 720)
    assert source.source_id == "camera:1"


def test_obs_source_accepts_a_capture_size_and_keeps_its_id() -> None:
    source = OBSVirtualCameraSource(2, backend="dshow")
    assert source.capture_size is None
    assert source.source_id == "obs-virtual-camera"
    resized = OBSVirtualCameraSource(2, backend="dshow", capture_size=(1920, 1080))
    assert resized.capture_size == (1920, 1080)
    assert resized.source_id == "obs-virtual-camera"


def test_recorded_video_source_is_unaffected() -> None:
    # A file has its own size; the option must not leak into this class.
    source = RecordedVideoSource("clip.mp4")
    assert source.capture_size is None


@pytest.mark.parametrize(
    "capture_size",
    [(0, 720), (1280, 0), (-1, 720), (1280.0, 720), (1280, 720.5), (720,), (720, 480, 24)],
)
def test_invalid_capture_size_is_rejected(capture_size) -> None:
    with pytest.raises((ValueError, TypeError)):
        OpenCVFrameSource(0, "test", capture_size=capture_size)


def test_integer_like_values_are_normalised() -> None:
    source = OpenCVFrameSource(0, "test", capture_size=(1280, 720))
    width, height = source.capture_size
    assert type(width) is int and type(height) is int
