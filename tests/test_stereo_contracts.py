"""Check the optional rectified right view and stereo calibration.

These tests cover Stage 1 of the stereo plan: the frame contract carries one
left-right pair under one frame identity. They need no camera and no SDK.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
STREAM_SRC = ROOT / "modules" / "stream_handler" / "src"
if str(STREAM_SRC) not in sys.path:
    sys.path.insert(0, str(STREAM_SRC))

from stream_handler import (  # noqa: E402
    FramePacket,
    FrameQuality,
    IterableFrameSource,
    SourceFrame,
    StereoCalibration,
    StreamHandler,
    TimestampOrigin,
)


def _calibration(**overrides) -> StereoCalibration:
    values = dict(
        fx=524.08, fy=524.08, cx=640.42, cy=346.93, baseline_m=0.12004, width=64, height=48
    )
    values.update(overrides)
    return StereoCalibration(**values)


def _packet(**overrides) -> FramePacket:
    values = dict(
        frame_id=1,
        source_id="stereo-test",
        capture_timestamp_ns=1,
        timestamp_origin=TimestampOrigin.DECODE_MONOTONIC,
        rgb=np.zeros((48, 64, 3), np.uint8),
        quality=FrameQuality(0, 0, True),
    )
    values.update(overrides)
    return FramePacket(**values)


# ------------------------------------------------------------ calibration


def test_calibration_exposes_an_intrinsics_tuple():
    cal = _calibration()
    assert cal.intrinsics == (524.08, 524.08, 640.42, 346.93)


def test_calibration_matches_only_its_own_size():
    cal = _calibration()
    assert cal.matches(64, 48) is True
    assert cal.matches(1280, 720) is False


def test_calibration_is_frozen():
    cal = _calibration()
    with pytest.raises(Exception):
        cal.baseline_m = 0.5  # type: ignore[misc]


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"baseline_m": 0.0}, "baseline"),
        ({"baseline_m": -0.12}, "baseline"),
        ({"fx": 0.0}, "focal"),
        ({"fy": -1.0}, "focal"),
        ({"fx": float("nan")}, "finite"),
        ({"height": 0}, "dimensions"),
    ],
)
def test_calibration_rejects_bad_values(overrides, message):
    with pytest.raises(ValueError, match=message):
        _calibration(**overrides)


# ---------------------------------------------------------------- pairing


def test_packet_without_stereo_is_still_valid():
    """Every existing one-image source keeps working, and reports version 1.1."""
    packet = _packet()
    assert packet.right_rgb is None
    assert packet.stereo is None
    assert packet.schema_version == "1.1"
    envelope = packet.metadata()
    assert "right_image" not in envelope
    assert "stereo" not in envelope


def test_packet_carries_one_pair_under_one_identity():
    packet = _packet(
        right_rgb=np.zeros((48, 64, 3), np.uint8), stereo=_calibration()
    )
    envelope = packet.metadata()
    assert envelope["right_image"]["width"] == 64
    assert envelope["right_image"]["height"] == 48
    assert envelope["stereo"]["baseline_m"] == pytest.approx(0.12004)
    # One envelope means one frame ID and one timestamp for both views.
    assert envelope["frame_id"] == 1
    assert envelope["capture_timestamp_ns"] == 1


def test_right_image_without_calibration_is_rejected():
    with pytest.raises(ValueError, match="requires a stereo calibration"):
        _packet(right_rgb=np.zeros((48, 64, 3), np.uint8))


def test_calibration_without_right_image_is_rejected():
    with pytest.raises(ValueError, match="requires right_rgb"):
        _packet(stereo=_calibration())


def test_right_image_size_must_match_left():
    with pytest.raises(ValueError, match="must match rgb shape"):
        _packet(right_rgb=np.zeros((24, 32, 3), np.uint8), stereo=_calibration())


def test_calibration_size_must_match_the_frame():
    with pytest.raises(ValueError, match="calibration describes"):
        _packet(
            right_rgb=np.zeros((48, 64, 3), np.uint8),
            stereo=_calibration(width=1280, height=720),
        )


def test_right_image_must_be_uint8_three_channel():
    with pytest.raises(ValueError, match="right_rgb must be a uint8"):
        _packet(right_rgb=np.zeros((48, 64), np.uint8), stereo=_calibration())
    with pytest.raises(ValueError, match="right_rgb must be a uint8"):
        _packet(right_rgb=np.zeros((48, 64, 3), np.float32), stereo=_calibration())


# ------------------------------------------------------- handler forwarding


def _read_one(source: IterableFrameSource) -> FramePacket | None:
    handler = StreamHandler(source, queue_capacity=2, max_reconnect_attempts=1)
    handler.start()
    try:
        for _ in range(100):
            packet = handler.queue.get(timeout_s=0.5)
            if packet is not None:
                return packet
        return None
    finally:
        handler.stop()


def test_handler_forwards_the_right_view_and_calibration():
    left = np.zeros((48, 64, 3), np.uint8)
    left[..., 1] = 255
    right = np.zeros((48, 64, 3), np.uint8)
    right[..., 2] = 255
    frames = [
        SourceFrame(
            image=left,
            pixel_format="RGB8",
            source_timestamp_ns=index * 1_000_000_000,
            right_image=right,
            right_pixel_format="RGB8",
            stereo=_calibration(),
        )
        for index in range(2)
    ]
    packet = _read_one(IterableFrameSource(frames, source_id="stereo-test"))
    assert packet is not None
    assert packet.right_rgb is not None
    assert packet.right_rgb.shape == (48, 64, 3)
    # The right plane keeps its own colour: blue stays in channel 2.
    assert (packet.right_rgb[..., 2] == 255).all()
    assert packet.stereo is not None
    assert packet.stereo.baseline_m == pytest.approx(0.12004)


def test_handler_forwards_a_bgr_right_view_as_rgb():
    left = np.zeros((48, 64, 3), np.uint8)
    right_bgr = np.zeros((48, 64, 3), np.uint8)
    right_bgr[..., 0] = 255  # blue in a BGR buffer
    frames = [
        SourceFrame(
            image=left,
            pixel_format="RGB8",
            source_timestamp_ns=0,
            right_image=right_bgr,
            right_pixel_format="BGR8",
            stereo=_calibration(),
        )
    ]
    packet = _read_one(IterableFrameSource(frames, source_id="stereo-bgr"))
    assert packet is not None
    assert packet.right_rgb is not None
    # Blue must arrive in channel 2 after the conversion.
    assert (packet.right_rgb[..., 2] == 255).all()
    assert (packet.right_rgb[..., 0] == 0).all()


def test_handler_leaves_stereo_fields_empty_for_a_one_image_source():
    frames = [SourceFrame(image=np.zeros((48, 64, 3), np.uint8), pixel_format="RGB8")]
    packet = _read_one(IterableFrameSource(frames, source_id="mono"))
    assert packet is not None
    assert packet.right_rgb is None
    assert packet.stereo is None


# ------------------------------------------------------------------ schema


def test_schema_declares_the_stereo_fields():
    import json

    schema = json.loads(
        (ROOT / "shared" / "schemas" / "frame_packet.schema.json").read_text(encoding="utf-8")
    )
    properties = schema["properties"]
    assert properties["schema_version"]["enum"] == ["1.0", "1.1"]
    assert "right_image" in properties
    assert "stereo" in properties
    assert set(properties["stereo"]["required"]) == {
        "fx", "fy", "cx", "cy", "baseline_m", "width", "height",
    }
    # The envelope must not permit undeclared fields, so a new field needs a
    # deliberate schema change.
    assert schema["additionalProperties"] is False


def test_envelope_fields_are_all_declared_in_the_schema():
    """Every key the envelope emits must appear in the schema."""
    import json

    schema = json.loads(
        (ROOT / "shared" / "schemas" / "frame_packet.schema.json").read_text(encoding="utf-8")
    )
    declared = set(schema["properties"])
    packet = _packet(right_rgb=np.zeros((48, 64, 3), np.uint8), stereo=_calibration())
    undeclared = set(packet.metadata()) - declared
    assert not undeclared, f"envelope emits undeclared keys: {sorted(undeclared)}"

# ------------------------------------------------------- ZED stereo source


def test_zed_emit_right_requires_depth_mode_none():
    """The SDK depth path already computes depth from the pair."""
    from stream_handler.sources import ZedCameraSource

    with pytest.raises(ValueError, match="emit_right requires depth_mode NONE"):
        ZedCameraSource(0, depth_mode="QUALITY", emit_right=True)


def test_zed_stereo_capture_with_a_fake_sdk(monkeypatch):
    """The source returns one pair plus calibration, and never requests depth."""
    from stream_handler import sources as sources_module

    class _Mat:
        def __init__(self, plane):
            self._plane = plane

        def get_data(self):
            return self._plane

    class _FakeZed:
        class ERROR_CODE:
            SUCCESS = "SUCCESS"

        class RESOLUTION:
            VGA = "VGA"
            HD720 = "HD720"
            HD1080 = "HD1080"

        class DEPTH_MODE:
            NONE = "NONE"
            QUALITY = "QUALITY"

        class VIEW:
            LEFT = "LEFT"
            RIGHT = "RIGHT"

        class UNIT:
            MILLIMETER = "MILLIMETER"

        class MEASURE:
            DEPTH = "DEPTH"

        class TIME_REFERENCE:
            IMAGE = "IMAGE"

        class InitParameters:
            def set_from_camera_id(self, camera_id):
                self.camera_id = camera_id

        class RuntimeParameters:
            pass

        height, width = 4, 6
        left = np.zeros((height, width, 4), np.uint8)
        left[..., 0] = 30   # B
        left[..., 2] = 200  # R
        right = np.zeros((height, width, 4), np.uint8)
        right[..., 1] = 120  # G differs between views
        depth_requests = 0

        def Camera(self):
            return self

        def Mat(self):
            return _Mat(None)

        def open(self, init):
            return self.ERROR_CODE.SUCCESS

        def grab(self, runtime=None):
            return self.ERROR_CODE.SUCCESS

        def close(self):
            return None

        def retrieve_image(self, mat, view):
            mat._plane = self.left if view == self.VIEW.LEFT else self.right

        def retrieve_measure(self, mat, measure):
            type(self).depth_requests += 1
            mat._plane = np.zeros((self.height, self.width), np.float32)

        def get_camera_information(self):
            from types import SimpleNamespace

            calibration = SimpleNamespace(
                left_cam=SimpleNamespace(fx=262.0, fy=262.0, cx=335.0, cy=180.0),
                get_camera_baseline=lambda: 120.04,
            )
            configuration = SimpleNamespace(
                calibration_parameters=calibration,
                resolution=SimpleNamespace(width=6, height=4),
            )
            return SimpleNamespace(camera_configuration=configuration)

        def get_timestamp(self, reference):
            from types import SimpleNamespace

            return SimpleNamespace(get_nanoseconds=lambda: 1234)

    fake = _FakeZed()
    monkeypatch.setattr(sources_module, "_load_zed_sdk", lambda: fake)

    source = sources_module.ZedCameraSource(0, depth_mode="NONE", emit_right=True)
    source.open()
    frame = source.read()

    assert frame.right_image is not None
    assert frame.right_image.shape == (4, 6, 3)
    assert frame.stereo is not None
    assert frame.stereo.baseline_m == pytest.approx(0.12004)
    assert frame.stereo.width == 6 and frame.stereo.height == 4
    assert frame.depth is None
    # Depth must not be requested when the SDK is not computing it.
    assert _FakeZed.depth_requests == 0
    # BGRA becomes RGB: the red plane moves from channel 2 to channel 0.
    assert (frame.image[..., 0] == 200).all()
    source.close()


def test_zed_rejects_an_unusable_baseline(monkeypatch):
    from stream_handler import sources as sources_module

    class _FakeZed:
        class ERROR_CODE:
            SUCCESS = "SUCCESS"

        class RESOLUTION:
            VGA = "VGA"
            HD720 = "HD720"

        class DEPTH_MODE:
            NONE = "NONE"

        class VIEW:
            LEFT = "LEFT"
            RIGHT = "RIGHT"

        class UNIT:
            MILLIMETER = "MILLIMETER"

        class MEASURE:
            DEPTH = "DEPTH"

        class TIME_REFERENCE:
            IMAGE = "IMAGE"

        class InitParameters:
            def set_from_camera_id(self, camera_id):
                self.camera_id = camera_id

        class RuntimeParameters:
            pass

        def Camera(self):
            return self

        def Mat(self):
            return _NoData()

        def open(self, init):
            return self.ERROR_CODE.SUCCESS

        def get_camera_information(self):
            from types import SimpleNamespace

            calibration = SimpleNamespace(
                left_cam=SimpleNamespace(fx=262.0, fy=262.0, cx=335.0, cy=180.0),
                get_camera_baseline=lambda: 0.0,
            )
            configuration = SimpleNamespace(
                calibration_parameters=calibration,
                resolution=SimpleNamespace(width=6, height=4),
            )
            return SimpleNamespace(camera_configuration=configuration)

    class _NoData:
        def get_data(self):
            return None

    monkeypatch.setattr(sources_module, "_load_zed_sdk", lambda: _FakeZed())
    source = sources_module.ZedCameraSource(0, depth_mode="NONE", emit_right=True)
    with pytest.raises(sources_module.SourceReadError, match="baseline is unusable"):
        source.open()
