"""Check the ZED source boundary and the source-selection rule.

These tests do not need a ZED camera or the ZED SDK. They check the parts that
can fail without hardware: the depth convention, the availability probe, the
package exports, and the ``auto`` rule.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
STREAM_SRC = ROOT / "modules" / "stream_handler" / "src"


def _stream_handler():
    """Import the ``stream_handler.sources`` module from the component tree.

    The private SDK loader and the source class live in ``sources``. The package
    re-exports the public names, so both views must stay importable.
    """
    text = str(STREAM_SRC)
    if text not in sys.path:
        sys.path.insert(0, text)
    from stream_handler import sources

    return sources


class _FakeZed:
    """A small stand-in for ``pyzed.sl`` that returns a chosen depth plane."""

    def __init__(self, depth_mm: np.ndarray) -> None:
        self._depth_mm = depth_mm
        self.closed = False
        self.open_calls = 0
        self.grab_result = self.ERROR_CODE.SUCCESS
        self.image = np.zeros((*depth_mm.shape, 4), np.uint8)

    # -- nested API shims ------------------------------------------------
    class ERROR_CODE:
        SUCCESS = "SUCCESS"
        NOT_A_CAMERA_SETTINGS_FILE = "FAIL"

    class RESOLUTION:
        HD720 = "HD720"
        HD1080 = "HD1080"
        VGA = "VGA"
        HD2K = "HD2K"

    class DEPTH_MODE:
        NEURAL = "NEURAL"
        NEURAL_PLUS = "NEURAL_PLUS"
        ULTRA = "ULTRA"
        QUALITY = "QUALITY"
        PERFORMANCE = "PERFORMANCE"

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
        camera_resolution = None
        camera_fps = 0
        depth_mode = None
        coordinate_units = None

        def set_from_camera_id(self, camera_id: int) -> None:
            self.camera_id = camera_id

    class RuntimeParameters:
        pass

    def Camera(self):  # noqa: N802 - mirrors the SDK spelling
        return self

    def Mat(self):  # noqa: N802 - mirrors the SDK spelling
        # One slot object: retrieve_image and retrieve_measure fill it, and
        # get_data reads it back.
        return _MatSlot()

    def open(self, init):
        self.open_calls += 1
        self.init = init
        return self.ERROR_CODE.SUCCESS

    def grab(self, runtime=None):
        return self.grab_result

    def close(self) -> None:
        self.closed = True

    def retrieve_image(self, mat, view) -> None:
        mat.set_data(self.image)

    def retrieve_measure(self, mat, measure) -> None:
        mat.set_data(self._depth_mm)

    def get_camera_information(self):
        left = SimpleNamespace(fx=522.5, fy=522.5, cx=640.4, cy=346.9)
        return SimpleNamespace(
            camera_configuration=SimpleNamespace(
                calibration_parameters=SimpleNamespace(left_cam=left)
            )
        )

    def get_timestamp(self, reference):
        return SimpleNamespace(get_nanoseconds=lambda: 1_234_567_890)


class _MatSlot:
    """A ``sl.Mat`` stand-in that holds the plane written into it."""

    def __init__(self) -> None:
        self._data = None

    def set_data(self, data) -> None:
        self._data = data

    def get_data(self):
        return self._data


@pytest.fixture
def zed_module():
    return _stream_handler()


def _source_with_depth(zed_module, depth_mm: np.ndarray):
    fake = _FakeZed(depth_mm)

    class _Patched(zed_module.ZedCameraSource):
        def __init__(self, depth_enabled: bool = True):
            self._sl = fake
            self.source_id = "zed:test"
            self.camera_id = 0
            self.fps = 30
            self.resolution = fake.RESOLUTION.HD720
            self.depth_mode = fake.DEPTH_MODE.QUALITY
            self.colour_view = fake.VIEW.LEFT
            # DEPTH_MODE.NONE sets this to False, and then read() reports no depth.
            self.depth_enabled = bool(depth_enabled)
            self._zed = fake
            self._image_mat = fake.Mat()
            self._depth_mat = fake.Mat()
            self._runtime = fake.RuntimeParameters()
            self._intrinsics = None
            self._opened = False

    return _Patched(), fake


def test_depth_uses_nan_and_metres(zed_module):
    """Invalid SDK depth becomes NaN, and every valid pixel becomes metres."""
    depth_mm = np.array(
        [
            [1000.0, 2500.0, 0.0],
            [np.inf, np.nan, -5.0],
            [500.0, 3000.0, 2000.0],
        ],
        dtype=np.float32,
    )
    source, _fake = _source_with_depth(zed_module, depth_mm)
    source.open()
    frame = source.read()

    assert frame.depth_units == "metres"
    assert frame.depth is not None
    depth = frame.depth
    assert depth.dtype == np.float32
    # Valid pixels are converted to metres.
    assert depth[0, 0] == pytest.approx(1.0)
    assert depth[0, 1] == pytest.approx(2.5)
    assert depth[2, 2] == pytest.approx(2.0)
    # Zero, infinite, NaN, and negative values are all unknown.
    for row, column in ((0, 2), (1, 0), (1, 1), (1, 2)):
        assert np.isnan(depth[row, column]), (row, column)
    # The project convention never uses zero or infinity for unknown depth.
    assert not np.isinf(depth).any()
    assert (depth[~np.isnan(depth)] > 0).all()


def test_depth_plane_is_contiguous_float32(zed_module):
    depth_mm = np.full((4, 6), 1500.0, np.float32)
    source, _fake = _source_with_depth(zed_module, depth_mm)
    source.open()
    frame = source.read()
    assert frame.depth.flags["C_CONTIGUOUS"]
    assert frame.depth.shape == depth_mm.shape


def test_intrinsics_are_read_after_open(zed_module):
    """Intrinsics come from the SDK calibration, not the fixed-FOV estimate."""
    source, _fake = _source_with_depth(zed_module, np.full((4, 4), 1000.0, np.float32))
    assert source._intrinsics is None
    source.open()
    frame = source.read()
    assert frame.intrinsics == pytest.approx((522.5, 522.5, 640.4, 346.9))


def test_read_before_open_raises(zed_module):
    source, _fake = _source_with_depth(zed_module, np.full((4, 4), 1000.0, np.float32))
    with pytest.raises(zed_module.SourceReadError):
        source.read()


def test_grab_failure_raises_source_read_error(zed_module):
    source, fake = _source_with_depth(zed_module, np.full((4, 4), 1000.0, np.float32))
    source.open()
    fake.grab_result = fake.ERROR_CODE.NOT_A_CAMERA_SETTINGS_FILE
    with pytest.raises(zed_module.SourceReadError):
        source.read()


def test_unknown_resolution_and_mode_are_rejected(zed_module):
    """A bad name fails at construction, before the camera is touched."""
    fake = _FakeZed(np.full((4, 4), 1000.0, np.float32))

    class _Bare(zed_module.ZedCameraSource):
        def __init__(self, **kwargs):
            self._sl = fake
            super().__init__(**kwargs)

    with pytest.raises(ValueError, match="Unknown ZED resolution"):
        _Bare(resolution="NOT_A_MODE")
    with pytest.raises(ValueError, match="Unknown ZED depth mode"):
        _Bare(depth_mode="NOT_A_MODE")
    with pytest.raises(ValueError, match="Unknown ZED view"):
        _Bare(colour_view="NOT_A_VIEW")


def test_zed_source_is_exported(zed_module):
    import stream_handler

    assert "ZedCameraSource" in stream_handler.__all__
    assert "zed_is_available" in stream_handler.__all__
    assert "describe_zed_devices" in stream_handler.__all__
    assert stream_handler.ZedCameraSource is zed_module.ZedCameraSource


def test_zed_probe_never_raises(zed_module):
    """The probe reports a boolean and never raises, whatever the machine has."""
    assert isinstance(zed_module.zed_is_available(), bool)
    devices = zed_module.describe_zed_devices()
    assert isinstance(devices, list)
    assert all(isinstance(entry, str) for entry in devices)


def test_zed_probe_returns_false_without_the_sdk(zed_module, monkeypatch):
    """A missing SDK reports 'not available' and does not break the webcam path."""

    def _raise():
        raise zed_module.SourceReadError("no sdk")

    monkeypatch.setattr(zed_module, "_load_zed_sdk", _raise)
    assert zed_module.zed_is_available() is False
    assert zed_module.describe_zed_devices() == []


def test_zed_probe_returns_false_without_a_camera(zed_module, monkeypatch):
    class _NoDevices:
        class Camera:  # noqa: N801 - mirrors the SDK spelling
            @staticmethod
            def get_device_list():
                return []

    monkeypatch.setattr(zed_module, "_load_zed_sdk", lambda: _NoDevices)
    assert zed_module.zed_is_available() is False
    assert zed_module.describe_zed_devices() == []


def test_auto_prefers_webcam_when_zed_is_absent():
    """The ``auto`` rule keeps the webcam unless a ZED is ready."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib.util;"
            f"spec=importlib.util.spec_from_file_location('app', r'{ROOT / 'scripts' / 'run_integration_app.py'}');"
            "m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);"
            "m.zed_is_available=lambda: False;"
            "print(m.resolve_source_choice('auto'));"
            "m.zed_is_available=lambda: True;"
            "print(m.resolve_source_choice('auto'))",
        ],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=str(ROOT),
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip() in {"webcam", "zed"}]
    assert result.returncode == 0, result.stderr[-2000:]
    assert lines == ["webcam", "zed"]


def test_explicit_choice_is_never_overridden():
    """An explicit source is returned unchanged, whatever the probe says."""
    from importlib import util

    spec = util.spec_from_file_location("app", ROOT / "scripts" / "run_integration_app.py")
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for choice in ("webcam", "zed", "obs", "video", "camera"):
        assert module.resolve_source_choice(choice) == choice

def test_rgb_only_mode_reports_no_depth(zed_module):
    """DEPTH_MODE.NONE captures RGB and leaves depth to a selected-frame model."""
    source, _fake = _source_with_depth(zed_module, np.full((4, 4), 1000.0, np.float32))
    source.depth_enabled = False
    source.open()
    frame = source.read()
    assert frame.depth is None
    assert frame.depth_units is None
    assert frame.image.ndim == 3


def test_depth_mode_none_is_offered_by_the_cli():
    """The CLI accepts NONE for the RGB-only benchmark path."""
    result = subprocess.run(
        [sys.executable, "-m", "scripts.run_integration_app", "--help"],
        capture_output=True, text=True, timeout=180, cwd=str(ROOT),
    )
    assert result.returncode == 0, result.stderr[-2000:]
    # argparse prints the choices in braces for this option.
    assert "NEURAL_PLUS,ULTRA,QUALITY,PERFORMANCE,NONE" in result.stdout.replace(", ", ",")
