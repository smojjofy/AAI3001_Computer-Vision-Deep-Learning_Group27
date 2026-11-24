"""Input adapters. They only obtain frames; StreamHandler assigns packet identity."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import os
from pathlib import Path
import threading
import time
from typing import Iterable, Protocol

import numpy as np

class StreamHealth(StrEnum):
    CONNECTING = "connecting"
    HEALTHY = "healthy"
    STALLED = "stalled"
    RECONNECTING = "reconnecting"
    CLOSED = "closed"


class ReplayMode(StrEnum):
    """How a finite recorded source should present frames."""

    PACED = "paced"
    FASTEST = "fastest"


class SourceExhausted(StopIteration):
    """A finite source ended normally (for example, a replay video)."""


class SourceReadError(RuntimeError):
    """A live source failed and may be reopened by the handler."""


def _load_zed_sdk():
    """Import the Stereolabs ZED SDK on demand.

    The SDK is optional: only the ZED source needs it. Importing it lazily keeps
    camera, OBS, video, and RTSP sources usable on machines without the SDK.

    On Windows the ``pyzed`` extension module links against ``sl_zed64.dll`` in
    the SDK ``bin`` directory. That directory is not always on ``PATH``, and a
    missing entry gives ``ImportError: DLL load failed`` rather than a clear
    message. This function adds the directory with ``os.add_dll_directory``,
    which is the supported way to make the dependency visible on Python 3.8+.
    It does not change ``PATH``, so it cannot disturb other libraries.
    """
    _add_zed_dll_directory()
    try:
        import pyzed.sl as zed
    except ModuleNotFoundError as error:
        raise SourceReadError(
            "The Stereolabs ZED SDK (pyzed) is not installed. Install it from the "
            "ZED SDK installer, or select a different --source."
        ) from error
    except ImportError as error:
        raise SourceReadError(
            "The Stereolabs ZED SDK (pyzed) is installed but its native library "
            "did not load. Check that the ZED SDK is installed and that "
            "sl_zed64.dll is in the SDK bin directory."
        ) from error
    return zed


def _add_zed_dll_directory() -> None:
    """Add the ZED SDK ``bin`` directory to the Windows DLL search path.

    The directory is found from ``ZED_SDK_ROOT`` when it is set, otherwise from
    the default install locations. A missing directory is not an error here,
    because a machine without the SDK must still reach the clear import message
    in :func:`_load_zed_sdk`. Adding the same directory twice is harmless.
    """
    if not hasattr(os, "add_dll_directory"):
        return
    candidates: list[os.PathLike[str] | str] = []
    root = os.environ.get("ZED_SDK_ROOT")
    if root:
        candidates.append(root)
    candidates.extend(
        (
            r"C:\Program Files (x86)\ZED SDK",
            r"C:\Program Files\ZED SDK",
        )
    )
    for base in candidates:
        bin_dir = os.path.join(os.fspath(base), "bin")
        if not os.path.isdir(bin_dir):
            continue
        key = os.path.normcase(os.path.abspath(bin_dir))
        if key in _ZED_DLL_DIRECTORIES:
            return
        try:
            handle = os.add_dll_directory(bin_dir)
        except OSError:
            continue
        # Keep the handle alive: removing it unloads the search path.
        _ZED_DLL_DIRECTORIES.add(key)
        _ZED_DLL_HANDLES.append(handle)
        return


_ZED_DLL_DIRECTORIES: set[str] = set()
_ZED_DLL_HANDLES: list[object] = []


@dataclass(frozen=True)
class SourceFrame:
    image: np.ndarray
    pixel_format: str = "BGR8"
    source_timestamp_ns: int | None = None
    depth: np.ndarray | None = None
    depth_units: str | None = None
    intrinsics: tuple[float, float, float, float] | None = None


class FrameSource(Protocol):
    source_id: str

    def open(self) -> None: ...
    def read(self) -> SourceFrame: ...
    def close(self) -> None: ...
    def health(self) -> StreamHealth: ...


class IterableFrameSource:
    """Deterministic source useful for replay tests and downstream development."""

    def __init__(self, frames: Iterable[SourceFrame], source_id: str = "replay") -> None:
        self.source_id = source_id
        self._frames = iter(frames)
        self._opened = False

    def open(self) -> None:
        self._opened = True

    def read(self) -> SourceFrame:
        if not self._opened:
            raise SourceReadError("source is not open")
        try:
            return next(self._frames)
        except StopIteration as error:
            raise SourceExhausted() from error

    def close(self) -> None:
        self._opened = False

    def health(self) -> StreamHealth:
        return StreamHealth.HEALTHY if self._opened else StreamHealth.CLOSED


class OpenCVFrameSource:
    """Shared OpenCV adapter for camera devices, files, and URL streams."""

    def __init__(
        self,
        location: int | str,
        source_id: str,
        finite: bool = False,
        replay_mode: ReplayMode = ReplayMode.FASTEST,
        camera_backend: str = "auto",
        capture_size: tuple[int, int] | None = None,
    ) -> None:
        if camera_backend not in {"auto", "dshow", "msmf"}:
            raise ValueError("camera_backend must be auto, dshow, or msmf")
        if capture_size is not None:
            width, height = capture_size
            if type(width) is not int or type(height) is not int or width < 1 or height < 1:
                raise ValueError("capture_size must be (width, height) positive integers, or None")
            capture_size = (int(width), int(height))
        self.location = location
        self.source_id = source_id
        self.finite = finite
        self.replay_mode = replay_mode
        self.camera_backend = camera_backend
        # None keeps the device's own default, so existing callers are unchanged.
        self.capture_size = capture_size
        self._capture = None
        self._capture_lock = threading.Lock()
        self._closed_event = threading.Event()
        self._replay_anchor_timestamp_ns: int | None = None
        self._replay_anchor_monotonic_ns: int | None = None

    def open(self) -> None:
        try:
            import cv2
        except ImportError as error:  # keeps replay-only tests dependency-light
            raise RuntimeError("OpenCV is required for camera, video, and RTSP sources") from error
        backend = {
            "auto": cv2.CAP_ANY,
            "dshow": cv2.CAP_DSHOW,
            "msmf": cv2.CAP_MSMF,
        }[self.camera_backend]
        capture = cv2.VideoCapture(self.location, backend)
        if self.capture_size is not None:
            # Requested before the first read; the driver may return its closest
            # supported size, so the delivered frame size stays authoritative.
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_size[0])
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_size[1])
        if not capture.isOpened():
            capture.release()
            raise SourceReadError(f"could not open source: {self.location}")
        with self._capture_lock:
            self._capture = capture
            self._closed_event.clear()
            self._replay_anchor_timestamp_ns = None
            self._replay_anchor_monotonic_ns = None

    def read(self) -> SourceFrame:
        with self._capture_lock:
            capture = self._capture
        if capture is None:
            raise SourceReadError("source is not open")
        ok, frame = capture.read()
        if not ok or frame is None:
            if self.finite:
                raise SourceExhausted()
            raise SourceReadError("frame decode failed or stream ended")
        timestamp_ns = None
        if self.finite:
            import cv2
            position_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
            if position_ms >= 0:
                timestamp_ns = int(position_ms * 1_000_000)
            if timestamp_ns is not None and self.replay_mode is ReplayMode.PACED:
                self._pace_replay(timestamp_ns)
        return SourceFrame(frame, "BGR8", timestamp_ns)

    def _pace_replay(self, timestamp_ns: int) -> None:
        """Wait until a file PTS is due, while allowing ``close`` to interrupt."""
        now_ns = time.monotonic_ns()
        if self._replay_anchor_timestamp_ns is None:
            self._replay_anchor_timestamp_ns = timestamp_ns
            self._replay_anchor_monotonic_ns = now_ns
            return
        assert self._replay_anchor_monotonic_ns is not None
        due_ns = self._replay_anchor_monotonic_ns + max(
            0, timestamp_ns - self._replay_anchor_timestamp_ns
        )
        remaining_s = (due_ns - now_ns) / 1_000_000_000
        if remaining_s > 0:
            self._closed_event.wait(remaining_s)
        if self._closed_event.is_set():
            raise SourceReadError("source was closed during recorded replay")

    def close(self) -> None:
        self._closed_event.set()
        with self._capture_lock:
            capture = self._capture
            self._capture = None
        if capture is not None:
            capture.release()

    def health(self) -> StreamHealth:
        with self._capture_lock:
            return StreamHealth.HEALTHY if self._capture is not None else StreamHealth.CLOSED


class CameraSource(OpenCVFrameSource):
    def __init__(
        self,
        device_index: int = 0,
        source_id: str | None = None,
        *,
        backend: str = "auto",
        capture_size: tuple[int, int] | None = None,
    ) -> None:
        super().__init__(
            device_index,
            source_id or f"camera:{device_index}",
            camera_backend=backend,
            capture_size=capture_size,
        )


class OBSVirtualCameraSource(CameraSource):
    """OBS Virtual Camera is an ordinary OS camera device; pass its device index."""

    def __init__(
        self,
        device_index: int = 0,
        *,
        backend: str = "auto",
        capture_size: tuple[int, int] | None = None,
    ) -> None:
        super().__init__(
            device_index,
            "obs-virtual-camera",
            backend=backend,
            capture_size=capture_size,
        )


class RecordedVideoSource(OpenCVFrameSource):
    def __init__(
        self,
        path: str | Path,
        source_id: str = "recorded-video",
        replay_mode: ReplayMode = ReplayMode.PACED,
    ) -> None:
        super().__init__(str(path), source_id, finite=True, replay_mode=replay_mode)


class RTSPSource(OpenCVFrameSource):
    def __init__(self, url: str, source_id: str = "rtsp") -> None:
        super().__init__(url, source_id)

class ZedCameraSource:
    """FrameSource implementation for the Stereolabs ZED SDK.

    The SDK is imported on first use, so importing the package does not require
    it. The source returns the left image as RGB8 plus the SDK depth map in
    metres. ``StreamHandler`` forwards ``depth``, ``depth_units``, and
    ``intrinsics`` into the packet, so a ZED run uses real calibrated
    intrinsics instead of the fixed-field-of-view estimate.

    Depth convention: the project treats invalid depth as ``NaN``, never 0 and
    never infinity. The SDK returns ``NaN``, 0, and ``inf`` for different
    failure cases, so this source maps every non-finite, zero, or negative
    value to ``NaN`` before it leaves the source. A caller in metres then needs
    no special case.
    """

    #: Depth reaches the packet in metres, matching the project convention.
    DEPTH_UNITS = "metres"

    def __init__(
        self,
        camera_id: int = 0,
        source_id: str | None = None,
        resolution: str | object | None = None,
        *,
        fps: int = 30,
        depth_mode: str = "QUALITY",
        colour_view: str = "LEFT",
    ) -> None:
        self._sl = _load_zed_sdk()
        self.source_id = source_id or f"zed:{camera_id}"
        self.camera_id = camera_id
        self.fps = int(fps)
        self._depth_mode_name = str(depth_mode).upper()
        self._colour_view_name = str(colour_view).upper()
        # HD720 is the default: it matches the recorded replay clips, holds
        # 30 fps with depth enabled, and is lighter than HD1080 on a laptop GPU.
        if resolution is None:
            self.resolution = self._sl.RESOLUTION.HD720
        elif isinstance(resolution, str):
            try:
                self.resolution = getattr(self._sl.RESOLUTION, resolution.upper())
            except AttributeError as error:
                raise ValueError(f"Unknown ZED resolution: {resolution!r}") from error
        else:
            self.resolution = resolution
        try:
            self.depth_mode = getattr(self._sl.DEPTH_MODE, self._depth_mode_name)
        except AttributeError as error:
            raise ValueError(f"Unknown ZED depth mode: {depth_mode!r}") from error
        # DEPTH_MODE.NONE captures RGB only. The depth map is skipped and the
        # module is not requested from the SDK, so `depth` stays None and a
        # downstream model can supply depth for the selected frames only.
        self.depth_enabled = self._depth_mode_name != "NONE"
        try:
            self.colour_view = getattr(self._sl.VIEW, self._colour_view_name)
        except AttributeError as error:
            raise ValueError(f"Unknown ZED view: {colour_view!r}") from error

        self._zed = self._sl.Camera()
        self._image_mat = self._sl.Mat()
        self._depth_mat = self._sl.Mat()
        self._runtime = self._sl.RuntimeParameters()
        self._intrinsics: tuple[float, float, float, float] | None = None
        self._opened = False

    def open(self) -> None:
        zed = self._sl
        init_params = zed.InitParameters()
        init_params.camera_resolution = self.resolution
        init_params.camera_fps = self.fps
        init_params.depth_mode = self.depth_mode
        init_params.coordinate_units = zed.UNIT.MILLIMETER
        init_params.set_from_camera_id(self.camera_id)

        err = self._zed.open(init_params)
        if err != zed.ERROR_CODE.SUCCESS:
            raise SourceReadError(f"Failed to open ZED camera: {err}")
        # Intrinsics must be read after a successful open; before that the call
        # returns an empty configuration.
        self._intrinsics = self._read_intrinsics()
        self._opened = True

    def _read_intrinsics(self) -> tuple[float, float, float, float] | None:
        """Read the left-camera pinhole intrinsics from the SDK, when available."""
        try:
            calibration = (
                self._zed.get_camera_information()
                .camera_configuration.calibration_parameters.left_cam
            )
            return (float(calibration.fx), float(calibration.fy), float(calibration.cx), float(calibration.cy))
        except Exception:
            return None

    def read(self) -> SourceFrame:
        zed = self._sl
        if not self._opened:
            raise SourceReadError("ZED source is not open")

        if self._zed.grab(self._runtime) != zed.ERROR_CODE.SUCCESS:
            raise SourceReadError("Failed to grab frame from ZED camera")

        self._zed.retrieve_image(self._image_mat, self.colour_view)
        image = self._image_mat.get_data()
        try:
            import cv2
        except ImportError as error:
            raise SourceReadError("OpenCV is required for the ZED colour conversion") from error
        # VIEW.LEFT normally returns BGRA. Convert explicitly: a plain channel
        # slice keeps the alpha plane and can break on a padded stride.
        if image.ndim == 3 and image.shape[2] == 4:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.ndim == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise SourceReadError(f"unexpected ZED image shape: {image.shape}")
        rgb_image = np.ascontiguousarray(rgb_image)

        depth_m = self._read_depth_metres() if self.depth_enabled else None

        timestamp_ns = int(self._zed.get_timestamp(zed.TIME_REFERENCE.IMAGE).get_nanoseconds())

        return SourceFrame(
            image=rgb_image,
            pixel_format="RGB8",
            source_timestamp_ns=timestamp_ns,
            depth=depth_m,
            depth_units=self.DEPTH_UNITS if depth_m is not None else None,
            intrinsics=self._intrinsics,
        )

    def _read_depth_metres(self) -> np.ndarray:
        """Return the depth map in metres with ``NaN`` for every invalid pixel.

        A pixel is invalid when it is non-finite, zero, or negative. The SDK
        uses ``inf`` for some no-measurement cases, and the project convention
        accepts only ``NaN`` for unknown depth.
        """
        zed = self._sl
        self._zed.retrieve_measure(self._depth_mat, zed.MEASURE.DEPTH)
        depth_mm = self._depth_mat.get_data().astype(np.float32, copy=True)
        valid = np.isfinite(depth_mm) & (depth_mm > 0.0)
        depth_m = np.full(depth_mm.shape, np.nan, dtype=np.float32)
        np.multiply(depth_mm, np.float32(0.001), out=depth_m, where=valid)
        return np.ascontiguousarray(depth_m, dtype=np.float32)

    def close(self) -> None:
        if self._opened:
            self._zed.close()
            self._opened = False

    def health(self) -> StreamHealth:
        return StreamHealth.HEALTHY if self._opened else StreamHealth.CLOSED


def zed_is_available() -> bool:
    """True when the ZED SDK imports and at least one ZED camera is connected.

    This never raises. It is used to decide whether the ``zed`` source can be
    offered, so a machine without the SDK or without a camera keeps the webcam
    path working.
    """
    try:
        zed = _load_zed_sdk()
    except (SourceReadError, ImportError, OSError):
        return False
    try:
        return len(zed.Camera.get_device_list()) > 0
    except Exception:
        return False


def describe_zed_devices() -> list[str]:
    """Return one short description for each connected ZED camera.

    The list is empty when the SDK or the cameras are absent. This never raises.
    """
    try:
        zed = _load_zed_sdk()
    except (SourceReadError, ImportError, OSError):
        return []
    try:
        devices = zed.Camera.get_device_list()
    except Exception:
        return []
    described: list[str] = []
    for index, device in enumerate(devices):
        model = getattr(device, "camera_model", "ZED")
        serial = getattr(device, "serial_number", None)
        described.append(f"{index}: {model}" + (f" (serial {serial})" if serial else ""))
    return described