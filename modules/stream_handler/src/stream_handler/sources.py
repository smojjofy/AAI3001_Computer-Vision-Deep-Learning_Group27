"""Input adapters. They only obtain frames; StreamHandler assigns packet identity."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
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
    """
    try:
        import pyzed.sl as zed
    except ModuleNotFoundError as error:
        raise SourceReadError(
            "The Stereolabs ZED SDK (pyzed) is not installed. Install it from the "
            "ZED SDK installer, or select a different --source."
        ) from error
    return zed


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
    it. The source returns the left image as BGR8 plus the SDK depth map in
    millimetres. ``StreamHandler`` does not forward ``SourceFrame.depth`` yet;
    the depth map stays available for the refresh stage.
    """

    def __init__(
        self,
        camera_id: int = 0,
        source_id: str | None = None,
        resolution: str | object | None = None,
    ) -> None:
        self._sl = _load_zed_sdk()
        self.source_id = source_id or f"zed:{camera_id}"
        self.camera_id = camera_id
        if resolution is None:
            self.resolution = self._sl.RESOLUTION.HD1080
        elif isinstance(resolution, str):
            try:
                self.resolution = getattr(self._sl.RESOLUTION, resolution.upper())
            except AttributeError as error:
                raise ValueError(f"Unknown ZED resolution: {resolution!r}") from error
        else:
            self.resolution = resolution

        self._zed = self._sl.Camera()
        self._image_mat = self._sl.Mat()
        self._depth_mat = self._sl.Mat()
        self._intrinsics: tuple[float, float, float, float] | None = None
        self._opened = False

    def open(self) -> None:
        zed = self._sl
        init_params = zed.InitParameters()
        init_params.camera_resolution = self.resolution
        init_params.depth_mode = zed.DEPTH_MODE.ULTRA
        init_params.coordinate_units = zed.UNIT.MILLIMETER
        init_params.set_from_camera_id(self.camera_id)

        err = self._zed.open(init_params)
        if err != zed.ERROR_CODE.SUCCESS:
            raise SourceReadError(f"Failed to open ZED camera: {err}")
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

        if self._zed.grab() == zed.ERROR_CODE.SUCCESS:
            self._zed.retrieve_image(self._image_mat, zed.VIEW.LEFT)
            image = self._image_mat.get_data()
            try:
                import cv2
            except ImportError as error:
                raise SourceReadError("OpenCV is required for the ZED colour conversion") from error
            # VIEW.LEFT normally returns BGRA. Convert explicitly: a plain
            # channel slice keeps the alpha plane and can break on a padded
            # stride.
            if image.ndim == 3 and image.shape[2] == 4:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            elif image.ndim == 3 and image.shape[2] == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                raise SourceReadError(f"unexpected ZED image shape: {image.shape}")
            rgb_image = np.ascontiguousarray(rgb_image)

            self._zed.retrieve_measure(self._depth_mat, zed.MEASURE.DEPTH)
            depth_map = self._depth_mat.get_data().copy()
            timestamp_ns = self._zed.get_timestamp(zed.TIME_REFERENCE.IMAGE).get_nanoseconds()

            return SourceFrame(
                image=rgb_image,
                pixel_format="RGB8",
                source_timestamp_ns=timestamp_ns,
                depth=np.ascontiguousarray(depth_map, dtype=np.float32),
                depth_units="millimetres",
                intrinsics=self._intrinsics,
            )
        raise SourceReadError("Failed to grab frame from ZED camera")

    def close(self) -> None:
        if self._opened:
            self._zed.close()
            self._opened = False

    def health(self) -> StreamHealth:
        return StreamHealth.HEALTHY if self._opened else StreamHealth.CLOSED