"""Input adapters. They only obtain frames; StreamHandler assigns packet identity."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np


class StreamHealth(StrEnum):
    CONNECTING = "connecting"
    HEALTHY = "healthy"
    STALLED = "stalled"
    RECONNECTING = "reconnecting"
    CLOSED = "closed"


class SourceExhausted(StopIteration):
    """A finite source ended normally (for example, a replay video)."""


class SourceReadError(RuntimeError):
    """A live source failed and may be reopened by the handler."""


@dataclass(frozen=True)
class SourceFrame:
    image: np.ndarray
    pixel_format: str = "BGR8"
    source_timestamp_ns: int | None = None


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

    def __init__(self, location: int | str, source_id: str, finite: bool = False) -> None:
        self.location = location
        self.source_id = source_id
        self.finite = finite
        self._capture = None

    def open(self) -> None:
        try:
            import cv2
        except ImportError as error:  # keeps replay-only tests dependency-light
            raise RuntimeError("OpenCV is required for camera, video, and RTSP sources") from error
        self._capture = cv2.VideoCapture(self.location)
        if not self._capture.isOpened():
            self._capture.release()
            self._capture = None
            raise SourceReadError(f"could not open source: {self.location}")

    def read(self) -> SourceFrame:
        if self._capture is None:
            raise SourceReadError("source is not open")
        ok, frame = self._capture.read()
        if not ok or frame is None:
            if self.finite:
                raise SourceExhausted()
            raise SourceReadError("frame decode failed or stream ended")
        timestamp_ns = None
        if self.finite:
            import cv2
            position_ms = self._capture.get(cv2.CAP_PROP_POS_MSEC)
            if position_ms >= 0:
                timestamp_ns = int(position_ms * 1_000_000)
        return SourceFrame(frame, "BGR8", timestamp_ns)

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def health(self) -> StreamHealth:
        return StreamHealth.HEALTHY if self._capture is not None else StreamHealth.CLOSED


class CameraSource(OpenCVFrameSource):
    def __init__(self, device_index: int = 0, source_id: str | None = None) -> None:
        super().__init__(device_index, source_id or f"camera:{device_index}")


class OBSVirtualCameraSource(CameraSource):
    """OBS Virtual Camera is an ordinary OS camera device; pass its device index."""

    def __init__(self, device_index: int = 0) -> None:
        super().__init__(device_index, "obs-virtual-camera")


class RecordedVideoSource(OpenCVFrameSource):
    def __init__(self, path: str | Path, source_id: str = "recorded-video") -> None:
        super().__init__(str(path), source_id, finite=True)


class RTSPSource(OpenCVFrameSource):
    def __init__(self, url: str, source_id: str = "rtsp") -> None:
        super().__init__(url, source_id)
