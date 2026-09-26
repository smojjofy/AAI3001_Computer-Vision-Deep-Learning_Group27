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

    def __init__(
        self,
        location: int | str,
        source_id: str,
        finite: bool = False,
        replay_mode: ReplayMode = ReplayMode.FASTEST,
    ) -> None:
        self.location = location
        self.source_id = source_id
        self.finite = finite
        self.replay_mode = replay_mode
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
        capture = cv2.VideoCapture(self.location)
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
    def __init__(self, device_index: int = 0, source_id: str | None = None) -> None:
        super().__init__(device_index, source_id or f"camera:{device_index}")


class OBSVirtualCameraSource(CameraSource):
    """OBS Virtual Camera is an ordinary OS camera device; pass its device index."""

    def __init__(self, device_index: int = 0) -> None:
        super().__init__(device_index, "obs-virtual-camera")


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
