"""Timestamping, RGB normalisation, queueing, and reconnect supervision."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time

import numpy as np

from .contracts import FramePacket, FrameQuality, TimestampOrigin
from .queue import LatestFrameQueue
from .sources import FrameSource, SourceExhausted, SourceFrame, SourceReadError, StreamHealth


@dataclass
class StreamMetrics:
    emitted_frames: int = 0
    reconnect_count: int = 0
    timestamp_duplicates: int = 0
    timestamp_backwards: int = 0
    last_error: str | None = None
    state: StreamHealth = StreamHealth.CLOSED


class StreamHandler:
    """Turns one `FrameSource` into ordered `FramePacket` instances.

    Model inference is deliberately absent. Consumers pull the freshest available
    packet from `queue` and may run at a different rate from capture.
    """

    def __init__(
        self,
        source: FrameSource,
        queue_capacity: int = 3,
        reconnect_initial_s: float = 0.25,
        reconnect_max_s: float = 5.0,
    ) -> None:
        if reconnect_initial_s <= 0 or reconnect_max_s < reconnect_initial_s:
            raise ValueError("invalid reconnect backoff")
        self.source = source
        self.queue = LatestFrameQueue(queue_capacity)
        self.metrics = StreamMetrics()
        self._next_frame_id = 0
        self._last_source_timestamp_ns: int | None = None
        self._reconnect_initial_s = reconnect_initial_s
        self._reconnect_max_s = reconnect_max_s
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def open(self) -> None:
        self.metrics.state = StreamHealth.CONNECTING
        self.source.open()
        self.metrics.state = StreamHealth.HEALTHY

    def close(self) -> None:
        self.source.close()
        self.metrics.state = StreamHealth.CLOSED

    def pump_once(self) -> FramePacket:
        """Read one source frame, normalise it, then publish it to the bounded queue."""
        started_ns = time.monotonic_ns()
        source_frame = self.source.read()
        received_ns = time.monotonic_ns()
        rgb = self._as_rgb8(source_frame)
        timestamp_ns, origin = self._timestamp(source_frame, received_ns)
        status = self._timestamp_status(source_frame.source_timestamp_ns)
        age_available = origin is TimestampOrigin.DECODE_MONOTONIC
        packet = FramePacket(
            frame_id=self._next_frame_id,
            source_id=self.source.source_id,
            capture_timestamp_ns=timestamp_ns,
            timestamp_origin=origin,
            received_timestamp_ns=received_ns,
            rgb=rgb,
            quality=FrameQuality(
                decode_latency_ns=received_ns - started_ns,
                frame_age_ns=max(0, time.monotonic_ns() - timestamp_ns) if age_available else 0,
                frame_age_available=age_available,
                timestamp_status=status,
            ),
        )
        self._next_frame_id += 1
        result = self.queue.put(packet)
        packet.quality.queue_depth = result.depth
        packet.quality.dropped_frame_count = result.dropped_total
        self.metrics.emitted_frames += 1
        self.metrics.state = StreamHealth.HEALTHY
        return packet

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            raise RuntimeError("stream handler is already running")
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="stream-handler", daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = 2.0) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout_s)
        self.close()
        self.queue.close()

    def _run(self) -> None:
        delay_s = self._reconnect_initial_s
        while not self._stop_event.is_set():
            try:
                self.open()
                delay_s = self._reconnect_initial_s
                while not self._stop_event.is_set():
                    self.pump_once()
            except SourceExhausted:
                self.close()
                return
            except (SourceReadError, OSError, RuntimeError) as error:
                self.metrics.last_error = str(error)
                self.metrics.reconnect_count += 1
                self.metrics.state = StreamHealth.RECONNECTING
                self.source.close()
                self._stop_event.wait(delay_s)
                delay_s = min(delay_s * 2, self._reconnect_max_s)
        self.close()

    def _timestamp(self, frame: SourceFrame, received_ns: int) -> tuple[int, TimestampOrigin]:
        if frame.source_timestamp_ns is not None:
            return frame.source_timestamp_ns, TimestampOrigin.SOURCE_PRESENTATION
        return received_ns, TimestampOrigin.DECODE_MONOTONIC

    def _timestamp_status(self, source_timestamp_ns: int | None) -> str:
        if source_timestamp_ns is None:
            return "ok"
        status = "ok"
        if self._last_source_timestamp_ns is not None:
            if source_timestamp_ns == self._last_source_timestamp_ns:
                status = "duplicate"
                self.metrics.timestamp_duplicates += 1
            elif source_timestamp_ns < self._last_source_timestamp_ns:
                status = "backwards"
                self.metrics.timestamp_backwards += 1
        self._last_source_timestamp_ns = source_timestamp_ns
        return status

    @staticmethod
    def _as_rgb8(frame: SourceFrame) -> np.ndarray:
        image = frame.image
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("source image must be uint8 with shape [height, width, 3]")
        if frame.pixel_format == "RGB8":
            return np.ascontiguousarray(image)
        if frame.pixel_format == "BGR8":
            return np.ascontiguousarray(image[:, :, ::-1])
        raise ValueError(f"unsupported source pixel format: {frame.pixel_format}")
