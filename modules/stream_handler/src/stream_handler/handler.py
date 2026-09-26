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
    capture_fps: float = 0.0
    last_successful_frame_ns: int | None = None
    decode_latency_mean_ns: float = 0.0
    decode_latency_max_ns: int = 0


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
        stall_timeout_s: float | None = 5.0,
        max_reconnect_attempts: int | None = None,
    ) -> None:
        if reconnect_initial_s <= 0 or reconnect_max_s < reconnect_initial_s:
            raise ValueError("invalid reconnect backoff")
        if stall_timeout_s is not None and stall_timeout_s <= 0:
            raise ValueError("stall_timeout_s must be positive or None")
        if max_reconnect_attempts is not None and max_reconnect_attempts < 1:
            raise ValueError("max_reconnect_attempts must be at least one or None")
        self.source = source
        self.queue = LatestFrameQueue(queue_capacity)
        self.metrics = StreamMetrics()
        self._next_frame_id = 0
        self._last_source_timestamp_ns: int | None = None
        self._reconnect_initial_s = reconnect_initial_s
        self._reconnect_max_s = reconnect_max_s
        self._stall_timeout_ns = None if stall_timeout_s is None else int(stall_timeout_s * 1_000_000_000)
        self._max_reconnect_attempts = max_reconnect_attempts
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._opened_ns: int | None = None
        self._first_successful_frame_ns: int | None = None

    def open(self) -> None:
        self.metrics.state = StreamHealth.CONNECTING
        self.source.open()
        self._opened_ns = time.monotonic_ns()
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
        self.queue.put(
            packet,
            lambda accounting: self._set_queue_quality(packet, accounting.depth, accounting.dropped_total),
        )
        self.metrics.emitted_frames += 1
        self._record_success(received_ns, packet.quality.decode_latency_ns)
        source_health = self.source.health()
        self.metrics.state = source_health if source_health is not StreamHealth.CLOSED else StreamHealth.HEALTHY
        return packet

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            raise RuntimeError("stream handler is already running")
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="stream-handler", daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = 2.0) -> None:
        self._stop_event.set()
        # Release first: OpenCV reads can block beyond the join timeout.
        self.source.close()
        if self._thread:
            self._thread.join(timeout_s)
        self.metrics.state = StreamHealth.CLOSED
        self.queue.close()

    def health(self) -> StreamHealth:
        """Return current health, deriving a stall from the last successful frame."""
        if self.metrics.state in (StreamHealth.CLOSED, StreamHealth.RECONNECTING, StreamHealth.CONNECTING):
            return self.metrics.state
        if self._stall_timeout_ns is not None:
            reference_ns = self.metrics.last_successful_frame_ns or self._opened_ns
            if reference_ns is not None and time.monotonic_ns() - reference_ns > self._stall_timeout_ns:
                self.metrics.state = StreamHealth.STALLED
                return StreamHealth.STALLED
        return self.metrics.state

    def _run(self) -> None:
        delay_s = self._reconnect_initial_s
        consecutive_failures = 0
        while not self._stop_event.is_set():
            try:
                self.open()
                while not self._stop_event.is_set():
                    self.pump_once()
                    consecutive_failures = 0
                    delay_s = self._reconnect_initial_s
            except SourceExhausted:
                self.close()
                return
            except (SourceReadError, OSError, RuntimeError) as error:
                if self._stop_event.is_set():
                    self.close()
                    return
                self.metrics.last_error = str(error)
                self.metrics.reconnect_count += 1
                consecutive_failures += 1
                self.metrics.state = StreamHealth.RECONNECTING
                self.source.close()
                if (
                    self._max_reconnect_attempts is not None
                    and consecutive_failures >= self._max_reconnect_attempts
                ):
                    self.metrics.state = StreamHealth.CLOSED
                    return
                self._stop_event.wait(delay_s)
                delay_s = min(delay_s * 2, self._reconnect_max_s)
        self.close()

    def _timestamp(self, frame: SourceFrame, received_ns: int) -> tuple[int, TimestampOrigin]:
        if frame.source_timestamp_ns is not None:
            return frame.source_timestamp_ns, TimestampOrigin.SOURCE_PRESENTATION
        return received_ns, TimestampOrigin.DECODE_MONOTONIC

    @staticmethod
    def _set_queue_quality(packet: FramePacket, depth: int, dropped_total: int) -> None:
        packet.quality.queue_depth = depth
        packet.quality.dropped_frame_count = dropped_total

    def _record_success(self, received_ns: int, decode_latency_ns: int) -> None:
        self.metrics.last_successful_frame_ns = received_ns
        if self._first_successful_frame_ns is None:
            self._first_successful_frame_ns = received_ns
        elif received_ns > self._first_successful_frame_ns:
            elapsed_s = (received_ns - self._first_successful_frame_ns) / 1_000_000_000
            self.metrics.capture_fps = (self.metrics.emitted_frames - 1) / elapsed_s
        if self.metrics.emitted_frames == 1:
            self.metrics.decode_latency_mean_ns = float(decode_latency_ns)
        else:
            count = self.metrics.emitted_frames
            self.metrics.decode_latency_mean_ns += (decode_latency_ns - self.metrics.decode_latency_mean_ns) / count
        self.metrics.decode_latency_max_ns = max(self.metrics.decode_latency_max_ns, decode_latency_ns)

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
