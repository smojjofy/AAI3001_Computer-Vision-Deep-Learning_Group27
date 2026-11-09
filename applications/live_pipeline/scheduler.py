"""Sample fresh camera packets and coalesce slow refresh work."""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Callable

from reranker import RerankerObservation, StreamRerankerConsumer
from reranker.contracts import FlowResult, RGBFrame
from reranker.reranker import SelectionDecision


@dataclass(frozen=True)
class RefreshJob:
    packet: RGBFrame
    flow: FlowResult | None
    decision: SelectionDecision


class RefreshDispatcher:
    """Run one refresh callback; retain at most one newer pending request.

    A pending request is replaced when a fresher refresh arrives. This class
    bounds work submission, but does not supply a Temporal Cache or validate
    reuse against completed reconstruction state.
    """

    def __init__(self, worker: Callable[[RefreshJob], None]) -> None:
        self._worker = worker
        self._condition = threading.Condition()
        self._pending: RefreshJob | None = None
        self._active = False
        self._closing = False
        self._drain = False
        self.submitted_total = 0
        self.started_total = 0
        self.completed_total = 0
        self.replaced_total = 0
        self.discarded_total = 0
        self.last_error: str | None = None
        self._thread = threading.Thread(target=self._run, name="refresh-worker", daemon=True)
        self._thread.start()

    def submit(self, job: RefreshJob) -> str:
        with self._condition:
            if self._closing:
                raise RuntimeError("refresh dispatcher is closed")
            outcome = "replaced" if self._pending is not None else "queued"
            if self._pending is not None:
                self.replaced_total += 1
            self._pending = job
            self.submitted_total += 1
            self._condition.notify()
            return outcome

    def snapshot(self) -> dict[str, object]:
        with self._condition:
            return {
                "submitted_total": self.submitted_total,
                "started_total": self.started_total,
                "completed_total": self.completed_total,
                "replaced_total": self.replaced_total,
                "discarded_total": self.discarded_total,
                "active": self._active,
                "pending_frame_id": None if self._pending is None else self._pending.packet.frame_id,
                "last_error": self.last_error,
            }

    def close(self, *, drain: bool = True, timeout_s: float = 10.0) -> None:
        with self._condition:
            self._closing = True
            self._drain = drain
            if not drain and self._pending is not None:
                self._pending = None
                self.discarded_total += 1
            self._condition.notify_all()
        self._thread.join(timeout_s)
        if self._thread.is_alive():
            raise TimeoutError("refresh worker did not stop within timeout")

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._closing:
                    self._condition.wait()
                if self._closing and (self._pending is None or not self._drain):
                    return
                job = self._pending
                assert job is not None
                self._pending = None
                self._active = True
                self.started_total += 1
            try:
                self._worker(job)
            except Exception as error:
                with self._condition:
                    self.last_error = repr(error)
            finally:
                with self._condition:
                    self._active = False
                    self.completed_total += 1
                    self._condition.notify_all()


class LiveRerankerScheduler:
    """Process the freshest packet no more often than one capture-time interval."""

    def __init__(
        self,
        consumer: StreamRerankerConsumer,
        dispatcher: RefreshDispatcher,
        sample_interval_ns: int = 250_000_000,
    ) -> None:
        if type(sample_interval_ns) is not int or sample_interval_ns < 1:
            raise ValueError("sample_interval_ns must be a positive integer")
        self.consumer = consumer
        self.dispatcher = dispatcher
        self.sample_interval_ns = sample_interval_ns
        self._last_timestamp_ns: int | None = None
        self._last_timestamp_origin: str | None = None
        self._last_source_id: str | None = None
        self.skipped_by_interval = 0
        self.processed_total = 0

    def poll(self, timeout_s: float | None = None) -> RerankerObservation | None:
        packet = self.consumer.handler.queue.get_latest(timeout_s=timeout_s)
        if packet is None:
            return None
        if (
            self._last_timestamp_ns is not None
            and packet.source_id == self._last_source_id
            and str(packet.timestamp_origin) == self._last_timestamp_origin
            and packet.capture_timestamp_ns - self._last_timestamp_ns < self.sample_interval_ns
        ):
            self.skipped_by_interval += 1
            return None
        observation = self.consumer.process(packet)
        self._last_timestamp_ns = packet.capture_timestamp_ns
        self._last_timestamp_origin = str(packet.timestamp_origin)
        self._last_source_id = packet.source_id
        self.processed_total += 1
        if observation.decision.action == "refresh":
            self.dispatcher.submit(RefreshJob(packet, observation.flow, observation.decision))
        return observation
