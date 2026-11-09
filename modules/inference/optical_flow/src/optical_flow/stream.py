"""Stream Handler queue to optical-flow handoff."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Protocol

from .flow import FlowResult, OpticalFlowTracker, RGBFrame


class FrameQueue(Protocol):
    def get(self, timeout_s: float | None = None) -> RGBFrame | None: ...


class HasFrameQueue(Protocol):
    queue: FrameQueue


@dataclass(frozen=True)
class FlowObservation:
    """Original packet and matching flow result for the future reranker."""

    packet: RGBFrame
    flow: FlowResult
    processing_time_ns: int


class StreamFlowConsumer:
    """Consume one Stream Handler queue and compare frames actually received.

    The Stream Handler owns capture, queue bounds, and frame drops. This adapter
    keeps exactly one previous dequeued packet through OpticalFlowTracker. It
    neither changes capture cadence nor assumes frame IDs are consecutive.
    Only one consumer should drain a given Stream Handler queue.
    """

    def __init__(self, handler: HasFrameQueue, tracker: OpticalFlowTracker | None = None) -> None:
        self.handler = handler
        self.tracker = tracker or OpticalFlowTracker()

    def poll(self, timeout_s: float | None = None) -> FlowObservation | None:
        packet = self.handler.queue.get(timeout_s=timeout_s)
        if packet is None:
            return None
        started_ns = time.perf_counter_ns()
        flow = self.tracker.push(packet)
        return FlowObservation(packet, flow, time.perf_counter_ns() - started_ns)

    def reset(self) -> None:
        """Forget the previous pair when restarting or changing capture runs."""
        self.tracker.reset()
