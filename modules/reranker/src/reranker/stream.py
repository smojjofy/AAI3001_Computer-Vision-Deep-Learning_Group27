"""Direct Stream Handler -> Optical Flow -> Reranker connection."""

from __future__ import annotations

from dataclasses import dataclass
import time

from .contracts import FlowEstimator, FlowResult, HasFrameQueue, RGBFrame
from .reranker import Reranker, SelectionDecision


@dataclass(frozen=True)
class RerankerObservation:
    """One packet, its reference-to-current flow, and the resulting decision."""

    packet: RGBFrame
    flow: FlowResult | None
    decision: SelectionDecision
    processing_time_ns: int


class StreamRerankerConsumer:
    """Drain one Stream Handler queue and link optical flow directly to reranking.

    The reranker's retained significant frame is supplied as the previous flow
    frame.  This deliberately differs from the adjacent-frame flow tracker:
    significance is measured against the retained reference, not every frame.
    """

    def __init__(self, handler: HasFrameQueue, estimator: FlowEstimator, reranker: Reranker | None = None) -> None:
        self.handler = handler
        self.estimator = estimator
        self.reranker = reranker or Reranker()

    def poll(self, timeout_s: float | None = None) -> RerankerObservation | None:
        packet = self.handler.queue.get(timeout_s=timeout_s)
        if packet is None:
            return None
        started_ns = time.perf_counter_ns()
        reference = self.reranker.reference
        flow = None
        if (
            reference is not None
            and reference.source_id == packet.source_id
            and reference.rgb.shape == packet.rgb.shape
        ):
            flow = self.estimator.estimate(reference, packet)
        decision = self.reranker.decide(packet, flow)
        return RerankerObservation(packet, flow, decision, time.perf_counter_ns() - started_ns)

    def reset(self) -> None:
        self.reranker.reset()
