"""Tests for reference commit-on-success behavior.

The retained reference must change only when the caller commits a successful
refresh. A failed construction must not become the reference for later flow
comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from modules.reranker.src.reranker.reranker import Reranker, RerankerConfig


@dataclass
class FakeFrame:
    frame_id: int
    rgb: np.ndarray
    source_id: str = "cam"
    capture_timestamp_ns: int = 0
    timestamp_origin: str = "source_presentation"


@dataclass
class FakeFlow:
    """The subset of FlowResult that the Reranker reads."""

    status: str
    source_id: str
    previous_frame_id: int | None
    current_frame_id: int
    previous_capture_timestamp_ns: int | None
    current_capture_timestamp_ns: int
    flow: np.ndarray
    valid_mask: np.ndarray

    @property
    def valid_fraction(self) -> float:
        return float(self.valid_mask.mean())


def flow_ok(previous: FakeFrame, current: FakeFrame, magnitude: float) -> FakeFlow:
    height, width = current.rgb.shape[:2]
    return FakeFlow(
        status="ok",
        source_id=current.source_id,
        previous_frame_id=previous.frame_id,
        current_frame_id=current.frame_id,
        previous_capture_timestamp_ns=previous.capture_timestamp_ns,
        current_capture_timestamp_ns=current.capture_timestamp_ns,
        flow=np.full((height, width, 2), magnitude, np.float32),
        valid_mask=np.ones((height, width), bool),
    )


def make_frame(frame_id: int, timestamp: int = 0) -> FakeFrame:
    return FakeFrame(frame_id=frame_id, rgb=np.zeros((8, 8, 3), np.uint8), capture_timestamp_ns=timestamp)


def test_bootstrap_refresh_does_not_commit_reference() -> None:
    reranker = Reranker(RerankerConfig(max_reference_age_ns=500_000_000))
    first = make_frame(1)
    decision = reranker.decide(first, None)
    assert decision.action == "refresh"
    assert decision.reason == "bootstrap"
    assert reranker.reference is None


def test_commit_reference_enables_flow_comparison() -> None:
    reranker = Reranker(RerankerConfig(max_reference_age_ns=500_000_000))
    first = make_frame(1)
    reranker.decide(first, None)
    reranker.commit_reference(first)
    assert reranker.reference is not None
    second = make_frame(2, 10)
    decision = reranker.decide(second, flow_ok(first, second, 0.5))
    assert decision.action == "reuse"
    assert reranker.reference.frame_id == 1


def test_refresh_decision_leaves_reference_until_commit() -> None:
    reranker = Reranker(RerankerConfig(max_reference_age_ns=500_000_000))
    first = make_frame(1)
    reranker.decide(first, None)
    reranker.commit_reference(first)
    second = make_frame(2, 10)
    decision = reranker.decide(second, flow_ok(first, second, 5.0))
    assert decision.action == "refresh"
    assert reranker.reference.frame_id == 1
    reranker.commit_reference(second)
    assert reranker.reference.frame_id == 2


def test_expired_reference_still_requests_refresh() -> None:
    reranker = Reranker(RerankerConfig(max_reference_age_ns=500_000_000))
    first = make_frame(1, 0)
    reranker.decide(first, None)
    reranker.commit_reference(first)
    current = make_frame(2, 600_000_000)
    decision = reranker.decide(current, None)
    assert decision.action == "refresh"
    assert decision.reason == "reference_expired"
