"""Tests for the flow-only candidate window and decision rules."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from applications.live_pipeline.window import Candidate, WindowSelector


@dataclass
class FakeFrame:
    frame_id: int
    rgb: np.ndarray
    source_id: str = "cam"
    capture_timestamp_ns: int = 0
    timestamp_origin: str = "source_presentation"


@dataclass
class FakeFlow:
    status: str
    flow: np.ndarray
    valid_mask: np.ndarray


class FixedEstimator:
    """Return a flow field with a fixed per-pixel magnitude."""

    def __init__(self, magnitude: float, valid_fraction: float = 1.0) -> None:
        self.magnitude = magnitude
        self.valid_fraction = valid_fraction

    def estimate(self, previous, current) -> FakeFlow:
        height, width = current.rgb.shape[:2]
        flow = np.zeros((height, width, 2), np.float32)
        flow[..., 0] = self.magnitude
        mask = np.zeros((height, width), bool)
        count = int(round(self.valid_fraction * height * width))
        mask.reshape(-1)[:count] = True
        return FakeFlow(status="ok", flow=flow, valid_mask=mask)


def frame(frame_id: int, timestamp: int) -> FakeFrame:
    return FakeFrame(frame_id=frame_id, rgb=np.zeros((10, 20, 3), np.uint8), capture_timestamp_ns=timestamp)


def selector(magnitude: float, threshold: float = 0.02, age_ns: int = 500_000_000) -> WindowSelector:
    return WindowSelector(FixedEstimator(magnitude), threshold=threshold, active_age_ns=age_ns)


def test_bootstrap_refresh_without_reference() -> None:
    window = selector(1.0)
    candidate = window.observe(frame(1, 0))
    decision = window.select([candidate])
    assert decision.action == "refresh"
    assert decision.reason == "bootstrap"
    assert window.reference is None


def test_score_is_normalized_by_image_diagonal() -> None:
    window = selector(2.0)
    window.commit(frame(1, 0))
    candidate = window.observe(frame(2, 10))
    assert candidate.score is not None
    diagonal = float(np.hypot(20, 10))
    assert candidate.score == np.float64(2.0) / diagonal


def test_highest_score_candidate_wins_the_window() -> None:
    window = selector(1.0)
    first = frame(1, 0)
    window.commit(first)
    low = Candidate(packet=frame(2, 10), flow=None, score=0.01, flow_valid=True, reason="scored")
    high = Candidate(packet=frame(3, 20), flow=None, score=0.9, flow_valid=True, reason="scored")
    decision = window.select([low, high])
    assert decision.action == "refresh"
    assert decision.reason == "significant_difference"
    assert decision.selected.frame_id == 3


def test_low_motion_reuses_when_state_is_fresh() -> None:
    window = selector(0.001, threshold=0.02)
    first = frame(1, 0)
    window.commit(first)
    window.active_state_capture_ns = 4_000_000_000
    candidate = window.observe(frame(2, 4_100_000_000))
    decision = window.select([candidate])
    assert decision.action == "reuse"
    assert decision.reason == "not_significant"


def test_forced_refresh_after_active_age() -> None:
    window = selector(0.001, threshold=0.02, age_ns=500_000_000)
    first = frame(1, 0)
    window.commit(first)
    window.active_state_capture_ns = 1_000_000_000
    candidate = window.observe(frame(2, 1_600_000_000))
    decision = window.select([candidate])
    assert decision.action == "refresh"
    assert decision.reason == "reference_expired"
    assert decision.forced is True


def test_invalid_flow_requests_refresh() -> None:
    window = selector(1.0, threshold=0.5)
    first = frame(1, 0)
    window.commit(first)
    candidate = Candidate(packet=frame(2, 10), flow=None, score=None, flow_valid=False, reason="flow_low_validity")
    decision = window.select([candidate])
    assert decision.action == "refresh"
    assert decision.reason == "invalid_flow"


def test_commit_advances_reference_and_active_state() -> None:
    window = selector(1.0)
    first = frame(1, 100)
    window.commit(first)
    assert window.reference is not None
    assert window.reference.frame_id == 1
    assert window.active_state_capture_ns == 100
    assert window.active_state_frame_id == 1
    # The retained copy must not alias the caller's buffer.
    first.rgb[:] = 255
    assert window.reference.rgb.max() == 0


def test_generation_change_resets_reference_and_state() -> None:
    window = selector(1.0)
    first = frame(1, 100)
    window.commit(first)
    assert window.reference is not None
    window.set_generation(1)
    assert window.reference is None
    assert window.active_state_capture_ns is None
    assert window.active_state_frame_id is None
    assert window.generation == 1
    # The next candidate is a bootstrap frame, never a reuse of the old run.
    candidate = window.observe(frame(2, 200))
    assert candidate.reason == "bootstrap"
    assert candidate.generation == 1
    decision = window.select([candidate])
    assert decision.action == "refresh"
    assert decision.reason == "bootstrap"


def test_flow_is_not_compared_across_generations() -> None:
    window = selector(1.0)
    window.commit(frame(1, 100))
    # Simulate a stale reference from an earlier run that was not reset.
    window.reference = window.reference.__class__(
        frame_id=1,
        source_id="cam",
        capture_timestamp_ns=100,
        timestamp_origin="source_presentation",
        rgb=np.zeros((10, 20, 3), np.uint8),
        generation=7,
    )
    candidate = window.observe(frame(2, 200))
    assert candidate.reason == "reference_mismatch"


def test_commit_stamps_the_current_generation() -> None:
    window = selector(1.0)
    window.set_generation(3)
    window.commit(frame(1, 100))
    assert window.reference is not None
    assert window.reference.generation == 3


def test_commit_rejects_a_stale_generation() -> None:
    # RT-0024 item 1: a refresh started in generation 0 must not commit after a
    # reset to generation 1.
    window = selector(1.0)
    window.set_generation(0)
    window.commit(frame(5, 100), generation=0)
    old_reference = window.reference
    window.set_generation(1)
    assert window.commit(frame(9, 200), generation=0) is False
    assert window.reference is None
    assert old_reference is not None
    # A current-generation commit is accepted.
    assert window.commit(frame(10, 300), generation=1) is True
    assert window.reference is not None
    assert window.reference.frame_id == 10
    assert window.reference.generation == 1


def test_candidate_current_generation_filter() -> None:
    window = selector(1.0)
    window.set_generation(2)
    current = window.observe(frame(1, 100))
    assert window.candidate_is_current(current)
    window.set_generation(3)
    assert not window.candidate_is_current(current)
