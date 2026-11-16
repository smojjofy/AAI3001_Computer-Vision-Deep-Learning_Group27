"""Tests for the capture-run generation guard and queue draining."""

from __future__ import annotations

import threading
import time

import numpy as np

from applications.live_pipeline.runstate import (
    PacketRunTracker,
    RunCounters,
    RunGuard,
    drain_queue,
)
from stream_handler.contracts import FramePacket, FrameQuality, TimestampOrigin
from stream_handler.queue import LatestFrameQueue


def packet(frame_id: int, source_id: str = "cam", size: tuple[int, int] = (4, 4)) -> FramePacket:
    height, width = size
    return FramePacket(
        frame_id=frame_id,
        source_id=source_id,
        capture_timestamp_ns=frame_id + 1,
        timestamp_origin=TimestampOrigin.DECODE_MONOTONIC,
        rgb=np.zeros((height, width, 3), np.uint8),
        quality=FrameQuality(0, 0, True),
    )


def test_publish_rejected_from_an_old_generation() -> None:
    guard = RunGuard(0)
    calls: list[int] = []
    assert guard.publish_if_current(0, lambda: calls.append(guard.generation)) is True
    guard.advance()
    assert guard.publish_if_current(0, lambda: calls.append(-1)) is False
    assert calls == [0]


def test_failure_record_rejected_from_an_old_generation() -> None:
    guard = RunGuard(0)
    events: list[str] = []
    assert guard.record_failure_if_current(0, lambda: events.append("old")) is True
    guard.advance()
    assert guard.record_failure_if_current(0, lambda: events.append("stale")) is False
    assert events == ["old"]


def test_generation_cannot_change_during_publish() -> None:
    guard = RunGuard(0)
    started = threading.Event()
    release = threading.Event()
    observed: list[int] = []
    advanced: list[int] = []

    def publish() -> None:
        observed.append(guard.generation)
        started.set()
        release.wait(2)
        observed.append(guard.generation)

    def advance() -> None:
        started.wait(2)
        advanced.append(guard.advance())

    publisher = threading.Thread(target=lambda: guard.publish_if_current(0, publish))
    advancer = threading.Thread(target=advance)
    publisher.start()
    advancer.start()
    assert started.wait(2)
    # The advance must wait for the publish to finish.
    time.sleep(0.05)
    assert advanced == []
    release.set()
    publisher.join(2)
    advancer.join(2)
    assert observed == [0, 0]
    assert advanced == [1]
    assert guard.generation == 1


def test_only_one_of_many_racing_publishes_wins_after_advance() -> None:
    guard = RunGuard(0)
    accepted: list[int] = []

    def worker(job_generation: int) -> None:
        def publish() -> None:
            time.sleep(0.005)
            accepted.append(job_generation)

        guard.publish_if_current(job_generation, publish)

    guard.advance()  # generation is now 1; only generation 1 jobs may publish
    threads = [threading.Thread(target=worker, args=(index % 3,)) for index in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(2)
    assert accepted and all(value == 1 for value in accepted)


def test_drain_queue_removes_old_packets() -> None:
    queue = LatestFrameQueue(capacity=3)
    for frame_id in range(3):
        queue.put(packet(frame_id))
    assert queue.depth == 3
    assert drain_queue(queue) == 1  # get_latest removes the newest and clears the rest
    assert queue.depth == 0
    assert queue.get_latest(timeout_s=0) is None


def test_drain_queue_is_safe_when_empty() -> None:
    queue = LatestFrameQueue(capacity=2)
    assert drain_queue(queue) == 0


def test_packet_tracker_detects_one_change_then_trusts_the_new_baseline() -> None:
    # RT-0028 item 2: after a permanent source change, the next packet must
    # become the baseline instead of causing a run change on every packet.
    tracker = PacketRunTracker()
    first = packet(1)
    tracker.accept(first)
    assert tracker.change_reason(packet(2)) is None

    changed = packet(3, source_id="other-camera")
    assert tracker.change_reason(changed) == "source_or_size"
    tracker.reset()
    tracker.accept(changed)
    assert tracker.change_reason(packet(4, source_id="other-camera")) is None
    assert tracker.change_reason(packet(5, source_id="other-camera")) is None


def test_packet_tracker_detects_resolution_and_backward_time() -> None:
    tracker = PacketRunTracker()
    tracker.accept(packet(10))
    resized = packet(11, size=(8, 8))
    assert tracker.change_reason(resized) == "source_or_size"

    tracker.reset()
    tracker.accept(packet(20))
    backwards = packet(21)
    backwards.capture_timestamp_ns = 1
    assert tracker.change_reason(backwards) == "backward_time"


def test_run_counters_ignore_stale_results() -> None:
    counters = RunCounters(0)
    assert counters.record_published(0) is True
    assert counters.record_failure(0) is True
    assert counters.snapshot()["refresh_count"] == 1
    # A run change resets the current-run counts.
    counters.start_run(1)
    assert counters.snapshot() == {"refresh_count": 0, "refresh_failures": 0, "discarded_stale": 0}
    # An old success or failure must not change the new run's counts.
    assert counters.record_published(0) is False
    assert counters.record_failure(0) is False
    assert counters.snapshot()["refresh_count"] == 0
    assert counters.snapshot()["refresh_failures"] == 0
    assert counters.snapshot()["discarded_stale"] == 2
    assert counters.record_published(1) is True
    assert counters.snapshot()["refresh_count"] == 1
