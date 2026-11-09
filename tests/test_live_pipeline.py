"""Focused checks for latest-frame scheduling and refresh backpressure."""

from __future__ import annotations

from pathlib import Path
import sys
import threading
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "modules/stream_handler/src"),
    str(ROOT / "modules/inference/optical_flow/src"),
    str(ROOT / "modules/reranker/src"),
    str(ROOT / "applications"),
]

from live_pipeline import LiveRerankerScheduler, RefreshDispatcher, RefreshJob
from optical_flow import FlowResult
from reranker import Reranker, RerankerConfig, StreamRerankerConsumer
from stream_handler import FramePacket, FrameQuality, TimestampOrigin
from stream_handler.queue import LatestFrameQueue


def packet(frame_id: int, timestamp_ns: int) -> FramePacket:
    return FramePacket(
        frame_id=frame_id,
        source_id="fixture",
        capture_timestamp_ns=timestamp_ns,
        timestamp_origin=TimestampOrigin.DECODE_MONOTONIC,
        rgb=np.zeros((32, 32, 3), dtype=np.uint8),
        quality=FrameQuality(0, 0, True),
    )


class ZeroFlowEstimator:
    def estimate(self, previous, current) -> FlowResult:
        return FlowResult(
            status="ok",
            source_id=current.source_id,
            previous_frame_id=previous.frame_id,
            current_frame_id=current.frame_id,
            previous_capture_timestamp_ns=previous.capture_timestamp_ns,
            current_capture_timestamp_ns=current.capture_timestamp_ns,
            timestamp_origin=str(current.timestamp_origin),
            elapsed_ns=current.capture_timestamp_ns - previous.capture_timestamp_ns,
            flow=np.zeros((32, 32, 2), dtype=np.float32),
            valid_mask=np.ones((32, 32), dtype=np.bool_),
            flow_width=32,
            flow_height=32,
        )


class LivePipelineTests(unittest.TestCase):
    def test_latest_queue_discards_stale_packets(self) -> None:
        queue = LatestFrameQueue(capacity=3)
        for frame_id in range(3):
            queue.put(packet(frame_id, frame_id))
        latest = queue.get_latest(timeout_s=0)
        self.assertIsNotNone(latest)
        self.assertEqual(latest.frame_id, 2)
        self.assertEqual(queue.skipped_total, 2)
        self.assertEqual(queue.depth, 0)

    def test_scheduler_samples_by_capture_time_and_uses_reranker(self) -> None:
        class Handler:
            queue = LatestFrameQueue(capacity=3)

        handler = Handler()
        consumer = StreamRerankerConsumer(handler, ZeroFlowEstimator(), Reranker(RerankerConfig(max_reference_age_ns=500_000_000)))
        dispatcher = RefreshDispatcher(lambda _job: None)
        scheduler = LiveRerankerScheduler(consumer, dispatcher, sample_interval_ns=250_000_000)
        try:
            handler.queue.put(packet(0, 1_000_000_000))
            first = scheduler.poll(timeout_s=0)
            handler.queue.put(packet(1, 1_100_000_000))
            skipped = scheduler.poll(timeout_s=0)
            handler.queue.put(packet(2, 1_250_000_000))
            second = scheduler.poll(timeout_s=0)
            self.assertEqual(first.decision.reason, "bootstrap")
            self.assertIsNone(skipped)
            self.assertEqual(second.decision.reason, "not_significant")
            self.assertEqual(second.flow.previous_frame_id, 0)
            self.assertEqual(scheduler.skipped_by_interval, 1)
        finally:
            dispatcher.close()

    def test_refresh_dispatcher_replaces_one_pending_job(self) -> None:
        running = threading.Event()
        release = threading.Event()
        completed: list[int] = []

        def worker(job: RefreshJob) -> None:
            if job.packet.frame_id == 0:
                running.set()
                self.assertTrue(release.wait(2))
            completed.append(job.packet.frame_id)

        dispatcher = RefreshDispatcher(worker)
        try:
            for frame_id in (0, 1, 2):
                if frame_id == 1:
                    self.assertTrue(running.wait(2))
                fake_decision = type("Decision", (), {"selected_frame_id": frame_id})()
                dispatcher.submit(RefreshJob(packet(frame_id, frame_id), None, fake_decision))
        finally:
            release.set()
            dispatcher.close()
        self.assertEqual(completed, [0, 2])
        self.assertEqual(dispatcher.snapshot()["replaced_total"], 1)


if __name__ == "__main__":
    unittest.main()
