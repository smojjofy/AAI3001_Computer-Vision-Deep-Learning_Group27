"""Run with: python -m unittest discover -s modules/inference/optical_flow/tests."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
import unittest

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "modules/inference/optical_flow/src"))
sys.path.insert(0, str(ROOT / "modules/stream_handler/src"))

from optical_flow import DISFlowEstimator, FlowConfig, OpticalFlowTracker, StreamFlowConsumer, create_estimator  # noqa: E402
from stream_handler import (  # noqa: E402
    FramePacket, FrameQuality, IterableFrameSource, SourceFrame, StreamHandler,
    TimestampOrigin,
)


def packet(rgb: np.ndarray, frame_id: int, timestamp_ns: int, source_id: str = "camera") -> FramePacket:
    return FramePacket(
        frame_id=frame_id,
        source_id=source_id,
        capture_timestamp_ns=timestamp_ns,
        timestamp_origin=TimestampOrigin.DECODE_MONOTONIC,
        rgb=rgb,
        quality=FrameQuality(decode_latency_ns=0, frame_age_ns=0, frame_age_available=True),
    )


class DISFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.estimator = DISFlowEstimator(FlowConfig(flow_width=96, flow_height=64))
        rng = np.random.default_rng(27)
        noise = rng.integers(0, 256, (128, 192), dtype=np.uint8)
        self.image = cv2.cvtColor(cv2.GaussianBlur(noise, (5, 5), 0), cv2.COLOR_GRAY2RGB)

    def test_forward_direction_pixels_and_skipped_ids(self) -> None:
        translated = cv2.warpAffine(self.image, np.float32([[1, 0, 6], [0, 1, 3]]), (192, 128))
        result = self.estimator.estimate(packet(self.image, 4, 1_000), packet(translated, 9, 51_000))
        center = np.s_[24:104, 24:168]
        vectors = result.flow[center][result.valid_mask[center]]
        self.assertEqual(result.status, "ok")
        self.assertGreater(len(vectors), 1000)
        self.assertAlmostEqual(float(np.median(vectors[:, 0])), 6.0, delta=1.5)
        self.assertAlmostEqual(float(np.median(vectors[:, 1])), 3.0, delta=1.5)
        self.assertEqual(result.elapsed_ns, 50_000)
        self.assertEqual(result.flow.shape, (128, 192, 2))
        self.assertEqual(result.valid_mask.shape, (128, 192))
        self.assertFalse(result.valid_mask[:, -1].all())
        self.assertEqual(result.metadata()["direction"], "previous_to_current")
        self.assertEqual(result.metadata()["flow_width"], 96)

    def test_first_frame_and_source_switch(self) -> None:
        tracker = OpticalFlowTracker(self.estimator)
        first = tracker.push(packet(self.image, 1, 1_000))
        self.assertEqual(first.status, "no_previous_frame")
        self.assertFalse(first.valid_mask.any())
        switched = tracker.push(packet(self.image, 1, 2_000, "another_camera"))
        self.assertEqual(switched.status, "no_previous_frame")

    def test_bad_timestamp_and_scene_cut(self) -> None:
        previous = packet(self.image, 1, 10_000)
        duplicate = self.estimator.estimate(previous, packet(self.image, 2, 10_000))
        self.assertEqual(duplicate.status, "invalid_timestamps")
        self.assertFalse(duplicate.valid_mask.any())
        black = packet(np.zeros_like(self.image), 2, 20_000)
        white = packet(np.full_like(self.image, 255), 3, 30_000)
        cut = self.estimator.estimate(black, white)
        self.assertEqual(cut.status, "scene_cut")
        self.assertFalse(cut.valid_mask.any())

    def test_reset_and_invalid_input(self) -> None:
        tracker = OpticalFlowTracker(self.estimator)
        tracker.push(packet(self.image, 1, 1_000))
        tracker.reset()
        self.assertEqual(tracker.push(packet(self.image, 2, 2_000)).status, "no_previous_frame")
        with self.assertRaises(ValueError):
            self.estimator.estimate(None, packet(self.image.astype(np.float32), 3, 3_000))

    def test_config_rejects_unknown_model_and_invalid_size(self) -> None:
        with self.assertRaises(ValueError):
            FlowConfig(flow_model="unknown")
        with self.assertRaises(ValueError):
            FlowConfig(flow_width=0)
        self.assertIsInstance(create_estimator(FlowConfig()), DISFlowEstimator)

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("torchvision"), "RAFT dependencies unavailable")
    def test_raft_backend_contract_without_weight_download(self) -> None:
        estimator = create_estimator(FlowConfig(
            flow_model="raft_small", flow_width=128, flow_height=128,
            raft_weights="none", raft_device="cpu", raft_iterations=1,
        ))
        result = estimator.estimate(packet(self.image, 1, 1_000), packet(self.image, 2, 2_000))
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.backend, "raft_small")
        self.assertEqual(result.flow.shape, (128, 192, 2))
        self.assertEqual(result.valid_mask.shape, (128, 192))


class StreamConnectionTests(unittest.TestCase):
    def test_queue_pairs_last_dequeued_packets_and_preserves_drops(self) -> None:
        rng = np.random.default_rng(27)
        image = rng.integers(0, 256, (64, 96, 3), dtype=np.uint8)
        frames = [SourceFrame(image, "RGB8", timestamp) for timestamp in (1_000, 2_000, 3_000)]
        handler = StreamHandler(IterableFrameSource(frames), queue_capacity=1)
        consumer = StreamFlowConsumer(
            handler, OpticalFlowTracker(DISFlowEstimator(FlowConfig(flow_width=96, flow_height=64)))
        )
        handler.open()
        try:
            self.assertIsNone(consumer.poll(timeout_s=0))
            first_packet = handler.pump_once()
            first = consumer.poll(timeout_s=0)
            self.assertIsNotNone(first)
            assert first is not None
            self.assertIs(first.packet, first_packet)
            self.assertEqual(first.flow.status, "no_previous_frame")

            handler.pump_once()  # Frame 1 will be dropped before flow sees it.
            last_packet = handler.pump_once()
            last = consumer.poll(timeout_s=0)
            self.assertIsNotNone(last)
            assert last is not None
            self.assertIs(last.packet, last_packet)
            self.assertEqual(last.flow.status, "ok")
            self.assertEqual((last.flow.previous_frame_id, last.flow.current_frame_id), (0, 2))
            self.assertEqual(last.flow.elapsed_ns, 2_000)
            self.assertEqual(last.packet.quality.dropped_frame_count, 1)
            self.assertGreaterEqual(last.processing_time_ns, 0)
            self.assertIsNone(consumer.poll(timeout_s=0))
        finally:
            handler.close()


if __name__ == "__main__":
    unittest.main()
