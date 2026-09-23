from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

PACKAGE_ROOT = Path(__file__).parents[1] / "src"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from stream_handler import IterableFrameSource, SourceExhausted, SourceFrame, StreamHandler


class StreamHandlerTests(unittest.TestCase):
    def _handler(self, frames, capacity: int = 3) -> StreamHandler:
        handler = StreamHandler(IterableFrameSource(frames, "fixture-camera"), queue_capacity=capacity)
        handler.open()
        return handler

    def test_replay_preserves_ids_timestamps_and_converts_bgr_to_rgb(self) -> None:
        bgr = np.array([[[3, 2, 1]]], dtype=np.uint8)
        handler = self._handler([
            SourceFrame(bgr, "BGR8", 100),
            SourceFrame(bgr, "BGR8", 200),
        ])
        first = handler.pump_once()
        second = handler.pump_once()
        self.assertEqual([first.frame_id, second.frame_id], [0, 1])
        self.assertEqual([first.capture_timestamp_ns, second.capture_timestamp_ns], [100, 200])
        self.assertEqual(first.rgb[0, 0].tolist(), [1, 2, 3])
        self.assertFalse(first.quality.frame_age_available)
        self.assertEqual(first.quality.frame_age_ns, 0)
        self.assertEqual(first.metadata()["image"]["pixel_format"], "RGB8")

    def test_queue_drops_oldest_when_consumer_is_slow(self) -> None:
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        handler = self._handler([SourceFrame(image, "RGB8", timestamp) for timestamp in (1, 2, 3)], capacity=2)
        for _ in range(3):
            handler.pump_once()
        first_available = handler.queue.get(timeout_s=0.01)
        second_available = handler.queue.get(timeout_s=0.01)
        self.assertEqual([first_available.frame_id, second_available.frame_id], [1, 2])
        self.assertEqual(handler.queue.dropped_total, 1)
        self.assertEqual(second_available.quality.dropped_frame_count, 1)

    def test_timestamp_faults_are_visible_without_breaking_frame_ids(self) -> None:
        image = np.zeros((1, 1, 3), dtype=np.uint8)
        handler = self._handler([
            SourceFrame(image, "RGB8", 10),
            SourceFrame(image, "RGB8", 10),
            SourceFrame(image, "RGB8", 9),
        ])
        packets = [handler.pump_once() for _ in range(3)]
        self.assertEqual([packet.frame_id for packet in packets], [0, 1, 2])
        self.assertEqual([packet.quality.timestamp_status for packet in packets], ["ok", "duplicate", "backwards"])
        self.assertEqual(handler.metrics.timestamp_duplicates, 1)
        self.assertEqual(handler.metrics.timestamp_backwards, 1)

    def test_invalid_images_are_rejected(self) -> None:
        handler = self._handler([SourceFrame(np.zeros((2, 2), dtype=np.uint8), "RGB8", 1)])
        with self.assertRaises(ValueError):
            handler.pump_once()

    def test_replay_signals_normal_end_of_stream(self) -> None:
        handler = self._handler([])
        with self.assertRaises(SourceExhausted):
            handler.pump_once()


if __name__ == "__main__":
    unittest.main()
