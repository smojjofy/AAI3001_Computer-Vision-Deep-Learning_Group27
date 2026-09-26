from __future__ import annotations

import sys
import tempfile
import threading
import time
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

PACKAGE_ROOT = Path(__file__).parents[1] / "src"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from stream_handler import (
    IterableFrameSource,
    SourceExhausted,
    SourceFrame,
    StreamHandler,
    StreamHealth,
)
from stream_handler.sources import RecordedVideoSource, ReplayMode, SourceReadError
import stream_handler.sources as sources_module
from stream_handler.cli import _parser

try:
    import cv2
except ImportError:
    cv2 = None


class ReconnectingFixtureSource:
    """Fails two opens, emits one frame, then ends normally."""

    source_id = "reconnecting-fixture"

    def __init__(self) -> None:
        self.open_attempts = 0
        self.opened = False
        self.emitted = False

    def open(self) -> None:
        self.open_attempts += 1
        if self.open_attempts < 3:
            raise SourceReadError("fixture open failure")
        self.opened = True

    def read(self) -> SourceFrame:
        if self.emitted:
            raise SourceExhausted()
        self.emitted = True
        return SourceFrame(np.zeros((1, 1, 3), dtype=np.uint8), "RGB8")

    def close(self) -> None:
        self.opened = False

    def health(self) -> StreamHealth:
        return StreamHealth.HEALTHY if self.opened else StreamHealth.CLOSED


class BlockingFixtureSource:
    """A read that only unblocks when close is called."""

    source_id = "blocking-fixture"

    def __init__(self) -> None:
        self.read_started = threading.Event()
        self.release_read = threading.Event()
        self.opened = False
        self.close_calls = 0

    def open(self) -> None:
        self.opened = True

    def read(self) -> SourceFrame:
        self.read_started.set()
        self.release_read.wait(1.0)
        raise SourceReadError("read interrupted")

    def close(self) -> None:
        self.close_calls += 1
        self.opened = False
        self.release_read.set()

    def health(self) -> StreamHealth:
        return StreamHealth.HEALTHY if self.opened else StreamHealth.CLOSED


class StalledFixtureSource:
    source_id = "stalled-fixture"

    def open(self) -> None:
        pass

    def read(self) -> SourceFrame:
        return SourceFrame(np.zeros((1, 1, 3), dtype=np.uint8), "RGB8")

    def close(self) -> None:
        pass

    def health(self) -> StreamHealth:
        return StreamHealth.STALLED


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

    def test_queue_metadata_is_ready_when_pump_once_returns(self) -> None:
        image = np.zeros((1, 1, 3), dtype=np.uint8)
        handler = self._handler([SourceFrame(image, "RGB8", 1)], capacity=1)
        packet = handler.pump_once()
        self.assertEqual(packet.quality.queue_depth, 1)
        self.assertEqual(packet.quality.dropped_frame_count, 0)
        self.assertIs(handler.queue.get(timeout_s=0.01), packet)

    def test_stall_health_is_derived_from_last_successful_frame(self) -> None:
        source = BlockingFixtureSource()
        handler = StreamHandler(source, stall_timeout_s=0.01)
        handler.start()
        self.assertTrue(source.read_started.wait(0.2))
        time.sleep(0.03)
        self.assertEqual(handler.health(), StreamHealth.STALLED)
        handler.stop(timeout_s=0.2)
        self.assertGreaterEqual(source.close_calls, 1)
        self.assertFalse(handler._thread.is_alive())

    def test_pump_once_uses_source_reported_health(self) -> None:
        handler = StreamHandler(StalledFixtureSource())
        handler.open()
        handler.pump_once()
        self.assertEqual(handler.health(), StreamHealth.STALLED)

    def test_background_handler_reconnects_after_open_failures(self) -> None:
        source = ReconnectingFixtureSource()
        handler = StreamHandler(source, reconnect_initial_s=0.001, reconnect_max_s=0.002)
        handler.start()
        handler._thread.join(0.5)
        self.assertFalse(handler._thread.is_alive())
        self.assertEqual(source.open_attempts, 3)
        self.assertEqual(handler.metrics.reconnect_count, 2)
        self.assertEqual(handler.metrics.emitted_frames, 1)
        self.assertEqual(handler.health(), StreamHealth.CLOSED)

    def test_recorded_replay_waits_for_presentation_timestamp(self) -> None:
        source = sources_module.RecordedVideoSource("fixture.mp4")
        source._closed_event = MagicMock()
        source._closed_event.is_set.return_value = False
        with patch.object(sources_module.time, "monotonic_ns", side_effect=(100, 200)):
            source._pace_replay(1_000)
            source._pace_replay(1_000_001_000)
        source._closed_event.wait.assert_called_once_with(0.9999999)

    def test_cli_accepts_recorded_replay_mode(self) -> None:
        args = _parser().parse_args([
            "--source", "video", "--path", "fixture.mp4", "--replay-mode", "fastest",
        ])
        self.assertEqual(args.source, "video")
        self.assertEqual(args.replay_mode, "fastest")

    @unittest.skipUnless(cv2 is not None, "OpenCV is not installed")
    def test_opencv_recorded_video_decodes_rgb_and_orders_pts(self) -> None:
        """Exercise the real OpenCV file adapter with a generated, licence-clear fixture."""
        colours_bgr = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "primary-colours.avi"
            writer = cv2.VideoWriter(
                str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (16, 16)
            )
            self.assertTrue(writer.isOpened(), "OpenCV could not create the test video")
            try:
                for colour in colours_bgr:
                    writer.write(np.full((16, 16, 3), colour, dtype=np.uint8))
            finally:
                writer.release()

            handler = StreamHandler(
                RecordedVideoSource(video_path, replay_mode=ReplayMode.FASTEST), queue_capacity=3
            )
            handler.open()
            packets = [handler.pump_once() for _ in colours_bgr]
            with self.assertRaises(SourceExhausted):
                handler.pump_once()
            handler.close()

        self.assertEqual([packet.frame_id for packet in packets], [0, 1, 2])
        self.assertTrue(all(packet.timestamp_origin == "source_presentation" for packet in packets))
        timestamps = [packet.capture_timestamp_ns for packet in packets]
        self.assertEqual(timestamps, sorted(timestamps))
        self.assertEqual(len(set(timestamps)), len(timestamps))
        np.testing.assert_allclose(packets[0].rgb[0, 0], [255, 0, 0], atol=10)
        np.testing.assert_allclose(packets[1].rgb[0, 0], [0, 255, 0], atol=10)
        np.testing.assert_allclose(packets[2].rgb[0, 0], [0, 0, 255], atol=10)


if __name__ == "__main__":
    unittest.main()
