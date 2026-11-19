"""Replay identity, failed-refresh, and split rules."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from applications.live_pipeline.pipeline import RefreshError
from applications.live_pipeline.replay import ReplaySettings, run_replay
from applications.live_pipeline.window import WindowSelector


class Flow:
    status = "ok"

    def __init__(self) -> None:
        self.flow = np.zeros((8, 8, 2), np.float32)
        self.valid_mask = np.ones((8, 8), bool)

    def metadata(self) -> dict[str, str]:
        return {"direction": "previous_to_current", "units": "pixels_at_output_resolution"}


class FlowEstimator:
    def estimate(self, previous, current) -> Flow:
        return Flow()


class Processor:
    def __init__(self) -> None:
        self.attempts = 0

    def refresh(self, packet, generation: int):
        self.attempts += 1
        if self.attempts == 1:
            raise RefreshError("no person", person_missing=True)
        return SimpleNamespace(
            depth_m=np.ones((8, 8), np.float32), pose_joints=[{"index": 0, "x": 0.5, "y": 0.5}],
            mask=SimpleNamespace(mask=np.ones((8, 8), np.uint8), area_fraction=1.0),
            header={"mask": {"source": "pose_gradient", "fallback_reason": None}},
            splat_count=32, timings_ms={"total": 10.0},
        )

    def estimate_selected(self, packet):
        selection = SimpleNamespace(
            info=SimpleNamespace(mask=np.ones((8, 8), np.uint8), area_fraction=1.0),
            source="pose_gradient", fallback_reason=None,
        )
        return np.ones((8, 8), np.float32), [], selection


def packet(frame_id: int, timestamp_ns: int):
    return SimpleNamespace(
        frame_id=frame_id, capture_timestamp_ns=timestamp_ns, source_id="clip",
        timestamp_origin="source_presentation", rgb=np.zeros((8, 8, 3), np.uint8),
        width=8, height=8,
    )


def test_replay_keeps_reference_only_after_success(tmp_path: Path) -> None:
    frames = [packet(index, (index - 1) * 100_000_000) for index in range(1, 7)]
    selector = WindowSelector(FlowEstimator(), active_age_ns=1_000_000_000)
    processor = Processor()
    output = tmp_path / "clip"
    counts = run_replay(
        frames, selector=selector, processor=processor, output_dir=output,
        settings=ReplaySettings("session1", "clip1", "train"), models={"flow": "fake"},
    )
    rows = [json.loads(line) for line in (output / "rows.jsonl").read_text().splitlines()]
    assert counts == {"windows": 3, "refresh_ok": 1, "refresh_failed": 1, "reuse": 1}
    assert [row["selected_frame_id"] for row in rows] == [1, 3, 5]
    assert rows[0]["status"] == "person_missing" and rows[0]["reference_frame_id"] is None
    assert rows[1]["reference_frame_id"] is None
    assert rows[2]["reference_frame_id"] == 3
    assert rows[2]["reference_timestamp_ns"] == frames[2].capture_timestamp_ns
    assert rows[2]["reference_depth"] == rows[1]["selected_depth"]
    assert rows[2]["reference_rgb"] == rows[1]["selected_rgb"]
    assert rows[2]["mask_source"] == "pose_gradient"
    assert (output / rows[2]["flow"]).is_file()


def test_replay_settings_reject_unsafe_ids_and_split() -> None:
    with pytest.raises(ValueError):
        ReplaySettings("../other", "clip", "train")
    with pytest.raises(ValueError):
        ReplaySettings("session", "clip", "unknown")


def test_video_reader_uses_one_timestamp_origin(tmp_path: Path) -> None:
    import cv2
    from scripts.build_learning_replay import video_packets

    clip = tmp_path / "frames.avi"
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    if not writer.isOpened():
        pytest.skip("MJPG encoder is unavailable")
    for index in range(5):
        writer.write(np.full((48, 64, 3), 40 + index, np.uint8))
    writer.release()
    packets = list(video_packets(clip, "recorded:test", 5))
    assert len(packets) == 5
    assert len({packet.timestamp_origin for packet in packets}) == 1
    assert all(a.capture_timestamp_ns < b.capture_timestamp_ns for a, b in zip(packets, packets[1:]))
