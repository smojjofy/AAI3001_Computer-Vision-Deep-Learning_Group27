"""Replay identity, failed-refresh, split rules, and mask-input joints."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from applications.live_pipeline.adapters import DepthThresholdConfig
from applications.live_pipeline.pipeline import RefreshError
from applications.live_pipeline.replay import LiveReplayProcessor, ReplaySettings, run_replay
from applications.live_pipeline.window import WindowSelector

PREVIEW_JOINTS = [{"index": 0, "x": 0.5, "y": 0.5}]
# Values the preview encoder would round. The replay must save these unchanged.
REFRESH_MASK_JOINTS = [
    {"index": 0, "x": 0.123456789, "y": 0.987654321, "visibility": 0.8166, "depth_m": 1.234567},
]
REUSE_MASK_JOINTS = [
    {"index": 3, "x": 0.111111111, "y": 0.222222222, "visibility": 0.7123, "depth_m": None},
]


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
    def __init__(self, *, mask_pose_joints=REFRESH_MASK_JOINTS, reuse_joints=REUSE_MASK_JOINTS) -> None:
        self.attempts = 0
        self.mask_pose_joints = mask_pose_joints
        self.reuse_joints = reuse_joints

    def refresh(self, packet, generation: int):
        self.attempts += 1
        if self.attempts == 1:
            raise RefreshError("no person", person_missing=True)
        return SimpleNamespace(
            depth_m=np.ones((8, 8), np.float32), pose_joints=PREVIEW_JOINTS,
            mask_pose_joints=self.mask_pose_joints,
            mask=SimpleNamespace(mask=np.ones((8, 8), np.uint8), area_fraction=1.0),
            header={"mask": {"source": "pose_gradient", "fallback_reason": None}},
            splat_count=32, timings_ms={"total": 10.0},
        )

    def estimate_selected(self, packet):
        selection = SimpleNamespace(
            info=SimpleNamespace(mask=np.ones((8, 8), np.uint8), area_fraction=1.0),
            source="pose_gradient", fallback_reason=None,
        )
        return np.ones((8, 8), np.float32), self.reuse_joints, selection


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


def run_three_window_replay(tmp_path: Path, processor: Processor | None = None):
    frames = [packet(index, (index - 1) * 100_000_000) for index in range(1, 7)]
    selector = WindowSelector(FlowEstimator(), active_age_ns=1_000_000_000)
    output = tmp_path / "clip"
    run_replay(
        frames, selector=selector, processor=processor or Processor(), output_dir=output,
        settings=ReplaySettings("session1", "clip1", "train"), models={"flow": "fake"},
    )
    rows = [json.loads(line) for line in (output / "rows.jsonl").read_text().splitlines()]
    return output, rows


def test_refresh_row_saves_the_exact_mask_input_joints(tmp_path: Path) -> None:
    output, rows = run_three_window_replay(tmp_path)
    row = rows[1]
    assert row["status"] == "ok"
    assert row["mask_pose"] is not None
    payload = json.loads((output / row["mask_pose"]).read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1"
    assert payload["joints"] == REFRESH_MASK_JOINTS
    # The joint record carries the frame identity of the row it belongs to.
    assert payload["frame_id"] == row["selected_frame_id"]
    assert payload["timestamp_ns"] == row["selected_timestamp_ns"]
    assert payload["source_id"] == row["source_id"]
    # The display field keeps its rounded preview values.
    selected = json.loads((output / row["selected_pose"]).read_text(encoding="utf-8"))
    assert selected["joints"] == PREVIEW_JOINTS


def test_reuse_row_saves_the_joints_given_to_the_mask_selector(tmp_path: Path) -> None:
    output, rows = run_three_window_replay(tmp_path)
    row = rows[2]
    assert row["action"] == "reuse"
    payload = json.loads((output / row["mask_pose"]).read_text(encoding="utf-8"))
    assert payload["joints"] == REUSE_MASK_JOINTS
    assert payload["frame_id"] == row["selected_frame_id"]


def test_failed_refresh_row_has_no_mask_input_joints(tmp_path: Path) -> None:
    output, rows = run_three_window_replay(tmp_path)
    failed = rows[0]
    assert failed["status"] == "person_missing"
    assert failed["mask_pose"] is None
    key = "g{0}_f{1:08d}_mask_pose.json".format(failed["generation"], failed["selected_frame_id"])
    assert not (output / key).exists()


@pytest.mark.parametrize("joints", [None, []])
def test_row_is_null_when_the_outcome_has_no_mask_input_joints(tmp_path: Path, joints) -> None:
    _, rows = run_three_window_replay(tmp_path, Processor(mask_pose_joints=joints))
    assert rows[1]["status"] == "ok"
    assert rows[1]["mask_pose"] is None


def test_live_replay_processor_returns_the_joints_it_gives_the_selector(monkeypatch) -> None:
    from applications.live_pipeline import mask_selection as mask_selection_module

    captured: dict = {}

    def fake_select(depth_m, *, pose_joints, config=None):
        captured["joints"] = pose_joints
        return SimpleNamespace(
            info=SimpleNamespace(mask=np.ones((8, 8), np.uint8), area_fraction=0.5),
            source="pose_gradient", fallback_reason=None,
        )

    monkeypatch.setattr(mask_selection_module, "select_pose_gradient_mask", fake_select)
    landmarks = [SimpleNamespace(x=0.123456789, y=0.987654321, visibility=0.8166)]
    result = SimpleNamespace(
        depth=SimpleNamespace(depth=np.ones((8, 8), np.float32)),
        joint_depth=[1.234567],
        pose=SimpleNamespace(landmarks=landmarks),
    )
    estimator = SimpleNamespace(process_frame=lambda *args, **kwargs: result)
    pipeline = SimpleNamespace(
        _ensure_pipeline=lambda: estimator,
        mask_config=DepthThresholdConfig(),
    )
    _, joints, _ = LiveReplayProcessor(pipeline).estimate_selected(packet(9, 123))
    assert captured["joints"] is joints
    assert joints[0]["x"] == 0.123456789
    assert joints[0]["visibility"] == 0.8166
    assert joints[0]["depth_m"] == 1.234567


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
