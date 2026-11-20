"""Boundary tests for the stride-1 against stride-4 mask comparison.

The comparison runs the live selector twice on one saved frame. These tests check
that both strides use the same unpatched selector, that rows without exact joints
or saved depth leave the comparison with a named reason, that the paired overlay
is written, and that the report stays outside the replay data directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from applications.live_pipeline.adapters import DepthThresholdConfig
from applications.live_pipeline.mask_selection import select_pose_gradient_mask
from scripts import compare_mask_strides as module

WIDTH = 64
HEIGHT = 48


def depth_plane() -> np.ndarray:
    depth = np.full((HEIGHT, WIDTH), 4.0, np.float32)
    depth[14:38, 18:46] = 1.5
    return depth


def pose_joints() -> list[dict]:
    return [
        {"index": 0, "x": 0.40, "y": 0.40, "visibility": 0.9, "depth_m": 1.5},
        {"index": 1, "x": 0.55, "y": 0.40, "visibility": 0.9, "depth_m": 1.5},
        {"index": 2, "x": 0.40, "y": 0.60, "visibility": 0.9, "depth_m": 1.5},
    ]


def write_joint_file(path: Path, joints: list[dict], frame_id: int) -> None:
    path.write_text(
        json.dumps({"schema_version": "1", "frame_id": frame_id, "timestamp_ns": 1,
                    "source_id": "recorded:test:clip", "joints": joints}),
        encoding="utf-8",
    )


def write_replay(tmp_path: Path) -> Path:
    output = tmp_path / "session" / "clip"
    output.mkdir(parents=True)
    depth = depth_plane()
    np.savez_compressed(
        output / "row1_depth.npz", depth_m=depth, validity=np.isfinite(depth).astype(np.float32)
    )
    write_joint_file(output / "row1_mask_pose.json", pose_joints(), 1)
    # The saved mask comes from the live default, so a stride-4 recompute agrees.
    saved = select_pose_gradient_mask(
        depth, pose_joints=pose_joints(), config=DepthThresholdConfig()
    )
    assert saved.source == "pose_gradient"
    cv2.imwrite(str(output / "row1_mask.png"), saved.info.mask.astype(np.uint8) * 255)
    rgb = np.full((HEIGHT, WIDTH, 3), 70, np.uint8)
    cv2.imwrite(str(output / "row1_rgb.png"), rgb)
    write_joint_file(output / "row2_mask_pose.json", pose_joints(), 2)
    write_joint_file(output / "row3_mask_pose.json", pose_joints(), 3)
    rows = [
        {
            "generation": 0, "selected_frame_id": 1, "status": "ok",
            "mask_source": "pose_gradient", "fallback_reason": None,
            "selected_depth": "row1_depth.npz", "selected_rgb": "row1_rgb.png",
            "mask_pose": "row1_mask_pose.json", "baseline_mask": "row1_mask.png",
        },
        {
            "generation": 0, "selected_frame_id": 2, "status": "ok",
            "mask_source": "pose_gradient", "fallback_reason": None,
            "selected_depth": None, "selected_rgb": None,
            "mask_pose": "row2_mask_pose.json", "baseline_mask": None,
        },
        {
            "generation": 0, "selected_frame_id": 3, "status": "ok",
            "mask_source": None, "fallback_reason": None,
            "selected_depth": "row3_depth.npz", "selected_rgb": None,
            "mask_pose": None, "baseline_mask": None,
        },
    ]
    (output / "rows.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    return output


def compare(output: Path, report: Path) -> dict:
    return module.compare_replay(output, report_dir=report, log=lambda _message: None)


def test_both_strides_use_the_same_unpatched_selector(tmp_path: Path, monkeypatch) -> None:
    output = write_replay(tmp_path)
    seen: list[int] = []
    real = module.select_pose_gradient_mask

    def recording_selector(*args, **kwargs):
        seen.append(kwargs["gradient_stride"])
        return real(*args, **kwargs)

    monkeypatch.setattr(module, "select_pose_gradient_mask", recording_selector)
    report = compare(output, tmp_path / "report")
    assert seen == [4, 1]
    row = report["rows"][0]
    assert row["reference"]["stride"] == 4
    assert row["alternate"]["stride"] == 1
    assert report["reference_stride"] == 4
    assert report["alternate_stride"] == 1


def test_comparison_reports_union_and_saved_agreement(tmp_path: Path) -> None:
    output = write_replay(tmp_path)
    report = compare(output, tmp_path / "report")
    summary = report["summary"]
    assert summary["rows_total"] == 3
    assert summary["rows_compared"] == 1
    assert summary["fallback_sources"]["both_pose_gradient"] == 1
    assert summary["mask_iou"]["rows"] == 1
    assert 0.0 <= summary["mask_iou"]["p50"] <= 1.0
    # The stride-4 recompute must reproduce the saved live mask from the same data.
    assert summary["saved_mask_iou_reference"]["p50"] == pytest.approx(1.0)
    assert summary["saved_mask_mismatch_rows"] == 0
    assert summary["different_pixel_fraction"]["rows"] == 1
    assert summary["area_abs_delta"]["rows"] == 1


def test_rows_without_exact_joints_or_depth_are_excluded(tmp_path: Path) -> None:
    output = write_replay(tmp_path)
    report = compare(output, tmp_path / "report")
    rows = {row["frame_id"]: row for row in report["rows"]}
    assert rows[2]["excluded"] == "no_depth"
    assert rows[3]["excluded"] == "no_joint_record"
    assert report["summary"]["exclusions"] == {"no_depth": 1, "no_joint_record": 1}


def test_missing_depth_file_is_not_a_zero_comparison(tmp_path: Path) -> None:
    output = write_replay(tmp_path)
    rows_path = output / "rows.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["selected_depth"] = "does_not_exist.npz"
    rows_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    report = compare(output, tmp_path / "report")
    row = report["rows"][0]
    assert row["excluded"] == "missing_file"
    assert any("depth file missing" in problem for problem in row["problems"])
    assert report["summary"]["rows_compared"] == 0


def test_paired_overlay_is_written_outside_the_replay_data(tmp_path: Path) -> None:
    output = write_replay(tmp_path)
    report_dir = tmp_path / "report"
    report = compare(output, report_dir)
    row = report["rows"][0]
    overlay_path = report_dir / row["overlay"]
    assert overlay_path.is_file()
    image = cv2.imread(str(overlay_path))
    assert image.shape[1] == 2 * WIDTH
    assert image.shape[0] == HEIGHT
    assert not (output / "index.html").exists()


def test_report_directory_inside_the_data_directory_is_refused(tmp_path: Path) -> None:
    output = write_replay(tmp_path)
    with pytest.raises(ValueError, match="outside the replay data directory"):
        module.compare_replay(output, report_dir=output / "report")


def test_summary_prints_the_comparison(tmp_path: Path) -> None:
    output = write_replay(tmp_path)
    report = compare(output, tmp_path / "report")
    lines: list[str] = []
    module.print_report(report, log=lines.append)
    assert any("compared: 1" in line for line in lines)
    assert any("mask_iou" in line for line in lines)
    assert any("region=head" in line for line in lines)
