"""Boundary tests for the replay mask inspection report.

The report is a measurement tool, so these tests check the measurement rules:
separate denominators, missing files against zero coverage, files against size
mismatch, the visibility floor, exact against rounded joint records, and that the
report never writes into the replay data directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts import inspect_replay_masks as module

WIDTH = 64
HEIGHT = 48
# The head box with a 2-pixel pad is x 14..22 and y 6..14, so 9 by 9 pixels.
MASK = np.zeros((HEIGHT, WIDTH), np.uint8)
MASK[6:15, 14:23] = 255


def joint(index: int, px: int, py: int, *, visibility: float = 0.9) -> dict:
    return {
        "index": index,
        "x": px / (WIDTH - 1),
        "y": py / (HEIGHT - 1),
        "visibility": visibility,
        "depth_m": 2.0,
    }


def head_joints(*, visibility: float = 0.9) -> list[dict]:
    return [
        joint(0, 16, 8, visibility=visibility),
        joint(1, 20, 8, visibility=visibility),
        joint(2, 16, 12, visibility=visibility),
        joint(3, 20, 12, visibility=visibility),
    ]


def left_hand_joints() -> list[dict]:
    return [joint(15, 48, 36), joint(17, 52, 36), joint(19, 48, 40)]


def write_image(path: Path, image: np.ndarray) -> None:
    assert cv2.imwrite(str(path), image)


def write_joint_file(path: Path, joints: list[dict], *, frame_id: int = 1) -> None:
    payload = {
        "schema_version": "1",
        "frame_id": frame_id,
        "timestamp_ns": 100,
        "source_id": "recorded:session:clip",
        "joints": joints,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def base_row(index: int, **overrides) -> dict:
    row = {
        "generation": 0,
        "selected_frame_id": index,
        "status": "ok",
        "error": None,
        "mask_source": "pose_gradient",
        "fallback_reason": None,
        "mask_area_fraction": 0.1,
        "selected_rgb": None,
        "selected_pose": None,
        "mask_pose": None,
        "baseline_mask": None,
    }
    row.update(overrides)
    return row


@pytest.fixture()
def replay_dir(tmp_path: Path) -> Path:
    """Five rows that cover every exclusion and denominator rule."""
    output = tmp_path / "session" / "clip"
    output.mkdir(parents=True)
    rgb = np.full((HEIGHT, WIDTH, 3), 60, np.uint8)
    for name in ("row1_rgb.png", "row2_rgb.png", "row4_rgb.png", "row5_rgb.png"):
        write_image(output / name, rgb)
    for name in ("row1_mask.png", "row2_mask.png"):
        write_image(output / name, MASK)
    # Row 5 has a mask of the wrong size.
    write_image(output / "row5_mask.png", np.zeros((HEIGHT // 2, WIDTH // 2), np.uint8))

    write_joint_file(
        output / "row1_mask_pose.json", head_joints() + left_hand_joints() + [joint(16, 50, 38)]
    )
    write_image(output / "row1_mask.png", MASK)
    write_joint_file(
        output / "row4_mask_pose.json", head_joints(), frame_id=4
    )
    write_joint_file(output / "row5_mask_pose.json", head_joints(), frame_id=5)
    # Row 2 only has the rounded display joints.
    (output / "row2_pose.json").write_text(
        json.dumps({"joints": head_joints(visibility=0.85)}), encoding="utf-8"
    )

    rows = [
        base_row(1, selected_rgb="row1_rgb.png", baseline_mask="row1_mask.png",
                 mask_pose="row1_mask_pose.json"),
        base_row(2, selected_rgb="row2_rgb.png", baseline_mask="row2_mask.png",
                 selected_pose="row2_pose.json", mask_source="depth_threshold_fallback",
                 fallback_reason="no_pose"),
        base_row(3, status="person_missing", error="pose not detected"),
        base_row(4, selected_rgb="row4_rgb.png", baseline_mask="missing_mask.png",
                 mask_pose="row4_mask_pose.json"),
        base_row(5, selected_rgb="row5_rgb.png", baseline_mask="row5_mask.png",
                 mask_pose="row5_mask_pose.json"),
    ]
    (output / "rows.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    return output


def inspect(replay_dir: Path, report_dir: Path) -> dict:
    return module.inspect_replay(replay_dir, report_dir=report_dir, log=lambda _message: None)


def test_report_separates_every_denominator(replay_dir: Path, tmp_path: Path) -> None:
    report = inspect(replay_dir, tmp_path / "report")
    summary = report["summary"]
    assert summary["rows_total"] == 5
    assert summary["row_status"] == {"ok": 4, "person_missing": 1}
    assert summary["mask_presence"] == {"present": 3, "absent": 2}
    assert summary["pose_joint_record"] == {
        "absent": 1, "mask_pose": 3, "selected_pose_rounded": 1,
    }
    assert summary["rows_with_usable_pose"] == 4

    head = summary["regions"]["head"]
    assert head["denominator_rows"] == 2
    assert head["coverage"]["p50"] == pytest.approx(1.0)
    assert head["excluded"] == {
        "missing_file": 1, "no_mask": 1, "size_mismatch": 1,
    }
    left = summary["regions"]["left_hand"]
    assert left["denominator_rows"] == 1
    assert left["coverage"]["p50"] == pytest.approx(0.0)
    assert left["joint_center_coverage"]["p50"] == pytest.approx(0.0)
    assert left["excluded"]["insufficient_visible_joints"] == 1
    right = summary["regions"]["right_hand"]
    assert right["denominator_rows"] == 0
    assert right["excluded"]["insufficient_visible_joints"] == 2


def test_missing_file_is_not_zero_coverage(replay_dir: Path, tmp_path: Path) -> None:
    report = inspect(replay_dir, tmp_path / "report")
    rows = {row["frame_id"]: row for row in report["rows"]}
    missing = rows[4]
    assert missing["mask_present"] is False
    assert missing["regions"]["head"] is None
    assert missing["excluded"]["head"] == "missing_file"
    assert any("baseline_mask missing" in problem for problem in missing["problems"])
    mismatch = rows[5]
    assert mismatch["mask_present"] is True
    assert mismatch["excluded"]["head"] == "size_mismatch"
    assert any("size differ" in problem for problem in mismatch["problems"])
    absent = rows[3]
    assert absent["excluded"]["head"] == "no_mask"
    assert absent["status"] == "person_missing"


def test_exact_joint_record_and_rounded_record_are_labelled(
    replay_dir: Path, tmp_path: Path
) -> None:
    report = inspect(replay_dir, tmp_path / "report")
    rows = {row["frame_id"]: row for row in report["rows"]}
    assert rows[1]["pose_joint_source"] == "mask_pose"
    assert rows[1]["exact_mask_joints"] is True
    assert rows[2]["pose_joint_source"] == "selected_pose_rounded"
    assert rows[2]["exact_mask_joints"] is False
    assert rows[3]["pose_joint_source"] == "absent"


def test_head_region_uses_the_visibility_floor(replay_dir: Path, tmp_path: Path) -> None:
    # Every head joint is below the 0.15 floor, so the row leaves the denominator
    # instead of adding a zero-coverage sample.
    (replay_dir / "row1_mask_pose.json").write_text(
        json.dumps({"joints": head_joints(visibility=0.1)}), encoding="utf-8"
    )
    report = inspect(replay_dir, tmp_path / "report")
    rows = {row["frame_id"]: row for row in report["rows"]}
    assert rows[1]["regions"]["head"] is None
    assert rows[1]["excluded"]["head"] == "insufficient_visible_joints"
    assert report["summary"]["regions"]["head"]["denominator_rows"] == 1


def test_overlays_and_html_are_written_outside_the_replay_data(
    replay_dir: Path, tmp_path: Path
) -> None:
    report_dir = tmp_path / "report"
    report = inspect(replay_dir, report_dir)
    assert (report_dir / "report.json").is_file()
    page = (report_dir / "index.html").read_text(encoding="utf-8")
    assert "diagnostic proxy" in page
    assert "Replay mask inspection" in page
    overlays = sorted((report_dir / "overlays").glob("*.png"))
    assert len(overlays) == 2
    rows = {row["frame_id"]: row for row in report["rows"]}
    assert rows[1]["overlay"] is not None and (report_dir / rows[1]["overlay"]).is_file()
    assert rows[4]["overlay"] is None
    assert rows[5]["overlay"] is None
    # The replay data directory gains no report file.
    assert not (replay_dir / "report.json").exists()
    assert not (replay_dir / "index.html").exists()


def test_report_directory_inside_the_data_directory_is_refused(
    replay_dir: Path,
) -> None:
    with pytest.raises(ValueError, match="outside the replay data directory"):
        module.inspect_replay(replay_dir, report_dir=replay_dir / "report")


def test_missing_rows_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="rows.jsonl"):
        module.inspect_replay(tmp_path / "empty", report_dir=tmp_path / "report")


def test_summary_prints_every_source_group(replay_dir: Path, tmp_path: Path) -> None:
    report = inspect(replay_dir, tmp_path / "report")
    lines: list[str] = []
    module.print_report(report, log=lines.append)
    assert any("source=pose_gradient" in line for line in lines)
    assert any("source=depth_threshold_fallback" in line for line in lines)
    assert any("region=head" in line for line in lines)
