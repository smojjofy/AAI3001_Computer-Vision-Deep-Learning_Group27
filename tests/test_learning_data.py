"""Boundary tests for the learning-data tools.

Covers the review-set selection rules, the separate label store, the
checked-mask evaluation with coverage, leakage and timing, the training filter
with one reason for each exclusion, and the rule that no report is written into
the replay data directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts import learning_data as module

WIDTH = 64
HEIGHT = 48
REGION = (slice(10, 30), slice(20, 44))


def person_label() -> np.ndarray:
    label = np.zeros((HEIGHT, WIDTH), np.uint8)
    label[REGION] = 255
    return label


def joints() -> list[dict]:
    return [
        {"index": 0, "x": 24 / (WIDTH - 1), "y": 12 / (HEIGHT - 1), "visibility": 0.9, "depth_m": 2.0},
        {"index": 1, "x": 40 / (WIDTH - 1), "y": 12 / (HEIGHT - 1), "visibility": 0.9, "depth_m": 2.0},
        {"index": 15, "x": 24 / (WIDTH - 1), "y": 26 / (HEIGHT - 1), "visibility": 0.9, "depth_m": 2.0},
        {"index": 16, "x": 40 / (WIDTH - 1), "y": 26 / (HEIGHT - 1), "visibility": 0.9, "depth_m": 2.0},
    ]


def depth_plane() -> np.ndarray:
    depth = np.full((HEIGHT, WIDTH), 4.0, np.float32)
    depth[REGION] = 2.0
    return depth


def write_row_files(output: Path, index: int, frame_id: int, *, with_mask: bool = True) -> dict:
    key = f"g0_f{frame_id:08d}"
    rgb = np.full((HEIGHT, WIDTH, 3), 90, np.uint8)
    cv2.imwrite(str(output / f"{key}_rgb.png"), rgb)
    depth = depth_plane()
    np.savez_compressed(
        output / f"{key}_depth.npz", depth_m=depth, validity=np.isfinite(depth).astype(np.float32)
    )
    (output / f"{key}_mask_pose.json").write_text(
        json.dumps({"schema_version": "1", "frame_id": frame_id, "timestamp_ns": frame_id,
                    "source_id": "recorded:test:clip", "joints": joints()}),
        encoding="utf-8",
    )
    row = {
        "generation": 0, "selected_frame_id": frame_id, "selected_timestamp_ns": frame_id * 1_000_000,
        "status": "ok", "error": None, "action": "reuse", "reference_frame_id": frame_id - 1,
        "mask_source": "pose_gradient", "fallback_reason": None, "mask_area_fraction": 0.2,
        "selected_rgb": f"{key}_rgb.png", "selected_depth": f"{key}_depth.npz",
        "mask_pose": f"{key}_mask_pose.json", "baseline_mask": None, "flow": f"{key}_flow.npz",
    }
    if with_mask:
        mask = person_label()
        cv2.imwrite(str(output / f"{key}_baseline_mask.png"), mask)
        row["baseline_mask"] = f"{key}_baseline_mask.png"
    np.savez_compressed(output / f"{key}_flow.npz", flow=np.zeros((2, 2, 2), np.float32))
    return row


@pytest.fixture()
def replay(tmp_path: Path) -> Path:
    output = tmp_path / "data" / "processed" / "session" / "clip"
    output.mkdir(parents=True)
    rows = []
    # Row 1 has a different pose gradient result per stride is not needed here;
    # the stride report is written directly.
    rows.append(write_row_files(output, 0, 1))
    rows.append(write_row_files(output, 1, 2))
    rows.append({
        "generation": 0, "selected_frame_id": 3, "selected_timestamp_ns": 3_000_000,
        "status": "ok", "error": None, "action": "reuse", "reference_frame_id": 2,
        "mask_source": "depth_threshold_fallback", "fallback_reason": "no_pose",
        "mask_area_fraction": 0.24, "selected_rgb": None, "selected_depth": None,
        "mask_pose": None, "baseline_mask": None, "flow": None,
    })
    rows.append({
        "generation": 0, "selected_frame_id": 4, "selected_timestamp_ns": 4_000_000,
        "status": "person_missing", "error": "pose not detected", "action": "refresh",
        "reference_frame_id": None, "mask_source": None, "fallback_reason": None,
        "mask_area_fraction": None, "selected_rgb": None, "selected_depth": None,
        "mask_pose": None, "baseline_mask": None, "flow": None,
    })
    rows.append(write_row_files(output, 4, 5))
    (output / "rows.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    stride_dir = Path(str(output) + ".stride_compare")
    stride_dir.mkdir()
    (stride_dir / "report.json").write_text(
        json.dumps({
            "rows": [
                {"frame_id": 1, "mask_iou": 0.55},
                {"frame_id": 2, "mask_iou": 0.99},
                {"frame_id": 3, "mask_iou": None},
                {"frame_id": 4, "mask_iou": None},
                {"frame_id": 5, "mask_iou": 0.98},
            ]
        }),
        encoding="utf-8",
    )
    return output


def test_frame_key_matches_the_inspection_overlay_name() -> None:
    assert module.frame_key({"generation": 0, "selected_frame_id": 3308}) == "g0_f00003308"
    assert module.frame_key({"generation": 0, "selected_frame_id": None}) == "unknown"


def test_review_set_takes_low_agreement_fallback_and_missing_rows(
    replay: Path, tmp_path: Path
) -> None:
    index = module.build_review_set(
        replay, review_dir=tmp_path / "review", log=lambda _message: None
    )
    frames = {item["frame_id"]: item for item in index["items"]}
    assert index["rows_total"] == 5
    assert index["rows_compared_for_stride"] == 5
    # Frame 1 has stride union 0.55, below the 0.9 threshold.
    assert "low_agreement" in frames[1]["reasons"]
    assert frames[1]["stride_iou"] == pytest.approx(0.55)
    # Frame 2 and 5 are high-agreement samples.
    assert "high_agreement_sample" in frames[2]["reasons"]
    assert "high_agreement_sample" in frames[5]["reasons"]
    # Frame 3 is the no-pose fallback row.
    assert "no_pose_fallback" in frames[3]["reasons"]
    # Frame 4 is the only missing-person row.
    assert "missing_person_sample" in frames[4]["reasons"]
    assert (tmp_path / "review" / "review_index.json").is_file()
    assert (tmp_path / "review" / "index.html").is_file()
    images = sorted((tmp_path / "review" / "images").glob("*.png"))
    # Only frames 1, 2 and 5 have an RGB file.
    assert len(images) == 3


def test_review_set_never_writes_inside_the_replay_data(replay: Path) -> None:
    with pytest.raises(ValueError, match="outside the replay data directory"):
        module.build_review_set(replay, review_dir=replay / "review", log=lambda _message: None)


def test_review_set_without_a_stride_report_still_picks_fallback_and_missing(
    replay: Path, tmp_path: Path
) -> None:
    index = module.build_review_set(
        replay, review_dir=tmp_path / "review",
        stride_report_dir=tmp_path / "no_stride_report", log=lambda _message: None,
    )
    reasons = {reason for item in index["items"] for reason in item["reasons"]}
    assert reasons == {"no_pose_fallback", "missing_person_sample"}


def test_label_store_keeps_checked_masks_out_of_the_replay_data(
    replay: Path, tmp_path: Path
) -> None:
    labels_root = tmp_path / "labels"
    path = module.save_checked_mask(
        labels_root, "movement_01", "g0_f00000001", person_label(), source="manual", notes="test"
    )
    assert path == labels_root / "movement_01" / "masks" / "g0_f00000001.png"
    assert path.is_file()
    assert not (replay / "labels").exists()
    metadata = json.loads(
        (labels_root / "movement_01" / "mask_meta" / "g0_f00000001.json").read_text(encoding="utf-8")
    )
    assert metadata["source"] == "manual"
    assert metadata["person_pixels"] == int((person_label() > 0).sum())
    loaded = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    assert set(np.unique(loaded).tolist()) == {0, 255}
    masks = module.labelled_frames(labels_root, "movement_01")
    assert list(masks) == ["g0_f00000001"]


def test_evaluation_scores_coverage_leakage_and_time(replay: Path, tmp_path: Path) -> None:
    labels_root = tmp_path / "labels"
    module.save_checked_mask(
        labels_root, "movement_01", "g0_f00000001", person_label(), source="manual"
    )
    report = module.evaluate_labelled(
        replay, labels_root, "movement_01", report_dir=tmp_path / "eval", log=lambda _message: None
    )
    assert report["frames"] == 1
    assert report["live_stride"] == 4
    record = report["records"][0]
    assert record["frame_id"] == 1
    assert record["joint_record"] == "mask_pose"
    for name in ("baseline", "stride1", "stride2", "stride4"):
        assert record[name]["label_pixels"] == int((person_label() > 0).sum())
        assert record[name]["person_coverage"] is not None
        assert record[name]["background_leakage"] is not None
    assert record["baseline"]["timing_ms"] is None
    for name in ("stride1", "stride2", "stride4"):
        assert record[name]["stride"] == int(name.removeprefix("stride"))
        assert record[name]["timing_ms"] is not None
        assert record[name]["timing_ms"] >= 0.0
    methods = report["summary"]["methods"]
    assert set(methods) == {"baseline", "stride1", "stride2", "stride4"}
    assert methods["stride4"]["timing_ms"]["frames"] == 1
    assert methods["baseline"]["timing_ms"]["frames"] == 0
    assert (tmp_path / "eval" / "report.json").is_file()
    assert (tmp_path / "eval" / "index.html").is_file()


def test_evaluation_requires_checked_masks(replay: Path, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no checked masks"):
        module.evaluate_labelled(
            replay, tmp_path / "labels", "movement_01",
            report_dir=tmp_path / "eval", log=lambda _message: None,
        )


def test_leakage_and_coverage_are_measured_against_the_label() -> None:
    label = person_label()
    perfect = label.copy()
    over_grown = label.copy()
    over_grown[0:8, 0:12] = True  # background object leak
    under_grown = label.copy()
    under_grown[10:15, :] = False  # person rows missed

    joints_value = joints()
    regions = (("head", tuple(range(0, 13))),)
    perfect_metrics = module.compare_to_label(
        perfect, label, joints=joints_value, regions=regions, rgb=None, pad_px=2
    )
    assert perfect_metrics["person_coverage"] == pytest.approx(1.0)
    assert perfect_metrics["background_leakage"] == pytest.approx(0.0)
    assert perfect_metrics["iou"] == pytest.approx(1.0)
    assert perfect_metrics["largest_leak_px"] == 0

    leaky = module.compare_to_label(
        over_grown, label, joints=joints_value, regions=regions, rgb=None, pad_px=2
    )
    assert leaky["person_coverage"] == pytest.approx(1.0)
    assert leaky["background_leakage"] > 0.0
    assert leaky["largest_leak_px"] == 8 * 12

    missed = module.compare_to_label(
        under_grown, label, joints=joints_value, regions=regions, rgb=None, pad_px=2
    )
    assert missed["person_coverage"] < 1.0
    assert missed["background_leakage"] == pytest.approx(0.0)

    missing_mask = module.compare_to_label(
        None, label, joints=joints_value, regions=regions, rgb=None, pad_px=2
    )
    assert missing_mask["error"] == "no mask for this method"
    assert missing_mask["person_coverage"] is None


def test_training_filter_excludes_each_reason_and_needs_a_label(
    replay: Path, tmp_path: Path
) -> None:
    labels_root = tmp_path / "labels"
    # Without a label store no row may enter the training set.
    report = module.training_eligibility(replay, log=lambda _message: None)
    assert report["eligible_rows"] == 0
    assert report["exclusion_reasons"]["no_label_store"] == 5

    module.save_checked_mask(
        labels_root, "movement_01", "g0_f00000001", person_label(), source="manual"
    )
    report = module.training_eligibility(
        replay, labels_root=labels_root, dataset="movement_01", log=lambda _message: None
    )
    assert report["eligible_rows"] == 1
    assert report["eligible"][0]["frame_key"] == "g0_f00000001"
    reasons = report["exclusion_reasons"]
    # Row 2 has no checked label yet.
    assert reasons["no_checked_mask"] == 4
    # Row 3 is the no-pose fallback row.
    assert reasons["no_pose_fallback_mask"] == 1
    # Row 4 is the failed refresh.
    assert reasons["selected_status_person_missing"] == 1
    assert reasons["no_reference"] == 1
    path = Path(str(replay) + ".training_filter.json")
    assert path.is_file()
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["eligible_rows"] == 1
    assert "baseline_mask is an input" in written["note"]
