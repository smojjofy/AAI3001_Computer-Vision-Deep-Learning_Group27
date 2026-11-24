"""Tests for the controlled person-mask checkpoint test.

The tests use a synthetic miniature replay, a synthetic label store, and a tiny
``bounded_residual`` checkpoint. No real data and no camera are necessary. Every
write goes to ``tmp_path``.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from modules.inference.checkpoint import save_checkpoint
from modules.inference.paired_research.model import PairedMaskMotionNet
from scripts import person_mask_test as module

WIDTH = 64
HEIGHT = 48
TINY_WIDTH = 320
TINY_HEIGHT = 180


# ------------------------------------------------------------------ helpers


def face_joints(*, visibility: float = 0.9, count: int = 11) -> list[dict]:
    """Eleven face pose points in the middle of the image."""
    return [
        {"index": index, "x": 0.40 + 0.01 * index, "y": 0.35, "visibility": visibility,
         "depth_m": 2.0}
        for index in range(count)
    ]


def checked_person() -> np.ndarray:
    """A solid checked person mask."""
    mask = np.zeros((HEIGHT, WIDTH), bool)
    mask[8:40, 20:44] = True
    return mask


def fixture_depth() -> np.ndarray:
    """A depth plane with one nearer rectangle that becomes the person."""
    depth = np.full((HEIGHT, WIDTH), 4.0, np.float32)
    depth[8:40, 20:44] = 2.0
    return depth


def _live_selector_mask(depth: np.ndarray, joints: list[dict]) -> np.ndarray:
    """The live pose-gradient selection with the live defaults, as a boolean mask."""
    from applications.live_pipeline.adapters import DepthThresholdConfig
    from applications.live_pipeline.mask_selection import select_pose_gradient_mask

    selection = select_pose_gradient_mask(
        depth, pose_joints=joints, config=DepthThresholdConfig(), gradient_stride=4
    )
    return np.asarray(selection.info.mask) > 0


# --------------------------------------------------------------- percentile


def test_percentile_uses_nearest_rank() -> None:
    assert module.percentile([1.0, 2.0, 3.0], 95.0) == 3.0
    assert module.percentile([1.0, 2.0, 3.0], 50.0) == 2.0
    # An even list at p50 takes rank ceil(0.5 * 4) = 2, so the second value.
    assert module.percentile([4.0, 1.0, 3.0, 2.0], 50.0) == 2.0
    assert module.percentile([4.0, 1.0, 3.0, 2.0], 67.0) == 3.0


def test_percentile_clamps_at_the_ends() -> None:
    assert module.percentile([5.0, 1.0, 9.0], 0.0) == 1.0
    assert module.percentile([5.0, 1.0, 9.0], 100.0) == 9.0


def test_percentile_of_an_empty_list_is_none() -> None:
    assert module.percentile([], 50.0) is None


# -------------------------------------------------------------- face region


def test_face_region_is_none_below_the_visibility_floor() -> None:
    joints = face_joints(visibility=0.10)
    assert module.face_region_from_joints(
        joints, width=WIDTH, height=HEIGHT
    ) is None


def test_face_region_needs_three_visible_points() -> None:
    joints = [
        {"index": 0, "x": 0.40, "y": 0.35, "visibility": 0.9},
        {"index": 1, "x": 0.45, "y": 0.35, "visibility": 0.9},
        {"index": 2, "x": 0.50, "y": 0.35, "visibility": 0.05},
    ]
    assert module.face_region_from_joints(joints, width=WIDTH, height=HEIGHT) is None
    assert module.face_region_from_joints(None, width=WIDTH, height=HEIGHT) is None


def test_face_region_is_a_padded_box_for_a_normal_face() -> None:
    joints = face_joints()
    region = module.face_region_from_joints(joints, width=WIDTH, height=HEIGHT)
    assert region is not None
    left, right, top, bottom = region
    # Points span x 0.40 to 0.50 and y 0.35, so the raw box is 25 to 31 by 16.
    x_min = int(round(0.40 * (WIDTH - 1)))
    x_max = int(round(0.50 * (WIDTH - 1)))
    y_centre = int(round(0.35 * (HEIGHT - 1)))
    pad_x = max(1, int(round(0.10 * (x_max - x_min))))
    assert left == x_min - pad_x
    assert right == x_max + pad_x
    assert top <= y_centre <= bottom
    assert right > left and bottom > top


def test_face_region_is_clipped_to_the_image() -> None:
    joints = [
        {"index": 0, "x": 0.00, "y": 0.00, "visibility": 0.9},
        {"index": 1, "x": 0.02, "y": 0.01, "visibility": 0.9},
        {"index": 2, "x": 1.00, "y": 1.00, "visibility": 0.9},
        {"index": 3, "x": 0.98, "y": 0.99, "visibility": 0.9},
    ]
    region = module.face_region_from_joints(joints, width=WIDTH, height=HEIGHT)
    assert region is not None
    left, right, top, bottom = region
    assert left == 0
    assert top == 0
    assert right == WIDTH - 1
    assert bottom == HEIGHT - 1


# ------------------------------------------------------------- face edge mask


def test_face_edge_mask_keeps_only_the_boundary_inside_the_region() -> None:
    checked = checked_person()
    # The region covers only the top part of the person.
    region = (20, 44, 8, 20)
    edge = module.face_edge_mask(checked, region, edge_px=3)
    assert edge.dtype == bool
    assert edge.shape == checked.shape
    assert edge.any()
    assert np.logical_and(edge, ~checked).sum() == 0
    # Every edge pixel is inside the region.
    rows, columns = np.nonzero(edge)
    assert rows.min() >= region[2]
    assert rows.max() <= region[3]
    assert columns.min() >= region[0]
    assert columns.max() <= region[1]
    # A pixel at the centre of the person is not a boundary pixel.
    assert not edge[24, 32]
    # Without a region nothing is selected.
    assert not module.face_edge_mask(checked, None).any()


# ----------------------------------------------------------------- score mask


def test_score_mask_perfect_mask() -> None:
    checked = checked_person()
    scores = module.score_mask(checked, checked, face_region=(20, 44, 8, 40))
    assert scores["iou"] == pytest.approx(1.0)
    assert scores["person_coverage"] == pytest.approx(1.0)
    assert scores["background_leakage"] == pytest.approx(0.0)
    assert scores["largest_leak_px"] == 0
    assert scores["face_edge_coverage"] == pytest.approx(1.0)
    assert scores["checked_pixels"] == int(checked.sum())
    assert scores["mask_pixels"] == int(checked.sum())
    assert scores["face_region_pixels"] == 25 * 33


def test_score_mask_over_grown_mask_leaks() -> None:
    checked = checked_person()
    grown = checked.copy()
    grown[0:6, 0:10] = True
    scores = module.score_mask(grown, checked, face_region=None)
    assert scores["person_coverage"] == pytest.approx(1.0)
    assert scores["background_leakage"] > 0.0
    assert scores["largest_leak_px"] == 6 * 10
    assert scores["face_edge_coverage"] is None


def test_score_mask_under_grown_mask_misses_person() -> None:
    checked = checked_person()
    missed = checked.copy()
    missed[8:14, :] = False
    scores = module.score_mask(missed, checked, face_region=None)
    assert scores["person_coverage"] < 1.0
    assert scores["background_leakage"] == pytest.approx(0.0)
    assert scores["iou"] < 1.0


def test_score_mask_returns_an_error_for_a_different_shape() -> None:
    checked = checked_person()
    scores = module.score_mask(np.zeros((10, 10), bool), checked, face_region=None)
    assert scores["error"] == "mask and label size differ"
    assert scores["iou"] is None
    assert scores["checked_pixels"] == int(checked.sum())


def test_score_mask_missing_mask_is_named_not_zero() -> None:
    checked = checked_person()
    scores = module.score_mask(None, checked, face_region=None)
    assert scores["error"] == "no mask for this method"
    assert scores["iou"] is None


def test_score_mask_degenerate_checked_mask_gives_none() -> None:
    empty = np.zeros((HEIGHT, WIDTH), bool)
    scores = module.score_mask(empty, empty, face_region=None)
    assert scores["iou"] is None
    assert scores["person_coverage"] is None
    assert scores["background_leakage"] is None
    assert scores["checked_pixels"] == 0
    assert scores["face_edge_coverage"] is None


# ------------------------------------------------------------ pairwise summary


def test_pairwise_summary_counts_and_interval() -> None:
    diffs = [0.10, -0.05, 0.02, 0.00, 0.20, -0.10, 0.08, 0.03]
    summary = module.pairwise_summary(diffs, bootstrap=200, seed=7)
    assert summary["frames"] == 8
    assert summary["mean"] == pytest.approx(0.035, abs=1e-9)
    assert summary["median"] == pytest.approx(0.025, abs=1e-9)
    assert summary["wins"] == 5
    assert summary["ties"] == 1
    assert summary["losses"] == 2
    assert summary["lower_95"] is not None
    assert summary["upper_95"] is not None
    assert summary["lower_95"] <= summary["upper_95"]
    again = module.pairwise_summary(diffs, bootstrap=200, seed=7)
    assert again["lower_95"] == summary["lower_95"]
    assert again["upper_95"] == summary["upper_95"]
    other = module.pairwise_summary(diffs, bootstrap=200, seed=99)
    assert (other["lower_95"], other["upper_95"]) != (
        summary["lower_95"], summary["upper_95"]
    )


def test_pairwise_summary_single_frame_has_a_zero_width_interval() -> None:
    summary = module.pairwise_summary([0.05], bootstrap=100, seed=7)
    assert summary["lower_95"] == summary["upper_95"] == pytest.approx(0.05)
    assert summary["wins"] == 1


def test_pairwise_summary_empty_list_is_all_none() -> None:
    summary = module.pairwise_summary([], bootstrap=10, seed=7)
    assert summary["frames"] == 0
    assert summary["mean"] is None
    assert summary["lower_95"] is None
    assert summary["upper_95"] is None


# -------------------------------------------------------------------- decide


def _metrics(*, lower: float, mean: float, wins: int, losses: int) -> dict:
    return {"lower_95": lower, "mean": mean, "wins": wins, "losses": losses}


def test_decide_promotes_only_when_all_conditions_hold() -> None:
    promoted = module.decide(_metrics(lower=0.01, mean=0.05, wins=8, losses=4))
    assert promoted["outcome"] == "offline_gate_passed"
    assert promoted["offline_gate_passed"] is True
    # The offline gate never promotes a live model by itself.
    assert promoted["live_promoted"] is False
    assert promoted["limits"]
    assert promoted["reason"] == []


def test_decide_names_every_failed_condition() -> None:
    below = module.decide(_metrics(lower=-0.01, mean=0.05, wins=8, losses=4))
    assert below["outcome"] == "offline_gate_failed"
    assert below["reason"] == ["lower_bootstrap_bound_not_above_zero"]

    small = module.decide(_metrics(lower=0.01, mean=0.02, wins=8, losses=4))
    assert small["reason"] == ["mean_gain_below_minimum"]

    many_losses = module.decide(_metrics(lower=0.01, mean=0.05, wins=4, losses=8))
    assert many_losses["reason"] == ["losses_exceed_wins"]

    everything = module.decide(_metrics(lower=0.0, mean=0.0, wins=1, losses=9))
    assert everything["outcome"] == "offline_gate_failed"
    assert set(everything["reason"]) == {
        "lower_bootstrap_bound_not_above_zero",
        "mean_gain_below_minimum",
        "losses_exceed_wins",
    }


# ------------------------------------------------------------ synthetic replay


def _write_pose(path: Path, frame_id: int, joints: list[dict]) -> None:
    path.write_text(
        json.dumps({
            "schema_version": "1", "frame_id": frame_id, "timestamp_ns": frame_id,
            "source_id": "recorded:test:clip", "joints": joints,
        }),
        encoding="utf-8",
    )


def _write_row_files(
    replay: Path, labels: Path, dataset: str, frame_id: int, reference_id: int,
    *, rgba_shift: int = 0,
) -> dict:
    """Write one complete replay pair and its checked mask."""
    key = f"g0_f{frame_id:08d}"
    reference_key = f"g0_f{reference_id:08d}"
    rgb = np.full((HEIGHT, WIDTH, 3), 60 + rgba_shift, np.uint8)
    cv2.imwrite(str(replay / f"{key}_rgb.png"), rgb)
    cv2.imwrite(str(replay / f"{reference_key}_rgb.png"), rgb)
    depth = fixture_depth()
    for name in (key, reference_key):
        np.savez_compressed(
            replay / f"{name}_depth.npz", depth_m=depth,
            validity=np.isfinite(depth).astype(np.float32),
        )
    joints = face_joints()
    _write_pose(replay / f"{key}_mask_pose.json", frame_id, joints)
    _write_pose(replay / f"{reference_key}_mask_pose.json", reference_id, joints)
    # The saved accepted mask is the live selector's own output, so the script
    # recompute agrees with it.
    accepted = _live_selector_mask(depth, joints)
    cv2.imwrite(str(replay / f"{key}_baseline_mask.png"), accepted.astype(np.uint8) * 255)
    np.savez_compressed(
        replay / f"{key}_flow.npz",
        flow=np.zeros((HEIGHT, WIDTH, 2), np.float32),
        valid_mask=np.ones((HEIGHT, WIDTH), np.uint8),
    )
    # The checked mask is the true person.
    checked_dir = labels / dataset / "masks"
    checked_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(checked_dir / f"{key}.png"), checked_person().astype(np.uint8) * 255)
    return {
        "split": "test", "status": "ok",
        "selected_frame_id": frame_id, "reference_frame_id": reference_id,
        "selected_timestamp_ns": frame_id * 1_000_000,
        "selected_rgb": f"{key}_rgb.png", "reference_rgb": f"{reference_key}_rgb.png",
        "selected_depth": f"{key}_depth.npz", "reference_depth": f"{reference_key}_depth.npz",
        "mask_pose": f"{key}_mask_pose.json",
        "flow": f"{key}_flow.npz",
        "flow_metadata": {
            "previous_frame_id": reference_id, "current_frame_id": frame_id,
            "direction": "previous_to_current",
        },
        "baseline_mask": f"{key}_baseline_mask.png",
        "generation": 0, "mask_source": "pose_gradient", "fallback_reason": None,
    }


def _write_checkpoint(path: Path, *, trained_heads: list[str] | None = None) -> None:
    """Write a tiny bounded_residual checkpoint that reads the 15 channels.

    The mask head is not zero, so the candidate output differs from the accepted
    mask. A fresh checkpoint starts with a zero head, which would only make ties.
    """
    torch.manual_seed(3)
    model = PairedMaskMotionNet(
        in_channels=15, base_channels=4, mask_mode="bounded_residual", gate_radius_px=12
    )
    with torch.no_grad():
        model.mask_head.weight.normal_(mean=0.0, std=0.5)
        model.mask_head.bias.fill_(0.4)
    save_checkpoint(
        path, task="paired_mask_motion_research", model=model,
        model_config={
            "in_channels": 15, "base_channels": 4, "mask_mode": "bounded_residual",
            "gate_radius_px": 12, "residual_scale": 4.0,
            "trained_heads": ["mask"] if trained_heads is None else trained_heads,
            "session_id": "person_train_01", "clip_sha256": "trainclip",
        },
        epoch=1, global_step=1,
    )


@pytest.fixture()
def fixture(tmp_path: Path) -> dict:
    """A three-frame synthetic replay, a label store, and a tiny checkpoint."""
    replay = tmp_path / "data" / "person_test_01" / "test_01"
    replay.mkdir(parents=True)
    labels = tmp_path / "data" / "person_test_01" / "labels"
    rows = [
        _write_row_files(replay, labels, "test_01", index + 1, index, rgba_shift=index)
        for index in range(3)
    ]
    (replay / "rows.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    (replay / "manifest.json").write_text(json.dumps({
        "schema_version": "1", "session_id": "person_test_01", "clip_id": "test_01",
        "split": "test", "clip_sha256": "testclip",
    }), encoding="utf-8")
    filter_path = tmp_path / "data" / "person_test_01" / "test_01.training_filter.json"
    filter_path.write_text(json.dumps({
        "schema_version": "1", "output_dir": str(replay), "labels_root": str(labels),
        "dataset": "test_01",
        "eligible": [
            {
                "index": index, "frame_key": f"g0_f{index + 1:08d}",
                "frame_id": index + 1, "reference_frame_id": index,
                "label": f"test_01/masks/g0_f{index + 1:08d}.png",
            }
            for index in range(3)
        ],
    }), encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.pt"
    _write_checkpoint(checkpoint)
    return {
        "replay": replay, "labels": labels, "filter": filter_path,
        "checkpoint": checkpoint, "root": tmp_path,
    }


def _run(fixture: dict, report_dir: Path, **overrides) -> dict:
    arguments = {
        "dataset": "test_01", "expected_split": "test", "report_dir": report_dir,
        "bootstrap": 50, "seed": 7, "device": "cpu", "log": lambda _message: None,
    }
    arguments.update(overrides)
    return module.run_offline(
        fixture["checkpoint"], fixture["replay"], fixture["labels"],
        fixture["filter"], **arguments,
    )


def test_offline_end_to_end_writes_the_report_and_overlays(
    fixture: dict, tmp_path: Path
) -> None:
    report_dir = tmp_path / "report"
    report = _run(fixture, report_dir)
    assert report["frames_scored"] == 3
    assert report["split"] == "test"
    assert report["session_id"] == "person_test_01"
    assert report["clip_sha256"] == "testclip"
    assert len(report["checkpoint_sha256"]) == 64
    assert report["mask_mode"] == "bounded_residual"
    assert report["data_integrity"]["mismatch_frames"] == 0
    for key in (
        "rules", "methods", "records", "limits", "scope_notes", "data_integrity",
        "checkpoint_sha256", "clip_sha256", "split", "frames_scored",
    ):
        assert key in report
    assert "baseline_mask" in report["rules"]["accepted_input"]
    method_names = {(entry["method"], entry["size"]) for entry in report["methods"]}
    assert ("candidate", "source") in method_names
    assert ("candidate_restricted", "source") in method_names
    assert ("candidate", "research") in method_names
    for entry in report["methods"]:
        if entry.get("reference") is not None:
            assert entry["paired"]["frames"] == 3
            assert entry["decision"]["outcome"] in ("offline_gate_passed", "offline_gate_failed")
    assert entry["decision"]["live_promoted"] is False
    record = report["records"][0]
    assert record["frame_key"] == "g0_f00000001"
    assert record["result"] in ("win", "loss", "tie")
    assert set(record["scores"]["source"]) == {
        "accepted", "candidate", "candidate_restricted", "checked",
    }
    assert record["scores"]["source"]["checked"]["iou"] == pytest.approx(1.0)
    assert (report_dir / "report.json").is_file()
    assert (report_dir / "index.html").is_file()
    overlays = sorted((report_dir / "overlays").glob("*.png"))
    assert overlays
    image = cv2.imread(str(overlays[0]))
    assert image.shape[0] == HEIGHT
    assert image.shape[1] == 4 * WIDTH
    written = json.loads((report_dir / "report.json").read_text(encoding="utf-8"))
    assert written["frames_scored"] == 3


def test_offline_refuses_a_report_dir_inside_the_replay(fixture: dict) -> None:
    with pytest.raises(ValueError, match="outside the replay data directory"):
        _run(fixture, fixture["replay"] / "report")


def test_offline_refuses_a_mismatched_sha256(fixture: dict, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not match the expected value"):
        _run(fixture, tmp_path / "report", expect_sha256="0" * 64)


def test_offline_accepts_the_correct_sha256(fixture: dict, tmp_path: Path) -> None:
    expected = module.sha256_file(fixture["checkpoint"])
    report = _run(fixture, tmp_path / "report", expect_sha256=expected.upper())
    assert report["checkpoint_sha256"] == expected


def test_offline_refuses_an_untrained_mask_head(fixture: dict, tmp_path: Path) -> None:
    model = PairedMaskMotionNet(base_channels=4, mask_mode="bounded_residual")
    untrained = tmp_path / "untrained.pt"
    save_checkpoint(
        untrained, task="paired_mask_motion_research", model=model,
        model_config={
            "in_channels": 15, "base_channels": 4, "mask_mode": "bounded_residual",
            "gate_radius_px": 12, "trained_heads": [],
            "session_id": "person_train_01", "clip_sha256": "trainclip",
        },
        epoch=1, global_step=1,
    )
    with pytest.raises(ValueError, match="mask head is not trained"):
        module.run_offline(
            untrained, fixture["replay"], fixture["labels"], fixture["filter"],
            dataset="test_01", expected_split="test",
            report_dir=tmp_path / "report", bootstrap=10, log=lambda _message: None,
        )


def test_offline_refuses_a_session_that_matches_the_training_data(
    fixture: dict, tmp_path: Path
) -> None:
    manifest_path = fixture["replay"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["session_id"] = "person_train_01"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="matches the training data"):
        _run(fixture, tmp_path / "report")


def test_offline_frames_filter_restricts_the_scored_set(
    fixture: dict, tmp_path: Path
) -> None:
    report = _run(fixture, tmp_path / "report", frames=["g0_f00000003", "g0_f00000001"])
    assert [record["frame_key"] for record in report["records"]] == [
        "g0_f00000003", "g0_f00000001",
    ]
    assert report["frames_scored"] == 2


def test_offline_overlay_cap_keeps_all_losses_first(
    fixture: dict, tmp_path: Path
) -> None:
    report = _run(fixture, tmp_path / "report", max_overlays=1)
    assert report["max_overlays"] == 1
    assert report["overlays_written"] <= 1


def test_offline_flags_a_disagreeing_accepted_mask(
    fixture: dict, tmp_path: Path
) -> None:
    """A saved accepted mask that the recompute does not reproduce is a warning."""
    replay = fixture["replay"]
    saved = cv2.imread(str(replay / "g0_f00000002_baseline_mask.png"), cv2.IMREAD_GRAYSCALE)
    # Overwrite the saved accepted mask of one frame with shifted pixels. The
    # stored mask is model-side data, so a disagreement is a data problem.
    shifted = np.roll(saved, 6, axis=1)
    cv2.imwrite(str(replay / "g0_f00000002_baseline_mask.png"), shifted)
    report = _run(fixture, tmp_path / "report")
    integrity = report["data_integrity"]
    assert integrity["mismatch_frames"] == 1
    assert integrity["mismatches"][0]["frame_key"] == "g0_f00000002"
    assert integrity["warnings"]
    assert any("does not agree" in warning for warning in integrity["warnings"])
    assert integrity["mismatches"][0]["agreement_stride4"] < 1.0


def test_restricted_candidate_uses_the_source_equivalent_radius() -> None:
    accepted = np.zeros((HEIGHT, WIDTH), bool)
    accepted[20:24, 30:34] = True
    candidate = np.ones((HEIGHT, WIDTH), bool)
    restricted, radii = module.candidate_restricted_mask(
        accepted, candidate, radius_px=12
    )
    radius_x = max(1, int(round(12 * WIDTH / 320)))
    radius_y = max(1, int(round(12 * HEIGHT / 180)))
    assert radii == (radius_x, radius_y)
    assert not restricted[0, 0]
    assert restricted[22, 32]
    assert restricted.sum() < candidate.sum()


def test_overlay_selection_puts_losses_first_then_wins_by_gain() -> None:
    def record(key: str, gain: float, result: str) -> dict:
        return {"frame_key": key, "gains": {"candidate": gain}, "result": result}

    records = [
        record("win_small", 0.01, "win"),
        record("loss_big", -0.10, "loss"),
        record("win_big", 0.20, "win"),
        record("tie", 0.0, "tie"),
        record("loss_small", -0.02, "loss"),
    ]
    chosen = module._overlay_selection(records, 10)
    assert [item["frame_key"] for item in chosen] == [
        "loss_big", "loss_small", "win_big", "win_small",
    ]
    assert [item["frame_key"] for item in module._overlay_selection(records, 2)] == [
        "loss_big", "loss_small",
    ]


def test_live_shadow_prints_the_command_without_a_log(tmp_path: Path) -> None:
    lines: list[str] = []
    status = module.run_live_shadow(
        tmp_path / "missing.jsonl", log=lines.append
    )
    assert status == 0
    text = "\n".join(lines)
    assert "scripts.run_integration_app" in text
    assert "--shadow-mask-checkpoint" in text
    assert "--shadow-mask-log" in text
    assert "--shadow-mask-sha256" in text
    assert "--duration-s" in text


def test_live_shadow_report_dir_rule_needs_a_replay_dir(tmp_path: Path) -> None:
    replay = tmp_path / "replay"
    replay.mkdir()
    with pytest.raises(ValueError, match="outside the replay data directory"):
        module.run_live_shadow(
            tmp_path / "missing.jsonl", report_dir=replay / "report",
            replay_dir=replay, log=lambda _message: None,
        )


def test_build_parser_has_both_subcommands() -> None:
    parser = module.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    args = parser.parse_args([
        "offline", "--checkpoint", "a.pt", "--replay-dir", "b",
        "--training-filter", "c.json", "--split", "test", "--report-dir", "d",
    ])
    assert args.command == "offline"
    assert args.bootstrap == module.DEFAULT_BOOTSTRAP
    assert args.max_overlays == module.DEFAULT_MAX_OVERLAYS
    shadow = parser.parse_args(["live-shadow", "--shadow-log", "e.jsonl"])
    assert shadow.command == "live-shadow"


def test_the_shadow_module_is_imported_only_inside_the_subcommand() -> None:
    """The scoring code must not import the live shadow module at import time."""
    import ast

    source = Path(module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    top_imports = [
        node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    names = "\n".join(ast.unparse(node) for node in top_imports)
    assert "paired_research.shadow" not in names
    nested = [
        node
        for node in ast.walk(tree)
        if (isinstance(node, ast.Import)
            and any(alias.name == "modules.inference.paired_research.shadow" for alias in node.names))
        or (isinstance(node, ast.ImportFrom)
            and node.module == "modules.inference.paired_research.shadow")
    ]
    assert nested
    for node in nested:
        assert node.col_offset > 0
