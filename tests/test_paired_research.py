"""Check paired replay identity and channel meaning before model training."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest
import torch

from modules.inference.paired_research import PairedMaskMotionNet, PairedReplayDataset
from modules.inference.paired_research.augment import augment_pairs
from modules.inference.paired_research.evaluate import evaluate
from modules.inference.checkpoint import save_checkpoint


def _fixture(tmp_path):
    replay = tmp_path / "replay"
    labels = tmp_path / "labels"
    checked = labels / "clip" / "masks"
    replay.mkdir()
    checked.mkdir(parents=True)
    reference = np.zeros((8, 8, 3), np.uint8)
    reference[..., 2] = 255
    selected = np.zeros((8, 8, 3), np.uint8)
    selected[..., 0] = 255
    cv2.imwrite(str(replay / "g0_f00000001_rgb.png"), reference)
    cv2.imwrite(str(replay / "g0_f00000002_rgb.png"), selected)
    mask = np.zeros((8, 8), np.uint8)
    mask[:, :4] = 255
    cv2.imwrite(str(replay / "g0_f00000002_baseline_mask.png"), mask)
    cv2.imwrite(str(checked / "g0_f00000002.png"), mask)
    for frame_id, depth_m in ((1, 2.0), (2, 3.0)):
        depth = np.full((8, 8), depth_m, np.float32)
        validity = np.ones((8, 8), np.float32)
        if frame_id == 2:
            depth[0, 0] = np.nan
            validity[0, 0] = 0
        np.savez_compressed(replay / f"g0_f{frame_id:08d}_depth.npz", depth_m=depth, validity=validity)
        (replay / f"g0_f{frame_id:08d}_mask_pose.json").write_text(json.dumps({
            "frame_id": frame_id,
            "joints": [{"index": 0, "x": 0.5, "y": 0.5, "visibility": 0.9}],
        }), encoding="utf-8")
    flow = np.zeros((8, 8, 2), np.float32)
    flow[..., 0] = 8.0
    flow[..., 1] = 4.0
    np.savez_compressed(replay / "g0_f00000002_flow.npz", flow=flow, valid_mask=np.ones((8, 8), np.uint8))
    row = {
        "split": "train", "status": "ok", "selected_frame_id": 2,
        "reference_frame_id": 1, "selected_rgb": "g0_f00000002_rgb.png",
        "reference_rgb": "g0_f00000001_rgb.png",
        "selected_depth": "g0_f00000002_depth.npz",
        "reference_depth": "g0_f00000001_depth.npz",
        "mask_pose": "g0_f00000002_mask_pose.json",
        "flow": "g0_f00000002_flow.npz",
        "flow_metadata": {"previous_frame_id": 1, "current_frame_id": 2, "direction": "previous_to_current"},
        "baseline_mask": "g0_f00000002_baseline_mask.png",
    }
    (replay / "rows.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    (replay / "manifest.json").write_text(json.dumps({"split": "train", "clip_sha256": "abc"}), encoding="utf-8")
    filter_path = tmp_path / "filter.json"
    filter_path.write_text(json.dumps({
        "output_dir": str(replay), "labels_root": str(labels),
        "eligible": [{"index": 0, "frame_key": "g0_f00000002", "frame_id": 2,
                      "reference_frame_id": 1, "label": "clip/masks/g0_f00000002.png"}],
    }), encoding="utf-8")
    return replay, labels, filter_path, row


def test_paired_channels_and_model_output(tmp_path):
    replay, labels, filter_path, _ = _fixture(tmp_path)
    dataset = PairedReplayDataset(replay, labels, filter_path)
    inputs, target = dataset[0]
    assert inputs.shape == (15, 180, 320)
    assert target.shape == (1, 180, 320)
    assert torch.allclose(inputs[0], torch.ones_like(inputs[0]))  # reference red
    assert torch.allclose(inputs[5], torch.ones_like(inputs[5]))  # selected blue
    assert torch.allclose(inputs[6], torch.full_like(inputs[6], 0.2))
    assert inputs[9, 0, 0] == 0  # invalid selected depth
    assert torch.allclose(inputs[10], torch.ones_like(inputs[10]))
    assert torch.allclose(inputs[11], torch.full_like(inputs[11], 0.5))
    assert inputs[12].max() > 0 and inputs[13].max() > 0
    assert torch.equal(inputs[14:15], target)
    model = PairedMaskMotionNet(base_channels=4)
    with torch.inference_mode():
        mask_logits, residual = model(inputs.unsqueeze(0))
    assert mask_logits.shape == (1, 1, 180, 320)
    assert residual.shape == (1, 2, 180, 320)


def test_paired_loader_rejects_wrong_flow_identity(tmp_path):
    replay, labels, filter_path, row = _fixture(tmp_path)
    row["flow_metadata"]["previous_frame_id"] = 99
    (replay / "rows.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="flow frame identity"):
        PairedReplayDataset(replay, labels, filter_path)


def test_validation_evaluation_needs_separate_session(tmp_path):
    replay, labels, filter_path, row = _fixture(tmp_path)
    row["split"] = "validation"
    (replay / "rows.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    manifest = {"split": "validation", "session_id": "validation01", "clip_id": "clip", "clip_sha256": "newclip"}
    (replay / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    checkpoint = tmp_path / "model.pt"
    model = PairedMaskMotionNet(base_channels=4)
    save_checkpoint(checkpoint, task="paired_mask_motion_research", model=model,
                    model_config={"in_channels": 15, "base_channels": 4,
                                  "trained_heads": ["mask"], "session_id": "train01",
                                  "clip_sha256": "oldclip"}, epoch=1, global_step=1)
    report = evaluate(checkpoint, replay, labels, filter_path, expected_split="validation")
    assert report["frames_scored"] == 1
    assert report["summary"]["baseline"]["iou"]["mean"] == 1.0
    assert report["summary"]["source_baseline"]["iou"]["mean"] == 1.0
    assert 0.0 <= report["summary"]["source_model"]["iou"]["mean"] <= 1.0
    manifest["session_id"] = "train01"
    (replay / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="matches training data"):
        evaluate(checkpoint, replay, labels, filter_path, expected_split="validation")


def test_bounded_mask_starts_from_baseline_and_blocks_remote_regions():
    inputs = torch.zeros((1, 15, 32, 32))
    inputs[:, 14, 12:20, 12:20] = 1
    model = PairedMaskMotionNet(base_channels=4, mask_mode="bounded_residual", gate_radius_px=3)
    with torch.inference_mode():
        first, _ = model(inputs)
    assert torch.equal(first >= 0, inputs[:, 14:15] >= 0.5)
    with torch.no_grad():
        model.mask_head.bias.fill_(10.0)
    with torch.inference_mode():
        changed, _ = model(inputs)
    assert changed[0, 0, 0, 0] < 0  # no person can appear away from baseline
    assert changed[0, 0, 12, 10] > 0  # a nearby missing pixel can be added


def test_paired_augmentation_flips_flow_and_masks_together():
    inputs = torch.zeros((1, 15, 16, 16))
    inputs[:, 10] = 0.25
    inputs[:, 11] = 0.5
    inputs[:, 14, 3:9, 1:5] = 1
    target = inputs[:, 14:15].clone()
    changed, checked = augment_pairs(
        inputs, target, generator=torch.Generator().manual_seed(3),
        scale_range=(1.0, 1.0), shift_fraction=(0.0, 0.0),
        flip_probability=1.0, colour_gain_range=(1.0, 1.0),
    )
    assert torch.allclose(changed[:, 10], torch.full_like(changed[:, 10], -0.25))
    assert torch.allclose(changed[:, 11], torch.full_like(changed[:, 11], 0.5))
    assert torch.equal(changed[:, 14:15], checked)
    assert torch.equal(checked, torch.flip(target, dims=(-1,)))


def test_paired_augmentation_scales_flow_vectors():
    inputs = torch.zeros((1, 15, 16, 16))
    inputs[:, 10] = 0.2
    inputs[:, 11] = -0.1
    inputs[:, 14, 5:11, 5:11] = 1
    target = inputs[:, 14:15].clone()
    changed, checked = augment_pairs(
        inputs, target, generator=torch.Generator().manual_seed(5),
        scale_range=(2.0, 2.0), shift_fraction=(0.0, 0.0),
        flip_probability=0.0, colour_gain_range=(1.0, 1.0),
    )
    assert changed[0, 10, 8, 8] == pytest.approx(0.4)
    assert changed[0, 11, 8, 8] == pytest.approx(-0.2)
    assert torch.equal(changed[:, 14:15], checked)
