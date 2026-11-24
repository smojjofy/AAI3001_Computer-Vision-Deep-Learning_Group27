"""Compare a fixed mask checkpoint with the baseline on a separate session."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import cv2
import torch
from torch.nn import functional as F

from modules.inference.checkpoint import load_checkpoint
from modules.inference.paired_research.data import PairedReplayDataset
from modules.inference.paired_research.model import PairedMaskMotionNet


def _read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"cannot read source mask: {path}")
    return mask >= 128


def _scores(predicted: np.ndarray, checked: np.ndarray) -> dict[str, float]:
    predicted = predicted.astype(bool)
    checked = checked.astype(bool)
    intersection = int(np.logical_and(predicted, checked).sum())
    union = int(np.logical_or(predicted, checked).sum())
    person = int(checked.sum())
    false_person = int(np.logical_and(predicted, ~checked).sum())
    return {
        "iou": round(intersection / union, 6) if union else 1.0,
        "person_coverage": round(intersection / person, 6) if person else 1.0,
        "background_leakage": round(false_person / person, 6) if person else float(false_person),
    }


def evaluate(
    checkpoint: Path, replay_dir: Path, labels_root: Path,
    filter_path: Path, *, expected_split: str, device: str = "cpu",
) -> dict:
    if expected_split not in ("validation", "test"):
        raise ValueError("evaluation needs a validation or test session")
    payload = load_checkpoint(checkpoint, expected_task="paired_mask_motion_research", map_location="cpu")
    config = payload["model_config"]
    if "mask" not in config.get("trained_heads", []):
        raise ValueError("checkpoint mask head is not trained")
    dataset = PairedReplayDataset(replay_dir, labels_root, filter_path, expected_split=expected_split)
    if (dataset.manifest["session_id"] == config.get("session_id")
            or dataset.manifest["clip_sha256"] == config.get("clip_sha256")):
        raise ValueError("evaluation session or clip matches training data")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    model = PairedMaskMotionNet(
        in_channels=config["in_channels"], base_channels=config["base_channels"],
        mask_mode=config.get("mask_mode", "absolute"),
        gate_radius_px=config.get("gate_radius_px", 12),
        residual_scale=config.get("residual_scale", 4.0),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    records = []
    with torch.inference_mode():
        for key, (inputs, target), row, label_path in zip(
            dataset.frame_keys, dataset.samples, dataset.rows, dataset.label_paths
        ):
            mask_logits, _ = model(inputs.unsqueeze(0).to(device))
            predicted = (torch.sigmoid(mask_logits)[0, 0].cpu().numpy() >= 0.5)
            checked = target[0].numpy() >= 0.5
            baseline = inputs[14].numpy() >= 0.5
            source_checked = _read_mask(label_path)
            source_baseline = _read_mask(dataset.replay_dir / row["baseline_mask"])
            if source_checked.shape != source_baseline.shape:
                raise ValueError(f"source mask dimensions do not match: {key}")
            source_logits = F.interpolate(
                mask_logits, size=source_checked.shape, mode="bilinear", align_corners=False
            )
            source_predicted = source_logits[0, 0].cpu().numpy() >= 0
            records.append({
                "frame_key": key,
                "model": _scores(predicted, checked),
                "baseline": _scores(baseline, checked),
                "source_model": _scores(source_predicted, source_checked),
                "source_baseline": _scores(source_baseline, source_checked),
            })
    summary = {}
    for name in ("model", "baseline", "source_model", "source_baseline"):
        summary[name] = {
            metric: {
                "mean": round(float(np.mean([r[name][metric] for r in records])), 6),
                "median": round(float(np.median([r[name][metric] for r in records])), 6),
            }
            for metric in ("iou", "person_coverage", "background_leakage")
        }
    return {
        "split": expected_split,
        "session_id": dataset.manifest["session_id"],
        "clip_id": dataset.manifest["clip_id"],
        "clip_sha256": dataset.manifest["clip_sha256"],
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_train_session": config["session_id"],
        "mask_mode": config.get("mask_mode", "absolute"),
        "training_augmentation": config.get("augmentation", "none"),
        "frames_scored": len(records),
        "threshold": 0.5,
        "source_upsample_rule": "bilinear logits, align_corners=False, threshold zero",
        "summary": summary,
        "records": records,
        "note": "Model and baseline are measured at 320x180 and source size. The motion head is untrained and is not measured.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--training-filter", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    report = evaluate(
        args.checkpoint, args.replay_dir, args.labels_root,
        args.training_filter, expected_split=args.split, device=args.device,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"scored {report['frames_scored']} {args.split} frames: {args.report}")


if __name__ == "__main__":
    main()
