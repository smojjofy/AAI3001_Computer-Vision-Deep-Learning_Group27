"""Train the mask head of the paired research model on checked replay masks.

This is a train-session diagnostic. It does not validate or enable a live model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from modules.inference.checkpoint import save_checkpoint
from modules.inference.paired_research.augment import augment_pairs
from modules.inference.paired_research.data import CHANNEL_ORDER, HEIGHT, WIDTH, PairedReplayDataset
from modules.inference.paired_research.model import PairedMaskMotionNet
from modules.inference.segmentation.src.loss import segmentation_loss


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--training-filter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--base-channels", type=int, default=12)
    parser.add_argument("--mask-mode", choices=("absolute", "bounded_residual"), default="absolute")
    parser.add_argument("--augmentation", choices=("none", "paired"), default="none")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def _label_manifest_id(dataset: PairedReplayDataset, filter_path: Path) -> str:
    filtered = json.loads(filter_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(dataset.manifest["clip_sha256"].encode("ascii"))
    for entry in filtered["eligible"]:
        digest.update(entry["frame_key"].encode("ascii"))
        digest.update((dataset.labels_root / entry["label"]).read_bytes())
    return digest.hexdigest()


def _iou(logits: torch.Tensor, target: torch.Tensor) -> float:
    predicted = torch.sigmoid(logits) >= 0.5
    actual = target >= 0.5
    intersection = (predicted & actual).flatten(1).sum(dim=1).float()
    union = (predicted | actual).flatten(1).sum(dim=1).float()
    return float(((intersection + 1) / (union + 1)).mean().item())


def train(args: argparse.Namespace) -> dict:
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise ValueError("epochs, batch size, and learning rate must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
    torch.set_num_threads(min(4, torch.get_num_threads()))

    dataset = PairedReplayDataset(args.replay_dir, args.labels_root, args.training_filter)
    data_id = _label_manifest_id(dataset, args.training_filter)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(args.seed),
    )
    model = PairedMaskMotionNet(base_channels=args.base_channels, mask_mode=args.mask_mode).to(args.device)
    augment_generator = torch.Generator().manual_seed(args.seed + 1)
    for parameter in model.motion_head.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    history = []
    global_step = 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = bce_total = dice_total = 0.0
        for inputs, target in loader:
            if args.augmentation == "paired":
                inputs, target = augment_pairs(inputs, target, generator=augment_generator)
            inputs = inputs.to(args.device)
            target = target.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            mask_logits, _ = model(inputs)
            loss, parts = segmentation_loss(mask_logits, target)
            loss.backward()
            optimizer.step()
            count = inputs.shape[0]
            total += float(loss.item()) * count
            bce_total += float(parts["bce"].item()) * count
            dice_total += float(parts["dice"].item()) * count
            global_step += 1
        model.eval()
        iou_total = baseline_total = 0.0
        with torch.inference_mode():
            for inputs, target in DataLoader(dataset, batch_size=args.batch_size, num_workers=0):
                inputs = inputs.to(args.device)
                target = target.to(args.device)
                mask_logits, _ = model(inputs)
                count = inputs.shape[0]
                iou_total += _iou(mask_logits, target) * count
                baseline_logits = (inputs[:, 14:15] - 0.5) * 20.0
                baseline_total += _iou(baseline_logits, target) * count
        record = {
            "epoch": epoch,
            "train_loss": round(total / len(dataset), 6),
            "train_bce": round(bce_total / len(dataset), 6),
            "train_dice_loss": round(dice_total / len(dataset), 6),
            "train_mask_iou": round(iou_total / len(dataset), 6),
            "train_baseline_iou": round(baseline_total / len(dataset), 6),
            "elapsed_s": round(time.perf_counter() - started, 2),
        }
        history.append(record)
        print(json.dumps(record), flush=True)

    final = history[-1]
    per_frame = []
    with torch.inference_mode():
        for key, (inputs, target) in zip(dataset.frame_keys, dataset.samples):
            inputs = inputs.unsqueeze(0).to(args.device)
            target = target.unsqueeze(0).to(args.device)
            mask_logits, _ = model(inputs)
            per_frame.append({
                "frame_key": key,
                "train_mask_iou": round(_iou(mask_logits, target), 6),
                "train_baseline_iou": round(_iou((inputs[:, 14:15] - 0.5) * 20.0, target), 6),
            })
    config = {
        "in_channels": 15,
        "base_channels": args.base_channels,
        "mask_mode": args.mask_mode,
        "gate_radius_px": model.gate_radius_px,
        "residual_scale": model.residual_scale,
        "augmentation": args.augmentation,
        "width": WIDTH,
        "height": HEIGHT,
        "channel_order": list(CHANNEL_ORDER),
        "depth_rule": "clip valid metres to [0,10], divide by 10; invalid zero with validity channel",
        "flow_rule": "reference_to_selected pixels divided by source width and height; invalid zero",
        "pose_rule": "exact mask-input joints, visibility >= 0.15, radius 2 at 320x180",
        "target_rule": "checked binary person mask resized nearest",
        "trained_heads": ["mask"],
        "untrained_heads": ["motion"],
        "evaluation_status": "train_only_no_validation",
        "session_id": dataset.manifest["session_id"],
        "clip_id": dataset.manifest["clip_id"],
        "clip_sha256": dataset.manifest["clip_sha256"],
        "training_data_manifest_id": data_id,
    }
    save_checkpoint(
        args.output, task="paired_mask_motion_research", model=model,
        model_config=config, epoch=args.epochs, global_step=global_step,
        optimizer=optimizer,
        metrics={"train_loss": final["train_loss"], "train_mask_iou": final["train_mask_iou"], "train_baseline_iou": final["train_baseline_iou"]},
    )
    report = {
        "status": "train_only_no_validation",
        "session_id": dataset.manifest["session_id"],
        "clip_id": dataset.manifest["clip_id"],
        "training_pairs": len(dataset),
        "frame_keys": dataset.frame_keys,
        "training_data_manifest_id": data_id,
        "trained_heads": ["mask"],
        "untrained_heads": ["motion"],
        "checkpoint": str(args.output.resolve()),
        "device": args.device,
        "mask_mode": args.mask_mode,
        "augmentation": args.augmentation,
        "peak_cuda_memory_mb": round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1) if args.device == "cuda" else None,
        "history": history,
        "per_frame_train": per_frame,
        "note": "All measurements use the training session. The checkpoint is not approved for live use.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    args = _parser().parse_args()
    report = train(args)
    print(f"saved train-only checkpoint: {report['checkpoint']}")


if __name__ == "__main__":
    main()
