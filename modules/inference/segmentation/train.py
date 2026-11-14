"""Train the temporary foreground segmentation baseline on the APC sample."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from modules.inference.apc_dataset import APCTrainingSample
from modules.inference.checkpoint import save_checkpoint
from modules.inference.segmentation.src.loss import segmentation_loss
from modules.inference.segmentation.src.model import TinySegmentationNet


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("checkpoints/segmentation_baseline.pt")
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = APCTrainingSample(
        args.data,
        image_size=(args.height, args.width),
        augment=True,
        max_samples=args.max_samples,
    )
    generator = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(len(dataset), generator=generator).tolist()
    validation_count = max(1, round(len(order) * 0.2))
    train_loader = DataLoader(
        Subset(dataset, order[validation_count:]), batch_size=args.batch_size, shuffle=True
    )
    validation_loader = DataLoader(
        Subset(dataset, order[:validation_count]), batch_size=args.batch_size
    )

    model_config = {"in_channels": 3, "base_channels": args.base_channels}
    model = TinySegmentationNet(**model_config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_total = 0.0
        for batch in train_loader:
            image = batch["image"].to(args.device)
            target = batch["foreground_mask"].to(args.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(image)
            loss, _ = segmentation_loss(logits, target)
            loss.backward()
            optimizer.step()
            train_total += loss.item() * image.shape[0]
            global_step += 1

        model.eval()
        validation_total = 0.0
        validation_items = 0
        with torch.inference_mode():
            for batch in validation_loader:
                image = batch["image"].to(args.device)
                target = batch["foreground_mask"].to(args.device)
                loss, _ = segmentation_loss(model(image), target)
                validation_total += loss.item() * image.shape[0]
                validation_items += image.shape[0]

        metrics = {
            "train_loss": train_total / len(train_loader.dataset),
            "validation_loss": validation_total / validation_items,
        }
        save_checkpoint(
            args.output,
            task="segmentation",
            model=model,
            model_config=model_config,
            epoch=epoch,
            global_step=global_step,
            optimizer=optimizer,
            metrics=metrics,
        )
        print(f"epoch={epoch} train={metrics['train_loss']:.4f} val={metrics['validation_loss']:.4f}")


if __name__ == "__main__":
    main()
