"""Run the temporary depth and segmentation checkpoints on one RGB image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from modules.inference.depth import DepthEstimator
from modules.inference.segmentation import SegmentationEstimator


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--depth-checkpoint", type=Path, required=True)
    parser.add_argument("--segmentation-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--invert-foreground-probability",
        action="store_true",
        help="Use 1 - foreground probability as a temporary polarity correction.",
    )
    parser.add_argument(
        "--use-foreground-as-depth",
        action="store_true",
        help="Export the effective foreground probability as relative inverse depth.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _byte_image(values: torch.Tensor) -> np.ndarray:
    return (values.clamp(0.0, 1.0).mul(255).round().byte().numpy())


def main() -> None:
    args = _arguments()
    args.output.mkdir(parents=True, exist_ok=True)

    with Image.open(args.image) as source:
        rgb_image = source.convert("RGB")
        original_width, original_height = rgb_image.size
        input_array = np.array(
            rgb_image.resize((args.width, args.height), Image.Resampling.BILINEAR),
            dtype=np.uint8,
            copy=True,
        )
        original_rgb = np.array(rgb_image, dtype=np.uint8, copy=True)

    tensor = torch.from_numpy(input_array).permute(2, 0, 1).float() / 255.0
    depth_estimator = DepthEstimator.from_checkpoint(
        str(args.depth_checkpoint), device=args.device
    )
    segmentation_estimator = SegmentationEstimator.from_checkpoint(
        str(args.segmentation_checkpoint), device=args.device, threshold=args.threshold
    )
    depth = depth_estimator.predict(tensor)
    segmentation = segmentation_estimator.predict(tensor)

    target_size = (original_height, original_width)
    relative_depth = F.interpolate(
        depth.relative_inverse_depth, size=target_size, mode="bilinear", align_corners=False
    )[0, 0]
    depth_confidence = F.interpolate(
        depth.confidence, size=target_size, mode="bilinear", align_corners=False
    )[0, 0]
    foreground_probability = F.interpolate(
        segmentation.foreground_probability,
        size=target_size,
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    segmentation_confidence = F.interpolate(
        segmentation.confidence, size=target_size, mode="bilinear", align_corners=False
    )[0, 0]
    if args.invert_foreground_probability:
        foreground_probability = 1.0 - foreground_probability
    if args.use_foreground_as_depth:
        relative_depth = foreground_probability
    mask = foreground_probability >= args.threshold

    depth_bytes = _byte_image(relative_depth)
    depth_confidence_bytes = _byte_image(depth_confidence)
    probability_bytes = _byte_image(foreground_probability)
    segmentation_confidence_bytes = _byte_image(segmentation_confidence)
    mask_bytes = mask.byte().mul(255).numpy()
    rgba = np.dstack((original_rgb, mask_bytes))

    Image.fromarray(depth_bytes).save(args.output / "relative_inverse_depth.png")
    Image.fromarray(depth_confidence_bytes).save(
        args.output / "depth_confidence.png"
    )
    Image.fromarray(probability_bytes).save(
        args.output / "foreground_probability.png"
    )
    Image.fromarray(segmentation_confidence_bytes).save(
        args.output / "segmentation_confidence.png"
    )
    Image.fromarray(mask_bytes).save(args.output / "foreground_mask.png")
    Image.fromarray(rgba).save(args.output / "rgba.png")

    metadata = {
        "source": str(args.image),
        "source_resolution": [original_width, original_height],
        "inference_resolution": [args.width, args.height],
        "depth_semantics": "relative_inverse_depth_near_is_bright",
        "depth_source": (
            "effective_foreground_probability"
            if args.use_foreground_as_depth
            else "depth_model"
        ),
        "depth_range": [float(relative_depth.min()), float(relative_depth.max())],
        "depth_confidence_range": [
            float(depth_confidence.min()),
            float(depth_confidence.max()),
        ],
        "segmentation_threshold": args.threshold,
        "foreground_probability_inverted": args.invert_foreground_probability,
        "foreground_probability_range": [
            float(foreground_probability.min()),
            float(foreground_probability.max()),
        ],
        "foreground_fraction": float(mask.float().mean()),
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
