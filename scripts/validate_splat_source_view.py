"""Construct and validate one aligned RGB/depth/mask frame through the native bridge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from modules.reconstruction_validation import validate_source_view
from modules.splat_constructor import (
    ConstructionConfig,
    ConstructionFrame,
    DepthSemantics,
    DepthUnits,
    construct_splat_state,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument("--depth", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--frame-id", default="0001")
    parser.add_argument("--source-id", default="0001")
    parser.add_argument("--timestamp-ns", type=int, required=True)
    parser.add_argument("--fov", type=float, default=50.0)
    parser.add_argument("--near", type=float, default=1.0)
    parser.add_argument("--far", type=float, default=4.0)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--mask-threshold", type=int, default=128)
    parser.add_argument(
        "--depth-semantics",
        choices=("relative_inverse", "pseudo_inverse_from_segmentation", "metric_camera_z"),
        default="relative_inverse",
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=1.0 / 255.0,
        help="Multiply grayscale depth by this before construction; use metric units for metric camera-Z.",
    )
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    rgb = np.asarray(Image.open(args.rgb).convert("RGB"), dtype=np.uint8)
    depth = np.asarray(Image.open(args.depth).convert("F"), dtype=np.float32) * args.depth_scale
    mask = (
        np.asarray(Image.open(args.mask).convert("L"), dtype=np.uint8) >= args.mask_threshold
    ).astype(np.uint8)
    height, width = depth.shape
    if rgb.shape[:2] != (height, width) or mask.shape != (height, width):
        raise ValueError("RGB, depth, and mask must have identical dimensions")
    focal = width / (2.0 * np.tan(np.deg2rad(args.fov) / 2.0))
    semantics = DepthSemantics[args.depth_semantics.upper()]
    units = DepthUnits.METERS if semantics == DepthSemantics.METRIC_CAMERA_Z else DepthUnits.UNITLESS
    frame = ConstructionFrame(
        frame_id=args.frame_id,
        timestamp_ns=args.timestamp_ns,
        source_id=args.source_id,
        rgb=rgb,
        depth=depth,
        foreground_mask=mask,
        intrinsics=(float(focal), float(focal), (width - 1.0) / 2.0, (height - 1.0) / 2.0),
        depth_semantics=semantics,
        depth_units=units,
    )
    config = ConstructionConfig(
        sample_stride=args.stride, near_depth=args.near, far_depth=args.far
    )
    state = construct_splat_state(frame, config)
    report = validate_source_view(frame, state, config)
    print(json.dumps({
        "gaussian_count": int(state.opacity_logits.size),
        "center_reprojection_rmse_px": report.center_reprojection_rmse_px,
        "covered_rgb_mae": report.covered_rgb_mae,
        "covered_depth_mae": report.covered_depth_mae,
        "mask_iou": report.mask_iou,
        "foreground_coverage": report.foreground_coverage,
    }, indent=2))


if __name__ == "__main__":
    main()
