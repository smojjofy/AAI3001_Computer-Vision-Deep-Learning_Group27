"""Convert aligned PNG/JPEG constructor inputs to dependency-free PPM/PGM files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument("--depth", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    args.output.mkdir(parents=True, exist_ok=True)
    with Image.open(args.rgb) as source:
        rgb = source.convert("RGB")
    with Image.open(args.depth) as source:
        depth = source.convert("L")
    with Image.open(args.mask) as source:
        mask = source.convert("L")

    if rgb.size != depth.size or rgb.size != mask.size:
        raise ValueError(
            f"unaligned inputs: RGB={rgb.size}, depth={depth.size}, mask={mask.size}"
        )
    binary_mask = np.where(np.asarray(mask) >= 128, 255, 0).astype(np.uint8)
    rgb.save(args.output / "rgb.ppm", format="PPM")
    depth.save(args.output / "depth.pgm", format="PPM")
    Image.fromarray(binary_mask).save(args.output / "mask.pgm", format="PPM")
    print(f"Prepared {rgb.width}x{rgb.height} fixture at {args.output}")


if __name__ == "__main__":
    main()
