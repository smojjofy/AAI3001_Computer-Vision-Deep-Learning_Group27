"""Inspect a standard binary little-endian 3DGS PLY without external PLY tools."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ply", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    properties: list[str] = []
    vertex_count: int | None = None
    with args.ply.open("rb") as stream:
        while True:
            raw_line = stream.readline()
            if not raw_line:
                raise ValueError("PLY header ended before end_header")
            line = raw_line.decode("ascii").strip()
            if line == "format binary_little_endian 1.0":
                pass
            elif line.startswith("element vertex "):
                vertex_count = int(line.split()[-1])
            elif line.startswith("property float "):
                properties.append(line.split()[-1])
            elif line == "end_header":
                break
        if vertex_count is None:
            raise ValueError("PLY has no vertex count")
        values = np.fromfile(stream, dtype="<f4", count=vertex_count * len(properties))

    expected = vertex_count * len(properties)
    if values.size != expected:
        raise ValueError(f"truncated PLY: expected {expected} float values, got {values.size}")
    vertices = values.reshape(vertex_count, len(properties))
    index = {name: offset for offset, name in enumerate(properties)}

    def value_range(names: list[str]) -> list[float]:
        selected = vertices[:, [index[name] for name in names]]
        return [float(selected.min()), float(selected.max())]

    def component_ranges(names: list[str]) -> dict[str, list[float]]:
        return {name: value_range([name]) for name in names}

    rotation = vertices[:, [index[f"rot_{component}"] for component in range(4)]]

    report = {
        "path": str(args.ply),
        "vertex_count": vertex_count,
        "property_count": len(properties),
        "all_finite": bool(np.isfinite(vertices).all()),
        "position_ranges": component_ranges(["x", "y", "z"]),
        "opacity_logit_range": value_range(["opacity"]),
        "log_scale_ranges": component_ranges(["scale_0", "scale_1", "scale_2"]),
        "rotation_norm_range": [
            float(np.linalg.norm(rotation, axis=1).min()),
            float(np.linalg.norm(rotation, axis=1).max()),
        ],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
