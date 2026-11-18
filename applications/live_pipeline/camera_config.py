"""Per-camera pinhole intrinsics loaded from a JSON file.

The pipeline uses capture intrinsics when a source supplies them, otherwise
estimated webcam intrinsics. This file lets a known camera supply calibrated
pinhole values without a code change. Only the ``pinhole`` model is supported.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from .adapters import webcam_intrinsics


@dataclass(frozen=True)
class CameraConfig:
    """Validated pinhole intrinsics for one capture resolution."""

    width: int
    height: int
    intrinsics: tuple[float, float, float, float]
    source: str = "file"

    def matches(self, width: int, height: int) -> bool:
        """True when this calibration applies to the frame size."""
        return int(width) == self.width and int(height) == self.height


def _positive_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return number


def load_camera_config(path: str | Path) -> CameraConfig:
    """Read and validate a pinhole intrinsics file.

    Required: ``model`` = "pinhole", ``width``, ``height``, and either the four
    values ``fx``, ``fy``, ``cx``, ``cy`` or ``horizontal_fov_deg``.
    """
    config_path = Path(path)
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as error:
        raise ValueError(f"camera config is not valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("camera config must be a JSON object")

    model = payload.get("model", "pinhole")
    if model != "pinhole":
        raise ValueError(f"unsupported camera model {model!r}; only 'pinhole' is supported")

    width = payload.get("width")
    height = payload.get("height")
    if isinstance(width, bool) or not isinstance(width, int) or width < 1:
        raise ValueError("width must be a positive integer")
    if isinstance(height, bool) or not isinstance(height, int) or height < 1:
        raise ValueError("height must be a positive integer")

    if "horizontal_fov_deg" in payload:
        fov = _positive_float(payload["horizontal_fov_deg"], "horizontal_fov_deg")
        if fov >= 180.0:
            raise ValueError("horizontal_fov_deg must be less than 180")
        intrinsics = webcam_intrinsics(width, height, fov)
    else:
        fx = _positive_float(payload.get("fx"), "fx")
        fy = _positive_float(payload.get("fy"), "fy")
        cx = payload.get("cx")
        cy = payload.get("cy")
        if isinstance(cx, bool) or not isinstance(cx, (int, float)):
            raise ValueError("cx must be a number")
        if isinstance(cy, bool) or not isinstance(cy, (int, float)):
            raise ValueError("cy must be a number")
        if not 0.0 <= float(cx) <= width or not 0.0 <= float(cy) <= height:
            raise ValueError("principal point must lie inside the image")
        intrinsics = (fx, fy, float(cx), float(cy))

    return CameraConfig(width=width, height=height, intrinsics=intrinsics, source=str(config_path))
