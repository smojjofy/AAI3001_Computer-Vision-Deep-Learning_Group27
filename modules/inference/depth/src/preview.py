from __future__ import annotations

import base64
import math

import cv2
import numpy as np

from .pipeline import PipelineResult
from .pose_depth import LANDMARK_NAMES, draw_pose_depth
from .utils import colorize

SCHEMA_VERSION = "1.0"


def _jpeg_base64(bgr: np.ndarray, quality: int = 75) -> str:
    ok, jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("could not encode pose/depth preview image")
    return base64.b64encode(jpeg).decode("ascii")


def _num(value: float, digits: int = 3) -> float | None:
    """JSON has no NaN: unknown values become null."""
    value = float(value)
    return round(value, digits) if math.isfinite(value) else None


def encode_pose_depth_preview(rgb: np.ndarray, result: PipelineResult, *, max_width: int = 640) -> dict:
    """Display-only preview of one pose + depth result. rgb: the frame it was computed on."""
    depth = result.depth
    valid = depth.valid
    # The colour scale: 2nd..98th percentile of valid depth, so one stray pixel doesn't squash it.
    if valid.any():
        near, far = (float(v) for v in np.percentile(depth.depth[valid], [2, 98]))
    else:
        near = far = float("nan")

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    depth_bgr = colorize(depth, max_depth_m=far if math.isfinite(far) else None)
    if depth_bgr.shape[:2] != bgr.shape[:2]:
        depth_bgr = cv2.resize(depth_bgr, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

    height, width = bgr.shape[:2]
    if width > max_width:                       # shrink first: cheaper to draw and to send
        size = (max_width, max(1, round(height * max_width / width)))
        bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
        depth_bgr = cv2.resize(depth_bgr, size, interpolation=cv2.INTER_NEAREST)

    joints = []
    if result.detected:
        for i, lm in enumerate(result.pose.landmarks):
            joints.append({
                "index": i,
                "name": LANDMARK_NAMES[i] if i < len(LANDMARK_NAMES) else str(i),
                "x": _num(lm.x, 4),
                "y": _num(lm.y, 4),
                "visibility": None if lm.visibility is None else _num(lm.visibility, 2),
                "depth_m": _num(result.joint_depth[i]),
            })

    return {
        "schema_version": SCHEMA_VERSION,
        "frame_id": result.frame_id,
        "source_id": result.source_id,
        "capture_timestamp_ns": str(result.capture_timestamp_ns),
        "width": int(bgr.shape[1]),
        "height": int(bgr.shape[0]),
        "depth_kind": depth.kind,               # "metric_depth" -> metres
        "depth_source": result.depth_source,    # "zed" or "yolo"
        "near_m": _num(near, 2),
        "far_m": _num(far, 2),
        "valid_fraction": _num(depth.valid_fraction, 3),
        "person_detected": result.detected,
        "body_depth_m": _num(result.body_depth, 2),
        "joints": joints,
        "timings_ms": {k: _num(v, 1) for k, v in result.timings_ms.items()},
        "depth_jpeg_base64": _jpeg_base64(depth_bgr),
        "skeleton_jpeg_base64": _jpeg_base64(draw_pose_depth(bgr, result.frame)),
    }