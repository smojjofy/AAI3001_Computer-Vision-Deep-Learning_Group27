"""Integration adapters between capture, depth, the constructor, and the viewer.

The frozen component modules use different depth conventions and identity types.
This module is the single place that converts between them:

- Metrics depth from YOLO is metres, camera-axis distance, larger = farther, and
  ``NaN`` where invalid.
- The constructor wants ``METRIC_CAMERA_Z`` depth in metres with finite values.
  Invalid samples are written as a finite placeholder and carried in
  ``depth_validity``.
- ``FramePacket`` uses integer frame IDs. ``ConstructionFrame`` needs a string.
  The integration convention is ``str(frame_id)`` with no padding, so the value
  round-trips to an integer without losing information. The source ID and the
  capture timestamp are preserved.

All functions are pure. They do not read files or networks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from modules.splat_constructor import (
    ConstructionConfig,
    ConstructionFrame,
    DepthSemantics,
    DepthUnits,
    SplatState,
)

# The constructor ignores any depth value at or beyond this placeholder when the
# matching depth_validity is 0, so the exact value only has to be finite and
# positive. It is set to the far end of the accepted metric range.
INVALID_DEPTH_PLACEHOLDER_M = 30.0

# Viewer transport array order. Each array is flattened row-major and packed back
# to back in this order. Keep this list in sync with the browser decoder.
TRANSPORT_ARRAYS: tuple[tuple[str, np.dtype], ...] = (
    ("positions", np.dtype("<f4")),
    ("sh_dc", np.dtype("<f4")),
    ("opacity_logits", np.dtype("<f4")),
    ("log_scales", np.dtype("<f4")),
    ("rotations", np.dtype("<f4")),
    ("reconstruction_weights", np.dtype("<f4")),
    ("source_pixels", np.dtype("<u4")),
)


def frame_id_string(frame_id: int) -> str:
    """Convert the capture integer frame ID to the constructor string ID."""
    if not isinstance(frame_id, int) or isinstance(frame_id, bool) or frame_id < 0:
        raise ValueError("frame_id must be a non-negative integer")
    return str(frame_id)


def webcam_intrinsics(
    width: int,
    height: int,
    horizontal_fov_deg: float = 60.0,
) -> tuple[float, float, float, float]:
    """Estimate pinhole intrinsics for a camera without calibration data.

    The constructor assumes square pixels and a principal point at
    ``((W-1)/2, (H-1)/2)`` when intrinsics come from a field of view, so this
    function keeps that half-pixel convention. The field of view is an assumption,
    not a calibration; state that limit in diagnostics.
    """
    if width < 1 or height < 1:
        raise ValueError("width and height must be positive")
    if not np.isfinite(horizontal_fov_deg) or not 1.0 < horizontal_fov_deg < 179.0:
        raise ValueError("horizontal_fov_deg must be in (1, 179)")
    fx = 0.5 * float(width) / np.tan(np.radians(horizontal_fov_deg) / 2.0)
    fx = float(fx)
    return (fx, fx, (width - 1) / 2.0, (height - 1) / 2.0)


def depth_to_metres(depth: np.ndarray, units: str | None) -> np.ndarray:
    """Convert a capture depth plane to float32 metres without changing validity."""
    if units not in ("millimetres", "metres"):
        raise ValueError("depth units must be 'millimetres' or 'metres'")
    values = np.asarray(depth, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("depth must be a 2-D array")
    if units == "millimetres":
        return values * np.float32(0.001)
    return values.copy()


@dataclass(frozen=True)
class DepthThresholdConfig:
    """Parameters of the depth-threshold person-mask experiment.

    The camera is static, so the foreground person is the nearest large region.
    The mask keeps valid depths inside a band above the closest valid depth.
    These values are starting settings. Change them only with recorded evidence.
    """

    near_percentile: float = 2.0
    band_m: float = 0.75
    near_margin_m: float = 0.20
    min_area_fraction: float = 0.005
    closing_px: int = 7
    fill_holes: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.near_percentile <= 50.0:
            raise ValueError("near_percentile must be in [0, 50]")
        if self.band_m <= 0.0:
            raise ValueError("band_m must be positive")
        if self.near_margin_m < 0.0:
            raise ValueError("near_margin_m must be non-negative")
        if not 0.0 <= self.min_area_fraction < 1.0:
            raise ValueError("min_area_fraction must be in [0, 1)")
        if self.closing_px < 1 or self.closing_px % 2 == 0:
            raise ValueError("closing_px must be an odd positive integer")


@dataclass(frozen=True)
class DepthThresholdMask:
    """The mask, its validity plane, and the measured values used to build them."""

    mask: np.ndarray
    validity: np.ndarray
    near_m: float
    far_m: float
    valid_fraction: float
    area_fraction: float
    confidence: float

    @property
    def detected(self) -> bool:
        return bool(self.mask.any()) and self.area_fraction > 0.0


def depth_threshold_mask(
    depth_m: np.ndarray,
    config: DepthThresholdConfig | None = None,
) -> DepthThresholdMask:
    """Build a person-mask candidate from metric depth alone.

    Steps: keep finite positive depths, find the closest valid depth percentile,
    keep a band above it, close small gaps, keep the largest connected region,
    and optionally fill holes. Returns zero mask when no region passes the area
    limit.

    The confidence value is a heuristic from the region area and the fraction of
    valid depth inside the mask. It is not a calibrated probability.
    """
    config = config or DepthThresholdConfig()
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError("depth_m must be a 2-D array")
    height, width = depth.shape
    validity = np.isfinite(depth) & (depth > 0.0)
    valid_fraction = float(validity.mean()) if validity.size else 0.0
    empty = np.zeros((height, width), dtype=np.uint8)
    if not validity.any():
        return DepthThresholdMask(empty, np.zeros((height, width), np.float32), np.nan, np.nan, 0.0, 0.0, 0.0)

    valid_values = depth[validity]
    near_m = float(np.percentile(valid_values, config.near_percentile))
    band = config.band_m
    if band <= 0.0:
        band = max(0.5, 0.25 * near_m)
    far_m = near_m + band
    band_mask = validity & (depth >= near_m - config.near_margin_m) & (depth <= far_m)

    binary = band_mask.astype(np.uint8)
    if config.closing_px > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.closing_px, config.closing_px))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return DepthThresholdMask(empty, np.zeros((height, width), np.float32), near_m, far_m, valid_fraction, 0.0, 0.0)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + int(np.argmax(areas))
    area = int(areas[largest - 1])
    keep = (labels == largest).astype(np.uint8)
    if config.fill_holes:
        filled = keep.copy()
        flood = np.zeros((height + 2, width + 2), np.uint8)
        cv2.floodFill(filled, flood, (0, 0), 1)
        keep = np.maximum(keep, 1 - filled)

    area_fraction = area / float(height * width)
    if area_fraction < config.min_area_fraction:
        return DepthThresholdMask(empty, np.zeros((height, width), np.float32), near_m, far_m, valid_fraction, area_fraction, 0.0)

    mask_bool = keep.astype(bool)
    inside = depth[mask_bool]
    valid_inside = float(np.isfinite(inside).mean()) if inside.size else 0.0
    # Heuristic confidence: large enough area and a high valid-depth fraction.
    confidence = float(np.clip(min(area_fraction / 0.15, 1.0) * valid_inside, 0.0, 1.0))
    validity_plane = (validity & mask_bool).astype(np.float32)
    return DepthThresholdMask(keep, validity_plane, near_m, far_m, valid_fraction, float(area_fraction), confidence)


def construction_frame_from_refresh(
    *,
    rgb: np.ndarray,
    depth_m: np.ndarray,
    foreground_mask: np.ndarray,
    frame_id: int,
    timestamp_ns: int,
    source_id: str,
    intrinsics: tuple[float, float, float, float] | None = None,
    depth_validity: np.ndarray | None = None,
    camera_to_world: np.ndarray | None = None,
    placeholder_m: float = INVALID_DEPTH_PLACEHOLDER_M,
) -> ConstructionFrame:
    """Build a validated metric ``ConstructionFrame`` from one refresh result.

    ``depth_m`` may contain ``NaN``. Invalid samples become a finite placeholder
    and get validity 0, because the native constructor rejects non-finite depth
    and non-finite validity planes.
    """
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must be a uint8 H x W x 3 array")
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.shape != rgb.shape[:2]:
        raise ValueError("depth must be aligned to the RGB frame")
    mask = np.asarray(foreground_mask)
    if mask.shape != rgb.shape[:2]:
        raise ValueError("foreground_mask must be aligned to the RGB frame")
    mask = (mask != 0).astype(np.uint8)

    height, width = depth.shape
    if intrinsics is None:
        intrinsics = webcam_intrinsics(width, height)
    if len(intrinsics) != 4 or not all(np.isfinite(float(value)) for value in intrinsics):
        raise ValueError("intrinsics must be four finite values")
    if float(intrinsics[0]) <= 0.0 or float(intrinsics[1]) <= 0.0:
        raise ValueError("fx and fy must be positive")

    valid = np.isfinite(depth) & (depth > 0.0)
    if depth_validity is None:
        validity_plane = valid.astype(np.float32)
    else:
        validity_plane = np.asarray(depth_validity, dtype=np.float32)
        if validity_plane.shape != depth.shape:
            raise ValueError("depth_validity must be aligned to the RGB frame")
        if not np.isfinite(validity_plane).all():
            raise ValueError("depth_validity must be finite")
        validity_plane = np.clip(validity_plane, 0.0, 1.0) * valid.astype(np.float32)

    safe_depth = np.where(valid, depth, np.float32(placeholder_m)).astype(np.float32)
    if not np.isfinite(safe_depth).all() or (safe_depth < 0.0).any():
        raise ValueError("metric depth must be finite and non-negative after conversion")

    pose = np.eye(4, dtype=np.float32) if camera_to_world is None else np.asarray(camera_to_world, dtype=np.float32)
    if pose.shape != (4, 4):
        raise ValueError("camera_to_world must be a 4x4 matrix")

    return ConstructionFrame(
        frame_id=frame_id_string(int(frame_id)),
        timestamp_ns=int(timestamp_ns),
        source_id=str(source_id),
        rgb=np.ascontiguousarray(rgb),
        depth=np.ascontiguousarray(safe_depth),
        foreground_mask=np.ascontiguousarray(mask),
        intrinsics=(float(intrinsics[0]), float(intrinsics[1]), float(intrinsics[2]), float(intrinsics[3])),
        depth_semantics=DepthSemantics.METRIC_CAMERA_Z,
        depth_units=DepthUnits.METERS,
        depth_validity=np.ascontiguousarray(validity_plane),
        camera_to_world=np.ascontiguousarray(pose),
    )


def splat_state_payload(state: SplatState) -> tuple[dict[str, Any], bytes]:
    """Serialize one ``SplatState`` for the browser renderer.

    Returns a JSON-safe header and one binary buffer. The header describes the
    buffer layout; the browser reconstructs typed arrays from it. Identity fields
    and the camera convention travel with the data.
    """
    if not isinstance(state, SplatState):
        raise TypeError("state must be a SplatState")
    header: dict[str, Any] = {
        "schema_version": "1.0",
        "frame_id": state.frame_id,
        "source_id": state.source_id,
        "timestamp_ns": int(state.timestamp_ns),
        "count": int(state.positions.shape[0]),
        "intrinsics": [float(value) for value in state.intrinsics],
        "depth_semantics": int(state.depth_semantics),
        "depth_units": int(state.depth_units),
        "camera_to_world": [float(value) for value in np.asarray(state.camera_to_world).reshape(-1)],
        "arrays": [],
        "coordinate_system": "right-handed, +X right, +Y up, camera forward -Z",
        "rotation_order": "wxyz",
        "scale_encoding": "natural_log",
        "opacity_encoding": "logit",
    }
    chunks: list[np.ndarray] = []
    offset = 0
    arrays = []
    for name, dtype in TRANSPORT_ARRAYS:
        values = np.ascontiguousarray(getattr(state, name), dtype=dtype).reshape(-1)
        chunk = values.tobytes(order="C")
        arrays.append(
            {
                "name": name,
                "dtype": str(dtype),
                "count": int(values.size),
                "byte_offset": offset,
                "byte_length": len(chunk),
            }
        )
        chunks.append(values)
        offset += len(chunk)
    header["arrays"] = arrays
    header["byte_length"] = offset
    buffer = b"".join(chunk.tobytes(order="C") for chunk in chunks)
    return header, buffer


def construct_splat_from_refresh(
    *,
    rgb: np.ndarray,
    depth_m: np.ndarray,
    foreground_mask: np.ndarray,
    frame_id: int,
    timestamp_ns: int,
    source_id: str,
    intrinsics: tuple[float, float, float, float] | None = None,
    depth_validity: np.ndarray | None = None,
    camera_to_world: np.ndarray | None = None,
    config: ConstructionConfig | None = None,
) -> SplatState:
    """Build the frame and call the native constructor in one step."""
    from modules.splat_constructor import construct_splat_state

    frame = construction_frame_from_refresh(
        rgb=rgb,
        depth_m=depth_m,
        foreground_mask=foreground_mask,
        frame_id=frame_id,
        timestamp_ns=timestamp_ns,
        source_id=source_id,
        intrinsics=intrinsics,
        depth_validity=depth_validity,
        camera_to_world=camera_to_world,
    )
    return construct_splat_state(frame, config or ConstructionConfig(sample_stride=2))
