"""Deterministic source-camera reprojection validation.

This is intentionally a point-center rasterizer, not a production 3D Gaussian
rasterizer. It verifies constructor geometry and source association without
coupling the native constructor to a renderer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from modules.splat_constructor import (
    ConstructionConfig,
    ConstructionFrame,
    DepthSemantics,
    SplatState,
)


_SH_C0 = np.float32(0.28209479177387814)


@dataclass(frozen=True)
class SourceViewReport:
    rgb: NDArray[np.float32]
    alpha: NDArray[np.float32]
    depth: NDArray[np.float32]
    coverage: NDArray[np.bool_]
    center_reprojection_rmse_px: float
    covered_rgb_mae: float
    covered_depth_mae: float
    mask_iou: float
    foreground_coverage: float


def _camera_points(state: SplatState) -> NDArray[np.float32]:
    pose = state.camera_to_world
    rotation = pose[:3, :3]
    translation = pose[:3, 3]
    return (state.positions - translation) @ rotation


def _expected_distance(
    frame: ConstructionFrame, config: ConstructionConfig
) -> NDArray[np.float32]:
    if frame.depth_semantics == DepthSemantics.METRIC_CAMERA_Z:
        return frame.depth
    return config.near_depth + (1.0 - frame.depth) * (config.far_depth - config.near_depth)


def validate_source_view(
    frame: ConstructionFrame,
    state: SplatState,
    config: ConstructionConfig = ConstructionConfig(),
) -> SourceViewReport:
    """Project a state into its originating camera and return quantitative checks."""

    if (frame.frame_id, frame.timestamp_ns, frame.source_id) != (
        state.frame_id, state.timestamp_ns, state.source_id
    ):
        raise ValueError("frame and state provenance must match for source-view validation")
    height, width = frame.depth.shape
    if frame.rgb.shape != (height, width, 3):
        raise ValueError("ConstructionFrame RGB and depth must be aligned")
    fx, fy, cx, cy = state.intrinsics
    camera = _camera_points(state)
    distance = -camera[:, 2]
    valid = distance > 0.0
    projected_u = fx * camera[:, 0] / np.maximum(distance, 1.0e-8) + cx
    projected_v = -fy * camera[:, 1] / np.maximum(distance, 1.0e-8) + cy
    source_u = state.source_pixels[:, 0].astype(np.float32)
    source_v = state.source_pixels[:, 1].astype(np.float32)
    reprojection_squared = (projected_u - source_u) ** 2 + (projected_v - source_v) ** 2
    rmse = float(np.sqrt(np.mean(reprojection_squared[valid]))) if valid.any() else float("inf")

    pixel_u = np.rint(projected_u).astype(np.int64)
    pixel_v = np.rint(projected_v).astype(np.int64)
    valid &= (pixel_u >= 0) & (pixel_u < width) & (pixel_v >= 0) & (pixel_v < height)
    rgb = np.zeros((height, width, 3), dtype=np.float32)
    alpha = np.zeros((height, width), dtype=np.float32)
    depth = np.zeros((height, width), dtype=np.float32)
    coverage = np.zeros((height, width), dtype=bool)
    if valid.any():
        indices = np.flatnonzero(valid)
        linear_pixels = pixel_v[indices] * width + pixel_u[indices]
        order = np.lexsort((distance[indices], linear_pixels))
        sorted_indices = indices[order]
        sorted_pixels = linear_pixels[order]
        winners = np.r_[True, sorted_pixels[1:] != sorted_pixels[:-1]]
        winners = sorted_indices[winners]
        winner_u = pixel_u[winners]
        winner_v = pixel_v[winners]
        rgb[winner_v, winner_u] = np.clip(state.sh_dc[winners] * _SH_C0 + 0.5, 0.0, 1.0)
        alpha[winner_v, winner_u] = 1.0 / (1.0 + np.exp(-state.opacity_logits[winners]))
        depth[winner_v, winner_u] = distance[winners]
        coverage[winner_v, winner_u] = True

    target_rgb = frame.rgb.astype(np.float32) / 255.0
    expected_depth = _expected_distance(frame, config)
    covered_rgb_mae = float(np.abs(rgb[coverage] - target_rgb[coverage]).mean()) if coverage.any() else float("inf")
    covered_depth_mae = float(np.abs(depth[coverage] - expected_depth[coverage]).mean()) if coverage.any() else float("inf")
    foreground = frame.foreground_mask.astype(bool)
    intersection = np.logical_and(coverage, foreground).sum()
    union = np.logical_or(coverage, foreground).sum()
    mask_iou = float(intersection / union) if union else 1.0
    foreground_count = foreground.sum()
    foreground_coverage = float(intersection / foreground_count) if foreground_count else 1.0
    return SourceViewReport(
        rgb=rgb,
        alpha=alpha,
        depth=depth,
        coverage=coverage,
        center_reprojection_rmse_px=rmse,
        covered_rgb_mae=covered_rgb_mae,
        covered_depth_mae=covered_depth_mae,
        mask_iou=mask_iou,
        foreground_coverage=foreground_coverage,
    )
