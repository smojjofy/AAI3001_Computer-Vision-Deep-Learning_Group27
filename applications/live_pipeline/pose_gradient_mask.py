"""Pose-seeded watershed mask on the depth gradient.

The person's silhouette is a depth discontinuity. Watershed on the depth-gradient
image, with the pose joints as foreground seeds and the area outside a pose box
as background, finds the region that wraps the pose. This does not use a mean
depth threshold, so a room that is closer than the person cannot invert the mask.

No model or camera dependency; the caller supplies a metric depth plane and pose
joints in normalized image coordinates.
"""

from __future__ import annotations

import cv2
import numpy as np

MIN_JOINT_VISIBILITY = 0.15
# Joint index groups used to fill the head and the hands with hull regions.
HEAD_JOINTS = tuple(range(0, 13))
LEFT_HAND_JOINTS = (15, 17, 19, 21)
RIGHT_HAND_JOINTS = (16, 18, 20, 22)


def _fill_invalid(depth: np.ndarray) -> np.ndarray:
    depth = depth.astype(np.float32)
    invalid = ~np.isfinite(depth) | (depth <= 0.0)
    if not invalid.any():
        return depth
    if not np.isfinite(depth).any():
        return np.zeros_like(depth)
    filled = depth.copy()
    filled[invalid] = 0.0
    for _ in range(3):
        blurred = cv2.medianBlur(filled, 5)
        filled[invalid] = blurred[invalid]
    return filled


def joint_hulls(
    joints: list[dict],
    groups: tuple[tuple[int, ...], ...],
    *,
    width: int,
    height: int,
) -> np.ndarray:
    """Union of convex hulls for groups of joint indices."""
    output = np.zeros((height, width), np.uint8)
    for group in groups:
        points: list[tuple[int, int]] = []
        for joint in joints:
            index = joint.get("index")
            if index not in group:
                continue
            visibility = joint.get("visibility")
            if visibility is not None and float(visibility) < MIN_JOINT_VISIBILITY:
                continue
            x = joint.get("x")
            y = joint.get("y")
            if x is None or y is None:
                continue
            px = int(round(float(x) * (width - 1)))
            py = int(round(float(y) * (height - 1)))
            if 0 <= px < width and 0 <= py < height:
                points.append((px, py))
        if len(points) >= 3:
            hull = cv2.convexHull(np.array(points, dtype=np.int32))
            cv2.fillConvexPoly(output, hull, 1)
    return output


def pose_seed_mask(
    joints: list[dict],
    *,
    width: int,
    height: int,
    radius: int = 3,
) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
    """Foreground seeds at visible joints, plus their bounding box."""
    seeds = np.zeros((height, width), np.uint8)
    xs: list[float] = []
    ys: list[float] = []
    for joint in joints:
        visibility = joint.get("visibility")
        if visibility is not None and float(visibility) < MIN_JOINT_VISIBILITY:
            continue
        x = joint.get("x")
        y = joint.get("y")
        if x is None or y is None:
            continue
        try:
            fx = float(x)
            fy = float(y)
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(fx) and np.isfinite(fy)):
            continue
        xs.append(fx)
        ys.append(fy)
        px = int(round(fx * (width - 1)))
        py = int(round(fy * (height - 1)))
        if 0 <= px < width and 0 <= py < height:
            cv2.circle(seeds, (px, py), max(1, int(radius)), 1, -1)
    if not xs:
        return seeds, None
    return seeds, (min(xs), min(ys), max(xs), max(ys))


def joint_depth_range(
    joints: list[dict] | None,
    *,
    margin_m: float = 0.35,
    low_percentile: float = 10.0,
    high_percentile: float = 90.0,
) -> tuple[float, float] | None:
    """Depth range covered by the visible joints, widened by a margin."""
    depths: list[float] = []
    for joint in joints or []:
        visibility = joint.get("visibility")
        if visibility is not None and float(visibility) < MIN_JOINT_VISIBILITY:
            continue
        depth = joint.get("depth_m")
        if depth is None:
            continue
        try:
            value = float(depth)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value) and value > 0.0:
            depths.append(value)
    if not depths:
        return None
    low = float(np.percentile(depths, low_percentile)) - float(margin_m)
    high = float(np.percentile(depths, high_percentile)) + float(margin_m)
    return max(0.0, low), high


def pose_watershed_mask(
    depth_m: np.ndarray,
    joints: list[dict] | None,
    *,
    pose_margin: float = 0.12,
    gradient_percentile: float = 97.0,
    min_contrast_m: float = 0.15,
    min_area_fraction: float = 0.005,
    max_area_fraction: float = 0.60,
) -> tuple[np.ndarray, str]:
    """Return a person mask that wraps the pose, and a reason.

    ``reason`` is ``ok``, ``no_pose``, ``too_small``, ``too_large``, or
    ``no_region``.
    """
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError("depth_m must be a 2-D array")
    height, width = depth.shape
    empty = np.zeros((height, width), np.uint8)
    if not joints:
        return empty, "no_pose"

    seeds, bounds = pose_seed_mask(joints, width=width, height=height)
    if bounds is None or not seeds.any():
        return empty, "no_pose"

    filled = _fill_invalid(depth)
    gx = cv2.Sobel(filled, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(filled, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    gradient = np.hypot(gx, gy)

    min_x, min_y, max_x, max_y = bounds
    margin_x = max(4, int(round(pose_margin * width)))
    margin_y = max(4, int(round(pose_margin * height)))
    left = max(0, int(min_x * width) - margin_x)
    right = min(width, int(max_x * width) + margin_x)
    top = max(0, int(min_y * height) - margin_y)
    bottom = min(height, int(max_y * height) + margin_y)
    if right <= left or bottom <= top:
        return empty, "no_region"

    # Without any depth discontinuity in the pose box there is no silhouette to
    # find. A flat plane would otherwise be split arbitrarily by watershed.
    box_depths = filled[top:bottom, left:right]
    box_valid = box_depths[np.isfinite(box_depths) & (box_depths > 0.0)]
    if box_valid.size < 16:
        return empty, "no_region"
    contrast = float(np.percentile(box_valid, 95) - np.percentile(box_valid, 5))
    if contrast < float(min_contrast_m):
        return empty, "flat_depth"

    markers = np.zeros((height, width), np.int32)
    markers[seeds.astype(bool)] = 1
    box = np.zeros((height, width), np.uint8)
    box[top:bottom, left:right] = 1
    markers[box == 0] = 2
    markers[0, :] = 2
    markers[-1, :] = 2
    markers[:, 0] = 2
    markers[:, -1] = 2

    # A stride-4 subsample estimates the full-image percentile. Measured against
    # the full image on three real frames, the resulting mask differs by about
    # 1 percent (intersection over union 0.988 to 0.998) at about a quarter of
    # the cost. It is an estimate, not the identical statistic.
    sample = gradient[::4, ::4]
    scale = max(1e-3, float(np.percentile(sample, gradient_percentile)))
    gradient_u8 = np.clip(gradient / scale * 255.0, 0, 255).astype(np.uint8)
    color = cv2.cvtColor(gradient_u8, cv2.COLOR_GRAY2BGR)
    cv2.watershed(color, markers)

    region = ((markers == 1) & (box == 1)).astype(np.uint8)
    if not region.any():
        return empty, "no_region"
    region = cv2.morphologyEx(region, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_region = np.zeros_like(region)
    cv2.drawContours(filled_region, contours, -1, 1, -1)

    # Keep only pixels whose depth is inside the person's joint-depth range.
    # This removes background that leaks through weak gradient edges, for
    # example between the fingers or behind the head.
    depth_range = joint_depth_range(joints)
    if depth_range is not None:
        low, high = depth_range
        in_range = np.isfinite(depth) & (depth > 0.0) & (depth >= low) & (depth <= high)
        # Dilate the gate by a few pixels so silhouette edges are not erased.
        in_range = cv2.dilate(in_range.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        filled_region = np.logical_and(filled_region.astype(bool), in_range).astype(np.uint8)
        # Re-add small discs at the joints themselves, so fingertips that sit on
        # a weak gradient edge are still part of the mask.
        discs = np.zeros_like(filled_region)
        for joint in joints:
            visibility = joint.get("visibility")
            if visibility is not None and float(visibility) < MIN_JOINT_VISIBILITY:
                continue
            x = joint.get("x")
            y = joint.get("y")
            if x is None or y is None:
                continue
            px = int(round(float(x) * (width - 1)))
            py = int(round(float(y) * (height - 1)))
            if 0 <= px < width and 0 <= py < height:
                cv2.circle(discs, (px, py), 4, 1, -1)
        discs = np.logical_and(discs.astype(bool), in_range).astype(np.uint8)
        filled_region = np.maximum(filled_region, discs)
        # Fill the head and hands with pose hulls so a weak depth edge at the
        # hairline or a fist does not leave part of the person as background.
        hulls = joint_hulls(
            joints,
            (HEAD_JOINTS, LEFT_HAND_JOINTS, RIGHT_HAND_JOINTS),
            width=width,
            height=height,
        )
        hulls = np.logical_and(hulls.astype(bool), in_range).astype(np.uint8)
        filled_region = np.maximum(filled_region, hulls)
        contours, _ = cv2.findContours(filled_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled_region = np.zeros_like(filled_region)
        cv2.drawContours(filled_region, contours, -1, 1, -1)

    area_fraction = float(filled_region.mean())
    if area_fraction < min_area_fraction:
        return empty, "too_small"
    if area_fraction > max_area_fraction:
        return empty, "too_large"
    return filled_region, "ok"


def gate_validity_to_joint_depth(
    mask: np.ndarray,
    depth_m: np.ndarray,
    joints: list[dict] | None,
) -> np.ndarray:
    """Validity plane limited to the person's joint-depth range.

    Mask pixels whose depth belongs to the background cannot build a person
    splat. Marking them invalid removes stray background splats inside the
    silhouette, which otherwise warp the face and body surface.
    """
    depth = np.asarray(depth_m, dtype=np.float32)
    depth_range = joint_depth_range(joints)
    valid = np.isfinite(depth) & (depth > 0.0) & np.asarray(mask, bool)
    if depth_range is None:
        return valid.astype(np.float32)
    low, high = depth_range
    return np.logical_and(valid, (depth >= low) & (depth <= high)).astype(np.float32)
