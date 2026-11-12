"""Turn raw Depth Anything output into the shared depth convention.

Pure NumPy (no torch, no model), so it can be unit-tested on hand-made arrays.

    raw relative output = disparity (bigger = closer, unknown scale/shift)
    raw metric output   = metres (bigger = farther), capped at max_depth by the model

    -> depth (bigger = farther, NaN = invalid), valid mask, confidence 0..1
"""

import cv2
import numpy as np

from .contracts import DepthEstimation, DepthKind


# Pixels whose log-depth changes faster than this per pixel are on an object edge.
# Depth Anything blends foreground and background there ("flying pixels").
EDGE_SCALE = 0.05

# A frame whose relative output spans less than this has no usable structure
# (e.g. an all-black frame from a covered lens).
MIN_RELATIVE_RANGE = 1e-4


def edge_confidence(depth: np.ndarray, valid: np.ndarray, edge_scale: float = EDGE_SCALE) -> np.ndarray:
    """Heuristic confidence: 1 on smooth surfaces, falling towards 0 at depth edges.

    Depth Anything gives no confidence of its own. Its least reliable pixels are at
    depth discontinuities, so confidence is based on the local gradient of log-depth
    (scale-invariant: the same for metres or relative values).
    """

    if not valid.any():
        return np.zeros(depth.shape, dtype=np.float32)

    # Relative depth reaches 0 at the nearest pixel, so offset it before the log.
    offset = 0.05 if np.nanmax(np.where(valid, depth, np.nan)) <= 1.0 else 0.0
    filled = np.where(valid, depth, np.nanmedian(depth[valid])) + offset
    log_depth = np.log(np.maximum(filled, 1e-6)).astype(np.float32)

    # np.gradient needs at least 2 samples along an axis (a 1-pixel-tall strip has none vertically).
    gy = np.gradient(log_depth, axis=0) if log_depth.shape[0] > 1 else np.zeros_like(log_depth)
    gx = np.gradient(log_depth, axis=1) if log_depth.shape[1] > 1 else np.zeros_like(log_depth)
    grad = np.hypot(gx, gy).astype(np.float32)

    # An invalid neighbour also makes a pixel less trustworthy.
    grad[~valid] = 1e3

    # Take the worst gradient in each 3x3 neighbourhood, so both sides of an edge drop.
    grad = cv2.dilate(grad, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_REPLICATE)

    confidence = 1.0 / (1.0 + (grad / edge_scale) ** 2)
    confidence[~valid] = 0.0
    return np.clip(confidence, 0.0, 1.0).astype(np.float32)


def normalize_prediction(
    raw: np.ndarray,
    kind: DepthKind,
    model_name: str = "",
    max_depth: float | None = None,
    valid_range: tuple[float, float] | None = None,
) -> DepthEstimation:
    """Raw model output [H, W] -> DepthEstimation in the shared convention.

    kind:        "relative_depth" (raw is disparity) or "metric_depth" (raw is metres).
    max_depth:   metric models' cap (20 m indoor / 80 m outdoor). Pixels at the cap
                 mean "beyond range" and are marked invalid.
    valid_range: optional (min_m, max_m) for metric depth, e.g. (0.3, 10.0) to match
                 a sensor's working range. Outside it -> invalid.
    """

    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 2:
        raise ValueError("raw prediction must have shape [height, width]")

    finite = np.isfinite(raw)

    if kind == "relative_depth":
        disparity = raw.copy()
        valid = finite & (raw >= 0.0)          # model ends in ReLU; negative = corrupt

        if valid.any():
            lo = float(raw[valid].min())
            hi = float(raw[valid].max())
        else:
            lo = hi = 0.0

        if hi - lo < MIN_RELATIVE_RANGE:
            valid[:] = False                   # no structure -> nothing trustworthy
            depth = np.full(raw.shape, np.nan, dtype=np.float32)
        else:
            # Flip so bigger = farther, then scale this frame to 0..1.
            depth = (hi - raw) / (hi - lo)
            depth = np.clip(depth, 0.0, 1.0).astype(np.float32)

    elif kind == "metric_depth":
        disparity = None
        valid = finite & (raw > 0.0)
        if max_depth is not None:
            valid &= raw < max_depth * 0.999   # saturated at the model's cap
        depth = raw.astype(np.float32, copy=True)

    else:
        raise ValueError(f"unknown depth kind: {kind}")

    if valid_range is not None:
        if kind != "metric_depth":
            raise ValueError("valid_range (metres) only applies to metric depth")
        near, far = valid_range
        if not (0 < near < far):
            raise ValueError("valid_range must be (min_m, max_m) with 0 < min_m < max_m")
        valid &= (depth >= near) & (depth <= far)

    depth = np.where(valid, depth, np.nan).astype(np.float32)
    confidence = edge_confidence(depth, valid)

    return DepthEstimation(
        depth=depth,
        kind=kind,
        model_name=model_name,
        valid=valid,
        confidence=confidence,
        disparity=disparity,
    )
