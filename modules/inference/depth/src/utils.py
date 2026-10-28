"""Helpers for viewing depth and combining it with pose landmarks."""

import cv2
import numpy as np

from .contracts import DepthEstimation


# colorize is for displaying depth maps as BGR images, where near objects are red/yellow and far objects are blue. Invalid pixels (NaN) are drawn black. It returns a BGR image suitable for display with cv2.imshow.
def colorize(
    estimation: DepthEstimation,
    max_depth_m: float | None = None,
) -> np.ndarray:
    """Depth -> BGR image for cv2.imshow. Near = red/yellow, far = blue, invalid = black.

    Metric depth uses a fixed range (0 .. max_depth_m) so colours mean the same
    distance in every frame. Relative depth is already 0..1 per frame.
    """

    values = estimation.depth
    valid = estimation.valid

    # Both kinds are bigger = farther, so "nearness" is the flipped, 0..1-scaled depth.
    if estimation.is_metric:
        far = max_depth_m or (float(np.percentile(values[valid], 98)) if valid.any() else 1.0)
        nearness = 1.0 - np.clip(np.nan_to_num(values, nan=far) / max(far, 1e-6), 0.0, 1.0)
    else:
        nearness = 1.0 - np.nan_to_num(values, nan=1.0)

    # the normalized values are then converted to a grayscale image and colorized using OpenCV's COLORMAP_TURBO colormap.
    gray = (np.clip(nearness, 0.0, 1.0) * 255).astype(np.uint8)
    image = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    image[~valid] = 0
    return image


def colorize_confidence(estimation: DepthEstimation) -> np.ndarray:
    """Confidence -> BGR greyscale image. White = trusted, black = unreliable/invalid."""

    gray = (np.clip(estimation.confidence, 0.0, 1.0) * 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# sample depth at a normalized point (x, y in 0..1, like MediaPipe landmarks). Takes the median over a small window, which is steadier than one pixel (landmarks often sit right on a body edge). Invalid pixels are ignored. Returns NaN if off-screen or nothing valid nearby.
def sample_depth(
    estimation: DepthEstimation,
    x: float,
    y: float,
    radius: int = 2,
) -> float:
    """Depth at a NORMALISED point (x, y in 0..1, like MediaPipe landmarks).

    Median over a small window of valid pixels, which is steadier than one pixel
    (landmarks often sit right on a body edge). NaN if off-screen or all invalid.
    """

    height, width = estimation.depth.shape
    col = int(round(x * (width - 1)))
    row = int(round(y * (height - 1)))

    if not (0 <= col < width and 0 <= row < height):
        return float("nan")

    window = estimation.depth[
        max(row - radius, 0): row + radius + 1,
        max(col - radius, 0): col + radius + 1,
    ]

    window = window[np.isfinite(window)]
    return float(np.median(window)) if window.size else float("nan")


def temporal_flicker(
    previous: DepthEstimation,
    current: DepthEstimation,
) -> float:
    """Median relative change between two consecutive frames' depth.

    Only pixels valid in both frames count. On a static camera and scene this
    should be close to 0. High values mean depth "breathes" frame to frame,
    which shows up as jitter in splats.
    """

    both = previous.valid & current.valid
    if not both.any():
        return float("nan")

    a, b = previous.depth[both], current.depth[both]
    # relative depth reaches 0 at the nearest pixel, so compare against the frame's range
    scale = np.maximum(np.abs(a), 0.05 if not previous.is_metric else 1e-6)
    return float(np.median(np.abs(b - a) / scale))
