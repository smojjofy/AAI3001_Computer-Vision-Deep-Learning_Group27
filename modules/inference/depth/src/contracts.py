from dataclasses import dataclass
from typing import Literal

import numpy as np

# Define a type alias for depth kind. Both kinds follow the same convention (see
# DEPTH_CONVENTION.md): bigger = farther, invalid = NaN.
#   "metric_depth"   -> metres along the camera axis (z-depth)
#   "relative_depth" -> unitless 0..1 per frame (0 = nearest, 1 = farthest), ordering only
DepthKind = Literal["metric_depth", "relative_depth"]

DEPTH_KINDS = ("metric_depth", "relative_depth")


# A dataclass is used along with frozen=True to create an immutable class that represents a depth estimation for a single frame. The __post_init__ method validates the input data, so every DepthEstimation that exists is guaranteed to follow the shared depth convention.
@dataclass(frozen=True)
class DepthEstimation:
    """Dense depth for one frame, in the shared depth convention.

    depth:       [H, W] float32. Bigger = farther. NaN where invalid (never 0).
                 metric_depth   -> metres along the camera's optical axis.
                 relative_depth -> 0..1 within this frame (0 = nearest, 1 = farthest).
                                   Ordering only: 0.6 is farther than 0.3, but not 2x.
    valid:       [H, W] bool. True where depth is usable. Always equals isfinite(depth).
    confidence:  [H, W] float32 in 0..1. How much to trust each pixel; 0 where invalid.
    disparity:   [H, W] float32 or None. Raw relative-model output (bigger = closer,
                 unknown scale and shift). Kept for alignment to COLMAP/ZED, which
                 needs the un-normalised values. None for metric models.
    """

    depth: np.ndarray
    kind: DepthKind
    model_name: str
    valid: np.ndarray
    confidence: np.ndarray
    disparity: np.ndarray | None = None

    # different from __init__ because we want to validate the inputs. The __post_init__ method is called automatically after the dataclass is initialized, allowing us to perform checks on the arrays' dimensions and data types, as well as ensuring that the values follow the convention. If any of these checks fail, a ValueError is raised with an appropriate message.
    def __post_init__(self) -> None:
        if self.depth.ndim != 2:
            raise ValueError(
                "depth must have shape [height, width]"
            )

        # the depth array must be of type float32, as this is the expected format for depth values in many computer vision applications. This ensures that downstream processing can rely on the depth values being in a consistent and expected format.
        if self.depth.dtype != np.float32:
            raise ValueError(
                "depth must use dtype float32"
            )

        if self.kind not in DEPTH_KINDS:
            raise ValueError(
                f"unknown depth kind: {self.kind} (expected one of {DEPTH_KINDS})"
            )

        if self.valid.shape != self.depth.shape or self.valid.dtype != np.bool_:
            raise ValueError(
                "valid must be a bool array with the same shape as depth"
            )

        if self.confidence.shape != self.depth.shape or self.confidence.dtype != np.float32:
            raise ValueError(
                "confidence must be a float32 array with the same shape as depth"
            )

        if self.disparity is not None and (
            self.disparity.shape != self.depth.shape or self.disparity.dtype != np.float32
        ):
            raise ValueError(
                "disparity must be a float32 array with the same shape as depth"
            )

        # Invalid pixels are NaN, and only invalid pixels are NaN.
        if not np.array_equal(self.valid, np.isfinite(self.depth)):
            raise ValueError(
                "valid must be exactly where depth is finite (invalid pixels must be NaN)"
            )

        if not (np.all(self.confidence >= 0.0) and np.all(self.confidence <= 1.0)):
            raise ValueError(
                "confidence must be within 0..1"
            )

        if np.any(self.confidence[~self.valid] != 0.0):
            raise ValueError(
                "confidence must be 0 where depth is invalid"
            )

        values = self.depth[self.valid]
        if values.size:
            if self.kind == "metric_depth" and values.min() <= 0.0:
                raise ValueError(
                    "metric depth must be > 0 m where valid"
                )
            if self.kind == "relative_depth" and (values.min() < 0.0 or values.max() > 1.0):
                raise ValueError(
                    "relative depth must be within 0..1 where valid"
                )

    # Why property? Because these are derived attributes. By defining them as properties, we provide a convenient way to check them without storing additional attributes.
    @property
    def is_metric(self) -> bool:
        return self.kind == "metric_depth"

    @property
    def valid_fraction(self) -> float:
        """Share of pixels with usable depth (0..1)."""
        return float(self.valid.mean())

    @property
    def shape(self) -> tuple[int, int]:
        return self.depth.shape


# Dataclass again because we want immutability and automatic init. DepthFrameResult is an immutable dataclass that represents the result of a depth estimation tied to its original source frame. It contains the frame ID, source ID, capture timestamp in nanoseconds, and the depth estimation itself. The __post_init__ method is used to validate the input data, ensuring that the frame ID is non-negative, the source ID is not empty, and the capture timestamp is non-negative. This class allows for easy matching of depth results with corresponding pose results based on the same identity fields (source_id and frame_id).
@dataclass(frozen=True)
class DepthFrameResult:
    """Depth result tied to its original source frame.

    Uses the same identity fields as pose_estimator.PoseFrameResult, so pose and
    depth for the same frame can be matched on (source_id, frame_id).
    """

    frame_id: int
    source_id: str
    capture_timestamp_ns: int
    depth: DepthEstimation

    def __post_init__(self) -> None:
        if self.frame_id < 0:
            raise ValueError(
                "frame_id must be non-negative"
            )

        if not self.source_id:
            raise ValueError(
                "source_id must not be empty"
            )

        if self.capture_timestamp_ns < 0:
            raise ValueError(
                "capture_timestamp_ns must be non-negative"
            )
