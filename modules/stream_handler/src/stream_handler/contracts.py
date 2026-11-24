"""In-process representation of the `shared/schemas/frame_packet` contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np


class TimestampOrigin(StrEnum):
    SOURCE_PRESENTATION = "source_presentation"
    DECODE_MONOTONIC = "decode_monotonic"


@dataclass
class FrameQuality:
    decode_latency_ns: int
    frame_age_ns: int
    frame_age_available: bool
    queue_depth: int = 0
    dropped_frame_count: int = 0
    timestamp_status: str = "ok"
    blur_flag: bool | None = None
    exposure_flag: bool | None = None


@dataclass(frozen=True)
class StereoCalibration:
    """Immutable rectified stereo calibration for one image size.

    Both rectified views of a ZED camera share the focal lengths and the
    principal point, so one record describes the pair. ``baseline_m`` is the
    distance between the two rectified camera centres, in metres.

    The record is frozen because a packet must never describe pixels from one
    calibration with numbers from another. Build a new record when the
    resolution changes.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    baseline_m: float
    width: int
    height: int

    def __post_init__(self) -> None:
        values = (self.fx, self.fy, self.cx, self.cy, self.baseline_m)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("stereo calibration must be finite")
        if self.fx <= 0.0 or self.fy <= 0.0:
            raise ValueError("stereo calibration focal lengths must be positive")
        if self.baseline_m <= 0.0:
            raise ValueError("stereo baseline must be positive")
        if int(self.width) < 1 or int(self.height) < 1:
            raise ValueError("stereo calibration dimensions must be positive")

    @property
    def intrinsics(self) -> tuple[float, float, float, float]:
        """The ``(fx, fy, cx, cy)`` tuple that the depth adapters take."""
        return (float(self.fx), float(self.fy), float(self.cx), float(self.cy))

    def matches(self, width: int, height: int) -> bool:
        """True when this record describes an image of exactly this size."""
        return int(self.width) == int(width) and int(self.height) == int(height)

    def as_dict(self) -> dict[str, Any]:
        """Plain data for a JSON envelope."""
        return {
            "fx": float(self.fx),
            "fy": float(self.fy),
            "cx": float(self.cx),
            "cy": float(self.cy),
            "baseline_m": float(self.baseline_m),
            "width": int(self.width),
            "height": int(self.height),
        }


@dataclass
class FramePacket:
    """A decoded RGB8 frame and identity metadata.

    `rgb` stays as an in-process buffer. Use :meth:`metadata` when a transport
    needs the matching JSON envelope; attach pixels as a separate binary plane.

    Optional capture extras:
    - ``depth``: float32 ``[H, W]`` camera-axis distance. It may contain ``NaN``
      for invalid samples. Sources without a depth sensor leave it as ``None``.
    - ``depth_units``: ``"millimetres"`` or ``"metres"``. Required with depth.
    - ``depth_validity``: optional float32 ``[H, W]`` weight in ``[0, 1]``.
    - ``intrinsics``: ``(fx, fy, cx, cy)`` in pixels for this image size.
    - ``right_rgb``: optional uint8 ``[H, W, 3]`` rectified right view of the
      same frame. One packet owns one left-right pair, so both views always
      share one frame ID, one source ID, and one capture timestamp. Sources
      without a second view leave it as ``None``.
    - ``stereo``: optional :class:`StereoCalibration` that describes the pair.
      Required with ``right_rgb``, and its size must match the image.
    """

    frame_id: int
    source_id: str
    capture_timestamp_ns: int
    timestamp_origin: TimestampOrigin
    rgb: np.ndarray
    quality: FrameQuality
    received_timestamp_ns: int | None = None
    camera: dict[str, Any] | None = None
    depth: np.ndarray | None = None
    depth_units: str | None = None
    depth_validity: np.ndarray | None = None
    intrinsics: tuple[float, float, float, float] | None = None
    right_rgb: np.ndarray | None = None
    stereo: StereoCalibration | None = None
    schema_version: str = "1.1"

    def __post_init__(self) -> None:
        if self.frame_id < 0:
            raise ValueError("frame_id must be non-negative")
        if not self.source_id:
            raise ValueError("source_id must not be empty")
        if self.capture_timestamp_ns < 0:
            raise ValueError("capture_timestamp_ns must be non-negative")
        if self.rgb.dtype != np.uint8 or self.rgb.ndim != 3 or self.rgb.shape[2] != 3:
            raise ValueError("rgb must be a uint8 array with shape [height, width, 3]")
        if self.rgb.shape[0] < 1 or self.rgb.shape[1] < 1:
            raise ValueError("rgb dimensions must be positive")
        if self.depth is not None:
            if self.depth.dtype != np.float32 or self.depth.shape != self.rgb.shape[:2]:
                raise ValueError("depth must be a float32 array with shape [height, width]")
            if self.depth_units not in ("millimetres", "metres"):
                raise ValueError("depth_units must be 'millimetres' or 'metres' when depth is set")
            if self.depth_validity is not None:
                if self.depth_validity.dtype != np.float32 or self.depth_validity.shape != self.rgb.shape[:2]:
                    raise ValueError("depth_validity must be a float32 array with shape [height, width]")
                if (self.depth_validity < 0.0).any() or (self.depth_validity > 1.0).any():
                    raise ValueError("depth_validity values must be in [0, 1]")
        elif self.depth_units is not None or self.depth_validity is not None:
            raise ValueError("depth_units and depth_validity require depth")
        if self.intrinsics is not None:
            fx, fy, cx, cy = self.intrinsics
            if not all(np.isfinite(value) for value in (fx, fy, cx, cy)):
                raise ValueError("intrinsics must be finite")
            if fx <= 0.0 or fy <= 0.0:
                raise ValueError("fx and fy must be positive")
        self._check_stereo()

    def _check_stereo(self) -> None:
        """Validate the optional right view and its calibration as one pair.

        The two views must describe the same image size, because one rectified
        stereo pair has one row alignment and one principal point. A right view
        without calibration, or calibration without a right view, is half a
        pair and is rejected rather than left for a later stage to discover.
        """
        if self.right_rgb is None:
            if self.stereo is not None:
                raise ValueError("stereo calibration requires right_rgb")
            return
        if self.stereo is None:
            raise ValueError("right_rgb requires a stereo calibration record")
        right = self.right_rgb
        if right.dtype != np.uint8 or right.ndim != 3 or right.shape[2] != 3:
            raise ValueError("right_rgb must be a uint8 array with shape [height, width, 3]")
        if right.shape != self.rgb.shape:
            raise ValueError(
                f"right_rgb shape {right.shape} must match rgb shape {self.rgb.shape}"
            )
        if not self.stereo.matches(self.width, self.height):
            raise ValueError(
                "stereo calibration describes "
                f"{self.stereo.width}x{self.stereo.height} but the frame is "
                f"{self.width}x{self.height}"
            )

    @property
    def width(self) -> int:
        return int(self.rgb.shape[1])

    @property
    def height(self) -> int:
        return int(self.rgb.shape[0])

    @property
    def stride_bytes(self) -> int:
        return int(self.rgb.strides[0])

    def metadata(self) -> dict[str, Any]:
        """Return the variable-payload JSON envelope defined by the shared schema."""
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "frame_id": self.frame_id,
            "source_id": self.source_id,
            "capture_timestamp_ns": self.capture_timestamp_ns,
            "timestamp_origin": str(self.timestamp_origin),
            "image": {
                "width": self.width,
                "height": self.height,
                "stride_bytes": self.stride_bytes,
                "pixel_format": "RGB8",
                "payload_encoding": "raw-rgb8",
                "payload_bytes": int(self.rgb.nbytes),
            },
            "quality": {key: value for key, value in asdict(self.quality).items() if value is not None},
        }
        if self.received_timestamp_ns is not None:
            payload["received_timestamp_ns"] = self.received_timestamp_ns
        if self.camera is not None:
            payload["camera"] = self.camera
        if self.depth is not None:
            payload["depth"] = {
                "width": int(self.depth.shape[1]),
                "height": int(self.depth.shape[0]),
                "units": self.depth_units,
                "payload_encoding": "raw-float32",
                "payload_bytes": int(self.depth.nbytes),
                "has_validity": self.depth_validity is not None,
            }
        if self.intrinsics is not None:
            payload["intrinsics"] = [float(value) for value in self.intrinsics]
        if self.right_rgb is not None and self.stereo is not None:
            payload["right_image"] = {
                "width": int(self.right_rgb.shape[1]),
                "height": int(self.right_rgb.shape[0]),
                "stride_bytes": int(self.right_rgb.strides[0]),
                "pixel_format": "RGB8",
                "payload_encoding": "raw-rgb8",
                "payload_bytes": int(self.right_rgb.nbytes),
            }
            payload["stereo"] = self.stereo.as_dict()
        return payload
