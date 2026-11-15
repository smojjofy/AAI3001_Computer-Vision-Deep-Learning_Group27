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
    schema_version: str = "1.0"

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
        return payload
