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
    """

    frame_id: int
    source_id: str
    capture_timestamp_ns: int
    timestamp_origin: TimestampOrigin
    rgb: np.ndarray
    quality: FrameQuality
    received_timestamp_ns: int | None = None
    camera: dict[str, Any] | None = None
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
        return payload
