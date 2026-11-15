"""Reuse-path motion preview for the browser warp.

The browser samples the selected reuse flow at each splat source pixel and
offsets only the displayed position. This module prepares a small, bounded
motion payload: a downsampled flow field in normalized image units, its
validity, and the pose joints that the browser uses to limit the warp by body
region.

Nothing here changes stored 3D positions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

DEFAULT_MOTION_SIZE = (40, 30)  # width, height
DEFAULT_MAX_OFFSET_NDC = 0.15


@dataclass(frozen=True)
class MotionPreview:
    """A downsampled reuse flow field plus pose joints, ready for transport.

    ``source_id``, ``generation``, and ``reference_frame_id`` identify the
    splat state that this motion may move. The browser applies the warp only
    when all three equal the displayed splat state, and only when
    ``frame_id`` is later than ``reference_frame_id``.
    """

    frame_id: int
    reference_frame_id: int | None
    source_id: str
    generation: int
    splat_frame_id: int | None
    width: int
    height: int
    source_width: int
    source_height: int
    flow_ndc: np.ndarray  # [height, width, 2] float32, normalized image units
    valid: np.ndarray  # [height, width] bool
    joints: list[dict[str, Any]]
    max_offset_ndc: float

    def identity_valid(self) -> bool:
        """True when this motion describes a later frame of the same active state."""
        if self.reference_frame_id is None or self.splat_frame_id is None:
            return False
        return self.splat_frame_id == self.reference_frame_id and self.frame_id > self.reference_frame_id

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "frame_id": int(self.frame_id),
            "reference_frame_id": None if self.reference_frame_id is None else int(self.reference_frame_id),
            "source_id": self.source_id,
            "generation": int(self.generation),
            "splat_frame_id": None if self.splat_frame_id is None else int(self.splat_frame_id),
            "width": int(self.width),
            "height": int(self.height),
            "source_width": int(self.source_width),
            "source_height": int(self.source_height),
            "max_offset_ndc": float(self.max_offset_ndc),
            "flow_ndc": np.round(self.flow_ndc.reshape(-1), 5).tolist(),
            "valid": self.valid.reshape(-1).astype(np.uint8).tolist(),
            "joints": self.joints,
        }


def build_motion_preview(
    candidate,
    joints: list[dict[str, Any]] | None,
    *,
    source_id: str,
    generation: int,
    splat_frame_id: int | None,
    size: tuple[int, int] = DEFAULT_MOTION_SIZE,
    max_offset_ndc: float = DEFAULT_MAX_OFFSET_NDC,
) -> MotionPreview | None:
    """Prepare the reuse-flow payload for one scored candidate, or None.

    Returns None when the flow is unusable or the identity is not valid for the
    active splat state. The caller publishes only a non-None payload.
    """
    flow = getattr(candidate, "flow", None)
    if flow is None or getattr(flow, "status", None) != "ok":
        return None
    vectors = np.asarray(getattr(flow, "flow", None), dtype=np.float32)
    valid = np.asarray(getattr(flow, "valid_mask", None), dtype=bool)
    if vectors.ndim != 3 or vectors.shape[2] != 2 or valid.shape != vectors.shape[:2]:
        return None
    height, width = vectors.shape[:2]
    if height < 2 or width < 2:
        return None

    # Convert to normalized image units: x to [-1, 1] across width, y flipped
    # because image y grows down and clip-space y grows up.
    normalized = np.empty_like(vectors)
    normalized[..., 0] = vectors[..., 0] * (2.0 / width)
    normalized[..., 1] = -vectors[..., 1] * (2.0 / height)
    np.clip(normalized, -max_offset_ndc, max_offset_ndc, out=normalized)

    small = cv2.resize(normalized, size, interpolation=cv2.INTER_AREA)
    small_valid = cv2.resize(valid.astype(np.float32), size, interpolation=cv2.INTER_AREA) > 0.999
    small = np.where(small_valid[..., None], small, 0.0).astype(np.float32)

    reference = getattr(candidate, "reference_frame_id", None)
    preview = MotionPreview(
        frame_id=int(candidate.frame_id),
        reference_frame_id=None if reference is None else int(reference),
        source_id=str(source_id),
        generation=int(generation),
        splat_frame_id=None if splat_frame_id is None else int(splat_frame_id),
        width=size[0],
        height=size[1],
        source_width=width,
        source_height=height,
        flow_ndc=small,
        valid=small_valid,
        joints=list(joints or []),
        max_offset_ndc=float(max_offset_ndc),
    )
    return preview if preview.identity_valid() else None
