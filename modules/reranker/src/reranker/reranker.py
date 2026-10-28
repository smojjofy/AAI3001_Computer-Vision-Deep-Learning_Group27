"""One-reference replacement reranker."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np

from .contracts import FlowResult, RGBFrame


DecisionAction = Literal["reuse", "refresh"]
DecisionReason = Literal[
    "bootstrap", "significant_difference", "not_significant", "flow_unavailable",
]


@dataclass(frozen=True)
class RerankerConfig:
    """Thresholds for comparing a current frame to the retained reference."""

    significant_difference_px: float = 2.0
    minimum_valid_fraction: float = 0.25

    def __post_init__(self) -> None:
        if not math.isfinite(self.significant_difference_px) or self.significant_difference_px <= 0:
            raise ValueError("significant_difference_px must be finite and positive")
        if not math.isfinite(self.minimum_valid_fraction) or not 0 <= self.minimum_valid_fraction <= 1:
            raise ValueError("minimum_valid_fraction must be between zero and one")


@dataclass(frozen=True)
class RetainedFrame:
    """Immutable copy of the only camera frame retained by the reranker."""

    frame_id: int
    source_id: str
    capture_timestamp_ns: int
    timestamp_origin: str
    rgb: np.ndarray

    @classmethod
    def copy_of(cls, frame: RGBFrame) -> "RetainedFrame":
        rgb = np.array(frame.rgb, dtype=np.uint8, copy=True, order="C")
        rgb.setflags(write=False)
        return cls(
            frame_id=frame.frame_id,
            source_id=frame.source_id,
            capture_timestamp_ns=frame.capture_timestamp_ns,
            timestamp_origin=str(frame.timestamp_origin),
            rgb=rgb,
        )


@dataclass(frozen=True)
class SelectionDecision:
    """A decision matching `shared/schemas/selection_decision.schema.json`."""

    action: DecisionAction
    source_id: str
    selected_frame_id: int
    selected_capture_timestamp_ns: int
    flow_current_frame_id: int
    flow_current_capture_timestamp_ns: int
    reference_frame_id: int | None
    reference_capture_timestamp_ns: int | None
    reason: DecisionReason
    difference_score: float | None

    def metadata(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": "1.0",
            "action": self.action,
            "source_id": self.source_id,
            "selected_frame_id": self.selected_frame_id,
            "selected_capture_timestamp_ns": self.selected_capture_timestamp_ns,
            "flow_current_frame_id": self.flow_current_frame_id,
            "flow_current_capture_timestamp_ns": self.flow_current_capture_timestamp_ns,
            "reference_frame_id": self.reference_frame_id,
            "reference_capture_timestamp_ns": self.reference_capture_timestamp_ns,
            "reason": self.reason,
        }
        if self.difference_score is not None:
            payload["difference_score"] = self.difference_score
        return payload


class Reranker:
    """Replace one retained reference when a new frame differs significantly."""

    def __init__(self, config: RerankerConfig | None = None) -> None:
        self.config = config or RerankerConfig()
        self._reference: RetainedFrame | None = None

    @property
    def reference(self) -> RetainedFrame | None:
        """The sole retained significant frame; callers supply it to Optical Flow."""
        return self._reference

    def reset(self) -> None:
        """Flush the retained reference, for example when a capture run restarts."""
        self._reference = None

    def decide(self, current: RGBFrame, flow: FlowResult | None = None) -> SelectionDecision:
        """Return `refresh` for a significant current/reference difference.

        When a reference exists, `flow` must describe that exact reference as
        previous and `current` as current.  The caller owns the optical-flow
        invocation, keeping the Reranker independent from DIS or RAFT.
        """
        reference = self._reference
        if reference is None:
            return self._refresh(current, None, "bootstrap", None)

        if (
            flow is None
            or flow.status != "ok"
            or flow.source_id != current.source_id
            or flow.previous_frame_id != reference.frame_id
            or flow.current_frame_id != current.frame_id
            or flow.previous_capture_timestamp_ns != reference.capture_timestamp_ns
            or flow.current_capture_timestamp_ns != current.capture_timestamp_ns
            or current.source_id != reference.source_id
            or current.rgb.shape != reference.rgb.shape
        ):
            return self._refresh(current, reference, "flow_unavailable", None)

        difference_score = self._difference_score(flow)
        if difference_score is None:
            return self._refresh(current, reference, "flow_unavailable", None)
        if difference_score >= self.config.significant_difference_px:
            return self._refresh(current, reference, "significant_difference", difference_score)
        return SelectionDecision(
            action="reuse",
            source_id=current.source_id,
            selected_frame_id=current.frame_id,
            selected_capture_timestamp_ns=current.capture_timestamp_ns,
            flow_current_frame_id=current.frame_id,
            flow_current_capture_timestamp_ns=current.capture_timestamp_ns,
            reference_frame_id=reference.frame_id,
            reference_capture_timestamp_ns=reference.capture_timestamp_ns,
            reason="not_significant",
            difference_score=difference_score,
        )

    def _difference_score(self, flow: FlowResult) -> float | None:
        if flow.valid_mask.shape != flow.flow.shape[:2] or flow.flow.ndim != 3 or flow.flow.shape[2] != 2:
            return None
        if not math.isfinite(float(flow.valid_fraction)) or flow.valid_fraction < self.config.minimum_valid_fraction:
            return None
        vectors = flow.flow[flow.valid_mask]
        if len(vectors) == 0 or not np.isfinite(vectors).all():
            return None
        return float(np.median(np.linalg.norm(vectors, axis=1)))

    def _refresh(
        self,
        current: RGBFrame,
        reference: RetainedFrame | None,
        reason: DecisionReason,
        difference_score: float | None,
    ) -> SelectionDecision:
        # Assignment flushes the old copy before retaining this new frame.
        self._reference = RetainedFrame.copy_of(current)
        return SelectionDecision(
            action="refresh",
            source_id=current.source_id,
            selected_frame_id=current.frame_id,
            selected_capture_timestamp_ns=current.capture_timestamp_ns,
            flow_current_frame_id=current.frame_id,
            flow_current_capture_timestamp_ns=current.capture_timestamp_ns,
            reference_frame_id=None if reference is None else reference.frame_id,
            reference_capture_timestamp_ns=None if reference is None else reference.capture_timestamp_ns,
            reason=reason,
            difference_score=difference_score,
        )
