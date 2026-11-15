"""Flow-only candidate window and refresh/reuse selection.

`CONTEXT.md` current contract: optical flow runs for each frame in the 100 ms
candidate window, the reranker keeps one retained refresh reference, and the
candidate with the highest normalized flow score wins the window. Depth, the
depth-threshold mask, and pose run only for the selected refresh frame.

The retained reference advances only through :meth:`WindowSelector.commit`,
which the application calls after a refresh produced a valid active state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

DEFAULT_WINDOW_NS = 100_000_000
DEFAULT_THRESHOLD = 0.02
DEFAULT_ACTIVE_AGE_NS = 500_000_000
DEFAULT_MIN_VALID_FRACTION = 0.25


@dataclass(frozen=True)
class RetainedReference:
    """One retained successful refresh frame."""

    frame_id: int
    source_id: str
    capture_timestamp_ns: int
    timestamp_origin: str
    rgb: np.ndarray

    @classmethod
    def copy_of(cls, packet) -> "RetainedReference":
        return cls(
            frame_id=int(packet.frame_id),
            source_id=str(packet.source_id),
            capture_timestamp_ns=int(packet.capture_timestamp_ns),
            timestamp_origin=str(packet.timestamp_origin),
            rgb=np.array(packet.rgb, copy=True),
        )


@dataclass(frozen=True)
class Candidate:
    """One window candidate with its reference-to-candidate flow score."""

    packet: Any
    flow: Any | None
    score: float | None
    flow_valid: bool
    reason: str
    reference_frame_id: int | None = None

    @property
    def frame_id(self) -> int:
        return int(self.packet.frame_id)

    @property
    def capture_timestamp_ns(self) -> int:
        return int(self.packet.capture_timestamp_ns)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "capture_timestamp_ns": self.capture_timestamp_ns,
            "flow_score": None if self.score is None else round(float(self.score), 5),
            "flow_valid": self.flow_valid,
            "reason": self.reason,
            "reference_frame_id": self.reference_frame_id,
        }


@dataclass(frozen=True)
class WindowDecision:
    """The window result: action, reason, selected candidate, and diagnostics."""

    action: str
    reason: str
    selected: Candidate
    candidates: int
    window_start_ns: int
    window_end_ns: int
    forced: bool = False

    @property
    def score(self) -> float | None:
        return self.selected.score

    def diagnostics(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "reason": self.reason,
            "selected_frame_id": self.selected.frame_id,
            "flow_score": None if self.selected.score is None else round(float(self.selected.score), 5),
            "candidates": int(self.candidates),
            "window_start_ns": self.window_start_ns,
            "window_end_ns": self.window_end_ns,
            "forced": bool(self.forced),
        }


class WindowSelector:
    """Keep one flow reference and score candidates against it."""

    def __init__(
        self,
        estimator,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        active_age_ns: int = DEFAULT_ACTIVE_AGE_NS,
        min_valid_fraction: float = DEFAULT_MIN_VALID_FRACTION,
    ) -> None:
        if not np.isfinite(threshold) or threshold < 0.0:
            raise ValueError("threshold must be finite and non-negative")
        if active_age_ns < 1:
            raise ValueError("active_age_ns must be positive")
        if not 0.0 <= min_valid_fraction <= 1.0:
            raise ValueError("min_valid_fraction must be in [0, 1]")
        self.estimator = estimator
        self.threshold = float(threshold)
        self.active_age_ns = int(active_age_ns)
        self.min_valid_fraction = float(min_valid_fraction)
        self.reference: RetainedReference | None = None
        self.active_state_capture_ns: int | None = None
        self.active_state_frame_id: int | None = None

    def reset(self) -> None:
        self.reference = None
        self.active_state_capture_ns = None
        self.active_state_frame_id = None

    def observe(self, packet) -> Candidate:
        """Compute the reference-to-candidate flow score for one frame."""
        reference = self.reference
        if reference is None:
            return Candidate(packet=packet, flow=None, score=None, flow_valid=False, reason="bootstrap")
        reference_frame_id = reference.frame_id
        if reference.source_id != packet.source_id or reference.rgb.shape != packet.rgb.shape:
            return Candidate(packet=packet, flow=None, score=None, flow_valid=False, reason="reference_mismatch", reference_frame_id=reference_frame_id)
        try:
            flow = self.estimator.estimate(reference, packet)
        except (ValueError, RuntimeError) as error:
            return Candidate(packet=packet, flow=None, score=None, flow_valid=False, reason=f"flow_error:{error}", reference_frame_id=reference_frame_id)
        return self._score_candidate(packet, flow, reference_frame_id)

    def _score_candidate(self, packet, flow, reference_frame_id: int) -> Candidate:
        if flow is None or getattr(flow, "status", None) != "ok":
            return Candidate(packet=packet, flow=flow, score=None, flow_valid=False, reason="flow_unavailable", reference_frame_id=reference_frame_id)
        valid_mask = getattr(flow, "valid_mask", None)
        vectors = getattr(flow, "flow", None)
        if valid_mask is None or vectors is None or vectors.ndim != 3 or vectors.shape[2] != 2:
            return Candidate(packet=packet, flow=flow, score=None, flow_valid=False, reason="flow_malformed", reference_frame_id=reference_frame_id)
        valid_fraction = float(valid_mask.mean()) if valid_mask.size else 0.0
        if valid_fraction < self.min_valid_fraction:
            return Candidate(packet=packet, flow=flow, score=None, flow_valid=False, reason="flow_low_validity", reference_frame_id=reference_frame_id)
        magnitudes = np.linalg.norm(vectors[valid_mask], axis=1)
        if magnitudes.size == 0 or not np.isfinite(magnitudes).all():
            return Candidate(packet=packet, flow=flow, score=None, flow_valid=False, reason="flow_empty", reference_frame_id=reference_frame_id)
        height, width = vectors.shape[:2]
        diagonal = float(np.hypot(width, height))
        score = float(magnitudes.mean()) / diagonal
        return Candidate(packet=packet, flow=flow, score=score, flow_valid=True, reason="scored", reference_frame_id=reference_frame_id)

    def _age_exceeded(self, capture_timestamp_ns: int) -> bool:
        if self.active_state_capture_ns is None:
            return True
        return capture_timestamp_ns - self.active_state_capture_ns >= self.active_age_ns

    def select(self, candidates: list[Candidate]) -> WindowDecision:
        """Pick the window candidate and the refresh/reuse action.

        The forced refresh uses the active state age measured on capture time.
        It requests a refresh but never makes a failed construction active.
        """
        if not candidates:
            raise ValueError("a window needs at least one candidate")
        first = candidates[0]
        ordered = sorted(
            candidates,
            key=lambda item: -1.0 if item.score is None else item.score,
            reverse=True,
        )
        best = ordered[0]
        if self.reference is None or self.active_state_capture_ns is None:
            return WindowDecision("refresh", "bootstrap", first, len(candidates), first.capture_timestamp_ns, candidates[-1].capture_timestamp_ns)
        if best.score is None:
            return WindowDecision("refresh", "invalid_flow", best, len(candidates), first.capture_timestamp_ns, candidates[-1].capture_timestamp_ns)
        if best.score >= self.threshold:
            return WindowDecision("refresh", "significant_difference", best, len(candidates), first.capture_timestamp_ns, candidates[-1].capture_timestamp_ns)
        if self._age_exceeded(best.capture_timestamp_ns):
            return WindowDecision("refresh", "reference_expired", best, len(candidates), first.capture_timestamp_ns, candidates[-1].capture_timestamp_ns, forced=True)
        return WindowDecision("reuse", "not_significant", best, len(candidates), first.capture_timestamp_ns, candidates[-1].capture_timestamp_ns)

    def commit(self, packet) -> None:
        """Retain ``packet`` after its refresh produced a valid active state."""
        self.reference = RetainedReference.copy_of(packet)
        self.active_state_capture_ns = int(packet.capture_timestamp_ns)
        self.active_state_frame_id = int(packet.frame_id)
