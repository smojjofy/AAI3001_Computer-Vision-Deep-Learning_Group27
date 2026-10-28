"""Structural contracts shared with Stream Handler and Optical Flow."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class RGBFrame(Protocol):
    """The identity and RGB plane needed for a retained frame reference."""

    frame_id: int
    source_id: str
    capture_timestamp_ns: int
    timestamp_origin: str
    rgb: np.ndarray


class FlowResult(Protocol):
    """The subset of the Optical Flow result used for significance scoring."""

    status: str
    source_id: str
    previous_frame_id: int | None
    current_frame_id: int
    previous_capture_timestamp_ns: int | None
    current_capture_timestamp_ns: int
    flow: np.ndarray
    valid_mask: np.ndarray

    @property
    def valid_fraction(self) -> float: ...


class FlowEstimator(Protocol):
    """Existing DIS or RAFT estimator; receives retained-reference to current flow."""

    def estimate(self, previous: RGBFrame | None, current: RGBFrame) -> FlowResult: ...


class FrameQueue(Protocol):
    def get(self, timeout_s: float | None = None) -> RGBFrame | None: ...


class HasFrameQueue(Protocol):
    queue: FrameQueue
