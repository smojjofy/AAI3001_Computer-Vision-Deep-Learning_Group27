"""Reduced-resolution DIS flow with an explicit previous-to-current contract."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

import cv2
import numpy as np


class RGBFrame(Protocol):
    """Minimal boundary shared by FramePacket and future ProcessedFrame."""

    frame_id: int
    source_id: str
    capture_timestamp_ns: int
    timestamp_origin: str
    rgb: np.ndarray


@dataclass(frozen=True)
class FlowConfig:
    flow_model: str = "opencv_dis"
    flow_width: int = 320
    flow_height: int = 180
    dis_preset: str = "fast"
    forward_backward_threshold_px: float = 1.5
    scene_cut_mean_difference: float = 0.55
    raft_weights: str = "default"
    raft_checkpoint: str | None = None
    raft_device: str = "auto"
    raft_iterations: int = 12

    def __post_init__(self) -> None:
        if self.flow_model not in {"opencv_dis", "raft_small"}:
            raise ValueError("flow_model must be opencv_dis or raft_small")
        if any(type(value) is not int for value in (self.flow_width, self.flow_height)):
            raise ValueError("flow dimensions must be integers")
        if self.flow_width < 32 or self.flow_height < 32:
            raise ValueError("flow dimensions must be at least 32 pixels")
        if self.dis_preset not in {"ultrafast", "fast", "medium"}:
            raise ValueError("dis_preset must be ultrafast, fast, or medium")
        if not math.isfinite(self.forward_backward_threshold_px) or self.forward_backward_threshold_px <= 0:
            raise ValueError("forward_backward_threshold_px must be positive")
        if not math.isfinite(self.scene_cut_mean_difference) or not 0 <= self.scene_cut_mean_difference <= 1:
            raise ValueError("scene_cut_mean_difference must be between 0 and 1")
        if self.raft_weights not in {"default", "none"}:
            raise ValueError("raft_weights must be default or none")
        if self.raft_device not in {"auto", "cpu", "cuda"}:
            raise ValueError("raft_device must be auto, cpu, or cuda")
        if type(self.raft_iterations) is not int or self.raft_iterations < 1:
            raise ValueError("raft_iterations must be a positive integer")
        if self.raft_checkpoint is not None and not self.raft_checkpoint:
            raise ValueError("raft_checkpoint must be a non-empty path or null")


@dataclass(frozen=True)
class FlowResult:
    status: str
    source_id: str
    previous_frame_id: int | None
    current_frame_id: int
    previous_capture_timestamp_ns: int | None
    current_capture_timestamp_ns: int
    timestamp_origin: str
    elapsed_ns: int | None
    flow: np.ndarray
    valid_mask: np.ndarray
    flow_width: int
    flow_height: int
    backend: str = "opencv_dis"

    @property
    def height(self) -> int:
        return int(self.flow.shape[0])

    @property
    def width(self) -> int:
        return int(self.flow.shape[1])

    @property
    def valid_fraction(self) -> float:
        return float(np.mean(self.valid_mask))

    def metadata(self) -> dict[str, object]:
        """Return the JSON envelope; transport flow and mask planes separately."""
        return {
            "schema_version": "1.0",
            "status": self.status,
            "source_id": self.source_id,
            "previous_frame_id": self.previous_frame_id,
            "current_frame_id": self.current_frame_id,
            "previous_capture_timestamp_ns": self.previous_capture_timestamp_ns,
            "current_capture_timestamp_ns": self.current_capture_timestamp_ns,
            "timestamp_origin": str(self.timestamp_origin),
            "elapsed_ns": self.elapsed_ns,
            "width": self.width,
            "height": self.height,
            "direction": "previous_to_current",
            "units": "pixels_at_output_resolution",
            "backend": self.backend,
            "flow_width": self.flow_width,
            "flow_height": self.flow_height,
            "valid_fraction": self.valid_fraction,
        }


class DISFlowEstimator:
    """Calculate dense flow without taking ownership of scheduler/cache policy."""

    def __init__(self, config: FlowConfig | None = None) -> None:
        self.config = config or FlowConfig()
        if self.config.flow_model != "opencv_dis":
            raise ValueError("DISFlowEstimator requires flow_model=opencv_dis")
        preset = {
            "ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
            "fast": cv2.DISOPTICAL_FLOW_PRESET_FAST,
            "medium": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
        }[self.config.dis_preset]
        self._forward = cv2.DISOpticalFlow_create(preset)
        self._backward = cv2.DISOpticalFlow_create(preset)

    def _size(self, width: int, height: int) -> tuple[int, int]:
        scale = min(1.0, self.config.flow_width / width, self.config.flow_height / height)
        return max(32, round(width * scale)), max(32, round(height * scale))

    def _empty(self, current: RGBFrame, previous: RGBFrame | None, status: str) -> FlowResult:
        height, width = current.rgb.shape[:2]
        return FlowResult(
            status=status,
            source_id=current.source_id,
            previous_frame_id=None if previous is None else previous.frame_id,
            current_frame_id=current.frame_id,
            previous_capture_timestamp_ns=None if previous is None else previous.capture_timestamp_ns,
            current_capture_timestamp_ns=current.capture_timestamp_ns,
            timestamp_origin=str(current.timestamp_origin),
            elapsed_ns=None,
            flow=np.zeros((height, width, 2), dtype=np.float32),
            valid_mask=np.zeros((height, width), dtype=np.bool_),
            flow_width=min(width, self.config.flow_width),
            flow_height=min(height, self.config.flow_height),
        )

    @staticmethod
    def _validate(frame: RGBFrame) -> None:
        rgb = frame.rgb
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("flow input rgb must be uint8 [height, width, 3]")
        if min(rgb.shape[:2]) < 32:
            raise ValueError("flow input dimensions must be at least 32 pixels")

    def estimate(self, previous: RGBFrame | None, current: RGBFrame) -> FlowResult:
        """Return forward flow at the input frame resolution.

        Frame IDs can skip. Timestamps must increase on a common origin.
        """
        self._validate(current)
        if previous is None:
            return self._empty(current, None, "no_previous_frame")
        self._validate(previous)
        if previous.source_id != current.source_id:
            raise ValueError("flow frames must have the same source_id")
        if previous.rgb.shape != current.rgb.shape:
            raise ValueError("flow frames must have the same processed dimensions")
        if (
            str(previous.timestamp_origin) != str(current.timestamp_origin)
            or current.frame_id <= previous.frame_id
            or current.capture_timestamp_ns <= previous.capture_timestamp_ns
        ):
            return self._empty(current, previous, "invalid_timestamps")

        mean_difference = float(np.mean(cv2.absdiff(previous.rgb, current.rgb))) / 255.0
        if mean_difference >= self.config.scene_cut_mean_difference:
            return self._empty(current, previous, "scene_cut")

        height, width = current.rgb.shape[:2]
        small_width, small_height = self._size(width, height)
        size = (small_width, small_height)
        gray_previous = cv2.cvtColor(cv2.resize(previous.rgb, size, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
        gray_current = cv2.cvtColor(cv2.resize(current.rgb, size, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
        forward = self._forward.calc(gray_previous, gray_current, None)
        backward = self._backward.calc(gray_current, gray_previous, None)

        # DIS returns displacements in its input pixels. Upscale both vector
        # components separately before computing validity at output resolution.
        flow = cv2.resize(forward, (width, height), interpolation=cv2.INTER_LINEAR)
        reverse = cv2.resize(backward, (width, height), interpolation=cv2.INTER_LINEAR)
        flow = np.ascontiguousarray(flow, dtype=np.float32)
        reverse = np.ascontiguousarray(reverse, dtype=np.float32)
        flow[..., 0] *= width / small_width
        flow[..., 1] *= height / small_height
        reverse[..., 0] *= width / small_width
        reverse[..., 1] *= height / small_height

        valid = forward_backward_valid_mask(flow, reverse, self.config.forward_backward_threshold_px)
        return FlowResult(
            status="ok",
            source_id=current.source_id,
            previous_frame_id=previous.frame_id,
            current_frame_id=current.frame_id,
            previous_capture_timestamp_ns=previous.capture_timestamp_ns,
            current_capture_timestamp_ns=current.capture_timestamp_ns,
            timestamp_origin=str(current.timestamp_origin),
            elapsed_ns=current.capture_timestamp_ns - previous.capture_timestamp_ns,
            flow=flow,
            valid_mask=valid,
            flow_width=small_width,
            flow_height=small_height,
        )


class OpticalFlowTracker:
    """Keep exactly one previous frame; accept gaps from latest-frame-wins queues."""

    def __init__(self, estimator: FlowEstimator | None = None) -> None:
        self.estimator = estimator or DISFlowEstimator()
        self._previous: RGBFrame | None = None

    def push(self, frame: RGBFrame) -> FlowResult:
        previous = self._previous
        if previous is not None and (
            previous.source_id != frame.source_id
            or previous.rgb.shape != frame.rgb.shape
        ):
            previous = None
        result = self.estimator.estimate(previous, frame)
        self._previous = frame
        return result

    def reset(self) -> None:
        self._previous = None


class FlowEstimator(Protocol):
    def estimate(self, previous: RGBFrame | None, current: RGBFrame) -> FlowResult: ...


def forward_backward_valid_mask(flow: np.ndarray, reverse: np.ndarray, threshold_px: float) -> np.ndarray:
    """Check whether a forward vector returns near its starting pixel."""
    height, width = flow.shape[:2]
    x, y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    target_x = x + flow[..., 0]
    target_y = y + flow[..., 1]
    reverse_at_target = cv2.remap(reverse, target_x, target_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    residual = np.linalg.norm(flow + reverse_at_target, axis=2)
    return (
        (target_x >= 0)
        & (target_x <= width - 1)
        & (target_y >= 0)
        & (target_y <= height - 1)
        & np.isfinite(flow).all(axis=2)
        & np.isfinite(reverse_at_target).all(axis=2)
        & (residual <= threshold_px)
    )


def create_estimator(config: FlowConfig | None = None) -> FlowEstimator:
    """Select a backend without importing PyTorch for DIS users."""
    selected = config or FlowConfig()
    if selected.flow_model == "opencv_dis":
        return DISFlowEstimator(selected)
    from .raft import RaftSmallFlowEstimator

    return RaftSmallFlowEstimator(selected)
