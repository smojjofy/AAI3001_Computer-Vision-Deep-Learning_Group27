"""Optional Torchvision RAFT Small backend for the common FlowResult contract."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .flow import DISFlowEstimator, FlowConfig, FlowResult, RGBFrame, forward_backward_valid_mask


class RaftSmallFlowEstimator:
    """Pretrained or locally fine-tuned RAFT Small inference.

    PyTorch imports are lazy so the DIS-only installation stays lightweight.
    """

    def __init__(self, config: FlowConfig) -> None:
        if config.flow_model != "raft_small":
            raise ValueError("RaftSmallFlowEstimator requires flow_model=raft_small")
        try:
            import torch
            from torchvision.models.optical_flow import Raft_Small_Weights, raft_small
        except ImportError as exc:
            raise RuntimeError("RAFT Small requires compatible torch and torchvision; install the optical-flow raft extra") from exc

        self.config = config
        self._torch = torch
        if config.raft_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("raft_device=cuda requested but CUDA is unavailable")
        self.device = "cuda" if config.raft_device == "auto" and torch.cuda.is_available() else config.raft_device
        if self.device == "auto":
            self.device = "cpu"

        weights = Raft_Small_Weights.DEFAULT if config.raft_weights == "default" and config.raft_checkpoint is None else None
        self._transform = Raft_Small_Weights.DEFAULT.transforms()
        model = raft_small(weights=weights, progress=False)
        if config.raft_checkpoint is not None:
            checkpoint = Path(config.raft_checkpoint)
            if not checkpoint.is_file():
                raise FileNotFoundError(f"RAFT checkpoint not found: {checkpoint}")
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            model.load_state_dict(state)
        self._model = model.to(self.device).eval()

    def _empty(self, current: RGBFrame, previous: RGBFrame | None, status: str) -> FlowResult:
        height, width = current.rgb.shape[:2]
        return FlowResult(
            status=status,
            source_id=current.source_id,
            previous_frame_id=previous.frame_id if previous is not None else None,
            current_frame_id=current.frame_id,
            previous_capture_timestamp_ns=previous.capture_timestamp_ns if previous is not None else None,
            current_capture_timestamp_ns=current.capture_timestamp_ns,
            timestamp_origin=str(current.timestamp_origin),
            elapsed_ns=None,
            flow=np.zeros((height, width, 2), dtype=np.float32),
            valid_mask=np.zeros((height, width), dtype=np.bool_),
            flow_width=min(width, self.config.flow_width),
            flow_height=min(height, self.config.flow_height),
            backend="raft_small",
        )

    def _size(self, width: int, height: int) -> tuple[int, int]:
        scale = min(1.0, self.config.flow_width / width, self.config.flow_height / height)
        # Torchvision RAFT needs both axes >=128 and divisible by 8.
        return max(128, round(width * scale / 8) * 8), max(128, round(height * scale / 8) * 8)

    def _tensor(self, rgb: np.ndarray, size: tuple[int, int]):
        interpolation = cv2.INTER_AREA if size[0] <= rgb.shape[1] and size[1] <= rgb.shape[0] else cv2.INTER_LINEAR
        resized = cv2.resize(rgb, size, interpolation=interpolation)
        chw = np.ascontiguousarray(resized.transpose(2, 0, 1))
        return self._torch.from_numpy(chw).unsqueeze(0)

    def _predict(self, first, second) -> np.ndarray:
        with self._torch.inference_mode():
            prediction = self._model(first, second, num_flow_updates=self.config.raft_iterations)[-1]
        return np.ascontiguousarray(prediction[0].permute(1, 2, 0).cpu().numpy(), dtype=np.float32)

    def estimate(self, previous: RGBFrame | None, current: RGBFrame) -> FlowResult:
        DISFlowEstimator._validate(current)
        if previous is None:
            return self._empty(current, None, "no_previous_frame")
        DISFlowEstimator._validate(previous)
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
        first = self._tensor(previous.rgb, size)
        second = self._tensor(current.rgb, size)
        if self._transform is not None:
            first, second = self._transform(first, second)
        else:
            first = first.float().div(127.5).sub(1)
            second = second.float().div(127.5).sub(1)
        first = first.to(self.device)
        second = second.to(self.device)
        forward = self._predict(first, second)
        backward = self._predict(second, first)

        flow = np.ascontiguousarray(cv2.resize(forward, (width, height)), dtype=np.float32)
        reverse = np.ascontiguousarray(cv2.resize(backward, (width, height)), dtype=np.float32)
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
            backend="raft_small",
        )
