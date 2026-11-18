"""One refresh: pose + depth -> depth-threshold mask -> native constructor.

The pipeline is intentionally synchronous and single-frame. The caller owns the
thread and the active-state logic. This keeps the GPU work in one place and makes
the refresh cost measurable.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic_ns, perf_counter
from typing import Any

import numpy as np

from .background_calibration import BackgroundBaseline
from .camera_config import CameraConfig
from .mask_selection import select_person_mask
from .telemetry import capture_age_ms as capture_age_ms_at, stage_total

from .adapters import (
    DepthThresholdConfig,
    DepthThresholdMask,
    construction_frame_from_refresh,
    depth_threshold_mask,
    depth_to_metres,
    splat_state_payload,
    webcam_intrinsics,
)

ROOT = Path(__file__).resolve().parents[2]
DEPTH_ROOT = ROOT / "modules" / "inference" / "depth"


class RefreshError(RuntimeError):
    """A refresh cannot produce a valid state; the caller keeps the active state.

    ``person_missing`` is True when the selected frame has no valid person pose
    or no separable person mask. That is the invalid-person event from
    `CONTEXT.md` that starts the 500 ms hold and fade. Other failures (model
    load, native construction) are recorded but do not start that timer.
    """

    def __init__(self, message: str, *, person_missing: bool = False) -> None:
        super().__init__(message)
        self.person_missing = bool(person_missing)


@dataclass(frozen=True)
class RefreshOutcome:
    """Everything one refresh produced, with identity preserved.

    ``pose_joints`` lives here, not beside the state holder, so a splat state
    and the pose joints that belong to it are always published together.
    ``person_missing`` is True only for pose or mask absence, which is the
    invalid-person event that starts the hold-then-fade timer.
    """

    frame_id: int
    source_id: str
    capture_timestamp_ns: int
    generation: int
    state: Any
    header: dict[str, Any]
    buffer: bytes
    mask: DepthThresholdMask
    depth_m: np.ndarray
    timings_ms: dict[str, float]
    capture_age_ms: float | None = None
    pose_preview: dict[str, Any] | None = None
    pose_joints: list[dict[str, Any]] | None = None
    pose_confidence: float = 0.0
    depth_source: str = "yolo"

    @property
    def splat_count(self) -> int:
        return int(self.header["count"])


def _import_depth_stack():
    """Import the frozen depth/pose package without installing it.

    The package is imported as ``src`` from ``modules/inference/depth``, matching
    the module's own run scripts. ``PoseDepthPipeline`` adds the pose package to
    ``sys.path`` itself.
    """
    depth_parent = str(DEPTH_ROOT)
    if depth_parent not in sys.path:
        sys.path.insert(0, depth_parent)
    from src import CameraIntrinsics, PoseDepthPipeline  # type: ignore[import-not-found]
    from src.preview import encode_pose_depth_preview  # type: ignore[import-not-found]

    return CameraIntrinsics, PoseDepthPipeline, encode_pose_depth_preview


@dataclass
class RefreshPipeline:
    """Build one person splat state per selected refresh frame."""

    mask_config: DepthThresholdConfig = field(default_factory=DepthThresholdConfig)
    sample_stride: int = 2
    horizontal_fov_deg: float = 60.0
    camera_config: CameraConfig | None = None
    background_baseline: BackgroundBaseline | None = None
    yolo_model: str | None = None
    yolo_imgsz: int = 640
    device: str = "auto"
    require_pose: bool = True
    generation: int = 0
    _pipeline: Any = field(default=None, init=False, repr=False)
    _intrinsics: tuple[float, float, float, float] | None = field(default=None, init=False, repr=False)
    _intrinsics_key: tuple[int, int, tuple[float, float, float, float] | None] | None = field(
        default=None, init=False, repr=False
    )
    refresh_count: int = field(default=0, init=False)
    failure_count: int = field(default=0, init=False)
    last_error: str | None = field(default=None, init=False)

    def _ensure_pipeline(self):
        if self._pipeline is None:
            CameraIntrinsics, PoseDepthPipeline, _ = _import_depth_stack()
            kwargs: dict[str, Any] = {"device": self.device, "imgsz": self.yolo_imgsz}
            if self.yolo_model is not None:
                kwargs["yolo_model"] = self.yolo_model
            self._pipeline = PoseDepthPipeline(**kwargs)
        return self._pipeline

    def configure_for_frame(self, width: int, height: int, intrinsics: tuple[float, float, float, float] | None) -> None:
        """Use capture intrinsics when present, else a fixed field-of-view estimate."""
        if intrinsics is not None:
            self._intrinsics = tuple(float(value) for value in intrinsics)  # type: ignore[assignment]
        else:
            self._intrinsics = webcam_intrinsics(width, height, self.horizontal_fov_deg)
        self._intrinsics_key = self._intrinsics_key_for(width, height, intrinsics)

    @staticmethod
    def _intrinsics_key_for(
        width: int,
        height: int,
        intrinsics: tuple[float, float, float, float] | None,
    ) -> tuple[int, int, tuple[float, float, float, float] | None]:
        if intrinsics is None:
            return (int(width), int(height), None)
        return (int(width), int(height), tuple(float(value) for value in intrinsics))

    def _ensure_intrinsics(self, packet) -> tuple[float, float, float, float]:
        """Select intrinsics for this packet; re-derive them when the identity changes.

        A resolution change, a calibration change, or a source restart must not
        keep focal lengths from an earlier image size.
        """
        intrinsics_arg = packet.intrinsics
        if (
            intrinsics_arg is None
            and self.camera_config is not None
            and self.camera_config.matches(packet.width, packet.height)
        ):
            # A calibrated camera supplies pinhole intrinsics for this
            # resolution. A different resolution falls back to the estimate.
            intrinsics_arg = self.camera_config.intrinsics
        key = self._intrinsics_key_for(packet.width, packet.height, intrinsics_arg)
        if self._intrinsics is None or self._intrinsics_key != key:
            self.configure_for_frame(packet.width, packet.height, intrinsics_arg)
        intrinsics = self._intrinsics
        assert intrinsics is not None
        return intrinsics

    def set_background_baseline(self, baseline: BackgroundBaseline | None) -> None:
        """Use an empty-scene background baseline for later masks.

        The baseline is only applied when its source, generation, and image size
        match the refresh frame, so a stale scene cannot leak into a new run.
        """
        self.background_baseline = baseline

    def capture_background_depth(self, packet, generation: int | None = None) -> tuple[np.ndarray, bool]:
        """Run pose and depth for one calibration frame.

        Returns ``(depth_m, person_detected)``. Calibration runs while the scene
        is empty, so the pose requirement does not apply, but a detected person
        is reported so the caller can refuse to learn a person as background.
        This does not build a mask or a splat state.
        """
        intrinsics = self._ensure_intrinsics(packet)
        pipeline = self._ensure_pipeline()
        depth_input = None
        depth_units = None
        if packet.depth is not None:
            depth_input = depth_to_metres(packet.depth, packet.depth_units)
            depth_units = "metres"
        result = pipeline.process_frame(
            packet.rgb,
            packet.frame_id,
            packet.source_id,
            packet.capture_timestamp_ns,
            depth=depth_input,
        )
        detected = bool(getattr(result, "detected", False))
        if depth_input is not None:
            # A source that already supplies depth returns that depth.
            return np.asarray(depth_input, dtype=np.float32), detected
        return np.asarray(result.depth.depth, dtype=np.float32), detected

    def process(self, packet, generation: int | None = None) -> RefreshOutcome:
        """Run pose, depth, mask, and construction for one selected frame.

        `generation` is the capture run of this request. It is stamped on the
        outcome and never read again from mutable state, so a slow refresh that
        finishes after a run change cannot claim the newer generation.
        """
        started = perf_counter()
        job_generation = int(self.generation if generation is None else generation)
        intrinsics = self._ensure_intrinsics(packet)

        pipeline = self._ensure_pipeline()
        depth_input = None
        depth_units = None
        if packet.depth is not None:
            depth_input = depth_to_metres(packet.depth, packet.depth_units)
            depth_units = "metres"
        try:
            result = pipeline.process_frame(
                packet.rgb,
                packet.frame_id,
                packet.source_id,
                packet.capture_timestamp_ns,
                depth=depth_input,
            )
        except Exception as error:  # the caller keeps the active state
            self.failure_count += 1
            self.last_error = f"pose/depth failed: {error!r}"
            raise RefreshError(self.last_error) from error
        t_depth = perf_counter()

        if self.require_pose and not bool(result.detected):
            # CONTEXT.md: a refresh without a valid pose keeps the last valid
            # state. Do not build a person state from background splats.
            self.failure_count += 1
            self.last_error = "pose not detected in the selected refresh frame"
            raise RefreshError(self.last_error, person_missing=True)

        depth_m = np.asarray(result.depth.depth, dtype=np.float32)
        pose_joints_for_mask = None
        try:
            pose_joints_for_mask = [
                {
                    "x": float(landmark.x),
                    "y": float(landmark.y),
                    "visibility": None if landmark.visibility is None else float(landmark.visibility),
                }
                for landmark in result.pose.landmarks
            ]
        except (AttributeError, TypeError, ValueError):
            pose_joints_for_mask = None
        mask_selection = select_person_mask(
            depth_m,
            baseline=self.background_baseline,
            source_id=packet.source_id,
            generation=job_generation,
            config=self.mask_config,
            pose_joints=pose_joints_for_mask,
        )
        mask_info = mask_selection.info
        if not mask_info.detected:
            self.failure_count += 1
            reason = "mask area {:.3f} outside [{:.3f}, {:.3f}]".format(
                mask_info.area_fraction,
                self.mask_config.min_area_fraction,
                self.mask_config.max_area_fraction,
            )
            self.last_error = f"depth threshold found no person region ({reason})"
            raise RefreshError(self.last_error, person_missing=True)
        t_mask = perf_counter()

        try:
            frame = construction_frame_from_refresh(
                rgb=packet.rgb,
                depth_m=depth_m,
                foreground_mask=mask_info.mask,
                frame_id=packet.frame_id,
                timestamp_ns=packet.capture_timestamp_ns,
                source_id=packet.source_id,
                intrinsics=intrinsics,
                depth_validity=mask_info.validity,
            )
        except Exception as error:
            self.failure_count += 1
            self.last_error = f"construction frame rejected: {error!r}"
            raise RefreshError(self.last_error) from error

        from modules.splat_constructor import ConstructionConfig, construct_splat_state

        try:
            state = construct_splat_state(frame, ConstructionConfig(sample_stride=self.sample_stride))
            header, buffer = splat_state_payload(
            state, generation=job_generation, source_frame_id=int(packet.frame_id)
        )
        except Exception as error:
            self.failure_count += 1
            self.last_error = f"native construction failed: {error!r}"
            raise RefreshError(self.last_error) from error
        t_build = perf_counter()

        pose_preview = None
        pose_confidence = 0.0
        try:
            _, _, encode_pose_depth_preview = _import_depth_stack()
            pose_preview = encode_pose_depth_preview(packet.rgb, result)
            joints = pose_preview.get("joints") or []
            visibilities = [
                float(joint["visibility"])
                for joint in joints
                if joint.get("visibility") is not None
            ]
            pose_confidence = float(np.mean(visibilities)) if visibilities else 0.0
        except Exception:
            pose_preview = None
        t_preview = perf_counter()

        # Latency is the sum of every processing stage, so the logged total is
        # the stage sum and not an independent clock reading that can drift.
        timings = {
            "pose_depth": (t_depth - started) * 1000.0,
            "mask": (t_mask - t_depth) * 1000.0,
            "construct": (t_build - t_mask) * 1000.0,
            "pose_preview": (t_preview - t_build) * 1000.0,
        }
        timings["total"] = stage_total(timings)
        capture_age_ms = capture_age_ms_at(
            packet.capture_timestamp_ns,
            monotonic_ns(),
            getattr(packet, "timestamp_origin", None),
        )
        header["timings_ms"] = {key: round(value, 2) for key, value in timings.items()}
        header["capture_age_ms"] = None if capture_age_ms is None else round(capture_age_ms, 2)
        header["image_width"] = int(packet.width)
        header["image_height"] = int(packet.height)
        header["intrinsics"] = [float(value) for value in intrinsics]
        header["pose_confidence"] = round(pose_confidence, 3)
        header["person_detected"] = bool(result.detected)
        header["mask"] = {
            "near_m": None if not np.isfinite(mask_info.near_m) else round(float(mask_info.near_m), 3),
            "far_m": None if not np.isfinite(mask_info.far_m) else round(float(mask_info.far_m), 3),
            "area_fraction": round(float(mask_info.area_fraction), 4),
            "valid_fraction": round(float(mask_info.valid_fraction), 4),
            "confidence": round(float(mask_info.confidence), 3),
            "detected": bool(mask_info.detected),
        }
        header["depth_source"] = result.depth_source
        header["depth_units_metres"] = True
        header["mask"]["source"] = mask_selection.source
        header["mask"]["fallback_reason"] = mask_selection.fallback_reason
        header["mask"]["background_area_fraction"] = mask_selection.background_area_fraction
        pose_joints = None
        if pose_preview is not None:
            candidate_joints = pose_preview.get("joints")
            pose_joints = list(candidate_joints) if isinstance(candidate_joints, list) else None
        self.refresh_count += 1
        return RefreshOutcome(
            frame_id=packet.frame_id,
            source_id=packet.source_id,
            capture_timestamp_ns=packet.capture_timestamp_ns,
            generation=job_generation,
            state=state,
            header=header,
            buffer=buffer,
            mask=mask_info,
            depth_m=depth_m,
            timings_ms=timings,
            capture_age_ms=capture_age_ms,
            pose_preview=pose_preview,
            pose_joints=pose_joints,
            pose_confidence=float(pose_confidence),
            depth_source=str(result.depth_source),
        )

    def close(self) -> None:
        if self._pipeline is not None:
            self._pipeline.close()
            self._pipeline = None
