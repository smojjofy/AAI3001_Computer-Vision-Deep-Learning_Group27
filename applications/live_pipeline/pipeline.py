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
from .mask_selection import MaskSelection, pose_region_mask, select_person_mask, select_pose_gradient_mask
from .pose_gradient_mask import gate_validity_to_joint_depth
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

    ``reason`` is a short machine-readable class for a failure report. A caller
    groups failures by this value instead of matching message text, so a count
    of failure causes stays correct when a message changes. The classes are:

    - ``pose_depth_error``: the depth or pose model raised.
    - ``pose_missing``: the pose model ran and found no person.
    - ``mask_rejected``: a person was found but the mask area failed its guard.
    - ``construction_rejected``: the adapter rejected the construction frame.
    - ``construction_failed``: the native constructor raised.
    - ``depth_invalid``: the depth plane had too little usable value.
    - ``other``: anything without a specific class.
    """

    def __init__(
        self,
        message: str,
        *,
        person_missing: bool = False,
        reason: str = "other",
    ) -> None:
        super().__init__(message)
        self.person_missing = bool(person_missing)
        self.reason = str(reason)


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
    # The unrounded joints and joint depths that were passed to the mask
    # selector for this frame. ``pose_joints`` above comes from the preview
    # encoder, which rounds x, y, and visibility for display.
    mask_pose_joints: list[dict[str, Any]] | None = None
    pose_confidence: float = 0.0
    depth_source: str = "yolo"
    # Face depth correction. ``face_depth_info`` is None when no rule ran.
    face_depth_info: dict[str, Any] | None = None
    face_depth_timings_ms: dict[str, float] = field(default_factory=dict)
    # Stereo matching record for this frame. None when stereo was not enabled,
    # otherwise it holds the match decision and, on a fallback, the reason.
    stereo_info: dict[str, Any] | None = None

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
    depth_confidence: bool = False
    generation: int = 0
    # Off by default. The rule runs after mask selection and before
    # construction, and it changes only finite, positive, already-valid pixels
    # inside the person mask.
    face_depth_rule: Any = None
    # Stereo depth matching, off by default. It needs a packet that carries a
    # rectified right view and a stereo calibration. When the frame keeps the
    # match, the published depth source is `stereo_sgbm`; when the match is
    # unusable the whole frame uses the YOLO estimate and the reason is recorded.
    stereo_enabled: bool = False
    stereo_config: Any = None
    stereo_fallback: Any = None
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
            kwargs: dict[str, Any] = {
                "device": self.device,
                "imgsz": self.yolo_imgsz,
                # The live mask path reads depth and validity only. The per-pixel
                # edge-confidence plane costs milliseconds and is unused here.
                "compute_confidence": self.depth_confidence,
            }
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

    def capture_background_depth(
        self, packet, generation: int | None = None
    ) -> tuple[np.ndarray, bool, str]:
        """Run pose and depth for one calibration frame.

        Returns ``(depth_m, person_detected, depth_source)``. Calibration runs
        while the scene is empty, so the pose requirement does not apply. A
        detected person is reported so the caller can reject that background.
        This does not build a mask or a splat state.
        """
        intrinsics = self._ensure_intrinsics(packet)
        pipeline = self._ensure_pipeline()
        depth_input = None
        depth_source = None
        if self.stereo_enabled:
            depth_input, depth_source, stereo_info = self._stereo_depth_for(packet)
            if depth_input is None:
                reason = stereo_info.get("fallback_reason", "stereo_depth_unusable")
                raise RefreshError(f"stereo calibration depth is unusable: {reason}", reason="depth_invalid")
        elif packet.depth is not None:
            depth_input = depth_to_metres(packet.depth, packet.depth_units)
        result = pipeline.process_frame(
            packet.rgb,
            packet.frame_id,
            packet.source_id,
            packet.capture_timestamp_ns,
            depth=depth_input,
            **({"depth_source": depth_source} if depth_source is not None else {}),
        )
        detected = bool(getattr(result, "detected", False))
        if depth_input is not None:
            # A source that already supplies depth returns that depth.
            return np.asarray(depth_input, dtype=np.float32), detected, str(result.depth_source)
        return np.asarray(result.depth.depth, dtype=np.float32), detected, str(result.depth_source)

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
        supplied_source = None
        stereo_info: dict[str, Any] | None = None
        if packet.depth is not None:
            depth_input = depth_to_metres(packet.depth, packet.depth_units)
            depth_units = "metres"
        elif self.stereo_enabled:
            # Stereo replaces the supplied depth with a match from this pair.
            # The decision is whole-frame: either this frame keeps the match, or
            # it uses the YOLO estimate, never a mix of the two.
            depth_input, supplied_source, stereo_info = self._stereo_depth_for(packet)
        try:
            process_kwargs: dict[str, Any] = {"depth": depth_input}
            if supplied_source is not None:
                # Only a caller that names a source passes the keyword, so a
                # depth stack or a test double with the older signature still
                # works and keeps its historical label.
                process_kwargs["depth_source"] = supplied_source
            result = pipeline.process_frame(
                packet.rgb,
                packet.frame_id,
                packet.source_id,
                packet.capture_timestamp_ns,
                **process_kwargs,
            )
        except Exception as error:  # the caller keeps the active state
            self.failure_count += 1
            self.last_error = f"pose/depth failed: {error!r}"
            raise RefreshError(self.last_error, reason="pose_depth_error") from error
        t_depth = perf_counter()

        if self.require_pose and not bool(result.detected):
            # CONTEXT.md: a refresh without a valid pose keeps the last valid
            # state. Do not build a person state from background splats.
            self.failure_count += 1
            self.last_error = "pose not detected in the selected refresh frame"
            raise RefreshError(self.last_error, person_missing=True, reason="pose_missing")

        depth_m = np.asarray(result.depth.depth, dtype=np.float32)
        pose_joints_for_mask, mask_selection, mask_info = self._mask_for_result(
            result, depth_m, packet, job_generation
        )
        if stereo_info is not None and stereo_info.get("status") == "used":
            _import_depth_stack()
            from src.pipeline import StereoFallbackConfig, stereo_frame_usable

            person_region = (
                pose_region_mask(pose_joints_for_mask, width=packet.width, height=packet.height)
                if pose_joints_for_mask is not None else None
            )
            valid = np.isfinite(depth_m) & (depth_m > 0.0)
            if person_region is None:
                usable, reason, detail = False, "pose_region_unavailable", {"person_pixels": 0}
            else:
                usable, reason, detail = stereo_frame_usable(
                    depth_m, valid, person_mask=person_region,
                    config=self.stereo_fallback or StereoFallbackConfig(),
                )
            stereo_info["person_decision"] = detail
            if usable and not mask_info.detected:
                usable, reason = False, "mask_rejected"
            if not usable:
                stereo_info.update({"status": "fallback", "fallback_reason": reason})
                try:
                    result = pipeline.replace_depth_with_yolo(packet.rgb, result)
                except Exception as error:
                    self.failure_count += 1
                    self.last_error = f"YOLO fallback failed: {error!r}"
                    raise RefreshError(self.last_error, reason="pose_depth_error") from error
                t_depth = perf_counter()
                depth_m = np.asarray(result.depth.depth, dtype=np.float32)
                pose_joints_for_mask, mask_selection, mask_info = self._mask_for_result(
                    result, depth_m, packet, job_generation
                )
        if not mask_info.detected:
            self.failure_count += 1
            reason = "mask area {:.3f} outside [{:.3f}, {:.3f}]".format(
                mask_info.area_fraction,
                self.mask_config.min_area_fraction,
                self.mask_config.max_area_fraction,
            )
            self.last_error = f"depth threshold found no person region ({reason})"
            raise RefreshError(self.last_error, person_missing=True, reason="mask_rejected")
        t_mask = perf_counter()

        # Face depth correction. It is off unless a rule is set. It runs after
        # mask selection and before construction above, so the mask and the
        # validity that construction reads are the corrected values.
        face_depth_info: dict[str, Any] | None = None
        face_depth_timings: dict[str, float] = {}
        if self.face_depth_rule is not None:
            face_depth_info, face_depth_timings, depth_m, mask_info = self._correct_face_depth(
                depth_m, mask_info, pose_joints_for_mask
            )

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
            raise RefreshError(self.last_error, reason="construction_rejected") from error

        from modules.splat_constructor import ConstructionConfig, construct_splat_state

        try:
            state = construct_splat_state(frame, ConstructionConfig(sample_stride=self.sample_stride))
            header, buffer = splat_state_payload(
            state, generation=job_generation, source_frame_id=int(packet.frame_id)
        )
        except Exception as error:
            self.failure_count += 1
            self.last_error = f"native construction failed: {error!r}"
            raise RefreshError(self.last_error, reason="construction_failed") from error
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
        if face_depth_timings:
            # The correction runs inside the mask stage. Report it as its own
            # stage so its cost stays measurable, and take it out of "mask" so
            # the stage sum is still one total.
            correction_ms = float(face_depth_timings.get("face_depth", 0.0))
            timings["mask"] = max(0.0, timings["mask"] - correction_ms)
            timings["face_depth"] = correction_ms
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
            mask_pose_joints=pose_joints_for_mask,
            pose_confidence=float(pose_confidence),
            depth_source=str(result.depth_source),
            face_depth_info=face_depth_info,
            face_depth_timings_ms=face_depth_timings,
            stereo_info=stereo_info if self.stereo_enabled else None,
        )

    def _mask_for_result(
        self, result, depth_m: np.ndarray, packet, generation: int
    ) -> tuple[list[dict[str, Any]] | None, MaskSelection, DepthThresholdMask]:
        pose_joints = None
        try:
            joint_depths = getattr(result, "joint_depth", None)
            pose_joints = []
            for index, landmark in enumerate(result.pose.landmarks):
                depth_value = None
                if joint_depths is not None and index < len(joint_depths):
                    candidate = joint_depths[index]
                    if candidate is not None and np.isfinite(candidate):
                        depth_value = float(candidate)
                pose_joints.append({
                    "index": index,
                    "x": float(landmark.x),
                    "y": float(landmark.y),
                    "visibility": None if landmark.visibility is None else float(landmark.visibility),
                    "depth_m": depth_value,
                })
        except (AttributeError, TypeError, ValueError):
            pose_joints = None

        baseline = self.background_baseline
        if self.stereo_enabled and baseline is not None and baseline.depth_source != result.depth_source:
            selection = MaskSelection(
                depth_threshold_mask(depth_m, self.mask_config),
                "depth_threshold_fallback", "baseline_depth_source_mismatch",
            )
        elif self.stereo_enabled and baseline is not None:
            selection = select_person_mask(
                depth_m, baseline=baseline, source_id=packet.source_id,
                generation=generation, config=self.mask_config, pose_joints=pose_joints,
            )
        else:
            selection = select_pose_gradient_mask(
                depth_m, pose_joints=pose_joints, config=self.mask_config,
            )
        mask_info = selection.info
        if selection.source == "pose_gradient":
            gated_validity = gate_validity_to_joint_depth(mask_info.mask, depth_m, pose_joints)
            mask_info = DepthThresholdMask(
                mask=mask_info.mask,
                validity=gated_validity,
                near_m=mask_info.near_m,
                far_m=mask_info.far_m,
                valid_fraction=mask_info.valid_fraction,
                area_fraction=mask_info.area_fraction,
                confidence=mask_info.confidence,
            )
        return pose_joints, selection, mask_info

    def _stereo_depth_for(
        self, packet
    ) -> tuple[np.ndarray | None, str | None, dict[str, Any]]:
        """Match this packet's pair, or decide that the frame uses YOLO depth.

        Returns ``(depth_m, source_label, info)``. A ``None`` depth means the
        caller should let the pipeline estimate depth with YOLO, and
        ``info["fallback_reason"]`` says why.

        The frame is matched before the person region is known. ``process``
        checks that region after pose and mask selection. It retries with YOLO
        depth for the whole frame if that later check fails.
        """
        info: dict[str, Any] = {"status": "not_attempted"}
        right = getattr(packet, "right_rgb", None)
        calibration = getattr(packet, "stereo", None)
        if right is None or calibration is None:
            info.update({"status": "fallback", "fallback_reason": "no_stereo_pair"})
            return None, None, info
        if not calibration.matches(packet.width, packet.height):
            info.update({"status": "fallback", "fallback_reason": "calibration_size_mismatch"})
            return None, None, info
        # The depth package is imported lazily, matching how the pose and depth
        # stack is loaded: the package is used without being installed, so its
        # parent directory is only on sys.path after _import_depth_stack runs.
        _import_depth_stack()
        from src.pipeline import StereoFallbackConfig
        from src.pipeline import stereo_frame_usable
        from src.stereo_depth import StereoMatcherConfig, stereo_depth

        matcher_config = self.stereo_config if self.stereo_config is not None else StereoMatcherConfig()
        started = perf_counter()
        try:
            depth, match_info = stereo_depth(
                packet.rgb,
                right,
                fx=calibration.fx,
                baseline_m=calibration.baseline_m,
                config=matcher_config,
            )
        except Exception as error:  # a match fault must not stop the live run
            info.update({"status": "fallback", "fallback_reason": "match_error", "error": repr(error)})
            return None, None, info
        elapsed_ms = (perf_counter() - started) * 1000.0
        valid = np.isfinite(depth) & (depth > 0.0)
        fallback_config = self.stereo_fallback if self.stereo_fallback is not None else StereoFallbackConfig()
        usable, reason, detail = stereo_frame_usable(depth, valid, config=fallback_config)
        info.update(
            {
                "status": "used" if usable else "fallback",
                "match_ms": round(elapsed_ms, 3),
                "match": match_info,
                "decision": detail,
            }
        )
        if not usable:
            info["fallback_reason"] = reason
            return None, None, info
        return depth, "stereo_sgbm", info

    def _correct_face_depth(
        self,
        depth_m: np.ndarray,
        mask_info: DepthThresholdMask,
        pose_joints: list[dict[str, Any]] | None,
    ) -> tuple[dict[str, Any], dict[str, float], np.ndarray, DepthThresholdMask]:
        """Run the face depth rule and rebuild the mask record from its result.

        The rule owns the mask and validity invariants: the mask is unchanged and
        the validity never increases. This method checks both and reports a
        violation in ``info`` instead of raising, so one bad frame cannot stop
        the live run.
        """
        from scripts.face_depth_test import correct_face_depth

        started = perf_counter()
        result = correct_face_depth(
            depth_m,
            np.asarray(mask_info.validity, dtype=np.float32),
            np.asarray(mask_info.mask),
            pose_joints,
            self.face_depth_rule,
        )
        elapsed_ms = (perf_counter() - started) * 1000.0
        info = dict(result.info)
        mask_unchanged = bool(np.array_equal(np.asarray(result.mask), np.asarray(mask_info.mask)))
        validity_ok = bool(
            np.all(np.asarray(result.validity) <= np.asarray(mask_info.validity) + np.float32(1e-6))
        )
        info["pipeline_checks"] = {
            "mask_unchanged": mask_unchanged,
            "validity_not_increased": validity_ok,
        }
        if not (mask_unchanged and validity_ok):
            # Keep the accepted mask and depth if the rule broke an invariant.
            info["applied"] = False
            info["reason"] = "invariant_violation"
            return info, {"face_depth": round(elapsed_ms, 3)}, depth_m, mask_info
        corrected_depth = np.asarray(result.depth_m, dtype=np.float32)
        corrected_mask = DepthThresholdMask(
            mask=mask_info.mask,
            validity=np.asarray(result.validity, dtype=np.float32),
            near_m=mask_info.near_m,
            far_m=mask_info.far_m,
            valid_fraction=mask_info.valid_fraction,
            area_fraction=mask_info.area_fraction,
            confidence=mask_info.confidence,
        )
        return info, {"face_depth": round(elapsed_ms, 3)}, corrected_depth, corrected_mask

    def close(self) -> None:
        if self._pipeline is not None:
            self._pipeline.close()
            self._pipeline = None
