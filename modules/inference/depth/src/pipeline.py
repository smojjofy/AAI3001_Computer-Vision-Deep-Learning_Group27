"""One pipeline for pose + depth: MediaPipe Pose Lite + (ZED depth or YOLO depth).

The frame processor calls process_frame() once per frame:

    with PoseDepthPipeline() as pipe:
        out = pipe.process_frame(rgb, frame_id, source_id, capture_timestamp_ns)              # YOLO depth
        out = pipe.process_frame(rgb, frame_id, source_id, capture_timestamp_ns,
                                 depth=zed_depth_m)                                          # ZED depth

        out.pose.landmarks     # 33 joints (x, y in 0..1), for the GUI
        out.depth.depth        # [H, W] float32 metres, NaN = invalid
        out.joint_depth        # [33] metres per joint (NaN = unknown)
        out.body_depth         # one number: how far the person is
        out.joint_xyz          # [33, 3] metres in camera space, if intrinsics were given
        out.depth_source       # "zed" or "yolo"
        out.timings_ms         # {"pose": .., "depth": .., "total": ..}

ZED frames:
  * the image comes as BGRA (sl.Mat VIEW.LEFT): convert with zed_to_rgb() first.
  * depth comes as float32 per pixel, NaN / +inf / -inf where unknown. The ZED SDK
    defaults to MILLIMETRES unless InitParameters.coordinate_units = sl.UNIT.METER;
    pass depth_scale=0.001 if yours is in millimetres.
  * YOLO is only loaded the first time a frame arrives WITHOUT depth, so a pure-ZED
    run never pays for it.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from time import perf_counter

import cv2
import numpy as np

from .contracts import DepthEstimation, DepthFrameResult
from .normalize import normalize_prediction
from .pose_depth import PoseDepthFrameResult, fuse_pose_depth
from .yolo_depth import DEFAULT_YOLO_DEPTH, DEFAULT_YOLO_IMGSZ

DEPTH_ROOT = Path(__file__).resolve().parents[1]                       # .../inference/depth
POSE_ROOT = DEPTH_ROOT.parent / "pose"                                  # .../inference/pose
DEFAULT_POSE_MODEL = POSE_ROOT / "models" / "pose_landmarker_lite.task"


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics in pixels, for the image size the frames arrive at.

    From the ZED: zed.get_camera_information().camera_configuration
                  .calibration_parameters.left_cam -> fx, fy, cx, cy
    """

    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class PipelineResult:
    """Everything the pipeline knows about one frame."""

    frame: PoseDepthFrameResult
    depth_source: str                          # "zed" or "yolo"
    timings_ms: dict
    joint_xyz: np.ndarray | None = field(default=None)   # [N, 3] metres, camera space (x right, y down, z forward)

    # Shortcuts, so callers don't need to know about .frame
    @property
    def frame_id(self) -> int:
        return self.frame.frame_id

    @property
    def source_id(self) -> str:
        return self.frame.source_id

    @property
    def capture_timestamp_ns(self) -> int:
        return self.frame.capture_timestamp_ns

    @property
    def pose(self):
        return self.frame.pose

    @property
    def depth(self) -> DepthEstimation:
        return self.frame.depth

    @property
    def detected(self) -> bool:
        return self.frame.detected

    @property
    def joint_depth(self) -> np.ndarray:
        return self.frame.joint_depth

    @property
    def body_depth(self) -> float:
        return self.frame.body_depth


def zed_to_rgb(image: np.ndarray) -> np.ndarray:
    """ZED / OpenCV colour image (BGRA or BGR, uint8) -> contiguous RGB uint8."""
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"expected an [H, W, 3|4] colour image, got shape {image.shape}")
    code = cv2.COLOR_BGRA2RGB if image.shape[2] == 4 else cv2.COLOR_BGR2RGB
    return np.ascontiguousarray(cv2.cvtColor(image, code))


def external_depth(
    depth: np.ndarray,
    size: tuple[int, int],
    depth_scale: float = 1.0,
    model_name: str = "zed",
    valid_range: tuple[float, float] | None = None,
) -> DepthEstimation:
    """A depth map from another source (ZED) -> the shared depth convention, at `size` (H, W).

    NaN / +inf / -inf / <= 0 become invalid (NaN). Resized with nearest-neighbour so
    no made-up depths appear between foreground and background.
    """
    depth = np.asarray(depth)
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[:, :, 0]
    if depth.ndim != 2:
        raise ValueError(f"depth must be [H, W], got shape {depth.shape}")

    metres = depth.astype(np.float32) * np.float32(depth_scale)
    metres[~np.isfinite(metres) | (metres <= 0)] = np.nan

    height, width = size
    if metres.shape != (height, width):
        metres = cv2.resize(metres, (width, height), interpolation=cv2.INTER_NEAREST)

    return normalize_prediction(metres, "metric_depth", model_name=model_name, valid_range=valid_range)


def joints_to_xyz(result: PoseDepthFrameResult, intrinsics: CameraIntrinsics) -> np.ndarray:
    """Back-project each joint (pixel + its depth) to 3D camera coordinates in metres."""
    if not result.detected:
        return np.zeros((0, 3), np.float32)
    height, width = result.depth.shape
    u = np.array([lm.x * (width - 1) for lm in result.pose.landmarks], np.float32)
    v = np.array([lm.y * (height - 1) for lm in result.pose.landmarks], np.float32)
    z = result.joint_depth.astype(np.float32)
    x = (u - intrinsics.cx) * z / intrinsics.fx
    y = (v - intrinsics.cy) * z / intrinsics.fy
    return np.stack([x, y, z], axis=1)          # NaN rows where the joint has no depth


def _import_pose_estimator():
    """Import the pose package from its source folder.

    The pose package is a sibling module, not a pip dependency, so its source
    folder must be on sys.path. Adding it unconditionally (rather than only
    after a failed import) keeps the import deterministic when another package
    already put a different path on sys.path.
    """
    pose_src = str(POSE_ROOT / "src")
    if pose_src not in sys.path:
        sys.path.insert(0, pose_src)
    from pose_estimator import PoseEstimator
    return PoseEstimator


class PoseDepthPipeline:
    """MediaPipe Pose (Lite) + depth (ZED if given per frame, else YOLO depth)."""

    def __init__(
        self,
        pose_model: str | Path = DEFAULT_POSE_MODEL,
        yolo_model: str | Path = DEFAULT_YOLO_DEPTH,
        device: str = "auto",
        imgsz: int = DEFAULT_YOLO_IMGSZ,
        depth_scale: float = 1.0,
        intrinsics: CameraIntrinsics | None = None,
        joint_radius: int = 2,
        min_visibility: float = 0.5,
        compute_confidence: bool = True,
        pose_estimator=None,
        depth_estimator=None,
    ) -> None:
        """
        pose_model / yolo_model  model files (the YOLO one downloads into models/ if missing)
        device                   for YOLO: auto, cpu, cuda, cuda:N, mps
        imgsz                    YOLO input size: bigger = sharper edges, slower
        depth_scale              multiplies ZED depth to get metres (0.001 if it's in mm)
        intrinsics               camera fx, fy, cx, cy -> also returns 3D joint positions
        joint_radius             window (2*r+1 px square) used to read each joint's depth
        min_visibility           joints MediaPipe is less sure about than this get NaN depth
        compute_confidence       compute the per-pixel depth edge-confidence plane. The
                                 live mask path reads depth and validity only, and the
                                 plane costs several milliseconds per frame; set False
                                 when nothing consumes ``DepthEstimation.confidence``
        pose_estimator / depth_estimator   pass ready-made objects instead (tests, sharing)
        """
        self.pose_model = Path(pose_model)
        self.yolo_model = yolo_model
        self.device = device
        self.imgsz = int(imgsz)
        self.depth_scale = float(depth_scale)
        self.intrinsics = intrinsics
        self.joint_radius = joint_radius
        self.min_visibility = min_visibility
        self.compute_confidence = bool(compute_confidence)

        if pose_estimator is None:
            PoseEstimator = _import_pose_estimator()
            pose_estimator = PoseEstimator(self.pose_model)    # FileNotFoundError if the .task is missing
        self._pose = pose_estimator
        self._depth = depth_estimator                          # YOLO: created lazily

    @property
    def depth_estimator(self):
        """The YOLO depth estimator, loaded on first use."""
        if self._depth is None:
            from .yolo_depth import YoloDepthEstimator
            self._depth = YoloDepthEstimator(
                self.yolo_model,
                device=self.device,
                imgsz=self.imgsz,
                compute_confidence=self.compute_confidence,
            )
        return self._depth

    def process_frame(
        self,
        rgb_frame: np.ndarray,
        frame_id: int,
        source_id: str,
        capture_timestamp_ns: int,
        depth: np.ndarray | DepthEstimation | None = None,
        depth_source: str | None = None,
    ) -> PipelineResult:
        """Pose + depth for one RGB uint8 frame.

        depth: ZED depth for this frame (array, metres x depth_scale, or a DepthEstimation).
               None = estimate it with YOLO depth.
        depth_source: label for a supplied depth plane, for example ``zed`` or
               ``stereo_sgbm``. It is recorded on the result and published in the
               splat header. None keeps the historical ``zed`` label.
        """
        if self._pose is None:
            raise RuntimeError("this PoseDepthPipeline was closed")
        if rgb_frame.ndim != 3 or rgb_frame.shape[2] != 3 or rgb_frame.dtype != np.uint8:
            raise ValueError("rgb_frame must be [H, W, 3] uint8 RGB (use zed_to_rgb() for ZED BGRA images)")
        rgb_frame = np.ascontiguousarray(rgb_frame)
        identity = {"frame_id": frame_id, "source_id": source_id, "capture_timestamp_ns": capture_timestamp_ns}

        t0 = perf_counter()
        pose = self._pose.estimate_frame(rgb_frame, **identity)
        t1 = perf_counter()

        if depth is None:
            depth_frame = self.depth_estimator.estimate_frame(rgb_frame, **identity)
            source = "yolo"
        else:
            # A supplied depth plane is not necessarily ZED SDK depth: it may
            # come from the stereo matcher or a recorded array. Label it with
            # what the caller says it is, and only fall back to "zed" when the
            # caller gives no label. Labelling every external plane "zed"
            # published stereo results under the wrong name.
            if isinstance(depth, DepthEstimation):
                depth_estimation = depth
            else:
                depth_estimation = external_depth(depth, rgb_frame.shape[:2], self.depth_scale)
            if depth_estimation.shape != rgb_frame.shape[:2]:
                raise ValueError(
                    f"depth is {depth_estimation.shape} but the frame is {rgb_frame.shape[:2]}"
                )
            depth_frame = DepthFrameResult(depth=depth_estimation, **identity)
            source = str(depth_source) if depth_source is not None else "zed"
        t2 = perf_counter()

        fused = fuse_pose_depth(pose, depth_frame, radius=self.joint_radius, min_visibility=self.min_visibility)
        xyz = joints_to_xyz(fused, self.intrinsics) if self.intrinsics is not None else None
        t3 = perf_counter()

        timings = {"pose": (t1 - t0) * 1000, "depth": (t2 - t1) * 1000,
                   "fuse": (t3 - t2) * 1000, "total": (t3 - t0) * 1000}
        return PipelineResult(frame=fused, depth_source=source, timings_ms=timings, joint_xyz=xyz)

    def close(self) -> None:
        for part in (self._pose, self._depth):
            if part is not None and hasattr(part, "close"):
                part.close()
        self._pose = self._depth = None

    def __enter__(self) -> "PoseDepthPipeline":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


@dataclass(frozen=True)
class StereoFallbackConfig:
    """When a selected frame keeps its stereo depth and when it uses YOLO.

    The plan requires a whole-frame decision: stereo depth is either used for
    the frame or replaced by YOLO depth, never mixed pixel by pixel in this
    first experiment.

    - ``min_valid_fraction``: the smallest share of pixels that must carry a
      usable stereo depth.
    - ``min_person_fraction``: with a person mask, the smallest share of the
      masked person that must carry a usable stereo depth. The face and the
      hands are the hard cases, so a mask that is mostly unmatched is not
      usable even when the whole frame looks healthy.
    - ``max_depth_m``: optional upper bound. None keeps the plan's literal rule
      that any positive disparity is a valid depth.
    """

    min_valid_fraction: float = 0.15
    min_person_fraction: float = 0.50
    max_depth_m: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_valid_fraction <= 1.0:
            raise ValueError("min_valid_fraction must be in [0, 1]")
        if not 0.0 <= self.min_person_fraction <= 1.0:
            raise ValueError("min_person_fraction must be in [0, 1]")
        if self.max_depth_m is not None and (
            not np.isfinite(self.max_depth_m) or self.max_depth_m <= 0.0
        ):
            raise ValueError("max_depth_m must be positive and finite when set")


def apply_stereo_depth_cap(
    depth: np.ndarray, validity: np.ndarray, max_depth_m: float
) -> tuple[np.ndarray, np.ndarray, int]:
    """Invalidate depths beyond ``max_depth_m`` and report how many were cut.

    The plan allows any positive disparity, so a distant background match can
    imply a depth of tens of metres. This helper is off by default; it exists so
    a caller can bound the result and see the count rather than guess.
    """
    if not np.isfinite(max_depth_m) or max_depth_m <= 0.0:
        raise ValueError("max_depth_m must be positive and finite")
    values = np.array(depth, dtype=np.float32, copy=True)
    flags = np.array(validity, dtype=bool, copy=True)
    too_far = flags & np.isfinite(values) & (values > float(max_depth_m))
    count = int(np.count_nonzero(too_far))
    flags &= ~too_far
    values[~flags] = np.nan
    return values, flags, count


def stereo_frame_usable(
    depth: np.ndarray,
    validity: np.ndarray,
    *,
    person_mask: np.ndarray | None = None,
    config: StereoFallbackConfig | None = None,
) -> tuple[bool, str, dict[str, Any]]:
    """Decide whether this frame keeps stereo depth, and say why.

    Returns ``(usable, reason, detail)``. ``reason`` is ``ok`` when stereo is
    kept, otherwise it names the first condition that failed so a report can
    count fallbacks by cause instead of lumping them together.
    """
    settings = config or StereoFallbackConfig()
    values = np.asarray(depth, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"depth must be [H, W], got shape {values.shape}")
    flags = np.asarray(validity, dtype=bool)
    if flags.shape != values.shape:
        raise ValueError("validity must have the shape of the depth plane")
    # A pixel counts as usable only when it is marked valid AND holds a
    # positive finite value, so a caller cannot claim a pixel it cannot fill.
    flags = flags & np.isfinite(values) & (values > 0.0)

    total = int(values.size)
    valid_count = int(np.count_nonzero(flags))
    valid_fraction = float(valid_count / total) if total else 0.0
    detail: dict[str, Any] = {
        "valid_pixels": valid_count,
        "total_pixels": total,
        "valid_fraction": round(valid_fraction, 4),
        "min_valid_fraction": settings.min_valid_fraction,
    }
    if valid_count == 0:
        return False, "no_valid_pixels", detail
    if valid_fraction < settings.min_valid_fraction:
        return False, "valid_fraction_below_minimum", detail

    if person_mask is not None:
        mask = np.asarray(person_mask, dtype=bool)
        if mask.shape != values.shape:
            raise ValueError("person_mask must have the shape of the depth plane")
        person_pixels = int(np.count_nonzero(mask))
        person_valid = int(np.count_nonzero(flags & mask))
        person_fraction = float(person_valid / person_pixels) if person_pixels else 0.0
        detail.update(
            {
                "person_pixels": person_pixels,
                "person_valid_pixels": person_valid,
                "person_valid_fraction": round(person_fraction, 4),
                "min_person_fraction": settings.min_person_fraction,
            }
        )
        if person_pixels == 0:
            return False, "empty_person_mask", detail
        if person_fraction < settings.min_person_fraction:
            return False, "person_fraction_below_minimum", detail

    if settings.max_depth_m is not None:
        capped, capped_flags, cut = apply_stereo_depth_cap(
            values, flags, settings.max_depth_m
        )
        detail["pixels_beyond_max_depth"] = cut
        if not capped_flags.any():
            return False, "all_depths_beyond_maximum", detail

    return True, "ok", detail
