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
    try:
        from pose_estimator import PoseEstimator
    except ImportError:
        sys.path.insert(0, str(POSE_ROOT / "src"))      # teammate's module, not pip-installed
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
            self._depth = YoloDepthEstimator(self.yolo_model, device=self.device, imgsz=self.imgsz)
        return self._depth

    def process_frame(
        self,
        rgb_frame: np.ndarray,
        frame_id: int,
        source_id: str,
        capture_timestamp_ns: int,
        depth: np.ndarray | DepthEstimation | None = None,
    ) -> PipelineResult:
        """Pose + depth for one RGB uint8 frame.

        depth: ZED depth for this frame (array, metres x depth_scale, or a DepthEstimation).
               None = estimate it with YOLO depth.
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
            if not isinstance(depth, DepthEstimation):
                depth = external_depth(depth, rgb_frame.shape[:2], self.depth_scale)
            elif depth.shape != rgb_frame.shape[:2]:
                raise ValueError(f"depth is {depth.shape} but the frame is {rgb_frame.shape[:2]}")
            depth_frame = DepthFrameResult(depth=depth, **identity)
            source = "zed"
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
