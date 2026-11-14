"""Depth inference: YOLO/Depth-Anything pipeline, pose fusion, and tiny baselines."""

from .contracts import DEPTH_KINDS, DepthEstimation, DepthFrameResult, DepthKind
from .estimator import DepthEstimate, DepthEstimator
from .model import TinyDepthNet
from .normalize import edge_confidence, normalize_prediction
from .pipeline import (
    CameraIntrinsics,
    PipelineResult,
    PoseDepthPipeline,
    external_depth,
    zed_to_rgb,
)
from .pose_depth import (
    LANDMARK_NAMES,
    PoseDepthFrameResult,
    draw_pose_depth,
    fuse_pose_depth,
    joint_depths,
)
from .utils import MODELS_DIR, resolve_device
from .yolo_depth import DEFAULT_YOLO_DEPTH, DEFAULT_YOLO_IMGSZ, YoloDepthEstimator

__all__ = [
    "DEPTH_KINDS",
    "DepthEstimation",
    "DepthFrameResult",
    "DepthKind",
    "DepthEstimate",
    "DepthEstimator",
    "TinyDepthNet",
    "edge_confidence",
    "normalize_prediction",
    "CameraIntrinsics",
    "PipelineResult",
    "PoseDepthPipeline",
    "external_depth",
    "zed_to_rgb",
    "LANDMARK_NAMES",
    "PoseDepthFrameResult",
    "draw_pose_depth",
    "fuse_pose_depth",
    "joint_depths",
    "MODELS_DIR",
    "resolve_device",
    "DEFAULT_YOLO_DEPTH",
    "DEFAULT_YOLO_IMGSZ",
    "YoloDepthEstimator",
]
