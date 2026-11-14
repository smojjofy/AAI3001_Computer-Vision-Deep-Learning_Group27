from dataclasses import dataclass

import cv2
import numpy as np

from .contracts import DepthEstimation, DepthFrameResult
from .utils import sample_depth

# MediaPipe Pose Landmarker's 33 landmarks, in index order.
LANDMARK_NAMES = (
    "nose", "left eye (inner)", "left eye", "left eye (outer)", "right eye (inner)", "right eye",
    "right eye (outer)", "left ear", "right ear", "mouth (left)", "mouth (right)",
    "left shoulder", "right shoulder", "left elbow", "right elbow", "left wrist", "right wrist",
    "left pinky", "right pinky", "left index", "right index", "left thumb", "right thumb",
    "left hip", "right hip", "left knee", "right knee", "left ankle", "right ankle",
    "left heel", "right heel", "left foot index", "right foot index",
)
# Skeleton lines (same as MediaPipe's PoseLandmarksConnections.POSE_LANDMARKS).
POSE_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10), (11, 12), (11, 13),
    (13, 15), (15, 17), (15, 19), (15, 21), (17, 19), (12, 14), (14, 16), (16, 18), (16, 20),
    (16, 22), (18, 20), (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
)
# Joints whose distance is written next to them on the drawing (the rest just get a dot).
LABELLED = {0: "nose", 15: "L wrist", 16: "R wrist", 25: "L knee", 26: "R knee", 27: "L ankle", 28: "R ankle"}

NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP = 0, 11, 12, 23, 24


@dataclass(frozen=True)
class PoseDepthFrameResult:
    """One frame's pose and depth, plus the depth at every joint.

    joint_depth[i] is the depth at landmark i (same units as depth.kind: metres for
    metric_depth, 0..1 for relative_depth). NaN when the pose wasn't detected, the joint
    is outside the image, it isn't visible enough, or the depth there is invalid.
    """

    frame_id: int
    source_id: str
    capture_timestamp_ns: int
    pose: object                      # pose_estimator.PoseEstimation
    depth: DepthEstimation
    joint_depth: np.ndarray           # float32 [num_landmarks]

    @property
    def detected(self) -> bool:
        return bool(self.pose.detected)

    @property
    def body_depth(self) -> float:
        """Median depth of the torso joints (shoulders + hips): one number for 'how far is the person'."""
        torso = self.joint_depth[[LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]] \
            if self.joint_depth.size > RIGHT_HIP else self.joint_depth
        torso = torso[np.isfinite(torso)]
        return float(np.median(torso)) if torso.size else float("nan")


def joint_depths(
    pose,
    depth: DepthEstimation,
    radius: int = 2,
    min_visibility: float = 0.5,
) -> np.ndarray:
    """Depth at each pose landmark: median of a small (2*radius+1)^2 window, ignoring invalid pixels.

    A window rather than a single pixel, because joints often sit right on the person's edge
    where one pixel can be background. Joints MediaPipe isn't confident about
    (visibility < min_visibility) get NaN.
    """
    if not pose.detected:
        return np.zeros(0, np.float32)

    values = np.full(len(pose.landmarks), np.nan, np.float32)
    for i, lm in enumerate(pose.landmarks):
        if lm.visibility is not None and lm.visibility < min_visibility:
            continue
        values[i] = sample_depth(depth, lm.x, lm.y, radius=radius)
    return values


def fuse_pose_depth(
    pose_result,
    depth_result: DepthFrameResult,
    radius: int = 2,
    min_visibility: float = 0.5,
) -> PoseDepthFrameResult:
    """Combine a PoseFrameResult and a DepthFrameResult of the SAME frame.

    Raises ValueError if they come from different frames: mixing frames would put joints
    at the wrong distances without any visible error.
    """
    pose_id = (pose_result.frame_id, pose_result.source_id, pose_result.capture_timestamp_ns)
    depth_id = (depth_result.frame_id, depth_result.source_id, depth_result.capture_timestamp_ns)
    if pose_id != depth_id:
        raise ValueError(f"pose and depth are from different frames: {pose_id} vs {depth_id}")

    return PoseDepthFrameResult(
        frame_id=pose_result.frame_id,
        source_id=pose_result.source_id,
        capture_timestamp_ns=pose_result.capture_timestamp_ns,
        pose=pose_result.pose,
        depth=depth_result.depth,
        joint_depth=joint_depths(pose_result.pose, depth_result.depth, radius, min_visibility),
    )


def draw_pose_depth(image_bgr: np.ndarray, result: PoseDepthFrameResult) -> np.ndarray:
    """Skeleton on a copy of the image, with the distance written at the main joints.

    Works on the photo or on a colourised depth map (any BGR image of any size).
    """
    out = image_bgr.copy()
    if not result.detected:
        return out
    h, w = out.shape[:2]
    scale = max(h, w) / 800
    thick = max(1, round(2 * scale))
    unit = " m" if result.depth.is_metric else ""
    pts = [(round(lm.x * (w - 1)), round(lm.y * (h - 1))) for lm in result.pose.landmarks]
    seen = np.isfinite(result.joint_depth)

    for a, b in POSE_CONNECTIONS:
        if a < len(pts) and b < len(pts):
            colour = (0, 255, 0) if seen[a] and seen[b] else (0, 140, 0)
            cv2.line(out, pts[a], pts[b], colour, thick, cv2.LINE_AA)
    for i, (x, y) in enumerate(pts):
        cv2.circle(out, (x, y), max(2, round(4 * scale)), (0, 0, 255) if seen[i] else (90, 90, 90), -1, cv2.LINE_AA)
        if i in LABELLED and seen[i]:
            text = f"{LABELLED[i]} {result.joint_depth[i]:.2f}{unit}"
            org = (x + round(8 * scale), y - round(6 * scale))
            cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
            cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (255, 255, 255), thick, cv2.LINE_AA)
    return out


def joint_table(result: PoseDepthFrameResult) -> list[list]:
    """Rows of [index, joint, x px, y px, visibility, depth] for printing or a table widget."""
    if not result.detected:
        return []
    h, w = result.depth.shape
    rows = []
    for i, lm in enumerate(result.pose.landmarks):
        name = LANDMARK_NAMES[i] if i < len(LANDMARK_NAMES) else str(i)
        d = float(result.joint_depth[i])
        vis = float(lm.visibility) if lm.visibility is not None else float("nan")
        rows.append([i, name, round(lm.x * (w - 1)), round(lm.y * (h - 1)), round(vis, 2),
                     round(d, 3) if np.isfinite(d) else None])
    return rows
