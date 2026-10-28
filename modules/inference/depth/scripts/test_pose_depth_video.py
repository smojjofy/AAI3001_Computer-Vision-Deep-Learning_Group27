"""MediaPipe skeleton + Depth Anything depth on the same frames.

MediaPipe gives WHERE each body joint is in the image (x, y in 0..1).
Depth Anything gives HOW FAR every pixel is. Sampling the depth at each joint
turns the 2D skeleton into joints with real distances (with a metric model).
"""

import argparse
import sys
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))                 # .../depth
sys.path.insert(0, str(HERE.parents[2] / "pose"))        # .../pose (pose_estimator package)

from mediapipe.tasks.python.vision import pose_landmarker

from depth_estimator import DepthEstimator
from depth_estimator.cli import add_estimator_arguments, estimator_kwargs
from depth_estimator.utils import colorize, sample_depth
from pose_estimator import VideoPoseEstimator


ROOT = HERE.parents[4]

POSE_MODEL_PATH = (
    ROOT / "modules" / "inference" / "pose" / "models" / "pose_landmarker.task"
)

VIDEO_PATH = (
    ROOT / "modules" / "inference" / "pose" / "test_video.mp4"
)

# MediaPipe landmark indices worth labelling on screen.
LABELLED = {0: "nose", 15: "L wrist", 16: "R wrist", 27: "L ankle", 28: "R ankle"}


def draw_pose_with_depth(frame, landmarks, depth_result) -> list[float]:
    height, width = frame.shape[:2]
    unit = "m" if depth_result.is_metric else " rel"

    for connection in pose_landmarker.PoseLandmarksConnections.POSE_LANDMARKS:
        start = landmarks[connection.start]
        end = landmarks[connection.end]
        cv2.line(frame,
                 (round(start.x * width), round(start.y * height)),
                 (round(end.x * width), round(end.y * height)),
                 (0, 255, 0), 2)

    joint_depths = []
    for index, landmark in enumerate(landmarks):
        x, y = round(landmark.x * width), round(landmark.y * height)
        d = sample_depth(depth_result, landmark.x, landmark.y)
        joint_depths.append(d)
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

        if index in LABELLED and np.isfinite(d):
            cv2.putText(frame, f"{LABELLED[index]} {d:.2f}{unit}", (x + 6, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return joint_depths


def main() -> None:
    parser = argparse.ArgumentParser()
    add_estimator_arguments(parser)
    parser.add_argument("--video", default=str(VIDEO_PATH))
    args = parser.parse_args()

    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open video: {args.video}")

    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        source_fps = 30.0

    frame_count = 0
    pose_ms = depth_ms = 0.0
    identity_mismatch_count = 0

    with VideoPoseEstimator(POSE_MODEL_PATH) as pose_estimator, \
            DepthEstimator(**estimator_kwargs(args)) as depth_estimator:

        while True:
            success, frame_bgr = capture.read()
            if not success:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_id = frame_count
            source_id = "test-video"
            capture_timestamp_ns = round(frame_id * 1_000_000_000 / source_fps)

            t0 = perf_counter()
            pose = pose_estimator.estimate_frame(frame_rgb, frame_id, source_id, capture_timestamp_ns)
            t1 = perf_counter()
            depth = depth_estimator.estimate_frame(frame_rgb, frame_id, source_id, capture_timestamp_ns)
            t2 = perf_counter()
            pose_ms += (t1 - t0) * 1000
            depth_ms += (t2 - t1) * 1000
            frame_count += 1

            # Both results must describe the same frame before we combine them.
            if (pose.frame_id, pose.source_id) != (depth.frame_id, depth.source_id):
                identity_mismatch_count += 1
                continue

            depth_view = colorize(depth.depth, max_depth_m=8.0)
            if pose.pose.detected:
                draw_pose_with_depth(frame_bgr, pose.pose.landmarks, depth.depth)
                draw_pose_with_depth(depth_view, pose.pose.landmarks, depth.depth)

            cv2.putText(frame_bgr, f"Frame {frame_count}  pose {(t1-t0)*1000:.0f} ms  "
                        f"depth {(t2-t1)*1000:.0f} ms", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("MediaPipe Pose + Depth Anything", np.hstack([frame_bgr, depth_view]))

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    capture.release()
    cv2.destroyAllWindows()

    if frame_count:
        print()
        print("Pose + Depth Summary")
        print("--------------------")
        print(f"Frames processed: {frame_count}")
        print(f"Average pose inference: {pose_ms / frame_count:.2f} ms")
        print(f"Average depth inference: {depth_ms / frame_count:.2f} ms")
        print(f"Combined FPS: {1000 * frame_count / (pose_ms + depth_ms):.2f}")
        print(f"Identity mismatches: {identity_mismatch_count}")


if __name__ == "__main__":
    main()
