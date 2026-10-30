from pathlib import Path

import numpy as np

from mediapipe.tasks.python.core import base_options
from mediapipe.tasks.python.vision import pose_landmarker
from mediapipe.tasks.python.vision.core import image as image_module

from .contracts import (PoseEstimation, PoseFrameResult,PoseLandmark,)


class PoseEstimator:
    """MediaPipe pose estimator for one RGB frame."""

    def __init__(self, model_path: str | Path) -> None:
        model_path = Path(model_path)

        if not model_path.is_file():
            raise FileNotFoundError(
                f"Pose Landmarker model not found: {model_path}"
            )

        options = pose_landmarker.PoseLandmarkerOptions(
            base_options=base_options.BaseOptions(
                model_asset_path=str(model_path)
            ),
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )

        self._landmarker = (
            pose_landmarker.PoseLandmarker.create_from_options(options)
        )

    def estimate(self, rgb_frame: np.ndarray) -> PoseEstimation:
        """Estimate a human pose from one RGB uint8 frame."""

        if rgb_frame.ndim != 3 or rgb_frame.shape[2] != 3:
            raise ValueError(
                "rgb_frame must have shape [height, width, 3]"
            )

        if rgb_frame.dtype != np.uint8:
            raise ValueError(
                "rgb_frame must use dtype uint8"
            )

        rgb_frame = np.ascontiguousarray(rgb_frame)

        mp_image = image_module.Image(
            image_format=image_module.ImageFormat.SRGB,
            data=rgb_frame,
        )

        result = self._landmarker.detect(mp_image)

        if not result.pose_landmarks:
            return PoseEstimation(
                detected=False,
                landmarks=(),
            )

        pose = result.pose_landmarks[0]

        landmarks = tuple(
            PoseLandmark(
                x=landmark.x,
                y=landmark.y,
                z=landmark.z,
                visibility=landmark.visibility,
                presence=landmark.presence,
            )
            for landmark in pose
        )

        return PoseEstimation(
            detected=True,
            landmarks=landmarks,
        )

    def estimate_frame(
        self,
        rgb_frame: np.ndarray,
        frame_id: int,
        source_id: str,
        capture_timestamp_ns: int,
    ) -> PoseFrameResult:
        """Estimate pose while preserving source-frame identity."""

        if frame_id < 0:
            raise ValueError(
                "frame_id must be non-negative"
            )

        if not source_id:
            raise ValueError(
                "source_id must not be empty"
            )

        if capture_timestamp_ns < 0:
            raise ValueError(
                "capture_timestamp_ns must be non-negative"
            )

        pose = self.estimate(
            rgb_frame,
        )

        return PoseFrameResult(
            frame_id=frame_id,
            source_id=source_id,
            capture_timestamp_ns=capture_timestamp_ns,
            pose=pose,
        )

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "PoseEstimator":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


