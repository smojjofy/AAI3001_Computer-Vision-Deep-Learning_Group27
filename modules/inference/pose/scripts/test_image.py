from pathlib import Path

import cv2

from pose_estimator import PoseEstimator


ROOT = Path(__file__).resolve().parents[4]

MODEL_PATH = (
    ROOT
    / "modules"
    / "inference"
    / "pose"
    / "models"
    / "pose_landmarker.task"
)

IMAGE_PATH = (
    ROOT
    / "modules"
    / "inference"
    / "pose"
    / "test_input.jpg"
)


def main() -> None:
    image_bgr = cv2.imread(str(IMAGE_PATH))

    if image_bgr is None:
        raise FileNotFoundError(
            f"Unable to read test image: {IMAGE_PATH}"
        )

    # OpenCV loads images as BGR.
    # MediaPipe expects RGB.
    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    with PoseEstimator(MODEL_PATH) as estimator:
        result = estimator.estimate(image_rgb)

    print(f"Pose detected: {result.detected}")
    print(f"Landmark count: {len(result.landmarks)}")

    if result.detected:
        for index, landmark in enumerate(result.landmarks):
            print(
                f"{index:02d}: "
                f"x={landmark.x:.3f}, "
                f"y={landmark.y:.3f}, "
                f"z={landmark.z:.3f}, "
                f"visibility={landmark.visibility}"
            )


if __name__ == "__main__":
    main()