from pathlib import Path

import cv2

from mediapipe.tasks.python.vision import pose_landmarker

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

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    with PoseEstimator(MODEL_PATH) as estimator:
        result = estimator.estimate(image_rgb)

    if not result.detected:
        print("No pose detected.")
        return

    height, width = image_bgr.shape[:2]

    # Draw skeleton connections first.
    for connection in pose_landmarker.PoseLandmarksConnections.POSE_LANDMARKS:
        start = result.landmarks[connection.start]
        end = result.landmarks[connection.end]

        start_point = (
            int(start.x * width),
            int(start.y * height),
        )

        end_point = (
            int(end.x * width),
            int(end.y * height),
        )

        cv2.line(
            image_bgr,
            start_point,
            end_point,
            (0, 255, 0),
            2,
        )

    # Draw the individual landmarks.
    for landmark in result.landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)

        cv2.circle(
            image_bgr,
            (x, y),
            4,
            (0, 0, 255),
            -1,
        )

    cv2.imshow(
        "MediaPipe Pose Estimator Prototype",
        image_bgr,
    )

    print(f"Pose detected: {result.detected}")
    print(f"Landmarks: {len(result.landmarks)}")
    print("Press any key in the image window to close.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()