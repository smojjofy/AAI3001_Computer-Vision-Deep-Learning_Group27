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

    # Draw official MediaPipe pose connections.
    for connection in (
        pose_landmarker
        .PoseLandmarksConnections
        .POSE_LANDMARKS
    ):
        start = result.landmarks[connection.start]
        end = result.landmarks[connection.end]

        start_point = (
            round(start.x * width),
            round(start.y * height),
        )

        end_point = (
            round(end.x * width),
            round(end.y * height),
        )

        cv2.line(
            image_bgr,
            start_point,
            end_point,
            (0, 255, 0),
            2,
        )

    # Draw and label every landmark.
    for index, landmark in enumerate(result.landmarks):
        x = round(landmark.x * width)
        y = round(landmark.y * height)

        cv2.circle(
            image_bgr,
            (x, y),
            4,
            (0, 0, 255),
            -1,
        )

        cv2.putText(
            image_bgr,
            str(index),
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )

    cv2.imshow(
        "Pose Landmark Inspection",
        image_bgr,
    )

    print("\nLower-body landmarks:")
    for index in range(23, 33):
        landmark = result.landmarks[index]

        x_pixel = round(
            landmark.x * width
        )

        y_pixel = round(
            landmark.y * height
        )

        print(
            f"{index:02d}: "
            f"pixel=({x_pixel}, {y_pixel}), "
            f"visibility={landmark.visibility:.3f}"
        )

    print("\nPress any key to close.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()