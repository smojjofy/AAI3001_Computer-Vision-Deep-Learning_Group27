from pathlib import Path
from time import perf_counter

import cv2

from mediapipe.tasks.python.vision import pose_landmarker

from pose_estimator import VideoPoseEstimator


ROOT = Path(__file__).resolve().parents[4]

MODEL_PATH = (
    ROOT
    / "modules"
    / "inference"
    / "pose"
    / "models"
    / "pose_landmarker.task"
)

VIDEO_PATH = (
    ROOT
    / "modules"
    / "inference"
    / "pose"
    / "test_video.mp4"
)


def draw_pose(
    frame,
    landmarks,
) -> None:
    height, width = frame.shape[:2]

    # Skeleton connections.
    for connection in (
        pose_landmarker
        .PoseLandmarksConnections
        .POSE_LANDMARKS
    ):
        start = landmarks[connection.start]
        end = landmarks[connection.end]

        start_point = (
            round(start.x * width),
            round(start.y * height),
        )

        end_point = (
            round(end.x * width),
            round(end.y * height),
        )

        cv2.line(
            frame,
            start_point,
            end_point,
            (0, 255, 0),
            2,
        )

    # Landmark points.
    for landmark in landmarks:
        x = round(landmark.x * width)
        y = round(landmark.y * height)

        cv2.circle(
            frame,
            (x, y),
            4,
            (0, 0, 255),
            -1,
        )


def main() -> None:
    capture = cv2.VideoCapture(
        str(VIDEO_PATH)
    )




    if not capture.isOpened():
        raise FileNotFoundError(
            f"Unable to open video: {VIDEO_PATH}"
        )



    source_fps = capture.get(
        cv2.CAP_PROP_FPS
    )

    if source_fps <= 0:
        source_fps = 30.0

    frame_count = 0
    pose_detected_count = 0

    total_inference_ms = 0.0


    with VideoPoseEstimator(
        MODEL_PATH
    ) as estimator:

        while True:

            success, frame_bgr = capture.read()


            if not success:
                break

            # OpenCV gives us BGR.
            frame_rgb = cv2.cvtColor(
                frame_bgr,
                cv2.COLOR_BGR2RGB,
            )

            #/*
            #Timestamp is derived from the frame number
            #rather than wall-clock execution time.
            #*/

            timestamp_ms = round(
                frame_count
                * 1000.0
                / source_fps
            )

            start_time = perf_counter()

            result = estimator.estimate(
                frame_rgb,
                timestamp_ms,
            )

            inference_ms = (
                perf_counter() - start_time
            ) * 1000.0

            total_inference_ms += inference_ms
            frame_count += 1

            if result.detected:
                pose_detected_count += 1
                draw_pose(
                    frame_bgr,
                    result.landmarks,
                )

            pose_inference_fps = (
                1000.0 / inference_ms
                if inference_ms > 0
                else 0.0
            )

            status = (
                "DETECTED"
                if result.detected
                else "NO POSE"
            )

            cv2.putText(
                frame_bgr,
                f"Frame: {frame_count}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame_bgr,
                f"Pose: {status}",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame_bgr,
                f"Inference: {inference_ms:.1f} ms",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame_bgr,
                f"Processing FPS: {pose_inference_fps:.1f}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.imshow(
                "MediaPipe Video Pose Estimator",
                frame_bgr,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    capture.release()
    cv2.destroyAllWindows()

    if frame_count > 0:
        average_ms = (
            total_inference_ms
            / frame_count
        )

        average_fps = (
            1000.0 / average_ms
            if average_ms > 0
            else 0.0
        )

        pose_missed_count = (
            frame_count
            - pose_detected_count
        )

        detection_rate = (
            pose_detected_count
            / frame_count
            * 100.0
        )

        print()
        print("Video Pose Estimator Summary")
        print("----------------------------")

        print(
            f"Frames processed: "
            f"{frame_count}"
        )

        print(
            f"Pose detected: "
            f"{pose_detected_count}"
        )

        print(
            f"Pose missed: "
            f"{pose_missed_count}"
        )

        print(
            f"Detection rate: "
            f"{detection_rate:.2f}%"
        )

        print(
            f"Average pose inference: "
            f"{average_ms:.2f} ms"
        )

        print(
            f"Average pose-inference FPS: "
            f"{average_fps:.2f}"
        )


if __name__ == "__main__":
    main()