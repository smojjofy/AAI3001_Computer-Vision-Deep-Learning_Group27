from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean
from time import perf_counter

import cv2

from mediapipe.tasks.python.vision import pose_landmarker

from pose_estimator import PoseEstimator


ROOT = Path(__file__).resolve().parents[4]

POSE_ROOT = (
    ROOT
    / "modules"
    / "inference"
    / "pose"
)

MODELS = {
    "lite": (
        POSE_ROOT
        / "models"
        / "pose_landmarker_lite.task"
    ),
    "full": (
        POSE_ROOT
        / "models"
        / "pose_landmarker_full.task"
    ),
}

TEST_IMAGES = {
    "front_facing_runner": (
        POSE_ROOT
        / "test_images"
        / "front_facing_runner.jpg"
    ),

    "sideways_runner": (
        POSE_ROOT
        / "test_images"
        / "sideways_runner.jpg"
    ),

    "lower_body_cutoff": (
        POSE_ROOT
        / "test_images"
        / "runner_with_lower_body_cutoff.jpg"
    ),

    "multiple_people": (
        POSE_ROOT
        / "test_images"
        / "multiple_people.jpg"
    ),

    "multiple_people_varying_distances": (
        POSE_ROOT
        / "test_images"
        / "multiple_people_varying_distances.jpg"
    ),

    "low_light_visible_body": (
        POSE_ROOT
        / "test_images"
        / "Low_light_test.png"
    ),

    "extreme_low_light_silhouette": (
        POSE_ROOT
        / "test_images"
        / "Low_light_edge case_test.jpg"
    ),

    "moderate_motion_blur": (
        POSE_ROOT
        / "test_images"
        / "Moderate_motion_blur_test.png"
    ),

    "extreme_motion_blur": (
        POSE_ROOT
        / "test_images"
        / "extreme_motion_blur_test.png"
    ),
    "face_mask_occlusion": (
        POSE_ROOT
        / "test_images"
        / "face_mask_occlusion.png"
    ),

    "object_occlusion_full_body": (
        POSE_ROOT
        / "test_images"
        / "object_occlusion_full_body.png"
    ),
}

OUTPUT_DIR = (
    POSE_ROOT
    / "comparison_outputs_expanded_with_occlusion"
)

CSV_PATH = (
    OUTPUT_DIR
    / "comparison_results_expanded_with_occlusion.csv"
)

WARMUP_RUNS = 5
TIMED_RUNS = 30


def load_image(path: Path):
    image_bgr = cv2.imread(str(path))

    if image_bgr is None:
        raise FileNotFoundError(
            f"Unable to read image: {path}"
        )

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    return image_bgr, image_rgb


def draw_pose(
    frame_bgr,
    landmarks,
) -> None:
    height, width = frame_bgr.shape[:2]

    
    line_thickness = max(
        2,
        round(min(width, height) / 300),
    )

    landmark_radius = max(
        4,
        line_thickness * 2,
    )


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
            frame_bgr,
            start_point,
            end_point,
            (0, 255, 0),
            line_thickness,
        )


    for landmark in landmarks:
        x = round(landmark.x * width)
        y = round(landmark.y * height)

        cv2.circle(
            frame_bgr,
            (x, y),
            landmark_radius,
            (0, 0, 255),
            -1,
        )


def average_visibility(landmarks) -> float:
    values = [
        landmark.visibility
        for landmark in landmarks
        if landmark.visibility is not None
    ]

    if not values:
        return 0.0

    return mean(values)


def lower_body_visibility(landmarks) -> float:

    values = [
        landmark.visibility
        for landmark in landmarks[23:33]
        if landmark.visibility is not None
    ]

    if not values:
        return 0.0

    return mean(values)


def benchmark_model(
    model_name: str,
    model_path: Path,
):
    results = []

    print()
    print("=" * 70)
    print(f"MODEL: {model_name.upper()}")
    print("=" * 70)

    with PoseEstimator(model_path) as estimator:

        for test_name, image_path in TEST_IMAGES.items():

            image_bgr, image_rgb = load_image(
                image_path
            )


            for _ in range(WARMUP_RUNS):
                estimator.estimate(
                    image_rgb
                )

            timings_ms = []
            pose_result = None

            for _ in range(TIMED_RUNS):

                start = perf_counter()

                pose_result = estimator.estimate(
                    image_rgb
                )

                elapsed_ms = (
                    perf_counter()
                    - start
                ) * 1000.0

                timings_ms.append(
                    elapsed_ms
                )

            avg_ms = mean(
                timings_ms
            )

            inference_fps = (
                1000.0 / avg_ms
                if avg_ms > 0
                else 0.0
            )

            if pose_result.detected:
                landmark_count = len(
                    pose_result.landmarks
                )

                avg_visibility = (
                    average_visibility(
                        pose_result.landmarks
                    )
                )

                lower_visibility = (
                    lower_body_visibility(
                        pose_result.landmarks
                    )
                )

                annotated = (
                    image_bgr.copy()
                )

                draw_pose(
                    annotated,
                    pose_result.landmarks,
                )

                output_path = (
                    OUTPUT_DIR
                    / f"{model_name}_{test_name}.jpg"
                )

                cv2.imwrite(
                    str(output_path),
                    annotated,
                )

            else:
                landmark_count = 0
                avg_visibility = 0.0
                lower_visibility = 0.0

                output_path = (
                    OUTPUT_DIR
                    / f"{model_name}_{test_name}.jpg"
                )

                cv2.imwrite(
                    str(output_path),
                    image_bgr,
                )

            print()
            print(
                f"Test: {test_name}"
            )

            print(
                f"Pose detected: "
                f"{pose_result.detected}"
            )

            print(
                f"Landmarks: "
                f"{landmark_count}"
            )

            print(
                f"Average visibility: "
                f"{avg_visibility:.3f}"
            )

            print(
                f"Lower-body visibility: "
                f"{lower_visibility:.3f}"
            )

            print(
                f"Average inference: "
                f"{avg_ms:.2f} ms"
            )

            print(
                f"Pose-inference FPS: "
                f"{inference_fps:.2f}"
            )

            results.append(
                {
                    "model": model_name,
                    "test": test_name,
                    "detected": pose_result.detected,
                    "landmarks": landmark_count,
                    "average_visibility": (
                        round(avg_visibility, 4)
                    ),
                    "lower_body_visibility": (
                        round(lower_visibility, 4)
                    ),
                    "average_inference_ms": (
                        round(avg_ms, 2)
                    ),
                    "pose_inference_fps": (
                        round(inference_fps, 2)
                    ),
                }
            )

    return results


def save_csv(results) -> None:

    fieldnames = [
        "model",
        "test",
        "detected",
        "landmarks",
        "average_visibility",
        "lower_body_visibility",
        "average_inference_ms",
        "pose_inference_fps",
    ]

    with CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)


def main() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_results = []

    for model_name, model_path in MODELS.items():

        model_results = benchmark_model(
            model_name,
            model_path,
        )

        all_results.extend(
            model_results
        )

    save_csv(
        all_results
    )

    print()
    print("=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)

    print(
        f"Annotated images saved to: "
        f"{OUTPUT_DIR}"
    )

    print(
        f"Results CSV saved to: "
        f"{CSV_PATH}"
    )


if __name__ == "__main__":
    main()