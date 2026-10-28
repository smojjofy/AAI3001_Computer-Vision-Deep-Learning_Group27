import argparse
import sys
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depth_estimator import DepthEstimator
from depth_estimator.cli import add_estimator_arguments, describe_values, estimator_kwargs
from depth_estimator.utils import colorize, sample_depth, temporal_flicker

# We use parents [4] to get the root of the repository, as this script is located in modules/inference/depth/scripts/
ROOT = Path(__file__).resolve().parents[4]
# The default video path is set to a test video located in the pose inference module. This path can be overridden by providing a different video path as a command-line argument when running the script.
VIDEO_PATH = (
    ROOT
    / "modules"
    / "inference"
    / "pose"
    / "test_video.mp4"
)

# Put text lines on the frame at specified positions. The function takes a frame and a list of lines as input, and uses OpenCV's putText function to draw each line of text on the frame at the specified position. The text is drawn in white color with a font size of 0.7 and a thickness of 2 pixels.
def put_lines(frame, lines) -> None:
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (20, 30 + 30 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

# We have to parse the command-line arguments to get the model path, video path, maximum number of frames to process, save directory for depth maps, and whether to show the output or not. The main function is responsible for loading the video, initializing the depth estimator, processing each frame of the video, estimating the depth, and displaying or saving the results as specified by the user. It also calculates and prints summary statistics about the processing time and temporal flicker.
def main() -> None:
    parser = argparse.ArgumentParser()
    add_estimator_arguments(parser)
    parser.add_argument("--video", default=str(VIDEO_PATH))
    parser.add_argument("--max-frames", type=int, default=0,
                        help="stop after N frames (0 = whole video)")
    parser.add_argument("--save-dir", default="",
                        help="save depth per frame as <frame_id>.npy here "
                             "(float32; metres if a metric model)")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    capture = cv2.VideoCapture(args.video)

    if not capture.isOpened():
        raise FileNotFoundError(
            f"Unable to open video: {args.video}"
        )

    source_fps = capture.get(cv2.CAP_PROP_FPS)

# Check fps because some video files may not have this information. If the fps is less than or equal to 0, we set it to a default value of 30.0 fps. This ensures that the script can still process the video even if the fps information is missing or incorrect.
    if source_fps <= 0:
        source_fps = 30.0

    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
# Initialize variables for frame count, total inference time, and a list to store temporal flicker values. The previous depth map is also initialized to None. These variables will be used to keep track of the number of frames processed, the total time taken for depth estimation, and the temporal flicker between consecutive frames.
    frame_count = 0
    total_inference_ms = 0.0
    flickers = []
    previous = None

    print(f"Loading {args.model} ...")

    with DepthEstimator(**estimator_kwargs(args)) as estimator:
        print(f"Device: {estimator.device}, output: {estimator.kind}")

        while True:
            success, frame_bgr = capture.read()

            if not success:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Same identity scheme as the pose video script, so depth and pose results line up frame for frame.
            frame_id = frame_count
            source_id = "test-video"
            capture_timestamp_ns = round(
                frame_id * 1_000_000_000 / source_fps
            )

            start_time = perf_counter()

            result = estimator.estimate_frame(
                rgb_frame=frame_rgb,
                frame_id=frame_id,
                source_id=source_id,
                capture_timestamp_ns=capture_timestamp_ns,
            )

            inference_ms = (perf_counter() - start_time) * 1000.0
            total_inference_ms += inference_ms
            frame_count += 1

# Temporal flicker is calculated by comparing the current depth map with the previous one. If a previous depth map exists, the temporal_flicker function is called to compute the flicker value, which is then appended to the flickers list. The previous depth map is updated to the current depth map for the next iteration. If a save directory is specified, the current depth map is saved as a .npy file in that directory with a filename based on the frame ID.
            if previous is not None:
                flickers.append(temporal_flicker(previous, result.depth))
            previous = result.depth

            if save_dir:
                np.save(save_dir / f"{frame_id:06d}.npy", result.depth.depth)
                np.save(save_dir / f"{frame_id:06d}_confidence.npy", result.depth.confidence)

            if not args.no_show:
                depth_view = colorize(result.depth, max_depth_m=8.0)
                centre = sample_depth(result.depth, 0.5, 0.5)
                unit = " m" if result.depth.is_metric else " (rel)"
                put_lines(depth_view, [
                    f"Frame: {frame_count}",
                    f"Inference: {inference_ms:.1f} ms",
                    f"Processing FPS: {1000.0 / max(inference_ms, 1e-6):.1f}",
                    f"Centre: {centre:.2f}{unit}",
                ])
                cv2.imshow("Depth Anything V2 Video",
                           np.hstack([frame_bgr, depth_view]))

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.max_frames and frame_count >= args.max_frames:
                break

    capture.release()
    cv2.destroyAllWindows()

# After processing the video, if any frames were processed, we calculate the average inference time in milliseconds and print a summary of the video depth estimation results. This includes the total number of frames processed, the average depth inference time, and the average depth-inference frames per second (FPS). If temporal flicker values were recorded, we also print the median frame-to-frame change as a percentage. Finally, if a save directory was specified, we print the location where the depth maps were saved.
    if frame_count > 0:
        average_ms = total_inference_ms / frame_count

        print()
        print("Video Depth Estimator Summary")
        print("-----------------------------")
        print(f"Frames processed: {frame_count}")
        print(f"Average depth inference: {average_ms:.2f} ms")
        print(f"Average depth-inference FPS: {1000.0 / average_ms:.2f}")

# flickers is a list that contains the temporal flicker values calculated for each frame. If there are any flicker values recorded, we calculate the median of these values and print it as a percentage. This provides an indication of how much the depth estimation changes from frame to frame, which can be useful for assessing the stability of the depth estimation in dynamic scenes.
        if flickers:
            print(f"Temporal flicker (median frame-to-frame change): "
                  f"{np.median(flickers) * 100:.2f}%")
            print("  (only meaningful if the camera AND scene are still: then aim for < ~2%."
                  " A moving camera changes depth for real.)")

        if save_dir:
            print(f"Depth maps saved to: {save_dir}")


if __name__ == "__main__":
    main()
