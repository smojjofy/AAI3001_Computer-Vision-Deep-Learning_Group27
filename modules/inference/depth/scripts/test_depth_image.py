import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Make `src` importable when running this file directly
# (remove if the package is installed / on PYTHONPATH like pose_estimator).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import DEFAULT_YOLO_DEPTH, DEFAULT_YOLO_IMGSZ, YoloDepthEstimator
from src.utils import colorize


ROOT = Path(__file__).resolve().parents[4]

# Same test image the pose prototype uses, so results can be compared directly.
IMAGE_PATH = (
    ROOT
    / "modules"
    / "inference"
    / "pose"
    / "test_input.jpg"
)

OUTPUT_PATH = (
    ROOT
    / "modules"
    / "inference"
    / "depth"
    / "test_output_depth.png"
)

# Parse command-line arguments to get the model path, image path, and whether to show the output or not. The main function is responsible for loading the image, initializing the depth estimator, estimating the depth, and displaying or saving the results as specified by the user. It also prints summary statistics about the depth estimation results.

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo-model", default=DEFAULT_YOLO_DEPTH, help="yolo26n-depth.pt (fastest), s, m")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_YOLO_IMGSZ, help="YOLO input size (lower = faster)")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, mps")
    parser.add_argument("--image", default=str(IMAGE_PATH))
    parser.add_argument("--no-show", action="store_true",
                        help="don't open a window (just save the PNG)")
    args = parser.parse_args()

    image_path = Path(args.image.strip('"')).expanduser().resolve()

    if not image_path.is_file():
        raise FileNotFoundError(
            f"No file at: {image_path}\n"
            f"Check the path, or pass one with --image \"C:\\path\\to\\photo.jpg\""
        )

    # np.fromfile + imdecode also works with non-English characters in Windows paths,
    # which plain cv2.imread silently fails on.
    image_bgr = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise ValueError(
            f"File exists but OpenCV can't decode it: {image_path}\n"
            f"Use a .jpg or .png (HEIC/iPhone photos and some .webp files won't load)."
        )

    # OpenCV loads images as BGR. YOLO depth (like MediaPipe) is given RGB.
    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    print(f"Loading {args.yolo_model} ...")

    with YoloDepthEstimator(args.yolo_model, device=args.device, imgsz=args.imgsz) as estimator:
        print(f"Device: {estimator.device}, input size: {estimator.imgsz}")
        result = estimator.estimate(image_rgb)

    depth = result.depth
    unit = "m"

    print(f"Kind: {result.kind}  ->  metres from the camera (bigger = farther)")
    print(f"Shape: {depth.shape}  (input was {image_rgb.shape[:2]})")
    print(f"Valid pixels: {result.valid_fraction * 100:.1f}%")
    print(f"Min: {np.nanmin(depth):.3f} {unit}")
    print(f"Median: {np.nanmedian(depth):.3f} {unit}")
    print(f"Max: {np.nanmax(depth):.3f} {unit}")
    print(f"Mean confidence: {result.confidence.mean():.2f}  (1 = smooth surface, low = edges)")

    side_by_side = np.hstack([image_bgr, colorize(result)])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT_PATH), side_by_side)
    print(f"Saved: {OUTPUT_PATH}")
# implement a check to see if the user has requested to show the output image. If the --no-show flag is not set, the script will display the side-by-side comparison of the original image and the depth map in a window. The user can close the window by pressing any key, and the script will then clean up by destroying all OpenCV windows.
    if not args.no_show:
        cv2.imshow("YOLO depth", side_by_side)
        print("Press any key in the image window to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
