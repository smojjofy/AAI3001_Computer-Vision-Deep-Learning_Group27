"""This file is to run depth estimation on a folder of images. 
It takes an input folder containing images, processes each image to estimate depth, and saves the results in an output folder. 
The script also provides options to specify the model to use for depth estimation and whether to save the depth maps as .npy files in addition to the PNG images.
Without this script, the user would have to manually run depth estimation on each image one by one, which can be time-consuming and inefficient.
"""

import argparse
import sys
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depth_estimator import DepthEstimator
from depth_estimator.cli import add_estimator_arguments, describe_values, estimator_kwargs
from depth_estimator.utils import colorize


DEPTH_ROOT = Path(__file__).resolve().parents[1]
IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Read an image from a given path, handling non-English characters in Windows paths. The function uses OpenCV's imdecode method to read the image as a BGR array. If the image cannot be decoded, it returns None.
def read_image(path: Path):
    """Like cv2.imread, but also works with non-English characters in Windows paths."""
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)

# ok stands for whether the image was successfully encoded, and encoded is the encoded image data. If the encoding is successful, the function saves the encoded image to the specified path using tofile method. This function is used to save the output images after depth estimation.
def write_image(path: Path, image) -> None:
    ok, encoded = cv2.imencode(path.suffix, image)
    if ok:
        encoded.tofile(str(path))

# Parse command-line arguments to get the input folder, output folder, model path, and whether to save .npy files. The main function is responsible for loading the images from the input folder, initializing the depth estimator, processing each image to estimate depth, and saving the results in the output folder. It also prints summary statistics about the processing time and any skipped images.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEPTH_ROOT / "test_images"))
    parser.add_argument("--output", default=str(DEPTH_ROOT / "test_outputs"))
    add_estimator_arguments(parser)
    parser.add_argument("--no-npy", action="store_true", help="only save the PNGs")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.is_dir():
        input_dir.mkdir(parents=True)
        print(f"Created {input_dir}. Put some .jpg/.png images in it and run again.")
        return

    images = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_TYPES)

    if not images:
        print(f"No images in {input_dir}. Put some .jpg/.png files there and run again.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model} ...")
    times, skipped = [], []
# checks if the model can be loaded, and if not, it prints an error message and exits the script. If the model is loaded successfully, it prints the repository path (if available), the device being used for inference, and the output type of the depth estimator. It also prints the total number of images to be processed from the input directory.
    with DepthEstimator(**estimator_kwargs(args)) as estimator:
        if getattr(estimator, "repo_path", None):
            print(f"Repo code: {estimator.repo_path}")
        print(f"Device: {estimator.device}, input size: {estimator.input_size}")
        print(f"Output: {estimator.kind} -> {describe_values(estimator.kind)}")
        print(f"Processing {len(images)} images from {input_dir}\n")

        for i, path in enumerate(images, 1):
            image_bgr = read_image(path)

            if image_bgr is None:
                skipped.append(path.name)
                print(f"[{i}/{len(images)}] SKIP {path.name} (can't decode; is it really a jpg/png?)")
                continue

            start = perf_counter()
            result = estimator.estimate(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
            ms = (perf_counter() - start) * 1000
            times.append(ms)

            write_image(output_dir / f"{path.stem}_depth.png",
                        np.hstack([image_bgr, colorize(result)]))
            if not args.no_npy:
                np.save(output_dir / f"{path.stem}.npy", result.depth)
                np.save(output_dir / f"{path.stem}_confidence.npy", result.confidence)

            d = result.depth
            print(f"[{i}/{len(images)}] {path.name}  {image_bgr.shape[1]}x{image_bgr.shape[0]}  "
                  f"{ms:.0f} ms  range {np.nanmin(d):.2f}..{np.nanmax(d):.2f}  "
                  f"valid {result.valid_fraction * 100:.0f}%")

    print()
    print("Folder Depth Summary")
    print("--------------------")
    print(f"Images processed: {len(times)}")
    if skipped:
        print(f"Skipped: {len(skipped)} ({', '.join(skipped)})")
    if times:
        # first image includes warm-up, so report the rest separately
        steady = times[1:] or times
        print(f"First image: {times[0]:.0f} ms (includes warm-up)")
        print(f"Average after that: {np.mean(steady):.0f} ms  ({1000 / np.mean(steady):.1f} images/s)")
    print(f"Results in: {output_dir}")


if __name__ == "__main__":
    main()
