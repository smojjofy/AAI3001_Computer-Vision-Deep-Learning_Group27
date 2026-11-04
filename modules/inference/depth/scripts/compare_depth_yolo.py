"""Visual comparison: Depth Anything V2 vs YOLO depth on the same frame.

    python scripts/compare_depth_yolo.py --image photo.jpg
    python scripts/compare_depth_yolo.py                      # every image in test_images/
    python scripts/compare_depth_yolo.py --video clip.mp4     # live window, q to quit
    python scripts/compare_depth_yolo.py --video clip.mp4 --no-show --save-video out.mp4

Each output is a 2x2 grid, all depth panels on the SAME colour scale (metres):
    Original                         | Depth Anything V2 (scaled to YOLO's metres)
    YOLO depth (metres)              | Where they disagree (0% black -> 30%+ yellow)

Depth Anything's relative output has no units, so for display its one scale + shift is
fitted to YOLO's metres (shape unchanged). The ordering agreement number needs no fitting.

Needs `pip install ultralytics` (YOLO weights download into models/ on first run).
"""

import argparse
import sys
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

DEPTH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEPTH_ROOT))

from depth_estimator import DEFAULT_YOLO_DEPTH, DepthEstimator, YoloDepthEstimator, resolve_device
from depth_estimator.cli import add_estimator_arguments, estimator_kwargs
from depth_estimator.compare import align_to_metric, ordering_agreement, relative_difference
from depth_estimator.yolo_depth import DEFAULT_YOLO_IMGSZ

IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".bmp"}
FONT = cv2.FONT_HERSHEY_SIMPLEX
DIFF_CAP = 0.30   # disagreement colour scale tops out at 30%


def read_image(path: Path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def write_image(path: Path, image) -> None:
    ok, encoded = cv2.imencode(path.suffix, image)
    if ok:
        encoded.tofile(str(path))


def colour_metres(depth: np.ndarray, near: float, far: float) -> np.ndarray:
    """Metres -> BGR on a fixed range shared by both models. Near = red, far = blue, invalid = black."""
    t = np.clip((np.nan_to_num(depth, nan=far) - near) / max(far - near, 1e-6), 0.0, 1.0)
    image = cv2.applyColorMap(((1.0 - t) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    image[~np.isfinite(depth)] = 0
    return image


def colour_difference(diff: np.ndarray) -> np.ndarray:
    gray = (np.clip(np.nan_to_num(diff, nan=0.0) / DIFF_CAP, 0, 1) * 255).astype(np.uint8)
    image = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
    image[~np.isfinite(diff)] = (60, 60, 60)
    return image


def title_bar(panel, text):
    out = panel.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 30), (30, 30, 30), -1)
    cv2.putText(out, text, (10, 21), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def timed(fn, frame, sync):
    sync()
    start = perf_counter()
    out = fn(frame)
    sync()
    return out, (perf_counter() - start) * 1000


def compare_panels(rgb, depth_anything, yolo, sync):
    """Run both depth models on one RGB frame.

    Returns (panels as BGR images, stats dict). Used by this script and by the app's comparison tab.
    panels = {"original", "depth_anything", "yolo", "difference"}
    """
    da, da_ms = timed(depth_anything.estimate, rgb, sync)
    yo, yolo_ms = timed(yolo.estimate, rgb, sync)

    if da.is_metric:
        da_metres = da.depth                        # metric checkpoint: already metres
        fit = {"scale": 1.0, "shift": 0.0, "inliers": 1.0}
    else:
        da_metres, fit = align_to_metric(da, yo)

    diff = relative_difference(da_metres, yo)
    near, far = np.nanpercentile(yo.depth, [2, 98]) if yo.valid.any() else (0.5, 10.0)

    stats = {
        "depth_anything_ms": da_ms,
        "yolo_ms": yolo_ms,
        "ordering_agreement": ordering_agreement(da, yo),
        "median_disagreement": float(np.nanmedian(diff)) if np.isfinite(diff).any() else float("nan"),
        "yolo_range_m": (float(np.nanmin(yo.depth)), float(np.nanmax(yo.depth))) if yo.valid.any() else (np.nan, np.nan),
        "fit": fit,
        "depth_anything_kind": da.kind,
        "colour_range_m": (float(near), float(far)),
    }
    panels = {
        "original": cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        "depth_anything": colour_metres(da_metres, near, far),
        "yolo": colour_metres(yo.depth, near, far),
        "difference": colour_difference(diff),
    }
    return panels, stats


def compare_grid(rgb, depth_anything, yolo, sync):
    panels, stats = compare_panels(rgb, depth_anything, yolo, sync)
    scaled = "" if stats["depth_anything_kind"] == "metric_depth" else ", scaled to YOLO's metres"
    near, far = stats["colour_range_m"]
    titled = [
        title_bar(panels["original"], f"Original   (colours: red {near:.1f} m -> blue {far:.1f} m)"),
        title_bar(panels["depth_anything"], f"Depth Anything V2  {stats['depth_anything_ms']:.0f} ms{scaled}"),
        title_bar(panels["yolo"], f"YOLO depth  {stats['yolo_ms']:.0f} ms  (metres)"),
        title_bar(panels["difference"], f"Disagreement: median {stats['median_disagreement'] * 100:.0f}%, "
                                        f"order agrees {stats['ordering_agreement'] * 100:.0f}%"),
    ]
    return np.vstack([np.hstack(titled[:2]), np.hstack(titled[2:])]), stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_estimator_arguments(parser)
    parser.add_argument("--image", default="", help="one image")
    parser.add_argument("--images", default=str(DEPTH_ROOT / "test_images"), help="a folder of images")
    parser.add_argument("--video", default="", help="a video file (live window)")
    parser.add_argument("--output", default=str(DEPTH_ROOT / "compare_outputs"))
    parser.add_argument("--yolo-model", default=DEFAULT_YOLO_DEPTH)
    parser.add_argument("--yolo-imgsz", type=int, default=DEFAULT_YOLO_IMGSZ)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--save-video", default="", help="with --video: write the grid to this .mp4")
    args = parser.parse_args()

    device = resolve_device(args.device)
    import torch
    sync = torch.cuda.synchronize if device.startswith("cuda") else (lambda: None)

    with DepthEstimator(**{**estimator_kwargs(args), "device": device}) as da, \
            YoloDepthEstimator(args.yolo_model, device=device, imgsz=args.yolo_imgsz) as yolo:
        print(f"Depth Anything: {Path(da.model_name).name} ({da.kind}) | "
              f"YOLO: {Path(yolo.model_name).name} | device: {device}")

        if args.video:
            cap = cv2.VideoCapture(args.video)
            if not cap.isOpened():
                raise FileNotFoundError(f"Unable to open video: {args.video}")
            writer = None
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                grid, _ = compare_grid(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), da, yolo, sync)
                if args.save_video:
                    if writer is None:
                        fps = cap.get(cv2.CAP_PROP_FPS) or 15
                        writer = cv2.VideoWriter(args.save_video, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                                 (grid.shape[1], grid.shape[0]))
                    writer.write(grid)
                if not args.no_show:
                    cv2.imshow("Depth Anything V2 vs YOLO depth", grid)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            cap.release()
            if writer is not None:
                writer.release()
                print(f"Saved: {args.save_video}")
            cv2.destroyAllWindows()
            return

        paths = [Path(args.image)] if args.image else sorted(
            p for p in Path(args.images).iterdir() if p.suffix.lower() in IMAGE_TYPES)
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)

        for path in paths:
            bgr = read_image(path)
            if bgr is None:
                print(f"SKIP {path.name} (can't decode)")
                continue
            grid, stats = compare_grid(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), da, yolo, sync)
            out = out_dir / f"{path.stem}_compare.png"
            write_image(out, grid)
            lo, hi = stats["yolo_range_m"]
            print(f"{path.name}: Depth Anything {stats['depth_anything_ms']:.0f} ms | YOLO {stats['yolo_ms']:.0f} ms | "
                  f"order agrees {stats['ordering_agreement'] * 100:.0f}% | "
                  f"median disagreement {stats['median_disagreement'] * 100:.0f}% | YOLO range {lo:.1f}-{hi:.1f} m")
            print(f"  -> {out}")
            if not args.no_show:
                cv2.imshow("Depth Anything V2 vs YOLO depth", grid)
                cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
