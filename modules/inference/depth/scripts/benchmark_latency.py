"""Latency benchmark: Depth Anything V2 vs YOLO depth, same frames, same device.

    python scripts/benchmark_latency.py                        # test_images/, both models
    python scripts/benchmark_latency.py --video path/to/clip.mp4 --frames 100
    python scripts/benchmark_latency.py --device cuda --yolo-model yolo26s-depth.pt
    python scripts/benchmark_latency.py --only depth-anything  # or --only yolo

Needs `pip install ultralytics` for the YOLO part (weights download into models/ on first run).

Notes on fairness:
  * Both run on the same frames, at the same device and precision (fp16 on GPU).
  * Both are timed end to end (estimate(): model + resize + shared convention + confidence)
    and model-only (predict_raw()).
  * The first --warmup frames are run but NOT counted (CUDA/cuDNN set-up, lazy init).
  * GPU timings wait for the GPU to finish (torch.cuda.synchronize), otherwise
    you'd only measure how long it takes to *queue* the work.
  * Input resolution matters a lot: compare at similar sizes (--input-size for Depth
    Anything, --yolo-imgsz for YOLO), and say which you used.
"""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

DEPTH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEPTH_ROOT))

from depth_estimator import DEFAULT_YOLO_DEPTH, DepthEstimator, YoloDepthEstimator, resolve_device
from depth_estimator.cli import add_estimator_arguments, estimator_kwargs
from depth_estimator.yolo_depth import DEFAULT_YOLO_IMGSZ

IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".bmp"}


# ------------------------------------------------------------ frames

def load_frames(args) -> list[np.ndarray]:
    """RGB frames, cycled from a folder of images or read from a video."""
    frames = []
    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise FileNotFoundError(f"Unable to open video: {args.video}")
        while len(frames) < args.frames:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
    else:
        folder = Path(args.images)
        paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_TYPES) if folder.is_dir() else []
        if not paths:
            raise FileNotFoundError(f"No .jpg/.png images in {folder}")
        images = [cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR) for p in paths]
        images = [im for im in images if im is not None]
        frames = [images[i % len(images)] for i in range(args.frames)]

    if args.resize:
        w, h = args.resize
        frames = [cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) for f in frames]
    if not frames:
        raise RuntimeError("No frames loaded")
    return [np.ascontiguousarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]


# ------------------------------------------------------------ timing

def make_sync(device: str):
    """Returns a function that waits for queued GPU work, so timings are real."""
    import torch
    if device.startswith("cuda"):
        return torch.cuda.synchronize
    if device == "mps" and hasattr(torch, "mps"):
        return torch.mps.synchronize
    return lambda: None


def time_calls(fn, frames, warmup, sync) -> np.ndarray:
    """Per-frame latency in ms, warm-up frames excluded."""
    times = []
    for i, frame in enumerate(frames):
        sync()
        start = perf_counter()
        fn(frame)
        sync()
        if i >= warmup:
            times.append((perf_counter() - start) * 1000)
    return np.array(times)


def summarize(name, ms, extra=None) -> dict:
    row = {
        "model": name,
        "frames": int(ms.size),
        "mean_ms": float(ms.mean()),
        "median_ms": float(np.median(ms)),
        "p95_ms": float(np.percentile(ms, 95)),
        "min_ms": float(ms.min()),
        "fps": float(1000 / ms.mean()),
    }
    if extra:
        row.update(extra)
    return row


def bench(label, estimator, frames, warmup, input_res) -> list[dict]:
    sync = make_sync(estimator.device)
    full = time_calls(estimator.estimate, frames, warmup, sync)
    model_only = time_calls(estimator.predict_raw, frames, warmup, sync)
    extra = {"device": estimator.device, "input": input_res, "kind": estimator.kind}
    return [
        summarize(f"{label} (full: model + convention + confidence)", full, extra),
        summarize(f"{label} (model only)", model_only, extra),
    ]


# ------------------------------------------------------------ main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_estimator_arguments(parser)
    parser.add_argument("--images", default=str(DEPTH_ROOT / "test_images"))
    parser.add_argument("--video", default="", help="use frames from this video instead of --images")
    parser.add_argument("--frames", type=int, default=30, help="frames to time (after warm-up)")
    parser.add_argument("--warmup", type=int, default=5, help="extra frames run first and not counted")
    parser.add_argument("--resize", type=int, nargs=2, metavar=("W", "H"),
                        help="resize every frame first, e.g. --resize 1280 720 to match the ZED")
    parser.add_argument("--yolo-model", default=DEFAULT_YOLO_DEPTH,
                        help="yolo26n-depth.pt (fastest), yolo26s-depth.pt, yolo26m-depth.pt, ...")
    parser.add_argument("--yolo-imgsz", type=int, default=DEFAULT_YOLO_IMGSZ)
    parser.add_argument("--only", choices=["depth-anything", "yolo"], help="benchmark just one model")
    parser.add_argument("--save", default="", help="also write results to this .json file")
    args = parser.parse_args()

    args.frames += args.warmup
    frames = load_frames(args)
    h, w = frames[0].shape[:2]
    device = resolve_device(args.device)
    print(f"{len(frames)} frames at {w}x{h} on {device} ({args.warmup} warm-up, not counted)\n")

    rows = []
    if args.only != "yolo":
        with DepthEstimator(**{**estimator_kwargs(args), "device": device}) as est:
            print(f"Depth Anything V2: {Path(est.model_name).name} ({est.encoder}), "
                  f"input_size={est.input_size}, dtype={est.settings['dtype']}, output={est.kind}")
            rows += bench("Depth Anything V2", est, frames, args.warmup, est.input_size)

    if args.only != "depth-anything":
        try:
            with YoloDepthEstimator(args.yolo_model, device=device, imgsz=args.yolo_imgsz) as est:
                print(f"YOLO depth: {Path(est.model_name).name}, imgsz={est.imgsz}, "
                      f"dtype={est.settings['dtype']}, output={est.kind}")
                rows += bench(f"YOLO depth {Path(args.yolo_model).stem}", est, frames, args.warmup, est.imgsz)
        except ImportError as e:
            print(f"\nSkipping YOLO: {e}")

    print()
    print(f"{'model':<66} {'median':>8} {'mean':>8} {'p95':>8} {'FPS':>7}")
    print("-" * 101)
    for r in rows:
        print(f"{r['model']:<66} {r['median_ms']:>6.1f}ms {r['mean_ms']:>6.1f}ms {r['p95_ms']:>6.1f}ms {r['fps']:>7.1f}")

    full = [r for r in rows if "full" in r["model"]]
    if len(full) == 2:
        ratio = full[0]["median_ms"] / full[1]["median_ms"]
        faster = "YOLO depth" if ratio > 1 else "Depth Anything V2"
        print(f"\n{faster} is {max(ratio, 1 / ratio):.1f}x faster per frame (median, end to end).")

    if args.save:
        Path(args.save).write_text(json.dumps({"frame_size": [w, h], "device": device, "results": rows}, indent=2),
                                   encoding="utf-8")
        print(f"Saved: {args.save}")


if __name__ == "__main__":
    main()
