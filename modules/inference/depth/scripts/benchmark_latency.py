
import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
 
import cv2
import numpy as np
 
DEPTH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEPTH_ROOT))
 
from src import DEFAULT_YOLO_DEPTH, YoloDepthEstimator, resolve_device
from src.yolo_depth import DEFAULT_YOLO_IMGSZ
 
IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".bmp"}
 
 
# ------------------------------------------------------------ frames
 
def load_frames(args) -> list[np.ndarray]:
    """RGB frames, cycled from a folder of images."""
    folder = Path(args.images)
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_TYPES) if folder.is_dir() else []
    if not paths:
        raise FileNotFoundError(f"No .jpg/.png images in {folder}")
    images = [cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR) for p in paths]
    images = [im for im in images if im is not None]
    if not images:
        raise RuntimeError(f"None of the images in {folder} could be read")
    frames = [images[i % len(images)] for i in range(args.frames)]
 
    if args.resize:
        w, h = args.resize
        frames = [cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) for f in frames]
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
 
 
def bench(label, estimator, frames, warmup) -> list[dict]:
    sync = make_sync(estimator.device)
    full = time_calls(estimator.estimate, frames, warmup, sync)
    model_only = time_calls(estimator.predict_raw, frames, warmup, sync)
    extra = {"device": estimator.device, "input": estimator.imgsz, "kind": estimator.kind}
    return [
        summarize(f"{label} (full: model + convention + confidence)", full, extra),
        summarize(f"{label} (model only)", model_only, extra),
    ]
 
 
# ------------------------------------------------------------ main
 
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", default=str(DEPTH_ROOT / "test_images"))
    parser.add_argument("--frames", type=int, default=30, help="frames to time (after warm-up)")
    parser.add_argument("--warmup", type=int, default=5, help="extra frames run first and not counted")
    parser.add_argument("--resize", type=int, nargs=2, metavar=("W", "H"),
                        help="resize every frame first, e.g. --resize 1280 720 to match the ZED")
    parser.add_argument("--yolo-model", default=DEFAULT_YOLO_DEPTH,
                        help="yolo26n-depth.pt (fastest), yolo26s-depth.pt, yolo26m-depth.pt, ...")
    parser.add_argument("--yolo-imgsz", type=int, default=DEFAULT_YOLO_IMGSZ)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, mps")
    parser.add_argument("--save", default="", help="also write results to this .json file")
    args = parser.parse_args()
 
    args.frames += args.warmup
    frames = load_frames(args)
    h, w = frames[0].shape[:2]
    device = resolve_device(args.device)
    print(f"{len(frames)} frames at {w}x{h} on {device} ({args.warmup} warm-up, not counted)\n")
 
    with YoloDepthEstimator(args.yolo_model, device=device, imgsz=args.yolo_imgsz) as est:
        print(f"YOLO depth: {Path(est.model_name).name}, imgsz={est.imgsz}, "
              f"dtype={est.settings['dtype']}, output={est.kind}")
        rows = bench(f"YOLO depth {Path(args.yolo_model).stem}", est, frames, args.warmup)
 
    print()
    print(f"{'model':<66} {'median':>8} {'mean':>8} {'p95':>8} {'FPS':>7}")
    print("-" * 101)
    for r in rows:
        print(f"{r['model']:<66} {r['median_ms']:>6.1f}ms {r['mean_ms']:>6.1f}ms {r['p95_ms']:>6.1f}ms {r['fps']:>7.1f}")
 
    if args.save:
        Path(args.save).write_text(json.dumps({"frame_size": [w, h], "device": device, "results": rows}, indent=2),
                                   encoding="utf-8")
        print(f"Saved: {args.save}")
 
 
if __name__ == "__main__":
    main()