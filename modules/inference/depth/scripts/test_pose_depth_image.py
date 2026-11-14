"""PoseDepthPipeline on one image: MediaPipe Pose Lite + YOLO depth (or a saved ZED depth map).

    python scripts/test_pose_depth_image.py --image photo.jpg
    python scripts/test_pose_depth_image.py --image left.png --zed-depth depth.npy      # metres
    python scripts/test_pose_depth_image.py --image left.png --zed-depth depth.npy --depth-scale 0.001   # mm
    python scripts/test_pose_depth_image.py --image photo.jpg --imgsz 1024 --no-show

Saves a side-by-side PNG (skeleton on the photo | on the depth map) and a .json with every joint.
Needs mediapipe, and pose_landmarker_lite.task in modules/inference/pose/models/.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

DEPTH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEPTH_ROOT))

from src import DEFAULT_YOLO_DEPTH, DEFAULT_YOLO_IMGSZ, PoseDepthPipeline, zed_to_rgb
from src.pipeline import DEFAULT_POSE_MODEL
from src.pose_depth import draw_pose_depth, joint_table
from src.utils import colorize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", required=True, help="a .jpg/.png (BGR or BGRA, e.g. a saved ZED left image)")
    parser.add_argument("--zed-depth", default="", help="optional .npy depth map for this image (skips YOLO)")
    parser.add_argument("--depth-scale", type=float, default=1.0, help="multiply --zed-depth by this to get metres")
    parser.add_argument("--yolo-model", default=DEFAULT_YOLO_DEPTH)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_YOLO_IMGSZ, help="YOLO input size (lower = faster)")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, mps")
    parser.add_argument("--pose-model", default=str(DEFAULT_POSE_MODEL))
    parser.add_argument("--output", default=str(DEPTH_ROOT / "test_outputs" / "pose_depth.png"))
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    image = cv2.imdecode(np.fromfile(args.image, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 3:
        raise FileNotFoundError(f"Unable to read a colour image from: {args.image}")
    rgb = zed_to_rgb(image)                       # BGR or BGRA -> RGB
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    zed_depth = np.load(args.zed_depth) if args.zed_depth else None

    with PoseDepthPipeline(args.pose_model, args.yolo_model, device=args.device, imgsz=args.imgsz,
                           depth_scale=args.depth_scale) as pipe:
        out = pipe.process_frame(rgb, frame_id=0, source_id=Path(args.image).name, capture_timestamp_ns=0,
                                 depth=zed_depth)

    t = out.timings_ms
    print(f"Depth source: {out.depth_source}  ·  pose {t['pose']:.0f} ms  depth {t['depth']:.0f} ms  "
          f"total {t['total']:.0f} ms (first run includes model warm-up)")
    rows = joint_table(out.frame)
    if not out.detected:
        print("No person detected.")
    else:
        print(f"Person (torso median): {out.body_depth:.2f} m\n")
        print(f"{'#':>2}  {'joint':<18} {'x':>5} {'y':>5} {'vis':>5}  depth")
        for i, name, x, y, vis, d in rows:
            print(f"{i:>2}  {name:<18} {x:>5} {y:>5} {vis:>5.2f}  {'-' if d is None else f'{d:.3f} m'}")

    panel = np.hstack([draw_pose_depth(bgr, out.frame), draw_pose_depth(colorize(out.depth), out.frame)])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(output.suffix or ".png", panel)[1].tofile(str(output))
    output.with_suffix(".json").write_text(json.dumps({
        "image": args.image, "depth_source": out.depth_source, "timings_ms": t,
        "detected": out.detected, "body_depth_m": None if np.isnan(out.body_depth) else out.body_depth,
        "joints": [dict(zip(["index", "name", "x", "y", "visibility", "depth_m"], r)) for r in rows],
    }, indent=2), encoding="utf-8")
    print(f"\nSaved: {output} (+ .json)")

    if not args.no_show:
        cv2.imshow("Pose + depth (any key to close)", panel)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
