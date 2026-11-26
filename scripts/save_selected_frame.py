"""Save one selected live frame as an aligned NPZ for the CUDA ray experiment.

The ray experiment in ``experiment/cuda-depth-rays`` needs the RGB image, the
depth plane, the person mask, and the intrinsics from ONE frame, so the splat
viewer and the ray renderer describe the same subject. This script runs the
normal live path until a refresh succeeds, then writes those four arrays.

    python scripts/save_selected_frame.py --source zed --out outputs/frame.npz

The saved depth follows the shared convention: metres, camera Z axis, NaN where
a pixel has no measurement. The mask is the selected person mask for that same
refresh. The intrinsics are the ones the refresh used, so the arrays line up.

Only the frozen live path is used, and nothing here changes it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "applications"),
    str(ROOT / "modules/stream_handler/src"),
    str(ROOT / "modules/inference/depth"),
]

from live_pipeline.adapters import DepthThresholdConfig  # noqa: E402
from live_pipeline.pipeline import RefreshError, RefreshPipeline  # noqa: E402
from stream_handler import (  # noqa: E402
    CameraSource,
    RecordedVideoSource,
    ReplayMode,
    StreamHandler,
    ZedCameraSource,
)


def build_source(args):
    if args.source == "video":
        if args.path is None or not Path(args.path).is_file():
            raise SystemExit("--source video needs an existing --path")
        return RecordedVideoSource(args.path, replay_mode=ReplayMode.PACED if args.video_paced else ReplayMode.FASTEST)
    if args.source == "zed":
        return ZedCameraSource(
            args.zed_camera_id,
            resolution=args.zed_resolution,
            fps=args.zed_fps,
            depth_mode=args.zed_depth_mode,
        )
    return CameraSource(args.device, backend=args.camera_backend)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=("webcam", "camera", "zed", "video"), default="zed")
    parser.add_argument("--path", type=Path, default=None)
    parser.add_argument("--video-paced", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--camera-backend", choices=("auto", "dshow", "msmf"), default="dshow")
    parser.add_argument("--zed-camera-id", type=int, default=0)
    parser.add_argument("--zed-resolution", choices=("VGA", "HD720", "HD1080", "HD2K"), default="HD720")
    parser.add_argument("--zed-fps", type=int, default=30)
    parser.add_argument(
        "--zed-depth-mode",
        choices=("NEURAL", "NEURAL_PLUS", "ULTRA", "QUALITY", "PERFORMANCE", "NONE"),
        default="QUALITY",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output .npz path")
    parser.add_argument("--report", type=Path, default=None, help="Output .json beside the npz")
    parser.add_argument("--sample-stride", type=int, default=2)
    parser.add_argument("--horizontal-fov-deg", type=float, default=60.0)
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument("--yolo-model", default=None)
    parser.add_argument("--device-inference", default="auto")
    parser.add_argument("--depth-band-m", type=float, default=0.75)
    parser.add_argument("--depth-near-percentile", type=float, default=2.0)
    parser.add_argument("--max-mask-area", type=float, default=0.60)
    parser.add_argument("--timeout-s", type=float, default=60.0, help="Give up after this many seconds")
    args = parser.parse_args()

    pipeline = RefreshPipeline(
        mask_config=DepthThresholdConfig(
            band_m=args.depth_band_m,
            near_percentile=args.depth_near_percentile,
            max_area_fraction=args.max_mask_area,
        ),
        sample_stride=args.sample_stride,
        horizontal_fov_deg=args.horizontal_fov_deg,
        yolo_imgsz=args.yolo_imgsz,
        yolo_model=args.yolo_model,
        device=args.device_inference,
        require_pose=True,
        generation=0,
    )
    handler = StreamHandler(build_source(args), queue_capacity=4, max_reconnect_attempts=3)
    handler.start()
    deadline = time.monotonic() + float(args.timeout_s)
    outcome = None
    attempts = 0
    failures: dict[str, int] = {}
    try:
        while time.monotonic() < deadline:
            packet = handler.queue.get(timeout_s=0.05)
            if packet is None:
                continue
            attempts += 1
            try:
                outcome = pipeline.process(packet, generation=0)
            except RefreshError as error:
                reason = str(getattr(error, "reason", "other"))
                failures[reason] = failures.get(reason, 0) + 1
                continue
            break
    finally:
        try:
            handler.stop()
        except Exception:
            pass

    if outcome is None:
        pipeline.close()
        raise SystemExit(f"no usable refresh in {args.timeout_s:.0f} s; failures: {json.dumps(failures)}")

    depth_m = np.asarray(outcome.depth_m, dtype=np.float32)
    mask = np.asarray(outcome.mask.mask)
    # The packet that produced this refresh carries the image and calibration.
    rgb = np.asarray(packet.rgb, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise SystemExit(f"rgb has shape {rgb.shape}, expected [H, W, 3]")
    height, width = depth_m.shape
    if rgb.shape[:2] != (height, width):
        raise SystemExit(f"rgb is {rgb.shape[:2]} but depth is {depth_m.shape}")
    if mask.shape != (height, width):
        raise SystemExit(f"mask is {mask.shape} but depth is {depth_m.shape}")
    if packet.intrinsics is None:
        raise SystemExit("this frame carries no intrinsics; the ray experiment needs them")
    intrinsics = np.asarray(packet.intrinsics, dtype=np.float64)
    if intrinsics.shape != (4,):
        raise SystemExit(f"intrinsics has shape {intrinsics.shape}, expected [fx, fy, cx, cy]")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        rgb=rgb,
        depth_m=depth_m,
        mask=mask.astype(np.uint8),
        intrinsics=intrinsics,
    )

    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    mask_bool = mask.astype(bool)
    report = {
        "npz": str(args.out),
        "frame_id": str(getattr(packet, "frame_id", "")),
        "source_id": str(getattr(packet, "source_id", "")),
        "capture_timestamp_ns": int(getattr(packet, "capture_timestamp_ns", 0)),
        "generation": int(getattr(outcome, "frame_id", 0)) and 0,
        "depth_source": str(outcome.depth_source),
        "resolution": [int(width), int(height)],
        "intrinsics_fx_fy_cx_cy": [float(v) for v in intrinsics],
        "depth_units": "metres",
        "depth_valid_fraction": round(float(valid.mean()), 4),
        "depth_valid_in_mask_fraction": round(
            float(np.count_nonzero(valid & mask_bool) / max(1, np.count_nonzero(mask_bool))), 4
        ),
        "mask_area_fraction": round(float(outcome.mask.area_fraction), 4),
        "splat_count": int(outcome.splat_count),
        "timings_ms": {k: round(float(v), 2) for k, v in outcome.timings_ms.items()},
        "attempts_before_success": attempts,
        "failures_before_success": failures,
        "saved_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    report_path = args.report if args.report is not None else args.out.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    pipeline.close()
    print(json.dumps({"saved": report}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
