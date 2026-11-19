"""Extract selected frame pairs and accepted baseline signals from one clip.

Example:
    python -m scripts.build_learning_replay --clip data/recordings/person01.mp4 \
        --session-id person01 --split train
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT), str(ROOT / "modules/stream_handler/src"),
    str(ROOT / "modules/inference/optical_flow/src"),
    str(ROOT / "modules/reranker/src"),
]

from applications.live_pipeline.pipeline import RefreshPipeline  # noqa: E402
from applications.live_pipeline.replay import LiveReplayProcessor, ReplaySettings, run_replay  # noqa: E402
from applications.live_pipeline.window import WindowSelector  # noqa: E402
from optical_flow import FlowConfig, create_estimator  # noqa: E402
from stream_handler.contracts import FramePacket, FrameQuality, TimestampOrigin  # noqa: E402


def video_packets(path: Path, source_id: str, max_frames: int = 0):
    """Read frames in order with one timestamp origin for the complete clip."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OSError(f"could not open clip: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        capture.release()
        raise ValueError("clip has no valid frame rate; capture timestamps cannot be recovered")
    # Flow rejects a pair when the timestamp origin changes. Check presentation
    # timestamps before decoding so the complete clip uses one origin.
    pts: list[int] = []
    while (not max_frames or len(pts) < max_frames) and capture.grab():
        position_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
        pts.append(round(position_ms * 1_000_000) if np.isfinite(position_ms) else -1)
    capture.release()
    use_pts = bool(pts) and pts[0] >= 0 and all(left < right for left, right in zip(pts, pts[1:]))
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OSError(f"could not reopen clip: {path}")
    frame_id = 0
    try:
        while not max_frames or frame_id < max_frames:
            ok, bgr = capture.read()
            if not ok:
                break
            frame_id += 1
            timestamp_ns = pts[frame_id - 1] if use_pts else round((frame_id - 1) * 1_000_000_000 / fps)
            origin = TimestampOrigin.SOURCE_PRESENTATION if use_pts else TimestampOrigin.DECODE_MONOTONIC
            yield FramePacket(
                frame_id=frame_id, source_id=source_id, capture_timestamp_ns=timestamp_ns,
                timestamp_origin=origin, rgb=np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)),
                quality=FrameQuality(decode_latency_ns=0, frame_age_ns=0, frame_age_available=False),
            )
    finally:
        capture.release()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--clip-id", default=None)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/processed/learning_replay")
    parser.add_argument("--max-frames", type=int, default=0, help="0 reads the complete clip")
    parser.add_argument("--threshold", type=float, default=0.02)
    parser.add_argument("--window-ms", type=float, default=100.0)
    parser.add_argument("--active-age-ms", type=float, default=500.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--yolo-model", default=None)
    args = parser.parse_args()
    clip = args.clip.resolve()
    if not clip.is_file():
        parser.error(f"clip does not exist: {clip}")
    if args.max_frames < 0 or args.window_ms <= 0 or args.active_age_ms <= 0:
        parser.error("max-frames must be non-negative; window and active age must be positive")
    settings = ReplaySettings(
        session_id=args.session_id, clip_id=args.clip_id or clip.stem, split=args.split,
        window_ns=round(args.window_ms * 1_000_000),
    )
    session_dir = args.output_root / settings.session_id
    output_dir = session_dir / settings.clip_id
    session_dir.mkdir(parents=True, exist_ok=True)
    split_file = session_dir / "split.json"
    if split_file.exists():
        old = json.loads(split_file.read_text(encoding="utf-8"))
        if old.get("split") != settings.split:
            raise ValueError("one recording session cannot be assigned to different splits")
    if output_dir.exists():
        raise FileExistsError(f"replay output already exists: {output_dir}")
    split_file.write_text(json.dumps({"session_id": settings.session_id, "split": settings.split}, indent=2), encoding="utf-8")

    flow_config = FlowConfig()
    selector = WindowSelector(
        create_estimator(flow_config), threshold=args.threshold,
        active_age_ns=round(args.active_age_ms * 1_000_000),
    )
    pipeline = RefreshPipeline(device=args.device, yolo_model=args.yolo_model)
    pose_path = ROOT / "modules/inference/pose/models/pose_landmarker_lite.task"
    depth_name = args.yolo_model or "yolo26n-depth.pt"
    depth_path = Path(depth_name)
    if not depth_path.is_file():
        depth_path = ROOT / "modules/inference/depth/models" / depth_path.name
    models = {
        "flow": f"{flow_config.flow_model}:{flow_config.dis_preset}:opencv-{cv2.__version__}",
        "pose": f"{pose_path.name}:sha256:{_sha256(pose_path)}" if pose_path.is_file() else str(pose_path),
        "depth": f"{depth_path.name}:sha256:{_sha256(depth_path)}" if depth_path.is_file() else str(depth_path),
        "mask": "pose_gradient_with_depth_band_fallback",
        "depth_imgsz": str(pipeline.yolo_imgsz),
        "depth_confidence": str(pipeline.depth_confidence),
    }
    try:
        counts = run_replay(
            video_packets(clip, f"recorded:{settings.session_id}:{settings.clip_id}", args.max_frames),
            selector=selector, processor=LiveReplayProcessor(pipeline), output_dir=output_dir,
            settings=settings, models=models,
        )
    finally:
        pipeline.close()
    manifest = {
        "schema_version": "1", "session_id": settings.session_id, "clip_id": settings.clip_id,
        "split": settings.split, "clip_sha256": _sha256(clip), "clip_path": str(clip),
        "source_id": f"recorded:{settings.session_id}:{settings.clip_id}",
        "window_ns": settings.window_ns, "threshold": args.threshold,
        "active_age_ns": round(args.active_age_ms * 1_000_000),
        "timestamp_rule": "whole-clip increasing presentation timestamps; frame index / FPS otherwise",
        "models": models, "research_size": [settings.research_width, settings.research_height],
        "counts": counts, "note": "offline synchronous replay; timings exclude capture queue and browser draw",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_dir), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
