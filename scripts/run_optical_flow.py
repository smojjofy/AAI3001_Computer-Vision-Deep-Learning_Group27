"""Inspect Stream Handler -> optical flow on live or recorded input.

From the repository root:
  python scripts/run_optical_flow.py --source video --path recording.mp4
  python scripts/run_optical_flow.py --source camera --device 0 --duration-s 20

Outputs JSON diagnostics only; no image or flow planes are written.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "modules/stream_handler/src"))
sys.path.insert(0, str(ROOT / "modules/inference/optical_flow/src"))

from optical_flow import FlowConfig, OpticalFlowTracker, StreamFlowConsumer, create_estimator  # noqa: E402
from stream_handler import (  # noqa: E402
    CameraSource, OBSVirtualCameraSource, RecordedVideoSource, ReplayMode,
    RTSPSource, StreamHandler, StreamHealth,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("video", "camera", "obs", "rtsp"), required=True)
    parser.add_argument("--path", type=Path, help="Recorded video path for --source video")
    parser.add_argument("--device", type=int, default=0, help="Device index for camera or OBS")
    parser.add_argument("--url", help="RTSP URL; avoid credentials in shell history")
    parser.add_argument("--replay-mode", choices=("paced", "fastest"), default="paced")
    parser.add_argument("--duration-s", type=float, default=15.0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--queue-capacity", type=int, default=3)
    parser.add_argument("--summary-only", action="store_true",
                        help="Suppress per-frame JSON and reduce diagnostic output overhead")
    parser.add_argument("--config", type=Path, default=ROOT / "config/optical_flow.example.json")
    args = parser.parse_args()
    if args.duration_s <= 0 or args.queue_capacity < 1 or (args.max_frames is not None and args.max_frames < 1):
        parser.error("duration, queue capacity, and max frames must be positive")
    if args.source == "video":
        if args.path is None or not args.path.is_file():
            parser.error("--source video requires an existing --path")
        source = RecordedVideoSource(args.path, replay_mode=ReplayMode(args.replay_mode))
    elif args.source == "camera":
        source = CameraSource(args.device)
    elif args.source == "obs":
        source = OBSVirtualCameraSource(args.device)
    else:
        if not args.url:
            parser.error("--source rtsp requires --url")
        source = RTSPSource(args.url)

    config = FlowConfig(**json.loads(args.config.read_text(encoding="utf-8")))
    if config.flow_model == "raft_small":
        import torch

        torch.hub.set_dir(str(ROOT / "outputs/torch_cache"))
    handler = StreamHandler(source, queue_capacity=args.queue_capacity)
    consumer = StreamFlowConsumer(handler, OpticalFlowTracker(create_estimator(config)))
    durations_ms: list[float] = []
    valid_fractions: list[float] = []
    statuses: dict[str, int] = {}
    seen = 0
    deadline = time.monotonic() + args.duration_s
    started = time.monotonic()
    handler.start()
    try:
        while time.monotonic() < deadline and (args.max_frames is None or seen < args.max_frames):
            observation = consumer.poll(timeout_s=0.25)
            if observation is None:
                if args.source == "video" and handler.health() is StreamHealth.CLOSED and handler.queue.depth == 0 and handler.metrics.emitted_frames:
                    break
                continue
            packet, result = observation.packet, observation.flow
            seen += 1
            statuses[result.status] = statuses.get(result.status, 0) + 1
            duration_ms = observation.processing_time_ns / 1_000_000
            if result.status == "ok":
                durations_ms.append(duration_ms)
                valid_fractions.append(result.valid_fraction)
            if not args.summary_only:
                valid_vectors = result.flow[result.valid_mask]
                median_motion = np.median(valid_vectors, axis=0).tolist() if len(valid_vectors) else None
                print(json.dumps({
                    "source_id": packet.source_id,
                    "frame_id": packet.frame_id,
                    "previous_frame_id": result.previous_frame_id,
                    "capture_timestamp_ns": packet.capture_timestamp_ns,
                    "elapsed_ns": result.elapsed_ns,
                    "status": result.status,
                    "flow_backend": result.backend,
                    "flow_ms": round(duration_ms, 3),
                    "valid_fraction": round(result.valid_fraction, 3),
                    "median_motion_px": median_motion,
                    "queue_drops_total": packet.quality.dropped_frame_count,
                }), flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        handler.stop()
    elapsed_s = time.monotonic() - started
    ordered_ms = sorted(durations_ms)
    print(json.dumps({
        "summary": {
            "flow_model": config.flow_model,
            "elapsed_s": round(elapsed_s, 3),
            "frames_consumed": seen,
            "frames_emitted": handler.metrics.emitted_frames,
            "queue_drops_total": handler.queue.dropped_total,
            "queue_drop_fraction": round(handler.queue.dropped_total / handler.metrics.emitted_frames, 3) if handler.metrics.emitted_frames else None,
            "statuses": statuses,
            "flow_ms_p50": round(statistics.median(durations_ms), 3) if durations_ms else None,
            "flow_ms_p95": round(ordered_ms[min(len(ordered_ms) - 1, int(0.95 * len(ordered_ms)))], 3) if ordered_ms else None,
            "valid_fraction_mean": round(statistics.mean(valid_fractions), 3) if valid_fractions else None,
        }
    }), flush=True)


if __name__ == "__main__":
    main()
