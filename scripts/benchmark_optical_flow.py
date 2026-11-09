"""Benchmark DIS or RAFT Small on the same recorded Stream Handler frames.

Run from the repository root with the environment that has OpenCV installed:
python scripts/benchmark_optical_flow.py path/to/recording.mp4 --max-frames 300
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "modules/stream_handler/src"))
sys.path.insert(0, str(ROOT / "modules/inference/optical_flow/src"))

from optical_flow import FlowConfig, OpticalFlowTracker, create_estimator  # noqa: E402
from stream_handler import RecordedVideoSource, ReplayMode, SourceExhausted, StreamHandler  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "config/optical_flow.example.json")
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--warmup-pairs", type=int, default=5,
                        help="Exclude this many initial valid pairs from timing statistics")
    args = parser.parse_args()
    if args.max_frames < 2:
        parser.error("--max-frames must be at least 2")
    if args.warmup_pairs < 0:
        parser.error("--warmup-pairs must be non-negative")
    if not args.video.is_file():
        parser.error(f"video does not exist: {args.video}")

    config = FlowConfig(**json.loads(args.config.read_text(encoding="utf-8")))
    if config.flow_model == "raft_small":
        import torch

        torch.hub.set_dir(str(ROOT / "outputs/torch_cache"))
    handler = StreamHandler(RecordedVideoSource(args.video, replay_mode=ReplayMode.FASTEST))
    tracker = OpticalFlowTracker(create_estimator(config))
    latencies_ms: list[float] = []
    valid_fractions: list[float] = []
    statuses: dict[str, int] = {}
    ok_pairs = 0
    handler.open()
    try:
        for _ in range(args.max_frames):
            try:
                handler.pump_once()
            except SourceExhausted:
                break
            frame = handler.queue.get(timeout_s=0)
            assert frame is not None
            start = time.perf_counter_ns()
            result = tracker.push(frame)
            duration_ms = (time.perf_counter_ns() - start) / 1_000_000
            statuses[result.status] = statuses.get(result.status, 0) + 1
            if result.status == "ok":
                ok_pairs += 1
                if ok_pairs > args.warmup_pairs:
                    latencies_ms.append(duration_ms)
                    valid_fractions.append(result.valid_fraction)
    finally:
        handler.close()

    report = {
        "video": str(args.video),
        "frames": sum(statuses.values()),
        "warmup_pairs_skipped": min(ok_pairs, args.warmup_pairs),
        "measured_pairs": len(latencies_ms),
        "statuses": statuses,
        "flow_model": config.flow_model,
        "dis_preset": config.dis_preset if config.flow_model == "opencv_dis" else None,
        "raft_iterations": config.raft_iterations if config.flow_model == "raft_small" else None,
        "flow_max_size": [config.flow_width, config.flow_height],
        "flow_latency_ms_p50": round(statistics.median(latencies_ms), 3) if latencies_ms else None,
        "flow_latency_ms_p95": round(sorted(latencies_ms)[min(len(latencies_ms) - 1, int(0.95 * len(latencies_ms)))], 3) if latencies_ms else None,
        "valid_fraction_mean": round(statistics.mean(valid_fractions), 3) if valid_fractions else None,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
