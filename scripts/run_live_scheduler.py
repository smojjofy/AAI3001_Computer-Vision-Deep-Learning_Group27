"""Exercise latest-frame flow, reranking, and a simulated slow refresh worker.
 
Run from the repository root. By default this command does not create depth, pose, splats,
or a viewer; the refresh worker only waits for the configured duration.
 
--pose-depth makes the refresh worker run the real PoseDepthPipeline instead
(MediaPipe Pose Lite + YOLO depth, modules/inference/depth) on each refreshed frame.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
import threading
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "modules/stream_handler/src"),
    str(ROOT / "modules/inference/optical_flow/src"),
    str(ROOT / "modules/reranker/src"),
    str(ROOT / "applications"),
]
sys.path.insert(0, str(ROOT / "modules/inference/depth"))   # pose + depth pipeline ("from src import ...")

from backend.preview_server import PreviewServer  # noqa: E402
from live_pipeline import LiveRerankerScheduler, RefreshDispatcher  # noqa: E402
from optical_flow import FlowConfig, create_estimator  # noqa: E402
from reranker import Reranker, RerankerConfig, StreamRerankerConsumer  # noqa: E402
from stream_handler import (  # noqa: E402
    CameraSource, OBSVirtualCameraSource, RecordedVideoSource, ReplayMode,
    StreamHandler, StreamHealth,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("video", "camera", "obs"), required=True)
    parser.add_argument("--path", type=Path, help="Recorded video for --source video")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--camera-backend", choices=("auto", "dshow", "msmf"), default="auto")
    parser.add_argument("--config", type=Path, default=ROOT / "config/optical_flow_raft.example.json")
    parser.add_argument("--duration-s", type=float, default=20.0,
                        help="Run length in seconds; 0 runs until Ctrl+C")
    parser.add_argument("--sample-ms", type=float, default=250.0)
    parser.add_argument("--max-reference-age-ms", type=float, default=500.0)
    parser.add_argument("--refresh-work-ms", type=float, default=1000.0,
                        help="Simulated refresh time (ignored with --pose-depth)")
    parser.add_argument("--pose-depth", action="store_true",
                        help="Run MediaPipe pose + YOLO depth as the refresh work instead of sleeping")
    parser.add_argument("--queue-capacity", type=int, default=3)
    parser.add_argument("--gui-preview", action="store_true",
                        help="Serve the same captured frames to the local GUI on port 8765")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    if (
        not all(math.isfinite(value) for value in (args.duration_s, args.sample_ms, args.max_reference_age_ms, args.refresh_work_ms))
        or args.duration_s < 0 or args.sample_ms <= 0 or args.max_reference_age_ms < 0.000001
        or args.refresh_work_ms < 0 or args.queue_capacity < 1
    ):
        parser.error("duration and worker time must be non-negative; sample interval and reference age must be positive; queue capacity at least one")
    if args.source == "video":
        if args.path is None or not args.path.is_file():
            parser.error("--source video requires an existing --path")
        source = RecordedVideoSource(args.path, replay_mode=ReplayMode.PACED)
    elif args.source == "camera":
        source = CameraSource(args.device, backend=args.camera_backend)
    else:
        source = OBSVirtualCameraSource(args.device, backend=args.camera_backend)

    config = FlowConfig(**json.loads(args.config.read_text(encoding="utf-8")))
    if config.flow_model == "raft_small":
        import torch

        torch.hub.set_dir(str(ROOT / "outputs/torch_cache"))

    handler = StreamHandler(source, queue_capacity=args.queue_capacity, max_reconnect_attempts=3)
    reranker = Reranker(RerankerConfig(max_reference_age_ns=round(args.max_reference_age_ms * 1_000_000)))
    consumer = StreamRerankerConsumer(handler, create_estimator(config), reranker)

    latest_pose_depth: dict[str, object] = {}       # newest pose + depth result (for the GUI later)
    pose_depth = None

    if args.pose_depth:
        from src import PoseDepthPipeline
        from src.preview import encode_pose_depth_preview

        print("Loading pose + depth models ...", flush=True)
        pose_depth = PoseDepthPipeline(yolo_model="yolo26n-depth_openvino_model")           

        def refresh(job) -> None:
            packet = job.packet
            # packet.rgb: RGB uint8 [H, W, 3]. No ZED depth on the packet yet -> YOLO depth.
            out = pose_depth.process_frame(packet.rgb, packet.frame_id, packet.source_id,
                                           packet.capture_timestamp_ns)
            latest_pose_depth["result"] = out
            if args.gui_preview:                      # ready-made images + joints for /api/pose-depth
                latest_pose_depth["preview"] = encode_pose_depth_preview(packet.rgb, out)
            if not args.summary_only:
                print(json.dumps({
                    "pose_depth_frame_id": out.frame_id,
                    "person_detected": out.detected,
                    "person_m": None if math.isnan(out.body_depth) else round(out.body_depth, 2),
                    "depth_source": out.depth_source,
                    "ms": {k: round(v, 1) for k, v in out.timings_ms.items()},
                }), flush=True)
    else:
        def refresh(_job) -> None:
            time.sleep(args.refresh_work_ms / 1000)   # simulated refresh work

    dispatcher = RefreshDispatcher(refresh)
    scheduler = LiveRerankerScheduler(consumer, dispatcher, round(args.sample_ms * 1_000_000))
    preview_server = (
        PreviewServer(handler, pose_depth=lambda: latest_pose_depth.get("preview"))
        if args.gui_preview else None
    )
    preview_thread = (
        threading.Thread(target=preview_server.serve_forever, name="gui-preview", daemon=True)
        if preview_server is not None else None
    )
    actions: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    started = time.monotonic()
    deadline = math.inf if args.duration_s == 0 else started + args.duration_s
    handler.start()
    try:
        if preview_thread is not None:
            preview_thread.start()
            print("GUI preview ready at http://127.0.0.1:8765/api/preview", flush=True)
            if args.pose_depth:
                print("Pose + depth preview at http://127.0.0.1:8765/api/pose-depth", flush=True)
        while time.monotonic() < deadline:
            observation = scheduler.poll(timeout_s=0.25)
            if observation is None:
                if (
                    handler.health() is StreamHealth.CLOSED
                    and handler.queue.depth == 0
                ):
                    break
                continue
            decision = observation.decision
            actions[decision.action] += 1
            reasons[decision.reason] += 1
            if not args.summary_only:
                print(json.dumps({
                    "frame_id": decision.selected_frame_id,
                    "reference_frame_id": decision.reference_frame_id,
                    "action": decision.action,
                    "reason": decision.reason,
                    "flow_status": None if observation.flow is None else observation.flow.status,
                    "processing_ms": round(observation.processing_time_ns / 1_000_000, 2),
                }), flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        capture_finished = time.monotonic()
        capture_health_before_stop = str(handler.health())
        handler.stop()
        if preview_server is not None:
            if preview_thread is not None and preview_thread.is_alive():
                preview_server.shutdown()
                preview_thread.join(timeout=2)
            preview_server.server_close()
        dispatcher.close(drain=True)
        if pose_depth is not None:
            pose_depth.close()

    print(json.dumps({"summary": {
        "source": args.source,
        "device": args.device if args.source != "video" else None,
        "camera_backend": args.camera_backend if args.source != "video" else None,
        "flow_model": config.flow_model,
        "sample_ms": args.sample_ms,
        "max_reference_age_ms": args.max_reference_age_ms,
        "capture_loop_s": round(capture_finished - started, 3),
        "elapsed_s_including_shutdown": round(time.monotonic() - started, 3),
        "frames_emitted": handler.metrics.emitted_frames,
        "capture_overall_fps": round(handler.metrics.emitted_frames / (capture_finished - started), 3),
        "capture_active_fps": round(handler.metrics.capture_fps, 3),
        "capture_health_before_stop": capture_health_before_stop,
        "capture_idle_before_stop_s": (
            round(max(0.0, capture_finished - handler.metrics.last_successful_frame_ns / 1_000_000_000), 3)
            if handler.metrics.last_successful_frame_ns is not None else None
        ),
        "decode_latency_mean_ms": round(handler.metrics.decode_latency_mean_ns / 1_000_000, 3),
        "decode_latency_max_ms": round(handler.metrics.decode_latency_max_ns / 1_000_000, 3),
        "capture_reconnects": handler.metrics.reconnect_count,
        "capture_last_error": handler.metrics.last_error,
        "frames_processed": scheduler.processed_total,
        "skipped_by_interval": scheduler.skipped_by_interval,
        "queue_drops_total": handler.queue.dropped_total,
        "queue_skipped_to_latest_total": handler.queue.skipped_total,
        "actions": dict(actions),
        "reasons": dict(reasons),
        "refresh_worker": dispatcher.snapshot(),
        "pose_depth": None if pose_depth is None or "result" not in latest_pose_depth else {
            "last_frame_id": latest_pose_depth["result"].frame_id,
            "last_person_m": (None if math.isnan(latest_pose_depth["result"].body_depth)
                              else round(latest_pose_depth["result"].body_depth, 2)),
            "last_timings_ms": {k: round(v, 1) for k, v in latest_pose_depth["result"].timings_ms.items()},
        },
    }}, indent=2), flush=True)
    if handler.metrics.emitted_frames == 0:
        parser.exit(2, "No frames captured. Check the device index, camera access, OBS Virtual Camera, or try --camera-backend dshow.\n")


if __name__ == "__main__":
    main()
