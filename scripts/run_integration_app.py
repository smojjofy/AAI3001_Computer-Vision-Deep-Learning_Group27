"""Run the integrated live splat application on this computer.

    python scripts/run_integration_app.py --source webcam --device 0

The application captures RGB frames, scores each frame in a 100 ms candidate
window with reference-to-candidate optical flow, selects one candidate, and runs
pose + YOLO depth + a depth-threshold mask + the native Splat Constructor only
for a refresh. The newest complete splat state is published to the browser.

Endpoints (default port 8765):
    /api/preview      latest captured frame (JPEG)
    /api/pose-depth   depth map + skeleton for the latest refresh
    /api/mask         depth-threshold mask (PNG)
    /api/splat        splat state (framed JSON header + binary arrays)
    /api/diagnostics  counters, timings, decision, cache age
"""

from __future__ import annotations

import argparse
from collections import Counter
import cv2
import json
import math
import numpy as np
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "applications"),
    str(ROOT / "modules/stream_handler/src"),
    str(ROOT / "modules/inference/optical_flow/src"),
    str(ROOT / "modules/reranker/src"),
]

from backend.pipeline_server import PipelineServer, PipelineState  # noqa: E402
from live_pipeline.adapters import DepthThresholdConfig  # noqa: E402
from live_pipeline.pipeline import RefreshError, RefreshPipeline  # noqa: E402
from live_pipeline.scheduler import RefreshDispatcher, RefreshJob  # noqa: E402
from live_pipeline.window import WindowSelector  # noqa: E402
from optical_flow import FlowConfig, create_estimator  # noqa: E402
from stream_handler import (  # noqa: E402
    CameraSource,
    OBSVirtualCameraSource,
    RecordedVideoSource,
    ReplayMode,
    StreamHandler,
)


def build_source(args):
    if args.source == "video":
        if args.path is None or not args.path.is_file():
            raise SystemExit("--source video requires an existing --path")
        return RecordedVideoSource(args.path, replay_mode=ReplayMode.PACED)
    if args.source == "obs":
        return OBSVirtualCameraSource(args.device, backend=args.camera_backend)
    return CameraSource(args.device, backend=args.camera_backend)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=("webcam", "camera", "obs", "video"), default="webcam")
    parser.add_argument("--path", type=Path, help="Recorded video for --source video")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--camera-backend", choices=("auto", "dshow", "msmf"), default="dshow")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--window-ms", type=float, default=100.0)
    parser.add_argument("--threshold", type=float, default=0.02, help="Fixed normalized flow refresh threshold")
    parser.add_argument("--active-age-ms", type=float, default=500.0, help="Forced refresh age of the active state")
    parser.add_argument("--queue-capacity", type=int, default=3)
    parser.add_argument("--sample-stride", type=int, default=2, help="Constructor pixel stride")
    parser.add_argument("--horizontal-fov-deg", type=float, default=60.0)
    parser.add_argument("--depth-band-m", type=float, default=0.75)
    parser.add_argument("--depth-near-percentile", type=float, default=2.0)
    parser.add_argument("--yolo-model", default=None, help="YOLO depth weights name or path")
    parser.add_argument("--device-inference", default="auto", help="Torch device for YOLO depth")
    parser.add_argument("--duration-s", type=float, default=0.0, help="0 runs until Ctrl+C")
    parser.add_argument("--summary-json", type=Path, default=None, help="Write the final summary here")
    args = parser.parse_args()

    state = PipelineState()
    pipeline = RefreshPipeline(
        mask_config=DepthThresholdConfig(
            band_m=args.depth_band_m,
            near_percentile=args.depth_near_percentile,
        ),
        sample_stride=args.sample_stride,
        horizontal_fov_deg=args.horizontal_fov_deg,
        yolo_model=args.yolo_model,
        device=args.device_inference,
    )
    estimator = create_estimator(FlowConfig())
    selector = WindowSelector(
        estimator,
        threshold=args.threshold,
        active_age_ns=round(args.active_age_ms * 1_000_000),
    )

    counters: Counter[str] = Counter()
    latest: dict[str, object] = {"decision": None, "outcome_header": None, "last_error": None}
    lock = threading.Lock()

    def publish_diagnostics() -> None:
        with lock:
            decision = latest["decision"]
            header = latest["outcome_header"]
            last_error = latest["last_error"]
        active_capture = selector.active_state_capture_ns
        last_capture = latest.get("last_capture_ns")
        cache_age_ms = None
        if active_capture is not None and isinstance(last_capture, int):
            cache_age_ms = round(max(0, last_capture - active_capture) / 1_000_000.0, 1)
        state.publish_diagnostics(
            {
                "schema_version": "1.0",
                "action": None if decision is None else decision.action,
                "reason": None if decision is None else decision.reason,
                "selected_frame_id": None if decision is None else decision.selected.frame_id,
                "flow_score": None if decision is None or decision.score is None else round(decision.score, 5),
                "flow_threshold": selector.threshold,
                "candidates": None if decision is None else decision.candidates,
                "cache_age_ms": cache_age_ms,
                "active_state_frame_id": selector.active_state_frame_id,
                "refresh_count": pipeline.refresh_count,
                "refresh_failures": pipeline.failure_count,
                "last_error": last_error,
                "splat": None if header is None else {
                    "frame_id": header.get("frame_id"),
                    "count": header.get("count"),
                    "timings_ms": header.get("timings_ms"),
                    "mask": header.get("mask"),
                    "depth_source": header.get("depth_source"),
                    "pose_confidence": header.get("pose_confidence"),
                    "person_detected": header.get("person_detected"),
                },
                "counters": dict(counters),
                "stream_health": str(handler.health()),
                "queue": {
                    "depth": handler.queue.depth,
                    "dropped_total": handler.queue.dropped_total,
                    "skipped_total": handler.queue.skipped_total,
                },
            }
        )

    def refresh_worker(job) -> None:
        try:
            outcome = pipeline.process(job.packet)
        except RefreshError as error:
            counters["refresh_failed"] += 1
            with lock:
                latest["last_error"] = str(error)
            raise
        state.publish_splat(outcome.header, outcome.buffer)
        if outcome.pose_preview is not None:
            state.publish_pose_depth(outcome.pose_preview)
        state.publish_mask(outcome.mask.mask)
        with lock:
            latest["outcome_header"] = outcome.header
            latest["last_error"] = None
        counters["refresh_published"] += 1
        print(
            json.dumps(
                {
                    "refresh": outcome.frame_id,
                    "splats": outcome.splat_count,
                    "ms": {key: round(value, 1) for key, value in outcome.timings_ms.items()},
                    "mask_area": round(float(outcome.mask.area_fraction), 3),
                }
            ),
            flush=True,
        )

    dispatcher = RefreshDispatcher(refresh_worker, on_success=lambda job: selector.commit(job.packet))
    handler = StreamHandler(build_source(args), queue_capacity=args.queue_capacity, max_reconnect_attempts=3)
    server = PipelineServer(handler, state, port=args.port)
    server_thread = threading.Thread(target=server.serve_forever, name="pipeline-server", daemon=True)

    window_ns = round(args.window_ms * 1_000_000)
    window: list = []
    window_start_capture: int | None = None
    window_started_wall: float | None = None

    def flush_window() -> None:
        nonlocal window, window_start_capture, window_started_wall
        if not window:
            return
        decision = selector.select(window)
        counters["windows"] += 1
        with lock:
            latest["decision"] = decision
        if decision.action == "refresh":
            counters["refresh_requested"] += 1
            dispatcher.submit(RefreshJob(decision.selected.packet, decision.selected.flow, decision))
        else:
            counters["reuse"] += 1
        window = []
        window_start_capture = None
        window_started_wall = None
        publish_diagnostics()

    deadline = math.inf if args.duration_s == 0 else time.monotonic() + args.duration_s
    started = time.monotonic()
    handler.start()
    server_thread.start()
    print(f"Integrated app: http://127.0.0.1:{args.port}/  (Ctrl+C stops)", flush=True)
    try:
        while time.monotonic() < deadline:
            packet = handler.queue.get(timeout_s=0.02)
            if packet is not None:
                counters["frames"] += 1
                latest["last_capture_ns"] = int(packet.capture_timestamp_ns)
                candidate = selector.observe(packet)
                window.append(candidate)
                if window_start_capture is None:
                    window_start_capture = int(packet.capture_timestamp_ns)
                    window_started_wall = time.monotonic()
                if int(packet.capture_timestamp_ns) - window_start_capture >= window_ns:
                    flush_window()
            else:
                if window and window_started_wall is not None and time.monotonic() - window_started_wall > 0.25:
                    flush_window()
    except KeyboardInterrupt:
        pass
    finally:
        elapsed = time.monotonic() - started
        handler.stop()
        flush_window()
        dispatcher.close(drain=True)
        server.shutdown()
        server.server_close()
        pipeline.close()

    summary = {
        "elapsed_s": round(elapsed, 2),
        "frames": counters["frames"],
        "windows": counters["windows"],
        "refresh_requested": counters["refresh_requested"],
        "refresh_published": counters["refresh_published"],
        "refresh_failed": counters["refresh_failed"],
        "reuse": counters["reuse"],
        "windows_per_s": round(counters["windows"] / elapsed, 2) if elapsed > 0 else None,
        "capture_fps": round(handler.metrics.capture_fps, 2),
        "queue_drops": handler.queue.dropped_total,
        "last_error": latest["last_error"],
    }
    print(json.dumps({"summary": summary}, indent=2), flush=True)
    if args.summary_json is not None:
        args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
