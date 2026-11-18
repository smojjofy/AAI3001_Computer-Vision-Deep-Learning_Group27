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
from live_pipeline.motion import build_motion_preview  # noqa: E402
from live_pipeline.runstate import PacketRunTracker, RunCounters, RunGuard, drain_queue  # noqa: E402
from live_pipeline.telemetry import summarize_counts, summarize_ms  # noqa: E402
from live_pipeline.background_calibration import (  # noqa: E402
    BackgroundCalibrator,
    CalibrationError,
)
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
    parser.add_argument(
        "--camera-config",
        type=Path,
        default=None,
        help="Optional pinhole intrinsics JSON for the capture resolution",
    )
    parser.add_argument("--depth-band-m", type=float, default=0.75)
    parser.add_argument("--depth-near-percentile", type=float, default=2.0)
    parser.add_argument("--max-mask-area", type=float, default=0.60, help="Reject masks larger than this image fraction")
    parser.add_argument("--allow-missing-pose", action="store_true", help="Publish a mask-only state when pose is absent")
    parser.add_argument("--yolo-model", default=None, help="YOLO depth weights name or path")
    parser.add_argument("--device-inference", default="auto", help="Torch device for YOLO depth")
    parser.add_argument("--duration-s", type=float, default=0.0, help="0 runs until Ctrl+C")
    parser.add_argument("--summary-json", type=Path, default=None, help="Write the final summary here")
    parser.add_argument(
        "--force-run-change-after-s",
        type=float,
        default=0.0,
        help="Diagnostic: start one new capture run after this many seconds, to test run boundaries. 0 disables it.",
    )
    args = parser.parse_args()

    state = PipelineState()
    # One capture run. A source reopen, resolution change, or backward capture
    # time advances it, so flow, motion, and splat identity never span two runs.
    run_guard = RunGuard(0)
    run_counters = RunCounters(0)
    packet_tracker = PacketRunTracker()
    camera_config = None
    if args.camera_config is not None:
        from live_pipeline.camera_config import load_camera_config

        try:
            camera_config = load_camera_config(args.camera_config)
        except (OSError, ValueError) as error:
            raise SystemExit(f"camera config {args.camera_config} is invalid: {error}")
    pipeline = RefreshPipeline(
        mask_config=DepthThresholdConfig(
            band_m=args.depth_band_m,
            near_percentile=args.depth_near_percentile,
            max_area_fraction=args.max_mask_area,
        ),
        sample_stride=args.sample_stride,
        horizontal_fov_deg=args.horizontal_fov_deg,
        camera_config=camera_config,
        yolo_model=args.yolo_model,
        device=args.device_inference,
        require_pose=not args.allow_missing_pose,
        generation=run_guard.generation,
    )
    estimator = create_estimator(FlowConfig())
    selector = WindowSelector(
        estimator,
        threshold=args.threshold,
        active_age_ns=round(args.active_age_ms * 1_000_000),
        generation=run_guard.generation,
    )

    counters: Counter[str] = Counter()
    # Cascade offsets at the start of the current capture run. The live cascade
    # must describe one run, while the dispatcher and window counters are
    # process lifetime totals.
    run_cascade_base: dict[str, int] = {
        "windows": 0,
        "submitted": 0,
        "started": 0,
        "replaced": 0,
        "reuse": 0,
    }
    # Placeholder until the dispatcher is constructed below; the first run
    # starts at zero for every field, so the base is correct as written.
    dispatcher: object | None = None
    latency_samples: list[float] = []
    capture_age_samples: list[float] = []
    splat_samples: list[int] = []
    latest: dict[str, object] = {"decision": None, "outcome_header": None, "last_error": None}
    lock = threading.Lock()

    def advance_generation(reason: str) -> None:
        """Start a new capture run and drop every old-run selection and job."""
        nonlocal window, window_start_capture, window_started_wall
        generation = run_guard.advance()
        run_counters.start_run(generation)
        if dispatcher is not None:
            # Drop an old-run pending job before the offsets are taken, so it
            # cannot start inside the new run and inflate its cascade.
            dispatcher.discard_pending()
            offsets = dispatcher.run_offsets()
            run_cascade_base.update(
                {
                    "windows": int(counters["windows"]),
                    "submitted": offsets["submitted"],
                    "started": offsets["started"],
                    "replaced": offsets["replaced"],
                    "reuse": int(counters["reuse"]),
                }
            )
        packet_tracker.reset()
        selector.set_generation(generation)
        pipeline.generation = generation
        # The new run has no valid state yet. Drop the old snapshot, identity,
        # loss latch, and motion, and clear the diagnostics that describe them,
        # so a viewer cannot show old-run data beside the new run generation.
        state.clear_active()
        pipeline.set_background_baseline(None)
        with lock:
            latest["outcome_header"] = None
            latest["last_error"] = None
            latest["decision"] = None
        drained = drain_queue(handler.queue)
        if drained:
            counters["packets_discarded_run"] += 1
        window = []
        window_start_capture = None
        window_started_wall = None
        counters["run_generation_changes"] += 1
        counters[f"run_change_{reason}"] += 1
        # Publish boundary diagnostics immediately, so /api/diagnostics cannot
        # serve the previous run's action, splat, or error before the next
        # window publishes.
        publish_diagnostics()
        print(json.dumps({"run_generation": generation, "reason": reason}), flush=True)

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
                "refresh_count": run_counters.refresh_count,
                "refresh_failures": run_counters.refresh_failures,
                "run_generation": run_guard.generation,
                "last_error": last_error,
                "splat": None if header is None else {
                    "frame_id": header.get("frame_id"),
                    "count": header.get("count"),
                    "timings_ms": header.get("timings_ms"),
                    "capture_age_ms": header.get("capture_age_ms"),
                    "mask": header.get("mask"),
                    "depth_source": header.get("depth_source"),
                    "pose_confidence": header.get("pose_confidence"),
                    "person_detected": header.get("person_detected"),
                },
                "counters": dict(counters),
                "cascade": {
                    # Current capture run only: every field uses the run-start
                    # offset, so all fields describe the same run.
                    "scope": "run",
                    "run_generation": run_guard.generation,
                    "selected": int(counters["windows"]) - run_cascade_base["windows"],
                    "submitted": int(dispatcher.submitted_total) - run_cascade_base["submitted"],
                    "started": int(dispatcher.started_total) - run_cascade_base["started"],
                    "replaced": int(dispatcher.replaced_total) - run_cascade_base["replaced"],
                    "published": run_counters.refresh_count,
                    "failed": run_counters.refresh_failures,
                    "reuse": int(counters["reuse"]) - run_cascade_base["reuse"],
                },
                "refresh_worker": dispatcher.snapshot(),
                "stream_health": str(handler.health()),
                "queue": {
                    "depth": handler.queue.depth,
                    "dropped_total": handler.queue.dropped_total,
                    "skipped_total": handler.queue.skipped_total,
                },
            }
        )

    calibration = BackgroundCalibrator()

    def run_calibration_step(job, job_generation: int, request: dict) -> None:
        """Capture one calibration frame; build and freeze the baseline when due."""
        requested_at = request.get("requested_at_s") or time.monotonic()
        seconds = float(request.get("seconds") or 2.0)
        elapsed = time.monotonic() - float(requested_at)
        try:
            depth_m, person_detected = pipeline.capture_background_depth(job.packet, generation=job_generation)
            if person_detected:
                # Never learn a person as background. Stop and report instead of
                # collecting a bad baseline.
                calibration.reset()
                state.update_calibration(
                    active=False,
                    status="failed",
                    frames=0,
                    last_error="person detected during calibration; clear the scene and retry",
                )
                publish_diagnostics()
                return
            calibration.add(depth_m)
            state.record_calibration_frame(calibration.count)
            if elapsed < seconds and calibration.count < 20:
                publish_diagnostics()
                return
            if calibration.count < 3:
                raise CalibrationError("too few calibration frames were captured; retry")
            baseline = calibration.build(
                source_id=job.packet.source_id,
                generation=job_generation,
            )
        except (CalibrationError, RuntimeError) as error:
            calibration.reset()
            state.update_calibration(active=False, status="failed", last_error=str(error))
            publish_diagnostics()
            return
        pipeline.set_background_baseline(baseline)
        calibration.reset()
        state.update_calibration(
            active=False,
            status="calibrated",
            frames=baseline.frames_used,
            baseline=baseline.stats(),
            last_error=None,
        )
        publish_diagnostics()

    def refresh_worker(job) -> None:
        job_generation = int(getattr(job, "generation", run_guard.generation))
        if job_generation != run_guard.generation:
            counters["refresh_discarded_stale"] += 1
            return
        calibration = state.get_calibration()
        if calibration.get("active"):
            run_calibration_step(job, job_generation, calibration)
            return
        try:
            outcome = pipeline.process(job.packet, generation=job_generation)
        except RefreshError as error:
            def record_failure() -> None:
                counters["refresh_failed"] += 1
                run_counters.record_failure(job_generation)
                if getattr(error, "person_missing", False):
                    # CONTEXT.md invalid-person event: hold starts now and
                    # latches until a successful refresh publishes a state.
                    state.note_invalid_person()
                    counters["person_lost_events"] += 1
                with lock:
                    latest["last_error"] = str(error)

            # RT-0026 item 2: a stale failure must not start person loss in the
            # new run or overwrite the current error.
            if not run_guard.record_failure_if_current(job_generation, record_failure):
                counters["refresh_discarded_stale"] += 1
                return
            raise
        # One consistent snapshot: splat, pose-depth image, mask, and the pose
        # joints that belong to this state. A missing preview is published as
        # None so the viewer cannot show an older frame.
        pose_depth = None
        if outcome.pose_preview is not None:
            pose_depth = dict(outcome.pose_preview)
            pose_depth["generation"] = job_generation
            pose_depth["splat_frame_id"] = int(outcome.frame_id)
        def publish() -> None:
            run_counters.record_published(job_generation)
            state.publish_snapshot(
                outcome.header,
                outcome.buffer,
                pose_depth=pose_depth,
                mask=outcome.mask.mask,
                pose_joints=outcome.pose_joints,
            )
            with lock:
                latest["outcome_header"] = outcome.header
                latest["last_error"] = None
            counters["refresh_published"] += 1
            latency_samples.append(float(outcome.timings_ms["total"]))
            if outcome.capture_age_ms is not None:
                capture_age_samples.append(float(outcome.capture_age_ms))
            splat_samples.append(int(outcome.splat_count))
            print(
                json.dumps(
                    {
                        "refresh": outcome.frame_id,
                        "splats": outcome.splat_count,
                        "ms": {key: round(value, 1) for key, value in outcome.timings_ms.items()},
                        "capture_age_ms": None if outcome.capture_age_ms is None else round(outcome.capture_age_ms, 1),
                        "mask_area": round(float(outcome.mask.area_fraction), 3),
                    }
                ),
                flush=True,
            )

        # RT-0026 item 3: validation and publication share one generation
        # boundary. A run change cannot occur between them.
        if not run_guard.publish_if_current(job_generation, publish):
            counters["refresh_discarded_stale"] += 1
            return

    dispatcher = RefreshDispatcher(
        refresh_worker,
        on_success=lambda job: selector.commit(job.packet, generation=int(getattr(job, "generation", -1))),
    )
    handler = StreamHandler(build_source(args), queue_capacity=args.queue_capacity, max_reconnect_attempts=3)
    server = PipelineServer(handler, state, port=args.port)
    server_thread = threading.Thread(target=server.serve_forever, name="pipeline-server", daemon=True)

    window_ns = round(args.window_ms * 1_000_000)
    window: list = []
    window_start_capture: int | None = None
    window_started_wall: float | None = None

    def flush_window() -> None:
        nonlocal window, window_start_capture, window_started_wall, window_generation_value
        if not window:
            return
        if window_generation_value != run_guard.generation:
            # A capture run changed while this window was open. Drop it; the new
            # run starts from a bootstrap window, never from old candidates.
            counters["windows_discarded_generation"] += 1
            window = []
            window_start_capture = None
            window_started_wall = None
            return
        current = [item for item in window if selector.candidate_is_current(item)]
        if not current:
            counters["windows_discarded_generation"] += 1
            window = []
            window_start_capture = None
            window_started_wall = None
            return
        decision = selector.select(current)
        counters["windows"] += 1
        with lock:
            latest["decision"] = decision
        if decision.action == "refresh":
            counters["refresh_requested"] += 1
            dispatcher.submit(
                RefreshJob(decision.selected.packet, decision.selected.flow, decision, generation=window_generation_value)
            )
        else:
            counters["reuse"] += 1
            # Take the identity from the published state, not from a local copy,
            # so a fetch of the served splat and the motion always agree.
            identity = state.get_active_identity()
            if identity is None:
                state.publish_motion(None)
            else:
                # Motion may move only the displayed state, and its pose joints
                # must be the joints published with that state.
                preview = build_motion_preview(
                    decision.selected,
                    state.get_pose_joints(),
                    source_id=identity.source_id,
                    generation=identity.generation,
                    splat_frame_id=identity.reference_frame_id,
                )
                state.publish_motion(None if preview is None else preview.to_payload())
        window = []
        window_start_capture = None
        window_started_wall = None
        publish_diagnostics()

    deadline = math.inf if args.duration_s == 0 else time.monotonic() + args.duration_s
    started = time.monotonic()
    last_reconnect_count = int(handler.metrics.reconnect_count)
    window_generation_value = run_guard.generation
    force_change_after_s = max(0.0, args.force_run_change_after_s)
    force_change_done = force_change_after_s <= 0.0
    handler.start()
    server_thread.start()
    print(f"Integrated app: http://127.0.0.1:{args.port}/  (Ctrl+C stops)", flush=True)
    try:
        while time.monotonic() < deadline:
            if not force_change_done and time.monotonic() - started >= force_change_after_s:
                force_change_done = True
                advance_generation("forced_test")
                counters["forced_run_changes"] += 1
            packet = handler.queue.get(timeout_s=0.02)
            if packet is not None:
                counters["frames"] += 1
                # A new capture run after a source reopen or a resolution change
                # resets the retained reference, the active state identity, and
                # any motion, so no flow or warp spans two runs.
                generation_before = run_guard.generation
                reconnect_now = int(handler.metrics.reconnect_count)
                if reconnect_now != last_reconnect_count:
                    # A live OpenCV source has no source timestamp, so a reopen
                    # is invisible in the packet identity. The handler counter
                    # is the only reliable reopen signal.
                    last_reconnect_count = reconnect_now
                    advance_generation("reconnect")
                else:
                    reason = packet_tracker.change_reason(packet)
                    if reason is not None:
                        advance_generation(f"identity_{reason}")
                if run_guard.generation != generation_before:
                    # This packet may predate the run change (it was already
                    # dequeued). Drop it; the drain removed the rest of the old
                    # queue and the tracker reset makes the next packet the new
                    # baseline, so one permanent change cannot loop.
                    counters["packets_discarded_run"] += 1
                    continue
                packet_tracker.accept(packet)
                latest["last_capture_ns"] = int(packet.capture_timestamp_ns)
                candidate = selector.observe(packet)
                window.append(candidate)
                if window_start_capture is None:
                    window_start_capture = int(packet.capture_timestamp_ns)
                    window_started_wall = time.monotonic()
                    window_generation_value = run_guard.generation
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
        "latency_ms": summarize_ms(latency_samples),
        # The first refresh after model load includes warm-up and is not a
        # steady-state latency sample.
        "latency_warm_ms": summarize_ms(latency_samples[1:]),
        "capture_age_ms": summarize_ms(capture_age_samples),
        "splat_count": summarize_counts(splat_samples),
        "cascade": {
            "selected": counters["windows"],
            "submitted": dispatcher.submitted_total,
            "started": dispatcher.started_total,
            "replaced": dispatcher.replaced_total,
            "published": counters["refresh_published"],
            "failed": counters["refresh_failed"],
            "reuse": counters["reuse"],
        },
    }
    print(json.dumps({"summary": summary}, indent=2), flush=True)
    if args.summary_json is not None:
        args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
