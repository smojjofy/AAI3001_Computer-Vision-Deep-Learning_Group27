"""Run the integrated live splat application on this computer.

    python scripts/run_integration_app.py --source webcam --device 0
    python scripts/run_integration_app.py --source zed
    python scripts/run_integration_app.py --source auto
    python scripts/run_integration_app.py --list-zed

The application captures RGB frames, scores each frame in a 100 ms candidate
window with reference-to-candidate optical flow, selects one candidate, and runs
pose + YOLO depth + a depth-threshold mask + the native Splat Constructor only
for a refresh. The newest complete splat state is published to the browser.

Sources: ``webcam`` is the default. ``zed`` uses a Stereolabs ZED camera on
USB. ``auto`` tries the webcam first and selects ``zed`` only when the ZED SDK
imports and a ZED camera answers. A ZED source passes its depth map and its
calibrated intrinsics into the packet.

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
    # The depth inference package is imported as ``src`` by the live pipeline
    # when it needs the stereo matcher. Adding it once here keeps that import
    # from depending on the order the clip is imported in.
    str(ROOT / "modules/inference/depth"),
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
from live_pipeline.pipeline import (  # noqa: E402
    RefreshError,
    RefreshOutcome,
    RefreshPipeline,
)
from live_pipeline.scheduler import RefreshDispatcher, RefreshJob  # noqa: E402
from live_pipeline.window import WindowSelector  # noqa: E402
from optical_flow import FlowConfig, create_estimator  # noqa: E402
from stream_handler import (  # noqa: E402
    CameraSource,
    OBSVirtualCameraSource,
    RecordedVideoSource,
    ReplayMode,
    StreamHandler,
    ZedCameraSource,
    describe_zed_devices,
    zed_is_available,
)


def resolve_source_choice(requested: str) -> str:
    """Return the source to use, applying the ``auto`` rule.

    ``auto`` prefers the webcam and uses the ZED only when the SDK imports and a
    ZED camera is connected. The ZED SDK probe takes about one second, so it is
    skipped for every explicit choice. The function never raises.
    """
    if requested != "auto":
        return requested
    if zed_is_available():
        print(
            json.dumps({"source": {"requested": "auto", "selected": "zed", "devices": describe_zed_devices()}}),
            flush=True,
        )
        return "zed"
    print(json.dumps({"source": {"requested": "auto", "selected": "webcam"}}), flush=True)
    return "webcam"


def build_source(args):
    if args.source == "video":
        if args.path is None or not args.path.is_file():
            raise SystemExit("--source video requires an existing --path")
        return RecordedVideoSource(args.path, replay_mode=ReplayMode.PACED)
    if args.source == "zed":
        return ZedCameraSource(
            args.zed_camera_id,
            resolution=args.zed_resolution,
            fps=args.zed_fps,
            depth_mode=args.zed_depth_mode,
            emit_right=bool(getattr(args, "stereo_depth", False)),
        )
    capture_size = None
    if args.capture_width > 0 and args.capture_height > 0:
        capture_size = (args.capture_width, args.capture_height)
    elif args.capture_width > 0 or args.capture_height > 0:
        raise SystemExit("--capture-width and --capture-height must be given together")
    if args.source == "obs":
        return OBSVirtualCameraSource(
            args.device, backend=args.camera_backend, capture_size=capture_size
        )
    return CameraSource(args.device, backend=args.camera_backend, capture_size=capture_size)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source",
        choices=("auto", "webcam", "camera", "obs", "zed", "video"),
        default="webcam",
        help=(
            "Capture source. 'webcam' is the default. 'auto' uses the webcam and "
            "selects 'zed' only when the ZED SDK and a ZED camera are available."
        ),
    )
    parser.add_argument("--path", type=Path, help="Recorded video for --source video")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--camera-backend", choices=("auto", "dshow", "msmf"), default="dshow")
    parser.add_argument(
        "--zed-camera-id",
        type=int,
        default=0,
        help="ZED camera index for --source zed. Run with --list-zed to see the connected devices.",
    )
    parser.add_argument(
        "--zed-resolution",
        choices=("HD720", "HD1080", "HD2K", "VGA"),
        default="HD720",
        help="ZED resolution for --source zed. HD720 matches the recorded replay clips.",
    )
    parser.add_argument("--zed-fps", type=int, default=30, help="ZED capture rate for --source zed")
    parser.add_argument(
        "--zed-depth-mode",
        choices=("NEURAL", "NEURAL_PLUS", "ULTRA", "QUALITY", "PERFORMANCE", "NONE"),
        default="QUALITY",
        help=(
            "ZED depth mode for --source zed. QUALITY opens in about 2 s. NONE "
            "captures RGB only and leaves depth to the selected-frame YOLO model, "
            "which is the fair comparison for the depth-latency benchmark. NEURAL "
            "and NEURAL_PLUS build a TensorRT engine on first use; that build can "
            "run for many minutes and it happens before the first frame."
        ),
    )
    parser.add_argument(
        "--stereo-depth",
        action="store_true",
        help=(
            "EXPERIMENTAL. Match the rectified stereo pair with StereoSGBM on each "
            "selected refresh frame instead of using YOLO depth. Needs --source zed "
            "with --zed-depth-mode NONE. The published depth source becomes "
            "stereo_sgbm; a frame whose match is unusable falls back to the "
            "whole-frame YOLO estimate and the reason is recorded. Off by default."
        ),
    )
    parser.add_argument(
        "--stereo-half-resolution",
        action="store_true",
        help="EXPERIMENTAL. Match the stereo pair at half resolution. Needs --stereo-depth.",
    )
    parser.add_argument(
        "--list-zed",
        action="store_true",
        help="Print the connected ZED cameras and exit without starting the pipeline.",
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--window-ms", type=float, default=100.0)
    parser.add_argument("--threshold", type=float, default=0.02, help="Fixed normalized flow refresh threshold")
    parser.add_argument("--active-age-ms", type=float, default=500.0, help="Forced refresh age of the active state")
    parser.add_argument("--queue-capacity", type=int, default=3)
    parser.add_argument("--sample-stride", type=int, default=2, help="Constructor pixel stride")
    parser.add_argument("--horizontal-fov-deg", type=float, default=60.0)
    parser.add_argument(
        "--capture-width",
        type=int,
        default=0,
        help="Requested camera capture width. 0 keeps the device default (640 for this camera)",
    )
    parser.add_argument(
        "--capture-height",
        type=int,
        default=0,
        help="Requested camera capture height. 0 keeps the device default (480 for this camera)",
    )
    parser.add_argument(
        "--yolo-imgsz",
        type=int,
        default=640,
        help="YOLO depth input size. 640 matches the default; larger is slower",
    )
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
    # Shadow mask test. A shadow run computes the research mask candidate for
    # each refresh and records it. It never changes the published splat.
    parser.add_argument(
        "--shadow-mask-checkpoint",
        type=Path,
        default=None,
        help="Research mask checkpoint for the shadow run. The published splat does not change.",
    )
    parser.add_argument(
        "--shadow-mask-log",
        type=Path,
        default=None,
        help="JSONL file for shadow mask records. One line for each poll.",
    )
    parser.add_argument(
        "--shadow-mask-sha256",
        default=None,
        help="Expected SHA-256 of the shadow checkpoint. The run stops on a mismatch.",
    )
    parser.add_argument(
        "--shadow-mask-device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Device for the shadow mask model only.",
    )
    parser.add_argument(
        "--shadow-mask-session",
        default=None,
        help="Session label written to each shadow record. Default is the log file stem.",
    )
    parser.add_argument(
        "--shadow-mask-interval-s",
        type=float,
        default=2.0,
        help="Minimum seconds between shadow polls. The shadow run is not on the refresh path.",
    )
    # Face depth correction. Off by default.
    parser.add_argument(
        "--face-depth-correction",
        action="store_true",
        help="Apply the bounded face depth correction after mask selection. Off by default.",
    )
    parser.add_argument(
        "--face-depth-rule-json",
        type=Path,
        default=None,
        help="Parameter override for the face depth correction. Needs --face-depth-correction.",
    )
    parser.add_argument(
        "--face-depth-log",
        type=Path,
        default=None,
        help="JSONL file with one record for each corrected refresh.",
    )
    args = parser.parse_args()
    if args.list_zed:
        devices = describe_zed_devices()
        if devices:
            print(json.dumps({"zed_devices": devices}, indent=2), flush=True)
        else:
            print(
                json.dumps(
                    {
                        "zed_devices": [],
                        "reason": (
                            "no ZED camera was found, or the ZED SDK (pyzed) is not "
                            "installed for this Python"
                        ),
                    },
                    indent=2,
                ),
                flush=True,
            )
        return 0
    # 'auto' prefers the webcam and selects the ZED only when it is available.
    args.source = resolve_source_choice(args.source)
    if args.face_depth_correction and args.face_depth_rule_json is None:
        raise SystemExit("--face-depth-correction needs --face-depth-rule-json")
    if (not args.face_depth_correction) and (
        args.face_depth_rule_json is not None or args.face_depth_log is not None
    ):
        raise SystemExit("--face-depth-rule-json and --face-depth-log need --face-depth-correction")
    if (args.shadow_mask_checkpoint is None) != (args.shadow_mask_log is None):
        raise SystemExit("--shadow-mask-checkpoint and --shadow-mask-log must be given together")

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
        yolo_imgsz=args.yolo_imgsz,
        camera_config=camera_config,
        yolo_model=args.yolo_model,
        device=args.device_inference,
        require_pose=not args.allow_missing_pose,
        generation=run_guard.generation,
        stereo_enabled=bool(getattr(args, "stereo_depth", False)),
    )
    if args.stereo_depth:
        # The experimental stereo path needs a pair. Refuse a configuration that
        # cannot produce one instead of silently publishing YOLO depth under a
        # stereo run.
        if args.source != "zed":
            raise SystemExit("--stereo-depth needs --source zed")
        if args.zed_depth_mode != "NONE":
            raise SystemExit(
                "--stereo-depth needs --zed-depth-mode NONE: the SDK depth path "
                "would already spend the time the matcher is meant to replace"
            )
        from src.pipeline import StereoFallbackConfig
        from src.stereo_depth import StereoMatcherConfig

        pipeline.stereo_config = StereoMatcherConfig(
            resolution_divisor=2 if args.stereo_half_resolution else 1
        )
        pipeline.stereo_fallback = StereoFallbackConfig()
    # Face depth correction. Off by default, so a normal run is unchanged.
    face_depth_rule = None
    face_depth_log = None
    if args.face_depth_correction:
        from scripts.face_depth_test import load_rule

        try:
            face_depth_rule = load_rule(args.face_depth_rule_json)
        except (OSError, ValueError) as error:
            raise SystemExit(f"face depth rule {args.face_depth_rule_json} is invalid: {error}")
        pipeline.face_depth_rule = face_depth_rule
        face_depth_log = open(args.face_depth_log, "a", encoding="utf-8", newline="\n")
        counters_face_depth = Counter()
    else:
        counters_face_depth = None

    # Shadow mask run. It runs in a child process and only writes records.
    shadow_runner = None
    shadow_pipeline = None
    shadow_final: dict = {}
    if args.shadow_mask_checkpoint is not None:
        from modules.inference.paired_research.shadow import ShadowProcessWriter

        try:
            shadow_pipeline = ShadowProcessWriter(
                checkpoint=args.shadow_mask_checkpoint,
                log_path=args.shadow_mask_log,
                expected_sha256=args.shadow_mask_sha256,
                device=args.shadow_mask_device,
                session_id=args.shadow_mask_session or args.shadow_mask_log.stem,
            )
        except (OSError, ValueError) as error:
            raise SystemExit(f"shadow mask checkpoint is rejected: {error}")
        print(
            json.dumps(
                {
                    "shadow_mask": {
                        "checkpoint": str(args.shadow_mask_checkpoint),
                        "sha256": args.shadow_mask_sha256,
                        "device": args.shadow_mask_device,
                        "log": str(args.shadow_mask_log),
                        "process": "child",
                        "published_splat_unchanged": True,
                    }
                }
            ),
            flush=True,
        )
    # A list holds the mutable value, so the refresh worker's nested helper can
    # read and write it without a nonlocal binding to main().
    shadow_last_poll = [0.0]
    estimator = create_estimator(FlowConfig())
    selector = WindowSelector(
        estimator,
        threshold=args.threshold,
        active_age_ns=round(args.active_age_ms * 1_000_000),
        generation=run_guard.generation,
    )

    counters: Counter[str] = Counter()
    # One record for each failed refresh, grouped by class in the summary.
    failure_log: list[dict[str, object]] = []
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
    latest: dict[str, object] = {"decision": None, "outcome_header": None, "stereo_info": None, "last_error": None}
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
            latest["stereo_info"] = None
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
            stereo_info = latest["stereo_info"]
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
                    "stereo": stereo_info,
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
            depth_m, person_detected, depth_source = pipeline.capture_background_depth(
                job.packet, generation=job_generation
            )
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
            calibration.add(depth_m, depth_source=depth_source)
            state.record_calibration_frame(calibration.count)
            if elapsed < seconds and calibration.count < 20:
                publish_diagnostics()
                return
            if calibration.count < 3:
                raise CalibrationError("too few calibration frames were captured; retry")
            baseline = calibration.build(
                source_id=job.packet.source_id,
                generation=job_generation,
                depth_source=depth_source,
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
            if face_depth_log is not None:
                # The correction runs inside the refresh and changes
                # outcome.depth_m and outcome.mask.validity. The record keeps the
                # invariants so the log can prove what the rule did.
                counters_face_depth["refreshes"] += 1
                counters_face_depth["corrected"] += int(bool(outcome.face_depth_info.get("applied")))
                face_depth_log.write(
                    json.dumps(
                        {
                            "schema_version": "1",
                            "kind": "live_face_depth",
                            "session_id": args.face_depth_log.stem,
                            "generation": job_generation,
                            "frame_id": int(outcome.frame_id),
                            "source_id": outcome.source_id,
                            "capture_timestamp_ns": str(outcome.capture_timestamp_ns),
                            "mask_source": outcome.header.get("mask", {}).get("source"),
                            "face_depth": outcome.face_depth_info,
                            "timings_ms": {
                                key: round(value, 3) for key, value in outcome.face_depth_timings_ms.items()
                            },
                        }
                    )
                    + "\n"
                )
                face_depth_log.flush()
        except RefreshError as error:
            def record_failure() -> None:
                counters["refresh_failed"] += 1
                run_counters.record_failure(job_generation)
                # Group every failure by class, not only the last message. RT-0087
                # needs a count for each cause.
                reason = getattr(error, "reason", "other")
                counters[f"refresh_failed_{reason}"] += 1
                with lock:
                    failure_log.append(
                        {
                            "frame_id": int(job.packet.frame_id),
                            "generation": int(job_generation),
                            "reason": str(reason),
                            "person_missing": bool(getattr(error, "person_missing", False)),
                            "message": str(error)[:300],
                        }
                    )
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
        def run_shadow_once(shadow_outcome, shadow_packet, shadow_generation: int) -> None:
            try:
                shadow_pipeline.submit(shadow_packet, shadow_outcome, generation=shadow_generation)
            except Exception as error:  # a shadow fault must not stop the run
                counters["shadow_write_failed"] += 1
                print(json.dumps({"shadow_error": f"{type(error).__name__}: {error}"}), flush=True)

        if shadow_pipeline is not None:
            # The shadow run must never delay the refresh or touch the published
            # state. submit() hands the frame to a child process, which holds its
            # own interpreter lock, and returns at once.
            if time.monotonic() - shadow_last_poll[0] >= max(0.0, args.shadow_mask_interval_s):
                shadow_last_poll[0] = time.monotonic()
                run_shadow_once(outcome, job.packet, job_generation)
                counters["shadow_submitted"] += 1

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
                latest["stereo_info"] = outcome.stereo_info
                latest["last_error"] = None
            counters["refresh_published"] += 1
            if outcome.stereo_info is not None:
                stereo_status = str(outcome.stereo_info.get("status", "unknown"))
                counters[f"stereo_{stereo_status}"] += 1
                if stereo_status == "fallback":
                    reason = str(outcome.stereo_info.get("fallback_reason", "unknown"))
                    counters[f"stereo_fallback_{reason}"] += 1
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
                        "mask_source": outcome.header.get("mask", {}).get("source"),
                        "bg_area": outcome.header.get("mask", {}).get("background_area_fraction"),
                        "stereo": outcome.stereo_info,
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
        # A shutdown step must never prevent the summary. The ZED SDK close path
        # can raise from native code, and losing the whole run record because of
        # a teardown error hides the measurement the run was made for.
        shutdown_errors: list[str] = []

        def try_shutdown(name: str, action) -> None:
            try:
                action()
            except BaseException as error:  # noqa: BLE001 - shutdown is best effort
                shutdown_errors.append(f"{name}: {type(error).__name__}: {error}")
                print(json.dumps({"shutdown_error": shutdown_errors[-1]}), flush=True)

        try_shutdown("handler.stop", handler.stop)
        try_shutdown("flush_window", flush_window)
        # Shutdown must not wait for in-flight model work. A slow refresh would
        # otherwise hang the process for the full dispatcher timeout, which is
        # what a finite video source shows when it reaches its end.
        def close_dispatcher() -> None:
            try:
                dispatcher.close(drain=False, timeout_s=2.0)
            except TimeoutError:
                print("warning: refresh worker did not stop in time; exiting anyway", flush=True)

        try_shutdown("dispatcher.close", close_dispatcher)
        try_shutdown("server.shutdown", server.shutdown)
        try_shutdown("server.server_close", server.server_close)
        try_shutdown("pipeline.close", pipeline.close)
        if shadow_pipeline is not None:
            try:
                shadow_final = shadow_pipeline.close()
            except BaseException as error:  # noqa: BLE001 - shutdown is best effort
                shutdown_errors.append(f"shadow_pipeline.close: {type(error).__name__}: {error}")
                print(json.dumps({"shutdown_error": shutdown_errors[-1]}), flush=True)
        if face_depth_log is not None:
            try_shutdown("face_depth_log.close", face_depth_log.close)

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
        # RT-0087: every failure grouped by class, with a small sample of each.
        "refresh_failure_classes": {
            reason: int(count)
            for key, count in sorted(counters.items())
            for reason in [key.removeprefix("refresh_failed_")]
            if key.startswith("refresh_failed_") and key != "refresh_failed"
        },
        "refresh_failure_samples": failure_log[:40],
        "stereo_decisions": {
            "used": counters["stereo_used"],
            "fallback": counters["stereo_fallback"],
            "fallback_reasons": {
                key.removeprefix("stereo_fallback_"): int(count)
                for key, count in sorted(counters.items())
                if key.startswith("stereo_fallback_")
            },
        },
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
    if shadow_pipeline is not None:
        summary["shadow_mask"] = {
            "log": str(args.shadow_mask_log),
            "process": "child",
            "published_splat_unchanged": True,
            "submitted": int(counters["shadow_submitted"]),
            **shadow_final,
        }
    if counters_face_depth is not None:
        summary["face_depth_correction"] = {
            "log": str(args.face_depth_log),
            **dict(counters_face_depth),
        }
    print(json.dumps({"summary": summary}, indent=2), flush=True)
    if args.summary_json is not None:
        args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
