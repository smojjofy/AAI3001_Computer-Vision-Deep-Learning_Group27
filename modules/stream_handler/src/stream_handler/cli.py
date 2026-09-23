"""Diagnostics-only command line entry point for capture setup and smoke tests."""

from __future__ import annotations

import argparse
import logging
import time

from .handler import StreamHandler
from .sources import CameraSource, OBSVirtualCameraSource, RecordedVideoSource, RTSPSource


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect a live-splat frame source without storing imagery.")
    parser.add_argument("--source", choices=("camera", "obs", "video", "rtsp"), default="obs")
    parser.add_argument("--device", type=int, default=0, help="Camera index for --source camera or obs.")
    parser.add_argument("--path", help="Path for --source video.")
    parser.add_argument("--url", help="RTSP URL for --source rtsp; do not put credentials in shell history.")
    parser.add_argument("--duration", type=float, default=15.0, help="Diagnostic run duration in seconds.")
    parser.add_argument("--queue-capacity", type=int, default=3, choices=range(1, 5))
    return parser


def _source_from_args(args: argparse.Namespace):
    if args.source == "obs":
        return OBSVirtualCameraSource(args.device)
    if args.source == "camera":
        return CameraSource(args.device)
    if args.source == "video":
        if not args.path:
            raise SystemExit("--path is required for --source video")
        return RecordedVideoSource(args.path)
    if not args.url:
        raise SystemExit("--url is required for --source rtsp")
    return RTSPSource(args.url)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parser().parse_args()
    handler = StreamHandler(_source_from_args(args), queue_capacity=args.queue_capacity)
    handler.start()
    deadline = time.monotonic() + args.duration
    last_report = 0.0
    try:
        while time.monotonic() < deadline:
            packet = handler.queue.get(timeout_s=0.5)
            now = time.monotonic()
            if packet and now - last_report >= 1.0:
                logging.info(
                    "frame=%s %sx%s state=%s queue=%s dropped=%s timestamp=%s",
                    packet.frame_id, packet.width, packet.height, handler.metrics.state,
                    handler.queue.depth, handler.queue.dropped_total, packet.quality.timestamp_status,
                )
                last_report = now
    except KeyboardInterrupt:
        logging.info("stopped by user")
    finally:
        handler.stop()
        logging.info("final metrics: %s", handler.metrics)


if __name__ == "__main__":
    main()
