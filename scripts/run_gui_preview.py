"""Serve the Stream Handler's raw camera preview to the local Vite GUI.

Run from the repository root, then start the GUI with ``npm run dev``.
This server does not run optical flow, depth, or splat reconstruction.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "modules/stream_handler/src"), str(ROOT / "applications")]

from backend.preview_server import PreviewServer  # noqa: E402
from stream_handler import (  # noqa: E402
    CameraSource, OBSVirtualCameraSource, RecordedVideoSource, ReplayMode,
    RTSPSource, StreamHandler,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("camera", "obs", "video", "rtsp"), required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--camera-backend", choices=("auto", "dshow", "msmf"), default="auto")
    parser.add_argument("--path", type=Path, help="Recorded video for --source video")
    parser.add_argument("--url", help="RTSP URL; avoid credentials in shell history")
    args = parser.parse_args()

    if args.source == "camera":
        source = CameraSource(args.device, backend=args.camera_backend)
    elif args.source == "obs":
        source = OBSVirtualCameraSource(args.device, backend=args.camera_backend)
    elif args.source == "video":
        if args.path is None or not args.path.is_file():
            parser.error("--source video requires an existing --path")
        source = RecordedVideoSource(args.path, replay_mode=ReplayMode.PACED)
    else:
        if not args.url:
            parser.error("--source rtsp requires --url")
        source = RTSPSource(args.url)

    handler = StreamHandler(source, queue_capacity=3, max_reconnect_attempts=3)
    server = PreviewServer(handler)
    handler.start()
    print("Preview ready at http://127.0.0.1:8765/api/preview", flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        handler.stop()


if __name__ == "__main__":
    main()
