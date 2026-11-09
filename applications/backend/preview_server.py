"""Local HTTP preview of the Stream Handler's latest RGB packet."""

from __future__ import annotations

import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from urllib.parse import urlsplit

import cv2

from stream_handler import FramePacket, StreamHandler


def encode_preview(packet: FramePacket, *, max_width: int = 640) -> dict[str, object]:
    """Make a display-only JPEG while retaining the capture packet's identity."""
    image = packet.rgb
    if packet.width > max_width:
        height = max(1, round(packet.height * max_width / packet.width))
        image = cv2.resize(image, (max_width, height), interpolation=cv2.INTER_AREA)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    encoded, jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
    if not encoded:
        raise RuntimeError("could not encode preview frame")
    return {
        "schema_version": "1.0",
        "frame_id": packet.frame_id,
        "source_id": packet.source_id,
        "capture_timestamp_ns": str(packet.capture_timestamp_ns),
        "timestamp_origin": str(packet.timestamp_origin),
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "jpeg_base64": base64.b64encode(jpeg).decode("ascii"),
    }


class PreviewServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, handler: StreamHandler, port: int = 8765) -> None:
        self.stream_handler = handler
        super().__init__(("127.0.0.1", port), PreviewRequestHandler)


class PreviewRequestHandler(BaseHTTPRequestHandler):
    server: PreviewServer

    def do_GET(self) -> None:
        if urlsplit(self.path).path != "/api/preview":
            self.send_error(404)
            return
        packet = self.server.stream_handler.latest_packet
        if packet is None:
            self.send_response(204)
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Stream-Health", str(self.server.stream_handler.health()))
            self.end_headers()
            return
        try:
            payload = encode_preview(packet)
        except (cv2.error, RuntimeError):
            self.send_error(500, "Could not encode preview frame")
            return
        payload["stream_health"] = str(self.server.stream_handler.health())
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, format: str, *args: object) -> None:
        # The browser requests several previews each second; avoid noisy logs.
        pass
