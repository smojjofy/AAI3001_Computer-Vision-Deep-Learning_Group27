"""Integrated HTTP server: preview frames, pose/depth, splat state, diagnostics.

One server serves the built GUI and the local API:

- ``/api/preview``     latest captured RGB frame as JPEG (identity preserved)
- ``/api/pose-depth``  latest refresh depth + skeleton JPEGs and joint values
- ``/api/mask``        latest refresh mask as PNG
- ``/api/splat``       latest splat state: 4-byte header length, JSON header, arrays
- ``/api/diagnostics`` counters, timings, decision, and cache age
- ``/`` and assets     the built Vite GUI, when ``applications/gui/dist`` exists

The response for ``/api/splat`` is ``application/octet-stream`` so one request
delivers the header and all arrays with no chance of a header/data mismatch.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
import json
from pathlib import Path
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

import cv2
import numpy as np

from backend.preview_server import encode_preview

ROOT = Path(__file__).resolve().parents[2]
GUI_DIST = ROOT / "applications" / "gui" / "dist"

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".svg": "image/svg+xml",
    ".wasm": "application/wasm",
    ".ico": "image/x-icon",
}


@dataclass
class PipelineState:
    """Thread-safe holder for the newest publishable results."""

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    splat_header: dict[str, Any] | None = None
    splat_buffer: bytes = b""
    pose_depth: dict[str, Any] | None = None
    mask_png: bytes | None = None
    motion: dict[str, Any] | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    last_publish_wall_ns: int = 0

    def publish_splat(self, header: dict[str, Any], buffer: bytes) -> None:
        with self._lock:
            self.splat_header = header
            self.splat_buffer = buffer
            self.last_publish_wall_ns = time.monotonic_ns()

    def publish_motion(self, payload: dict[str, Any] | None) -> None:
        with self._lock:
            self.motion = payload

    def get_motion(self) -> dict[str, Any] | None:
        with self._lock:
            return self.motion

    def publish_pose_depth(self, payload: dict[str, Any] | None) -> None:
        with self._lock:
            self.pose_depth = payload

    def publish_mask(self, mask: np.ndarray | None) -> None:
        with self._lock:
            if mask is None:
                self.mask_png = None
                return
            image = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
            ok, encoded = cv2.imencode(".png", image)
            self.mask_png = encoded.tobytes() if ok else None

    def publish_diagnostics(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self.diagnostics = payload

    def splat_response(self) -> bytes | None:
        with self._lock:
            if self.splat_header is None:
                return None
            header = dict(self.splat_header)
            age_ns = time.monotonic_ns() - self.last_publish_wall_ns
            header["state_age_ms"] = round(age_ns / 1_000_000.0, 1)
            # Hold the state for 500 ms, then fade it over 500 ms.
            hold_ms = 500.0
            fade_ms = 500.0
            age_ms = age_ns / 1_000_000.0
            if age_ms <= hold_ms:
                header["display_alpha"] = 1.0
            else:
                header["display_alpha"] = round(max(0.0, 1.0 - (age_ms - hold_ms) / fade_ms), 3)
            encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
            return struct.pack("<I", len(encoded)) + encoded + self.splat_buffer

    def get_pose_depth(self) -> dict[str, Any] | None:
        with self._lock:
            return self.pose_depth

    def get_mask_png(self) -> bytes | None:
        with self._lock:
            return self.mask_png

    def get_diagnostics(self) -> dict[str, Any]:
        with self._lock:
            return dict(self.diagnostics)


class PipelineServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        handler,
        state: PipelineState,
        port: int = 8765,
        gui_dist: Path | None = None,
    ) -> None:
        self.stream_handler = handler
        self.state = state
        self.gui_dist = gui_dist if gui_dist is not None else GUI_DIST
        super().__init__(("127.0.0.1", port), PipelineRequestHandler)


class PipelineRequestHandler(BaseHTTPRequestHandler):
    server: PipelineServer

    def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        path = urlsplit(self.path).path
        if path == "/api/preview":
            self._send_preview()
        elif path == "/api/pose-depth":
            self._send_json_or_204(self.server.state.get_pose_depth())
        elif path == "/api/mask":
            self._send_mask()
        elif path == "/api/motion":
            self._send_json_or_204(self.server.state.get_motion())
        elif path == "/api/splat":
            self._send_splat()
        elif path == "/api/diagnostics":
            self._send_json(self.server.state.get_diagnostics())
        elif path == "/api/health":
            self._send_json({"status": "ok", "stream_health": str(self.server.stream_handler.health())})
        else:
            self._serve_static(path)

    def _send_preview(self) -> None:
        packet = self.server.stream_handler.latest_packet
        if packet is None:
            self._send_no_content()
            return
        try:
            payload = encode_preview(packet)
        except (cv2.error, RuntimeError):
            self.send_error(500, "Could not encode preview frame")
            return
        payload["stream_health"] = str(self.server.stream_handler.health())
        self._send_json(payload)

    def _send_mask(self) -> None:
        payload = self.server.state.get_mask_png()
        if payload is None:
            self._send_no_content()
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_splat(self) -> None:
        payload = self.server.state.splat_response()
        if payload is None:
            self._send_no_content()
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_json_or_204(self, payload: dict[str, Any] | None) -> None:
        if payload is None:
            self._send_no_content()
        else:
            self._send_json(payload)

    def _serve_static(self, path: str) -> None:
        root = self.server.gui_dist
        relative = path.lstrip("/") or "index.html"
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError:
            self.send_error(404)
            return
        if not candidate.is_file():
            # Single-page app fallback: unknown non-file routes serve index.html.
            candidate = root / "index.html"
            if not candidate.is_file():
                self.send_error(404, "GUI is not built; run npm run build in applications/gui")
                return
        body = candidate.read_bytes()
        content_type = _CONTENT_TYPES.get(candidate.suffix.lower(), "application/octet-stream")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_no_content(self) -> None:
        self.send_response(204)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Stream-Health", str(self.server.stream_handler.health()))
        self.end_headers()

    def _send_json(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":"), default=_json_default).encode("utf-8")
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
        pass


def _json_default(value: Any):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return str(value)
