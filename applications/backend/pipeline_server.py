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


@dataclass(frozen=True)
class ActiveStateIdentity:
    """Identity of the displayed splat state and the motion allowed to move it."""

    source_id: str
    generation: int
    frame_id: int
    reference_frame_id: int | None

    def same_run(self, source_id: str, generation: int) -> bool:
        return self.source_id == source_id and self.generation == generation

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "generation": int(self.generation),
            "frame_id": int(self.frame_id),
            "reference_frame_id": None if self.reference_frame_id is None else int(self.reference_frame_id),
        }


@dataclass
class PipelineState:
    """Thread-safe holder for the newest publishable results.

    The active state is one splat header and one binary buffer. The browser
    keeps the newest published state only; earlier Gaussian states are not
    merged or interpolated. ``person_lost_mono_ns`` latches the first
    invalid-person refresh attempt and clears on a successful publish, so
    repeated failures do not restart the hold timer.
    """

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    splat_header: dict[str, Any] | None = None
    splat_buffer: bytes = b""
    pose_depth: dict[str, Any] | None = None
    mask_png: bytes | None = None
    motion: dict[str, Any] | None = None
    motion_published_total: int = 0
    motion_rejected_total: int = 0
    diagnostics: dict[str, Any] = field(default_factory=dict)
    last_publish_wall_ns: int = 0
    active_identity: ActiveStateIdentity | None = None
    pose_joints: list[dict[str, Any]] | None = None
    person_lost_mono_ns: int | None = None
    last_invalid_person_mono_ns: int | None = None

    def publish_snapshot(
        self,
        header: dict[str, Any],
        buffer: bytes,
        *,
        pose_depth: dict[str, Any] | None,
        mask: np.ndarray | None,
        pose_joints: list[dict[str, Any]] | None = None,
    ) -> None:
        """Publish one complete state under one lock.

        The splat, pose-depth image, and mask in this snapshot all come from the
        same refresh, so a poll cannot mix results from two refreshes. A pose
        preview that failed is published as None, never as the previous frame.
        """
        mask_png = None
        if mask is not None:
            image = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
            ok, encoded = cv2.imencode(".png", image)
            mask_png = encoded.tobytes() if ok else None
        with self._lock:
            self.splat_header = header
            self.splat_buffer = buffer
            self.pose_depth = pose_depth
            self.mask_png = mask_png
            self.pose_joints = list(pose_joints) if pose_joints else None
            self.last_publish_wall_ns = time.monotonic_ns()
            display_frame = int(header.get("frame_id_int", header.get("frame_id", 0)) or 0)
            self.active_identity = ActiveStateIdentity(
                source_id=str(header.get("source_id", "")),
                generation=int(header.get("generation", 0) or 0),
                frame_id=display_frame,
                reference_frame_id=display_frame,
            )
            self.person_lost_mono_ns = None
            self.motion = None

    def note_invalid_person(self) -> None:
        """Latch the first invalid-person event; later failures keep the latch."""
        with self._lock:
            now = time.monotonic_ns()
            self.last_invalid_person_mono_ns = now
            if self.person_lost_mono_ns is None:
                self.person_lost_mono_ns = now

    def person_loss_age_ms(self) -> float | None:
        """Milliseconds since the first invalid-person event, or None."""
        with self._lock:
            if self.person_lost_mono_ns is None:
                return None
            return round((time.monotonic_ns() - self.person_lost_mono_ns) / 1_000_000.0, 1)

    def publish_motion(self, payload: dict[str, Any] | None) -> None:
        """Publish reuse motion only when it matches the displayed splat state.

        Counts separate accepted and rejected motion. A reuse decision is not a
        displayed warp; only accepted motion is counted as published.
        """
        with self._lock:
            if payload is None:
                self.motion = None
                return
            identity = self.active_identity
            rejected = (
                identity is None
                or str(payload.get("source_id", "")) != identity.source_id
                or int(payload.get("generation", -1)) != identity.generation
                or payload.get("reference_frame_id") != identity.frame_id
                or payload.get("splat_frame_id") != identity.frame_id
                or int(payload.get("frame_id", 0)) <= identity.frame_id
            )
            if rejected:
                self.motion = None
                self.motion_rejected_total += 1
                return
            self.motion = payload
            self.motion_published_total += 1

    def get_motion(self) -> dict[str, Any] | None:
        with self._lock:
            return self.motion

    def publish_diagnostics(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self.diagnostics = payload

    def get_pose_joints(self) -> list[dict[str, Any]] | None:
        with self._lock:
            return self.pose_joints

    def get_active_identity(self) -> ActiveStateIdentity | None:
        with self._lock:
            return self.active_identity

    def splat_response(self) -> bytes | None:
        with self._lock:
            if self.splat_header is None:
                return None
            header = dict(self.splat_header)
            age_ns = time.monotonic_ns() - self.last_publish_wall_ns
            header["state_age_ms"] = round(age_ns / 1_000_000.0, 1)
            # One display clock owns the 500 ms hold and 500 ms fade: the
            # browser runs it from person_loss_age_ms. The server never forces
            # alpha to zero, because a response during the hold would bypass it.
            header["display_alpha"] = 1.0
            header["person_lost"] = self.person_lost_mono_ns is not None
            if self.active_identity is not None:
                header["active_identity"] = self.active_identity.to_dict()
            encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
            return struct.pack("<I", len(encoded)) + encoded + self.splat_buffer

    def mask_response(self) -> tuple[ActiveStateIdentity | None, bytes] | None:
        """Return the mask and the identity of the state that produced it.

        Both values come from one lock hold, so a snapshot cannot be published
        between them. The browser compares the identity with the state it asked
        for and never displays a mask from a different state.
        """
        with self._lock:
            if self.mask_png is None:
                return None
            return (self.active_identity, self.mask_png)

    def get_pose_depth(self) -> dict[str, Any] | None:
        with self._lock:
            return self.pose_depth

    def get_diagnostics(self) -> dict[str, Any]:
        with self._lock:
            payload = dict(self.diagnostics)
            payload["person_loss_age_ms"] = (
                None
                if self.person_lost_mono_ns is None
                else round((time.monotonic_ns() - self.person_lost_mono_ns) / 1_000_000.0, 1)
            )
            payload["person_lost"] = self.person_lost_mono_ns is not None
            payload["active_identity"] = (
                None if self.active_identity is None else self.active_identity.to_dict()
            )
            payload["motion_active"] = self.motion is not None
            payload["motion_published"] = self.motion_published_total
            payload["motion_rejected"] = self.motion_rejected_total
            return payload


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
        response = self.server.state.mask_response()
        if response is None:
            self._send_no_content()
            return
        identity, payload = response
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Cache-Control", "no-store")
        if identity is not None:
            self.send_header("X-Mask-Source-Id", identity.source_id)
            self.send_header("X-Mask-Generation", str(identity.generation))
            self.send_header("X-Mask-Frame-Id", str(identity.frame_id))
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
