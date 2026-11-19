"""Deterministic selected-frame replay for the first learning data slice.

This tool uses the live window selector and refresh processor. It reads a video
in decode order and has no capture queue or browser. Its timings are processing
timings, not live capture-to-display timings.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol
import json

import cv2
import numpy as np

from .pipeline import RefreshError
from .window import WindowSelector


class ReplayProcessor(Protocol):
    def refresh(self, packet, generation: int) -> Any: ...
    def estimate_selected(self, packet) -> tuple[np.ndarray, list[dict[str, Any]] | None, Any]: ...


@dataclass(frozen=True)
class ReplaySettings:
    session_id: str
    clip_id: str
    split: str
    window_ns: int = 100_000_000
    research_width: int = 320
    research_height: int = 180

    def __post_init__(self) -> None:
        for value in (self.session_id, self.clip_id):
            if not value or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in value):
                raise ValueError("session_id and clip_id must use letters, digits, '_' or '-'")
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")
        if self.window_ns < 1 or self.research_width < 1 or self.research_height < 1:
            raise ValueError("window and research dimensions must be positive")


def _write_png(path: Path, image: np.ndarray, *, rgb: bool = False) -> None:
    pixels = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if rgb else image
    if not cv2.imwrite(str(path), pixels):
        raise OSError(f"could not write {path}")


def _joints(result: Any) -> list[dict[str, Any]] | None:
    try:
        depths = result.joint_depth
        return [
            {
                "index": index,
                "x": float(landmark.x),
                "y": float(landmark.y),
                "visibility": None if landmark.visibility is None else float(landmark.visibility),
                "depth_m": float(depths[index]) if index < len(depths) and np.isfinite(depths[index]) else None,
            }
            for index, landmark in enumerate(result.pose.landmarks)
        ]
    except (AttributeError, TypeError, ValueError):
        return None


class LiveReplayProcessor:
    """Use one live refresh owner for refresh and selected reuse inference."""

    def __init__(self, pipeline: Any) -> None:
        self.pipeline = pipeline

    def refresh(self, packet, generation: int) -> Any:
        return self.pipeline.process(packet, generation=generation)

    def estimate_selected(self, packet) -> tuple[np.ndarray, list[dict[str, Any]] | None, Any]:
        from .mask_selection import select_pose_gradient_mask

        result = self.pipeline._ensure_pipeline().process_frame(
            packet.rgb, packet.frame_id, packet.source_id, packet.capture_timestamp_ns
        )
        depth = np.asarray(result.depth.depth, dtype=np.float32)
        joints = _joints(result)
        selection = select_pose_gradient_mask(depth, pose_joints=joints, config=self.pipeline.mask_config)
        return depth, joints, selection


def _save_estimate(
    root: Path, key: str, depth: np.ndarray, joints: list[dict[str, Any]] | None, mask: np.ndarray
) -> dict[str, str]:
    depth_path = f"{key}_depth.npz"
    pose_path = f"{key}_pose.json"
    mask_path = f"{key}_baseline_mask.png"
    depth = np.asarray(depth, dtype=np.float32)
    np.savez_compressed(root / depth_path, depth_m=depth, validity=np.isfinite(depth).astype(np.float32))
    (root / pose_path).write_text(json.dumps({"joints": joints}, allow_nan=False), encoding="utf-8")
    _write_png(root / mask_path, (np.asarray(mask) > 0).astype(np.uint8) * 255)
    return {"depth": depth_path, "pose": pose_path, "mask": mask_path}


def _depth_histogram(depth: np.ndarray) -> dict[str, int]:
    finite = np.asarray(depth)[np.isfinite(depth)]
    counts = np.histogram(finite, bins=[0, 1, 2, 3, 5, 10, np.inf])[0]
    return dict(zip(("0_1", "1_2", "2_3", "3_5", "5_10", "10_plus"), map(int, counts)))


def run_replay(
    packets: Iterable[Any],
    *,
    selector: WindowSelector,
    processor: ReplayProcessor,
    output_dir: Path,
    settings: ReplaySettings,
    models: dict[str, str],
) -> dict[str, int]:
    """Write one row for each selected window. The caller owns ``output_dir``.

    A refresh advances the reference only after the processor returns a valid
    state. This is the same selection rule as the live application. Replay is
    synchronous, so it does not simulate dispatcher drops or wall-clock delay.
    """
    output_dir.mkdir(parents=True, exist_ok=False)
    rows_path = output_dir / "rows.jsonl"
    window: list[Any] = []
    window_start: int | None = None
    retained: dict[str, Any] | None = None
    counts = {"windows": 0, "refresh_ok": 0, "refresh_failed": 0, "reuse": 0}
    source_shape: tuple[int, int] | None = None

    def flush(rows_file) -> None:
        nonlocal window, window_start, retained
        if not window:
            return
        decision = selector.select(window)
        selected = decision.selected
        packet = selected.packet
        counts["windows"] += 1
        key = f"g{selector.generation}_f{int(packet.frame_id):08d}"
        rgb_path = f"{key}_rgb.png"
        _write_png(output_dir / rgb_path, packet.rgb, rgb=True)
        flow_path = None
        if selected.flow is not None and getattr(selected.flow, "flow", None) is not None:
            flow_path = f"{key}_flow.npz"
            np.savez_compressed(
                output_dir / flow_path,
                flow=np.asarray(selected.flow.flow, dtype=np.float32),
                valid_mask=np.asarray(selected.flow.valid_mask, dtype=np.uint8),
            )
        row: dict[str, Any] = {
            "schema_version": "1", "session_id": settings.session_id, "clip_id": settings.clip_id,
            "split": settings.split, "source_id": str(packet.source_id), "generation": selector.generation,
            "selected_frame_id": int(packet.frame_id), "selected_timestamp_ns": int(packet.capture_timestamp_ns),
            "timestamp_origin": str(packet.timestamp_origin),
            "reference_frame_id": None if retained is None else retained["frame_id"],
            "reference_timestamp_ns": None if retained is None else retained["timestamp_ns"],
            "selected_rgb": rgb_path, "reference_rgb": None if retained is None else retained["rgb"],
            "selected_depth": None, "reference_depth": None if retained is None else retained["depth"],
            "selected_pose": None, "reference_pose": None if retained is None else retained["pose"],
            "flow": flow_path, "flow_metadata": None if selected.flow is None else selected.flow.metadata(),
            "action": decision.action, "reason": decision.reason, "flow_score": decision.score,
            "candidates": decision.candidates, "status": "ok", "error": None,
            "mask_source": None, "fallback_reason": None, "baseline_mask": None,
            "mask_area_fraction": None, "splat_count": None, "timings_ms": None,
            "depth_histogram": None, "pose_detected": None,
            "models": models,
            "resize": {"width": settings.research_width, "height": settings.research_height,
                       "rule": "resize RGB/depth/mask/pose to target; scale flow x by target_width/source_width and y by target_height/source_height"},
        }
        try:
            if decision.action == "refresh":
                outcome = processor.refresh(packet, selector.generation)
                depth = outcome.depth_m
                joints = outcome.pose_joints
                mask = outcome.mask.mask
                mask_header = outcome.header.get("mask", {})
                row["mask_source"] = mask_header.get("source")
                row["fallback_reason"] = mask_header.get("fallback_reason")
                row["mask_area_fraction"] = float(outcome.mask.area_fraction)
                row["splat_count"] = int(outcome.splat_count)
                row["timings_ms"] = outcome.timings_ms
                counts["refresh_ok"] += 1
            else:
                depth, joints, selection = processor.estimate_selected(packet)
                mask = selection.info.mask
                row["mask_source"] = selection.source
                row["fallback_reason"] = selection.fallback_reason
                row["mask_area_fraction"] = float(selection.info.area_fraction)
                counts["reuse"] += 1
            paths = _save_estimate(output_dir, key, depth, joints, mask)
            row["selected_depth"] = paths["depth"]
            row["selected_pose"] = paths["pose"]
            row["baseline_mask"] = paths["mask"]
            row["depth_histogram"] = _depth_histogram(depth)
            row["pose_detected"] = bool(joints)
            if decision.action == "refresh":
                selector.commit(packet, generation=selector.generation)
                retained = {"frame_id": int(packet.frame_id), "timestamp_ns": int(packet.capture_timestamp_ns),
                            "rgb": rgb_path, "depth": paths["depth"], "pose": paths["pose"]}
        except RefreshError as error:
            row["status"] = "person_missing" if error.person_missing else "error"
            row["error"] = str(error)
            counts["refresh_failed"] += 1
        rows_file.write(json.dumps(row, allow_nan=False) + "\n")
        window = []
        window_start = None

    with rows_path.open("w", encoding="utf-8") as rows_file:
        for packet in packets:
            shape = (int(packet.width), int(packet.height))
            if source_shape is None:
                source_shape = shape
            elif shape != source_shape:
                raise ValueError("recorded clip changed image size; split it into separate clips")
            candidate = selector.observe(packet)
            window.append(candidate)
            if window_start is None:
                window_start = int(packet.capture_timestamp_ns)
            if int(packet.capture_timestamp_ns) - window_start >= settings.window_ns:
                flush(rows_file)
        flush(rows_file)
    return counts
