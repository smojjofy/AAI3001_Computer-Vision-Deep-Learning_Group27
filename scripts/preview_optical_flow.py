"""Save a recorded frame pair with flow arrows and a color flow map.

Example: python scripts/preview_optical_flow.py recording.mp4 --frame-id 60
The output defaults to the ignored outputs/flow_previews directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "modules/stream_handler/src"))
sys.path.insert(0, str(ROOT / "modules/inference/optical_flow/src"))

from optical_flow import FlowConfig, OpticalFlowTracker, create_estimator  # noqa: E402
from stream_handler import RecordedVideoSource, ReplayMode, SourceExhausted, StreamHandler  # noqa: E402


def _panel_label(image: np.ndarray, label: str) -> np.ndarray:
    cv2.rectangle(image, (0, 0), (image.shape[1], 32), (0, 0, 0), -1)
    cv2.putText(image, label, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def _preview(previous, current, result, grid_step: int, color_max_px: float, arrow_scale: float) -> np.ndarray:
    arrows = cv2.cvtColor(previous.rgb.copy(), cv2.COLOR_RGB2BGR)
    current_bgr = cv2.cvtColor(current.rgb, cv2.COLOR_RGB2BGR)
    height, width = result.valid_mask.shape
    for y in range(grid_step // 2, height, grid_step):
        for x in range(grid_step // 2, width, grid_step):
            if not result.valid_mask[y, x]:
                continue
            dx, dy = result.flow[y, x]
            if np.hypot(dx, dy) >= 0.5:
                tip = (int(round(x + float(dx) * arrow_scale)), int(round(y + float(dy) * arrow_scale)))
                cv2.arrowedLine(arrows, (x, y), tip, (0, 0, 0), 3, cv2.LINE_AA, tipLength=0.3)
                cv2.arrowedLine(arrows, (x, y), tip, (40, 240, 40), 1, cv2.LINE_AA, tipLength=0.3)

    magnitude, angle = cv2.cartToPolar(result.flow[..., 0], result.flow[..., 1], angleInDegrees=True)
    hsv = np.zeros((height, width, 3), dtype=np.uint8)
    hsv[..., 0] = np.mod(angle / 2, 180).astype(np.uint8)
    hsv[..., 1] = np.where(result.valid_mask, 255, 0).astype(np.uint8)
    visible = result.valid_mask & (magnitude >= 0.25)
    hsv[..., 2] = np.where(visible, np.clip(120 + magnitude / color_max_px * 135, 0, 255), 0).astype(np.uint8)
    colors = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return np.hstack([
        _panel_label(arrows, f"previous {previous.frame_id} + arrows x{arrow_scale:g}"),
        _panel_label(current_bgr, f"current {current.frame_id}"),
        _panel_label(colors, f"flow {result.status}; valid {result.valid_fraction:.2f}"),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--frame-id", type=int, required=True, help="Current video frame ID, at least 1")
    parser.add_argument("--config", type=Path, default=ROOT / "config/optical_flow.example.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--grid-step", type=int, default=24)
    parser.add_argument("--flow-color-max-px", type=float, default=20.0)
    parser.add_argument("--arrow-scale", type=float, default=4.0,
                        help="Visual magnification of arrows; numeric flow is unchanged")
    args = parser.parse_args()
    if not args.video.is_file() or args.frame_id < 1 or args.grid_step < 4 or args.flow_color_max_px <= 0 or args.arrow_scale <= 0:
        parser.error("provide an existing video, frame ID >= 1, grid step >= 4, and positive preview scales")

    config = FlowConfig(**json.loads(args.config.read_text(encoding="utf-8")))
    if config.flow_model == "raft_small":
        import torch

        torch.hub.set_dir(str(ROOT / "outputs/torch_cache"))
    handler = StreamHandler(RecordedVideoSource(args.video, replay_mode=ReplayMode.FASTEST))
    tracker = OpticalFlowTracker(create_estimator(config))
    previous = None
    selected = None
    handler.open()
    try:
        while True:
            try:
                handler.pump_once()
            except SourceExhausted:
                break
            current = handler.queue.get(timeout_s=0)
            assert current is not None
            result = tracker.push(current)
            if current.frame_id == args.frame_id:
                selected = (previous, current, result)
                break
            previous = current
    finally:
        handler.close()
    if selected is None or selected[0] is None:
        parser.error(f"frame {args.frame_id} is not available in the video")

    image = _preview(*selected, args.grid_step, args.flow_color_max_px, args.arrow_scale)
    output = args.output or ROOT / "outputs/flow_previews" / f"{config.flow_model}_frame_{args.frame_id}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), image):
        raise RuntimeError(f"could not write preview: {output}")
    print(json.dumps({
        "output": str(output),
        "previous_frame_id": selected[0].frame_id,
        "current_frame_id": selected[1].frame_id,
        "status": selected[2].status,
        "valid_fraction": round(selected[2].valid_fraction, 3),
        "direction": "previous_to_current",
    }))


if __name__ == "__main__":
    main()
