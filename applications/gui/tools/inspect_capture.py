"""Developer helper: inspect captured GUI screenshots and the live preview.

This script is not part of the built application. It helps verifier runs read
pixel statistics from screenshots/ and brighten dark regions for inspection.

Usage (from applications/gui):
    ../../.venv/Scripts/python.exe tools/inspect_capture.py
"""

from __future__ import annotations

import base64
import json
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def brighten(image: np.ndarray, gain: float) -> np.ndarray:
    return np.clip(image.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def describe(name: str, image: np.ndarray) -> None:
    if image.size == 0:
        print(f"{name}: empty")
        return
    mean = image.reshape(-1, image.shape[-1]).mean(axis=0) if image.ndim == 3 else image.mean()
    std = image.reshape(-1, image.shape[-1]).std(axis=0) if image.ndim == 3 else image.std()
    print(f"{name}: shape={image.shape} mean={mean} std={std} max={image.max()}")


def main() -> None:
    # 1. Live preview brightness.
    try:
        preview = fetch_json("http://127.0.0.1:8765/api/preview")
        jpeg = base64.b64decode(preview["jpeg_base64"])
        bgr = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        describe("preview", bgr)
        bright_path = ROOT / "screenshots" / "debug-preview-bright.png"
        cv2.imwrite(str(bright_path), brighten(bgr, 8.0))
        print(f"wrote {bright_path}")
    except Exception as error:  # noqa: BLE001 - diagnostic helper
        print(f"preview fetch failed: {error!r}")

    # 2. Screenshot regions. The webcam scene is dark, so the evidence images
    # are brightened for inspection only; the application never brightens data.
    screenshot_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "screenshots" / "gui-live-main.png"
    screenshot = cv2.imread(str(screenshot_path), cv2.IMREAD_COLOR)
    print(f"screenshot: {screenshot_path} shape={screenshot.shape}")
    regions = {
        "original_input": screenshot[132:380, 15:686],
        "splat": screenshot[492:750, 715:1386],
    }
    for name, region in regions.items():
        describe(name, region)
        out = ROOT / "screenshots" / f"evidence-{name}-brightened.png"
        cv2.imwrite(str(out), brighten(region, 6.0))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
