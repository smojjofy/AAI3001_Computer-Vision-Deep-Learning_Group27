"""Draw a checked person mask over review frames, in a separate label store.

Example, after ``scripts.learning_data review-set``:

    python -m scripts.label_review_masks --review-dir <clip>.review \
        --labels-root data/processed/learning_replay/labels --dataset movement_01

The window shows the selected RGB frame. Paint the person wherever the mask must
keep pixels. Keys:

    left drag   paint, right drag  erase, wheel  brush size
    d           erase the complete frame
    n / p       next or previous frame
    g           save (writes a PNG mask and its metadata)
    s           skip this frame for now
    q or Esc    save the frame and quit

Only the standard library and OpenCV are used, because the project environment
does not include a GUI dependency. A checked mask is written to the label store,
never to the replay data, and never replaces ``baseline_mask``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.learning_data import (  # noqa: E402
    label_path,
    load_review_index,
    save_checked_mask,
)

WINDOW = "checked mask labeler"

BRUSH_STEPS = (5, 9, 15, 25, 41, 61, 91)
ALPHA = 0.45


class LabelState:
    def __init__(
        self, image: np.ndarray, existing: np.ndarray | None = None, brush_index: int = 2
    ) -> None:
        self.image = image
        self.paint = (
            np.zeros(image.shape[:2], np.uint8)
            if existing is None
            else (np.asarray(existing) > 0).astype(np.uint8) * 255
        )
        self.brush_index = max(0, min(brush_index, len(BRUSH_STEPS) - 1))
        self.painting: int | None = None

    @property
    def radius(self) -> int:
        return BRUSH_STEPS[self.brush_index] // 2

    def stroke(self, point: tuple[int, int], value: int) -> None:
        cv2.circle(self.paint, point, max(1, self.radius), value, -1)

    def clear(self) -> None:
        self.paint[:] = 0

    def erased(self) -> bool:
        return not bool(self.paint.any())

    def view(self) -> np.ndarray:
        output = self.image.copy()
        inside = self.paint > 0
        tint = np.array([0, 200, 0], dtype=np.float32)
        output[inside] = (output[inside].astype(np.float32) * (1 - ALPHA) + tint * ALPHA).astype(np.uint8)
        hint = f"brush {BRUSH_STEPS[self.brush_index]}px  |  n/p frame  g save  d clear  q quit"
        if self.erased():
            hint = "EMPTY: press d again on an empty frame to accept no person  |  g save"
        cv2.putText(output, hint, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return output


def _mouse(state: LabelState, event: int, x: int, y: int, flags: int, param) -> None:
    if event == cv2.EVENT_LBUTTONDOWN:
        state.painting = 255
        state.stroke((x, y), 255)
    elif event == cv2.EVENT_RBUTTONDOWN:
        state.painting = 0
        state.stroke((x, y), 0)
    elif event == cv2.EVENT_MOUSEMOVE and state.painting is not None:
        state.stroke((x, y), state.painting)
    elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
        state.painting = None


def run_session(
    review_index: dict,
    *,
    labels_root: Path,
    dataset: str,
    notes: str = "",
    start_at: int = 0,
) -> dict:
    """Open the labeler over the review frames. Returns counts of what was saved."""
    review_dir = Path(review_index["review_dir"])
    items = [item for item in review_index["items"] if item.get("image")]
    if not items:
        raise ValueError("review index has no images to label")
    saved = 0
    skipped = 0
    empty_saved = 0
    checked_dir = Path(labels_root) / dataset / "masks"
    checked_dir.mkdir(parents=True, exist_ok=True)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    position = max(0, min(int(start_at), len(items) - 1))
    keep_labeling = True
    try:
        while keep_labeling and 0 <= position < len(items):
            item = items[position]
            image_path = review_dir / item["image"]
            frame = cv2.imread(str(image_path))
            if frame is None:
                skipped += 1
                position += 1
                continue
            # The review image is RGB beside overlay. Use the left half.
            width = frame.shape[1] // 2
            rgb = frame[:, :width].copy()
            existing_path = label_path(labels_root, dataset, item["frame_key"])
            existing = cv2.imread(str(existing_path), cv2.IMREAD_GRAYSCALE)
            state = LabelState(rgb, existing)
            title = f"{position + 1}/{len(items)}  frame {item['frame_id']}  {item['frame_key']}"
            cv2.setMouseCallback(WINDOW, _mouse, state)
            action = "skip"
            move = 1
            editing = True
            while editing:
                view = state.view()
                cv2.putText(
                    view, title, (8, view.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
                )
                cv2.imshow(WINDOW, view)
                key = cv2.waitKey(20) & 0xFF
                if key == 255:
                    continue
                if key == ord("q") or key == 27:
                    # Quit after this frame. Save it only when it has paint.
                    action = "save" if state.paint.any() else "skip"
                    keep_labeling = False
                    editing = False
                elif key in (ord("g"), ord("n"), ord("s")):
                    action = "save" if key == ord("g") else "skip"
                    editing = False
                elif key == ord("p"):
                    action = "skip"
                    move = -1
                    editing = False
                elif key == ord("d"):
                    state.clear()
                elif key == ord("["):
                    state.brush_index = max(0, state.brush_index - 1)
                elif key == ord("]"):
                    state.brush_index = min(len(BRUSH_STEPS) - 1, state.brush_index + 1)
            if action == "save":
                save_checked_mask(
                    Path(labels_root), dataset, item["frame_key"], state.paint,
                    source="manual_labeler", notes=notes,
                )
                saved += 1
                if state.erased():
                    empty_saved += 1
            else:
                skipped += 1
            position += move
    finally:
        cv2.destroyWindow(WINDOW)
    return {"saved": saved, "skipped": skipped, "saved_empty_frames": empty_saved}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--notes", default="")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    index = load_review_index(args.review_dir)
    counts = run_session(index, labels_root=args.labels_root, dataset=args.dataset, notes=args.notes)
    print(counts)


if __name__ == "__main__":
    main()
