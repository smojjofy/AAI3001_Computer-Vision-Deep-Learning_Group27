"""State tests for the checked-mask labeler that do not open a window.

The window loop itself needs a display, so these tests cover the paint state,
the store location, and the rule that a checked mask never writes into the
replay data directory.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts import label_review_masks as labeller
from scripts import learning_data as module


def image() -> np.ndarray:
    return np.full((40, 60, 3), 80, np.uint8)


def test_paint_and_erase_change_the_mask() -> None:
    state = labeller.LabelState(image())
    assert state.erased() is True
    state.stroke((20, 20), 255)
    assert state.erased() is False
    assert int((state.paint > 0).sum()) > 0
    painted = int((state.paint > 0).sum())
    state.stroke((20, 20), 0)
    assert int((state.paint > 0).sum()) < painted
    state.clear()
    assert state.erased() is True


def test_brush_size_steps_within_bounds() -> None:
    state = labeller.LabelState(image(), brush_index=0)
    assert state.radius == labeller.BRUSH_STEPS[0] // 2
    state.stroke((20, 20), 255)
    small = int((state.paint > 0).sum())
    state.brush_index = len(labeller.BRUSH_STEPS) - 1
    state.clear()
    state.stroke((20, 20), 255)
    assert int((state.paint > 0).sum()) > small
    view = state.view()
    assert view.shape == (40, 60, 3)


def test_existing_checked_mask_is_loaded_for_correction(tmp_path: Path) -> None:
    labels_root = tmp_path / "labels"
    module.save_checked_mask(
        labels_root, "movement_01", "g0_f00000001", np.ones((40, 60), bool), source="manual"
    )
    existing = labeller.cv2.imread(
        str(module.label_path(labels_root, "movement_01", "g0_f00000001")),
        labeller.cv2.IMREAD_GRAYSCALE,
    )
    state = labeller.LabelState(image(), existing)
    assert state.erased() is False
    assert int((state.paint > 0).sum()) == 40 * 60


def test_view_overlays_paint_without_touching_the_source_image() -> None:
    state = labeller.LabelState(image())
    state.stroke((10, 10), 255)
    view = state.view()
    assert not np.array_equal(view, state.image)
    assert int(state.image.max()) == 80


def test_session_requires_images() -> None:
    with pytest.raises(ValueError, match="no images"):
        labeller.run_session(
            {"review_dir": "x", "items": [{"image": None, "frame_key": "k", "frame_id": 1}]},
            labels_root=Path("labels"), dataset="d",
        )
