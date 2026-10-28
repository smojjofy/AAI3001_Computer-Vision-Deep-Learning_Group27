"""Shared test setup.

Run from modules/inference/depth:
    python -m pytest tests -q

Tests that need the real model are skipped (with the reason) if torch, the .pth
checkpoint in models/, or the cloned Depth-Anything-V2 repo is missing.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

DEPTH_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
sys.path.insert(0, str(DEPTH_ROOT))


def _model_skip_reason() -> str | None:
    try:
        import torch  # noqa: F401
    except ImportError:
        return "torch not installed"

    from depth_estimator import find_default_model, find_repo

    checkpoint = find_default_model()
    if not checkpoint:
        return "no .pth checkpoint in models/"
    if find_repo("metric" in Path(checkpoint).name.lower())[0] is None:
        return "Depth-Anything-V2 repo not found (clone it or set DEPTH_ANYTHING_REPO)"
    return None


@pytest.fixture(scope="session")
def estimator():
    """One CPU estimator for the whole test run (loading the model takes a few seconds)."""
    reason = _model_skip_reason()
    if reason:
        pytest.skip(reason)

    from depth_estimator import DepthEstimator

    est = DepthEstimator(device="cpu")
    yield est
    est.close()


@pytest.fixture(scope="session")
def fast_estimator():
    """Lower input size, for tests that only check shapes and handling (much faster)."""
    reason = _model_skip_reason()
    if reason:
        pytest.skip(reason)

    from depth_estimator import DepthEstimator

    est = DepthEstimator(device="cpu", input_size=252)
    yield est
    est.close()


@pytest.fixture(scope="session")
def ordering_fixtures():
    data = json.loads((FIXTURES / "ordering_pairs.json").read_text(encoding="utf-8"))
    out = {}
    for name, info in data["images"].items():
        bgr = cv2.imread(str(FIXTURES / name), cv2.IMREAD_COLOR)
        assert bgr is not None, f"fixture image missing: {name}"
        out[name] = (cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), info)
    return out


@pytest.fixture
def rgb_image():
    """A small natural-ish RGB test frame: one fixture image, resized."""
    bgr = cv2.imread(str(FIXTURES / "room_4.png"), cv2.IMREAD_COLOR)
    return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
