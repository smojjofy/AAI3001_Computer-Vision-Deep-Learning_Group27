"""Relative-depth ordering on fixture images with KNOWN true depth.

fixtures/ordering_pairs.json lists pixel pairs (near point, far point) on synthetic
room renders where the far point is at least 1.3x farther and both points sit on
smooth surfaces. A depth estimator that understands the scene should get almost
all of them in the right order, whatever its scale.

Skipped automatically if torch / the .pth / the Depth-Anything-V2 repo is missing.
"""

import numpy as np
import pytest

# Share of pairs that must be ordered correctly. Depth Anything V2 Small gets ~100%
# on these; the margin allows for a different checkpoint or input size.
MIN_PAIR_ACCURACY = 0.90


def pair_accuracy(depth: np.ndarray, pairs) -> float:
    correct = 0
    for near_x, near_y, far_x, far_y in pairs:
        correct += depth[near_y, near_x] < depth[far_y, far_x]
    return correct / len(pairs)


@pytest.mark.parametrize("name", ["room_0.png", "room_4.png", "room_7.png"])
def test_known_pairs_are_ordered(estimator, ordering_fixtures, name):
    rgb, info = ordering_fixtures[name]
    est = estimator.estimate(rgb)
    accuracy = pair_accuracy(est.depth, info["pairs"])
    assert accuracy >= MIN_PAIR_ACCURACY, f"{name}: only {accuracy:.0%} of near/far pairs in the right order"


@pytest.mark.parametrize("name", ["room_0.png", "room_4.png", "room_7.png"])
def test_ordering_holds_at_low_resolution(fast_estimator, ordering_fixtures, name):
    rgb, info = ordering_fixtures[name]
    accuracy = pair_accuracy(fast_estimator.estimate(rgb).depth, info["pairs"])
    assert accuracy >= MIN_PAIR_ACCURACY, f"{name} at input_size=252: {accuracy:.0%}"


@pytest.mark.parametrize("name", ["room_0.png", "room_4.png", "room_7.png"])
def test_nearest_object_is_in_nearest_fifth(estimator, ordering_fixtures, name):
    """The truly nearest surface point should land among the nearest 20% of predicted depths."""
    rgb, info = ordering_fixtures[name]
    est = estimator.estimate(rgb)
    x, y = info["nearest_point"]
    value = est.depth[y, x]
    assert value <= np.nanpercentile(est.depth, 20), f"{name}: nearest point predicted at {value:.2f}"


def test_ordering_survives_brightness_change(estimator, ordering_fixtures):
    """Same scene, darker exposure: the near/far ordering should not change."""
    rgb, info = ordering_fixtures["room_4.png"]
    darker = (rgb.astype(np.float32) * 0.6).astype(np.uint8)
    accuracy = pair_accuracy(estimator.estimate(darker).depth, info["pairs"])
    assert accuracy >= MIN_PAIR_ACCURACY


def test_ordering_survives_horizontal_flip(estimator, ordering_fixtures):
    """Mirroring the image should mirror the depth, not change which points are nearer."""
    rgb, info = ordering_fixtures["room_0.png"]
    width = rgb.shape[1]
    flipped = np.ascontiguousarray(rgb[:, ::-1])
    depth = estimator.estimate(flipped).depth
    pairs = [[width - 1 - nx, ny, width - 1 - fx, fy] for nx, ny, fx, fy in info["pairs"]]
    assert pair_accuracy(depth, pairs) >= MIN_PAIR_ACCURACY
