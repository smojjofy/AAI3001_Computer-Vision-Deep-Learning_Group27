"""Normalisation to the shared convention, on hand-made raw outputs. No model needed."""

import numpy as np
import pytest

from depth_estimator import edge_confidence, normalize_prediction


def ramp_disparity(h=40, w=60):
    """Raw relative output: disparity falls from left (near) to right (far)."""
    return np.tile(np.linspace(5.0, 1.0, w, dtype=np.float32), (h, 1))


# ------------------------------------------------------------ relative models

def test_relative_flips_to_bigger_is_farther_and_spans_0_1():
    est = normalize_prediction(ramp_disparity(), "relative_depth")
    assert est.kind == "relative_depth"
    assert est.depth.dtype == np.float32
    assert np.nanmin(est.depth) == pytest.approx(0.0)
    assert np.nanmax(est.depth) == pytest.approx(1.0)
    row = est.depth[0]
    assert np.all(np.diff(row) > 0), "farther (lower disparity) must give bigger depth"


def test_relative_preserves_ordering_exactly():
    rng = np.random.default_rng(0)
    raw = rng.uniform(0.1, 9.0, (30, 30)).astype(np.float32)
    est = normalize_prediction(raw, "relative_depth")
    # bigger disparity = closer = smaller depth, for every pair of pixels
    order_raw = np.argsort(-raw.ravel(), kind="stable")
    assert np.all(np.diff(est.depth.ravel()[order_raw]) >= 0)


def test_relative_keeps_raw_disparity_for_alignment():
    raw = ramp_disparity()
    est = normalize_prediction(raw, "relative_depth")
    assert est.disparity is not None
    np.testing.assert_array_equal(est.disparity, raw)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, -1.0])
def test_relative_corrupt_values_become_invalid(bad):
    raw = ramp_disparity()
    raw[5, 7] = bad
    est = normalize_prediction(raw, "relative_depth")
    assert not est.valid[5, 7]
    assert np.isnan(est.depth[5, 7])
    assert est.confidence[5, 7] == 0.0
    assert est.valid.sum() == raw.size - 1


def test_relative_flat_frame_is_all_invalid():
    """A covered lens / black frame gives no structure: nothing should be trusted."""
    est = normalize_prediction(np.full((20, 20), 3.0, np.float32), "relative_depth")
    assert not est.valid.any()
    assert np.all(np.isnan(est.depth))
    assert np.all(est.confidence == 0.0)


def test_relative_rejects_metric_range_filter():
    with pytest.raises(ValueError, match="metric"):
        normalize_prediction(ramp_disparity(), "relative_depth", valid_range=(0.3, 5.0))


# ------------------------------------------------------------ metric models

def test_metric_keeps_metres_and_marks_nonpositive_invalid():
    raw = np.array([[0.0, 1.5, 2.5], [-1.0, 3.0, np.nan]], np.float32)
    est = normalize_prediction(raw, "metric_depth")
    assert est.is_metric and est.disparity is None
    np.testing.assert_array_equal(est.valid, [[False, True, True], [False, True, False]])
    assert est.depth[0, 1] == pytest.approx(1.5)


def test_metric_saturated_at_cap_is_invalid():
    raw = np.array([[2.0, 19.999, 20.0]], np.float32)
    est = normalize_prediction(raw, "metric_depth", max_depth=20.0)
    np.testing.assert_array_equal(est.valid, [[True, False, False]])


def test_metric_valid_range():
    raw = np.array([[0.1, 0.5, 4.0, 12.0]], np.float32)
    est = normalize_prediction(raw, "metric_depth", valid_range=(0.3, 10.0))
    np.testing.assert_array_equal(est.valid, [[False, True, True, False]])


@pytest.mark.parametrize("bad_range", [(0.0, 5.0), (5.0, 1.0), (-1.0, 2.0)])
def test_metric_bad_valid_range_rejected(bad_range):
    with pytest.raises(ValueError, match="valid_range"):
        normalize_prediction(np.ones((2, 2), np.float32), "metric_depth", valid_range=bad_range)


def test_unknown_kind_rejected():
    with pytest.raises(ValueError, match="kind"):
        normalize_prediction(np.ones((2, 2), np.float32), "disparity")


def test_raw_must_be_2d():
    with pytest.raises(ValueError, match="shape"):
        normalize_prediction(np.ones((2, 2, 1), np.float32), "metric_depth")


# ------------------------------------------------------------ confidence

def test_confidence_high_on_flat_low_at_depth_edge():
    depth = np.full((40, 40), 2.0, np.float32)
    depth[:, 20:] = 5.0                          # a wall behind an object: sharp edge at x=20
    valid = np.ones_like(depth, bool)
    conf = edge_confidence(depth, valid)
    assert conf.dtype == np.float32
    assert conf[20, 5] > 0.9 and conf[20, 35] > 0.9      # flat areas trusted
    assert conf[20, 19] < 0.1 and conf[20, 20] < 0.1     # both sides of the edge distrusted


def test_confidence_drops_next_to_invalid_pixels():
    depth = np.full((20, 20), 2.0, np.float32)
    valid = np.ones_like(depth, bool)
    depth[10, 10] = np.nan
    valid[10, 10] = False
    conf = edge_confidence(depth, valid)
    assert conf[10, 10] == 0.0
    assert conf[10, 11] < 0.1
    assert conf[2, 2] > 0.9


def test_confidence_range_on_random_input():
    rng = np.random.default_rng(1)
    est = normalize_prediction(rng.uniform(0.5, 8.0, (50, 50)).astype(np.float32), "metric_depth")
    assert est.confidence.min() >= 0.0 and est.confidence.max() <= 1.0
