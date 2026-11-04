"""Comparison helpers on hand-made depth maps. No model needed."""

import numpy as np
import pytest

from depth_estimator import normalize_prediction
from depth_estimator.compare import align_to_metric, ordering_agreement, relative_difference


def scene(h=60, w=80):
    """A simple 'room': floor getting farther towards the top, a near box in the middle."""
    rows = np.linspace(1.0, 6.0, h, dtype=np.float32)[::-1, None]
    metres = np.repeat(rows, w, axis=1)
    metres[20:40, 30:50] = 1.5
    return metres


def as_metric(metres):
    return normalize_prediction(metres.astype(np.float32), "metric_depth")


def as_relative(metres, scale=3.0, shift=0.2):
    """What a relative model would output for this scene: affine in disparity, unknown scale/shift."""
    return normalize_prediction((scale / metres + shift).astype(np.float32), "relative_depth")


def test_identical_ordering_scores_one():
    m = scene()
    assert ordering_agreement(as_relative(m), as_metric(m)) == pytest.approx(1.0)


def test_reversed_ordering_scores_zero():
    m = scene()
    assert ordering_agreement(as_metric(10.0 - m), as_metric(m)) == pytest.approx(0.0)


def test_alignment_recovers_metres_exactly():
    m = scene()
    aligned, fit = align_to_metric(as_relative(m, scale=7.0, shift=0.5), as_metric(m))
    np.testing.assert_allclose(aligned, m, rtol=1e-3)
    assert np.nanmax(relative_difference(aligned, as_metric(m))) < 1e-3
    assert fit["inliers"] > 0.99


def test_alignment_ignores_outliers():
    m = scene()
    rel = (3.0 / m + 0.2).astype(np.float32)
    rel[:5] = 50.0                                     # a strip of garbage
    aligned, _ = align_to_metric(normalize_prediction(rel, "relative_depth"), as_metric(m))
    good = np.ones_like(m, bool)
    good[:5] = False
    np.testing.assert_allclose(aligned[good], m[good], rtol=1e-2)


def test_alignment_needs_disparity_and_a_metric_reference():
    m = scene()
    with pytest.raises(ValueError, match="disparity"):
        align_to_metric(as_metric(m), as_metric(m))
    with pytest.raises(ValueError, match="metric"):
        align_to_metric(as_relative(m), as_relative(m))


def test_invalid_pixels_are_ignored():
    m = scene()
    metric_raw = m.copy()
    metric_raw[0:10] = np.nan
    metric = normalize_prediction(metric_raw, "metric_depth")
    aligned, _ = align_to_metric(as_relative(m), metric)
    diff = relative_difference(aligned, metric)
    assert np.all(np.isnan(diff[0:10]))
    assert np.nanmax(diff) < 1e-3
