"""The shared depth convention is enforced by DepthEstimation itself. No model needed."""

import numpy as np
import pytest

from depth_estimator import DepthEstimation, DepthFrameResult


def make(depth, kind="metric_depth", valid=None, confidence=None, disparity=None):
    depth = np.asarray(depth, dtype=np.float32)
    if valid is None:
        valid = np.isfinite(depth)
    if confidence is None:
        confidence = np.where(valid, 1.0, 0.0).astype(np.float32)
    return DepthEstimation(depth=depth, kind=kind, model_name="test", valid=valid,
                           confidence=confidence, disparity=disparity)


def test_valid_metric_estimation_is_accepted():
    est = make([[1.0, 2.0], [np.nan, 3.5]])
    assert est.is_metric
    assert est.shape == (2, 2)
    assert est.valid_fraction == 0.75


def test_valid_relative_estimation_is_accepted():
    est = make([[0.0, 0.5], [1.0, np.nan]], kind="relative_depth")
    assert not est.is_metric


@pytest.mark.parametrize("bad, message", [
    (np.zeros((2, 2, 1), np.float32), "shape"),
    (np.zeros((2, 2), np.float64), "float32"),
])
def test_depth_shape_and_dtype(bad, message):
    with pytest.raises(ValueError, match=message):
        DepthEstimation(depth=bad, kind="metric_depth", model_name="t",
                        valid=np.ones(bad.shape[:2], bool),
                        confidence=np.ones(bad.shape[:2], np.float32))


def test_unknown_kind_rejected():
    with pytest.raises(ValueError, match="kind"):
        make([[1.0]], kind="relative_disparity")   # old name, no longer allowed


def test_invalid_pixels_must_be_nan_not_zero():
    depth = np.array([[1.0, 0.0]], np.float32)
    valid = np.array([[True, False]])               # says pixel 2 invalid, but it isn't NaN
    with pytest.raises(ValueError, match="NaN"):
        make(depth, valid=valid, confidence=np.array([[1.0, 0.0]], np.float32))


def test_valid_mask_must_match_shape_and_dtype():
    with pytest.raises(ValueError, match="valid"):
        make([[1.0, 2.0]], valid=np.array([[1, 1]], np.uint8))


@pytest.mark.parametrize("confidence", [
    np.array([[1.5, 1.0]], np.float32),     # above 1
    np.array([[-0.1, 1.0]], np.float32),    # below 0
])
def test_confidence_must_be_0_to_1(confidence):
    with pytest.raises(ValueError, match="confidence"):
        make([[1.0, 2.0]], confidence=confidence)


def test_confidence_must_be_zero_where_invalid():
    with pytest.raises(ValueError, match="confidence must be 0"):
        make([[1.0, np.nan]], confidence=np.array([[1.0, 0.5]], np.float32))


def test_metric_depth_must_be_positive():
    with pytest.raises(ValueError, match="> 0"):
        make([[1.0, -2.0]])


def test_relative_depth_must_be_within_0_1():
    with pytest.raises(ValueError, match="0..1"):
        make([[0.2, 1.4]], kind="relative_depth")


def test_disparity_must_match_depth():
    with pytest.raises(ValueError, match="disparity"):
        make([[0.2, 0.4]], kind="relative_depth", disparity=np.zeros((3, 3), np.float32))


@pytest.mark.parametrize("kwargs, message", [
    ({"frame_id": -1}, "frame_id"),
    ({"source_id": ""}, "source_id"),
    ({"capture_timestamp_ns": -5}, "capture_timestamp_ns"),
])
def test_frame_identity_validation(kwargs, message):
    base = {"frame_id": 0, "source_id": "cam0", "capture_timestamp_ns": 0, "depth": make([[1.0]])}
    with pytest.raises(ValueError, match=message):
        DepthFrameResult(**{**base, **kwargs})
