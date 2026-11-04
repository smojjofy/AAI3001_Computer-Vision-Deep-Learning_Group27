"""YOLO depth as a second depth source: same convention, same identity, comparable ordering.

Skipped if ultralytics isn't installed or models/yolo26n-depth.pt isn't there
(tests never download weights; run the app or a script once to fetch them).
"""

from pathlib import Path

import numpy as np
import pytest

from depth_estimator import DepthEstimation, DepthFrameResult
from depth_estimator.compare import align_to_metric, ordering_agreement, relative_difference
from depth_estimator.estimator import MODELS_DIR

YOLO_WEIGHTS = MODELS_DIR / "yolo26n-depth.pt"


@pytest.fixture(scope="session")
def yolo():
    pytest.importorskip("ultralytics")
    if not YOLO_WEIGHTS.is_file():
        pytest.skip(f"{YOLO_WEIGHTS.name} not in models/ (run the app or a script once to download it)")
    from depth_estimator import YoloDepthEstimator
    est = YoloDepthEstimator(YOLO_WEIGHTS, device="cpu", imgsz=320)
    yield est
    est.close()


# ------------------------------------------------------------ convention and settings

@pytest.mark.parametrize("height, width", [(480, 640), (640, 480), (333, 517)])
def test_output_matches_input_size(yolo, rgb_image, height, width):
    import cv2
    frame = cv2.resize(rgb_image, (width, height), interpolation=cv2.INTER_AREA)
    est = yolo.estimate(frame)
    assert isinstance(est, DepthEstimation)
    assert est.depth.shape == est.valid.shape == est.confidence.shape == (height, width)


def test_is_metric_and_follows_convention(yolo, rgb_image):
    est = yolo.estimate(rgb_image)
    assert est.kind == "metric_depth" and est.is_metric and est.disparity is None
    assert est.valid_fraction > 0.99
    values = est.depth[est.valid]
    assert np.all(np.isfinite(values)) and values.min() > 0
    assert 0.1 < np.median(values) < 100, "an indoor/outdoor scene should be metres, not millimetres"


def test_output_size_setting(rgb_image):
    pytest.importorskip("ultralytics")
    if not YOLO_WEIGHTS.is_file():
        pytest.skip("YOLO depth weights not in models/")
    from depth_estimator import YoloDepthEstimator
    with YoloDepthEstimator(YOLO_WEIGHTS, device="cpu", imgsz=256, output_size=(90, 120)) as est:
        assert est.estimate(rgb_image).depth.shape == (90, 120)
        assert est.settings["imgsz"] == 256 and est.settings["kind"] == "metric_depth"


def test_bad_frames_rejected(yolo):
    with pytest.raises(ValueError, match="shape"):
        yolo.estimate(np.zeros((48, 64), np.uint8))
    with pytest.raises(ValueError, match="uint8"):
        yolo.estimate(np.zeros((48, 64, 3), np.float32))


def test_non_depth_checkpoint_rejected(tmp_path):
    pytest.importorskip("ultralytics")
    seg = MODELS_DIR / "yolo11n-seg.pt"
    if not seg.is_file():
        pytest.skip("no segmentation checkpoint around to test with")
    from depth_estimator import YoloDepthEstimator
    with pytest.raises(ValueError, match="not a depth model"):
        YoloDepthEstimator(seg, device="cpu")


def test_estimate_frame_keeps_identity(yolo, rgb_image):
    result = yolo.estimate_frame(rgb_image, frame_id=4, source_id="cam0", capture_timestamp_ns=99)
    assert isinstance(result, DepthFrameResult)
    assert (result.frame_id, result.source_id, result.capture_timestamp_ns) == (4, "cam0", 99)


# ------------------------------------------------------------ ordering on fixtures with known depth

@pytest.mark.parametrize("name", ["room_0.png", "room_4.png", "room_7.png"])
def test_known_pairs_are_ordered(yolo, ordering_fixtures, name):
    from test_ordering import MIN_PAIR_ACCURACY, pair_accuracy
    rgb, info = ordering_fixtures[name]
    accuracy = pair_accuracy(yolo.estimate(rgb).depth, info["pairs"])
    assert accuracy >= MIN_PAIR_ACCURACY, f"{name}: {accuracy:.0%}"


# ------------------------------------------------------------ comparing the two sources

def test_both_sources_agree_on_ordering(yolo, estimator, rgb_image):
    agreement = ordering_agreement(estimator.estimate(rgb_image), yolo.estimate(rgb_image))
    assert agreement > 0.9


def test_alignment_puts_relative_depth_in_metres(yolo, estimator, rgb_image):
    relative, metric = estimator.estimate(rgb_image), yolo.estimate(rgb_image)
    aligned, fit = align_to_metric(relative, metric)
    diff = relative_difference(aligned, metric)
    assert aligned.shape == metric.depth.shape
    assert fit["inliers"] > 0.5
    assert np.nanmedian(diff) < 0.25, "after matching scale, the two should mostly agree"
