
import cv2
import numpy as np
import pytest
 
from src import DepthEstimation, DepthFrameResult
from src.utils import MODELS_DIR
 
YOLO_WEIGHTS = MODELS_DIR / "yolo26n-depth.pt"
 
 
def _skip_without_yolo():
    pytest.importorskip("ultralytics")
    if not YOLO_WEIGHTS.is_file():
        pytest.skip(f"{YOLO_WEIGHTS.name} not in models/ (run the app or a script once to download it)")
 
 
@pytest.fixture(scope="session")
def yolo():
    _skip_without_yolo()
    from src import YoloDepthEstimator
    est = YoloDepthEstimator(YOLO_WEIGHTS, device="cpu", imgsz=320)
    yield est
    est.close()
 
 
@pytest.mark.parametrize("height, width", [(480, 640), (640, 480), (333, 517)])
def test_output_matches_input_size(yolo, rgb_image, height, width):
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
    assert 0.1 < np.median(values) < 100, "should be metres, not millimetres"
 
 
def test_output_size_setting(rgb_image):
    _skip_without_yolo()
    from src import YoloDepthEstimator
    with YoloDepthEstimator(YOLO_WEIGHTS, device="cpu", imgsz=256, output_size=(90, 120)) as est:
        assert est.estimate(rgb_image).depth.shape == (90, 120)
        assert est.settings["imgsz"] == 256 and est.settings["kind"] == "metric_depth"
 
 
def test_bad_frames_rejected(yolo):
    with pytest.raises(ValueError, match="shape"):
        yolo.estimate(np.zeros((48, 64), np.uint8))
    with pytest.raises(ValueError, match="uint8"):
        yolo.estimate(np.zeros((48, 64, 3), np.float32))
 
 
def test_estimate_frame_keeps_identity(yolo, rgb_image):
    result = yolo.estimate_frame(rgb_image, frame_id=4, source_id="cam0", capture_timestamp_ns=99)
    assert isinstance(result, DepthFrameResult)
    assert (result.frame_id, result.source_id, result.capture_timestamp_ns) == (4, "cam0", 99)