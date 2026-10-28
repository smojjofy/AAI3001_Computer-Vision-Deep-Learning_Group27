"""The real model: loading, settings, output dimensions, finite values, invalid input.

Skipped automatically if torch / the .pth / the Depth-Anything-V2 repo is missing.
"""

import numpy as np
import pytest

from depth_estimator import DepthEstimation, DepthFrameResult


# ------------------------------------------------------------ dimensions and finite values

@pytest.mark.parametrize("height, width", [
    (480, 640),     # landscape VGA
    (640, 480),     # portrait
    (333, 517),     # odd sizes, not multiples of 14
    (720, 1280),    # HD
    (24, 32),       # tiny
])
def test_output_matches_input_size(fast_estimator, rgb_image, height, width):
    import cv2
    frame = cv2.resize(rgb_image, (width, height), interpolation=cv2.INTER_AREA)
    est = fast_estimator.estimate(frame)
    assert isinstance(est, DepthEstimation)
    assert est.depth.shape == (height, width)
    assert est.valid.shape == est.confidence.shape == (height, width)


def test_values_are_finite_where_valid(estimator, rgb_image):
    est = estimator.estimate(rgb_image)
    assert est.depth.dtype == np.float32
    assert est.valid_fraction > 0.99, "a normal photo should be almost fully valid"
    assert np.all(np.isfinite(est.depth[est.valid]))
    assert np.all(np.isnan(est.depth[~est.valid]))
    assert np.all(np.isfinite(est.confidence))
    if est.disparity is not None:
        assert np.all(np.isfinite(est.disparity))


def test_follows_convention_for_its_kind(estimator, rgb_image):
    est = estimator.estimate(rgb_image)
    assert est.kind == estimator.kind
    values = est.depth[est.valid]
    if est.is_metric:
        assert values.min() > 0
    else:
        assert values.min() == pytest.approx(0.0, abs=1e-6)
        assert values.max() == pytest.approx(1.0, abs=1e-6)


def test_output_size_setting(rgb_image):
    from depth_estimator import DepthEstimator
    from conftest import _model_skip_reason
    if _model_skip_reason():
        pytest.skip(_model_skip_reason())
    with DepthEstimator(device="cpu", input_size=252, output_size=(120, 160)) as est:
        assert est.estimate(rgb_image).depth.shape == (120, 160)


def test_input_size_changes_detail_not_shape(estimator, fast_estimator, rgb_image):
    full = estimator.estimate(rgb_image)
    fast = fast_estimator.estimate(rgb_image)
    assert full.depth.shape == fast.depth.shape == rgb_image.shape[:2]
    # same scene, same ordering: the two resolutions should agree strongly
    a, b = full.depth.ravel(), fast.depth.ravel()
    assert np.corrcoef(a, b)[0, 1] > 0.95


def test_matches_official_repo_inference(estimator, rgb_image):
    """Our preprocessing must give the same raw output as the repo's own infer_image()."""
    import sys
    if estimator.kind != "relative_depth":
        pytest.skip("reference check written for the relative checkpoint")
    sys.path.insert(0, str(estimator.repo_path))
    try:
        reference = estimator._model.infer_image(np.ascontiguousarray(rgb_image[:, :, ::-1]), estimator.input_size)
    finally:
        sys.path.remove(str(estimator.repo_path))
    raw = estimator.predict_raw(rgb_image)
    np.testing.assert_allclose(raw, reference, rtol=1e-3, atol=1e-3)


# ------------------------------------------------------------ settings

def test_settings_are_reported(fast_estimator):
    s = fast_estimator.settings
    assert s["device"] == "cpu"
    assert s["input_size"] == 252
    assert s["kind"] in ("relative_depth", "metric_depth")
    assert s["encoder"] in ("vits", "vitb", "vitl", "vitg")


def test_bad_settings_are_rejected():
    from depth_estimator import DepthEstimator, resolve_device
    from conftest import _model_skip_reason
    if _model_skip_reason():
        pytest.skip(_model_skip_reason())
    with pytest.raises(ValueError, match="input_size"):
        DepthEstimator(device="cpu", input_size=5)
    with pytest.raises(ValueError, match="output_size"):
        DepthEstimator(device="cpu", output_size=(0, 100))
    with pytest.raises(ValueError, match="unknown device"):
        resolve_device("tpu")


def test_cuda_request_without_gpu_gives_clear_error():
    import torch
    from depth_estimator import resolve_device
    if torch.cuda.is_available():
        pytest.skip("this machine has a GPU")
    with pytest.raises(ValueError, match="CUDA"):
        resolve_device("cuda")


def test_missing_checkpoint_gives_clear_error(tmp_path):
    from depth_estimator import DepthEstimator
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        DepthEstimator(tmp_path / "nope.pth", device="cpu")


# ------------------------------------------------------------ invalid input / invalid depth

@pytest.mark.parametrize("frame, message", [
    (np.zeros((48, 64), np.uint8), "shape"),          # greyscale, no channel axis
    (np.zeros((48, 64, 4), np.uint8), "shape"),       # RGBA
    (np.zeros((48, 64, 3), np.float32), "uint8"),     # wrong dtype
])
def test_bad_frames_rejected(fast_estimator, frame, message):
    with pytest.raises(ValueError, match=message):
        fast_estimator.estimate(frame)


@pytest.mark.parametrize("value", [0, 255])
def test_blank_frames_do_not_crash_and_follow_convention(fast_estimator, value):
    """Covered lens / overexposed: output may be all-invalid, but must still be well-formed."""
    est = fast_estimator.estimate(np.full((96, 128, 3), value, np.uint8))
    assert est.depth.shape == (96, 128)
    assert np.array_equal(est.valid, np.isfinite(est.depth))
    assert np.all(est.confidence[~est.valid] == 0)


def test_closed_estimator_refuses_to_run(rgb_image):
    from depth_estimator import DepthEstimator
    from conftest import _model_skip_reason
    if _model_skip_reason():
        pytest.skip(_model_skip_reason())
    est = DepthEstimator(device="cpu", input_size=252)
    est.close()
    with pytest.raises(RuntimeError, match="closed"):
        est.estimate(rgb_image)


def test_confidence_lower_at_object_edges(estimator, ordering_fixtures):
    """On a real image, pixels at depth edges should get lower confidence than flat ones."""
    rgb, info = ordering_fixtures["room_4.png"]
    est = estimator.estimate(rgb)
    log_d = np.log(est.depth + 0.05)
    gy, gx = np.gradient(log_d)
    edges = np.hypot(gx, gy) > np.percentile(np.hypot(gx, gy), 98)
    flat = np.hypot(gx, gy) < np.percentile(np.hypot(gx, gy), 50)
    assert est.confidence[edges].mean() < est.confidence[flat].mean() - 0.3


# ------------------------------------------------------------ frame identity

def test_estimate_frame_keeps_identity(fast_estimator, rgb_image):
    result = fast_estimator.estimate_frame(rgb_image, frame_id=7, source_id="cam0",
                                           capture_timestamp_ns=123_000_000)
    assert isinstance(result, DepthFrameResult)
    assert (result.frame_id, result.source_id, result.capture_timestamp_ns) == (7, "cam0", 123_000_000)
    assert result.depth.depth.shape == rgb_image.shape[:2]
