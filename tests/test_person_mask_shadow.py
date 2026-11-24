"""Check the shadow person-mask path against the training path and the model."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from modules.inference.checkpoint import save_checkpoint
from modules.inference.paired_research import PairedMaskMotionNet, PairedReplayDataset
from modules.inference.paired_research.shadow import (
    PERMIT_KERNEL_PX,
    PERMIT_RADIUS_PX,
    ShadowMaskPipeline,
    ShadowMaskRunner,
    ShadowRecordWriter,
    build_research_input,
    compare_masks,
    describe_failure,
    dilate_baseline_ellipse,
    file_sha256,
    percentile,
    provisional_median_background,
    read_shadow_records,
    research_permitted_region,
    source_permitted_region,
    summarize_shadow_records,
)
from modules.inference.paired_research.shadow import _WindowFrame


# --------------------------------------------------------------------------
# Fixtures and stubs
# --------------------------------------------------------------------------


def _replay_fixture(tmp_path: Path):
    """Write a tiny replay directory like the one in ``test_paired_research``."""
    replay = tmp_path / "replay"
    labels = tmp_path / "labels"
    checked = labels / "clip" / "masks"
    replay.mkdir()
    checked.mkdir(parents=True)
    rng = np.random.default_rng(7)
    # On disk the PNG holds BGR. The loader converts BGR to RGB.
    reference_bgr = np.zeros((8, 8, 3), np.uint8)
    reference_bgr[..., 2] = 255
    selected_bgr = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
    cv2.imwrite(str(replay / "g0_f00000001_rgb.png"), reference_bgr)
    cv2.imwrite(str(replay / "g0_f00000002_rgb.png"), selected_bgr)
    mask = np.zeros((8, 8), np.uint8)
    mask[:, :4] = 255
    cv2.imwrite(str(replay / "g0_f00000002_baseline_mask.png"), mask)
    cv2.imwrite(str(checked / "g0_f00000002.png"), mask)
    reference_depth = np.full((8, 8), 2.0, np.float32)
    selected_depth = np.full((8, 8), 3.0, np.float32)
    selected_depth[0, 0] = np.nan
    reference_validity = np.ones((8, 8), np.float32)
    selected_validity = np.ones((8, 8), np.float32)
    selected_validity[0, 0] = 0.0
    np.savez_compressed(
        replay / "g0_f00000001_depth.npz", depth_m=reference_depth, validity=reference_validity
    )
    np.savez_compressed(
        replay / "g0_f00000002_depth.npz", depth_m=selected_depth, validity=selected_validity
    )
    joints = {
        1: [{"index": 0, "x": 0.25, "y": 0.5, "visibility": 0.9, "depth_m": 2.0}],
        2: [
            {"index": 0, "x": 0.5, "y": 0.5, "visibility": 0.9, "depth_m": 3.0},
            {"index": 1, "x": 0.125, "y": 0.375, "visibility": 0.4, "depth_m": 3.0},
            {"index": 2, "x": 0.1, "y": 0.2, "visibility": 0.1, "depth_m": 3.0},
        ],
    }
    for frame_id, records in joints.items():
        (replay / f"g0_f{frame_id:08d}_mask_pose.json").write_text(
            json.dumps({"frame_id": frame_id, "joints": records}), encoding="utf-8"
        )
    flow = np.zeros((8, 8, 2), np.float32)
    flow[..., 0] = 20.0
    flow[..., 1] = -10.0
    flow[2, 3] = np.nan
    valid = np.ones((8, 8), np.uint8)
    valid[6, 5] = 0
    np.savez_compressed(replay / "g0_f00000002_flow.npz", flow=flow, valid_mask=valid)
    row = {
        "split": "train", "status": "ok", "selected_frame_id": 2,
        "reference_frame_id": 1, "selected_rgb": "g0_f00000002_rgb.png",
        "reference_rgb": "g0_f00000001_rgb.png",
        "selected_depth": "g0_f00000002_depth.npz",
        "reference_depth": "g0_f00000001_depth.npz",
        "mask_pose": "g0_f00000002_mask_pose.json",
        "flow": "g0_f00000002_flow.npz",
        "flow_metadata": {
            "previous_frame_id": 1, "current_frame_id": 2,
            "direction": "previous_to_current",
        },
        "baseline_mask": "g0_f00000002_baseline_mask.png",
    }
    (replay / "rows.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    (replay / "manifest.json").write_text(
        json.dumps({"split": "train", "clip_sha256": "abc"}), encoding="utf-8"
    )
    filter_path = tmp_path / "filter.json"
    filter_path.write_text(json.dumps({
        "output_dir": str(replay), "labels_root": str(labels),
        "eligible": [{
            "index": 0, "frame_key": "g0_f00000002", "frame_id": 2,
            "reference_frame_id": 1, "label": "clip/masks/g0_f00000002.png",
        }],
    }), encoding="utf-8")
    return replay, labels, filter_path


def _reference_depth(tmp_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(tmp_path / "replay" / "g0_f00000001_depth.npz") as archive:
        return archive["depth_m"], archive["validity"]


def _selected_depth(tmp_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(tmp_path / "replay" / "g0_f00000002_depth.npz") as archive:
        return archive["depth_m"], archive["validity"]


def _reference_pose(tmp_path: Path) -> list[dict]:
    return json.loads((tmp_path / "replay" / "g0_f00000001_mask_pose.json").read_text(encoding="utf-8"))["joints"]


def _selected_pose(tmp_path: Path) -> list[dict]:
    return json.loads((tmp_path / "replay" / "g0_f00000002_mask_pose.json").read_text(encoding="utf-8"))["joints"]


def _reference_rgb(tmp_path: Path) -> np.ndarray:
    bgr = cv2.imread(str(tmp_path / "replay" / "g0_f00000001_rgb.png"), cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _selected_rgb(tmp_path: Path) -> np.ndarray:
    bgr = cv2.imread(str(tmp_path / "replay" / "g0_f00000002_rgb.png"), cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _flow(tmp_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(tmp_path / "replay" / "g0_f00000002_flow.npz") as archive:
        return archive["flow"], archive["valid_mask"]


def _baseline(tmp_path: Path) -> np.ndarray:
    mask = cv2.imread(
        str(tmp_path / "replay" / "g0_f00000002_baseline_mask.png"), cv2.IMREAD_GRAYSCALE
    )
    return mask >= 128


def _handmade_input(tmp_path: Path) -> torch.Tensor:
    reference_depth, reference_validity = _reference_depth(tmp_path)
    selected_depth, selected_validity = _selected_depth(tmp_path)
    flow, flow_valid = _flow(tmp_path)
    return build_research_input(
        reference_rgb=_reference_rgb(tmp_path),
        selected_rgb=_selected_rgb(tmp_path),
        reference_depth=reference_depth,
        reference_validity=reference_validity,
        selected_depth=selected_depth,
        selected_validity=selected_validity,
        flow=flow,
        flow_valid=flow_valid,
        reference_joints=_reference_pose(tmp_path),
        selected_joints=_selected_pose(tmp_path),
        reference_frame_id=1,
        selected_frame_id=2,
        baseline_mask=_baseline(tmp_path),
    )


def _checkpoint(
    tmp_path: Path,
    *,
    name: str = "candidate.pt",
    trained_heads: list[str] | None = None,
    mask_mode: str = "bounded_residual",
    gate_radius_px: int = PERMIT_RADIUS_PX,
    base_channels: int = 4,
) -> Path:
    path = tmp_path / name
    torch.manual_seed(0)
    model = PairedMaskMotionNet(
        base_channels=base_channels, mask_mode=mask_mode, gate_radius_px=gate_radius_px
    )
    save_checkpoint(
        path,
        task="paired_mask_motion_research",
        model=model,
        model_config={
            "in_channels": 15,
            "base_channels": base_channels,
            "mask_mode": mask_mode,
            "gate_radius_px": gate_radius_px,
            "residual_scale": 4.0,
            "trained_heads": ["mask"] if trained_heads is None else trained_heads,
            "session_id": "train01",
            "clip_sha256": "abc",
        },
        epoch=1,
        global_step=1,
    )
    return path


class _Landmark:
    def __init__(self, x: float, y: float, visibility: float) -> None:
        self.x = x
        self.y = y
        self.visibility = visibility


class _Pose:
    def __init__(self, landmarks: list[_Landmark]) -> None:
        self.landmarks = landmarks


class _DepthFrame:
    def __init__(self, depth: np.ndarray) -> None:
        self.depth = depth
        self.valid = np.isfinite(depth).astype(np.float32)


class _Result:
    def __init__(self, depth: np.ndarray, *, detected: bool = True) -> None:
        self.depth = _DepthFrame(depth)
        self.pose = _Pose([
            _Landmark(0.3, 0.4, 0.95),
            _Landmark(0.6, 0.5, 0.9),
            _Landmark(0.45, 0.7, 0.8),
        ])
        self.joint_depth = np.array([2.0, 2.2, 2.4], dtype=np.float64)
        self.detected = detected
        self.depth_source = "stub"


class _Packet:
    def __init__(self, frame_id: int, rgb: np.ndarray, timestamp_ns: int | None = None) -> None:
        self.frame_id = int(frame_id)
        self.source_id = "stub"
        self.capture_timestamp_ns = (
            int(timestamp_ns) if timestamp_ns is not None else 1_000_000_000 + frame_id * 1_000_000
        )
        self.rgb = rgb


class _RunnerStub:
    """Return a fixed candidate mask and return a logit plane as given."""

    def __init__(self, candidate_mask: np.ndarray, *, logits_maker=None) -> None:
        self.candidate_mask = candidate_mask
        self.model_config = {"mask_mode": "bounded_residual"}
        self.calls = 0
        self._logits_maker = logits_maker

    def predict_for_image(self, inputs, image_shape):
        self.calls += 1
        shape = (int(image_shape[0]), int(image_shape[1]))
        if self._logits_maker is None:
            logits = np.where(self.candidate_mask, 2.0, -2.0).astype(np.float32)
        else:
            logits = self._logits_maker(shape).astype(np.float32)
        return np.zeros((180, 320), bool), logits, logits >= 0


class _RaisingRunner:
    model_config = {"mask_mode": "bounded_residual"}

    def predict_for_image(self, inputs, image_shape):
        raise RuntimeError("model exploded")


class _Flow:
    def __init__(self, flow: np.ndarray) -> None:
        self.status = "ok"
        self.flow = flow
        self.valid_mask = np.ones(flow.shape[:2], dtype=bool)
        self.flow_width = flow.shape[1]
        self.flow_height = flow.shape[0]


class _FlowStub:
    """Return a fixed flow for every pair of frames."""

    def __init__(self, square_pixels: float = 0.0) -> None:
        self.pixels = float(square_pixels)
        self.calls = 0

    def estimate(self, previous, current) -> _Flow:
        self.calls += 1
        shape = (int(current.rgb.shape[0]), int(current.rgb.shape[1]))
        flow = np.zeros((*shape, 2), np.float32)
        flow[..., 0] = self.pixels
        return _Flow(flow)


def default_true_plane(size: int = 16, value: float = 2.0) -> np.ndarray:
    """Return one full depth plane."""
    return np.full((size, size), value, np.float32)


def _scene_depth(size: int = 32, person_value: float = 2.0) -> np.ndarray:
    """A wall at 4 m with a person box at the image centre."""
    depth = np.full((size, size), 4.0, np.float32)
    low = size // 4
    high = size - low
    depth[low:high, low:high] = np.float32(person_value)
    return depth


def _scene_rgb(size: int = 32) -> np.ndarray:
    rgb = np.zeros((size, size, 3), np.uint8)
    rgb[:, :] = 40
    low = size // 4
    high = size - low
    rgb[low:high, low:high] = 200
    return rgb


# --------------------------------------------------------------------------
# 1. The live path and the training path agree
# --------------------------------------------------------------------------


def test_research_input_matches_replay_loader(tmp_path):
    replay, labels, filter_path = _replay_fixture(tmp_path)
    dataset = PairedReplayDataset(replay, labels, filter_path)
    loader_inputs, target = dataset[0]
    handmade = _handmade_input(tmp_path)
    assert handmade.shape == (15, 180, 320)
    assert handmade.dtype == torch.float32
    assert torch.equal(handmade, loader_inputs)
    # The last channel is the baseline mask and it matches the checked target.
    assert torch.equal(handmade[14:15], target)
    # The invalid selected depth sample is zero, and so is its validity.
    assert handmade[7, 0, 0] == 0
    assert handmade[9, 0, 0] == 0
    # The pose floor keeps two of the three selected joints.
    assert float(handmade[13].max()) == pytest.approx(0.9)
    assert int((handmade[13] > 0).sum()) > 0


# --------------------------------------------------------------------------
# 2. Input checks
# --------------------------------------------------------------------------


def _valid_arguments(tmp_path: Path) -> dict:
    reference_depth, reference_validity = _reference_depth(tmp_path)
    selected_depth, selected_validity = _selected_depth(tmp_path)
    flow, flow_valid = _flow(tmp_path)
    return {
        "reference_rgb": _reference_rgb(tmp_path),
        "selected_rgb": _selected_rgb(tmp_path),
        "reference_depth": reference_depth,
        "reference_validity": reference_validity,
        "selected_depth": selected_depth,
        "selected_validity": selected_validity,
        "flow": flow,
        "flow_valid": flow_valid,
        "reference_joints": _reference_pose(tmp_path),
        "selected_joints": _selected_pose(tmp_path),
        "reference_frame_id": 1,
        "selected_frame_id": 2,
        "baseline_mask": _baseline(tmp_path),
    }


def test_research_input_rejects_bad_flow_shape(tmp_path):
    _replay_fixture(tmp_path)
    arguments = _valid_arguments(tmp_path)
    arguments["flow"] = arguments["flow"][..., :1]
    with pytest.raises(ValueError, match="flow must have shape"):
        build_research_input(**arguments)


def test_research_input_rejects_non_uint8_rgb(tmp_path):
    _replay_fixture(tmp_path)
    arguments = _valid_arguments(tmp_path)
    arguments["selected_rgb"] = arguments["selected_rgb"].astype(np.float32) / 255.0
    with pytest.raises(ValueError, match="selected_rgb must be uint8"):
        build_research_input(**arguments)


def test_research_input_rejects_non_finite_depth(tmp_path):
    _replay_fixture(tmp_path)
    arguments = _valid_arguments(tmp_path)
    arguments["selected_depth"] = arguments["selected_depth"].copy()
    arguments["selected_depth"][3, 3] = np.inf
    with pytest.raises(ValueError, match="selected_depth must be finite"):
        build_research_input(**arguments)
    # A NaN depth with a valid flag above 0.5 is a contradiction.
    arguments["selected_depth"][3, 3] = np.nan
    arguments["selected_validity"] = arguments["selected_validity"].copy()
    arguments["selected_validity"][3, 3] = 1.0
    with pytest.raises(ValueError, match="selected_depth must be finite"):
        build_research_input(**arguments)
    # A NaN depth with a valid flag of zero is the recorded invalid sample form.
    arguments["selected_validity"][3, 3] = 0.0
    tensor = build_research_input(**arguments)
    assert tensor.shape == (15, 180, 320)
    assert torch.isfinite(tensor).all()


def test_research_input_rejects_equal_frame_ids(tmp_path):
    _replay_fixture(tmp_path)
    arguments = _valid_arguments(tmp_path)
    arguments["reference_frame_id"] = 2
    with pytest.raises(ValueError, match="must differ"):
        build_research_input(**arguments)


# --------------------------------------------------------------------------
# 3. Runner construction checks
# --------------------------------------------------------------------------


def test_runner_rejects_wrong_sha256(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    expected = "0" * 64
    with pytest.raises(ValueError) as error:
        ShadowMaskRunner(checkpoint=checkpoint, device="cpu", expected_sha256=expected)
    text = str(error.value)
    assert expected in text
    actual = file_sha256(checkpoint)
    assert actual in text
    assert "SHA-256 mismatch" in text


def test_runner_accepts_correct_sha256(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    digest = file_sha256(checkpoint)
    runner = ShadowMaskRunner(checkpoint=checkpoint, device="cpu", expected_sha256=digest)
    assert runner.checkpoint_sha256 == digest
    assert runner.device == "cpu"
    assert runner.model_config["mask_mode"] == "bounded_residual"
    # The configuration is a copy.
    runner.model_config["mask_mode"] = "changed"
    assert runner.model_config["mask_mode"] == "bounded_residual"


def test_runner_rejects_checkpoint_without_mask_head(tmp_path):
    checkpoint = _checkpoint(tmp_path, trained_heads=["motion"])
    with pytest.raises(ValueError, match="trained mask head"):
        ShadowMaskRunner(checkpoint=checkpoint, device="cpu")


# --------------------------------------------------------------------------
# 4. Prediction rules
# --------------------------------------------------------------------------


def test_runner_predict_returns_bool_mask_and_source_logits(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    runner = ShadowMaskRunner(checkpoint=checkpoint, device="cpu")
    _replay_fixture(tmp_path)
    inputs = _handmade_input(tmp_path)
    research_mask, source_logits = runner.predict(inputs)
    assert research_mask.shape == (180, 320)
    assert research_mask.dtype == np.bool_
    assert source_logits.shape == (180, 320)
    assert source_logits.dtype == np.float32
    model = runner.model
    with torch.inference_mode():
        raw_logits, _motion = model(inputs.unsqueeze(0))
    expected_mask = (torch.sigmoid(raw_logits)[0, 0] >= 0.5).numpy()
    expected_logits = F.interpolate(
        raw_logits, size=(180, 320), mode="bilinear", align_corners=False
    )[0, 0].numpy()
    assert np.array_equal(research_mask, expected_mask)
    assert np.array_equal(source_logits, expected_logits)
    # Source mask follows the fixed rule.
    _mask, _logits, source_mask = runner.predict_for_image(inputs, (180, 320))
    assert np.array_equal(source_mask, expected_logits >= 0)


def test_runner_source_upsampling_rule_at_image_size(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    runner = ShadowMaskRunner(checkpoint=checkpoint, device="cpu")
    _replay_fixture(tmp_path)
    inputs = _handmade_input(tmp_path)
    image_shape = (64, 48)
    _research, source_logits, source_mask = runner.predict_for_image(inputs, image_shape)
    assert source_logits.shape == image_shape
    model = runner.model
    with torch.inference_mode():
        raw_logits, _motion = model(inputs.unsqueeze(0))
    expected = F.interpolate(
        raw_logits, size=image_shape, mode="bilinear", align_corners=False
    )[0, 0].numpy()
    assert np.array_equal(source_logits, expected)
    assert np.array_equal(source_mask, expected >= 0)


# --------------------------------------------------------------------------
# 5. The restriction matches the model
# --------------------------------------------------------------------------


def _model_permits(model: torch.nn.Module, baseline_research: torch.Tensor) -> np.ndarray:
    """Return the pixels that can be positive for this research baseline mask."""
    inputs = torch.zeros((1, 15, 180, 320))
    inputs[:, 14] = baseline_research.float()
    with torch.no_grad():
        model.mask_head.bias.fill_(20.0)
        model.mask_head.weight.fill_(0.0)
    with torch.inference_mode():
        logits, _motion = model(inputs)
    return (logits[0, 0] > 0).numpy()


def test_restriction_equals_model_window(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    runner = ShadowMaskRunner(checkpoint=checkpoint, device="cpu")
    assert PERMIT_KERNEL_PX == 2 * PERMIT_RADIUS_PX + 1
    rng = np.random.default_rng(11)
    baseline = np.zeros((180, 320), np.uint8)
    baseline[80:100, 150:170] = 1
    baseline[20:24, 30:38] = 1
    baseline[rng.integers(0, 180, 12), rng.integers(0, 320, 12)] = 1
    baseline_research = torch.from_numpy(baseline.astype(np.float32))
    permits = _model_permits(runner.model, baseline_research)
    mine = research_permitted_region(baseline >= 1)
    extra = int(np.logical_and(permits, ~mine).sum())
    missing = int(np.logical_and(mine, ~permits).sum())
    # No pixel outside my region can become positive.
    assert extra == 0, f"{extra} model-permitted pixels are outside the restriction"
    # And every pixel I permit can become positive.
    assert missing == 0, f"{missing} restricted pixels cannot become positive"
    assert np.array_equal(permits, mine)
    # A pixel far from the baseline stays background.
    assert not permits[0, 0]
    assert permits[95, 140]
    assert permits[12, 34]  # within 12 research pixels of the seed at (20, 30..37)
    assert not permits[7, 34]  # 13 research pixels away from that seed


def test_elliptical_dilation_leaves_corner_gaps():
    """The elliptical kernel is smaller than the square window of the model."""
    seed = np.zeros((180, 320), bool)
    seed[90, 160] = True
    research_seed = cv2.resize(
        seed.astype(np.uint8) * 255, (320, 180), interpolation=cv2.INTER_NEAREST
    ) > 127
    elliptical = dilate_baseline_ellipse(research_seed)
    square = research_permitted_region(seed)
    # The square window of the model keeps the diagonal neighbours of the seed.
    diagonal = (79, 149)
    assert square[diagonal]
    # The elliptical kernel does not keep that diagonal neighbour.
    assert not elliptical[diagonal]
    assert elliptical.sum() < square.sum()
    # The elliptical region is a subset of the square region.
    assert not np.logical_and(elliptical, ~square).any()


def test_source_region_is_research_region_at_image_size():
    baseline = np.zeros((180, 320), bool)
    baseline[90:100, 100:130] = True
    source = source_permitted_region(baseline)
    research = research_permitted_region(baseline)
    assert np.array_equal(source, research)
    small = np.zeros((72, 40), bool)
    small[20:26, 10:20] = True
    source_small = source_permitted_region(small)
    assert source_small.shape == (72, 40)
    assert source_small.sum() > 0
    assert not source_small[0, 0]


# --------------------------------------------------------------------------
# 6. The pipeline
# --------------------------------------------------------------------------


def _pipeline_inputs(tmp_path: Path):
    _replay_fixture(tmp_path)
    baseline = np.zeros((32, 32), bool)
    baseline[8:24, 8:24] = True
    return baseline


def test_pipeline_warms_up_and_records_skip(tmp_path):
    candidate = _pipeline_inputs(tmp_path)
    pipeline = ShadowMaskPipeline(_RunnerStub(candidate), min_images=2, flow_estimator=_FlowStub(), log=lambda _text: None)
    depth = _scene_depth()
    record = pipeline.poll(_Packet(1, _scene_rgb()), _Result(depth))
    assert record["shadow_status"] == "skipped"
    assert record["skip_reason"] == "warming_up"
    assert record["frame_id"] == 1
    assert record["schema_version"] == "1"
    assert record["kind"] == "live_shadow_mask"
    assert record["candidate"]["threshold"] == 0.5
    assert record["candidate"]["source_upsample_rule"] == "bilinear logits, align_corners=False, threshold zero"
    assert record["timings_ms"]["total"] >= 0.0
    counters = pipeline.counters
    assert counters["samples"] == 1
    assert counters["skipped"] == 1
    assert counters["skip_reasons"] == {"warming_up": 1}
    # JSON round-trip keeps the record serializable.
    json.dumps(record, allow_nan=False)


def test_pipeline_records_model_failure_without_raising(tmp_path):
    _pipeline_inputs(tmp_path)
    pipeline = ShadowMaskPipeline(
        _RaisingRunner(), min_images=2, flow_estimator=_FlowStub(), log=lambda _text: None
    )
    depth = _scene_depth()
    first = pipeline.poll(_Packet(1, _scene_rgb()), _Result(depth))
    assert first["shadow_status"] == "skipped"
    record = pipeline.poll(_Packet(2, _scene_rgb()), _Result(depth))
    assert record["shadow_status"] == "failed"
    assert record["error_type"] == "RuntimeError"
    assert "model exploded" in record["error"]
    assert record["skip_reason"] is None
    counters = pipeline.counters
    assert counters["failed"] == 1
    assert counters["error_types"] == {"RuntimeError": 1}
    json.dumps(record, allow_nan=False)


def test_pipeline_reports_no_pose_as_skip(tmp_path):
    candidate = _pipeline_inputs(tmp_path)
    pipeline = ShadowMaskPipeline(
        _RunnerStub(candidate), min_images=2, flow_estimator=_FlowStub(), log=lambda _text: None
    )
    depth = _scene_depth()
    pipeline.poll(_Packet(1, _scene_rgb()), _Result(depth))
    record = pipeline.poll(_Packet(2, _scene_rgb()), _Result(depth, detected=False))
    assert record["shadow_status"] == "skipped"
    assert record["skip_reason"] == "no_pose"
    assert pipeline.window_size == 0
    # A generation change also clears the window.
    pipeline.poll(_Packet(3, _scene_rgb()), _Result(depth))
    assert pipeline.window_size == 1
    record = pipeline.poll(_Packet(4, _scene_rgb()), _Result(depth), generation=3)
    assert pipeline.generation == 3
    assert record["generation"] == 3
    assert record["skip_reason"] == "warming_up"
    assert pipeline.window_size == 1


def test_pipeline_ok_record_is_self_consistent(tmp_path):
    candidate = _pipeline_inputs(tmp_path)
    candidate_pixels = int(candidate.sum())
    runner = _RunnerStub(candidate)
    pipeline = ShadowMaskPipeline(runner, min_images=2, flow_estimator=_FlowStub(), log=lambda _text: None)
    depth = _scene_depth()
    rgb = _scene_rgb()
    first = pipeline.poll(_Packet(10, rgb), _Result(depth))
    assert first["shadow_status"] == "skipped"
    record = pipeline.poll(_Packet(11, rgb), _Result(depth))
    assert record["shadow_status"] == "ok", record
    assert record["skip_reason"] is None
    assert record["error"] is None
    assert record["reference_frame_id"] == 10
    assert record["reference_timestamp_ns"] is not None
    assert record["window_frames"] == 2
    assert record["background"]["frames_used"] == 2
    assert record["background"]["frames_dropped"] == 0
    assert record["flow_score"] == 0.0
    assert runner.calls == 1

    accepted = record["accepted"]
    assert accepted["source"] in ("pose_gradient", "depth_threshold_fallback")
    assert accepted["pixels"] >= 0
    assert abs(accepted["area_fraction"] - accepted["pixels"] / (32 * 32)) < 1e-9
    accepted_pixels = accepted["pixels"]

    shadow = record["candidate"]
    compare = record["compare"]
    assert shadow["baseline_restricted"] is True
    assert shadow["pixels"] == compare["intersection"] + compare["added_pixels"]
    assert accepted["pixels"] == compare["intersection"] + compare["removed_pixels"]
    assert abs(shadow["area_fraction"] - shadow["pixels"] / (32 * 32)) < 1e-9
    assert compare["union"] == compare["intersection"] + compare["added_pixels"] + compare["removed_pixels"]
    if compare["union"]:
        assert compare["iou"] == pytest.approx(compare["intersection"] / compare["union"])
    # Added and removed counts must add up to the size change.
    assert compare["added_fraction_of_accepted"] is None or compare["added_fraction_of_accepted"] >= 0
    assert compare["removed_fraction_of_accepted"] is None or compare["removed_fraction_of_accepted"] >= 0
    if accepted_pixels:
        assert compare["removed_fraction_of_accepted"] == pytest.approx(
            compare["removed_pixels"] / accepted_pixels
        )
    assert record["timings_ms"]["total"] >= record["timings_ms"]["model"] >= 0.0
    assert pipeline.counters["ok"] == 1
    assert pipeline.counters["samples"] == 2
    json.dumps(record, allow_nan=False)


def test_pipeline_reference_requires_small_flow(tmp_path):
    candidate = _pipeline_inputs(tmp_path)
    pipeline = ShadowMaskPipeline(
        _RunnerStub(candidate), min_images=3, flow_estimator=_FlowStub(square_pixels=40.0),
        log=lambda _text: None,
    )
    depth = _scene_depth()
    rgb = _scene_rgb()
    records = [pipeline.poll(_Packet(frame, rgb), _Result(depth)) for frame in range(3)]
    assert [item["skip_reason"] for item in records] == ["warming_up", "warming_up", None]
    assert records[2]["shadow_status"] == "ok", records[2]
    record = pipeline.poll(_Packet(3, rgb), _Result(depth))
    # All window frames move, so the first frame that passes is the reference.
    # A window of three frames checks two candidates. When both fail, the oldest
    # window frame is the reference.
    assert record["shadow_status"] == "ok", record
    assert record["reference_frame_id"] == 1
    assert record["flow_score"] > 0.02

    # The search starts at the newest candidate. The oldest window frame is the
    # reference only when every candidate fails.
    strict = ShadowMaskPipeline(
        _RunnerStub(candidate), min_images=3,
        flow_estimator=_FlowStub(square_pixels=40.0), reference_threshold=0.02, log=lambda _text: None,
    )
    for frame_id in range(4):
        strict.poll(_Packet(frame_id, rgb), _Result(depth))
    record = strict.poll(_Packet(4, rgb), _Result(depth))
    assert record["shadow_status"] == "ok", record
    # The window holds frames 1, 2, 3. Both candidates fail, so the reference is
    # the oldest frame in the window, frame 1.
    assert [frame.frame_id for frame in strict._window] == [2, 3, 4]
    assert record["reference_frame_id"] == 2
    # The record reports the flow score of the chosen reference, which is above
    # the threshold in this case.
    assert record["flow_score"] > 0.02

    # A zero flow passes the test. The newest candidate is the reference.
    stepped = ShadowMaskPipeline(
        _RunnerStub(candidate), min_images=3,
        flow_estimator=_FlowStub(square_pixels=0.0), reference_threshold=0.02, log=lambda _text: None,
    )
    for frame_id in range(4):
        stepped.poll(_Packet(frame_id, rgb), _Result(depth))
    record = stepped.poll(_Packet(4, rgb), _Result(depth))
    assert record["shadow_status"] == "ok", record
    assert record["reference_frame_id"] == 3
    assert record["flow_score"] == 0.0

    # With min_images=2, the first sample that fills the window already has a
    # reference. The second sample then produces an ok record.
    assert stepped.counters["samples"] == 5
    assert stepped.counters["skipped"] == 2
    assert stepped.counters["ok"] == 3

    # A short window uses the oldest window frame when the only candidate does
    # not pass the test. The reference of that sample is not the selected frame,
    # because the poll adds the selected frame before the search.
    short = ShadowMaskPipeline(
        _RunnerStub(candidate), min_images=2,
        flow_estimator=_FlowStub(square_pixels=40.0), reference_threshold=0.02, log=lambda _text: None,
    )
    short.poll(_Packet(1, rgb), _Result(depth))
    record = short.poll(_Packet(2, rgb), _Result(depth))
    assert record["shadow_status"] == "ok", record
    assert record["reference_frame_id"] == 1
    assert record["flow_score"] > 0.02

    # The reference search uses the flow statistic of the live window selector:
    # the mean magnitude over the valid pixels divided by the image diagonal.
    # A 32 x 32 image has a diagonal of about 45.25 pixels.
    selector = ShadowMaskPipeline(
        _RunnerStub(candidate), min_images=2, flow_estimator=_FlowStub(square_pixels=9.05),
        log=lambda _text: None,
    )
    selector.poll(_Packet(0, rgb), _Result(depth))
    selector.poll(_Packet(1, rgb), _Result(depth))
    score = selector.flow_score(selector._estimate(selector._window[0], selector._window[1]))
    assert score == pytest.approx(9.05 / float(np.hypot(32, 32)))
    # A flow result with no valid pixel cannot be used as a reference.
    broken = _Flow(np.zeros((32, 32, 2), np.float32))
    broken.valid_mask = np.zeros((32, 32), bool)
    assert selector.flow_score(broken) is None
    assert selector.flow_score(None) is None
    assert selector._reference_score(broken) == float("inf")


def test_pipeline_compares_masks(tmp_path):
    accepted = np.zeros((8, 8), bool)
    accepted[2:6, 2:6] = True
    candidate = np.zeros((8, 8), bool)
    candidate[2:6, 2:6] = True
    candidate[0, 0] = True
    candidate[7, 3] = True
    compare = compare_masks(candidate, accepted)
    assert compare["intersection"] == 16
    assert compare["union"] == 18
    assert compare["added_pixels"] == 2
    assert compare["removed_pixels"] == 0
    assert compare["iou"] == pytest.approx(16 / 18)
    assert compare["added_fraction_of_accepted"] == pytest.approx(2 / 16)
    assert compare["removed_fraction_of_accepted"] == 0.0
    assert compare["net_area_change_fraction"] == pytest.approx(2 / 16)
    assert compare["shrink_ratio"] == 0.0
    empty = compare_masks(np.zeros((8, 8), bool), np.zeros((8, 8), bool))
    assert empty["iou"] is None


def test_provisional_background_is_a_median(tmp_path):
    stack = [np.full((16, 16), value, np.float32) for value in (1.0, 2.0, 3.0)]
    stack[1][0, 0] = np.nan
    stack[2][0, 0] = np.nan
    baseline, info = provisional_median_background(stack)
    assert baseline.dtype == np.float32
    assert float(baseline[5, 5]) == 2.0
    # Two of three planes hold no usable depth at (0, 0), which is at least half
    # of the planes. The baseline is NaN there.
    assert np.isnan(baseline[0, 0])
    # The interior pixels are usable in every plane.
    assert float(baseline[0, 8]) == 2.0
    assert info["frames_used"] == 3
    assert info["frames_dropped"] == 0
    assert 0.0 < info["valid_fraction"] <= 1.0
    # One plane with no usable border depth is dropped.
    dropped = default_true_plane(size=64)
    dropped[:, :60] = np.nan
    _baseline, dropped_info = provisional_median_background([default_true_plane(size=64) for _ in range(3)] + [dropped])
    assert dropped_info["frames_dropped"] == 1
    assert dropped_info["frames_used"] == 3
    with pytest.raises(ValueError):
        provisional_median_background([])
    with pytest.raises(ValueError):
        provisional_median_background([stack[0], np.ones((8, 8), np.float32)])
    with pytest.raises(ValueError):
        provisional_median_background([np.full((4, 4), np.nan, np.float32)])


# --------------------------------------------------------------------------
# 7. Records, summaries, and percentiles
# --------------------------------------------------------------------------


def _sample_record(status: str, **extra) -> dict:
    record = {
        "schema_version": "1",
        "kind": "live_shadow_mask",
        "generation": 0,
        "frame_id": 1,
        "source_id": "stub",
        "capture_timestamp_ns": "1000",
        "reference_frame_id": 0,
        "reference_timestamp_ns": "900",
        "shadow_status": status,
        "skip_reason": None,
        "error": None,
        "error_type": None,
        "timings_ms": {"data": 1.0, "model": 2.0, "total": 3.0},
        "accepted": {"source": "pose_gradient", "fallback_reason": None, "pixels": 10, "area_fraction": 0.1},
        "candidate": {"mask_mode": "bounded_residual", "threshold": 0.5,
                      "source_upsample_rule": "bilinear logits, align_corners=False, threshold zero",
                      "pixels": 8, "area_fraction": 0.08, "baseline_restricted": True},
        "compare": {"intersection": 6, "union": 12, "iou": 0.5, "added_pixels": 2,
                    "removed_pixels": 4, "added_fraction_of_accepted": 0.2,
                    "removed_fraction_of_accepted": 0.4, "net_area_change_fraction": -0.2,
                    "shrink_ratio": 2.0},
    }
    record.update(extra)
    return record


def test_record_writer_appends_and_reads_back(tmp_path):
    path = tmp_path / "shadow" / "rows.jsonl"
    with ShadowRecordWriter(path, session_id="s1") as writer:
        writer.write(_sample_record("ok"))
    first_size = path.stat().st_size
    with ShadowRecordWriter(path) as writer:
        writer.write(_sample_record("skipped", skip_reason="warming_up"))
        writer.write(_sample_record("failed", error_type="RuntimeError", error="RuntimeError: bad"))
    assert path.stat().st_size > first_size
    records = read_shadow_records(path)
    assert len(records) == 3
    assert records[0]["session_id"] == "s1"
    assert records[1]["session_id"] is None
    assert records[1]["skip_reason"] == "warming_up"
    raw = path.read_bytes()
    assert raw.count(b"\n") == 3
    assert b"\r\n" not in raw
    text = path.read_text(encoding="utf-8")
    assert '"session_id": "s1"' in text
    with pytest.raises(ValueError):
        ShadowRecordWriter(path).write({"shadow_status": float("nan")})


def test_summarize_counts_statuses_and_percentiles():
    records = [
        _sample_record("ok"),
        _sample_record("ok"),
        _sample_record("skipped", skip_reason="warming_up",
                       timings_ms={"data": 0.0, "model": 0.0, "total": 9.0},
                       compare={**_sample_record("ok")["compare"], "iou": None}),
        _sample_record("failed", error_type="ValueError", error="ValueError: bad",
                       skip_reason="warming_up"),
    ]
    summary = summarize_shadow_records(records)
    assert summary["records"] == 4
    assert summary["status_counts"] == {"ok": 2, "skipped": 1, "failed": 1}
    assert summary["skip_reasons"] == {"warming_up": 1}
    assert summary["error_types"] == {"ValueError": 1}
    assert summary["timings_ms_total"]["count"] == 4
    assert summary["timings_ms_total"]["p50"] == 3.0
    # Three records hold an IoU value; the fourth has None.
    assert summary["compare_iou"]["count"] == 3
    assert summary["compare_iou"]["p50"] == 0.5
    assert summary["compare_removed_fraction_of_accepted"]["p50"] == 0.4
    assert summary["compare_shrink_ratio"]["p50"] == 2.0
    empty = summarize_shadow_records([])
    assert empty["records"] == 0
    assert empty["timings_ms_total"] == {"count": 0, "p50": None, "p95": None}
    assert summary["candidate_area_fraction"]["count"] == 4


def test_percentile_uses_nearest_rank():
    # p50 of [10, 20, 30, 40]: rank ceil(0.5 * 4) = 2, so the value is 20.
    assert percentile([10.0, 20.0, 30.0, 40.0], 50.0) == 20.0
    # p95 of three values: rank ceil(0.95 * 3) = 3, so the value is 30.
    assert percentile([10.0, 20.0, 30.0], 95.0) == 30.0
    assert percentile([5.0], 50.0) == 5.0
    assert percentile([1.0, 2.0, 3.0], 0.0) == 1.0
    assert percentile([1.0, 2.0, 3.0], 100.0) == 3.0
    assert percentile([], 50.0) is None
    with pytest.raises(ValueError):
        percentile([1.0], 101.0)


def test_describe_failure_and_window_frame_helper():
    assert describe_failure(RuntimeError("a  b")) == "RuntimeError: a b"
    assert describe_failure(RuntimeError("")) == "RuntimeError"
    frame = _WindowFrame(
        frame_id=3, capture_timestamp_ns=9, source_id="s",
        rgb=np.zeros((4, 4, 3), np.uint8), depth=np.zeros((4, 4), np.float32),
        valid=np.zeros((4, 4), np.float32), joints=[],
    )
    assert frame.frame_id == 3

def test_process_writer_drops_when_busy_and_writes_through_child(tmp_path):
    """The child process writes records, and the parent never blocks on it."""
    from modules.inference.paired_research.shadow import ShadowProcessWriter

    checkpoint = _checkpoint(tmp_path, base_channels=4)
    log_path = tmp_path / "shadow_out" / "run.jsonl"
    writer = ShadowProcessWriter(
        checkpoint=checkpoint,
        log_path=log_path,
        device="cpu",
        min_images=2,
        session_id="proc01",
    )
    try:
        # A submit with no packet data is rejected without raising.
        import types

        empty = types.SimpleNamespace(rgb=None, frame_id=1, source_id="s", capture_timestamp_ns=1)
        outcome = types.SimpleNamespace(depth_m=np.zeros((8, 8), np.float32), mask=None, mask_pose_joints=[])
        assert writer.submit(empty, outcome, generation=0) is False
        counters = writer.counters
        assert counters["child_alive"] is True
        assert counters["writes"] == 0
        assert counters["rejected_requests"] == 1
    finally:
        final = writer.close()
    assert final["child_alive"] is False
    assert final["child_errors"] == 0


def test_process_writer_rejects_a_bad_checkpoint_digest(tmp_path):
    import types
    from modules.inference.paired_research.shadow import ShadowProcessWriter

    checkpoint = _checkpoint(tmp_path, base_channels=4)
    writer = ShadowProcessWriter(
        checkpoint=checkpoint,
        log_path=tmp_path / "bad.jsonl",
        expected_sha256="0" * 64,
        device="cpu",
        min_images=2,
    )
    try:
        packet = types.SimpleNamespace(
            rgb=np.zeros((8, 8, 3), np.uint8), frame_id=1, source_id="s", capture_timestamp_ns=1
        )
        outcome = types.SimpleNamespace(
            depth_m=np.ones((8, 8), np.float32), mask=None, mask_pose_joints=[]
        )
        writer.submit(packet, outcome, generation=0)
    finally:
        final = writer.close(timeout_s=30.0)
    # The child reports the digest mismatch instead of writing a record.
    assert final["child_errors"] >= 1
    assert final["writes"] == 0
