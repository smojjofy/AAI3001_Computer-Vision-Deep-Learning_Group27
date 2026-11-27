"""Tests for the integration adapters.

Run with the repository pytest configuration:
    python -m pytest applications/live_pipeline/tests -q
"""

from __future__ import annotations

import numpy as np
import pytest

from applications.live_pipeline.adapters import (
    INVALID_DEPTH_PLACEHOLDER_M,
    TRANSPORT_ARRAYS,
    DepthThresholdConfig,
    construction_frame_from_refresh,
    construct_splat_from_refresh,
    depth_threshold_mask,
    depth_to_metres,
    frame_id_string,
    splat_state_payload,
    webcam_intrinsics,
)
from modules.splat_constructor import DepthSemantics, DepthUnits
from modules.splat_constructor.python_bridge import default_library_path


def synthetic_depth(height: int = 60, width: int = 80) -> np.ndarray:
    """Background at 5 m with a 2.0-2.6 m person block in the centre."""
    depth = np.full((height, width), 5.0, np.float32)
    top, bottom = int(height * 0.25), int(height * 0.85)
    left, right = int(width * 0.32), int(width * 0.68)
    block_height, block_width = bottom - top, right - left
    depth[top:bottom, left:right] = np.linspace(
        2.0, 2.6, block_height * block_width, dtype=np.float32
    ).reshape(block_height, block_width)
    return depth


def test_frame_id_string_preserves_integer_round_trip() -> None:
    assert frame_id_string(0) == "0"
    assert frame_id_string(1234) == "1234"
    with pytest.raises(ValueError):
        frame_id_string(-1)


def test_webcam_intrinsics_uses_constructor_half_pixel_convention() -> None:
    fx, fy, cx, cy = webcam_intrinsics(1280, 720, 60.0)
    assert fx == fy
    assert fx > 0.0
    assert cx == pytest.approx((1280 - 1) / 2.0)
    assert cy == pytest.approx((720 - 1) / 2.0)
    with pytest.raises(ValueError):
        webcam_intrinsics(0, 720)
    with pytest.raises(ValueError):
        webcam_intrinsics(640, 480, 0.5)


def test_depth_to_metres_keeps_validity() -> None:
    depth_mm = np.array([[1000.0, np.nan, 2500.0]], np.float32)
    metres = depth_to_metres(depth_mm, "millimetres")
    assert metres[0, 0] == pytest.approx(1.0)
    assert np.isnan(metres[0, 1])
    assert metres[0, 2] == pytest.approx(2.5)
    with pytest.raises(ValueError):
        depth_to_metres(depth_mm, "furlongs")


def test_depth_threshold_mask_isolates_the_near_region() -> None:
    depth = synthetic_depth()
    result = depth_threshold_mask(depth, DepthThresholdConfig(band_m=0.9, closing_px=3))
    assert result.detected
    assert result.mask.shape == depth.shape
    assert result.mask.dtype == np.uint8
    assert set(np.unique(result.mask)) <= {0, 1}
    # The centre block must be covered, the corners must not be.
    assert result.mask[30, 40] == 1
    assert result.mask[0, 0] == 0
    assert result.area_fraction > 0.05
    assert 0.0 <= result.confidence <= 1.0
    assert np.isfinite(result.near_m) and np.isfinite(result.far_m)
    assert result.near_m < result.far_m


def test_depth_threshold_mask_rejects_blank_depth() -> None:
    depth = np.full((20, 30), np.nan, np.float32)
    result = depth_threshold_mask(depth)
    assert not result.detected
    assert result.mask.sum() == 0
    assert result.confidence == 0.0


def test_depth_threshold_mask_rejects_full_frame_band() -> None:
    # A uniform scene has no separable foreground: the band covers everything.
    depth = np.full((40, 50), 1.5, np.float32)
    result = depth_threshold_mask(depth, DepthThresholdConfig(band_m=0.9))
    assert not result.detected
    assert result.mask.sum() == 0
    assert result.area_fraction > 0.9


def test_hole_fill_does_not_invert_a_corner_touching_region() -> None:
    # A small band that touches the image corner made the old fill
    # return the whole frame. The mask must stay at the component size, and the
    # final area guard must judge the post-fill mask.
    height, width = 40, 50
    depth = np.full((height, width), 5.0, np.float32)
    depth[0:8, 0:8] = 1.2
    result = depth_threshold_mask(
        depth,
        DepthThresholdConfig(
            band_m=0.5,
            near_percentile=0.0,
            closing_px=1,
            fill_holes=True,
            min_area_fraction=0.005,
        ),
    )
    assert int(result.mask.sum()) == 64
    assert result.area_fraction == pytest.approx(64 / (height * width))
    assert result.mask[0, 0] == 1
    assert result.mask[height - 1, width - 1] == 0


def test_hole_fill_keeps_a_normal_person_region() -> None:
    depth = synthetic_depth()
    result = depth_threshold_mask(depth, DepthThresholdConfig(band_m=0.9, closing_px=3))
    assert result.detected
    assert result.area_fraction < 0.60
    assert result.mask[30, 40] == 1
    assert result.mask[0, 0] == 0


def test_hole_fill_adds_enclosed_holes_only() -> None:
    from applications.live_pipeline.adapters import _fill_holes_from_background

    mask = np.zeros((20, 30), np.uint8)
    mask[4:16, 6:24] = 1
    mask[8:12, 12:18] = 0  # enclosed hole
    filled = _fill_holes_from_background(mask)
    assert filled.dtype == np.uint8
    assert set(np.unique(filled)) <= {0, 1}
    assert int(filled.sum()) == int(mask[4:16, 6:24].size)
    assert filled[10, 15] == 1
    assert filled[0, 0] == 0


def test_hole_fill_keeps_a_corner_touching_region_exact() -> None:
    from applications.live_pipeline.adapters import _fill_holes_from_background

    mask = np.zeros((20, 30), np.uint8)
    mask[0:6, 0:6] = 1  # touches the top-left corner
    filled = _fill_holes_from_background(mask)
    assert np.array_equal(filled, mask)


def test_construction_frame_replaces_nan_and_keeps_validity() -> None:
    rgb = np.zeros((60, 80, 3), np.uint8)
    depth = synthetic_depth()
    depth[0, 0] = np.nan
    mask = np.zeros((60, 80), np.uint8)
    mask[15:50, 25:55] = 1
    frame = construction_frame_from_refresh(
        rgb=rgb,
        depth_m=depth,
        foreground_mask=mask,
        frame_id=7,
        timestamp_ns=123456789,
        source_id="webcam:0",
    )
    assert frame.frame_id == "7"
    assert frame.source_id == "webcam:0"
    assert frame.timestamp_ns == 123456789
    assert frame.depth_semantics == DepthSemantics.METRIC_CAMERA_Z
    assert frame.depth_units == DepthUnits.METERS
    assert np.isfinite(frame.depth).all()
    assert frame.depth[0, 0] == pytest.approx(INVALID_DEPTH_PLACEHOLDER_M)
    assert frame.depth_validity[0, 0] == 0.0
    assert frame.depth_validity[30, 40] == 1.0
    assert set(np.unique(frame.foreground_mask)) <= {0, 1}
    assert frame.foreground_mask[30, 40] == 1


def test_construction_frame_rejects_misaligned_inputs() -> None:
    rgb = np.zeros((10, 10, 3), np.uint8)
    with pytest.raises(ValueError):
        construction_frame_from_refresh(
            rgb=rgb,
            depth_m=np.zeros((5, 5), np.float32),
            foreground_mask=np.ones((10, 10), np.uint8),
            frame_id=1,
            timestamp_ns=1,
            source_id="s",
        )
    with pytest.raises(ValueError):
        construction_frame_from_refresh(
            rgb=rgb,
            depth_m=np.ones((10, 10), np.float32),
            foreground_mask=np.ones((5, 5), np.uint8),
            frame_id=1,
            timestamp_ns=1,
            source_id="s",
        )


def test_construction_frame_clamps_zero_replay_timestamp() -> None:
    rgb = np.zeros((60, 80, 3), np.uint8)
    depth = synthetic_depth()
    mask = np.zeros((60, 80), np.uint8)
    mask[15:50, 25:55] = 1
    frame = construction_frame_from_refresh(
        rgb=rgb,
        depth_m=depth,
        foreground_mask=mask,
        frame_id=1,
        timestamp_ns=0,
        source_id="video:replay",
    )
    assert frame.timestamp_ns == 1


def test_splat_state_native_round_trip() -> None:
    if not default_library_path().is_file():
        pytest.skip("native constructor is not built")
    rgb = np.random.default_rng(0).integers(0, 255, (48, 64, 3), dtype=np.uint8)
    depth = synthetic_depth(48, 64)
    mask = np.zeros((48, 64), np.uint8)
    mask[12:40, 20:44] = 1
    state = construct_splat_from_refresh(
        rgb=rgb,
        depth_m=depth,
        foreground_mask=mask,
        frame_id=3,
        timestamp_ns=10,
        source_id="webcam:0",
        intrinsics=webcam_intrinsics(64, 48),
    )
    assert state.positions.shape[0] > 0
    assert state.positions.shape[1] == 3
    assert state.rotations.shape[1] == 4
    header, buffer = splat_state_payload(state)
    assert header["count"] == state.positions.shape[0]
    assert header["frame_id"] == "3"
    assert header["byte_length"] == len(buffer)
    assert header["rotation_order"] == "wxyz"
    offset = 0
    for spec in header["arrays"]:
        assert spec["byte_offset"] == offset
        offset += spec["byte_length"]
        name = dict(((item[0], item[1]) for item in TRANSPORT_ARRAYS))[spec["name"]]
        source = getattr(state, spec["name"]).reshape(-1)
        decoded = np.frombuffer(buffer, dtype=name, count=spec["count"], offset=spec["byte_offset"])
        assert np.array_equal(decoded, source.astype(decoded.dtype))
    assert offset == len(buffer)
