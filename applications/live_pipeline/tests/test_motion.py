"""Tests for the reuse motion preview."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from applications.live_pipeline.motion import build_motion_preview


@dataclass
class FakeCandidate:
    frame_id: int
    flow: object | None
    reference_frame_id: int | None = 1


@dataclass
class FakeFlow:
    status: str
    flow: np.ndarray
    valid_mask: np.ndarray


def candidate(magnitude: float = 10.0, width: int = 80, height: int = 40) -> FakeCandidate:
    flow = np.zeros((height, width, 2), np.float32)
    flow[..., 0] = magnitude
    flow[..., 1] = -magnitude
    return FakeCandidate(frame_id=3, flow=FakeFlow("ok", flow, np.ones((height, width), bool)))


def test_flow_is_converted_to_normalized_units() -> None:
    preview = build_motion_preview(candidate(10.0, 80, 40), joints=None)
    assert preview is not None
    payload = preview.to_payload()
    assert payload["source_width"] == 80
    assert payload["source_height"] == 40
    flow = np.asarray(payload["flow_ndc"], np.float32).reshape(payload["height"], payload["width"], 2)
    # 10 px right of an 80 px image is 2*10/80 = 0.25, clamped to 0.15.
    # 10 px up (image y decreases) is +0.5 after the y flip, clamped to 0.15.
    assert np.allclose(flow[..., 0], 0.15, atol=1e-5)
    assert np.allclose(flow[..., 1], 0.15, atol=1e-5)


def test_invalid_flow_returns_none() -> None:
    bad = FakeCandidate(frame_id=4, flow=None)
    assert build_motion_preview(bad, joints=None) is None
    no_status = FakeCandidate(frame_id=4, flow=FakeFlow("scene_cut", np.zeros((4, 4, 2), np.float32), np.zeros((4, 4), bool)))
    assert build_motion_preview(no_status, joints=None) is None


def test_joints_are_carried() -> None:
    joints = [{"index": 0, "x": 0.5, "y": 0.4, "visibility": 0.9}]
    preview = build_motion_preview(candidate(), joints=joints)
    assert preview is not None
    assert preview.to_payload()["joints"] == joints


def test_motion_preview_preserves_frame_identity() -> None:
    preview = build_motion_preview(candidate(), joints=None)
    assert preview is not None
    payload = preview.to_payload()
    assert payload["frame_id"] == 3
    assert payload["reference_frame_id"] == 1
    assert len(payload["valid"]) == payload["width"] * payload["height"]
