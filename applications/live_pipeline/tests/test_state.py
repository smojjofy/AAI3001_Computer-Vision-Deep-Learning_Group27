"""Tests for the active splat state holder: identity, person-loss latch, motion gating."""

from __future__ import annotations

import json
import struct
import time

import numpy as np

from applications.backend.pipeline_server import ActiveStateIdentity, PipelineState


def header(frame_id: int = 3, generation: int = 2, source_id: str = "cam") -> dict:
    """A splat header. The committed reference equals the displayed frame."""
    return {
        "schema_version": "1.0",
        "frame_id": str(frame_id),
        "frame_id_int": frame_id,
        "source_id": source_id,
        "generation": generation,
    }


def mask(value: float = 1.0) -> np.ndarray:
    image = np.zeros((8, 8), np.uint8)
    image[2:6, 2:6] = 1 if value > 0 else 0
    return image


def publish(state: PipelineState, frame_id: int = 3, generation: int = 2) -> None:
    state.publish_snapshot(
        header(frame_id=frame_id, generation=generation),
        b"data",
        pose_depth={"frame_id": frame_id, "generation": generation},
        mask=mask(),
        pose_joints=[{"index": 11, "x": 0.4, "y": 0.3}],
    )


def motion(frame_id: int, reference_frame_id: int, generation: int = 2, source_id: str = "cam") -> dict:
    return {
        "schema_version": "1.0",
        "frame_id": frame_id,
        "reference_frame_id": reference_frame_id,
        "splat_frame_id": reference_frame_id,
        "source_id": source_id,
        "generation": generation,
    }


IDENTITY_FRAME = 3  # the frame id the active identity reports


def test_snapshot_publishes_splat_pose_mask_together() -> None:
    state = PipelineState()
    publish(state)
    identity = state.get_active_identity()
    assert identity == ActiveStateIdentity("cam", 2, IDENTITY_FRAME, IDENTITY_FRAME)
    assert state.get_pose_joints() == [{"index": 11, "x": 0.4, "y": 0.3}]
    assert state.get_pose_depth() is not None
    assert state.get_mask_png() is not None
    assert state.splat_response() is not None


def test_snapshot_without_pose_clears_the_old_pose() -> None:
    state = PipelineState()
    publish(state)
    state.publish_snapshot(header(frame_id=4), b"data2", pose_depth=None, mask=None, pose_joints=None)
    assert state.get_pose_depth() is None
    assert state.get_pose_joints() is None
    assert state.get_mask_png() is None


def test_motion_must_match_the_active_state() -> None:
    state = PipelineState()
    publish(state)
    state.publish_motion(motion(9, IDENTITY_FRAME))
    assert state.get_motion() is not None
    assert state.get_diagnostics()["motion_published"] == 1
    state.publish_motion(motion(9, 4))  # wrong reference
    assert state.get_motion() is None
    assert state.get_diagnostics()["motion_rejected"] == 1
    state.publish_motion(motion(9, IDENTITY_FRAME, generation=1))  # old run
    assert state.get_motion() is None
    state.publish_motion(motion(IDENTITY_FRAME, IDENTITY_FRAME))  # not later
    assert state.get_motion() is None


def test_old_motion_rejected_after_refresh() -> None:
    state = PipelineState()
    publish(state, frame_id=3)
    state.publish_motion(motion(9, 3))
    assert state.get_motion() is not None
    publish(state, frame_id=5)
    # The snapshot clears motion; motion for the previous reference is rejected.
    assert state.get_motion() is None
    state.publish_motion(motion(9, 3))
    assert state.get_motion() is None


def test_generation_change_rejects_old_motion() -> None:
    state = PipelineState()
    publish(state, frame_id=3, generation=0)
    state.publish_motion(motion(9, 3, generation=0))
    assert state.get_motion() is not None
    publish(state, frame_id=4, generation=1)
    state.publish_motion(motion(10, 4, generation=0))
    assert state.get_motion() is None


def test_person_loss_latches_until_success() -> None:
    state = PipelineState()
    publish(state)
    assert state.get_diagnostics()["person_loss_age_ms"] is None
    state.note_invalid_person()
    first = state.get_diagnostics()["person_loss_age_ms"]
    assert first is not None
    time.sleep(0.01)
    state.note_invalid_person()  # repeated failure must not restart the hold
    second = state.get_diagnostics()["person_loss_age_ms"]
    assert second >= first
    publish(state, frame_id=8)
    assert state.get_diagnostics()["person_loss_age_ms"] is None


def test_splat_response_reports_loss_without_forcing_alpha() -> None:
    state = PipelineState()
    assert state.splat_response() is None
    publish(state)
    state.note_invalid_person()
    response = state.splat_response()
    assert response is not None
    length = struct.unpack("<I", response[:4])[0]
    payload = json.loads(response[4 : 4 + length])
    assert payload["person_lost"] is True
    # The browser owns the hold and fade; the server keeps full alpha.
    assert payload["display_alpha"] == 1.0
    assert payload["active_identity"]["frame_id"] == IDENTITY_FRAME
