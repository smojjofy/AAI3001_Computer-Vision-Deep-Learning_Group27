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


def mask_full() -> np.ndarray:
    return np.ones((8, 8), np.uint8)


def mask_with_hole() -> np.ndarray:
    image = np.ones((8, 8), np.uint8)
    image[3:5, 3:5] = 0
    return image


def decode_png(data: bytes) -> np.ndarray:
    import cv2

    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)


def test_mask_response_pairs_mask_and_identity_atomically() -> None:
    # RT-0026 item 1: a reader must never see the mask of one state with the
    # identity of another. Two writers publish a recognisable mask/identity
    # pair while the reader checks the pair.
    import threading

    state = PipelineState()
    state.publish_snapshot(header(frame_id=1), b"a", pose_depth=None, mask=mask_full(), pose_joints=None)
    errors: list[str] = []
    done = threading.Event()

    def writer() -> None:
        frame = 2
        while not done.is_set():
            if frame % 2 == 0:
                state.publish_snapshot(header(frame_id=frame), b"b", pose_depth=None, mask=mask_with_hole(), pose_joints=None)
            else:
                state.publish_snapshot(header(frame_id=frame), b"b", pose_depth=None, mask=mask_full(), pose_joints=None)
            frame += 1

    def reader() -> None:
        for _ in range(300):
            response = state.mask_response()
            if response is None:
                continue
            identity, png = response
            if identity is None:
                continue
            decoded = decode_png(png)
            has_hole = int((decoded < 128).sum()) > 0
            # The pair is valid only when the mask content matches the parity
            # of the frame id that was published with it.
            expected_hole = identity.frame_id % 2 == 0
            if has_hole != expected_hole:
                errors.append(f"frame {identity.frame_id} paired with hole={has_hole}")

    writer_thread = threading.Thread(target=writer)
    reader_thread = threading.Thread(target=reader)
    writer_thread.start()
    reader_thread.start()
    reader_thread.join(10)
    done.set()
    writer_thread.join(10)
    assert not errors, errors[:3]


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
    assert state.mask_response() is not None
    assert state.splat_response() is not None


def test_snapshot_without_pose_clears_the_old_pose() -> None:
    state = PipelineState()
    publish(state)
    state.publish_snapshot(header(frame_id=4), b"data2", pose_depth=None, mask=None, pose_joints=None)
    assert state.get_pose_depth() is None
    assert state.get_pose_joints() is None
    assert state.mask_response() is None


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


def test_clear_active_drops_the_old_run_state() -> None:
    # RT-0030 item 1: a run change must not leave an old snapshot, identity,
    # loss latch, or motion beside the new run generation.
    state = PipelineState()
    publish(state)
    state.publish_motion(motion(9, IDENTITY_FRAME))
    state.note_invalid_person()
    assert state.splat_response() is not None
    assert state.mask_response() is not None
    assert state.get_pose_depth() is not None
    assert state.get_motion() is not None
    assert state.get_diagnostics()["person_lost"] is True

    state.clear_active()

    assert state.splat_response() is None
    assert state.mask_response() is None
    assert state.get_pose_depth() is None
    assert state.get_pose_joints() is None
    assert state.get_active_identity() is None
    assert state.get_motion() is None
    diagnostics = state.get_diagnostics()
    assert diagnostics["person_lost"] is False
    assert diagnostics["person_loss_age_ms"] is None
    assert diagnostics["active_identity"] is None
