from __future__ import annotations

import numpy as np
import pytest

from modules.splat_constructor import (
    ConstructionConfig,
    ConstructionFrame,
    DepthSemantics,
    DepthUnits,
    construct_splat_state,
)
from modules.splat_constructor.python_bridge import default_library_path


pytestmark = pytest.mark.skipif(
    not default_library_path().is_file(),
    reason="native bridge has not been built; run scripts/build_splat_constructor.ps1",
)


def test_python_bridge_constructs_copied_splat_state() -> None:
    frame = ConstructionFrame(
        frame_id="0001",
        timestamp_ns=123,
        source_id="0001",
        rgb=np.full((2, 2, 3), 127, dtype=np.uint8),
        depth=np.array([[2.0, 0.0], [3.0, 4.0]], dtype=np.float32),
        foreground_mask=np.ones((2, 2), dtype=np.uint8),
        intrinsics=(2.0, 2.0, 0.5, 0.5),
        depth_semantics=DepthSemantics.METRIC_CAMERA_Z,
        depth_units=DepthUnits.METERS,
        depth_validity=np.ones((2, 2), dtype=np.float32),
        foreground_weight=np.full((2, 2), 0.5, dtype=np.float32),
        camera_to_world=np.array(
            [[1.0, 0.0, 0.0, 10.0], [0.0, 1.0, 0.0, 0.0],
             [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
    )

    state = construct_splat_state(frame, ConstructionConfig(sample_stride=1))

    assert state.frame_id == "0001"
    assert state.timestamp_ns == 123
    assert state.source_id == "0001"
    assert state.positions.shape == (3, 3)
    assert state.source_pixels.tolist() == [[0, 0], [0, 1], [1, 1]]
    assert state.local_ids.tolist() == [0, 2, 3]
    assert state.positions[0, 0] == pytest.approx(9.5)
    assert state.positions[0, 2] == pytest.approx(-2.0)
    assert np.allclose(state.reconstruction_weights, 0.5)
    assert np.allclose(np.linalg.norm(state.rotations, axis=1), 1.0)

    frame.rgb[0, 0] = 0
    assert state.sh_dc[0, 0] != pytest.approx((0.0 - 0.5) / 0.28209479177387814)
