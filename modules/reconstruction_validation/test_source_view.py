from __future__ import annotations

import numpy as np
import pytest

from modules.reconstruction_validation import validate_source_view
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


def test_source_view_validation_reprojects_metric_fixture() -> None:
    frame = ConstructionFrame(
        frame_id="0001",
        timestamp_ns=1,
        source_id="0001",
        rgb=np.array(
            [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]],
            dtype=np.uint8,
        ),
        depth=np.array([[2.0, 2.0], [3.0, 3.0]], dtype=np.float32),
        foreground_mask=np.ones((2, 2), dtype=np.uint8),
        intrinsics=(2.0, 2.0, 0.5, 0.5),
        depth_semantics=DepthSemantics.METRIC_CAMERA_Z,
        depth_units=DepthUnits.METERS,
    )
    state = construct_splat_state(frame, ConstructionConfig(sample_stride=1))

    report = validate_source_view(frame, state, ConstructionConfig(sample_stride=1))

    assert report.center_reprojection_rmse_px < 1.0e-5
    assert report.covered_rgb_mae < 1.0e-5
    assert report.covered_depth_mae < 1.0e-5
    assert report.mask_iou == pytest.approx(1.0)
    assert report.foreground_coverage == pytest.approx(1.0)
