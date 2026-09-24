from __future__ import annotations

import numpy as np
import pytest
import torch

from modules.inference import ConstructionFrameMetadata, construction_frame_from_estimates
from modules.inference.depth.src.estimator import DepthEstimate
from modules.inference.segmentation.src.estimator import SegmentationEstimate


def _estimates(height: int, width: int) -> tuple[DepthEstimate, SegmentationEstimate]:
    depth = DepthEstimate(
        relative_inverse_depth=torch.full((1, 1, height, width), 0.75),
        confidence=torch.full((1, 1, height, width), 0.8),
    )
    mask = torch.tensor([[[[True, False, True], [False, True, True]]]])
    segmentation = SegmentationEstimate(
        mask=mask[:, :, :height, :width],
        foreground_probability=torch.full((1, 1, height, width), 0.6),
        confidence=torch.full((1, 1, height, width), 0.2),
    )
    return depth, segmentation


def test_constructor_handoff_preserves_metadata_and_contract() -> None:
    rgb = np.full((2, 3, 3), 127, dtype=np.uint8)
    depth, segmentation = _estimates(2, 3)
    metadata = ConstructionFrameMetadata(
        frame_id="0001",
        timestamp_ns=123,
        source_id="0001",
        intrinsics=(2.0, 2.0, 1.0, 0.5),
    )

    frame = construction_frame_from_estimates(rgb, depth, segmentation, metadata)

    assert (frame.frame_id, frame.timestamp_ns, frame.source_id) == ("0001", 123, "0001")
    assert frame.depth.dtype == np.float32
    assert frame.depth_validity is not None
    assert np.allclose(frame.depth_validity, 0.8)
    assert frame.foreground_weight is None
    assert frame.foreground_mask.tolist() == [[1, 0, 1], [0, 1, 1]]


def test_constructor_handoff_rejects_unaligned_estimates() -> None:
    rgb = np.full((2, 3, 3), 127, dtype=np.uint8)
    depth, segmentation = _estimates(2, 2)
    metadata = ConstructionFrameMetadata("0001", 123, "0001", (2.0, 2.0, 1.0, 0.5))

    with pytest.raises(ValueError, match="aligned"):
        construction_frame_from_estimates(rgb, depth, segmentation, metadata)
