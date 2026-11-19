"""Live refresh-construction test: the real pipeline wiring, not only units.

RT-0064 found that the unit suite passed while the live constructor raised
TypeError, because no test built a RefreshPipeline and ran one refresh through
the real PoseDepthPipeline. This test closes that gap. It needs the real model
files, so it skips when they are absent.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

# `_import_pose_estimator` appends the pose package to sys.path on first use. If
# another suite imported it earlier from a different path, a cached half-built
# package can remain and the model import fails with ModuleNotFoundError. Drop
# any cached pose modules before importing the live pipeline so this test owns
# the import.
for _name in [name for name in list(sys.modules) if name == "pose_estimator" or name.startswith("pose_estimator.")]:
    del sys.modules[_name]

from applications.live_pipeline.pipeline import RefreshError, RefreshPipeline

ROOT = Path(__file__).resolve().parents[3]
DEPTH_MODELS = ROOT / "modules" / "inference" / "depth" / "models"

requires_models = pytest.mark.skipif(
    not DEPTH_MODELS.is_dir(),
    reason="depth and pose model files are required for a live construction test",
)


class WarmPacket:
    """A packet shaped like the stream contract, with a photo-like frame."""

    def __init__(self, width: int = 640, height: int = 480, frame_id: int = 7) -> None:
        self.frame_id = frame_id
        self.source_id = "live-test"
        self.capture_timestamp_ns = 1_000_000_000
        self.intrinsics = None
        self.depth = None
        self.depth_units = None
        rng = np.random.default_rng(3)
        # A smooth gradient with a brighter block is enough to exercise the
        # model path; this test checks wiring and contracts, not detection.
        gradient = np.linspace(20, 220, width, dtype=np.float32)[None, :, None]
        image = np.repeat(gradient, height, axis=0).repeat(3, axis=2)
        image[height // 3 : 2 * height // 3, width // 3 : 2 * width // 3] += 30
        image += rng.normal(0, 2, image.shape).astype(np.float32)
        self.rgb = np.clip(image, 0, 255).astype(np.uint8)

    @property
    def width(self) -> int:
        return int(self.rgb.shape[1])

    @property
    def height(self) -> int:
        return int(self.rgb.shape[0])


@requires_models
def test_live_pipeline_constructs_and_runs_a_refresh() -> None:
    # The defect was that PoseDepthPipeline rejected a keyword the live pipeline
    # passed. Constructing the pipeline and processing a frame proves the whole
    # path is wired, which the unit tests could not.
    pipeline = RefreshPipeline(device="auto", require_pose=False)
    try:
        pipeline._ensure_pipeline()
        assert pipeline._pipeline is not None
        packet = WarmPacket()
        try:
            outcome = pipeline.process(packet, generation=0)
        except RefreshError as error:
            # A synthetic frame has no person, so a person-missing rejection is
            # a valid outcome; the wiring is what this test checks.
            assert "pose" in str(error).lower() or "mask" in str(error).lower(), str(error)
            return
        assert outcome.header["frame_id_int"] == 7
        assert outcome.generation == 0
        assert "timings_ms" in outcome.header
        assert outcome.splat_count >= 0
    finally:
        pipeline.close()


@requires_models
def test_live_pipeline_passes_the_confidence_option_to_the_estimator() -> None:
    # depth_confidence=False must reach the YOLO estimator through
    # PoseDepthPipeline; this is the option that skipped the unused plane.
    quiet = RefreshPipeline(device="auto", require_pose=False, depth_confidence=False)
    loud = RefreshPipeline(device="auto", require_pose=False, depth_confidence=True)
    try:
        quiet_pipe = quiet._ensure_pipeline()
        loud_pipe = loud._ensure_pipeline()
        assert quiet_pipe.compute_confidence is False
        assert loud_pipe.compute_confidence is True
        assert quiet_pipe.depth_estimator.compute_confidence is False
        assert loud_pipe.depth_estimator.compute_confidence is True
    finally:
        quiet.close()
        loud.close()
