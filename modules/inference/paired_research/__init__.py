"""Offline paired-frame research model and replay data loader."""

from .data import PairedReplayDataset
from .model import PairedMaskMotionNet
from .shadow import (
    ShadowMaskPipeline,
    ShadowMaskRunner,
    ShadowRecordWriter,
    build_research_input,
    common_joints_for_mask,
    compare_masks,
    describe_failure,
    percentile,
    read_shadow_records,
    summarize_shadow_records,
)

__all__ = [
    "PairedReplayDataset",
    "PairedMaskMotionNet",
    "ShadowMaskPipeline",
    "ShadowMaskRunner",
    "ShadowRecordWriter",
    "build_research_input",
    "common_joints_for_mask",
    "compare_masks",
    "describe_failure",
    "percentile",
    "read_shadow_records",
    "summarize_shadow_records",
]
