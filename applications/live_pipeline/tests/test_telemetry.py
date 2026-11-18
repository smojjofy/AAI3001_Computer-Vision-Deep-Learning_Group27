"""Tests for latency summaries, capture age, and the stage-sum rule."""

from __future__ import annotations

import pytest

from applications.live_pipeline.telemetry import (
    capture_age_ms,
    percentile,
    stage_total,
    summarize_counts,
    summarize_ms,
)
from stream_handler.contracts import TimestampOrigin


def test_percentile_boundaries_and_single_sample() -> None:
    assert percentile([42.0], 0) == 42.0
    assert percentile([42.0], 95) == 42.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 3.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 100) == 4.0
    assert percentile([4.0, 3.0, 2.0, 1.0], 0) == 1.0
    with pytest.raises(ValueError):
        percentile([], 50)
    with pytest.raises(ValueError):
        percentile([1.0], 101)


def test_summarize_ms_skips_missing_and_non_finite() -> None:
    assert summarize_ms([]) is None
    summary = summarize_ms([100.0, None, 200.0, float("nan")])
    assert summary is not None
    assert summary["count"] == 2
    assert summary["p50"] == 100.0
    assert summary["min"] == 100.0
    assert summary["max"] == 200.0


def test_summarize_counts_is_integer_only() -> None:
    assert summarize_counts([]) is None
    summary = summarize_counts([27000, 28000, 30000])
    assert summary is not None
    assert summary["count"] == 3
    assert summary["p50"] == 28000
    assert summary["min"] == 27000
    assert summary["max"] == 30000


def test_capture_age_uses_only_the_decoded_monotonic_clock() -> None:
    now_ns = 10_000_000_000
    assert capture_age_ms(now_ns - 250_000_000, now_ns, TimestampOrigin.DECODE_MONOTONIC) == 250.0
    # A recorded source uses presentation time; the age is undefined.
    assert capture_age_ms(5_000_000, now_ns, TimestampOrigin.SOURCE_PRESENTATION) is None
    assert capture_age_ms(5_000_000, now_ns, None) is None
    # A capture timestamp in the future clamps to zero instead of a negative age.
    assert capture_age_ms(now_ns + 50_000_000, now_ns, TimestampOrigin.DECODE_MONOTONIC) == 0.0


def test_stage_total_is_the_stage_sum() -> None:
    stages = {"pose_depth": 120.0, "mask": 12.0, "construct": 40.0, "pose_preview": 8.0}
    assert stage_total(stages) == 180.0
    assert stage_total({}) == 0.0
