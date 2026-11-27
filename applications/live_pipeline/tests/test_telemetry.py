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


def test_percentile_nearest_rank_boundaries_and_single_sample() -> None:
    assert percentile([42.0], 0) == 42.0
    assert percentile([42.0], 50) == 42.0
    assert percentile([42.0], 95) == 42.0
    assert percentile([42.0], 100) == 42.0
    # Nearest rank: the rank is ceil(p/100 * n), clamped to [1, n].
    assert percentile([1.0, 2.0, 3.0, 4.0], 0) == 1.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 25) == 1.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 75) == 3.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 95) == 4.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 100) == 4.0
    # Order does not matter.
    assert percentile([4.0, 3.0, 2.0, 1.0], 50) == 2.0
    # An odd count with a single middle value.
    assert percentile([1.0, 2.0, 3.0], 50) == 2.0
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


def test_summarize_ms_even_sample_count_uses_nearest_rank() -> None:
    # An even count must not pick the upper middle sample.
    summary = summarize_ms([1.0, 2.0, 3.0, 4.0])
    assert summary is not None
    assert summary["p50"] == 2.0
    assert summary["p95"] == 4.0
    # Ten samples: p50 is the fifth value, p95 the tenth.
    ten = summarize_ms([float(value) for value in range(1, 11)])
    assert ten is not None
    assert ten["p50"] == 5.0
    assert ten["p95"] == 10.0


def test_summarize_counts_is_integer_only() -> None:
    assert summarize_counts([]) is None
    summary = summarize_counts([27000, 28000, 30000])
    assert summary is not None
    assert summary["count"] == 3
    assert summary["p50"] == 28000
    assert summary["min"] == 27000
    assert summary["max"] == 30000
    # The same nearest-rank rule applies to an even count.
    even = summarize_counts([4, 1, 3, 2])
    assert even is not None
    assert even["p50"] == 2


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
