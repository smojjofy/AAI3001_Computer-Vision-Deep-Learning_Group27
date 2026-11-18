"""Latency and capture-age helpers.

The refresh log reports one latency value per published refresh: the sum of every
processing stage. These helpers keep that arithmetic and its summary in one
testable place. They have no model or camera dependency.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

from stream_handler.contracts import TimestampOrigin


def percentile(values: Sequence[float], percentile_value: float) -> float:
    """Nearest-rank percentile for a non-empty sequence."""
    if not values:
        raise ValueError("percentile needs at least one value")
    if not 0.0 <= percentile_value <= 100.0:
        raise ValueError("percentile must be between 0 and 100")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(percentile_value / 100.0 * (len(ordered) - 1))))
    return float(ordered[index])


def summarize_ms(values: Iterable[float]) -> dict[str, float | int] | None:
    """Summarize latency samples in milliseconds, or None when empty."""
    samples = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not samples:
        return None
    return {
        "count": len(samples),
        "mean": round(sum(samples) / len(samples), 2),
        "p50": round(percentile(samples, 50), 2),
        "p95": round(percentile(samples, 95), 2),
        "min": round(min(samples), 2),
        "max": round(max(samples), 2),
    }


def summarize_counts(values: Iterable[int]) -> dict[str, int] | None:
    """Summarize integer counts such as splats per refresh."""
    samples = [int(value) for value in values]
    if not samples:
        return None
    samples.sort()
    return {
        "count": len(samples),
        "mean": int(round(sum(samples) / len(samples))),
        "p50": samples[len(samples) // 2],
        "min": samples[0],
        "max": samples[-1],
    }


def capture_age_ms(
    capture_timestamp_ns: int,
    now_ns: int,
    origin: TimestampOrigin | str | None,
) -> float | None:
    """Age of a captured frame in milliseconds, or None when the clock differs.

    A live camera stamps packets with the same monotonic clock as this process,
    so the age is meaningful. A recorded source uses its own presentation clock,
    so subtracting it from the process clock would produce a meaningless value.
    """
    if origin is None:
        return None
    origin_value = origin.value if isinstance(origin, TimestampOrigin) else str(origin)
    if origin_value != TimestampOrigin.DECODE_MONOTONIC.value:
        return None
    return max(0.0, (int(now_ns) - int(capture_timestamp_ns)) / 1_000_000.0)


def stage_total(stages: dict[str, float]) -> float:
    """Sum the processing stages; the total must be the stage sum."""
    return float(sum(float(value) for value in stages.values()))
