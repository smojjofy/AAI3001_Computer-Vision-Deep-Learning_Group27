"""Capture-run generation guard.

One lock owns the generation value and the decision to publish. A refresh job
carries the generation it was created in. Publication and failure recording
take this lock, so a run change cannot interleave between the generation check
and the state change it authorizes.

This module has no model or camera dependency, so it is unit-testable.
"""

from __future__ import annotations

import threading
from typing import Callable


class RunGuard:
    """Guard a capture-run generation and the actions it authorizes."""

    def __init__(self, generation: int = 0) -> None:
        if type(generation) is not int or generation < 0:
            raise ValueError("generation must be a non-negative integer")
        # Reentrant: a guarded publish callback may read the generation.
        self._lock = threading.RLock()
        self._generation = int(generation)

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    def advance(self) -> int:
        """Start a new capture run and return its generation."""
        with self._lock:
            self._generation += 1
            return self._generation

    def publish_if_current(self, job_generation: int, publish: Callable[[], None]) -> bool:
        """Run ``publish`` only when the job still belongs to the current run.

        The generation cannot change while ``publish`` runs, so a publish is
        either fully in the old run or rejected. Returns True when it ran.
        """
        with self._lock:
            if int(job_generation) != self._generation:
                return False
            publish()
            return True

    def record_failure_if_current(self, job_generation: int, record: Callable[[], None]) -> bool:
        """Run ``record`` only for a failure of the current run.

        A stale "person missing" result from an earlier run must not start the
        invalid-person hold in the new run. Returns True when it ran.
        """
        with self._lock:
            if int(job_generation) != self._generation:
                return False
            record()
            return True


def drain_queue(queue, *, limit: int = 64) -> int:
    """Discard queued packets that may predate a run change.

    ``LatestFrameQueue.get_latest`` removes the newest packet and clears its
    predecessors, so one call empties the queue. The loop and limit guard
    against a producer that adds fresh packets while this runs; the caller
    treats those as new-run input.
    """
    if limit < 1:
        raise ValueError("limit must be positive")
    discarded = 0
    for _ in range(limit):
        packet = queue.get_latest(timeout_s=0)
        if packet is None:
            break
        discarded += 1
    return discarded


class PacketRunTracker:
    """Detect capture-run changes from packet identity.

    The tracker keeps the last accepted packet identity. After a run change the
    caller calls :meth:`reset`, so the next packet becomes the new baseline
    instead of comparing against the old run's packet. Without the reset, every
    later packet would look like another change and be discarded again.
    """

    def __init__(self) -> None:
        self._identity: tuple[str, tuple[int, int], int, str] | None = None

    @staticmethod
    def identity_of(packet) -> tuple[str, tuple[int, int], int, str]:
        return (
            str(packet.source_id),
            (int(packet.width), int(packet.height)),
            int(packet.capture_timestamp_ns),
            str(packet.timestamp_origin),
        )

    def change_reason(self, packet) -> str | None:
        """Return a run-change reason for this packet, or None.

        ``source_or_size`` covers a permanent source or image-size change;
        ``backward_time`` covers a capture clock restart inside one run.
        """
        identity = self.identity_of(packet)
        previous = self._identity
        if previous is None:
            return None
        if identity[0] != previous[0] or identity[1] != previous[1]:
            return "source_or_size"
        if identity[2] < previous[2]:
            return "backward_time"
        if identity[3] != previous[3]:
            return "timestamp_origin"
        return None

    def accept(self, packet) -> None:
        """Record this packet as the baseline for later comparisons."""
        self._identity = self.identity_of(packet)

    def reset(self) -> None:
        """Drop the baseline; the next accepted packet starts it again."""
        self._identity = None

    @property
    def identity(self) -> tuple[str, tuple[int, int], int, str] | None:
        return self._identity


class RunCounters:
    """Counters for the current capture run only.

    A stale job can finish after a run change. Its counts must not appear in the
    current run's diagnostics, so every update checks the generation first. A
    run change resets the counters.
    """

    def __init__(self, generation: int = 0) -> None:
        self._generation = int(generation)
        self.refresh_count = 0
        self.refresh_failures = 0
        self.discarded_stale = 0

    @property
    def generation(self) -> int:
        return self._generation

    def start_run(self, generation: int) -> None:
        self._generation = int(generation)
        self.refresh_count = 0
        self.refresh_failures = 0
        self.discarded_stale = 0

    def record_published(self, generation: int) -> bool:
        if int(generation) != self._generation:
            self.discarded_stale += 1
            return False
        self.refresh_count += 1
        return True

    def record_failure(self, generation: int) -> bool:
        if int(generation) != self._generation:
            self.discarded_stale += 1
            return False
        self.refresh_failures += 1
        return True

    def snapshot(self) -> dict[str, int]:
        return {
            "refresh_count": self.refresh_count,
            "refresh_failures": self.refresh_failures,
            "discarded_stale": self.discarded_stale,
        }
