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
