"""Boundary tests for the refresh dispatcher at a capture-run change."""

from __future__ import annotations

import threading
import time

from applications.live_pipeline.scheduler import RefreshDispatcher


class BlockingWorker:
    """Worker whose first job blocks until released, so a second job stays pending."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.started: list[int] = []
        self.first_started = threading.Event()

    def __call__(self, job) -> None:
        self.started.append(job.packet.frame_id)
        if len(self.started) == 1:
            self.first_started.set()
            self.release.wait(5)


def job(frame_id: int, generation: int = 0):
    from applications.live_pipeline.scheduler import RefreshJob

    class Packet:
        def __init__(self, frame_id: int) -> None:
            self.frame_id = frame_id

    return RefreshJob(packet=Packet(frame_id), flow=None, decision=None, generation=generation)  # type: ignore[arg-type]


def test_discard_pending_drops_old_run_job_before_it_starts() -> None:
    worker = BlockingWorker()
    dispatcher = RefreshDispatcher(worker)
    try:
        dispatcher.submit(job(1, generation=0))
        assert worker.first_started.wait(2)
        # A second job from the same old run is queued behind the running one.
        dispatcher.submit(job(2, generation=0))
        time.sleep(0.05)
        assert dispatcher.started_total == 1

        assert dispatcher.discard_pending() == 1
        offsets = dispatcher.run_offsets()
        worker.release.set()
        time.sleep(0.1)

        # The old pending job must not start after the run boundary.
        assert worker.started == [1]
        assert dispatcher.started_total == 1
        assert offsets["started"] == 1
        assert dispatcher.discarded_total == 1
    finally:
        worker.release.set()
        dispatcher.close(drain=False)


def test_discard_pending_is_safe_when_nothing_is_queued() -> None:
    dispatcher = RefreshDispatcher(lambda job: None)
    try:
        assert dispatcher.discard_pending() == 0
        assert dispatcher.run_offsets()["discarded"] == 0
    finally:
        dispatcher.close(drain=True)


def test_run_offsets_include_every_job_that_already_started() -> None:
    worker = BlockingWorker()
    dispatcher = RefreshDispatcher(worker)
    try:
        dispatcher.submit(job(10))
        assert worker.first_started.wait(2)
        offsets = dispatcher.run_offsets()
        assert offsets["submitted"] == 1
        assert offsets["started"] == 1
    finally:
        worker.release.set()
        dispatcher.close(drain=True)


def test_submit_after_discard_is_accepted_for_the_new_run() -> None:
    worker = BlockingWorker()
    dispatcher = RefreshDispatcher(worker)
    try:
        dispatcher.submit(job(1, generation=0))
        assert worker.first_started.wait(2)
        dispatcher.submit(job(2, generation=0))
        dispatcher.discard_pending()
        dispatcher.submit(job(3, generation=1))
        worker.release.set()
        time.sleep(0.15)
        assert worker.started == [1, 3]
        assert dispatcher.discarded_total == 1
    finally:
        worker.release.set()
        dispatcher.close(drain=True)
