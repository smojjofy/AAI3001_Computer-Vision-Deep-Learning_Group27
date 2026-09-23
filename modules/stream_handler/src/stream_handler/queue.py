"""A small thread-safe queue that favours the most recent live frames."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
import time
from typing import Deque

from .contracts import FramePacket


@dataclass(frozen=True)
class QueuePutResult:
    depth: int
    dropped_now: int
    dropped_total: int


class LatestFrameQueue:
    """Bounded FIFO that drops its oldest entry before accepting a new frame."""

    def __init__(self, capacity: int = 3) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least one")
        self.capacity = capacity
        self._items: Deque[FramePacket] = deque()
        self._dropped_total = 0
        self._closed = False
        self._condition = threading.Condition()

    def put(self, packet: FramePacket) -> QueuePutResult:
        with self._condition:
            if self._closed:
                raise RuntimeError("cannot put into a closed queue")
            dropped_now = 0
            if len(self._items) >= self.capacity:
                self._items.popleft()
                self._dropped_total += 1
                dropped_now = 1
            self._items.append(packet)
            result = QueuePutResult(len(self._items), dropped_now, self._dropped_total)
            self._condition.notify()
            return result

    def get(self, timeout_s: float | None = None) -> FramePacket | None:
        with self._condition:
            deadline = None if timeout_s is None else time.monotonic() + timeout_s
            while not self._items and not self._closed:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return None
                self._condition.wait(remaining)
            return self._items.popleft() if self._items else None

    @property
    def depth(self) -> int:
        with self._condition:
            return len(self._items)

    @property
    def dropped_total(self) -> int:
        with self._condition:
            return self._dropped_total

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
