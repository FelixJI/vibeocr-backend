"""DeviceScheduler: per-GPU single-heavy-lease with priority + aging.

Plan §3/§4 Phase 3 goals:

* Each GPU runs at most one heavy inference lease by default.
* Interactive priority preempts at microbatch boundaries; background jobs use
  priority aging to avoid starvation.
* The scheduler exposes ``enqueue/cancel/drain`` and a lease abstraction. It
  does NOT know about Paddle/MinerU types — adapters acquire a lease and run
  their own microbatches inside it.

This implementation is deterministic and uses an injected fake clock so tests
do not rely on wall-clock sleeps.
"""

from __future__ import annotations

import heapq
import itertools
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

from vibeocr.runtime_contracts import JobPriority

Clock = Callable[[], float]

_COUNTER = itertools.count()


@dataclass(order=True, slots=True)
class _QueueEntry:
    """Heap entry ordered by (effective_priority, age_bonus, seq)."""

    sort_key: tuple[float, int, int]
    job_id: str = field(compare=False)
    device: str = field(compare=False)
    priority: JobPriority = field(compare=False)


@dataclass(slots=True)
class DeviceLease:
    """A granted exclusive lease on a device for one heavy task."""

    device: str
    job_id: str
    acquired_at: float


class DeviceScheduler:
    """Single-heavy-lease scheduler with priority aging.

    ``acquire`` blocks the caller until the device is free; real adapters
    call it before running a microbatch. For unit tests we expose the
    non-blocking ``try_acquire`` and the queue introspection helpers.
    """

    def __init__(
        self,
        *,
        devices: list[str] | None = None,
        clock: Clock | None = None,
        aging_interval: float = 1.0,
    ) -> None:
        self._devices = list(devices or ["gpu:0"])
        self._clock = clock or _monotonic
        self._aging_interval = aging_interval
        self._lock = threading.RLock()
        self._changed = threading.Condition(self._lock)
        self._queue: list[_QueueEntry] = []
        self._leases: dict[str, DeviceLease] = {}  # device -> lease
        self._enqueued_at: dict[str, float] = {}  # job_id -> enqueue time
        self._cancelled: set[str] = set()

    # ------------------------------------------------------------------
    # Enqueue / cancel / drain
    # ------------------------------------------------------------------

    def enqueue(self, *, job_id: str, device: str, priority: JobPriority) -> None:
        if device not in self._devices:
            raise ValueError(f"unknown device: {device}")
        with self._changed:
            if job_id in self._enqueued_at:
                return
            self._enqueued_at[job_id] = self._clock()
            self._cancelled.discard(job_id)
            entry = self._new_entry(job_id, device, priority)
            heapq.heappush(self._queue, entry)
            self._changed.notify_all()

    def cancel(self, job_id: str) -> bool:
        with self._changed:
            if job_id in self._enqueued_at:
                self._cancelled.add(job_id)
                self._enqueued_at.pop(job_id, None)
                self._remove_from_heap(job_id)
                self._changed.notify_all()
                return True
            return False

    def drain(self) -> list[str]:
        """Cancel every queued job and return the cancelled ids."""
        with self._changed:
            drained = list(self._enqueued_at.keys())
            self._cancelled.update(drained)
            self._enqueued_at.clear()
            self._queue.clear()
            self._changed.notify_all()
            return drained

    # ------------------------------------------------------------------
    # Acquire / release
    # ------------------------------------------------------------------

    def try_acquire(self, job_id: str, device: str) -> DeviceLease | None:
        """Grant only the current queue head when the device is free."""
        with self._lock:
            if job_id in self._cancelled:
                # Drop the cancelled entry from the heap lazily.
                self._cancelled.discard(job_id)
                self._remove_from_heap(job_id)
                return None
            if self._leases.get(device) is not None:
                return None
            if job_id not in self._enqueued_at:
                return None
            if self.next_job_for(device) != job_id:
                return None
            lease = DeviceLease(device=device, job_id=job_id, acquired_at=self._clock())
            self._leases[device] = lease
            self._enqueued_at.pop(job_id, None)
            self._remove_from_heap(job_id)
            return lease

    def acquire(
        self,
        *,
        job_id: str,
        device: str,
        priority: JobPriority,
        cancelled: Callable[[], bool] | None = None,
    ) -> DeviceLease | None:
        """Block until this job is the dynamic priority head or is cancelled."""
        self.enqueue(job_id=job_id, device=device, priority=priority)
        with self._changed:
            while True:
                if cancelled is not None and cancelled():
                    self.cancel(job_id)
                    return None
                lease = self.try_acquire(job_id, device)
                if lease is not None:
                    return lease
                self._changed.wait(timeout=0.05)

    def _remove_from_heap(self, job_id: str) -> None:
        """Remove all heap entries matching ``job_id`` (there should be one)."""
        self._queue = [e for e in self._queue if e.job_id != job_id]
        heapq.heapify(self._queue)

    def release(self, lease: DeviceLease) -> None:
        with self._changed:
            current = self._leases.get(lease.device)
            if current is not None and current.job_id == lease.job_id:
                self._leases.pop(lease.device, None)
                self._changed.notify_all()

    @property
    def busy_devices(self) -> list[str]:
        with self._lock:
            return list(self._leases.keys())

    def queue_depth(self, device: str | None = None) -> int:
        with self._lock:
            if device is None:
                return len(self._enqueued_at)
            return sum(
                1
                for e in self._queue
                if e.device == device and e.job_id not in self._cancelled
            )

    # ------------------------------------------------------------------
    # Aging
    # ------------------------------------------------------------------

    def effective_priority(self, job_id: str, priority: JobPriority) -> float:
        """Compute effective priority with aging for background jobs.

        Interactive jobs always win; background jobs get a monotonic bonus the
        longer they wait so they cannot be starved indefinitely.
        """
        base = 0.0 if priority is JobPriority.INTERACTIVE else 1.0
        with self._lock:
            enq = self._enqueued_at.get(job_id)
        if enq is None or priority is JobPriority.INTERACTIVE:
            return base
        age = self._clock() - enq
        # Every aging interval reduces the effective priority by 0.1, but it
        # can cross below 0 (i.e. outrank a fresh interactive job) only after
        # a long starvation window.
        return max(0.0, base - age / self._aging_interval * 0.1)

    def _new_entry(
        self, job_id: str, device: str, priority: JobPriority
    ) -> _QueueEntry:
        eff = self.effective_priority(job_id, priority)
        seq = next(_COUNTER)
        # Lower sort_key = higher priority. Age bonus is encoded so an older
        # background job sorts before a newer one with the same base.
        age_seq = seq  # tiebreaker
        return _QueueEntry(
            sort_key=(eff, age_seq, seq),
            job_id=job_id,
            device=device,
            priority=priority,
        )

    def _current_key(self, entry: _QueueEntry) -> tuple[float, int, int]:
        """Recompute the priority component so aging is not frozen at enqueue."""
        return (
            self.effective_priority(entry.job_id, entry.priority),
            entry.sort_key[1],
            entry.sort_key[2],
        )

    def next_job_for(self, device: str) -> str | None:
        """Return the next non-cancelled job id for ``device`` (peek, no pop).

        Scans the queue once for the best matching entry rather than
        pop/push/continue, which could otherwise loop forever when the only
        queued entries are for other devices. The entry is removed from the
        queue when :meth:`try_acquire` grants the lease.
        """
        with self._lock:
            best: _QueueEntry | None = None
            for entry in self._queue:
                if entry.device != device:
                    continue
                if entry.job_id in self._cancelled:
                    continue
                if best is None or self._current_key(entry) < self._current_key(best):
                    best = entry
            return best.job_id if best is not None else None


def _monotonic() -> float:
    import time

    return time.monotonic()


__all__ = ["Clock", "DeviceLease", "DeviceScheduler"]
