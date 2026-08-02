"""Tests for the less-travelled branches of DeviceScheduler + RecoveryPolicy.

Existing test_scheduler_budgets_recovery.py covers the core happy path; this
file targets the branches left behind: re-enqueue idempotency, cancel of an
unknown job, try_acquire on an unknown job, the blocking ``acquire`` loop
(with + without cancellation), queue_depth's cancelled filter, and
``next_job_for`` skipping cancelled entries, plus the remaining
RecoveryPolicy.classify branches and the UNKNOWN-attempt decision matrix.
"""

from __future__ import annotations

import threading
import time

from vibeocr.backend.supervisor.inference.recovery import (
    FailureClass,
    RecoveryAction,
    RecoveryPolicy,
)
from vibeocr.backend.supervisor.inference.scheduler import DeviceScheduler
from vibeocr.runtime_contracts import JobPriority

# ---------------------------------------------------------------------------
# DeviceScheduler: idempotent enqueue, cancel unknown, try_acquire unknown
# ---------------------------------------------------------------------------


def test_enqueue_is_idempotent_for_same_job() -> None:
    """Re-enqueueing the same job_id is a no-op (line 84)."""
    sched = DeviceScheduler(devices=["gpu:0"])
    sched.enqueue(job_id="j1", device="gpu:0", priority=JobPriority.INTERACTIVE)
    sched.enqueue(job_id="j1", device="gpu:0", priority=JobPriority.BACKGROUND)
    # Only one entry exists.
    assert sched.queue_depth() == 1


def test_cancel_unknown_job_returns_false() -> None:
    """Cancelling a job that was never enqueued returns False (line 99)."""
    sched = DeviceScheduler(devices=["gpu:0"])
    assert sched.cancel("never-enqueued") is False


def test_try_acquire_unknown_job_returns_none() -> None:
    """try_acquire for a job not in the queue returns None (line 126)."""
    sched = DeviceScheduler(devices=["gpu:0"])
    assert sched.try_acquire("unknown", "gpu:0") is None


def test_queue_depth_filters_cancelled_entries_for_specific_device() -> None:
    """queue_depth(device) skips cancelled entries (line 176)."""
    sched = DeviceScheduler(devices=["gpu:0"])
    sched.enqueue(job_id="j1", device="gpu:0", priority=JobPriority.INTERACTIVE)
    sched.enqueue(job_id="j2", device="gpu:0", priority=JobPriority.INTERACTIVE)
    sched.cancel("j1")
    # Only j2 remains for gpu:0.
    assert sched.queue_depth("gpu:0") == 1


def test_next_job_for_skips_cancelled_entries() -> None:
    """A cancelled queue entry is skipped when peaking (line 229).

    ``cancel()`` removes the entry from the heap, so to exercise the
    in-queue cancellation guard we inject a cancelled entry directly.
    """
    sched = DeviceScheduler(devices=["gpu:0"])
    sched.enqueue(job_id="live-job", device="gpu:0", priority=JobPriority.INTERACTIVE)
    # Manually plant a second queue entry that is also flagged cancelled.
    # This mirrors the defensive scenario the loop guards against.
    sched._cancelled.add("ghost-job")  # type: ignore[attr-defined]
    sched._queue.append(  # type: ignore[attr-defined]
        sched._new_entry("ghost-job", "gpu:0", JobPriority.INTERACTIVE)
    )
    import heapq

    heapq.heapify(sched._queue)  # type: ignore[attr-defined]
    assert sched.next_job_for("gpu:0") == "live-job"


# ---------------------------------------------------------------------------
# DeviceScheduler.acquire: blocking happy path + cancellation
# ---------------------------------------------------------------------------


def test_acquire_blocks_until_device_free_then_grants() -> None:
    """acquire() blocks while the device is busy, then grants the lease
    (lines 144-153)."""
    sched = DeviceScheduler(devices=["gpu:0"])
    # Hold the device with job-1 first.
    sched.enqueue(job_id="j1", device="gpu:0", priority=JobPriority.INTERACTIVE)
    lease1 = sched.try_acquire("j1", "gpu:0")
    assert lease1 is not None

    granted: list = []

    def wait_for_j2():  # type: ignore[no-untyped-def]
        lease = sched.acquire(
            job_id="j2", device="gpu:0", priority=JobPriority.INTERACTIVE
        )
        granted.append(lease)

    t = threading.Thread(target=wait_for_j2, daemon=True)
    t.start()
    # Give the acquire loop a moment to enter the wait.
    time.sleep(0.1)
    assert granted == []  # still blocked
    sched.release(lease1)
    t.join(timeout=1.0)
    assert len(granted) == 1
    assert granted[0].job_id == "j2"


def test_acquire_returns_none_when_cancelled_before_grant() -> None:
    """A cancelled callback forces acquire() to return None (lines 147-149)."""
    sched = DeviceScheduler(devices=["gpu:0"])
    # Hold the device so j2 cannot acquire.
    sched.enqueue(job_id="j1", device="gpu:0", priority=JobPriority.INTERACTIVE)
    lease1 = sched.try_acquire("j1", "gpu:0")
    assert lease1 is not None

    cancel_flag = {"on": False}

    def cancelled():  # type: ignore[no-untyped-def]
        return cancel_flag["on"]

    granted: list = []

    def wait_for_j2():  # type: ignore[no-untyped-def]
        lease = sched.acquire(
            job_id="j2",
            device="gpu:0",
            priority=JobPriority.INTERACTIVE,
            cancelled=cancelled,
        )
        granted.append(lease)

    t = threading.Thread(target=wait_for_j2, daemon=True)
    t.start()
    time.sleep(0.1)
    cancel_flag["on"] = True  # signal cancel
    t.join(timeout=1.0)
    assert granted == [None]


# ---------------------------------------------------------------------------
# RecoveryPolicy.classify: remaining branches
# ---------------------------------------------------------------------------


def test_classify_deterministic_model_branches() -> None:
    """The 'shape'/'dtype'/'inference' substrings classify as DETERMINISTIC_MODEL
    (lines 71-72)."""
    policy = RecoveryPolicy()
    assert policy.classify("incompatible shape") is FailureClass.DETERMINISTIC_MODEL
    assert policy.classify("dtype mismatch") is FailureClass.DETERMINISTIC_MODEL
    assert policy.classify("inference failed") is FailureClass.DETERMINISTIC_MODEL


def test_classify_unknown_for_unrecognised_messages() -> None:
    """An unrecognised message returns UNKNOWN (line 73)."""
    policy = RecoveryPolicy()
    assert policy.classify("something completely different") is FailureClass.UNKNOWN


def test_classify_empty_message_returns_unknown() -> None:
    policy = RecoveryPolicy()
    assert policy.classify("") is FailureClass.UNKNOWN


# ---------------------------------------------------------------------------
# RecoveryPolicy.next_action: UNKNOWN decision matrix
# ---------------------------------------------------------------------------


def test_unknown_failure_retries_once_then_failfast() -> None:
    """An UNKNOWN failure on the first attempt retries; on the second it FAIL_FASTs
    (lines 139-143)."""
    policy = RecoveryPolicy()
    d1 = policy.next_action(
        failure=FailureClass.UNKNOWN, current_batch_size=4, attempt=0
    )
    assert d1.action is RecoveryAction.BACKOFF_RETRY
    assert d1.degraded is True
    d2 = policy.next_action(
        failure=FailureClass.UNKNOWN, current_batch_size=4, attempt=1
    )
    assert d2.action is RecoveryAction.FAIL_FAST
    assert d2.degraded is True
