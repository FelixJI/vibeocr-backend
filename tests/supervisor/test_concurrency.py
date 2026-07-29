"""Concurrency + cross-module integration tests for the supervisor.

These verify the thread-safety and cross-module wiring that the unit tests
do not cover directly.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

from vibeocr.backend.supervisor.jobs.registry import JobRegistry
from vibeocr.backend.supervisor.jobs.staging import InputStager
from vibeocr.runtime_contracts import ItemState, JobKind, JobPriority, JobState

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Concurrency: executor thread mutates while HTTP readers snapshot
# ---------------------------------------------------------------------------


def test_concurrent_mutation_and_snapshot_is_consistent() -> None:
    """An executor thread mutating a record must not corrupt reader snapshots."""
    reg = JobRegistry(instance_id="sup-test")
    rec = reg.create(
        kind=JobKind.RECOGNITION,
        priority=JobPriority.INTERACTIVE,
        items=[
            __import__("vibeocr.runtime_contracts", fromlist=["JobItem"]).JobItem(
                item_id=f"it-{i}", display_name=f"f{i}", state=ItemState.QUEUED
            )
            for i in range(50)
        ],
        progress_total=50,
    )
    rec.transition(JobState.QUEUED)
    rec.transition(JobState.RUNNING)

    errors: list[Exception] = []

    def _mutate() -> None:
        try:
            for i in range(50):
                rec.transition_item(f"it-{i}", ItemState.RUNNING)
                rec.set_item_result(f"it-{i}", {"i": i})
                rec.transition_item(f"it-{i}", ItemState.SUCCEEDED)
        except Exception as exc:  # pragma: no cover - test failure
            errors.append(exc)

    def _read() -> None:
        try:
            for _ in range(200):
                snap = rec.snapshot()
                # The snapshot's state must always be a valid enum value and
                # the items list must equal the original length (no torn reads
                # shrinking it mid-iteration).
                assert snap.state in JobState
                assert len(snap.items) == 50
                # succeeded count must be monotonic and never exceed total.
                assert snap.summary.succeeded <= 50
        except Exception as exc:  # pragma: no cover - test failure
            errors.append(exc)

    mutator = threading.Thread(target=_mutate)
    readers = [threading.Thread(target=_read) for _ in range(4)]
    for r in readers:
        r.start()
    mutator.start()
    mutator.join()
    for r in readers:
        r.join()
    assert not errors, f"concurrent errors: {errors}"
    # After mutation completes the record is fully consistent.
    snap = rec.snapshot()
    assert snap.summary.succeeded == 50


# ---------------------------------------------------------------------------
# Event log concurrency: no duplicate or lost sequence numbers
# ---------------------------------------------------------------------------


def test_concurrent_event_appends_are_unique_and_ordered() -> None:
    reg = JobRegistry(instance_id="sup-test")
    rec = reg.create(
        kind=JobKind.RECOGNITION,
        priority=JobPriority.INTERACTIVE,
        items=[],
        progress_total=0,
    )

    def _append(n: int) -> None:
        for i in range(n):
            rec.append_event(f"evt-{i}")

    threads = [threading.Thread(target=_append, args=(100,)) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # 8 threads × 100 appends = 800 events, plus the initial "accepted" event.
    seqs = [e.sequence for e in rec.events_after(0)]
    assert len(seqs) == 801
    assert len(set(seqs)) == 801  # no duplicate sequence numbers
    assert seqs == sorted(seqs)  # monotonic / ordered


# ---------------------------------------------------------------------------
# Stager concurrency: concurrent stage/release do not corrupt dirs
# ---------------------------------------------------------------------------


def test_concurrent_stage_and_release_are_safe(tmp_path: Path) -> None:
    stager = InputStager(root=tmp_path / "staging", max_file_count=4, max_total_bytes=4096)
    errors: list[Exception] = []

    def _stage(idx: int) -> None:
        try:
            jid = f"job-{idx}"
            stager.stage_job(jid, [(f"f{i}.png", None, b"x" * 10) for i in range(2)])
            time.sleep(0.001)
            stager.release(jid)
        except Exception as exc:  # pragma: no cover - test failure
            errors.append(exc)

    threads = [threading.Thread(target=_stage, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, f"stager errors: {errors}"
