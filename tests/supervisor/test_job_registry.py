"""Tests for the JobRegistry state machine and event log."""

from __future__ import annotations

import pytest

from vibeocr.backend.supervisor.jobs.registry import JobNotFoundError, JobRegistry
from vibeocr.runtime_contracts import (
    CancelMode,
    ContractError,
    ItemState,
    JobItem,
    JobKind,
    JobPriority,
    JobState,
    JobStateTransitionError,
)


def _make_registry() -> JobRegistry:
    return JobRegistry(instance_id="sup-test")


def _items(n: int) -> list[JobItem]:
    return [JobItem(item_id=f"it-{i}", display_name=f"f{i}.png", state=ItemState.QUEUED) for i in range(n)]


def test_create_appends_accepted_event_and_id_is_unique() -> None:
    reg = _make_registry()
    rec = reg.create(
        kind=JobKind.RECOGNITION,
        priority=JobPriority.INTERACTIVE,
        items=_items(2),
        progress_total=2,
    )
    assert rec.state is JobState.ACCEPTED
    assert rec.event_sequence == 1
    assert len(rec.events) == 1
    assert rec.events[0].stage == "accepted"


def test_transition_validates_state_machine() -> None:
    reg = _make_registry()
    rec = reg.create(
        kind=JobKind.RECOGNITION,
        priority=JobPriority.INTERACTIVE,
        items=_items(1),
        progress_total=1,
    )
    rec.transition(JobState.QUEUED)
    rec.transition(JobState.RUNNING)
    rec.transition(JobState.COMPLETED)
    # terminal cannot go back
    with pytest.raises(JobStateTransitionError):
        rec.transition(JobState.RUNNING)


def test_terminal_states_cannot_leave() -> None:
    reg = _make_registry()
    rec = reg.create(
        kind=JobKind.RECOGNITION,
        priority=JobPriority.INTERACTIVE,
        items=_items(1),
        progress_total=1,
    )
    rec.transition(JobState.QUEUED)
    rec.transition(JobState.FAILED)
    with pytest.raises(JobStateTransitionError):
        rec.transition(JobState.RUNNING)


def test_item_transition_succeeded_bumps_progress() -> None:
    reg = _make_registry()
    rec = reg.create(
        kind=JobKind.RECOGNITION,
        priority=JobPriority.INTERACTIVE,
        items=_items(2),
        progress_total=2,
    )
    rec.transition(JobState.QUEUED)
    rec.transition(JobState.RUNNING)
    rec.transition_item("it-0", ItemState.RUNNING)
    rec.transition_item("it-0", ItemState.SUCCEEDED)
    assert rec.progress_current == 1
    rec.transition_item("it-1", ItemState.RUNNING)
    rec.transition_item("it-1", ItemState.SUCCEEDED)
    assert rec.progress_current == 2


def test_item_result_and_error_recorded() -> None:
    reg = _make_registry()
    rec = reg.create(
        kind=JobKind.RECOGNITION,
        priority=JobPriority.INTERACTIVE,
        items=_items(1),
        progress_total=1,
    )
    rec.set_item_result("it-0", {"text": "hello"})
    rec.record_item_error("it-0", "OOM")
    assert rec.results["it-0"] == {"text": "hello"}
    assert rec.item_errors["it-0"] == "OOM"


def test_events_after_returns_strictly_greater() -> None:
    reg = _make_registry()
    rec = reg.create(
        kind=JobKind.RECOGNITION,
        priority=JobPriority.INTERACTIVE,
        items=_items(1),
        progress_total=1,
    )
    rec.append_event("queued")
    rec.append_event("running")
    after = rec.events_after(1)
    assert [e.sequence for e in after] == [2, 3]
    # reconnect scenario: ask for everything after the last -> empty
    assert rec.events_after(3) == []


def test_events_after_rejects_negative() -> None:
    reg = _make_registry()
    rec = reg.create(
        kind=JobKind.RECOGNITION,
        priority=JobPriority.INTERACTIVE,
        items=_items(1),
        progress_total=1,
    )
    with pytest.raises(ContractError):
        rec.events_after(-1)


def test_request_cancel_marks_request_not_terminal() -> None:
    reg = _make_registry()
    rec = reg.create(
        kind=JobKind.RECOGNITION,
        priority=JobPriority.INTERACTIVE,
        items=_items(1),
        progress_total=1,
    )
    rec.transition(JobState.QUEUED)
    rec.transition(JobState.RUNNING)
    mode = reg.request_cancel(rec.job_id, mode=CancelMode.COOPERATIVE)
    assert mode is CancelMode.COOPERATIVE
    snap = rec.snapshot()
    # Cancel requested → visible non-terminal state, NOT cancelled yet.
    assert snap.state is JobState.CANCEL_REQUESTED
    assert snap.cancel_mode is CancelMode.COOPERATIVE


def test_request_cancel_queued_only_transitions_straight_to_cancelled() -> None:
    reg = _make_registry()
    rec = reg.create(
        kind=JobKind.RECOGNITION,
        priority=JobPriority.INTERACTIVE,
        items=_items(1),
        progress_total=1,
    )
    rec.transition(JobState.QUEUED)
    mode = reg.request_cancel(rec.job_id, mode=CancelMode.QUEUED_ONLY)
    assert mode is CancelMode.QUEUED_ONLY
    assert rec.snapshot().state is JobState.CANCELLED


def test_request_cancel_on_terminal_is_idempotent() -> None:
    reg = _make_registry()
    rec = reg.create(
        kind=JobKind.RECOGNITION,
        priority=JobPriority.INTERACTIVE,
        items=_items(1),
        progress_total=1,
    )
    rec.transition(JobState.QUEUED)
    rec.transition(JobState.FAILED)
    mode = reg.request_cancel(rec.job_id, mode=CancelMode.COOPERATIVE)
    assert mode is CancelMode.QUEUED_ONLY


def test_retry_creates_new_job_referencing_source() -> None:
    reg = _make_registry()
    rec = reg.create(
        kind=JobKind.RECOGNITION,
        priority=JobPriority.INTERACTIVE,
        items=_items(2),
        progress_total=2,
    )
    rec.transition(JobState.QUEUED)
    rec.transition(JobState.RUNNING)
    rec.transition_item("it-0", ItemState.RUNNING)
    rec.transition_item("it-0", ItemState.SUCCEEDED)
    rec.transition_item("it-1", ItemState.RUNNING)
    rec.transition_item("it-1", ItemState.FAILED)
    rec.transition(JobState.COMPLETED_WITH_ERRORS)
    new = reg.create_retry(rec.job_id)
    assert new.job_id != rec.job_id
    assert new.source_job_id == rec.job_id
    assert new.source_item_ids == ("it-1",)
    assert len(new.items) == 1
    assert new.items[0].state is ItemState.QUEUED


def test_retry_rejects_non_terminal_source() -> None:
    reg = _make_registry()
    rec = reg.create(
        kind=JobKind.RECOGNITION,
        priority=JobPriority.INTERACTIVE,
        items=_items(1),
        progress_total=1,
    )
    rec.transition(JobState.QUEUED)
    rec.transition(JobState.RUNNING)
    with pytest.raises(ContractError):
        reg.create_retry(rec.job_id)


def test_retry_rejects_when_no_failed_items() -> None:
    reg = _make_registry()
    rec = reg.create(
        kind=JobKind.RECOGNITION,
        priority=JobPriority.INTERACTIVE,
        items=_items(1),
        progress_total=1,
    )
    rec.transition(JobState.QUEUED)
    rec.transition(JobState.RUNNING)
    rec.transition_item("it-0", ItemState.RUNNING)
    rec.transition_item("it-0", ItemState.SUCCEEDED)
    rec.transition(JobState.COMPLETED)
    with pytest.raises(ContractError):
        reg.create_retry(rec.job_id)


def test_purge_terminal_then_get_raises() -> None:
    reg = _make_registry()
    rec = reg.create(
        kind=JobKind.RECOGNITION,
        priority=JobPriority.INTERACTIVE,
        items=_items(1),
        progress_total=1,
    )
    rec.transition(JobState.QUEUED)
    rec.transition(JobState.FAILED)
    reg.purge(rec.job_id)
    with pytest.raises(JobNotFoundError):
        reg.get(rec.job_id)


def test_purge_non_terminal_raises() -> None:
    reg = _make_registry()
    rec = reg.create(
        kind=JobKind.RECOGNITION,
        priority=JobPriority.INTERACTIVE,
        items=_items(1),
        progress_total=1,
    )
    with pytest.raises(ContractError):
        reg.purge(rec.job_id)


def test_snapshot_summary_counts() -> None:
    reg = _make_registry()
    rec = reg.create(
        kind=JobKind.PDF_OCR,
        priority=JobPriority.BACKGROUND,
        items=_items(3),
        progress_total=3,
    )
    rec.transition(JobState.QUEUED)
    rec.transition(JobState.RUNNING)
    rec.transition_item("it-0", ItemState.RUNNING)
    rec.transition_item("it-0", ItemState.SUCCEEDED)
    rec.transition_item("it-1", ItemState.RUNNING)
    rec.transition_item("it-1", ItemState.FAILED)
    rec.transition_item("it-2", ItemState.CANCELLED)
    rec.transition(JobState.COMPLETED_WITH_ERRORS)
    snap = rec.snapshot()
    assert snap.summary.succeeded == 1
    assert snap.summary.failed == 1
    assert snap.summary.cancelled == 1
    assert snap.summary.total == 3
