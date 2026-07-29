"""Tests for the deeper state-machine branches of JobRegistry / JobRecord.

Existing test_job_registry.py covers the happy path; this file targets the
branches left behind: commit_item_success/failure/cancelled edge cases,
outcomes_after, observe, snapshot(purged), _require_item errors, and
all_job_ids filtering.
"""

from __future__ import annotations

import pytest

from vibeocr.backend.supervisor.jobs.registry import (
    JobNotFoundError,
    JobRecord,
    JobRegistry,
)
from vibeocr.runtime_contracts import (
    CancelMode,
    ContractError,
    ItemState,
    JobItem,
    JobKind,
    JobPriority,
    JobState,
)


def _make_registry() -> JobRegistry:
    return JobRegistry(instance_id="reg-test")


def _items(n: int) -> list[JobItem]:
    return [
        JobItem(item_id=f"it-{i}", display_name=f"f{i}.png", state=ItemState.QUEUED)
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# commit_item_success
# ---------------------------------------------------------------------------


def test_commit_item_success_rejects_empty_payload_type() -> None:
    reg = _make_registry()
    rec = reg.create(
        kind=JobKind.RECOGNITION,
        priority=JobPriority.INTERACTIVE,
        items=_items(1),
        progress_total=1,
    )
    rec.transition(JobState.QUEUED)
    rec.transition(JobState.RUNNING)
    with pytest.raises(ContractError, match="non-empty typed payload"):
        rec.commit_item_success("it-0", payload_type="", payload={"x": 1})


def test_commit_item_success_rejects_empty_payload() -> None:
    reg = _make_registry()
    rec = reg.create(
        kind=JobKind.RECOGNITION,
        priority=JobPriority.INTERACTIVE,
        items=_items(1),
        progress_total=1,
    )
    rec.transition(JobState.QUEUED)
    rec.transition(JobState.RUNNING)
    with pytest.raises(ContractError, match="non-empty typed payload"):
        rec.commit_item_success("it-0", payload_type="ocr.v1", payload={})


def test_commit_item_success_transitions_queued_item() -> None:
    reg = _make_registry()
    rec = reg.create(
        kind=JobKind.RECOGNITION,
        priority=JobPriority.INTERACTIVE,
        items=_items(1),
        progress_total=1,
    )
    rec.transition(JobState.QUEUED)
    rec.transition(JobState.RUNNING)
    # Item is still queued → commit auto-transitions queued → running → succeeded.
    outcome = rec.commit_item_success(
        "it-0", payload_type="ocr.v1", payload={"text": "ok"}
    )
    assert outcome.state is ItemState.SUCCEEDED
    assert outcome.payload_type == "ocr.v1"
    assert rec.snapshot().items[0].state is ItemState.SUCCEEDED


def test_commit_item_success_rejects_item_in_non_running_state() -> None:
    """Committing success for an already-terminal item raises ContractError."""
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
    with pytest.raises(ContractError, match="cannot commit success"):
        rec.commit_item_success(
            "it-0", payload_type="ocr.v1", payload={"text": "ok"}
        )


# ---------------------------------------------------------------------------
# commit_item_failure
# ---------------------------------------------------------------------------


def test_commit_item_failure_rejects_empty_error_code() -> None:
    reg = _make_registry()
    rec = reg.create(
        kind=JobKind.RECOGNITION,
        priority=JobPriority.INTERACTIVE,
        items=_items(1),
        progress_total=1,
    )
    rec.transition(JobState.QUEUED)
    rec.transition(JobState.RUNNING)
    with pytest.raises(ContractError, match="error_code"):
        rec.commit_item_failure("it-0", error_code="", error="boom")


def test_commit_item_failure_on_terminal_non_failed_item_raises() -> None:
    """Committing failure for a SUCCEEDED item raises ContractError."""
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
    with pytest.raises(ContractError, match="cannot commit failure"):
        rec.commit_item_failure("it-0", error_code="X", error="boom")


def test_commit_item_failure_idempotent_on_already_failed_item() -> None:
    """Re-committing failure for an already-failed item is idempotent."""
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
    rec.transition_item("it-0", ItemState.FAILED, error="first")
    # Second commit is idempotent (no state change, no raise).
    outcome = rec.commit_item_failure(
        "it-0", error_code="X", error="second", detail={"k": "v"}
    )
    assert outcome.state is ItemState.FAILED
    assert outcome.error_code == "X"


def test_commit_item_failure_records_detail_in_event() -> None:
    reg = _make_registry()
    rec = reg.create(
        kind=JobKind.RECOGNITION,
        priority=JobPriority.INTERACTIVE,
        items=_items(1),
        progress_total=1,
    )
    rec.transition(JobState.QUEUED)
    rec.transition(JobState.RUNNING)
    rec.commit_item_failure(
        "it-0", error_code="OOM", error="out of memory", detail={"retry": False}
    )
    event = next(e for e in rec.events if e.stage == "item_failed")
    assert event.detail["code"] == "OOM"
    assert event.detail["retry"] is False


# ---------------------------------------------------------------------------
# commit_item_cancelled
# ---------------------------------------------------------------------------


def test_commit_item_cancelled_transitions_non_terminal_item() -> None:
    reg = _make_registry()
    rec = reg.create(
        kind=JobKind.RECOGNITION,
        priority=JobPriority.INTERACTIVE,
        items=_items(1),
        progress_total=1,
    )
    rec.transition(JobState.QUEUED)
    rec.transition(JobState.RUNNING)
    outcome = rec.commit_item_cancelled("it-0")
    assert outcome.state is ItemState.CANCELLED
    assert outcome.error_code == "CANCELLED"


def test_commit_item_cancelled_idempotent_on_already_cancelled() -> None:
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
    rec.transition_item("it-0", ItemState.CANCELLED)
    outcome = rec.commit_item_cancelled("it-0")
    assert outcome.state is ItemState.CANCELLED


def test_commit_item_cancelled_on_terminal_non_cancelled_raises() -> None:
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
    with pytest.raises(ContractError, match="cannot cancel terminal item"):
        rec.commit_item_cancelled("it-0")


# ---------------------------------------------------------------------------
# outcomes_after + observe
# ---------------------------------------------------------------------------


def test_outcomes_after_returns_strictly_greater() -> None:
    reg = _make_registry()
    rec = reg.create(
        kind=JobKind.RECOGNITION,
        priority=JobPriority.INTERACTIVE,
        items=_items(2),
        progress_total=2,
    )
    rec.transition(JobState.QUEUED)
    rec.transition(JobState.RUNNING)
    rec.commit_item_success("it-0", payload_type="ocr.v1", payload={"text": "a"})
    rec.commit_item_success("it-1", payload_type="ocr.v1", payload={"text": "b"})
    outcomes = rec.outcomes_after(0)
    assert len(outcomes) == 2
    assert outcomes[0].item_id == "it-0"


def test_observe_rejects_negative_sequence() -> None:
    reg = _make_registry()
    rec = reg.create(
        kind=JobKind.RECOGNITION,
        priority=JobPriority.INTERACTIVE,
        items=_items(1),
        progress_total=1,
    )
    with pytest.raises(ContractError, match="after_sequence must be >= 0"):
        rec.observe(-1)


def test_observe_returns_events_and_outcomes_within_window() -> None:
    reg = _make_registry()
    rec = reg.create(
        kind=JobKind.RECOGNITION,
        priority=JobPriority.INTERACTIVE,
        items=_items(1),
        progress_total=1,
    )
    rec.transition(JobState.QUEUED)
    seq_after_queued = rec.event_sequence
    rec.transition(JobState.RUNNING)
    rec.commit_item_success("it-0", payload_type="ocr.v1", payload={"text": "ok"})
    update = rec.observe(seq_after_queued)
    # The update contains only events/outcomes after the queued transition.
    assert update.through_sequence == rec.event_sequence
    # commit_item_success emits an item_succeeded event in the window.
    assert any(event.stage == "item_succeeded" for event in update.events)
    assert any(o.payload_type == "ocr.v1" for o in update.outcomes)


# ---------------------------------------------------------------------------
# snapshot: purged record
# ---------------------------------------------------------------------------


def test_snapshot_on_purged_record_raises() -> None:
    reg = _make_registry()
    rec = reg.create(
        kind=JobKind.RECOGNITION,
        priority=JobPriority.INTERACTIVE,
        items=_items(1),
        progress_total=1,
    )
    rec.purged = True
    with pytest.raises(JobNotFoundError):
        rec.snapshot()


# ---------------------------------------------------------------------------
# _require_item
# ---------------------------------------------------------------------------


def test_transition_item_unknown_raises() -> None:
    reg = _make_registry()
    rec = reg.create(
        kind=JobKind.RECOGNITION,
        priority=JobPriority.INTERACTIVE,
        items=_items(1),
        progress_total=1,
    )
    rec.transition(JobState.QUEUED)
    rec.transition(JobState.RUNNING)
    with pytest.raises(ContractError, match="unknown item_id"):
        rec.transition_item("does-not-exist", ItemState.RUNNING)


def test_set_item_result_unknown_raises() -> None:
    reg = _make_registry()
    rec = reg.create(
        kind=JobKind.RECOGNITION,
        priority=JobPriority.INTERACTIVE,
        items=_items(1),
        progress_total=1,
    )
    with pytest.raises(ContractError, match="unknown item_id"):
        rec.set_item_result("nope", {"x": 1})


def test_record_item_error_unknown_raises() -> None:
    reg = _make_registry()
    rec = reg.create(
        kind=JobKind.RECOGNITION,
        priority=JobPriority.INTERACTIVE,
        items=_items(1),
        progress_total=1,
    )
    with pytest.raises(ContractError, match="unknown item_id"):
        rec.record_item_error("nope", "err")


# ---------------------------------------------------------------------------
# request_cancel: ACCEPTED state
# ---------------------------------------------------------------------------


def test_request_cancel_on_queued_transitions_to_cancel_requested() -> None:
    """A cooperative cancel on a QUEUED job transitions to CANCEL_REQUESTED
    (line 454 branch — note: ACCEPTED cannot transition to CANCEL_REQUESTED by
    contract, so we use QUEUED which can)."""
    reg = _make_registry()
    rec = reg.create(
        kind=JobKind.RECOGNITION,
        priority=JobPriority.INTERACTIVE,
        items=_items(1),
        progress_total=1,
    )
    rec.transition(JobState.QUEUED)
    mode = reg.request_cancel(rec.job_id, mode=CancelMode.COOPERATIVE)
    assert mode is CancelMode.COOPERATIVE
    assert rec.snapshot().state is JobState.CANCEL_REQUESTED


# ---------------------------------------------------------------------------
# create_retry: kind + priority override
# ---------------------------------------------------------------------------


def test_create_retry_overrides_kind_and_priority() -> None:
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
    rec.transition_item("it-0", ItemState.FAILED, error="x")
    rec.transition(JobState.COMPLETED_WITH_ERRORS)
    new = reg.create_retry(
        rec.job_id, kind=JobKind.MINERU_PARSE, priority=JobPriority.BACKGROUND
    )
    assert new.kind is JobKind.MINERU_PARSE
    assert new.priority is JobPriority.BACKGROUND


# ---------------------------------------------------------------------------
# all_job_ids + purge
# ---------------------------------------------------------------------------


def test_all_job_ids_excludes_purged() -> None:
    reg = _make_registry()
    rec1 = reg.create(
        kind=JobKind.RECOGNITION,
        priority=JobPriority.INTERACTIVE,
        items=_items(1),
        progress_total=1,
    )
    rec2 = reg.create(
        kind=JobKind.RECOGNITION,
        priority=JobPriority.INTERACTIVE,
        items=_items(1),
        progress_total=1,
    )
    rec1.transition(JobState.QUEUED)
    rec1.transition(JobState.FAILED)
    reg.purge(rec1.job_id)
    ids = set(reg.all_job_ids())
    assert rec1.job_id not in ids
    assert rec2.job_id in ids


def test_purge_unknown_raises_job_not_found() -> None:
    reg = _make_registry()
    with pytest.raises(JobNotFoundError):
        reg.purge("does-not-exist")


def test_get_purged_record_raises() -> None:
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


def test_events_on_unknown_job_raises() -> None:
    reg = _make_registry()
    with pytest.raises(JobNotFoundError):
        reg.events("nope")


def test_snapshot_on_unknown_job_raises() -> None:
    reg = _make_registry()
    with pytest.raises(JobNotFoundError):
        reg.snapshot("nope")


# ---------------------------------------------------------------------------
# create: terminal hook is wired
# ---------------------------------------------------------------------------


def test_create_wires_terminal_hook() -> None:
    """The registry's terminal hook is invoked on every terminal transition."""
    calls: list[str] = []

    def hook(record: JobRecord) -> None:
        calls.append(record.state.value if hasattr(record.state, "value") else str(record.state))

    reg = _make_registry()
    reg.set_terminal_hook(hook)
    rec = reg.create(
        kind=JobKind.RECOGNITION,
        priority=JobPriority.INTERACTIVE,
        items=_items(1),
        progress_total=1,
    )
    rec.transition(JobState.QUEUED)
    rec.transition(JobState.FAILED)
    assert "failed" in calls


# ---------------------------------------------------------------------------
# mark_degraded
# ---------------------------------------------------------------------------


def test_mark_degraded_sets_flag() -> None:
    reg = _make_registry()
    rec = reg.create(
        kind=JobKind.RECOGNITION,
        priority=JobPriority.INTERACTIVE,
        items=_items(1),
        progress_total=1,
    )
    rec.mark_degraded()
    assert rec.snapshot().degraded is True


# ---------------------------------------------------------------------------
# Registry properties + iteration
# ---------------------------------------------------------------------------


def test_registry_instance_id_property() -> None:
    """The instance_id property exposes the configured id (line 368)."""
    reg = _make_registry()
    assert reg.instance_id == "reg-test"


def test_registry_iter_yields_all_records() -> None:
    """``iter(registry)`` yields every record dict (lines 426-427)."""
    reg = _make_registry()
    rec1 = reg.create(
        kind=JobKind.RECOGNITION,
        priority=JobPriority.INTERACTIVE,
        items=_items(1),
        progress_total=1,
    )
    rec2 = reg.create(
        kind=JobKind.RECOGNITION,
        priority=JobPriority.INTERACTIVE,
        items=_items(1),
        progress_total=1,
    )
    iterated = {r.job_id for r in reg}
    assert iterated == {rec1.job_id, rec2.job_id}


def test_snapshot_uses_registry_schema_version() -> None:
    """A registry constructed with a custom schema_version propagates it to the
    snapshot (line 420)."""
    reg = JobRegistry(instance_id="x", schema_version=99)
    rec = reg.create(
        kind=JobKind.RECOGNITION,
        priority=JobPriority.INTERACTIVE,
        items=_items(1),
        progress_total=1,
    )
    snap = reg.snapshot(rec.job_id)
    assert snap.schema_version == 99
