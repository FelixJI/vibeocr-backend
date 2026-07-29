"""JobRegistry: the in-memory state machine for supervisor jobs.

Design constraints (see plan §4.2):

* Session-level state only — never persisted across supervisor restarts.
* Terminal jobs never return to non-terminal states.
* Retry creates a *new* job referencing the source; source history is
  immutable after the retry is created.
* Cancel being requested does not equal cancelled — only the executor's
  resource-stop transitions a job to ``cancelled``.
* Events have monotonically increasing per-job sequence numbers.
* Input order and result order are identical.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from vibeocr.runtime_contracts import (
    TERMINAL_ITEM_STATES,
    TERMINAL_JOB_STATES,
    CancelMode,
    ContractError,
    ItemOutcome,
    ItemState,
    JobItem,
    JobKind,
    JobPriority,
    JobSnapshot,
    JobState,
    JobSummary,
    JobUpdate,
    PipelineSelection,
    StageEvent,
    assert_item_transition,
    assert_job_transition,
    new_job_id,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator


class JobNotFoundError(KeyError):
    """Raised when a job id is unknown or has been purged."""


@dataclass(slots=True)
class JobRecord:
    """Mutable job record kept inside the registry.

    All mutation and snapshotting is guarded by ``_lock`` so the executor
    thread and HTTP handlers can read/modify a record concurrently without
    torn reads or lost events.
    """

    job_id: str
    kind: JobKind
    priority: JobPriority
    instance_id: str
    created_at: str
    state: JobState = JobState.ACCEPTED
    started_at: str | None = None
    finished_at: str | None = None
    stage: str | None = None
    progress_current: int = 0
    progress_total: int = 0
    items: list[JobItem] = field(default_factory=list)
    events: list[StageEvent] = field(default_factory=list)
    cancel_requested_at: str | None = None
    cancel_mode: CancelMode | None = None
    degraded: bool = False
    # Stable-ordered result payloads keyed by item_id.
    results: dict[str, dict] = field(default_factory=dict)
    item_errors: dict[str, str | None] = field(default_factory=dict)
    # For retry linkage (does not mutate source history).
    source_job_id: str | None = None
    source_item_ids: tuple[str, ...] = ()
    request_id: str | None = None
    pipeline: PipelineSelection | None = None
    purged: bool = False
    _next_seq: int = 1
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _terminal_at: str | None = None
    _outcomes: list[tuple[int, ItemOutcome]] = field(default_factory=list, repr=False)
    # Hook invoked on every terminal transition so the registry/retention
    # policy can record the timestamp without the executor knowing about it.
    _retention_hook: Callable[[JobRecord], None] = field(
        default=lambda _record: None, repr=False
    )

    # ------------------------------------------------------------------
    # Event log
    # ------------------------------------------------------------------

    def append_event(self, stage: str, *, item_id: str | None = None, detail: dict | None = None) -> StageEvent:
        with self._lock:
            seq = self._next_seq
            self._next_seq += 1
            event = StageEvent(sequence=seq, stage=stage, item_id=item_id, detail=detail)
            self.events.append(event)
            return event

    def events_after(self, after_sequence: int) -> list[StageEvent]:
        if after_sequence < 0:
            raise ContractError("after_sequence must be >= 0")
        with self._lock:
            return [e for e in self.events if e.sequence > after_sequence]

    @property
    def event_sequence(self) -> int:
        with self._lock:
            return self._next_seq - 1

    # ------------------------------------------------------------------
    # Transitions (validated)
    # ------------------------------------------------------------------

    def transition(self, target: JobState, *, now: datetime | None = None) -> None:
        with self._lock:
            assert_job_transition(self.state, target)
            self.state = target
            ts = (now or datetime.now(tz=UTC)).isoformat()
            if target is JobState.RUNNING and self.started_at is None:
                self.started_at = ts
            if target in TERMINAL_JOB_STATES and self.finished_at is None:
                self.finished_at = ts
                self._terminal_at = ts
            self._retention_hook(self)

    def transition_item(self, item_id: str, target: ItemState, *, error: str | None = None) -> None:
        with self._lock:
            item = self._require_item(item_id)
            assert_item_transition(item.state, target)
            new_attempt = item.attempt
            idx = self.items.index(item)
            self.items[idx] = JobItem(
                item_id=item.item_id,
                display_name=item.display_name,
                state=target,
                attempt=new_attempt,
                error=error,
                client_item_key=item.client_item_key,
                ordinal=item.ordinal,
                source_item_id=item.source_item_id,
            )
            if target in TERMINAL_ITEM_STATES:
                self.progress_current = min(self.progress_current + 1, self.progress_total)

    def set_item_result(self, item_id: str, payload: dict) -> None:
        with self._lock:
            self._require_item(item_id)
            self.results[item_id] = payload

    def record_item_error(self, item_id: str, error: str) -> None:
        with self._lock:
            self._require_item(item_id)
            self.item_errors[item_id] = error

    def mark_degraded(self) -> None:
        with self._lock:
            self.degraded = True

    def commit_item_success(
        self,
        item_id: str,
        *,
        payload_type: str,
        payload: dict,
    ) -> ItemOutcome:
        """Atomically commit a validated successful outcome."""
        if not payload_type or not isinstance(payload, dict) or not payload:
            raise ContractError("successful item outcome requires a non-empty typed payload")
        with self._lock:
            item = self._require_item(item_id)
            if item.state is ItemState.QUEUED:
                self.transition_item(item_id, ItemState.RUNNING)
                item = self._require_item(item_id)
            if item.state is not ItemState.RUNNING:
                raise ContractError(
                    f"cannot commit success for item {item_id} in state {item.state.value}"
                )
            self.results[item_id] = payload
            self.transition_item(item_id, ItemState.SUCCEEDED)
            event = self.append_event("item_succeeded", item_id=item_id)
            outcome = ItemOutcome(
                item_id=item_id,
                state=ItemState.SUCCEEDED,
                attempt=item.attempt,
                payload_type=payload_type,
                payload=payload,
            )
            self._outcomes.append((event.sequence, outcome))
            return outcome

    def commit_item_failure(
        self,
        item_id: str,
        *,
        error_code: str,
        error: str,
        detail: dict | None = None,
    ) -> ItemOutcome:
        """Atomically commit a failed outcome, including admission failures."""
        if not error_code:
            raise ContractError("failed item outcome requires an error_code")
        with self._lock:
            item = self._require_item(item_id)
            if item.state not in TERMINAL_ITEM_STATES:
                if item.state is ItemState.QUEUED:
                    self.transition_item(item_id, ItemState.RUNNING)
                    item = self._require_item(item_id)
                self.transition_item(item_id, ItemState.FAILED, error=error)
            elif item.state is not ItemState.FAILED:
                raise ContractError(
                    f"cannot commit failure for item {item_id} in state {item.state.value}"
                )
            elif not any(
                outcome.item_id == item_id for _sequence, outcome in self._outcomes
            ):
                # Admission/staging may construct the item directly in FAILED
                # state before a JobRecord exists; count that terminal exactly once.
                self.progress_current = min(
                    self.progress_current + 1, self.progress_total
                )
            self.item_errors[item_id] = error_code
            event = self.append_event(
                "item_failed",
                item_id=item_id,
                detail={"code": error_code, "message": error, **(detail or {})},
            )
            outcome = ItemOutcome(
                item_id=item_id,
                state=ItemState.FAILED,
                attempt=item.attempt,
                error_code=error_code,
                error_detail={"message": error, **(detail or {})},
            )
            self._outcomes.append((event.sequence, outcome))
            return outcome

    def commit_item_cancelled(self, item_id: str) -> ItemOutcome:
        with self._lock:
            item = self._require_item(item_id)
            if item.state not in TERMINAL_ITEM_STATES:
                self.transition_item(item_id, ItemState.CANCELLED)
            elif item.state is not ItemState.CANCELLED:
                raise ContractError(
                    f"cannot cancel terminal item {item_id} in state {item.state.value}"
                )
            event = self.append_event("item_cancelled", item_id=item_id)
            outcome = ItemOutcome(
                item_id=item_id,
                state=ItemState.CANCELLED,
                attempt=item.attempt,
                error_code="CANCELLED",
            )
            self._outcomes.append((event.sequence, outcome))
            return outcome

    def outcomes_after(self, after_sequence: int) -> list[ItemOutcome]:
        with self._lock:
            return [
                outcome
                for sequence, outcome in self._outcomes
                if sequence > after_sequence
            ]

    def observe(self, after_sequence: int = 0) -> JobUpdate:
        """Read snapshot, events and outcomes at one atomic sequence watermark."""
        if after_sequence < 0:
            raise ContractError("after_sequence must be >= 0")
        with self._lock:
            through = self._next_seq - 1
            return JobUpdate(
                snapshot=self.snapshot(),
                events=tuple(
                    event
                    for event in self.events
                    if after_sequence < event.sequence <= through
                ),
                outcomes=tuple(
                    outcome
                    for sequence, outcome in self._outcomes
                    if after_sequence < sequence <= through
                ),
                through_sequence=through,
            )

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot(self, schema_version: int = 2) -> JobSnapshot:
        with self._lock:
            if self.purged:
                raise JobNotFoundError(self.job_id)
            succeeded = sum(1 for it in self.items if it.state is ItemState.SUCCEEDED)
            failed = sum(1 for it in self.items if it.state is ItemState.FAILED)
            cancelled = sum(1 for it in self.items if it.state is ItemState.CANCELLED)
            summary = JobSummary(
                succeeded=succeeded, failed=failed, cancelled=cancelled, total=len(self.items)
            )
            result_available = bool(self.results) or any(
                it.state in TERMINAL_ITEM_STATES for it in self.items
            )
            return JobSnapshot(
                job_id=self.job_id,
                kind=self.kind,
                priority=self.priority,
                state=self.state,
                schema_version=schema_version,
                instance_id=self.instance_id,
                created_at=self.created_at,
                started_at=self.started_at,
                finished_at=self.finished_at,
                stage=self.stage,
                progress_current=self.progress_current,
                progress_total=self.progress_total,
                items=tuple(self.items),
                summary=summary,
                cancel_requested_at=self.cancel_requested_at,
                cancel_mode=self.cancel_mode,
                degraded=self.degraded,
                event_sequence=self._next_seq - 1,
                result_available=result_available,
                request_id=self.request_id,
                source_job_id=self.source_job_id,
                pipeline=self.pipeline,
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _require_item(self, item_id: str) -> JobItem:
        for it in self.items:
            if it.item_id == item_id:
                return it
        raise ContractError(f"unknown item_id: {item_id}")


class JobRegistry:
    """Thread-safe in-memory job store.

    Session-scoped: there is intentionally no persistence. The supervisor
    cleans stale staging on startup but does not recover job execution state.
    """

    def __init__(self, instance_id: str, *, schema_version: int = 2) -> None:
        self._instance_id = instance_id
        self._schema_version = schema_version
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.RLock()
        # Optional hook fired whenever a record transitions to a terminal
        # state. The RetentionPolicy attaches itself here so terminal timestamps
        # are recorded without executors knowing about retention.
        self._terminal_hook: Callable[[JobRecord], None] = lambda _record: None

    def set_terminal_hook(self, hook: Callable[[JobRecord], None]) -> None:
        self._terminal_hook = hook

    @property
    def instance_id(self) -> str:
        return self._instance_id

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    def create(
        self,
        *,
        kind: JobKind,
        priority: JobPriority,
        items: Iterable[JobItem],
        progress_total: int,
        stage: str = "accepted",
        source_job_id: str | None = None,
        source_item_ids: tuple[str, ...] = (),
        request_id: str | None = None,
        pipeline: PipelineSelection | None = None,
        job_id: str | None = None,
    ) -> JobRecord:
        record = JobRecord(
            job_id=job_id or new_job_id(),
            kind=kind,
            priority=priority,
            instance_id=self._instance_id,
            created_at=datetime.now(tz=UTC).isoformat(),
            progress_total=progress_total,
            stage=stage,
            items=list(items),
            source_job_id=source_job_id,
            source_item_ids=source_item_ids,
            request_id=request_id,
            pipeline=pipeline,
        )
        record._retention_hook = self._terminal_hook  # type: ignore[method-assign]
        with self._lock:
            self._jobs[record.job_id] = record
        record.append_event("accepted")
        return record

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, job_id: str) -> JobRecord:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None or record.purged:
                raise JobNotFoundError(job_id)
            return record

    def snapshot(self, job_id: str) -> JobSnapshot:
        return self.get(job_id).snapshot(self._schema_version)

    def events(self, job_id: str, after_sequence: int = 0) -> list[StageEvent]:
        return self.get(job_id).events_after(after_sequence)

    def __iter__(self) -> Iterator[JobRecord]:
        with self._lock:
            return iter(list(self._jobs.values()))

    # ------------------------------------------------------------------
    # Cancel / retry
    # ------------------------------------------------------------------

    def request_cancel(self, job_id: str, *, mode: CancelMode = CancelMode.COOPERATIVE) -> CancelMode:
        """Mark a cancel request following the contract state machine.

        For ``QUEUED_ONLY`` (the job has not started running) we transition
        directly to ``CANCELLED``. For cooperative/forced modes we transition
        to ``CANCEL_REQUESTED`` — the job is *not* cancelled until the
        executor actually stops its resources and transitions to
        ``CANCELLED``. Socket disconnect / cancel-signal delivery must never
        be reported as terminal.
        """
        with self._lock:
            record = self.get(job_id)
            if record.state in TERMINAL_JOB_STATES:
                # Idempotent: a terminal job stays terminal.
                return CancelMode.QUEUED_ONLY
            record.cancel_requested_at = datetime.now(tz=UTC).isoformat()
            record.cancel_mode = mode
            record.append_event("cancel_requested", detail={"mode": mode.value})
            if mode is CancelMode.QUEUED_ONLY:
                # Resource-free queued job can be cancelled immediately.
                record.transition(JobState.CANCELLED)
            elif record.state in (JobState.RUNNING, JobState.QUEUED, JobState.ACCEPTED):
                # Move into the visible cancel_requested state; the executor
                # will promote to CANCELLED once it stops.
                record.transition(JobState.CANCEL_REQUESTED)
            return mode

    def create_retry(
        self,
        source_job_id: str,
        *,
        kind: JobKind | None = None,
        priority: JobPriority | None = None,
    ) -> JobRecord:
        """Create a new job retrying only failed/cancelled items of the source."""
        source = self.get(source_job_id)
        if source.state not in TERMINAL_JOB_STATES:
            raise ContractError("source job must be terminal before retry")
        retry_items: list[JobItem] = []
        source_item_ids: list[str] = []
        for it in source.items:
            if it.state in (ItemState.FAILED, ItemState.CANCELLED):
                new_id = f"{new_job_id()}"
                retry_items.append(
                    JobItem(
                        item_id=new_id,
                        display_name=it.display_name,
                        state=ItemState.QUEUED,
                        attempt=it.attempt + 1,
                        client_item_key=it.client_item_key,
                        ordinal=it.ordinal,
                        source_item_id=it.item_id,
                    )
                )
                source_item_ids.append(it.item_id)
        if not retry_items:
            raise ContractError("source job has no failed/cancelled items to retry")
        return self.create(
            kind=kind or source.kind,
            priority=priority or source.priority,
            items=retry_items,
            progress_total=len(retry_items),
            stage="queued",
            source_job_id=source_job_id,
            source_item_ids=tuple(source_item_ids),
            request_id=source.request_id,
            pipeline=source.pipeline,
        )

    # ------------------------------------------------------------------
    # Purge / retention
    # ------------------------------------------------------------------

    def purge(self, job_id: str) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                raise JobNotFoundError(job_id)
            if record.state not in TERMINAL_JOB_STATES:
                raise ContractError("cannot purge a non-terminal job")
            record.purged = True

    def all_job_ids(self) -> list[str]:
        with self._lock:
            return [jid for jid, rec in self._jobs.items() if not rec.purged]


__all__ = ["JobNotFoundError", "JobRecord", "JobRegistry"]
