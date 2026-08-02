"""SupervisorModule: the in-process orchestrator.

The module owns the :class:`JobRegistry`, :class:`InputStager`,
:class:`RetentionPolicy` and a pluggable :class:`Executor`. It exposes the
observable behaviour the HTTP layer and tests interact with — submit,
status, events, result, cancel, retry, runtime, shutdown — without knowing
anything about Paddle/MinerU/PDF.

The executor seam is deliberately abstract so Phase 2 can complete a full
happy-path/partial/cancel/retry flow with a fake; Phase 4/5/6 plug real
adapters behind the same interface.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Protocol

from vibeocr.runtime_contracts import (
    TERMINAL_JOB_STATES,
    CancelMode,
    ContractError,
    JobKind,
    JobPriority,
    JobRef,
    JobSnapshot,
    JobState,
    JobUpdate,
    PipelineSelection,
    ResidencyStatus,
    ResultEntry,
    SettingsSnapshot,
    StageEvent,
    SubmitItem,
    SubmitRequest,
    new_job_id,
)

from .jobs.registry import JobRecord, JobRegistry
from .jobs.retention import RetentionPolicy
from .jobs.staging import InputExpiredError, InputStager, StagedInput, StagingQuotaError

if TYPE_CHECKING:
    from collections.abc import Iterable


class Executor(Protocol):
    """The seam real adapters (Paddle/MinerU/PDF) implement.

    Implementations run item processing in the background; they report
    progress by mutating the :class:`JobRecord` (thread-safe via the
    registry lock) and must respect ``record.cancel_mode`` at microbatch
    boundaries.
    """

    def execute(
        self, record: JobRecord, staged: Iterable[StagedInput]
    ) -> None:  # pragma: no cover - protocol
        ...

    def cancel_mode_for(
        self, record: JobRecord
    ) -> CancelMode:  # pragma: no cover - protocol
        ...

    def residency_status(self) -> ResidencyStatus:  # pragma: no cover - protocol
        ...

    def release_idle(
        self, pipeline: str | None = None
    ) -> ResidencyStatus:  # pragma: no cover - protocol
        ...

    def preload(
        self, pipelines: tuple[str, ...]
    ) -> ResidencyStatus:  # pragma: no cover - protocol
        ...

    def configure_settings(
        self, snapshot: SettingsSnapshot
    ) -> ResidencyStatus:  # pragma: no cover - protocol
        ...

    def close(self) -> None:  # pragma: no cover - protocol
        ...


@dataclass
class SupervisorOptions:
    """Configuration for a supervisor instance."""

    instance_id: str
    max_file_count: int = 64
    max_total_bytes: int = 256 * 1024 * 1024
    max_per_file_bytes: int = 64 * 1024 * 1024
    retention_seconds: float = 3600.0
    draining_grace_seconds: float = 5.0


class ShutdownRequested(Exception):
    """Raised when an operation is attempted on a draining/shutdown module."""


class SupervisorModule:
    """Process-wide supervisor orchestrator (UI-free)."""

    def __init__(
        self,
        *,
        options: SupervisorOptions,
        stager_root: Any,
        executor: Executor,
        pdf_adapter: Any = None,
    ) -> None:
        self.options = options
        self.registry = JobRegistry(options.instance_id)
        self.stager = InputStager(
            root=stager_root,
            max_file_count=options.max_file_count,
            max_total_bytes=options.max_total_bytes,
            max_per_file_bytes=options.max_per_file_bytes,
        )
        self.retention = RetentionPolicy(
            self.registry, retention_seconds=options.retention_seconds
        )
        # Wire the registry so every terminal transition records its
        # timestamp for the retention policy — executors do not need to know.
        self.registry.set_terminal_hook(self.retention.mark_terminal)
        self._executor = executor
        # Optional PDF child-process adapter (Phase 6). When present, the v2
        # PDF session routes proxy through it instead of the legacy
        # PdfBackendClient singleton. Tests / supervisor builds without PDF
        # support pass None and the routes return 503.
        self.pdf_adapter = pdf_adapter
        self._lock = threading.RLock()
        self._draining = False
        self._shutdown = False
        self._settings: SettingsSnapshot = SettingsSnapshot(
            default_ttl_seconds=300, pipelines=()
        )
        # Runtime status is observational and must remain available while a
        # heavy preload owns executor-internal locks. Keep the latest immutable
        # snapshot at the module boundary instead of forcing every status read
        # back through the active loader.
        self._residency_snapshot = ResidencyStatus(
            default_ttl_seconds=self._settings.default_ttl_seconds,
            pipelines=self._settings.pipelines,
        )
        self._preload_count = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def draining(self) -> bool:
        return self._draining

    @property
    def shutdown(self) -> bool:
        return self._shutdown

    def begin_drain(self) -> None:
        """Reject new jobs; queued jobs are cancelled; running jobs get grace."""
        with self._lock:
            self._draining = True
        for record in list(self.registry):
            if record.state is JobState.QUEUED:
                try:
                    self.registry.request_cancel(
                        record.job_id, mode=CancelMode.QUEUED_ONLY
                    )
                    record.transition(JobState.CANCELLED)
                except Exception:  # pragma: no cover - defensive
                    pass

    def shutdown_now(self) -> None:
        """Final shutdown: cancel queued, wait bounded for running, release staging."""
        self.begin_drain()
        with self._lock:
            self._shutdown = True
        # Wait a bounded window for running jobs to reach a terminal state
        # before tearing down staging out from under executor threads. The
        # plan requires "等待 bounded running, 再清理子进程"; we never block
        # forever — past the grace window we release regardless.
        import time as _time

        deadline = _time.monotonic() + self.options.draining_grace_seconds
        while _time.monotonic() < deadline:
            running = [r for r in self.registry if r.state not in TERMINAL_JOB_STATES]
            if not running:
                break
            _time.sleep(0.02)
        try:
            close = getattr(self._executor, "close", None)
            if callable(close):
                close()
        except Exception:  # pragma: no cover - defensive
            pass
        # Final purge of any expired-retention jobs and release all staging.
        try:
            self.retention.purge_expired()
        except Exception:  # pragma: no cover - defensive
            pass
        self.stager.release_all()
        # Tear down the PDF child if we own it.
        if self.pdf_adapter is not None:
            try:
                self.pdf_adapter.stop()
            except Exception:  # pragma: no cover - defensive
                pass

    # ------------------------------------------------------------------
    # Submit
    # ------------------------------------------------------------------

    def submit(
        self,
        *,
        kind: JobKind,
        priority: JobPriority,
        uploads: list[tuple[str, str | None, bytes]],
        request_id: str | None = None,
        pipeline: PipelineSelection | None = None,
        client_items: tuple[SubmitItem, ...] | None = None,
    ) -> JobRef:
        with self._lock:
            if self._draining or self._shutdown:
                raise ShutdownRequested("supervisor is draining")
        # Allocate a job id first so staging lives under the real job dir.
        job_id = new_job_id()
        staged, items = self.stager.stage_job_with_item_errors(job_id, uploads)
        if client_items is not None:
            if len(client_items) != len(items):
                self.stager.release(job_id)
                raise StagingQuotaError("client item manifest does not match uploads")
            items = [
                replace(
                    item,
                    client_item_key=client_items[index].client_item_key,
                    ordinal=client_items[index].ordinal,
                )
                for index, item in enumerate(items)
            ]
        record = self.registry.create(
            kind=kind,
            priority=priority,
            items=items,
            progress_total=len(items),
            stage="queued",
            job_id=job_id,
            request_id=request_id,
            pipeline=pipeline,
        )
        record.transition(JobState.QUEUED)
        record.append_event("queued", detail={"item_count": len(items)})
        for item in items:
            if item.state is not None and item.state.value == "failed":
                record.commit_item_failure(
                    item.item_id,
                    error_code="QUOTA_EXCEEDED",
                    error=item.error or "staging failed",
                )
        # Kick off the executor in the background.
        self._dispatch(record, staged)
        return JobRef(
            job_id=record.job_id,
            instance_id=self.options.instance_id,
            state=record.state,
            items=tuple(record.items),
        )

    def submit_request(
        self,
        request: SubmitRequest,
        attachments: dict[str, tuple[str | None, bytes]],
    ) -> JobRef:
        """Submit a strict logical manifest through the deep job interface."""
        uploads: list[tuple[str, str | None, bytes]] = []
        referenced: set[str] = set()
        for item in request.items:
            source_type = item.source.get("type")
            if source_type != "upload.v1":
                raise StagingQuotaError(f"source type is not wired yet: {source_type}")
            attachment = str(item.source["attachment"])
            if attachment not in attachments:
                raise StagingQuotaError(f"manifest attachment is missing: {attachment}")
            referenced.add(attachment)
            content_type, data = attachments[attachment]
            uploads.append((item.display_name, content_type, data))
        extras = sorted(set(attachments).difference(referenced))
        if extras:
            raise StagingQuotaError(
                "unreferenced multipart attachment(s): " + ", ".join(extras)
            )
        return self.submit(
            kind=request.kind,
            priority=request.priority,
            uploads=uploads,
            request_id=request.request_id,
            pipeline=request.pipeline,
            client_items=request.items,
        )

    def _dispatch(self, record: JobRecord, staged: list[StagedInput]) -> None:
        """Run the executor in a worker thread.

        Phase 2's fake executor is synchronous; real adapters (Phase 4/5/6)
        may run async work internally but must still return promptly so the
        HTTP handler is not blocked.
        """

        def _run() -> None:
            try:
                self._executor.execute(record, list(staged))
            except Exception as exc:  # pragma: no cover - defensive
                if record.state not in TERMINAL_JOB_STATES:
                    try:
                        record.transition(JobState.FAILED)
                        record.append_event(
                            "executor_failed", detail={"error": str(exc)}
                        )
                        self.retention.mark_terminal(record)
                    except Exception:  # pragma: no cover
                        pass

        thread = threading.Thread(
            target=_run, name=f"sup-job-{record.job_id[:8]}", daemon=True
        )
        thread.start()

    # ------------------------------------------------------------------
    # Status / events / result
    # ------------------------------------------------------------------

    def status(self, job_id: str) -> JobSnapshot:
        return self.registry.snapshot(job_id)

    def events(self, job_id: str, after_sequence: int = 0) -> list[StageEvent]:
        return self.registry.events(job_id, after_sequence)

    def observe(self, job_id: str, after_sequence: int = 0) -> JobUpdate:
        return self.registry.get(job_id).observe(after_sequence)

    def result(self, job_id: str) -> list[ResultEntry]:
        record = self.registry.get(job_id)
        out: list[ResultEntry] = []
        for item in record.items:
            payload = record.results.get(item.item_id, {})
            err = record.item_errors.get(item.item_id)
            out.append(
                ResultEntry(
                    item_id=item.item_id,
                    display_name=item.display_name,
                    payload=payload,
                    error_code=err,
                )
            )
        return out

    # ------------------------------------------------------------------
    # Cancel / retry / delete
    # ------------------------------------------------------------------

    def request_cancel(self, job_id: str) -> CancelMode:
        record = self.registry.get(job_id)
        if record.state in TERMINAL_JOB_STATES:
            raise ShutdownRequested("job is already terminal")
        mode = self._executor.cancel_mode_for(record)
        return self.registry.request_cancel(job_id, mode=mode)

    def retry(self, job_id: str) -> JobRef:
        source = self.registry.get(job_id)
        if source.state not in TERMINAL_JOB_STATES:
            raise ContractError("source job must be terminal before retry")
        retryable_source_ids = [
            item.item_id
            for item in source.items
            if item.state.value in {"failed", "cancelled"}
        ]
        if not retryable_source_ids:
            raise ValueError("source job has no failed/cancelled items to retry")
        missing = [
            item_id
            for item_id in retryable_source_ids
            if not self.stager.has_staged_item(job_id, item_id)
        ]
        if missing:
            raise InputExpiredError(
                "retry input expired or unavailable: " + ", ".join(missing)
            )
        new_record = self.registry.create_retry(job_id)
        staged = self.stager.clone_for_retry(
            source_job_id=job_id,
            retry_job_id=new_record.job_id,
            source_to_retry_item_ids=list(
                zip(
                    new_record.source_item_ids,
                    [item.item_id for item in new_record.items],
                )
            ),
        )
        new_record.transition(JobState.QUEUED)
        new_record.append_event("retry_queued", detail={"source": job_id})
        self._dispatch(new_record, staged)
        return JobRef(
            job_id=new_record.job_id,
            instance_id=self.options.instance_id,
            state=new_record.state,
            items=tuple(new_record.items),
        )

    def delete(self, job_id: str) -> None:
        record = self.registry.get(job_id)
        if record.state not in TERMINAL_JOB_STATES:
            raise ShutdownRequested("cannot delete a non-terminal job")
        self.stager.release(job_id)
        self.retention.forget(job_id)
        self.registry.purge(job_id)

    # ------------------------------------------------------------------
    # Runtime / settings
    # ------------------------------------------------------------------

    def residency(self) -> ResidencyStatus:
        with self._lock:
            if self._preload_count > 0:
                return self._residency_snapshot
        status = self._executor.residency_status()
        return self._remember_residency(status)

    def release_idle(self, pipeline: str | None = None) -> ResidencyStatus:
        status = self._executor.release_idle(pipeline)
        return self._remember_residency(status)

    def preload(self, pipelines: tuple[str, ...]) -> ResidencyStatus:
        with self._lock:
            self._preload_count += 1
            status = self._residency_snapshot
        try:
            # Preserve the existing sequential-loading contract at the module
            # boundary so every completed pipeline becomes observable while a
            # later heavy pipeline is still downloading or constructing.
            for pipeline in dict.fromkeys(pipelines):
                status = self._executor.preload((pipeline,))
                self._remember_residency(status)
            return status
        finally:
            with self._lock:
                self._preload_count -= 1

    def settings(self) -> SettingsSnapshot:
        return self._settings

    def update_settings(self, snapshot: SettingsSnapshot) -> SettingsSnapshot:
        configure = getattr(self._executor, "configure_settings", None)
        if callable(configure):
            status = configure(snapshot)
            if isinstance(status, ResidencyStatus):
                self._remember_residency(status)
        with self._lock:
            self._settings = snapshot
        return self._settings

    def _remember_residency(self, status: ResidencyStatus) -> ResidencyStatus:
        with self._lock:
            self._residency_snapshot = status
        return status


def _staging_key_placeholder() -> str:  # pragma: no cover - retained for back-compat
    return f"staging-placeholder-{threading.get_ident()}-{id(object())}"


__all__ = ["Executor", "ShutdownRequested", "SupervisorModule", "SupervisorOptions"]
