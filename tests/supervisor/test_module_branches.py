"""Tests for the defensive / less-travelled branches of SupervisorModule.

Existing test_module_lifecycle.py covers the happy path; this file targets
the branches left behind: begin_drain cancelling queued jobs, shutdown_now
bounded wait with running jobs, the PDF adapter teardown on shutdown,
submit_request's source-type/attachment checks, and the residency/cache
code paths.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

import pytest
from vibeocr.backend.supervisor.jobs.staging import StagingQuotaError
from vibeocr.backend.supervisor.module import (
    SupervisorModule,
    SupervisorOptions,
)
from vibeocr.runtime_contracts import (
    CancelMode,
    JobKind,
    JobPriority,
    JobState,
    PipelineSelection,
    ResidencyStatus,
    SettingsSnapshot,
    SubmitItem,
    SubmitRequest,
)

if TYPE_CHECKING:
    from pathlib import Path


class _NullExec:
    """Executor stub that does nothing (jobs stay in their initial state)."""

    def execute(self, record, staged):  # type: ignore[no-untyped-def]
        return

    def cancel_mode_for(self, record):  # type: ignore[no-untyped-def]
        return CancelMode.COOPERATIVE

    def residency_status(self) -> ResidencyStatus:
        return ResidencyStatus(default_ttl_seconds=300)

    def release_idle(self, pipeline=None):  # type: ignore[no-untyped-def]
        return ResidencyStatus(default_ttl_seconds=300)

    def preload(self, pipelines):  # type: ignore[no-untyped-def]
        return ResidencyStatus(default_ttl_seconds=300)

    def configure_settings(self, snapshot):  # type: ignore[no-untyped-def]
        return ResidencyStatus(
            default_ttl_seconds=snapshot.default_ttl_seconds,
            pipelines=snapshot.pipelines,
        )

    def close(self) -> None:
        return


# ---------------------------------------------------------------------------
# begin_drain: queued-job cancellation
# ---------------------------------------------------------------------------


def test_begin_drain_cancels_queued_job(tmp_path: Path) -> None:
    """A job still in QUEUED state when drain begins is cancelled (lines 160-162)."""
    executor = _NullExec()  # never transitions, so the job stays queued
    opts = SupervisorOptions(instance_id="drain-test")
    mod = SupervisorModule(options=opts, stager_root=tmp_path / "s", executor=executor)
    ref = mod.submit(
        kind=JobKind.RECOGNITION,
        priority=JobPriority.INTERACTIVE,
        uploads=[("a.png", None, b"1")],
    )
    # Sanity: the job is still queued (executor is a no-op).
    assert mod.status(ref.job_id).state is JobState.QUEUED
    mod.begin_drain()
    assert mod.status(ref.job_id).state is JobState.CANCELLED


# ---------------------------------------------------------------------------
# shutdown_now: bounded wait with running jobs + executor.close + pdf stop
# ---------------------------------------------------------------------------


def test_shutdown_now_waits_bounded_for_running_job(tmp_path: Path) -> None:
    """While running jobs exist the bounded wait loop sleeps (lines 178-186)."""
    release = threading.Event()

    class _StuckRunner(_NullExec):
        def execute(self, record, staged):  # type: ignore[no-untyped-def]
            record.transition(JobState.RUNNING)
            release.wait(timeout=2.0)  # hold the job non-terminal

    executor = _StuckRunner()
    opts = SupervisorOptions(instance_id="shutdown-test", draining_grace_seconds=0.2)
    mod = SupervisorModule(options=opts, stager_root=tmp_path / "s", executor=executor)
    mod.submit(
        kind=JobKind.RECOGNITION,
        priority=JobPriority.INTERACTIVE,
        uploads=[("a.png", None, b"1")],
    )
    # Run shutdown_now in a thread; it should block then exit after the grace.
    done = threading.Event()

    def run():  # type: ignore[no-untyped-def]
        mod.shutdown_now()
        done.set()

    t = threading.Thread(target=run, daemon=True)
    started = time.monotonic()
    t.start()
    # Must complete even though the running job never finishes on its own.
    assert done.wait(timeout=2.0)
    assert time.monotonic() - started >= 0.15  # grace window observed
    release.set()
    assert mod.shutdown is True


def test_shutdown_now_calls_executor_close(tmp_path: Path) -> None:
    close_calls: list[int] = []

    class _Closable(_NullExec):
        def close(self) -> None:
            close_calls.append(1)

    mod = SupervisorModule(
        options=SupervisorOptions(instance_id="x"),
        stager_root=tmp_path / "s",
        executor=_Closable(),
    )
    mod.shutdown_now()
    assert close_calls == [1]


def test_shutdown_now_calls_pdf_adapter_stop(tmp_path: Path) -> None:
    stop_calls: list[int] = []

    class _Pdf:
        def stop(self) -> None:
            stop_calls.append(1)

    mod = SupervisorModule(
        options=SupervisorOptions(instance_id="x"),
        stager_root=tmp_path / "s",
        executor=_NullExec(),
        pdf_adapter=_Pdf(),
    )
    mod.shutdown_now()
    assert stop_calls == [1]


def test_shutdown_now_does_not_call_close_when_missing(tmp_path: Path) -> None:
    """An executor without ``close`` must not break shutdown_now."""

    class _NoClose:
        def execute(self, record, staged):  # type: ignore[no-untyped-def]
            return

        def cancel_mode_for(self, record):  # type: ignore[no-untyped-def]
            return CancelMode.COOPERATIVE

    mod = SupervisorModule(
        options=SupervisorOptions(instance_id="x"),
        stager_root=tmp_path / "s",
        executor=_NoClose(),  # type: ignore[arg-type]
    )
    mod.shutdown_now()  # must not raise
    assert mod.shutdown is True


# ---------------------------------------------------------------------------
# submit: client_items mismatch
# ---------------------------------------------------------------------------


def test_submit_rejects_client_items_length_mismatch(tmp_path: Path) -> None:
    mod = SupervisorModule(
        options=SupervisorOptions(instance_id="x"),
        stager_root=tmp_path / "s",
        executor=_NullExec(),
    )
    items = (
        SubmitItem(
            client_item_key="k",
            ordinal=0,
            display_name="x",
            source={"type": "upload.v1", "attachment": "a"},
        ),
    )
    with pytest.raises(StagingQuotaError, match="client item manifest"):
        mod.submit(
            kind=JobKind.RECOGNITION,
            priority=JobPriority.INTERACTIVE,
            uploads=[("a.png", None, b"1"), ("b.png", None, b"2")],
            client_items=items,  # one item vs two uploads
        )


# ---------------------------------------------------------------------------
# submit_request: source-type + attachment validation
# ---------------------------------------------------------------------------


def _request(items):  # type: ignore[no-untyped-def]
    return SubmitRequest(
        request_id="r",
        kind=JobKind.RECOGNITION,
        priority=JobPriority.INTERACTIVE,
        pipeline=PipelineSelection("OCR"),
        items=items,
    )


def test_submit_request_rejects_non_upload_source(tmp_path: Path) -> None:
    """A source type that is not upload.v1 is rejected (line 277-279)."""
    mod = SupervisorModule(
        options=SupervisorOptions(instance_id="x"),
        stager_root=tmp_path / "s",
        executor=_NullExec(),
    )
    req = _request(
        (
            SubmitItem(
                client_item_key="k",
                ordinal=0,
                display_name="a.pdf",
                source={
                    "type": "pdf_page.v1",
                    "session_id": "s",
                    "session_revision": 1,
                    "page_index": 0,
                },
            ),
        )
    )
    with pytest.raises(StagingQuotaError, match="source type is not wired"):
        mod.submit_request(req, {})


def test_submit_request_rejects_missing_attachment(tmp_path: Path) -> None:
    """A referenced attachment not in the attachments dict is rejected (line 282-284)."""
    mod = SupervisorModule(
        options=SupervisorOptions(instance_id="x"),
        stager_root=tmp_path / "s",
        executor=_NullExec(),
    )
    req = _request(
        (
            SubmitItem(
                client_item_key="k",
                ordinal=0,
                display_name="a.png",
                source={"type": "upload.v1", "attachment": "missing"},
            ),
        )
    )
    with pytest.raises(StagingQuotaError, match="manifest attachment is missing"):
        mod.submit_request(req, {})


def test_submit_request_rejects_unreferenced_attachment(tmp_path: Path) -> None:
    """An attachment not referenced by any item is rejected (line 288-292)."""
    mod = SupervisorModule(
        options=SupervisorOptions(instance_id="x"),
        stager_root=tmp_path / "s",
        executor=_NullExec(),
    )
    req = _request(
        (
            SubmitItem(
                client_item_key="k",
                ordinal=0,
                display_name="a.png",
                source={"type": "upload.v1", "attachment": "a"},
            ),
        )
    )
    with pytest.raises(StagingQuotaError, match="unreferenced multipart"):
        mod.submit_request(
            req, {"a": ("image/png", b"x"), "extra": ("image/png", b"y")}
        )


# ---------------------------------------------------------------------------
# residency / release_idle / update_settings branches
# ---------------------------------------------------------------------------


def test_residency_returns_cached_snapshot_while_preloading(tmp_path: Path) -> None:
    """When ``_preload_count > 0``, residency returns the cached snapshot
    instead of querying the executor (line 419-420)."""
    calls: list[int] = []

    class _Counting(_NullExec):
        def residency_status(self) -> ResidencyStatus:
            calls.append(1)
            return ResidencyStatus(default_ttl_seconds=4321)

    executor = _Counting()
    mod = SupervisorModule(
        options=SupervisorOptions(instance_id="x"),
        stager_root=tmp_path / "s",
        executor=executor,
    )
    # Simulate an in-flight preload by bumping the counter directly.
    with mod._lock:  # type: ignore[attr-defined]
        mod._preload_count = 1  # type: ignore[attr-defined]
    status = mod.residency()
    assert calls == []  # executor not queried
    # Default snapshot has default_ttl_seconds=300 (the module's initial value).
    assert status.default_ttl_seconds == 300


def test_release_idle_returns_and_caches_status(tmp_path: Path) -> None:
    """release_idle calls the executor and remembers the snapshot (lines 425-426)."""
    mod = SupervisorModule(
        options=SupervisorOptions(instance_id="x"),
        stager_root=tmp_path / "s",
        executor=_NullExec(),
    )
    status = mod.release_idle("OCR")
    assert status.default_ttl_seconds == 300


def test_update_settings_caches_residency_when_returned(tmp_path: Path) -> None:
    """When configure_settings returns a ResidencyStatus, it's cached (line 451-452)."""
    cached: list[ResidencyStatus] = []

    class _ConfigCache(_NullExec):
        def configure_settings(self, snapshot):  # type: ignore[no-untyped-def]
            status = ResidencyStatus(
                default_ttl_seconds=snapshot.default_ttl_seconds,
            )
            cached.append(status)
            return status

    mod = SupervisorModule(
        options=SupervisorOptions(instance_id="x"),
        stager_root=tmp_path / "s",
        executor=_ConfigCache(),
    )
    snap = SettingsSnapshot(default_ttl_seconds=900)
    out = mod.update_settings(snap)
    assert out.default_ttl_seconds == 900
    # configure_settings was called and returned a ResidencyStatus that was cached.
    assert cached and cached[0].default_ttl_seconds == 900
    # The cached snapshot is observable while a preload owns the executor lock.
    with mod._lock:  # type: ignore[attr-defined]
        mod._preload_count = 1  # type: ignore[attr-defined]
    assert mod.residency().default_ttl_seconds == 900


def test_update_settings_tolerates_non_residency_return(tmp_path: Path) -> None:
    """When configure_settings returns None, residency is NOT replaced
    (the False branch of ``isinstance(status, ResidencyStatus)``)."""

    class _NoneConfig(_NullExec):
        def configure_settings(self, snapshot):  # type: ignore[no-untyped-def]
            return None  # not a ResidencyStatus

    mod = SupervisorModule(
        options=SupervisorOptions(instance_id="x"),
        stager_root=tmp_path / "s",
        executor=_NoneConfig(),
    )
    out = mod.update_settings(SettingsSnapshot(default_ttl_seconds=42))
    assert out.default_ttl_seconds == 42


def test_update_settings_tolerates_missing_configure(tmp_path: Path) -> None:
    """An executor without ``configure_settings`` must not break update_settings."""

    class _NoConfig:
        def execute(self, record, staged):  # type: ignore[no-untyped-def]
            return

        def cancel_mode_for(self, record):  # type: ignore[no-untyped-def]
            return CancelMode.COOPERATIVE

        def residency_status(self) -> ResidencyStatus:
            return ResidencyStatus(default_ttl_seconds=300)

    mod = SupervisorModule(
        options=SupervisorOptions(instance_id="x"),
        stager_root=tmp_path / "s",
        executor=_NoConfig(),  # type: ignore[arg-type]
    )
    snap = SettingsSnapshot(default_ttl_seconds=123)
    out = mod.update_settings(snap)
    assert out.default_ttl_seconds == 123
