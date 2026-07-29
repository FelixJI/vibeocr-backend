"""Tests for SupervisorModule: submit/status/events/result/cancel/retry/lifecycle.

Uses a deterministic fake executor so we exercise the full observable
behaviour without any OCR/Paddle/MinerU dependency.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

import pytest

from vibeocr.backend.supervisor.module import (
    ShutdownRequested,
    SupervisorModule,
    SupervisorOptions,
)
from vibeocr.runtime_contracts import (
    CancelMode,
    ContractError,
    ItemState,
    JobKind,
    JobPriority,
    JobState,
    ResidencyStatus,
    SettingsSnapshot,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from vibeocr.backend.supervisor.jobs.staging import StagedInput


class FakeExecutor:
    """Deterministic executor that succeeds every queued item.

    Honours cancel_mode_for and exposes residency/release stubs. A
    ``cancel_after`` callback (if set) is invoked when a cancel is requested
    so tests can simulate cooperative stop transitioning the job to
    cancelled.
    """

    def __init__(self) -> None:
        self.executed: list[str] = []
        self.cancel_calls: list[str] = []
        self.delay: float = 0.0
        self._lock = threading.Lock()

    def execute(self, record, staged: Iterable[StagedInput]) -> None:  # type: ignore[no-untyped-def]
        if self.delay:
            time.sleep(self.delay)
        with self._lock:
            self.executed.append(record.job_id)
        if record.state is JobState.QUEUED:
            record.transition(JobState.RUNNING)
            record.append_event("running")
        for item in list(record.items):
            if item.state is ItemState.QUEUED:
                record.transition_item(item.item_id, ItemState.RUNNING)
                record.set_item_result(item.item_id, {"text": f"ocr-{item.item_id}"})
                record.transition_item(item.item_id, ItemState.SUCCEEDED)
        terminal = JobState.COMPLETED
        # If a cancel was requested while running, follow the state machine:
        # running -> cancel_requested -> cancelled.
        if record.cancel_requested_at is not None:
            record.transition(JobState.CANCEL_REQUESTED)
            terminal = JobState.CANCELLED
        record.transition(terminal)
        record.append_event("done")

    def cancel_mode_for(self, record) -> CancelMode:  # type: ignore[no-untyped-def]
        with self._lock:
            self.cancel_calls.append(record.job_id)
        if record.state is JobState.QUEUED:
            return CancelMode.QUEUED_ONLY
        return CancelMode.COOPERATIVE

    def residency_status(self) -> ResidencyStatus:
        return ResidencyStatus(default_ttl_seconds=300)

    def release_idle(self, pipeline: str | None = None) -> ResidencyStatus:
        return ResidencyStatus(default_ttl_seconds=300)

    def preload(self, pipelines: tuple[str, ...]) -> ResidencyStatus:
        return ResidencyStatus(default_ttl_seconds=300)

    def configure_settings(self, snapshot: SettingsSnapshot) -> ResidencyStatus:
        return ResidencyStatus(
            default_ttl_seconds=snapshot.default_ttl_seconds,
            pipelines=snapshot.pipelines,
        )

    def close(self) -> None:
        return


@pytest.fixture()
def module(tmp_path: Path) -> SupervisorModule:
    opts = SupervisorOptions(instance_id="sup-test")
    return SupervisorModule(options=opts, stager_root=tmp_path / "staging", executor=FakeExecutor())


def _wait_for_terminal(module: SupervisorModule, job_id: str, timeout: float = 2.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        snap = module.status(job_id)
        if snap.state in {
            JobState.COMPLETED,
            JobState.COMPLETED_WITH_ERRORS,
            JobState.CANCELLED,
            JobState.FAILED,
        }:
            return
        time.sleep(0.005)
    raise AssertionError(f"job {job_id} did not reach terminal within {timeout}s")


# ---------------------------------------------------------------------------
# Submit / status / result
# ---------------------------------------------------------------------------


def test_submit_returns_accepted_then_completes(module: SupervisorModule) -> None:
    ref = module.submit(
        kind=JobKind.RECOGNITION,
        priority=JobPriority.INTERACTIVE,
        uploads=[("a.png", "image/png", b"x"), ("b.png", "image/png", b"y")],
    )
    assert ref.instance_id == "sup-test"
    _wait_for_terminal(module, ref.job_id)
    snap = module.status(ref.job_id)
    assert snap.state is JobState.COMPLETED
    assert snap.summary.succeeded == 2
    assert snap.summary.total == 2
    assert [it.display_name for it in snap.items] == ["a.png", "b.png"]


def test_result_preserves_input_order(module: SupervisorModule) -> None:
    ref = module.submit(
        kind=JobKind.RECOGNITION,
        priority=JobPriority.INTERACTIVE,
        uploads=[("first.png", None, b"1"), ("second.png", None, b"2"), ("third.png", None, b"3")],
    )
    _wait_for_terminal(module, ref.job_id)
    results = module.result(ref.job_id)
    assert [r.display_name for r in results] == ["first.png", "second.png", "third.png"]
    assert results[0].payload["text"] == "ocr-it-0000"


def test_events_long_poll_reconnect_no_loss_no_dup(module: SupervisorModule) -> None:
    ref = module.submit(
        kind=JobKind.RECOGNITION,
        priority=JobPriority.INTERACTIVE,
        uploads=[("a.png", None, b"1")],
    )
    _wait_for_terminal(module, ref.job_id)
    first = module.events(ref.job_id, after_sequence=0)
    last_seq = first[-1].sequence
    # Simulate a reconnect: client asks for everything strictly after.
    second = module.events(ref.job_id, after_sequence=last_seq)
    assert second == []
    # And asking again from 0 returns the same set with no duplication.
    again = module.events(ref.job_id, after_sequence=0)
    seqs = [e.sequence for e in again]
    assert seqs == sorted(seqs)
    assert len(seqs) == len(set(seqs))


# ---------------------------------------------------------------------------
# Cancel / retry
# ---------------------------------------------------------------------------


def test_cancel_request_returns_mode_and_eventually_cancells(tmp_path: Path) -> None:
    class HangingExecutor(FakeExecutor):
        """Enters running, then blocks until released, then honours cancel."""

        def __init__(self) -> None:
            super().__init__()
            self._enter = threading.Event()
            self._proceed = threading.Event()

        def execute(self, record, staged):  # type: ignore[no-untyped-def]
            try:
                if record.state is JobState.QUEUED:
                    record.transition(JobState.RUNNING)
                    record.append_event("running")
                self._enter.set()
                # Block until released (by the test after request_cancel).
                self._proceed.wait(timeout=2.0)
                if record.cancel_requested_at is not None:
                    # State machine requires running -> cancel_requested -> cancelled.
                    if record.state is JobState.RUNNING:
                        record.transition(JobState.CANCEL_REQUESTED)
                    if record.state is JobState.CANCEL_REQUESTED:
                        record.transition(JobState.CANCELLED)
                else:
                    if record.state not in (JobState.COMPLETED, JobState.CANCELLED, JobState.FAILED):
                        record.transition(JobState.COMPLETED)
                record.append_event("done")
            except Exception as exc:  # pragma: no cover - debug aid
                record.append_event("executor_error", detail={"error": str(exc)})
                raise

    opts = SupervisorOptions(instance_id="sup-test")
    hanging = HangingExecutor()
    mod = SupervisorModule(options=opts, stager_root=tmp_path / "staging", executor=hanging)
    ref = mod.submit(
        kind=JobKind.RECOGNITION,
        priority=JobPriority.INTERACTIVE,
        uploads=[("a.png", None, b"1")],
    )
    # Wait until the job is running before cancelling.
    assert hanging._enter.wait(timeout=2.0)
    mode = mod.request_cancel(ref.job_id)
    assert mode is CancelMode.COOPERATIVE
    # Now release the executor to finish (cancel_requested_at is set).
    hanging._proceed.set()
    _wait_for_terminal(mod, ref.job_id)
    snap = mod.status(ref.job_id)
    assert snap.state is JobState.CANCELLED


def test_retry_creates_new_job_for_failed_items(tmp_path: Path) -> None:
    class FailingExecutor(FakeExecutor):
        def execute(self, record, staged):  # type: ignore[no-untyped-def]
            if record.state is JobState.QUEUED:
                record.transition(JobState.RUNNING)
                record.append_event("running")
            for item in list(record.items):
                if item.state is ItemState.QUEUED:
                    record.transition_item(item.item_id, ItemState.RUNNING)
                    record.record_item_error(item.item_id, "OOM")
                    record.transition_item(item.item_id, ItemState.FAILED, error="OOM")
            record.transition(JobState.COMPLETED_WITH_ERRORS)
            record.append_event("done")

    opts = SupervisorOptions(instance_id="sup-test")
    mod = SupervisorModule(options=opts, stager_root=tmp_path / "staging", executor=FailingExecutor())
    ref = mod.submit(
        kind=JobKind.RECOGNITION,
        priority=JobPriority.INTERACTIVE,
        uploads=[("a.png", None, b"1")],
    )
    _wait_for_terminal(mod, ref.job_id)
    snap = mod.status(ref.job_id)
    assert snap.state is JobState.COMPLETED_WITH_ERRORS
    # Swap to a succeeding executor for the retry run.
    mod._executor = FakeExecutor()  # type: ignore[attr-defined]
    new_ref = mod.retry(ref.job_id)
    assert new_ref.job_id != ref.job_id
    _wait_for_terminal(mod, new_ref.job_id)
    new_snap = mod.status(new_ref.job_id)
    assert new_snap.state is JobState.COMPLETED
    assert new_snap.summary.succeeded == 1


def test_retry_rejects_non_terminal(tmp_path: Path) -> None:
    class BlockingExecutor(FakeExecutor):
        def __init__(self) -> None:
            super().__init__()
            self.entered = threading.Event()
            self.release = threading.Event()

        def execute(self, record, staged):  # type: ignore[no-untyped-def]
            if record.state is JobState.QUEUED:
                record.transition(JobState.RUNNING)
            self.entered.set()
            self.release.wait(timeout=2.0)
            if record.state is JobState.RUNNING:
                record.transition(JobState.COMPLETED)

    executor = BlockingExecutor()
    mod = SupervisorModule(
        options=SupervisorOptions(instance_id="sup-test"),
        stager_root=tmp_path / "staging",
        executor=executor,
    )
    ref = mod.submit(
        kind=JobKind.RECOGNITION,
        priority=JobPriority.INTERACTIVE,
        uploads=[("a.png", None, b"1")],
    )
    assert executor.entered.wait(timeout=2.0)
    try:
        with pytest.raises(ContractError):
            mod.retry(ref.job_id)
    finally:
        executor.release.set()


def test_delete_releases_terminal_job(module: SupervisorModule) -> None:
    ref = module.submit(
        kind=JobKind.RECOGNITION,
        priority=JobPriority.INTERACTIVE,
        uploads=[("a.png", None, b"1")],
    )
    _wait_for_terminal(module, ref.job_id)
    module.delete(ref.job_id)
    with pytest.raises(KeyError):
        module.status(ref.job_id)


def test_delete_rejects_non_terminal(tmp_path: Path) -> None:
    # Use an executor that never transitions the job, so it stays running.
    class StuckExecutor(FakeExecutor):
        def execute(self, record, staged):  # type: ignore[no-untyped-def]
            if record.state is JobState.QUEUED:
                record.transition(JobState.RUNNING)
            # deliberately do not finish

    opts = SupervisorOptions(instance_id="sup-test")
    mod = SupervisorModule(options=opts, stager_root=tmp_path / "staging", executor=StuckExecutor())
    ref = mod.submit(
        kind=JobKind.RECOGNITION,
        priority=JobPriority.INTERACTIVE,
        uploads=[("a.png", None, b"1")],
    )
    # Wait until the job is running (non-terminal).
    deadline = time.time() + 2.0
    while time.time() < deadline:
        if mod.status(ref.job_id).state is JobState.RUNNING:
            break
        time.sleep(0.005)
    with pytest.raises(ShutdownRequested):
        mod.delete(ref.job_id)


# ---------------------------------------------------------------------------
# Drain / shutdown
# ---------------------------------------------------------------------------


def test_drain_rejects_new_jobs(module: SupervisorModule) -> None:
    module.begin_drain()
    assert module.draining is True
    with pytest.raises(ShutdownRequested):
        module.submit(
            kind=JobKind.RECOGNITION,
            priority=JobPriority.INTERACTIVE,
            uploads=[("a.png", None, b"1")],
        )


def test_shutdown_releases_all_staging(module: SupervisorModule) -> None:
    ref = module.submit(
        kind=JobKind.RECOGNITION,
        priority=JobPriority.INTERACTIVE,
        uploads=[("a.png", None, b"1")],
    )
    _wait_for_terminal(module, ref.job_id)
    module.shutdown_now()
    assert module.shutdown is True
    # staging root should be empty after release_all
    assert not any(module.stager.root.iterdir()) if module.stager.root.exists() else True


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def test_settings_roundtrip(module: SupervisorModule) -> None:
    snap = SettingsSnapshot(default_ttl_seconds=600)
    out = module.update_settings(snap)
    assert out.default_ttl_seconds == 600
    assert module.settings().default_ttl_seconds == 600
