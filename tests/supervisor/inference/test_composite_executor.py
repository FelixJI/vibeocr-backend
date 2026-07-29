"""Tests for CompositeExecutor: kind routing + residency aggregation."""

from __future__ import annotations

from dataclasses import dataclass, field

from vibeocr.backend.supervisor.inference.composite_executor import CompositeExecutor
from vibeocr.runtime_contracts import (
    CancelMode,
    JobKind,
    JobState,
    ResidencyEntry,
    ResidencyKind,
    ResidencyStatus,
    SettingsSnapshot,
)


@dataclass
class _FakeItem:
    item_id: str = "it-0"
    state: str = "queued"


@dataclass
class _FakeRecord:
    job_id: str = "job-1"
    kind: JobKind = JobKind.RECOGNITION
    state: JobState = JobState.QUEUED
    items: list = field(default_factory=lambda: [_FakeItem()])
    events: list = field(default_factory=list)

    def transition(self, target: JobState) -> None:
        self.state = target

    def append_event(self, stage: str, *, detail: dict | None = None) -> None:
        self.events.append(stage)


class _FakeExecutor:
    """Records execute() calls and returns canned residency."""

    def __init__(
        self,
        *,
        entry: ResidencyEntry | None = None,
        cancel_mode: CancelMode = CancelMode.COOPERATIVE,
    ) -> None:
        self.execute_calls: list[tuple[str, JobKind]] = []
        self.release_calls: list[str | None] = []
        self.preload_calls: list[tuple[str, ...]] = []
        self._entry = entry
        self._cancel_mode = cancel_mode

    def execute(self, record, staged) -> None:  # type: ignore[no-untyped-def]
        self.execute_calls.append((record.job_id, record.kind))
        record.transition(JobState.COMPLETED)
        record.append_event("fake-done")

    def cancel_mode_for(self, record) -> CancelMode:  # type: ignore[no-untyped-def]
        return self._cancel_mode

    def residency_status(self) -> ResidencyStatus:
        entries = (self._entry,) if self._entry is not None else ()
        return ResidencyStatus(default_ttl_seconds=300, entries=entries)

    def release_idle(self, pipeline: str | None = None) -> ResidencyStatus:
        self.release_calls.append(pipeline)
        return self.residency_status()

    def preload(self, pipelines: tuple[str, ...]) -> ResidencyStatus:
        self.preload_calls.append(pipelines)
        return self.residency_status()

    def configure_settings(self, snapshot: SettingsSnapshot) -> ResidencyStatus:
        return ResidencyStatus(
            default_ttl_seconds=snapshot.default_ttl_seconds,
            pipelines=snapshot.pipelines,
        )

    def close(self) -> None:
        return


def _build(*, paddle: _FakeExecutor, mineru: _FakeExecutor) -> CompositeExecutor:
    return CompositeExecutor(
        [
            (paddle, frozenset({JobKind.RECOGNITION})),
            (mineru, frozenset({JobKind.MINERU_PARSE})),
        ]
    )


def test_recognition_routes_to_paddle_child() -> None:
    paddle = _FakeExecutor()
    mineru = _FakeExecutor()
    comp = _build(paddle=paddle, mineru=mineru)
    record = _FakeRecord(kind=JobKind.RECOGNITION)

    comp.execute(record, [])

    assert paddle.execute_calls == [("job-1", JobKind.RECOGNITION)]
    assert mineru.execute_calls == []
    assert record.state is JobState.COMPLETED


def test_mineru_parse_routes_to_mineru_child() -> None:
    paddle = _FakeExecutor()
    mineru = _FakeExecutor()
    comp = _build(paddle=paddle, mineru=mineru)
    record = _FakeRecord(kind=JobKind.MINERU_PARSE)

    comp.execute(record, [])

    assert mineru.execute_calls == [("job-1", JobKind.MINERU_PARSE)]
    assert paddle.execute_calls == []


def test_unhandled_kind_fails_job() -> None:
    paddle = _FakeExecutor()
    mineru = _FakeExecutor()
    comp = _build(paddle=paddle, mineru=mineru)
    record = _FakeRecord(kind=JobKind.PDF_OCR)

    comp.execute(record, [])

    assert paddle.execute_calls == []
    assert mineru.execute_calls == []
    assert record.state is JobState.FAILED
    assert "no_backend_for_kind" in record.events


def test_cancel_mode_uses_dispatched_child() -> None:
    paddle = _FakeExecutor(cancel_mode=CancelMode.QUEUED_ONLY)
    mineru = _FakeExecutor(cancel_mode=CancelMode.FORCED)
    comp = _build(paddle=paddle, mineru=mineru)
    record = _FakeRecord(kind=JobKind.MINERU_PARSE)

    comp.execute(record, [])
    # After dispatch, cancel_mode_for should reflect the MinerU child.
    assert comp.cancel_mode_for(record) is CancelMode.FORCED


def test_residency_status_unions_entries() -> None:
    paddle = _FakeExecutor(
        entry=ResidencyEntry(pipeline="OCR", kind=ResidencyKind.SOFT_TTL)
    )
    mineru = _FakeExecutor(
        entry=ResidencyEntry(pipeline="MinerU", kind=ResidencyKind.SOFT_TTL)
    )
    comp = _build(paddle=paddle, mineru=mineru)

    status = comp.residency_status()
    pipelines = {e.pipeline for e in status.entries}
    assert pipelines == {"OCR", "MinerU"}


def test_release_idle_fans_out_to_all_children() -> None:
    paddle = _FakeExecutor()
    mineru = _FakeExecutor()
    comp = _build(paddle=paddle, mineru=mineru)

    comp.release_idle("OCR")

    assert paddle.release_calls == ["OCR"]
    assert mineru.release_calls == ["OCR"]


def test_release_idle_none_releases_all() -> None:
    paddle = _FakeExecutor()
    mineru = _FakeExecutor()
    comp = _build(paddle=paddle, mineru=mineru)

    comp.release_idle(None)

    assert paddle.release_calls == [None]
    assert mineru.release_calls == [None]


def test_preload_fans_out_to_runtime_children() -> None:
    paddle = _FakeExecutor()
    mineru = _FakeExecutor()
    comp = _build(paddle=paddle, mineru=mineru)

    comp.preload(("OCR", "PP-StructureV3"))

    assert paddle.preload_calls == [("OCR", "PP-StructureV3")]
    assert mineru.preload_calls == [("OCR", "PP-StructureV3")]
