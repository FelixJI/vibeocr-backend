"""Tests for the residency aggregation / aggregation fallbacks of CompositeExecutor.

Existing test_composite_executor.py covers routing + happy aggregation; this
file targets the branches left behind: default cancel mode, ttl/vram/pipeline
aggregation, the TypeError fallback, configure_settings fan-out, and close.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vibeocr.backend.supervisor.inference.composite_executor import CompositeExecutor
from vibeocr.runtime_contracts import (
    CancelMode,
    JobKind,
    JobState,
    PipelineSpec,
    ResidencyStatus,
    SettingsSnapshot,
)


@dataclass
class _FakeRecord:
    job_id: str = "job-1"
    kind: JobKind = JobKind.RECOGNITION
    state: JobState = JobState.QUEUED
    items: list = field(default_factory=list)
    events: list = field(default_factory=list)

    def transition(self, target: JobState) -> None:
        self.state = target

    def append_event(self, stage: str, *, detail: dict | None = None) -> None:
        self.events.append(stage)


class _StubExecutor:
    """Executor whose residency can be customised; records every method."""

    def __init__(
        self,
        *,
        status: ResidencyStatus | None = None,
        release_raises: bool = False,
        close_raises: bool = False,
        preload_raises: bool = False,
        configure_raises: bool = False,
    ) -> None:
        self._status = status
        self.release_calls: list[str | None] = []
        self.preload_calls: list[tuple[str, ...]] = []
        self.configure_calls: list[SettingsSnapshot] = []
        self.close_calls: int = 0
        self._release_raises = release_raises
        self._close_raises = close_raises
        self._preload_raises = preload_raises
        self._configure_raises = configure_raises

    def execute(self, record, staged) -> None:  # type: ignore[no-untyped-def]
        return

    def cancel_mode_for(self, record) -> CancelMode:  # type: ignore[no-untyped-def]
        return CancelMode.COOPERATIVE

    def residency_status(self) -> ResidencyStatus:
        return self._status if self._status is not None else ResidencyStatus()

    def release_idle(self, pipeline=None):  # type: ignore[no-untyped-def]
        if self._release_raises:
            raise RuntimeError("release boom")
        self.release_calls.append(pipeline)
        return self.residency_status()

    def preload(self, pipelines):  # type: ignore[no-untyped-def]
        if self._preload_raises:
            raise RuntimeError("preload boom")
        self.preload_calls.append(pipelines)
        return self.residency_status()

    def configure_settings(self, snapshot):  # type: ignore[no-untyped-def]
        if self._configure_raises:
            raise RuntimeError("configure boom")
        self.configure_calls.append(snapshot)
        return self.residency_status()

    def close(self) -> None:
        self.close_calls += 1
        if self._close_raises:
            raise RuntimeError("close boom")


# ---------------------------------------------------------------------------
# cancel_mode_for default + dispatch miss
# ---------------------------------------------------------------------------


def test_cancel_mode_returns_cooperative_when_no_child_matches() -> None:
    """A record whose kind matches no child returns the default COOPERATIVE
    (line 81)."""
    paddle = _StubExecutor()
    comp = CompositeExecutor([(paddle, frozenset({JobKind.RECOGNITION}))])
    record = _FakeRecord(kind=JobKind.MINERU_PARSE)
    assert comp.cancel_mode_for(record) is CancelMode.COOPERATIVE


# ---------------------------------------------------------------------------
# Residency aggregation: ttl conversion, pipelines, vram
# ---------------------------------------------------------------------------


def test_residency_aggregates_pipelines_and_vram() -> None:
    """pipelines + vram_total_mb + vram_used_mb are unioned/maxed across children
    (lines 103-110)."""
    paddle = _StubExecutor(
        status=ResidencyStatus(
            default_ttl_seconds=300,
            pipelines=(PipelineSpec(name="OCR"),),
            vram_total_mb=4096,
            vram_used_mb=1024,
        )
    )
    mineru = _StubExecutor(
        status=ResidencyStatus(
            default_ttl_seconds=600,
            pipelines=(PipelineSpec(name="MinerU"),),
            vram_total_mb=8192,
            vram_used_mb=2048,
        )
    )
    comp = CompositeExecutor(
        [
            (paddle, frozenset({JobKind.RECOGNITION})),
            (mineru, frozenset({JobKind.MINERU_PARSE})),
        ]
    )
    status = comp.residency_status()
    pipelines = {spec.name for spec in status.pipelines}
    assert pipelines == {"OCR", "MinerU"}
    assert status.vram_total_mb == 8192  # max
    assert status.vram_used_mb == 2048  # max


def test_residency_tolerates_non_int_default_ttl() -> None:
    """A non-int default_ttl_seconds is ignored (line 99-100)."""
    paddle = _StubExecutor()
    # Inject a status whose default_ttl_seconds is a string.
    paddle.residency_status = lambda: ResidencyStatus(  # type: ignore[method-assign]
        default_ttl_seconds="not-an-int",  # type: ignore[arg-type]
    )
    comp = CompositeExecutor([(paddle, frozenset({JobKind.RECOGNITION}))])
    status = comp.residency_status()
    # Falls back to the prior default_ttl (300).
    assert status.default_ttl_seconds == 300


def test_residency_falls_back_to_empty_on_typeerror(monkeypatch) -> None:
    """If the final ResidencyStatus construction raises TypeError, the empty
    status is returned (lines 119-120)."""
    # Build the stub status BEFORE patching so construction works normally.
    paddle = _StubExecutor(
        status=ResidencyStatus(
            default_ttl_seconds=300,
            vram_total_mb=4096,
            vram_used_mb=1024,
        )
    )
    comp = CompositeExecutor([(paddle, frozenset({JobKind.RECOGNITION}))])

    real_init = ResidencyStatus.__init__
    call_count = {"n": 0}

    def boom_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        # Raise TypeError only on the FIRST construction (the composite's
        # aggregation call). The fallback's bare ResidencyStatus() must still
        # succeed.
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise TypeError("simulated constructor failure")
        return real_init(self, *args, **kwargs)

    monkeypatch.setattr(ResidencyStatus, "__init__", boom_init)
    status = comp.residency_status()
    # The TypeError fallback returns a bare ResidencyStatus() (default ttl 300).
    assert status.default_ttl_seconds == 300
    assert status.entries == ()


# ---------------------------------------------------------------------------
# execute: already-terminal job with no matching child
# ---------------------------------------------------------------------------


def test_unhandled_kind_skips_when_job_already_terminal() -> None:
    """An already-terminal job whose kind matches no child returns without
    appending the no_backend event (line 64->73)."""
    comp = CompositeExecutor([(_StubExecutor(), frozenset({JobKind.RECOGNITION}))])
    record = _FakeRecord(kind=JobKind.PDF_OCR)
    record.transition(JobState.FAILED)  # already terminal
    comp.execute(record, [])
    # No event appended because the state was already terminal.
    assert record.events == []


# ---------------------------------------------------------------------------
# configure_settings + close fan-out
# ---------------------------------------------------------------------------


def test_configure_settings_fans_out_to_all_children() -> None:
    paddle = _StubExecutor()
    mineru = _StubExecutor()
    comp = CompositeExecutor(
        [
            (paddle, frozenset({JobKind.RECOGNITION})),
            (mineru, frozenset({JobKind.MINERU_PARSE})),
        ]
    )
    snap = SettingsSnapshot(default_ttl_seconds=600)
    comp.configure_settings(snap)
    assert paddle.configure_calls == [snap]
    assert mineru.configure_calls == [snap]


def test_close_closes_every_child_and_clears_dispatch() -> None:
    paddle = _StubExecutor()
    mineru = _StubExecutor()
    comp = CompositeExecutor(
        [
            (paddle, frozenset({JobKind.RECOGNITION})),
            (mineru, frozenset({JobKind.MINERU_PARSE})),
        ]
    )
    comp._dispatch["old-job"] = comp._children[0]  # type: ignore[index]
    comp.close()
    assert paddle.close_calls == 1
    assert mineru.close_calls == 1
    assert comp._dispatch == {}


def test_close_tolerates_child_close_error() -> None:
    """A child whose close() raises must not break close() for the others."""
    boom = _StubExecutor(close_raises=True)
    ok = _StubExecutor()
    comp = CompositeExecutor(
        [
            (boom, frozenset({JobKind.RECOGNITION})),
            (ok, frozenset({JobKind.MINERU_PARSE})),
        ]
    )
    comp.close()  # must not raise
    assert ok.close_calls == 1


def test_release_idle_tolerates_child_error() -> None:
    boom = _StubExecutor(release_raises=True)
    ok = _StubExecutor()
    comp = CompositeExecutor(
        [
            (boom, frozenset({JobKind.RECOGNITION})),
            (ok, frozenset({JobKind.MINERU_PARSE})),
        ]
    )
    comp.release_idle("OCR")  # must not raise
    assert ok.release_calls == ["OCR"]
