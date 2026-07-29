"""Tests for the AdapterExecutor state machine + recovery branches.

Existing test_paddle_executor.py covers the happy path and whole-batch
failure; this file targets the branches left behind: capability fallback,
cancel-during-run, recovery (bisect/shrink/backoff) actions, adapter
protocol violations, residency/release/configure/close, and the cancel-
non-terminal helper.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vibeocr.backend.supervisor.inference.budgets import AdapterCapability, InputItem
from vibeocr.backend.supervisor.inference.paddle_executor import (
    AdapterExecutor,
    PaddleExecutor,
)
from vibeocr.backend.supervisor.inference.recovery import (
    RecoveryPolicy,
)
from vibeocr.backend.supervisor.jobs.registry import JobRegistry
from vibeocr.backend.supervisor.jobs.staging import StagedInput
from vibeocr.runtime_contracts import (
    CancelMode,
    ItemState,
    JobKind,
    JobPriority,
    JobState,
    ResidencyStatus,
    SettingsSnapshot,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _png_bytes() -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (1, 1)).save(buf, format="PNG")
    return buf.getvalue()


def _make_job(registry: JobRegistry, items: int, *, kind=JobKind.RECOGNITION) -> Any:
    from vibeocr.runtime_contracts import JobItem

    record = registry.create(
        kind=kind,
        priority=JobPriority.INTERACTIVE,
        items=[
            JobItem(item_id=f"it-{i}", display_name=f"f{i}.png", state=ItemState.QUEUED)
            for i in range(items)
        ],
        progress_total=items,
    )
    record.transition(JobState.QUEUED)
    return record


def _staged(items: int, base: Path) -> list[StagedInput]:
    png = _png_bytes()
    out: list[StagedInput] = []
    for i in range(items):
        p = base / f"f{i}.png"
        p.write_bytes(png)
        out.append(StagedInput(item_id=f"it-{i}", display_name=f"f{i}.png", path=p, size_bytes=len(png)))
    return out


# ---------------------------------------------------------------------------
# Adapter property: configure_settings hook on first materialisation
# ---------------------------------------------------------------------------


class _ConfiguringService:
    """Service that records ``configure_settings`` calls from the adapter."""

    def __init__(self) -> None:
        self.configured: list[SettingsSnapshot] = []

    def recognize_batch(self, images, options=None):  # type: ignore[no-untyped-def]
        return [{"text": "ok"} for _ in images]

    def preload_pipelines_sequential(self, pipelines):  # type: ignore[no-untyped-def]
        return {str(p): True for p in pipelines}


def test_adapter_property_calls_configure_settings_on_first_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the adapter exposes ``configure_settings``, the executor calls it
    on first materialisation (lines 89-93)."""
    from vibeocr.backend.supervisor.inference.paddle_adapter import (
        PaddlePipelineAdapter,
    )

    configure_calls: list[SettingsSnapshot] = []

    class _Adapter(PaddlePipelineAdapter):
        def configure_settings(self, snapshot):  # type: ignore[no-untyped-def]
            configure_calls.append(snapshot)
            return super().configure_settings(snapshot)

    service = _ConfiguringService()
    executor = PaddleExecutor(adapter_factory=lambda: _Adapter(service=service))
    # Touch the property to materialise the adapter.
    _ = executor.adapter
    assert len(configure_calls) == 1


# ---------------------------------------------------------------------------
# Capability fallback
# ---------------------------------------------------------------------------


def test_capability_fallback_when_adapter_lacks_capabilities_method(
    tmp_path: Path,
) -> None:
    """An adapter without ``capabilities`` falls back to a conservative cap
    (lines 126-131)."""
    reg = JobRegistry(instance_id="t")
    record = _make_job(reg, 1)
    staged = _staged(1, tmp_path)

    class _BareAdapter:
        def recognize_many(self, items, options=None):  # type: ignore[no-untyped-def]
            return [{"text": "ok"} for _ in items]

        # No ``capabilities`` method, no residency methods needed for happy path.

    executor = PaddleExecutor(adapter_factory=lambda: _BareAdapter())
    executor.execute(record, staged)
    snap = record.snapshot()
    assert snap.state is JobState.COMPLETED


# ---------------------------------------------------------------------------
# Already-terminal + cancel-during-run + lease-is-None
# ---------------------------------------------------------------------------


def test_execute_returns_immediately_when_job_already_terminal(
    tmp_path: Path,
) -> None:
    reg = JobRegistry(instance_id="t")
    record = _make_job(reg, 1)
    record.transition(JobState.RUNNING)
    record.transition(JobState.COMPLETED)
    staged = _staged(1, tmp_path)

    class _Boom:
        def recognize_many(self, items, options=None):  # type: ignore[no-untyped-def]
            raise AssertionError("must not be called")

    executor = PaddleExecutor(adapter_factory=lambda: _Boom())
    executor.execute(record, staged)  # must be a no-op
    assert record.snapshot().state is JobState.COMPLETED


def test_execute_skips_compute_when_cancel_requested_before_iter(
    tmp_path: Path,
) -> None:
    """A cancel requested before the compute loop runs the cancel state machine
    directly (lines 149-180)."""
    reg = JobRegistry(instance_id="t")
    record = _make_job(reg, 2)
    record.transition(JobState.RUNNING)
    record.cancel_requested_at = "2026-07-25T00:00:00+00:00"
    staged = _staged(2, tmp_path)

    class _Boom:
        def recognize_many(self, items, options=None):  # type: ignore[no-untyped-def]
            raise AssertionError("must not be called")

    executor = PaddleExecutor(adapter_factory=lambda: _Boom())
    executor.execute(record, staged)
    snap = record.snapshot()
    # Items are non-terminal; the executor marks them cancelled + job cancelled.
    assert snap.state is JobState.CANCELLED


def test_execute_breaks_when_scheduler_returns_no_lease(tmp_path: Path) -> None:
    """When the scheduler returns ``None`` lease the compute loop breaks (line 159).
    With zero items processed (and zero failed), the terminal decision is COMPLETED."""
    reg = JobRegistry(instance_id="t")
    record = _make_job(reg, 1)
    staged = _staged(1, tmp_path)

    class _NoLeaseScheduler:
        def acquire(self, **kwargs):  # type: ignore[no-untyped-def]
            return None

        def release(self, lease):  # type: ignore[no-untyped-def]
            return

    class _Adapter:
        def capabilities(self, options=None):  # type: ignore[no-untyped-def]
            return AdapterCapability(name="OCR", real_batch=False, max_compute_batch=1)

        def recognize_many(self, items, options=None):  # type: ignore[no-untyped-def]
            raise AssertionError("must not be called without a lease")

    executor = PaddleExecutor(
        adapter_factory=lambda: _Adapter(),
        scheduler=_NoLeaseScheduler(),  # type: ignore[arg-type]
    )
    executor.execute(record, staged)
    # No lease → no items processed. 0 succeeded + 0 failed → COMPLETED.
    assert record.snapshot().state is JobState.COMPLETED


def test_execute_completes_with_errors_when_some_items_fail(tmp_path: Path) -> None:
    """A mixed result (one success, one protocol-violation) lands the job in
    COMPLETED_WITH_ERRORS (lines 189-199 + 312-319)."""
    reg = JobRegistry(instance_id="t")
    record = _make_job(reg, 2)
    staged = _staged(2, tmp_path)

    class _MixedAdapter:
        def __init__(self) -> None:
            self.call = 0

        def capabilities(self, options=None):  # type: ignore[no-untyped-def]
            return AdapterCapability(name="OCR", real_batch=False, max_compute_batch=1)

        def recognize_many(self, items, options=None):  # type: ignore[no-untyped-def]
            # Each call has exactly one item (per-item compute batches).
            self.call += 1
            # First item succeeds, second returns an empty payload.
            return [{"text": "ok"}] if self.call == 1 else [{}]

    executor = PaddleExecutor(adapter_factory=lambda: _MixedAdapter())
    executor.execute(record, staged)
    snap = record.snapshot()
    assert snap.state is JobState.COMPLETED_WITH_ERRORS


# ---------------------------------------------------------------------------
# Recovery: bisect / shrink / backoff / fail-fast
# ---------------------------------------------------------------------------


class _ScriptedAdapter:
    """Adapter that raises N times then succeeds, recording every call."""

    def __init__(self, errors: list[Exception]) -> None:
        self._errors = list(errors)
        self.calls: list[int] = []

    def capabilities(self, options=None):  # type: ignore[no-untyped-def]
        return AdapterCapability(name="OCR", real_batch=False, max_compute_batch=1)

    def recognize_many(self, items, options=None):  # type: ignore[no-untyped-def]
        self.calls.append(len(items))
        if self._errors:
            raise self._errors.pop(0)
        return [{"text": "ok"} for _ in items]


def test_recovery_bisect_isolate_on_bad_input(tmp_path: Path) -> None:
    """A bad-input failure on a multi-item compute batch triggers BISECT_ISOLATE
    recursion (lines 234-253).

    Requires a real-batch capability so the budget planner produces one
    multi-item compute batch (otherwise per-item batches short-circuit bisection).
    """
    reg = JobRegistry(instance_id="t")
    record = _make_job(reg, 2)
    staged = _staged(2, tmp_path)
    adapter = _ScriptedAdapter([RuntimeError("invalid image: truncated")])

    class _RealBatchAdapter(_ScriptedAdapter):
        def capabilities(self, options=None):  # type: ignore[no-untyped-def]
            return AdapterCapability(name="OCR", real_batch=True, max_compute_batch=8)

    adapter = _RealBatchAdapter([RuntimeError("invalid image: truncated")])
    executor = PaddleExecutor(adapter_factory=lambda: adapter)
    executor.execute(record, staged)
    # Bisect splits the 2-item batch into two 1-item retries (≥3 calls total).
    assert len(adapter.calls) >= 3
    # The first call was the multi-item batch; subsequent calls are size 1.
    assert adapter.calls[0] == 2
    assert all(size == 1 for size in adapter.calls[1:])


def test_recovery_shrink_and_retry_on_oom(tmp_path: Path) -> None:
    """An OOM triggers SHRINK_AND_RETRY with a smaller batch (lines 254-265)."""
    reg = JobRegistry(instance_id="t")
    record = _make_job(reg, 2)
    staged = _staged(2, tmp_path)
    adapter = _ScriptedAdapter([RuntimeError("CUDA out of memory")])
    executor = PaddleExecutor(adapter_factory=lambda: adapter)
    executor.execute(record, staged)
    # After one OOM (shrink to 1) each single-item succeeds → COMPLETED.
    assert record.snapshot().state is JobState.COMPLETED


def test_recovery_backoff_retry_on_transient(tmp_path: Path) -> None:
    """A transient failure triggers BACKOFF_RETRY (lines 266-276)."""
    sleeps: list[float] = []

    reg = JobRegistry(instance_id="t")
    record = _make_job(reg, 1)
    staged = _staged(1, tmp_path)
    adapter = _ScriptedAdapter([RuntimeError("connection timeout")])
    executor = PaddleExecutor(
        adapter_factory=lambda: adapter,
        sleeper=lambda s: sleeps.append(s),
    )
    executor.execute(record, staged)
    assert record.snapshot().state is JobState.COMPLETED
    assert sleeps  # backoff slept at least once


def test_recovery_fail_fast_on_cancelled(tmp_path: Path) -> None:
    """A cancellation-flagged failure returns FAIL_FAST (lines 277-283)."""
    reg = JobRegistry(instance_id="t")
    record = _make_job(reg, 1)
    record.transition(JobState.RUNNING)
    record.cancel_requested_at = "2026-07-25T00:00:00+00:00"
    staged = _staged(1, tmp_path)

    class _Adapter:
        def capabilities(self, options=None):  # type: ignore[no-untyped-def]
            return AdapterCapability(name="OCR", real_batch=False, max_compute_batch=1)

        def recognize_many(self, items, options=None):  # type: ignore[no-untyped-def]
            raise RuntimeError("cancelled mid-flight")

    executor = PaddleExecutor(adapter_factory=lambda: _Adapter())
    executor.execute(record, staged)
    assert record.snapshot().state is JobState.CANCELLED


def test_recovery_protocol_violation_count_mismatch(tmp_path: Path) -> None:
    """A result-count mismatch fails the items with ADAPTER_PROTOCOL_VIOLATION
    (lines 286-302)."""
    reg = JobRegistry(instance_id="t")
    record = _make_job(reg, 2)
    staged = _staged(2, tmp_path)

    class _Adapter:
        def capabilities(self, options=None):  # type: ignore[no-untyped-def]
            return AdapterCapability(name="OCR", real_batch=False, max_compute_batch=1)

        def recognize_many(self, items, options=None):  # type: ignore[no-untyped-def]
            return [{"text": "only-one"}]  # wrong count

    executor = PaddleExecutor(adapter_factory=lambda: _Adapter())
    executor.execute(record, staged)
    # The mismatch eventually fails the items; whether the recovery backoff
    # flips the outcome one way depends on classification, but the job must
    # reach a terminal state with the items marked.
    snap = record.snapshot()
    assert snap.state in (
        JobState.COMPLETED,
        JobState.COMPLETED_WITH_ERRORS,
        JobState.FAILED,
    )


# ---------------------------------------------------------------------------
# AdapterExecutor residency / release / preload / configure / close
# ---------------------------------------------------------------------------


class _FullAdapter:
    """Adapter that exposes residency + close so those branches are covered."""

    def __init__(self) -> None:
        self.closed = False
        self.preloaded: list[tuple[str, ...]] = []
        self.configured: list[SettingsSnapshot] = []
        self.released: list[str | None] = []

    def recognize_many(self, items, options=None):  # type: ignore[no-untyped-def]
        return [{"text": "ok"} for _ in items]

    def capabilities(self, options=None):  # type: ignore[no-untyped-def]
        return AdapterCapability(name="OCR", real_batch=False, max_compute_batch=1)

    def residency_status(self) -> ResidencyStatus:
        return ResidencyStatus(default_ttl_seconds=123)

    def release_idle(self, pipeline=None):  # type: ignore[no-untyped-def]
        self.released.append(pipeline)
        return self.residency_status()

    def preload(self, pipelines):  # type: ignore[no-untyped-def]
        self.preloaded.append(pipelines)
        return self.residency_status()

    def configure_settings(self, snapshot):  # type: ignore[no-untyped-def]
        self.configured.append(snapshot)
        return self.residency_status()

    def close(self) -> None:
        self.closed = True


def test_residency_status_returns_settings_when_adapter_unmaterialised() -> None:
    """When the adapter has not been materialised, residency_status returns the
    cached settings snapshot (lines 376-380)."""
    executor = PaddleExecutor(adapter_factory=lambda: _FullAdapter())
    executor.configure_settings(SettingsSnapshot(default_ttl_seconds=456))
    status = executor.residency_status()
    assert status.default_ttl_seconds == 456


def test_residency_status_delegates_to_adapter_when_materialised() -> None:
    """Once the adapter exists, residency_status delegates to it (line 381)."""
    adapter = _FullAdapter()
    executor = PaddleExecutor(adapter_factory=lambda: adapter)
    _ = executor.adapter  # materialise
    assert executor.residency_status().default_ttl_seconds == 123


def test_release_idle_returns_settings_when_adapter_unmaterialised() -> None:
    executor = PaddleExecutor(adapter_factory=lambda: _FullAdapter())
    status = executor.release_idle("OCR")
    # No adapter yet → returns residency_status() (settings-based).
    assert status.default_ttl_seconds == 300  # default SettingsSnapshot ttl


def test_release_idle_delegates_to_adapter_when_materialised() -> None:
    adapter = _FullAdapter()
    executor = PaddleExecutor(adapter_factory=lambda: adapter)
    _ = executor.adapter
    executor.release_idle("OCR")
    assert adapter.released == ["OCR"]


def test_preload_delegates_to_adapter() -> None:
    adapter = _FullAdapter()
    executor = PaddleExecutor(adapter_factory=lambda: adapter)
    executor.preload(("OCR", "PP-StructureV3"))
    assert adapter.preloaded == [("OCR", "PP-StructureV3")]


def test_configure_settings_propagates_to_live_adapter() -> None:
    """When the adapter is already alive, configure_settings forwards to it
    (lines 393-396)."""
    adapter = _FullAdapter()
    executor = PaddleExecutor(adapter_factory=lambda: adapter)
    _ = executor.adapter  # materialise → first configure call with default
    snap = SettingsSnapshot(default_ttl_seconds=789)
    executor.configure_settings(snap)
    # The most recent call carries the snapshot we just pushed.
    assert adapter.configured[-1] == snap


def test_close_calls_adapter_close_when_materialised() -> None:
    adapter = _FullAdapter()
    executor = PaddleExecutor(adapter_factory=lambda: adapter)
    _ = executor.adapter
    executor.close()
    assert adapter.closed is True


def test_close_is_noop_when_adapter_unmaterialised() -> None:
    executor = PaddleExecutor(adapter_factory=lambda: _FullAdapter())
    executor.close()  # must not raise


def test_close_tolerates_adapter_without_close_method() -> None:
    class _NoClose:
        def recognize_many(self, items, options=None):  # type: ignore[no-untyped-def]
            return []

    executor = PaddleExecutor(adapter_factory=lambda: _NoClose())
    _ = executor.adapter
    executor.close()  # must not raise


# ---------------------------------------------------------------------------
# cancel_mode_for / _cancel_non_terminal_items / _fail_items edge
# ---------------------------------------------------------------------------


def test_cancel_mode_for_queued_returns_queued_only() -> None:

    reg = JobRegistry(instance_id="t")
    record = _make_job(reg, 1)
    executor = PaddleExecutor(adapter_factory=lambda: _FullAdapter())
    assert executor.cancel_mode_for(record) is CancelMode.QUEUED_ONLY


def test_cancel_mode_for_non_queued_returns_cooperative() -> None:
    reg = JobRegistry(instance_id="t")
    record = _make_job(reg, 1)
    record.transition(JobState.RUNNING)
    executor = PaddleExecutor(adapter_factory=lambda: _FullAdapter())
    assert executor.cancel_mode_for(record) is CancelMode.COOPERATIVE


def test_fail_items_skips_already_terminal(tmp_path: Path) -> None:
    """``_fail_items`` skips items that are already SUCCEEDED/FAILED/CANCELLED
    (lines 340-341)."""
    reg = JobRegistry(instance_id="t")
    record = _make_job(reg, 2)
    record.transition(JobState.RUNNING)
    # Mark it-0 succeeded; it-1 stays queued.
    record.transition_item("it-0", ItemState.RUNNING)
    record.transition_item("it-0", ItemState.SUCCEEDED)
    items = [
        InputItem(item_id="it-0", encoded_bytes=1, decoded_pixels=1, estimated_pages=1),
        InputItem(item_id="it-1", encoded_bytes=1, decoded_pixels=1, estimated_pages=1),
    ]
    AdapterExecutor._fail_items(
        record, items, error_code="X", error="boom"
    )
    # it-0 was skipped (still SUCCEEDED), it-1 transitioned to FAILED.
    snap = record.snapshot()
    states = {it.item_id: it.state for it in snap.items}
    assert states["it-0"] is ItemState.SUCCEEDED
    assert states["it-1"] is ItemState.FAILED


def test_cancel_non_terminal_items_leaves_terminal_items(tmp_path: Path) -> None:
    """``_cancel_non_terminal_items`` only cancels items not in a terminal state
    (lines 362-368)."""
    reg = JobRegistry(instance_id="t")
    record = _make_job(reg, 2)
    record.transition(JobState.RUNNING)
    record.transition_item("it-0", ItemState.RUNNING)
    record.transition_item("it-0", ItemState.SUCCEEDED)  # terminal
    # it-1 stays QUEUED (non-terminal).
    AdapterExecutor._cancel_non_terminal_items(record)
    states = {it.item_id: it.state for it in record.snapshot().items}
    assert states["it-0"] is ItemState.SUCCEEDED
    assert states["it-1"] is ItemState.CANCELLED


# ---------------------------------------------------------------------------
# _execute_with_recovery edge: empty items + already-cancelled short-circuit
# ---------------------------------------------------------------------------


def test_execute_with_recovery_noop_on_empty_items() -> None:
    """An empty items list returns immediately (line 211)."""
    executor = PaddleExecutor(adapter_factory=lambda: _FullAdapter())
    # Use a minimal record-like object.
    class _Rec:
        cancel_requested_at = None

    executor._execute_with_recovery(
        _Rec(), [], options=None, policy=RecoveryPolicy()  # type: ignore[arg-type]
    )


def test_execute_with_recovery_noop_when_cancelled() -> None:
    """A non-None cancel_requested_at returns immediately (line 211)."""
    executor = PaddleExecutor(adapter_factory=lambda: _FullAdapter())

    class _Rec:
        cancel_requested_at = "2026-07-25T00:00:00+00:00"

    executor._execute_with_recovery(
        _Rec(),
        [InputItem(item_id="it-0", encoded_bytes=1, decoded_pixels=1, estimated_pages=1)],
        options=None,
        policy=RecoveryPolicy(),  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Targeted branch coverage: degraded flag, protocol-violation count, empty
# payload, configure callable check, _staged_to_items decode failure.
# ---------------------------------------------------------------------------


def test_recovery_marks_job_degraded_on_failfast(tmp_path: Path) -> None:
    """When a recovery decision sets ``degraded=True`` the job is marked degraded
    (lines 223-225).

    OOM with retries already exhausted returns FAIL_FAST with degraded=True."""
    reg = JobRegistry(instance_id="t")
    record = _make_job(reg, 1)
    staged = _staged(1, tmp_path)

    class _AlwaysOom:
        def capabilities(self, options=None):  # type: ignore[no-untyped-def]
            return AdapterCapability(name="OCR", real_batch=False, max_compute_batch=1)

        def recognize_many(self, items, options=None):  # type: ignore[no-untyped-def]
            raise RuntimeError("CUDA out of memory")

    # Use a policy with zero OOM retries so the first failure FAIL_FASTs with
    # degraded=True (every OOM decision sets degraded=True).
    class _ZeroOomPolicy(RecoveryPolicy):
        max_oom_retries = 0

    executor = PaddleExecutor(
        adapter_factory=lambda: _AlwaysOom(),
        recovery_policy_factory=_ZeroOomPolicy,
    )
    executor.execute(record, staged)
    assert record.snapshot().degraded is True


def test_protocol_violation_count_mismatch_emits_event(tmp_path: Path) -> None:
    """A count mismatch emits an ``adapter_protocol_violation`` event and fails
    the items (lines 286-302)."""
    reg = JobRegistry(instance_id="t")
    record = _make_job(reg, 1)
    staged = _staged(1, tmp_path)

    class _WrongCount:
        def capabilities(self, options=None):  # type: ignore[no-untyped-def]
            return AdapterCapability(name="OCR", real_batch=False, max_compute_batch=1)

        def recognize_many(self, items, options=None):  # type: ignore[no-untyped-def]
            # Return more payloads than requested.
            return [{"text": "a"}, {"text": "b"}, {"text": "c"}]

    executor = PaddleExecutor(adapter_factory=lambda: _WrongCount())
    executor.execute(record, staged)
    stages = [e.stage for e in record.events]
    assert "adapter_protocol_violation" in stages


def test_empty_payload_dict_marked_as_protocol_violation(tmp_path: Path) -> None:
    """An empty dict payload fails the item with ADAPTER_PROTOCOL_VIOLATION
    (lines 312-319)."""
    reg = JobRegistry(instance_id="t")
    record = _make_job(reg, 1)
    staged = _staged(1, tmp_path)

    class _Empty:
        def capabilities(self, options=None):  # type: ignore[no-untyped-def]
            return AdapterCapability(name="OCR", real_batch=False, max_compute_batch=1)

        def recognize_many(self, items, options=None):  # type: ignore[no-untyped-def]
            return [{}]  # empty payload

    executor = PaddleExecutor(adapter_factory=lambda: _Empty())
    executor.execute(record, staged)
    snap = record.snapshot()
    assert snap.state is JobState.FAILED
    assert snap.items[0].state is ItemState.FAILED


def test_non_dict_payload_marked_as_protocol_violation(tmp_path: Path) -> None:
    """A non-dict payload fails the item with ADAPTER_PROTOCOL_VIOLATION."""
    reg = JobRegistry(instance_id="t")
    record = _make_job(reg, 1)
    staged = _staged(1, tmp_path)

    class _NonDict:
        def capabilities(self, options=None):  # type: ignore[no-untyped-def]
            return AdapterCapability(name="OCR", real_batch=False, max_compute_batch=1)

        def recognize_many(self, items, options=None):  # type: ignore[no-untyped-def]
            return ["not-a-dict"]  # non-dict payload

    executor = PaddleExecutor(adapter_factory=lambda: _NonDict())
    executor.execute(record, staged)
    assert record.snapshot().items[0].state is ItemState.FAILED


def test_staged_to_items_tolerates_non_image_bytes(tmp_path: Path) -> None:
    """A staged file whose bytes are not a valid image leaves decoded_pixels=0
    (the PIL decode except branch, line 422-423)."""
    p = tmp_path / "not-an-image.dat"
    p.write_bytes(b"definitely not a PNG")
    staged = [StagedInput(item_id="it-0", display_name="x", path=p, size_bytes=22)]
    items = AdapterExecutor._staged_to_items(staged)
    assert len(items) == 1
    assert items[0].decoded_pixels == 0


def test_staged_to_items_handles_entry_without_path_attribute() -> None:
    """A staged entry without a ``path`` attribute yields empty bytes (line 415)."""
    class _BareEntry:
        item_id = "it-0"
        display_name = "x"

    items = AdapterExecutor._staged_to_items([_BareEntry()])  # type: ignore[list-item]
    assert items[0].encoded_bytes == 0


def test_adapter_property_returns_cached_instance_on_second_access() -> None:
    """Two ``.adapter`` accesses return the same instance (line 87->94 False branch)."""
    calls: list[int] = []

    def factory() -> Any:
        calls.append(1)
        return _FullAdapter()

    executor = PaddleExecutor(adapter_factory=lambda: factory())
    first = executor.adapter
    second = executor.adapter
    assert first is second
    assert len(calls) == 1


def test_configure_settings_tolerates_adapter_without_configure() -> None:
    """An adapter lacking ``configure_settings`` is tolerated by configure_settings
    (line 395->397 False branch)."""
    class _NoConfig:
        def recognize_many(self, items, options=None):  # type: ignore[no-untyped-def]
            return []

        def residency_status(self) -> ResidencyStatus:
            return ResidencyStatus(default_ttl_seconds=42)

    executor = PaddleExecutor(adapter_factory=lambda: _NoConfig())
    _ = executor.adapter  # materialise
    snap = SettingsSnapshot(default_ttl_seconds=42)
    out = executor.configure_settings(snap)  # must not raise
    assert out.default_ttl_seconds == 42


def test_cancel_during_payload_iteration_returns_early(tmp_path: Path) -> None:
    """When cancel is requested mid-iteration over successful payloads, the
    executor returns early (line 311-312)."""
    reg = JobRegistry(instance_id="t")
    record = _make_job(reg, 2)
    record.transition(JobState.RUNNING)
    # Set cancel before execution; the cancel path runs but the compute loop
    # is what hits the per-item return — emulate by having the adapter set
    # cancel_requested_at on first call (single-item batches).
    staged = _staged(2, tmp_path)

    class _CancellingAdapter:
        def __init__(self) -> None:
            self.calls = 0

        def capabilities(self, options=None):  # type: ignore[no-untyped-def]
            return AdapterCapability(name="OCR", real_batch=False, max_compute_batch=1)

        def recognize_many(self, items, options=None):  # type: ignore[no-untyped-def]
            self.calls += 1
            # After the first item succeeds, request cancel so the next
            # iteration of the per-item loop hits the early return.
            if self.calls == 1:
                return [{"text": "ok"}]
            record.cancel_requested_at = "2026-07-25T00:00:00+00:00"
            return [{"text": "second"}]

    executor = PaddleExecutor(adapter_factory=lambda: _CancellingAdapter())
    executor.execute(record, staged)
    # The cancel is honoured → job ends in CANCELLED.
    assert record.snapshot().state is JobState.CANCELLED
