"""Production execution wiring: scheduling, microbatching and recovery."""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any

from vibeocr.backend.supervisor.inference.budgets import (
    AdapterCapability,
    BudgetPlanner,
)
from vibeocr.backend.supervisor.inference.paddle_executor import AdapterExecutor
from vibeocr.backend.supervisor.inference.scheduler import DeviceScheduler
from vibeocr.backend.supervisor.jobs.registry import JobRegistry
from vibeocr.backend.supervisor.jobs.staging import StagedInput
from vibeocr.runtime_contracts import (
    ItemState,
    JobItem,
    JobKind,
    JobPriority,
    JobState,
)

if TYPE_CHECKING:
    from pathlib import Path


def _record(
    registry: JobRegistry,
    job_id: str,
    names: list[str],
    *,
    priority: JobPriority = JobPriority.INTERACTIVE,
) -> Any:
    record = registry.create(
        job_id=job_id,
        kind=JobKind.RECOGNITION,
        priority=priority,
        items=[
            JobItem(
                item_id=f"{job_id}-{index}",
                display_name=name,
                state=ItemState.QUEUED,
                ordinal=index,
            )
            for index, name in enumerate(names)
        ],
        progress_total=len(names),
    )
    record.transition(JobState.QUEUED)
    return record


def _staged(record: Any, root: Path) -> list[StagedInput]:
    staged: list[StagedInput] = []
    for item in record.items:
        path = root / f"{item.item_id}.bin"
        path.write_bytes(item.display_name.encode())
        staged.append(
            StagedInput(
                item_id=item.item_id,
                display_name=item.display_name,
                path=path,
                size_bytes=path.stat().st_size,
            )
        )
    return staged


class _BatchAdapter:
    def __init__(self, *, max_batch: int = 8) -> None:
        self.max_batch = max_batch
        self.calls: list[list[str]] = []

    def capabilities(self, options=None):
        del options
        return AdapterCapability(
            name="test", real_batch=True, max_compute_batch=self.max_batch
        )

    def recognize_many(self, items, *, options=None):
        del options
        self.calls.append([item.item_id for item in items])
        return [{"raw_text": item.display_name} for item in items]

    def residency_status(self):
        from vibeocr.runtime_contracts import ResidencyStatus

        return ResidencyStatus()

    def release_idle(self, pipeline=None):
        del pipeline
        return self.residency_status()


def test_shared_scheduler_serializes_two_executor_threads(tmp_path: Path) -> None:
    scheduler = DeviceScheduler(devices=["gpu:0"])
    entered = threading.Event()
    allow_exit = threading.Event()
    lock = threading.Lock()
    active = 0
    max_active = 0

    class BlockingAdapter(_BatchAdapter):
        def recognize_many(self, items, *, options=None):
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            entered.set()
            allow_exit.wait(timeout=2)
            try:
                return super().recognize_many(items, options=options)
            finally:
                with lock:
                    active -= 1

    adapter = BlockingAdapter(max_batch=1)
    executor = AdapterExecutor(
        adapter_factory=lambda: adapter,
        scheduler=scheduler,
    )
    registry = JobRegistry("test")
    first = _record(registry, "first", ["a"])
    second = _record(registry, "second", ["b"])
    threads = [
        threading.Thread(target=executor.execute, args=(first, _staged(first, tmp_path))),
        threading.Thread(
            target=executor.execute, args=(second, _staged(second, tmp_path))
        ),
    ]
    threads[0].start()
    assert entered.wait(timeout=1)
    threads[1].start()
    time.sleep(0.05)
    assert max_active == 1
    allow_exit.set()
    for thread in threads:
        thread.join(timeout=2)
        assert not thread.is_alive()
    assert max_active == 1
    assert first.state is JobState.COMPLETED
    assert second.state is JobState.COMPLETED


def test_oom_shrinks_microbatch_and_preserves_all_items(tmp_path: Path) -> None:
    class OomAdapter(_BatchAdapter):
        def recognize_many(self, items, *, options=None):
            self.calls.append([item.item_id for item in items])
            if len(items) > 2:
                raise RuntimeError("CUDA out of memory")
            return [{"raw_text": item.display_name} for item in items]

    clears: list[None] = []
    adapter = OomAdapter(max_batch=8)
    executor = AdapterExecutor(
        adapter_factory=lambda: adapter,
        clear_cache=lambda: clears.append(None),
    )
    record = _record(JobRegistry("test"), "oom", ["a", "b", "c", "d"])

    executor.execute(record, _staged(record, tmp_path))

    assert [len(call) for call in adapter.calls] == [4, 2, 2]
    assert len(clears) == 1
    assert record.state is JobState.COMPLETED
    assert record.degraded is True
    assert all(item.state is ItemState.SUCCEEDED for item in record.items)


def test_bad_input_bisection_fails_only_corrupt_item(tmp_path: Path) -> None:
    class IsolatingAdapter(_BatchAdapter):
        def recognize_many(self, items, *, options=None):
            self.calls.append([item.item_id for item in items])
            if any(item.display_name == "bad" for item in items):
                raise ValueError("invalid image: corrupt")
            return [{"raw_text": item.display_name} for item in items]

    adapter = IsolatingAdapter(max_batch=8)
    executor = AdapterExecutor(adapter_factory=lambda: adapter)
    record = _record(JobRegistry("test"), "bad", ["a", "bad", "c", "d"])

    executor.execute(record, _staged(record, tmp_path))

    assert record.state is JobState.COMPLETED_WITH_ERRORS
    assert record.degraded is True
    assert [item.state for item in record.items] == [
        ItemState.SUCCEEDED,
        ItemState.FAILED,
        ItemState.SUCCEEDED,
        ItemState.SUCCEEDED,
    ]
    assert record.item_errors["bad-1"] == "BAD_INPUT"


def test_transient_retry_budget_is_local_to_each_compute_batch(
    tmp_path: Path,
) -> None:
    class TransientAdapter(_BatchAdapter):
        def __init__(self) -> None:
            super().__init__(max_batch=2)
            self.failures_left = 2

        def recognize_many(self, items, *, options=None):
            self.calls.append([item.item_id for item in items])
            if self.failures_left:
                self.failures_left -= 1
                raise RuntimeError("backend temporarily unavailable")
            return [{"raw_text": item.display_name} for item in items]

    delays: list[float] = []
    adapter = TransientAdapter()
    executor = AdapterExecutor(
        adapter_factory=lambda: adapter,
        sleeper=delays.append,
        budget_planner=BudgetPlanner(max_file_count=2),
    )
    record = _record(JobRegistry("test"), "transient", ["a", "b"])

    executor.execute(record, _staged(record, tmp_path))

    assert delays == [0.25, 0.5]
    assert record.state is JobState.COMPLETED
    assert record.degraded is True
