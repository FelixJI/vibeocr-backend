"""Deterministic unit tests for PaddleExecutor (the supervisor↔adapter bridge).

These exercise the bridge logic (queued→running→terminal, per-item mapping,
cancel handling, whole-batch failure) with a fake adapter, so the executor is
covered without the heavy real-model integration test.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vibeocr.backend.supervisor.inference.paddle_adapter import PaddlePipelineAdapter
from vibeocr.backend.supervisor.inference.paddle_executor import PaddleExecutor
from vibeocr.backend.supervisor.jobs.registry import JobRegistry
from vibeocr.backend.supervisor.jobs.staging import StagedInput
from vibeocr.runtime_contracts import ItemState, JobKind, JobPriority, JobState

if TYPE_CHECKING:
    from pathlib import Path


class _FakeService:
    """Returns one OCR-result-shaped dict per input, in order."""

    def __init__(self, texts: list[str], *, fail: bool = False) -> None:
        self._texts = texts
        self._fail = fail

    def recognize_batch(
        self, images: list[Any], options: Any | None = None
    ) -> list[dict[str, Any]]:
        if self._fail:
            raise RuntimeError("OOM during predict")
        return [{"text": t} for t in self._texts]

    def preload_pipelines_sequential(self, pipelines: list[Any]) -> dict[str, bool]:
        return {str(pipeline): True for pipeline in pipelines}


def _make_adapter(service: _FakeService) -> PaddlePipelineAdapter:
    return PaddlePipelineAdapter(service=service)


def _make_job(registry: JobRegistry, items: int) -> Any:
    record = registry.create(
        kind=JobKind.RECOGNITION,
        priority=JobPriority.INTERACTIVE,
        items=[
            __import__("vibeocr.runtime_contracts", fromlist=["JobItem"]).JobItem(
                item_id=f"it-{i}",
                display_name=f"f{i}.png",
                state=__import__(
                    "vibeocr.runtime_contracts", fromlist=["ItemState"]
                ).ItemState.QUEUED,
            )
            for i in range(items)
        ],
        progress_total=items,
    )
    record.transition(JobState.QUEUED)
    return record


def _valid_png_bytes() -> bytes:
    """A minimal 1x1 valid PNG so the real adapter's PIL decode succeeds."""
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (1, 1)).save(buf, format="PNG")
    return buf.getvalue()


def _staged(items: int, base: Path) -> list[StagedInput]:
    out: list[StagedInput] = []
    png = _valid_png_bytes()
    for i in range(items):
        p = base / f"f{i}.png"
        p.write_bytes(png)
        out.append(
            StagedInput(
                item_id=f"it-{i}", display_name=f"f{i}.png", path=p, size_bytes=len(png)
            )
        )
    return out


def test_execute_runs_job_to_completed_and_maps_results(tmp_path, monkeypatch) -> None:
    png = _valid_png_bytes()
    for i in range(2):
        (tmp_path / f"f{i}.png").write_bytes(png)
    reg = JobRegistry(instance_id="t")
    record = _make_job(reg, 2)
    staged = [
        StagedInput(
            item_id=f"it-{i}",
            display_name=f"f{i}.png",
            path=tmp_path / f"f{i}.png",
            size_bytes=len(png),
        )
        for i in range(2)
    ]
    executor = PaddleExecutor(
        adapter_factory=lambda: _make_adapter(_FakeService(["alpha", "beta"]))
    )
    executor.execute(record, staged)
    snap = record.snapshot()
    assert snap.state is JobState.COMPLETED
    assert [it.state for it in snap.items] == [ItemState.SUCCEEDED, ItemState.SUCCEEDED]
    results = [record.results[f"it-{i}"] for i in range(2)]
    assert [r["text"] for r in results] == ["alpha", "beta"]


def test_execute_isolates_whole_batch_failure(tmp_path: Path) -> None:
    reg = JobRegistry(instance_id="t")
    record = _make_job(reg, 2)
    executor = PaddleExecutor(
        adapter_factory=lambda: _make_adapter(_FakeService(["x"], fail=True))
    )
    executor.execute(record, _staged(2, tmp_path))
    snap = record.snapshot()
    assert snap.state is JobState.FAILED


def test_execute_honours_cancel_before_run(tmp_path: Path) -> None:
    reg = JobRegistry(instance_id="t")
    record = _make_job(reg, 1)
    # Simulate cancel requested before the executor runs.
    record.cancel_requested_at = "2026-07-25T00:00:00+00:00"
    record.transition(JobState.CANCEL_REQUESTED)
    executor = PaddleExecutor(
        adapter_factory=lambda: _make_adapter(_FakeService(["x"]))
    )
    executor.execute(record, _staged(1, tmp_path))
    assert record.snapshot().state is JobState.CANCELLED


def test_execute_empty_items_completes() -> None:
    reg = JobRegistry(instance_id="t")
    record = _make_job(reg, 0)
    executor = PaddleExecutor(adapter_factory=lambda: _make_adapter(_FakeService([])))
    executor.execute(record, [])
    assert record.snapshot().state is JobState.COMPLETED
