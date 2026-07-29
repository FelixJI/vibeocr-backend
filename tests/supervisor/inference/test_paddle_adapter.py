"""Tests for PaddlePipelineAdapter: recognize_many, capability honesty, order."""

from __future__ import annotations

import io
import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import pytest
from PIL import Image

from vibeocr.backend.models.ocr_result import OCRResult, TextBlock
from vibeocr.backend.supervisor.inference.budgets import InputItem
from vibeocr.backend.supervisor.inference.paddle_adapter import PaddlePipelineAdapter
from vibeocr.runtime_contracts import PipelineSelection

if TYPE_CHECKING:
    import numpy as np


def _png_bytes(label: str, color: tuple[int, int, int] = (255, 0, 0)) -> bytes:
    img = Image.new("RGB", (8, 8), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
    # Attach a label so the fake service can echo it back.


class _FakeOCRService:
    """Records calls and returns one OCRResult per input in order.

    Mirrors the real ``OCRService.recognize_batch`` contract (returns
    ``list[OCRResult]``) so the adapter's serializer is exercised end-to-end.
    """

    def __init__(self) -> None:
        self.calls: list[int] = []  # batch sizes
        self.predict_calls = 0
        self.preload_calls: list[list[object]] = []
        self.cache_manager: Any | None = None

    def recognize_batch(self, images: list[np.ndarray], options=None) -> list[OCRResult]:
        self.predict_calls += 1
        self.calls.append(len(images))
        return [
            OCRResult(
                raw_text=f"text-{i}",
                pipeline_type="OCR",
                text_blocks=[TextBlock(text=f"text-{i}", score=0.9, bbox=(0.0, 0.0, 1.0, 1.0))],
            )
            for i in range(len(images))
        ]

    def preload_pipelines_sequential(self, pipelines: list[object]) -> dict[str, bool]:
        self.preload_calls.append(pipelines)
        return {getattr(pipeline, "value", str(pipeline)): True for pipeline in pipelines}


def _raw_items(*labels: str) -> list[InputItem]:
    items: list[InputItem] = []
    for index, label in enumerate(labels):
        raw = _png_bytes(label)
        items.append(
            InputItem(
                item_id=f"it-{index}",
                data=raw,
                encoded_bytes=len(raw),
                decoded_pixels=64,
                estimated_pages=1,
            )
        )
    return items


# ---------------------------------------------------------------------------
# Capability honesty
# ---------------------------------------------------------------------------


def test_capability_reports_not_real_batch_without_registry() -> None:
    adapter = PaddlePipelineAdapter(service=_FakeOCRService(), pipeline_name="PP-StructureV3")
    cap = adapter.capabilities()
    # Without the real pipeline registry (not importable in unit tests) we
    # conservatively report not-real-batch.
    assert cap.real_batch is False
    assert cap.max_compute_batch == 1


def test_capability_cached_after_first_call() -> None:
    adapter = PaddlePipelineAdapter(service=_FakeOCRService(), pipeline_name="OCR")
    cap1 = adapter.capabilities()
    cap2 = adapter.capabilities()
    assert cap1 is cap2


# ---------------------------------------------------------------------------
# recognize_many
# ---------------------------------------------------------------------------


def test_recognize_many_preserves_order() -> None:
    service = _FakeOCRService()
    adapter = PaddlePipelineAdapter(service=service, pipeline_name="OCR")
    items = _raw_items("a", "b", "c")
    results = adapter.recognize_many(items)
    # The fake returns OCRResult(raw_text="text-N"); the serializer surfaces
    # it as the structured `raw_text` key (not the old broken `text`+repr).
    assert [r["raw_text"] for r in results] == ["text-0", "text-1", "text-2"]
    # And text_blocks survived serialization as a list of dicts.
    assert all(isinstance(r["text_blocks"], list) and r["text_blocks"] for r in results)


def test_single_image_is_one_element_batch() -> None:
    service = _FakeOCRService()
    adapter = PaddlePipelineAdapter(service=service, pipeline_name="OCR")
    results = adapter.recognize_many(_raw_items("only"))
    assert len(results) == 1
    assert service.calls == [1]


def test_recognize_many_empty_returns_empty() -> None:
    adapter = PaddlePipelineAdapter(service=_FakeOCRService(), pipeline_name="OCR")
    assert adapter.recognize_many([]) == []


def test_recognize_many_uses_one_predict_call_for_batch() -> None:
    service = _FakeOCRService()
    adapter = PaddlePipelineAdapter(service=service, pipeline_name="OCR")
    adapter.recognize_many(_raw_items("a", "b", "c", "d"))
    assert service.predict_calls == 1
    assert service.calls == [4]


def test_recognize_many_logs_pipeline_items_result_and_elapsed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    adapter = PaddlePipelineAdapter(service=_FakeOCRService(), pipeline_name="OCR")

    with caplog.at_level(
        logging.INFO,
        logger="vibeocr.backend.supervisor.inference.paddle_adapter",
    ):
        adapter.recognize_many(
            _raw_items("a", "b"),
            options=PipelineSelection("PP-StructureV3"),
        )

    message = next(
        record.getMessage()
        for record in caplog.records
        if "[Supervisor][Recognize]" in record.getMessage()
    )
    assert "pipeline=PP-StructureV3" in message
    assert "items=2" in message
    assert "result=success" in message
    assert "elapsed_ms=" in message


def test_recognize_many_raises_on_missing_raw_bytes() -> None:
    adapter = PaddlePipelineAdapter(service=_FakeOCRService(), pipeline_name="OCR")
    # Plain InputItem has no .data attribute.
    plain = InputItem(item_id="x", encoded_bytes=10, decoded_pixels=10, estimated_pages=1)
    with pytest.raises(ValueError, match="no raw bytes"):
        adapter.recognize_many([plain])


def test_preload_delegates_to_existing_ocr_service_loader() -> None:
    service = _FakeOCRService()
    adapter = PaddlePipelineAdapter(service=service)

    adapter.preload(("OCR", "PP-StructureV3"))

    assert [
        getattr(pipeline, "value", None) for pipeline in service.preload_calls[0]
    ] == ["OCR", "PP-StructureV3"]


def test_preload_logs_each_pipeline_result_and_elapsed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    adapter = PaddlePipelineAdapter(service=_FakeOCRService())

    with caplog.at_level(
        logging.INFO,
        logger="vibeocr.backend.supervisor.inference.paddle_adapter",
    ):
        adapter.preload(("OCR", "PP-StructureV3"))

    messages = [
        record.getMessage()
        for record in caplog.records
        if "[Supervisor][Preload]" in record.getMessage()
    ]
    assert len(messages) == 2
    assert any(
        "pipeline=OCR" in message
        and "result=success" in message
        and "elapsed_ms=" in message
        for message in messages
    )
    assert any(
        "pipeline=PP-StructureV3" in message
        and "result=success" in message
        and "elapsed_ms=" in message
        for message in messages
    )


def test_preload_missing_result_is_reported_as_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class EmptyResultService(_FakeOCRService):
        def preload_pipelines_sequential(
            self, pipelines: list[object]
        ) -> dict[str, bool]:
            self.preload_calls.append(pipelines)
            return {}

    adapter = PaddlePipelineAdapter(service=EmptyResultService())

    with (
        caplog.at_level(
            logging.ERROR,
            logger="vibeocr.backend.supervisor.inference.paddle_adapter",
        ),
        pytest.raises(RuntimeError, match="OCR"),
    ):
        adapter.preload(("OCR",))

    assert any(
        "pipeline=OCR" in record.getMessage()
        and "result=failed" in record.getMessage()
        for record in caplog.records
    )


def test_recognize_many_releases_residency_lease() -> None:
    service = _FakeOCRService()

    class PhysicalCache:
        def __init__(self) -> None:
            self.active = 0
            self.max_active = 0

        @contextmanager
        def lease(self, pipeline):
            assert pipeline == "OCR"
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            try:
                yield
            finally:
                self.active -= 1

        def status(self):
            return {
                "loaded_pipelines": ["OCR"],
                "pipeline_ttls": {"OCR": 100},
                "active_counts": {"OCR": self.active},
                "pinned_pipelines": [],
                "last_used_unix_ms": {},
            }

    cache = PhysicalCache()
    service.cache_manager = cache
    adapter = PaddlePipelineAdapter(service=service, pipeline_name="OCR")
    adapter.recognize_many(_raw_items("a"))
    status = adapter.residency_status()
    entry = next(e for e in status.entries if e.pipeline == "OCR")
    assert entry.active_leases == 0
    assert cache.max_active == 1


def test_result_payload_passes_through_dict_results() -> None:
    class _DictService:
        def recognize_batch(self, images, options=None):
            return [{"text": "raw"} for _ in images]

        def preload_pipelines_sequential(self, pipelines):
            return {str(pipeline): True for pipeline in pipelines}

    adapter = PaddlePipelineAdapter(service=_DictService(), pipeline_name="OCR")
    results = adapter.recognize_many(_raw_items("a"))
    assert results == [{"text": "raw"}]
