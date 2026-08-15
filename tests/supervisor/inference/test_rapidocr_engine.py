"""RapidOcrEngine adapter 测试：fake rapidocr 模块驱动，不依赖真实安装。

覆盖计划 §B3 验收的可测部分：descriptor 探测、payload 映射
（text/score/polygon 归一化）、空结果、初始化失败记忆与复位。
"""

from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np
import pytest
from vibeocr.backend.supervisor.inference.budgets import InputItem
from vibeocr.backend.supervisor.inference.ocr_engines import (
    REASON_ENGINE_INIT_FAILED,
    REASON_ENGINE_NOT_INSTALLED,
    EngineAvailability,
)
from vibeocr.backend.supervisor.inference.rapidocr_engine import RapidOcrEngine


class _FakeRapidOutput:
    """模拟 rapidocr v3 的 RapidOCROutput：txts/scores/boxes(N×4×2)。"""

    def __init__(self, txts, scores, boxes) -> None:
        self.txts = tuple(txts) if txts is not None else None
        self.scores = tuple(scores) if scores is not None else None
        self.boxes = np.array(boxes, dtype=float) if boxes is not None else None


class _FakeRapidOCR:
    instances: list["_FakeRapidOCR"] = []
    responses: list[Any] = []
    init_error: Exception | None = None

    def __init__(self, **params: Any) -> None:
        self.params = params
        self.calls: list[Any] = []
        _FakeRapidOCR.instances.append(self)
        if _FakeRapidOCR.init_error is not None:
            raise _FakeRapidOCR.init_error

    def __call__(self, image: Any) -> Any:
        self.calls.append(image)
        if _FakeRapidOCR.init_error is not None:
            raise _FakeRapidOCR.init_error
        response = _FakeRapidOCR.responses[len(self.calls) - 1]
        if isinstance(response, BaseException):
            raise response
        return response


@pytest.fixture()
def fake_rapidocr(monkeypatch: pytest.MonkeyPatch) -> type[_FakeRapidOCR]:
    module = types.ModuleType("rapidocr")
    module.RapidOCR = _FakeRapidOCR
    monkeypatch.setitem(sys.modules, "rapidocr", module)
    monkeypatch.setattr(_FakeRapidOCR, "instances", [])
    monkeypatch.setattr(_FakeRapidOCR, "responses", [])
    monkeypatch.setattr(_FakeRapidOCR, "init_error", None)
    return _FakeRapidOCR


def _png_bytes() -> bytes:
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (100, 50), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def _input(item_id: str = "r-1") -> InputItem:
    data = _png_bytes()
    return InputItem(
        item_id=item_id, encoded_bytes=len(data), decoded_pixels=5000, data=data
    )


class TestDescriptor:
    def test_unavailable_when_not_installed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "rapidocr", None)  # 强制 import 失败
        engine = RapidOcrEngine()
        descriptor = engine.descriptor()
        assert descriptor.availability is EngineAvailability.UNAVAILABLE
        assert descriptor.reason_code == REASON_ENGINE_NOT_INSTALLED
        assert descriptor.included_in_base is True

    def test_ready_when_installed(self, fake_rapidocr: type[_FakeRapidOCR]) -> None:
        engine = RapidOcrEngine()
        assert engine.descriptor().availability is EngineAvailability.READY


class TestRecognizeMany:
    def test_maps_txts_scores_boxes_to_stable_payload(
        self, fake_rapidocr: type[_FakeRapidOCR]
    ) -> None:
        # 100×50 图上的两个文本行（像素坐标）。
        fake_rapidocr.responses = [
            _FakeRapidOutput(
                txts=["hello", "world"],
                scores=[0.98, 0.42],
                boxes=[
                    [[10, 5], [60, 5], [60, 20], [10, 20]],
                    [[10, 25], [90, 25], [90, 45], [10, 45]],
                ],
            )
        ]
        engine = RapidOcrEngine()
        payloads = engine.recognize_many([_input()])
        assert len(payloads) == 1
        payload = payloads[0]
        assert payload["raw_text"] == "hello\nworld"
        assert payload["pipeline_type"] == "OCR"
        assert payload["image_width"] == 100
        assert payload["image_height"] == 50
        assert payload["avg_score"] == pytest.approx(0.7)
        assert payload["text_with_scores"] == [["hello", 0.98], ["world", 0.42]]
        assert payload["low_confidence_items"] == [["world", 0.42]]
        blocks = payload["text_blocks"]
        assert [b["text"] for b in blocks] == ["hello", "world"]
        assert [b["order"] for b in blocks] == [0, 1]
        # bbox 归一化到 [0,1000]：像素(10,5,60,20) → (100,100,600,400)。
        assert blocks[0]["bbox"] == pytest.approx([100.0, 100.0, 600.0, 400.0])
        assert blocks[0]["polygon"] is not None
        assert len(blocks[0]["polygon"]) == 8

    def test_empty_detection_returns_empty_text(
        self, fake_rapidocr: type[_FakeRapidOCR]
    ) -> None:
        fake_rapidocr.responses = [_FakeRapidOutput(txts=None, scores=None, boxes=None)]
        engine = RapidOcrEngine()
        payload = engine.recognize_many([_input()])[0]
        assert payload["raw_text"] == ""
        assert payload["text_blocks"] == []
        assert payload["avg_score"] == 0.0

    def test_batch_preserves_input_order(
        self, fake_rapidocr: type[_FakeRapidOCR]
    ) -> None:
        fake_rapidocr.responses = [
            _FakeRapidOutput(["a"], [0.9], [[[0, 0], [9, 0], [9, 9], [0, 9]]]),
            _FakeRapidOutput(["b"], [0.8], [[[0, 0], [9, 0], [9, 9], [0, 9]]]),
        ]
        engine = RapidOcrEngine()
        payloads = engine.recognize_many([_input("x"), _input("y")])
        assert [p["raw_text"] for p in payloads] == ["a", "b"]
        assert len(fake_rapidocr.instances) == 1

    def test_capabilities_are_conservative(self) -> None:
        capability = RapidOcrEngine().capabilities()
        assert capability.real_batch is False
        assert capability.max_compute_batch == 1


class TestInitFailureMemoization:
    def test_init_failure_is_memorized_until_reset(
        self, fake_rapidocr: type[_FakeRapidOCR]
    ) -> None:
        fake_rapidocr.init_error = RuntimeError("onnx load failed")
        engine = RapidOcrEngine()
        with pytest.raises(RuntimeError, match="init failed"):
            engine.recognize_many([_input()])
        # 第二次不再重新初始化，直接复用失败结论。
        with pytest.raises(RuntimeError, match="failed to init"):
            engine.recognize_many([_input()])
        assert engine.descriptor().reason_code == REASON_ENGINE_INIT_FAILED
        # release_idle 复位后允许重试。
        fake_rapidocr.init_error = None
        fake_rapidocr.responses = [
            _FakeRapidOutput(["ok"], [1.0], [[[0, 0], [9, 0], [9, 9], [0, 9]]])
        ]
        engine.release_idle()
        assert engine.recognize_many([_input()])[0]["raw_text"] == "ok"

    def test_input_without_bytes_raises(
        self, fake_rapidocr: type[_FakeRapidOCR]
    ) -> None:
        engine = RapidOcrEngine()
        with pytest.raises(ValueError, match="no raw bytes"):
            engine.recognize_many(
                [InputItem(item_id="e", encoded_bytes=0, decoded_pixels=0)]
            )

    def test_preload_loads_engine_once(
        self, fake_rapidocr: type[_FakeRapidOCR]
    ) -> None:
        engine = RapidOcrEngine()
        engine.preload(("OCR",))
        engine.preload(("OCR",))
        assert len(fake_rapidocr.instances) == 1
