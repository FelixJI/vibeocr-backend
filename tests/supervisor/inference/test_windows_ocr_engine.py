"""WindowsMediaOcrEngine adapter 测试：fake WinRT 对象驱动，不依赖 pywinrt。

覆盖计划 §B0.3/§B3 的可测部分：descriptor 探测（无 pywinrt / 无语言包 /
就绪）、行级 bbox 归一化、无 score 契约策略（固定 1.0）、指定语言缺
语言包时 OCR_ENGINE_LANGUAGE_UNAVAILABLE 且不回退其他语言。
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

import pytest
from vibeocr.backend.supervisor.inference.budgets import InputItem
from vibeocr.backend.supervisor.inference.ocr_engines import (
    REASON_ENGINE_INIT_FAILED,
    REASON_ENGINE_LANGUAGE_UNAVAILABLE,
    REASON_ENGINE_NOT_INSTALLED,
    EngineAvailability,
)
from vibeocr.backend.supervisor.inference.windows_ocr_engine import (
    WINDOWS_OCR_SCORE_POLICY,
    WindowsMediaOcrEngine,
)
from vibeocr.runtime_contracts.dtos import PipelineSelection


class _SyncRunner:
    """测试 seam：直接在当前线程跑协程（生产为 WinRT 线程循环）。"""

    def run(self, coro: Any) -> Any:
        return asyncio.run(coro)


class _FakeWinrtOcrEngine:
    """winrt.windows.media.ocr.OcrEngine 的最小 fake。"""

    user_profile_engine: Any = "engine-object"
    language_engines: dict[str, Any] = {}
    raise_on_create = False

    @staticmethod
    async def try_create_from_user_profile_languages() -> Any:
        if _FakeWinrtOcrEngine.raise_on_create:
            raise RuntimeError("winrt blew up")
        return _FakeWinrtOcrEngine.user_profile_engine

    @staticmethod
    async def try_create_from_language(language: Any) -> Any:
        return _FakeWinrtOcrEngine.language_engines.get(str(language.language_tag))


class _Rect:
    def __init__(self, x: float, y: float, width: float, height: float) -> None:
        self.x, self.y, self.width, self.height = x, y, width, height


class _Word:
    def __init__(self, rect: _Rect) -> None:
        self.bounding_rect = rect


class _Line:
    def __init__(self, text: str, words: list[_Word]) -> None:
        self.text = text
        self.words = words


class _FakeOcrResult:
    def __init__(self, lines: list[_Line]) -> None:
        self.lines = lines


@pytest.fixture()
def engine() -> WindowsMediaOcrEngine:
    instance = WindowsMediaOcrEngine(runner=_SyncRunner())
    instance._import_ocr_engine_cls = lambda: _FakeWinrtOcrEngine  # type: ignore[method-assign]

    class _FakeLanguage:
        def __init__(self, tag: str) -> None:
            self.language_tag = tag

    instance._import_language_cls = lambda: _FakeLanguage  # type: ignore[method-assign]
    return instance


def _input(item_id: str = "w-1") -> InputItem:
    return InputItem(
        item_id=item_id, encoded_bytes=16, decoded_pixels=100, data=b"\x89PNGfake"
    )


class TestDescriptor:
    def test_unavailable_without_projection(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 强制 winrt 导入失败（模块缓存置 None 即 ImportError）。
        monkeypatch.setitem(sys.modules, "winrt", None)
        descriptor = WindowsMediaOcrEngine().descriptor()
        assert descriptor.availability is EngineAvailability.UNAVAILABLE
        assert descriptor.reason_code == REASON_ENGINE_NOT_INSTALLED

    def test_ready_when_engine_available(self, engine: WindowsMediaOcrEngine) -> None:
        assert engine.descriptor().availability is EngineAvailability.READY
        assert engine.descriptor().included_in_base is True

    def test_no_language_pack_is_language_reason(
        self, engine: WindowsMediaOcrEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(_FakeWinrtOcrEngine, "user_profile_engine", None)
        descriptor = engine.descriptor()
        assert descriptor.availability is EngineAvailability.UNAVAILABLE
        # 系统没有 OCR 语言包：语言能力问题，不伪装成就绪。
        assert descriptor.reason_code == REASON_ENGINE_LANGUAGE_UNAVAILABLE

    def test_probe_exception_maps_init_failed(
        self, engine: WindowsMediaOcrEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(_FakeWinrtOcrEngine, "raise_on_create", True)
        assert engine.descriptor().reason_code == REASON_ENGINE_INIT_FAILED


class TestRecognize:
    @pytest.fixture()
    def wired_engine(self, engine: WindowsMediaOcrEngine) -> WindowsMediaOcrEngine:
        """绕过真实解码：_decode_bitmap 返回固定几何。"""

        async def _decode(raw: bytes) -> tuple[Any, int, int]:
            return ("bitmap", 200, 100)

        engine._decode_bitmap = _decode  # type: ignore[method-assign]
        return engine

    def _recognize_engine(self, lines: list[_Line]) -> Any:
        class _Engine:
            @staticmethod
            async def recognize_async(bitmap: Any) -> _FakeOcrResult:
                return _FakeOcrResult(lines)

        return _Engine()

    def test_line_bbox_union_and_score_policy(
        self, wired_engine: WindowsMediaOcrEngine
    ) -> None:
        lines = [
            _Line(
                "hello world",
                [_Word(_Rect(10, 5, 50, 15)), _Word(_Rect(70, 5, 30, 15))],
            ),
            _Line("second", [_Word(_Rect(10, 40, 60, 20))]),
        ]
        wired_engine._engine_cache[""] = self._recognize_engine(lines)
        payload = wired_engine.recognize_many([_input()])[0]
        assert payload["raw_text"] == "hello world\nsecond"
        assert payload["image_width"] == 200
        assert payload["image_height"] == 100
        blocks = payload["text_blocks"]
        assert [b["text"] for b in blocks] == ["hello world", "second"]
        # 无置信度契约：固定 1.0，绝不冒充低置信。
        assert all(b["score"] == WINDOWS_OCR_SCORE_POLICY for b in blocks)
        assert payload["avg_score"] == WINDOWS_OCR_SCORE_POLICY
        # 行 bbox 取词矩形并集并归一化到 [0,1000]：(10,5)-(100,20)/200×100。
        assert blocks[0]["bbox"] == pytest.approx([50.0, 50.0, 500.0, 200.0])
        assert blocks[1]["bbox"] == pytest.approx([50.0, 400.0, 350.0, 600.0])

    def test_requested_language_selects_language_engine(
        self, wired_engine: WindowsMediaOcrEngine
    ) -> None:
        wired_engine._engine_cache["ja-JP"] = self._recognize_engine(
            [_Line("こんにちは", [_Word(_Rect(0, 0, 100, 50))])]
        )
        options = PipelineSelection(pipeline_id="OCR", options={"language": "ja-JP"})
        payload = wired_engine.recognize_many([_input()], options=options)[0]
        assert payload["raw_text"] == "こんにちは"

    def test_missing_language_pack_fails_closed(
        self, engine: WindowsMediaOcrEngine
    ) -> None:
        # 目录探测就绪（默认语言可用），但请求的 ja-JP 无语言包：
        # 必须显式失败，不回退到默认语言引擎。
        engine._engine_cache[""] = self._recognize_engine([])
        from vibeocr.backend.supervisor.inference.ocr_engines import OcrEngineError

        with pytest.raises(OcrEngineError) as excinfo:
            engine._recognize_bytes(b"img", "ja-JP")
        assert excinfo.value.code.value == "OCR_ENGINE_LANGUAGE_UNAVAILABLE"
        assert excinfo.value.reason_code == REASON_ENGINE_LANGUAGE_UNAVAILABLE

    def test_language_engine_cached_per_language(
        self, wired_engine: WindowsMediaOcrEngine
    ) -> None:
        wired_engine._engine_cache["en-US"] = self._recognize_engine([])
        wired_engine._engine_cache["fr-FR"] = self._recognize_engine([])
        options_us = PipelineSelection(pipeline_id="OCR", options={"language": "en-US"})
        options_fr = PipelineSelection(pipeline_id="OCR", options={"language": "fr-FR"})
        wired_engine.recognize_many([_input()], options=options_us)
        wired_engine.recognize_many([_input()], options=options_fr)
        assert (
            wired_engine._engine_cache["en-US"]
            is not wired_engine._engine_cache["fr-FR"]
        )

    def test_empty_lines_payload_is_empty_text(
        self, wired_engine: WindowsMediaOcrEngine
    ) -> None:
        wired_engine._engine_cache[""] = self._recognize_engine([])
        payload = wired_engine.recognize_many([_input()])[0]
        assert payload["raw_text"] == ""
        assert payload["text_blocks"] == []
        assert payload["avg_score"] == 0.0

    def test_requested_language_reads_pipeline_options_only(
        self, engine: WindowsMediaOcrEngine
    ) -> None:
        assert engine._requested_language(None) is None
        assert (
            engine._requested_language(PipelineSelection(pipeline_id="OCR", options={}))
            is None
        )
        assert (
            engine._requested_language(
                PipelineSelection(pipeline_id="OCR", options={"language": "de-DE"})
            )
            == "de-DE"
        )


class TestLifecycle:
    def test_release_idle_clears_engine_cache(
        self, engine: WindowsMediaOcrEngine
    ) -> None:
        engine._engine_cache[""] = object()
        engine.release_idle()
        assert engine._engine_cache == {}

    def test_close_is_idempotent(self, engine: WindowsMediaOcrEngine) -> None:
        engine._engine_cache[""] = object()
        engine.close()
        engine.close()
        assert engine._engine_cache == {}
