"""Windows Media OCR 引擎 adapter：使用 Windows 自带 OCR 能力。

约束（docs/ocr-engine-runtime-profiles-execution-plan.md §B0.3/§B3）：

* adapter 随 base 运行时携带；系统 OCR/语言包由 Windows 动态探测，
  未通过探测时必须显示 unavailable，不能伪造成空识别结果。
* WinRT projection（pywinrt winrt-runtime）全部惰性导入；导入失败映射
  为结构化 unavailable，而不是让进程崩溃。
* 行级 bbox 归一化到 [0,1000]；Windows OCR 不提供置信度，
  契约策略为固定 score=1.0（缺失置信度不等于低置信度），由本模块
  显式决定并在此文档化。
* 请求指定 language 而系统缺对应语言包时抛
  OCR_ENGINE_LANGUAGE_UNAVAILABLE，绝不退回其他语言。
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

from vibeocr.runtime_contracts import ErrorCode, ResidencyStatus
from vibeocr.runtime_contracts.dtos import OcrEngine

from .budgets import AdapterCapability
from .ocr_engines import (
    OCR_PIPELINE_ID,
    REASON_ENGINE_INIT_FAILED,
    REASON_ENGINE_LANGUAGE_UNAVAILABLE,
    REASON_ENGINE_NOT_INSTALLED,
    EngineAvailability,
    EngineDescriptor,
    OcrEngineError,
)

logger = logging.getLogger(__name__)

# Windows OCR 未提供置信度：固定满置信并保持显式契约。
WINDOWS_OCR_SCORE_POLICY = 1.0


class _AsyncRunner:
    """进程内单例 WinRT 线程循环：WinRT async 需要非主线程的事件循环。"""

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def run(self, coro: Any) -> Any:
        with self._lock:
            if self._loop is None or not self._loop.is_running():
                self._start_locked()
        assert self._loop is not None
        return asyncio.run_coroutine_threadsafe(
            self._as_coroutine(coro), self._loop
        ).result()

    @staticmethod
    async def _as_coroutine(awaitable: Any) -> Any:
        """Adapt PyWinRT's generic Awaitable to asyncio's coroutine-only seam."""
        return await awaitable

    def _start_locked(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="winrt-ocr-loop", daemon=True
        )
        self._thread.start()

    def shutdown(self) -> None:
        with self._lock:
            loop = self._loop
            thread = self._thread
            self._loop = None
            self._thread = None
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)


class WindowsMediaOcrEngine:
    """``GeneralTextOcrEngine`` 的 Windows Media OCR 实现。"""

    engine_id = OcrEngine.WINDOWS
    included_in_base = True

    def __init__(self, *, runner: Any | None = None) -> None:
        # runner 注入点是测试 seam；生产使用进程内 WinRT 循环。
        self._owns_runner = runner is None
        self._runner = runner if runner is not None else _AsyncRunner()
        self._engine_cache: dict[str, Any] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Descriptor / capability
    # ------------------------------------------------------------------

    def descriptor(self) -> EngineDescriptor:
        reason = self._probe()
        if reason is not None:
            return EngineDescriptor(
                engine_id=self.engine_id,
                availability=EngineAvailability.UNAVAILABLE,
                included_in_base=self.included_in_base,
                reason_code=reason,
            )
        return EngineDescriptor(
            engine_id=self.engine_id,
            availability=EngineAvailability.READY,
            included_in_base=self.included_in_base,
        )

    def _probe(self) -> str | None:
        try:
            ocr_engine_cls = self._import_ocr_engine_cls()
        except Exception:
            return REASON_ENGINE_NOT_INSTALLED
        try:
            engine = self._runner.run(
                ocr_engine_cls.try_create_from_user_profile_languages()
            )
        except Exception:
            logger.exception("[Supervisor][WindowsOCR] probe failed")
            return REASON_ENGINE_INIT_FAILED
        if engine is None:
            # 系统没有可用的 OCR 语言包：这是语言能力问题而非安装问题。
            return REASON_ENGINE_LANGUAGE_UNAVAILABLE
        return None

    @staticmethod
    def _import_ocr_engine_cls() -> Any:
        from winrt.windows.media.ocr import OcrEngine

        return OcrEngine

    @staticmethod
    def _import_language_cls() -> Any:
        from winrt.windows.globalization import Language

        return Language

    def capabilities(self, options: Any | None = None) -> AdapterCapability:
        del options
        return AdapterCapability(
            name=OCR_PIPELINE_ID,
            real_batch=False,
            max_compute_batch=1,
        )

    # ------------------------------------------------------------------
    # recognize_many
    # ------------------------------------------------------------------

    def recognize_many(
        self,
        items: list[Any],
        *,
        options: Any | None = None,
        compute_batch: Any | None = None,
    ) -> list[dict[str, Any]]:
        del compute_batch
        if not items:
            return []
        language = self._requested_language(options)
        payloads: list[dict[str, Any]] = []
        for item in items:
            raw = self._raw_bytes(item)
            payloads.append(self._recognize_bytes(raw, language))
        return payloads

    def _recognize_bytes(self, raw: bytes, language: str | None) -> dict[str, Any]:
        engine = self._engine_for_language(language)
        if engine is None:
            # 目录探测已放行但该语言不可用：显式失败，不回退其他语言。
            raise OcrEngineError(
                ErrorCode.OCR_ENGINE_LANGUAGE_UNAVAILABLE,
                reason_code=REASON_ENGINE_LANGUAGE_UNAVAILABLE,
                engine=self.engine_id.value,
            )
        return self._runner.run(self._recognize_with_engine(engine, raw))

    async def _recognize_with_engine(self, engine: Any, raw: bytes) -> dict[str, Any]:
        bitmap, width, height = await self._decode_bitmap(raw)
        result = await engine.recognize_async(bitmap)
        return self._result_to_payload(result, width, height)

    async def _decode_bitmap(self, raw: bytes) -> tuple[Any, int, int]:
        from winrt.windows.graphics.imaging import (
            BitmapDecoder,
            BitmapPixelFormat,
            SoftwareBitmap,
        )
        from winrt.windows.storage.streams import (
            DataWriter,
            InMemoryRandomAccessStream,
        )

        stream = InMemoryRandomAccessStream()
        writer = DataWriter(stream.get_output_stream_at(0))
        writer.write_bytes(raw)
        await writer.store_async()
        decoder = await BitmapDecoder.create_async(stream)
        bitmap = await decoder.get_software_bitmap_async()
        if bitmap is None:
            raise RuntimeError("Windows OCR: failed to decode bitmap")
        pixel_width = int(bitmap.pixel_width)
        pixel_height = int(bitmap.pixel_height)
        # OcrEngine 只接受 Bgra8；其他格式显式转换。
        if bitmap.bitmap_pixel_format != BitmapPixelFormat.BGRA8:
            bitmap = SoftwareBitmap.convert(bitmap, BitmapPixelFormat.BGRA8)
        return bitmap, pixel_width, pixel_height

    def _engine_for_language(self, language: str | None) -> Any:
        cache_key = language or ""
        with self._lock:
            cached = self._engine_cache.get(cache_key)
            if cached is not None:
                return cached
        ocr_engine_cls = self._import_ocr_engine_cls()
        if language is None:
            engine = self._runner.run(
                ocr_engine_cls.try_create_from_user_profile_languages()
            )
        else:
            language_cls = self._import_language_cls()
            engine = self._runner.run(
                ocr_engine_cls.try_create_from_language(language_cls(language))
            )
        if engine is not None:
            with self._lock:
                self._engine_cache[cache_key] = engine
        return engine

    # ------------------------------------------------------------------
    # Result mapping
    # ------------------------------------------------------------------

    @staticmethod
    def _result_to_payload(result: Any, width: int, height: int) -> dict[str, Any]:
        from vibeocr.backend.models import ocr_result_to_payload
        from vibeocr.backend.models.ocr_result import OCRResult, TextBlock

        def _norm_x(value: float) -> float:
            return value / width * 1000 if width else value

        def _norm_y(value: float) -> float:
            return value / height * 1000 if height else value

        lines = list(getattr(result, "lines", None) or ())
        blocks: list[TextBlock] = []
        pairs: list[tuple[str, float]] = []
        for index, line in enumerate(lines):
            text = str(getattr(line, "text", "") or "")
            words = list(getattr(line, "words", None) or ())
            bbox = None
            if words:
                rects = [word.bounding_rect for word in words]
                # Word BoundingRect 是位图像素坐标：显式换算到 [0,1000]，
                # 不走 normalize_bbox 的范围启发式（小图会被误判）。
                bbox = (
                    _norm_x(min(float(r.x) for r in rects)),
                    _norm_y(min(float(r.y) for r in rects)),
                    _norm_x(max(float(r.x) + float(r.width) for r in rects)),
                    _norm_y(max(float(r.y) + float(r.height) for r in rects)),
                )
            blocks.append(
                TextBlock(
                    text=text,
                    score=WINDOWS_OCR_SCORE_POLICY,
                    bbox=bbox,
                    polygon=None,
                    order=index,
                )
            )
            pairs.append((text, WINDOWS_OCR_SCORE_POLICY))

        ocr_result = OCRResult(
            raw_text="\n".join(text for text, _ in pairs),
            markdown_text="\n".join(text for text, _ in pairs),
            html_text="",
            text_with_scores=pairs,
            avg_score=WINDOWS_OCR_SCORE_POLICY if pairs else 0.0,
            pipeline_type=OCR_PIPELINE_ID,
            text_blocks=blocks,
            image_width=int(width),
            image_height=int(height),
        )
        return ocr_result_to_payload(ocr_result)

    # ------------------------------------------------------------------
    # Lifecycle / residency
    # ------------------------------------------------------------------

    def preload(self, pipelines: tuple[str, ...]) -> ResidencyStatus:
        if OCR_PIPELINE_ID in pipelines:
            self._engine_for_language(None)
        return self.residency_status()

    def residency_status(self) -> ResidencyStatus:
        return ResidencyStatus()

    def release_idle(self, pipeline: str | None = None) -> ResidencyStatus:
        del pipeline
        with self._lock:
            self._engine_cache.clear()
        return self.residency_status()

    def configure_settings(self, snapshot: Any) -> None:
        del snapshot

    def close(self) -> None:
        with self._lock:
            self._engine_cache.clear()
        if self._owns_runner:
            self._runner.shutdown()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _requested_language(options: Any | None) -> str | None:
        if options is None:
            return None
        raw = getattr(options, "options", None) or {}
        language = raw.get("language") if isinstance(raw, dict) else None
        return str(language) if language else None

    @staticmethod
    def _raw_bytes(item: Any) -> bytes:
        data = getattr(item, "data", b"")
        if isinstance(data, (bytes, bytearray)) and len(data) > 0:
            return bytes(data)
        raise ValueError(
            f"InputItem {getattr(item, 'item_id', '?')} has no raw bytes; "
            "decode before calling recognize_many"
        )


__all__ = ["WindowsMediaOcrEngine", "WINDOWS_OCR_SCORE_POLICY"]
