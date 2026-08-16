"""OcrEngineRoutingAdapter：RECOGNITION 的通用引擎入口。

职责（docs/ocr-engine-runtime-profiles-execution-plan.md §3.2）：

* ``OCR`` pipeline（纯文本）按请求的 engine ID 经 resolver 路由到对应
  引擎 adapter；请求缺省 engine 时使用 Backend 默认引擎（rapidocr）。
* 其他 Paddle pipeline（表格/公式/版面/VL）不参与 engine 选择，直达
  fallback adapter——这是显式的 pipeline 分流，不是引擎静默回退。
* 引擎不可用等确定性失败以 ``OcrEngineError`` 抛出，由 executor 映射为
  per-item 协议错误码，绝不落入恢复重试或切换引擎。
* 本 adapter 与既有 adapter seam（capabilities/recognize_many/residency/
  lifecycle）同形，``PaddleExecutor`` 无需感知引擎差异。
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

from vibeocr.runtime_contracts import PipelineSelection, ResidencyStatus

from .ocr_engines import OCR_PIPELINE_ID, OcrEngineResolver

logger = logging.getLogger(__name__)


class OcrEngineRoutingAdapter:
    """RECOGNITION 通用入口：OCR pipeline 按 engine 路由，其余直达 fallback。"""

    def __init__(
        self,
        *,
        fallback_factory: Callable[[], Any],
        resolver: OcrEngineResolver,
    ) -> None:
        self._fallback_factory = fallback_factory
        self._fallback: Any = None
        self._fallback_lock = threading.Lock()
        self.resolver = resolver

    @property
    def fallback(self) -> Any:
        if self._fallback is not None:
            return self._fallback
        with self._fallback_lock:
            if self._fallback is None:
                self._fallback = self._fallback_factory()
        return self._fallback

    # ------------------------------------------------------------------
    # 路由判定
    # ------------------------------------------------------------------

    @staticmethod
    def _selection_parts(
        options: Any | None,
    ) -> tuple[str | None, Any]:
        """返回 (pipeline_id, engine)；非 PipelineSelection 时 (None, None)。"""
        if isinstance(options, PipelineSelection):
            return options.pipeline_id, getattr(options, "engine", None)
        return None, None

    def _engine_for(self, options: Any | None) -> Any:
        _, engine_id = self._selection_parts(options)
        return self.resolver.resolve(engine_id)

    # ------------------------------------------------------------------
    # Adapter seam
    # ------------------------------------------------------------------

    def capabilities(self, options: Any | None = None) -> Any:
        pipeline_id, _ = self._selection_parts(options)
        if pipeline_id == OCR_PIPELINE_ID:
            return self._engine_for(options).capabilities(options)
        return self.fallback.capabilities(options)

    def recognize_many(
        self,
        items: list[Any],
        *,
        options: Any | None = None,
        compute_batch: Any | None = None,
    ) -> list[dict[str, Any]]:
        pipeline_id, engine_id = self._selection_parts(options)
        if pipeline_id == OCR_PIPELINE_ID:
            engine = self.resolver.resolve(engine_id)
            logger.info(
                "[Supervisor][Recognize] route=pipeline-ocr engine=%s items=%d",
                engine.engine_id.value,
                len(items),
            )
            return engine.recognize_many(
                items, options=options, compute_batch=compute_batch
            )
        return self.fallback.recognize_many(
            items, options=options, compute_batch=compute_batch
        )

    # ------------------------------------------------------------------
    # Residency / lifecycle
    # ------------------------------------------------------------------

    def preload(self, pipelines: tuple[str, ...]) -> ResidencyStatus:
        """拆分预加载：OCR 走默认引擎，其余 Paddle 管道走 fallback。"""
        wants_ocr = OCR_PIPELINE_ID in pipelines
        others = tuple(name for name in pipelines if name != OCR_PIPELINE_ID)
        if others:
            self.fallback.preload(others)
        if wants_ocr:
            engine = self.resolver.resolve(None)
            if engine is not self.fallback:
                engine.preload((OCR_PIPELINE_ID,))
        return self.residency_status()

    def residency_status(self) -> ResidencyStatus:
        return self.fallback.residency_status()

    def release_idle(self, pipeline: str | None = None) -> ResidencyStatus:
        return self.fallback.release_idle(pipeline)

    def configure_settings(self, snapshot: Any) -> None:
        self.fallback.configure_settings(snapshot)

    def close(self) -> None:
        for engine in self.resolver.registry.engines():
            try:
                engine.close()
            except Exception:  # pragma: no cover - defensive
                logger.exception(
                    "[Supervisor][OcrEngines] close failed engine=%s",
                    getattr(engine, "engine_id", "?"),
                )
        self.fallback.close()


__all__ = ["OcrEngineRoutingAdapter"]
