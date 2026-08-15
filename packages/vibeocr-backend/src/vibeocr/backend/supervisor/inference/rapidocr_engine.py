"""RapidOCR 引擎 adapter：全局默认的轻量离线纯文本 OCR。

约束（docs/ocr-engine-runtime-profiles-execution-plan.md §B3）：

* 惰性初始化：首次 ``recognize_many`` 才加载模型；禁用网络下载，
  模型随 base-offline 运行时携带（wheel 内置或固定本地路径）。
* 复用进程内单份 ONNX Runtime——本 adapter 不自行携带 ORT 依赖，
  仅通过 rapidocr 使用现有闭包中的 onnxruntime。
* 输出映射到稳定 OCRResult payload：text/score/polygon（[0,1000]
  归一化）与阅读顺序，与 Paddle 引擎同构。
"""

from __future__ import annotations

import io
import logging
import threading
from typing import Any

from vibeocr.runtime_contracts import ResidencyStatus
from vibeocr.runtime_contracts.dtos import OcrEngine

from .budgets import AdapterCapability
from .ocr_engines import (
    OCR_PIPELINE_ID,
    REASON_ENGINE_INIT_FAILED,
    REASON_ENGINE_NOT_INSTALLED,
    EngineAvailability,
    EngineDescriptor,
)

logger = logging.getLogger(__name__)


class RapidOcrEngine:
    """``GeneralTextOcrEngine`` 的 RapidOCR 实现（线程安全、惰性加载）。"""

    engine_id = OcrEngine.RAPIDOCR
    included_in_base = True

    def __init__(self, *, engine_params: dict[str, Any] | None = None) -> None:
        self._engine_params = dict(engine_params or {})
        self._engine: Any = None
        self._init_error: str | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Descriptor / capability
    # ------------------------------------------------------------------

    def descriptor(self) -> EngineDescriptor:
        probe = self._probe()
        if probe is not None:
            return EngineDescriptor(
                engine_id=self.engine_id,
                availability=EngineAvailability.UNAVAILABLE,
                included_in_base=self.included_in_base,
                reason_code=probe,
            )
        return EngineDescriptor(
            engine_id=self.engine_id,
            availability=EngineAvailability.READY,
            included_in_base=self.included_in_base,
        )

    def _probe(self) -> str | None:
        """返回不可用 reason_code，可用返回 None。"""
        if self._init_error is not None:
            # 已证明初始化失败：保持 unavailable，直到 close() 复位。
            return REASON_ENGINE_INIT_FAILED
        try:
            __import__("rapidocr")
        except ImportError:
            return REASON_ENGINE_NOT_INSTALLED
        except Exception:
            logger.exception("[Supervisor][RapidOCR] probe import failed")
            return REASON_ENGINE_INIT_FAILED
        return None

    def capabilities(self, options: Any | None = None) -> AdapterCapability:
        del options
        # 保守契约：逐图推理，real batch 由外层 transport/compute 预算控制。
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
        del options, compute_batch
        if not items:
            return []
        engine = self._ensure_engine()
        payloads: list[dict[str, Any]] = []
        for item in items:
            image = self._to_ndarray(self._raw_bytes(item))
            payloads.append(self._recognize_one(engine, image))
        return payloads

    def _ensure_engine(self) -> Any:
        with self._lock:
            if self._engine is not None:
                return self._engine
            if self._init_error is not None:
                raise RuntimeError(
                    f"RapidOCR engine failed to init: {self._init_error}"
                )
            try:
                from rapidocr import RapidOCR

                self._engine = RapidOCR(**self._engine_params)
            except Exception as exc:
                self._init_error = str(exc)
                logger.exception("[Supervisor][RapidOCR] engine init failed")
                raise RuntimeError(f"RapidOCR engine init failed: {exc}") from exc
            logger.info("[Supervisor][RapidOCR] engine initialized")
            return self._engine

    def _recognize_one(self, engine: Any, image: Any) -> dict[str, Any]:
        try:
            result = engine(image)
        except Exception:
            logger.exception("[Supervisor][RapidOCR] recognize failed")
            raise
        return self._result_to_payload(result, image)

    # ------------------------------------------------------------------
    # Result mapping
    # ------------------------------------------------------------------

    @staticmethod
    def _result_to_payload(result: Any, image: Any) -> dict[str, Any]:
        """把 RapidOCROutput（txts/scores/boxes）映射为稳定 ocr.v1 payload。"""
        from vibeocr.backend.models import ocr_result_to_payload
        from vibeocr.backend.models.ocr_result import OCRResult, TextBlock

        height, width = (
            (image.shape[0], image.shape[1]) if image is not None else (0, 0)
        )
        raw_txts = getattr(result, "txts", None)
        raw_scores = getattr(result, "scores", None)
        raw_boxes = getattr(result, "boxes", None)
        # numpy 数组的真值判断是 ambiguous：必须显式判 None。
        txts = raw_txts if raw_txts is not None else ()
        scores = raw_scores if raw_scores is not None else ()
        boxes = raw_boxes if raw_boxes is not None else ()

        def _norm_x(value: float) -> float:
            return value / width * 1000 if width else value

        def _norm_y(value: float) -> float:
            return value / height * 1000 if height else value

        blocks: list[TextBlock] = []
        pairs: list[tuple[str, float]] = []
        for index, text in enumerate(txts):
            score = float(scores[index]) if index < len(scores) else 0.0
            box = boxes[index] if index < len(boxes) else None
            polygon: tuple[float, ...] | None = None
            bbox: tuple[float, float, float, float] | None = None
            if box is not None and len(box) > 0:
                xs = [float(point[0]) for point in box]
                ys = [float(point[1]) for point in box]
                # RapidOCR 输出原始像素坐标：显式换算到 [0,1000]，不走
                # normalize_bbox 的范围启发式（小图像像素值会被误判）。
                flat: list[float] = []
                for point in box:
                    flat.append(_norm_x(float(point[0])))
                    flat.append(_norm_y(float(point[1])))
                polygon = tuple(flat)
                bbox = (
                    _norm_x(min(xs)),
                    _norm_y(min(ys)),
                    _norm_x(max(xs)),
                    _norm_y(max(ys)),
                )
            blocks.append(
                TextBlock(
                    text=str(text),
                    score=score,
                    bbox=bbox,
                    polygon=polygon,
                    order=index,
                )
            )
            pairs.append((str(text), score))

        avg = sum(score for _, score in pairs) / len(pairs) if pairs else 0.0
        ocr_result = OCRResult(
            raw_text="\n".join(text for text, _ in pairs),
            markdown_text="\n".join(text for text, _ in pairs),
            html_text="",
            text_with_scores=pairs,
            avg_score=avg,
            low_confidence_items=[pair for pair in pairs if pair[1] < 0.6],
            pipeline_type=OCR_PIPELINE_ID,
            text_blocks=blocks,
            image_width=int(width),
            image_height=int(height),
        )
        return ocr_result_to_payload(ocr_result)

    # ------------------------------------------------------------------
    # Lifecycle / residency（轻量引擎：无 TTL 驻留语义）
    # ------------------------------------------------------------------

    def preload(self, pipelines: tuple[str, ...]) -> ResidencyStatus:
        if OCR_PIPELINE_ID in pipelines:
            self._ensure_engine()
        return self.residency_status()

    def residency_status(self) -> ResidencyStatus:
        return ResidencyStatus()

    def release_idle(self, pipeline: str | None = None) -> ResidencyStatus:
        del pipeline
        with self._lock:
            self._engine = None
            self._init_error = None
        return self.residency_status()

    def configure_settings(self, snapshot: Any) -> None:
        del snapshot

    def close(self) -> None:
        with self._lock:
            self._engine = None
            self._init_error = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _raw_bytes(item: Any) -> bytes:
        data = getattr(item, "data", b"")
        if isinstance(data, (bytes, bytearray)) and len(data) > 0:
            return bytes(data)
        raise ValueError(
            f"InputItem {getattr(item, 'item_id', '?')} has no raw bytes; "
            "decode before calling recognize_many"
        )

    @staticmethod
    def _to_ndarray(raw: bytes) -> Any:
        import numpy as np
        from PIL import Image as PILImage

        img = PILImage.open(io.BytesIO(raw))
        if img.mode != "RGB":
            img = img.convert("RGB")
        return np.array(img)


__all__ = ["RapidOcrEngine"]
