"""OCR 应用服务 facade。

封装 OCR adapter（如 OCRServiceSubprocess），对外暴露 OcrApplication 接口。
不发 Qt signal，不接触 widget。UI 和 WorkerHost 都通过此 facade 调用 OCR。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from vibeocr.backend.application.contracts import (
    CancelToken,
    OcrError,
    OcrRequest,
    OcrResult,
)


@runtime_checkable
class OcrAdapter(Protocol):
    """OCR adapter 协议：facade 委托的实际执行者。

    实现方可以是 OCRServiceSubprocess（生产）或 fake（测试）。
    """

    # Protocol 方法签名为 ``...`` body（无可执行分支）；coverage 的 ``->exit``
    # partial 是插桩噪音，非真实未覆盖代码。
    def recognize(
        self, request: OcrRequest, cancel: CancelToken
    ) -> OcrResult: ...  # pragma: no cover

    def recognize_batch(
        self, requests: list[OcrRequest], cancel: CancelToken
    ) -> list[OcrResult | None]: ...  # pragma: no cover


class OcrFacade:
    """OCR 应用服务实现。

    通过注入的 OcrAdapter 执行 OCR，包装异常为 OcrError。
    检查取消令牌并在取消时抛出 OcrError。

    Usage::

        adapter = OCRServiceSubprocess(...)
        facade = OcrFacade(adapter)
        result = facade.recognize(request, cancel)
    """

    def __init__(self, adapter: OcrAdapter) -> None:
        self._adapter = adapter

    def recognize(self, request: OcrRequest, cancel: CancelToken) -> OcrResult:
        """执行 OCR 识别。

        Args:
            request: OCR 请求（图片数据 + 管道）。
            cancel: 取消令牌。

        Returns:
            OcrResult。

        Raises:
            OcrError: 取消、adapter 异常或参数错误。
        """
        # 前置取消检查
        if cancel is not None and cancel.is_cancelled:
            raise OcrError("cancelled before start")

        try:
            return self._adapter.recognize(request, cancel)
        except OcrError:
            raise
        except Exception as e:
            raise OcrError(f"OCR recognize failed: {e}") from e

    def recognize_batch(
        self, requests: list[OcrRequest], cancel: CancelToken
    ) -> list[OcrResult | None]:
        """Execute one engine-level batch while preserving input order."""
        if cancel is not None and cancel.is_cancelled:
            raise OcrError("cancelled before start")
        if not requests:
            return []
        try:
            return self._adapter.recognize_batch(requests, cancel)
        except OcrError:
            raise
        except Exception as e:
            raise OcrError(f"OCR batch recognize failed: {e}") from e
