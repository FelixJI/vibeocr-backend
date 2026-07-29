"""PDF 应用服务 facade。

封装 PDF adapter（如 PdfSessionManager/PdfBackendClient），对外暴露
PdfApplication 接口。不发 Qt signal，不接触 widget。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from vibeocr.backend.application.contracts import (
    CancelToken,
    PdfError,
    PdfOpenRequest,
    PdfSessionDto,
)


@runtime_checkable
class PdfAdapter(Protocol):
    """PDF adapter 协议：facade 委托的实际执行者。"""

    def open(self, request: PdfOpenRequest, cancel: CancelToken) -> PdfSessionDto: ...


class PdfFacade:
    """PDF 应用服务实现。

    通过注入的 PdfAdapter 执行 PDF 操作，包装异常为 PdfError。
    """

    def __init__(self, adapter: PdfAdapter) -> None:
        self._adapter = adapter

    def open(self, request: PdfOpenRequest, cancel: CancelToken) -> PdfSessionDto:
        """打开 PDF 文件，返回会话 DTO。

        Args:
            request: PDF 打开请求。
            cancel: 取消令牌。

        Returns:
            PdfSessionDto。

        Raises:
            PdfError: 取消、adapter 异常或文件错误。
        """
        if cancel is not None and cancel.is_cancelled:
            raise PdfError("cancelled before start")

        try:
            return self._adapter.open(request, cancel)
        except PdfError:
            raise
        except Exception as e:
            raise PdfError(f"PDF open failed: {e}") from e
