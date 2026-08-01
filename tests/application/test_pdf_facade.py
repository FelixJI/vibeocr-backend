"""PdfApplication facade 测试。

以 fake adapter 验证 open 参数传递、取消 token、错误映射和结果 DTO。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from vibeocr.backend.application.contracts import (
    CancelToken,
    PdfApplication,
    PdfError,
    PdfOpenRequest,
    PdfSessionDto,
)
from vibeocr.backend.application.pdf_facade import PdfFacade


class _FakePdfAdapter:
    """fake PDF adapter：记录调用参数，可选模拟异常/取消。"""

    def __init__(
        self,
        *,
        page_count: int = 3,
        raise_exception: Exception | None = None,
    ) -> None:
        self._page_count = page_count
        self._raise = raise_exception
        self.last_request: PdfOpenRequest | None = None
        self.last_cancel: CancelToken | None = None
        self.call_count = 0

    def open(self, request: PdfOpenRequest, cancel: CancelToken) -> PdfSessionDto:
        self.last_request = request
        self.last_cancel = cancel
        self.call_count += 1
        if cancel is not None and cancel.is_cancelled:
            raise PdfError("cancelled")
        if self._raise is not None:
            raise self._raise
        return PdfSessionDto(
            session_id="fake-session-id",
            file_path=request.file_path,
            page_count=self._page_count,
        )


class TestPdfOpenRequest:
    def test_request_is_frozen(self):
        req = PdfOpenRequest(file_path=Path("/tmp/test.pdf"))
        with pytest.raises(Exception):  # noqa: B017
            req.file_path = Path("/other.pdf")  # type: ignore[misc]


class TestPdfSessionDto:
    def test_dto_is_frozen(self):
        dto = PdfSessionDto(
            session_id="sid", file_path=Path("/tmp/test.pdf"), page_count=5
        )
        with pytest.raises(Exception):  # noqa: B017
            dto.page_count = 10  # type: ignore[misc]


class TestPdfFacade:
    def test_facade_implements_protocol(self):
        adapter = _FakePdfAdapter()
        facade = PdfFacade(adapter)
        assert isinstance(facade, PdfApplication)

    def test_open_passes_request_and_cancel(self):
        adapter = _FakePdfAdapter()
        facade = PdfFacade(adapter)
        req = PdfOpenRequest(file_path=Path("/tmp/test.pdf"))
        cancel = CancelToken()

        result = facade.open(req, cancel)

        assert adapter.last_request is req
        assert adapter.last_cancel is cancel
        assert isinstance(result, PdfSessionDto)
        assert result.page_count == 3
        assert result.session_id == "fake-session-id"

    def test_open_propagates_pdf_error(self):
        adapter = _FakePdfAdapter(raise_exception=PdfError("corrupt pdf"))
        facade = PdfFacade(adapter)

        with pytest.raises(PdfError, match="corrupt pdf"):
            facade.open(PdfOpenRequest(file_path=Path("/tmp/test.pdf")), CancelToken())

    def test_open_wraps_generic_exception(self):
        adapter = _FakePdfAdapter(raise_exception=ValueError("bad path"))
        facade = PdfFacade(adapter)

        with pytest.raises(PdfError):
            facade.open(PdfOpenRequest(file_path=Path("/tmp/test.pdf")), CancelToken())

    def test_open_respects_cancel(self):
        adapter = _FakePdfAdapter()
        facade = PdfFacade(adapter)
        cancel = CancelToken()
        cancel.cancel()

        with pytest.raises(PdfError, match="cancel"):
            facade.open(PdfOpenRequest(file_path=Path("/tmp/test.pdf")), cancel)
