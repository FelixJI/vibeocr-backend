"""OcrApplication facade 测试。

以 fake adapter 验证参数传递、取消 token、错误映射和结果 DTO。
Facade 只依赖 dataclass/Protocol 和现有 service，不发 Qt signal。
"""

from __future__ import annotations

import pytest

from vibeocr.backend.application.contracts import (
    CancelToken,
    OcrApplication,
    OcrError,
    OcrRequest,
    OcrResult,
)
from vibeocr.backend.application.ocr_facade import OcrFacade


class _FakeOcrAdapter:
    """fake OCR adapter：记录调用参数，可选模拟延迟/异常/取消。"""

    def __init__(
        self,
        *,
        result_text: str = "fake OCR text",
        raise_exception: Exception | None = None,
        check_cancel: bool = True,
    ) -> None:
        self._result_text = result_text
        self._raise = raise_exception
        self._check_cancel = check_cancel
        self.last_request: OcrRequest | None = None
        self.last_cancel: CancelToken | None = None
        self.call_count = 0

    def recognize(self, request: OcrRequest, cancel: CancelToken) -> OcrResult:
        self.last_request = request
        self.last_cancel = cancel
        self.call_count += 1
        if self._check_cancel and cancel is not None and cancel.is_cancelled:
            raise OcrError("cancelled")
        if self._raise is not None:
            raise self._raise
        return OcrResult(text=self._result_text, raw_blocks=[], pipeline=request.pipeline)

    def recognize_batch(
        self, requests: list[OcrRequest], cancel: CancelToken
    ) -> list[OcrResult | None]:
        self.last_cancel = cancel
        self.call_count += 1
        if self._check_cancel and cancel is not None and cancel.is_cancelled:
            raise OcrError("cancelled")
        if self._raise is not None:
            raise self._raise
        return [
            OcrResult(text=self._result_text, raw_blocks=[], pipeline=r.pipeline)
            for r in requests
        ]


class TestOcrRequest:
    def test_ocr_request_is_frozen(self):
        req = OcrRequest(image_data=b"png", pipeline="OCR")
        with pytest.raises(Exception):  # noqa: B017
            req.pipeline = "Table"  # type: ignore[misc]

    def test_ocr_request_fields(self):
        req = OcrRequest(image_data=b"png", pipeline="OCR", language="ch")
        assert req.image_data == b"png"
        assert req.pipeline == "OCR"
        assert req.language == "ch"


class TestOcrResult:
    def test_ocr_result_is_frozen(self):
        result = OcrResult(text="hello", raw_blocks=[], pipeline="OCR")
        with pytest.raises(Exception):  # noqa: B017
            result.text = "world"  # type: ignore[misc]


class TestCancelToken:
    def test_cancel_token_default_not_cancelled(self):
        token = CancelToken()
        assert token.is_cancelled is False

    def test_cancel_token_cancel(self):
        token = CancelToken()
        token.cancel()
        assert token.is_cancelled is True


class TestOcrFacade:
    def test_facade_implements_protocol(self):
        """OcrFacade 应满足 OcrApplication Protocol。"""
        adapter = _FakeOcrAdapter()
        facade = OcrFacade(adapter)
        assert isinstance(facade, OcrApplication)

    def test_recognize_passes_request_and_cancel(self):
        adapter = _FakeOcrAdapter()
        facade = OcrFacade(adapter)
        req = OcrRequest(image_data=b"png", pipeline="OCR")
        cancel = CancelToken()

        result = facade.recognize(req, cancel)

        assert adapter.last_request is req
        assert adapter.last_cancel is cancel
        assert isinstance(result, OcrResult)
        assert result.text == "fake OCR text"

    def test_recognize_propagates_ocr_error(self):
        adapter = _FakeOcrAdapter(raise_exception=OcrError("backend failed"))
        facade = OcrFacade(adapter)
        req = OcrRequest(image_data=b"png", pipeline="OCR")

        with pytest.raises(OcrError, match="backend failed"):
            facade.recognize(req, CancelToken())

    def test_recognize_wraps_generic_exception(self):
        """非 OcrError 异常应被包装为 OcrError。"""
        adapter = _FakeOcrAdapter(raise_exception=RuntimeError("unexpected"))
        facade = OcrFacade(adapter)
        req = OcrRequest(image_data=b"png", pipeline="OCR")

        with pytest.raises(OcrError):
            facade.recognize(req, CancelToken())

    def test_recognize_respects_cancel(self):
        adapter = _FakeOcrAdapter()
        facade = OcrFacade(adapter)
        req = OcrRequest(image_data=b"png", pipeline="OCR")
        cancel = CancelToken()
        cancel.cancel()

        with pytest.raises(OcrError, match="cancel"):
            facade.recognize(req, cancel)

    def test_recognize_without_cancel(self):
        """cancel=None 时应正常执行。"""
        adapter = _FakeOcrAdapter(check_cancel=False)
        facade = OcrFacade(adapter)
        req = OcrRequest(image_data=b"png", pipeline="OCR")

        result = facade.recognize(req, None)  # type: ignore[arg-type]
        assert result.text == "fake OCR text"


class TestOcrFacadeBatch:
    """recognize_batch 各分支：cancel-pre-check、空请求、OcrError 透传、generic 包装。"""

    def _req(self):
        return OcrRequest(image_data=b"png", pipeline="OCR")

    def test_batch_empty_requests_returns_empty_list(self):
        """空请求列表直接返回 []，不调用 adapter（line 79-80）。"""
        adapter = _FakeOcrAdapter()
        facade = OcrFacade(adapter)
        assert facade.recognize_batch([], CancelToken()) == []

    def test_batch_passes_requests_and_cancel(self):
        adapter = _FakeOcrAdapter()
        facade = OcrFacade(adapter)
        reqs = [self._req(), self._req()]
        result = facade.recognize_batch(reqs, CancelToken())
        assert len(result) == 2

    def test_batch_respects_cancel(self):
        """cancel 已取消时 raise（line 77-78）。"""
        adapter = _FakeOcrAdapter()
        facade = OcrFacade(adapter)
        cancel = CancelToken()
        cancel.cancel()
        with pytest.raises(OcrError, match="cancel"):
            facade.recognize_batch([self._req()], cancel)

    def test_batch_without_cancel(self):
        """cancel=None 时正常执行。"""
        adapter = _FakeOcrAdapter(check_cancel=False)
        facade = OcrFacade(adapter)
        result = facade.recognize_batch([self._req()], None)
        assert len(result) == 1

    def test_batch_propagates_ocr_error(self):
        """OcrError 直接透传（line 83-84）。"""
        adapter = _FakeOcrAdapter(raise_exception=OcrError("backend"))
        facade = OcrFacade(adapter)
        with pytest.raises(OcrError, match="backend"):
            facade.recognize_batch([self._req()], CancelToken())

    def test_batch_wraps_generic_exception(self):
        """非 OcrError 异常包装为 OcrError（line 85-86）。"""
        adapter = _FakeOcrAdapter(raise_exception=RuntimeError("boom"))
        facade = OcrFacade(adapter)
        with pytest.raises(OcrError, match="batch"):
            facade.recognize_batch([self._req()], CancelToken())
