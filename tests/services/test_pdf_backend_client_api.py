"""PdfBackendClient 业务 API 委托方法单元测试。

既有 test_pdf_backend_*.py 启动真实子进程做集成测试（慢、依赖 fitz 后端进程）。
本文件用 mock 覆盖各业务 API 的参数转发与 schema 解析，快速、无副作用。

覆盖：
- health/open_session/close_session/get_model
- render_thumbnail/render_preview（默认值与自定义值）
- detect_text_layers
- rotate/delete_pages/insert_blank/insert_from/move_page/reorder
- add_text_layer/add_text_layer_batch/rewrite_text_layer/update_block_text
- delete_text_layers_stream/save/cancel/reset_cancel
- _post/_get 错误映射（PdfBackendError、HTTPError、>=400）
- _ensure_started 崩溃重启路径
"""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from vibeocr.backend.ipc.schemas import (
    DetectTextLayersResponse,
    HealthResponse,
    ModelDiff,
    MutateResponse,
    OpenResponse,
    PdfDocumentMirror,
    ProgressEvent,
    SaveResponse,
)
from vibeocr.backend.services.pdf_backend_client import (
    _HTTP_LONG_TIMEOUT,
    _HTTP_TIMEOUT,
    PdfBackendClient,
    PdfBackendError,
)


def _mock_response(content: bytes | dict[str, Any], status: int = 200) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status
    if isinstance(content, dict):
        resp.content = b""  # JSON via .json()
        resp.json.return_value = content
        resp.text = str(content)
    else:
        resp.content = content
        resp.text = content.decode("utf-8", errors="replace") if content else ""
    return resp


def _client_with_post() -> tuple[PdfBackendClient, MagicMock]:
    """构造一个不启动子进程、_post 被 mock 的客户端。"""
    client = PdfBackendClient()
    client._post = MagicMock()  # type: ignore[method-assign]
    return client, client._post


# ---------------------------------------------------------------------------
# 控制平面
# ---------------------------------------------------------------------------


class TestHealthOpenModel:
    def test_health_parses_response(self) -> None:
        client = PdfBackendClient()
        client._get = MagicMock(  # type: ignore[method-assign]
            return_value=_mock_response(
                HealthResponse(status="ok", sessions=2, pid=123)
                .model_dump_json()
                .encode()
            )
        )
        resp = client.health()
        assert resp.status == "ok"
        assert resp.sessions == 2
        client._get.assert_called_once_with("/health")

    def test_open_session(self) -> None:
        client, post = _client_with_post()
        open_resp = OpenResponse(session_id="s1", model=PdfDocumentMirror())
        post.return_value = _mock_response(open_resp.model_dump_json().encode())
        resp = client.open_session("doc.pdf")
        assert resp.session_id == "s1"
        post.assert_called_once()
        assert post.call_args.args[0] == "/session/open"

    def test_close_session(self) -> None:
        client, post = _client_with_post()
        client.close_session("s1")
        post.assert_called_once_with("/session/s1/close")

    def test_get_model(self) -> None:
        client, post = _client_with_post()
        post.return_value = _mock_response(
            PdfDocumentMirror(file_path="x.pdf").model_dump_json().encode()
        )
        model = client.get_model("s1")
        assert model.file_path == "x.pdf"
        assert post.call_args.args[0] == "/session/s1/model"


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------


class TestRender:
    def test_render_thumbnail_defaults(self) -> None:
        client, post = _client_with_post()
        post.return_value = _mock_response(b"PNG")
        assert client.render_thumbnail("s1", 0) == b"PNG"
        args, kwargs = post.call_args
        assert args[0] == "/session/s1/render_thumbnail"
        assert kwargs["timeout"] == _HTTP_TIMEOUT

    def test_render_thumbnail_custom_size(self) -> None:
        client, post = _client_with_post()
        post.return_value = _mock_response(b"PNG")
        client.render_thumbnail("s1", 2, size=256)
        payload = post.call_args.args[1]
        assert payload["size"] == 256
        assert payload["page"] == 2

    def test_render_preview_defaults(self) -> None:
        client, post = _client_with_post()
        post.return_value = _mock_response(b"PNG")
        client.render_preview("s1", 1)
        args, kwargs = post.call_args
        assert args[0] == "/session/s1/render_preview"
        assert kwargs["timeout"] == _HTTP_LONG_TIMEOUT

    def test_render_preview_custom_dpi(self) -> None:
        client, post = _client_with_post()
        post.return_value = _mock_response(b"PNG")
        client.render_preview("s1", 3, dpi=300)
        payload = post.call_args.args[1]
        assert payload["dpi"] == 300


# ---------------------------------------------------------------------------
# detect + mutate
# ---------------------------------------------------------------------------


class TestDetectAndMutate:
    def test_detect_text_layers(self) -> None:
        client, post = _client_with_post()
        post.return_value = _mock_response(
            DetectTextLayersResponse(text_layers=[]).model_dump_json().encode()
        )
        resp = client.detect_text_layers("s1", 5)
        assert isinstance(resp, DetectTextLayersResponse)
        payload = post.call_args.args[1]
        assert payload["page"] == 5

    def _client_returning_mutate(self) -> tuple[PdfBackendClient, MagicMock]:
        client, post = _client_with_post()
        post.return_value = _mock_response(
            MutateResponse(diff=ModelDiff()).model_dump_json().encode()
        )
        return client, post

    def test_rotate(self) -> None:
        client, post = self._client_returning_mutate()
        resp = client.rotate("s1", [0, 1], 90)
        assert isinstance(resp, MutateResponse)
        assert post.call_args.args[0] == "/session/s1/rotate"
        assert post.call_args.args[1] == {"pages": [0, 1], "angle": 90}

    def test_delete_pages(self) -> None:
        client, post = self._client_returning_mutate()
        client.delete_pages("s1", [2])
        assert post.call_args.args == ("/session/s1/delete_pages", {"pages": [2]})

    def test_insert_blank_defaults(self) -> None:
        client, post = self._client_returning_mutate()
        client.insert_blank("s1", 0)
        assert post.call_args.args[1] == {
            "after_index": 0,
            "width": 612.0,
            "height": 792.0,
        }

    def test_insert_blank_custom(self) -> None:
        client, post = self._client_returning_mutate()
        client.insert_blank("s1", 1, width=100.0, height=200.0)
        assert post.call_args.args[1]["width"] == 100.0

    def test_insert_from(self) -> None:
        client, post = self._client_returning_mutate()
        client.insert_from("s1", "src.pdf", 3)
        assert post.call_args.args[1] == {
            "source_path": "src.pdf",
            "after_index": 3,
        }

    def test_move_page(self) -> None:
        client, post = self._client_returning_mutate()
        client.move_page("s1", 0, 2)
        assert post.call_args.args == (
            "/session/s1/move_page",
            {"from_index": 0, "to_index": 2},
        )

    def test_reorder(self) -> None:
        client, post = self._client_returning_mutate()
        client.reorder("s1", [2, 0, 1])
        assert post.call_args.args == ("/session/s1/reorder", {"new_order": [2, 0, 1]})


class TestTextLayerOperations:
    def _client(self) -> tuple[PdfBackendClient, MagicMock]:
        client, post = _client_with_post()
        post.return_value = _mock_response(
            MutateResponse(diff=ModelDiff()).model_dump_json().encode()
        )
        return client, post

    def test_add_text_layer(self) -> None:
        client, post = self._client()
        client.add_text_layer(
            "s1", 1, {"text": "x"}, pdf_settings={"c": True}, overwrite=True
        )
        assert post.call_args.args[0] == "/session/s1/add_text_layer"
        assert post.call_args.args[1] == {
            "page": 1,
            "ocr_result": {"text": "x"},
            "pdf_settings": {"c": True},
            "overwrite": True,
        }

    def test_add_text_layer_batch(self) -> None:
        client, post = self._client()
        client.add_text_layer_batch(
            "s1",
            [{"page": 0, "ocr_result": {"a": 1}}, {"page": 1, "ocr_result": {}}],
            pdf_settings=None,
            overwrite=False,
            save=True,
        )
        args, kwargs = post.call_args
        assert args[0] == "/session/s1/add_text_layer_batch"
        assert kwargs["timeout"] == _HTTP_LONG_TIMEOUT
        payload = args[1]
        assert len(payload["pages"]) == 2
        assert payload["pages"][0]["page"] == 0
        assert payload["save"] is True

    def test_rewrite_text_layer(self) -> None:
        from vibeocr.backend.models.ocr_result import TextBlock

        client, post = self._client()
        blocks = [
            TextBlock(
                text="hi",
                score=0.9,
                bbox=(1, 2, 3, 4),
                polygon=(1, 2, 3, 4, 5, 6, 7, 8),
            )
        ]
        client.rewrite_text_layer("s1", 0, blocks, preproc_angle=90, pdf_settings=None)
        args = post.call_args.args
        assert args[0] == "/session/s1/rewrite_text_layer"
        assert args[1]["page"] == 0
        assert args[1]["preproc_angle"] == 90
        assert args[1]["text_blocks"][0]["text"] == "hi"

    def test_update_block_text(self) -> None:
        client, post = self._client()
        client.update_block_text("s1", 0, 2, "new")
        assert post.call_args.args == (
            "/session/s1/update_block_text",
            {"page": 0, "block_index": 2, "new_text": "new"},
        )


# ---------------------------------------------------------------------------
# save / cancel / reset_cancel
# ---------------------------------------------------------------------------


class TestSaveCancel:
    def test_save_with_all_params(self) -> None:
        client, post = _client_with_post()
        post.return_value = _mock_response(
            SaveResponse(path="out.pdf", diff=ModelDiff()).model_dump_json().encode()
        )
        resp = client.save("s1", "out.pdf", {"c": True}, rewrite_text_layers=False)
        assert resp.path == "out.pdf"
        args, kwargs = post.call_args
        assert args[0] == "/session/s1/save"
        assert kwargs["timeout"] == _HTTP_LONG_TIMEOUT
        assert args[1] == {
            "path": "out.pdf",
            "pdf_settings": {"c": True},
            "rewrite_text_layers": False,
        }

    def test_save_defaults(self) -> None:
        client, post = _client_with_post()
        post.return_value = _mock_response(
            SaveResponse(path="x", diff=ModelDiff()).model_dump_json().encode()
        )
        client.save("s1")
        assert post.call_args.args[1]["path"] is None
        assert post.call_args.args[1]["rewrite_text_layers"] is True

    def test_cancel(self) -> None:
        client, post = _client_with_post()
        client.cancel("s1")
        post.assert_called_once_with("/session/s1/cancel")

    def test_reset_cancel(self) -> None:
        client, post = _client_with_post()
        client.reset_cancel("s1")
        post.assert_called_once_with("/session/s1/reset_cancel")


# ---------------------------------------------------------------------------
# _post / _get 错误映射
# ---------------------------------------------------------------------------


class TestPostGetErrorMapping:
    def test_post_wraps_httperror(self) -> None:
        client = PdfBackendClient()
        client._ensure_started = MagicMock(  # type: ignore[method-assign]
            return_value=MagicMock()
        )
        mock_http_client = client._ensure_started.return_value
        mock_http_client.post.side_effect = httpx.ConnectError("refused")
        with pytest.raises(PdfBackendError, match="后端调用失败"):
            client._post("/x", {"a": 1})

    def test_post_maps_4xx_with_json_detail(self) -> None:
        client = PdfBackendClient()
        client._ensure_started = MagicMock(  # type: ignore[method-assign]
            return_value=MagicMock()
        )
        mock_http_client = client._ensure_started.return_value
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 400
        resp.text = "bad"
        resp.json.return_value = {"detail": "page out of range"}
        mock_http_client.post.return_value = resp
        with pytest.raises(PdfBackendError, match="page out of range"):
            client._post("/x", {"a": 1})

    def test_post_maps_5xx_without_json_detail(self) -> None:
        client = PdfBackendClient()
        client._ensure_started = MagicMock(  # type: ignore[method-assign]
            return_value=MagicMock()
        )
        mock_http_client = client._ensure_started.return_value
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 500
        resp.text = "internal error"
        resp.json.side_effect = ValueError("not json")
        mock_http_client.post.return_value = resp
        with pytest.raises(PdfBackendError, match="internal error"):
            client._post("/x", None)

    def test_get_wraps_httperror(self) -> None:
        client = PdfBackendClient()
        client._ensure_started = MagicMock(  # type: ignore[method-assign]
            return_value=MagicMock()
        )
        mock_http_client = client._ensure_started.return_value
        mock_http_client.get.side_effect = httpx.ReadTimeout("slow")
        with pytest.raises(PdfBackendError, match="后端调用失败"):
            client._get("/health")

    def test_get_maps_4xx(self) -> None:
        client = PdfBackendClient()
        client._ensure_started = MagicMock(  # type: ignore[method-assign]
            return_value=MagicMock()
        )
        mock_http_client = client._ensure_started.return_value
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 404
        resp.text = "not found"
        mock_http_client.get.return_value = resp
        with pytest.raises(PdfBackendError, match="404"):
            client._get("/missing")


# ---------------------------------------------------------------------------
# _ensure_started 重启路径
# ---------------------------------------------------------------------------


class TestEnsureStartedRestart:
    def test_returns_cached_client_when_alive(self) -> None:
        client = PdfBackendClient()
        client._started = True
        client._is_alive = MagicMock(return_value=True)  # type: ignore[method-assign]
        existing = MagicMock(spec=httpx.Client)
        tid = threading.get_ident()
        client._http_clients[tid] = existing
        assert client._ensure_started() is existing

    def test_starts_backend_and_creates_client_when_dead(self) -> None:
        client = PdfBackendClient()
        client._started = True
        client._is_alive = MagicMock(return_value=False)  # type: ignore[method-assign]
        client.start = MagicMock()  # type: ignore[method-assign]
        client._base_url = "http://x"
        result = client._ensure_started()
        client.start.assert_called_once()
        assert isinstance(result, httpx.Client)
        result.close()


# ---------------------------------------------------------------------------
# 生命周期: instance / stop / _stop_locked
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_instance_is_singleton(self) -> None:
        PdfBackendClient._instance = None
        a = PdfBackendClient.instance()
        b = PdfBackendClient.instance()
        assert a is b
        PdfBackendClient._instance = None

    def test_stop_closes_http_clients_and_process(self) -> None:
        client = PdfBackendClient()
        client._started = True
        proc = MagicMock()
        client._process = proc
        client._job_guard = MagicMock()
        mock_client = MagicMock(spec=httpx.Client)
        client._http_clients[threading.get_ident()] = mock_client
        client.stop()
        proc.terminate.assert_called_once()
        proc.wait.assert_called_once()
        mock_client.close.assert_called_once()
        assert client._started is False
        assert client._process is None

    def test_stop_falls_back_to_kill_on_terminate_wait_failure(self) -> None:
        client = PdfBackendClient()
        client._started = True
        proc = MagicMock()
        proc.terminate.side_effect = RuntimeError("x")
        proc.wait.side_effect = RuntimeError("x")
        client._process = proc
        client._job_guard = None
        client.stop()
        proc.kill.assert_called_once()
        assert client._started is False

    def test_stop_handles_job_guard_close_failure(self) -> None:
        client = PdfBackendClient()
        client._started = True
        client._process = None
        guard = MagicMock()
        guard.close.side_effect = RuntimeError("x")
        client._job_guard = guard
        client.stop()  # 不应抛
        assert client._job_guard is None


# ---------------------------------------------------------------------------
# 流式方法: load_stream / delete_text_layers_stream
# ---------------------------------------------------------------------------


class _StreamContextManager:
    """模拟 httpx client.stream() 返回的上下文管理器。"""

    def __init__(self, response: MagicMock) -> None:
        self._response = response

    def __enter__(self) -> MagicMock:
        return self._response

    def __exit__(self, *_args: object) -> None:
        return None


class TestStreamingMethods:
    def test_load_stream_yields_progress_events(self) -> None:
        client = PdfBackendClient()
        mock_http_client = MagicMock(spec=httpx.Client)
        client._ensure_started = MagicMock(return_value=mock_http_client)  # type: ignore[method-assign]
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.iter_lines.return_value = iter(
            [
                ProgressEvent(phase="load", current=0, total=2).model_dump_json(),
                ProgressEvent(phase="load", current=1, total=2).model_dump_json(),
            ]
        )
        mock_http_client.stream.return_value = _StreamContextManager(resp)
        events = list(client.load_stream("s1"))
        assert len(events) == 2
        assert all(isinstance(e, ProgressEvent) for e in events)
        mock_http_client.stream.assert_called_once()

    def test_load_stream_raises_on_4xx(self) -> None:
        client = PdfBackendClient()
        mock_http_client = MagicMock(spec=httpx.Client)
        client._ensure_started = MagicMock(return_value=mock_http_client)  # type: ignore[method-assign]
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 400
        resp.iter_lines.return_value = iter([])
        mock_http_client.stream.return_value = _StreamContextManager(resp)
        with pytest.raises(PdfBackendError, match="load 失败"):
            list(client.load_stream("s1"))

    def test_load_stream_wraps_httperror(self) -> None:
        client = PdfBackendClient()
        mock_http_client = MagicMock(spec=httpx.Client)
        client._ensure_started = MagicMock(return_value=mock_http_client)  # type: ignore[method-assign]
        mock_http_client.stream.side_effect = httpx.ConnectError("refused")
        with pytest.raises(PdfBackendError, match="流式调用失败"):
            list(client.load_stream("s1"))

    def test_load_stream_skips_empty_lines(self) -> None:
        client = PdfBackendClient()
        mock_http_client = MagicMock(spec=httpx.Client)
        client._ensure_started = MagicMock(return_value=mock_http_client)  # type: ignore[method-assign]
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.iter_lines.return_value = iter(
            ["", ProgressEvent(phase="load").model_dump_json(), ""]
        )
        mock_http_client.stream.return_value = _StreamContextManager(resp)
        events = list(client.load_stream("s1"))
        assert len(events) == 1

    def test_delete_text_layers_stream_yields_events(self) -> None:
        client = PdfBackendClient()
        mock_http_client = MagicMock(spec=httpx.Client)
        client._ensure_started = MagicMock(return_value=mock_http_client)  # type: ignore[method-assign]
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.iter_lines.return_value = iter(
            [ProgressEvent(phase="delete", current=1).model_dump_json()]
        )
        mock_http_client.stream.return_value = _StreamContextManager(resp)
        events = list(client.delete_text_layers_stream("s1", [0, 1]))
        assert len(events) == 1
        assert events[0].phase.value == "delete"

    def test_delete_text_layers_stream_raises_on_4xx(self) -> None:
        client = PdfBackendClient()
        mock_http_client = MagicMock(spec=httpx.Client)
        client._ensure_started = MagicMock(return_value=mock_http_client)  # type: ignore[method-assign]
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 500
        resp.iter_lines.return_value = iter([])
        mock_http_client.stream.return_value = _StreamContextManager(resp)
        with pytest.raises(PdfBackendError, match="删除文字层失败"):
            list(client.delete_text_layers_stream("s1", [0]))

    def test_delete_text_layers_stream_wraps_httperror(self) -> None:
        client = PdfBackendClient()
        mock_http_client = MagicMock(spec=httpx.Client)
        client._ensure_started = MagicMock(return_value=mock_http_client)  # type: ignore[method-assign]
        mock_http_client.stream.side_effect = httpx.ReadTimeout("slow")
        with pytest.raises(PdfBackendError, match="流式调用失败"):
            list(client.delete_text_layers_stream("s1", [0]))
