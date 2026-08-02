"""pdf_backend_client 覆盖补充：进程生命周期、错误处理、流式方法。

mock httpx + subprocess，不启动真实后端子进程。
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import httpx
import pytest
from vibeocr.backend.services.pdf_backend_client import (
    PdfBackendClient,
    PdfBackendError,
)


def _mock_resp(status=200, json_data=None, content=b"", text=None, reason="OK"):
    r = MagicMock(spec=httpx.Response)
    r.status_code = status
    r.reason_phrase = reason
    r.headers = {}
    if json_data is not None:
        r.json.return_value = json_data
        r.content = content or b"{}"
        r.text = text or "{}"
    else:
        r.content = content
        r.text = text or ""
        r.json.side_effect = ValueError("no json")
    return r


# ---- 进程生命周期 ------------------------------------------------------


class TestResolvePythonExe:
    def test_embedded_exists(self, monkeypatch):
        client = PdfBackendClient()
        from pathlib import Path

        import vibeocr.backend.env_manager as env

        monkeypatch.setattr(env, "get_project_root", lambda: Path("/fake"))
        monkeypatch.setattr(
            env, "get_embedded_python", lambda r: Path("/fake/embedded/python.exe")
        )
        monkeypatch.setattr(Path, "exists", lambda self: True)
        assert client._resolve_python_exe() == str(Path("/fake/embedded/python.exe"))

    def test_fallback_to_sys_executable(self, monkeypatch):
        import sys
        from pathlib import Path

        client = PdfBackendClient()
        import vibeocr.backend.env_manager as env

        monkeypatch.setattr(env, "get_project_root", lambda: Path("/fake"))
        monkeypatch.setattr(
            env, "get_embedded_python", lambda r: Path("/fake/none.exe")
        )
        result = client._resolve_python_exe()
        assert result == sys.executable


class TestGetBackendEnv:
    def test_removes_source_and_editable_environment(self, monkeypatch):
        client = PdfBackendClient()
        monkeypatch.setenv("PYTHONPATH", "/src/a")
        monkeypatch.setenv("PYTHONHOME", "/python/home")
        monkeypatch.setenv("VIRTUAL_ENV", "/venv")
        monkeypatch.setenv("VIBEOCR_REPOSITORY_ROOT", "/workspace")
        env_result = client._get_backend_env()
        assert "PYTHONPATH" not in env_result
        assert "PYTHONHOME" not in env_result
        assert "VIRTUAL_ENV" not in env_result
        assert "VIBEOCR_REPOSITORY_ROOT" not in env_result

    def test_frozen_mode_does_not_inject_meipass(self, monkeypatch):
        import sys

        client = PdfBackendClient()
        monkeypatch.setattr(sys, "frozen", True, raising=False)
        monkeypatch.setattr(sys, "_MEIPASS", "/fake/meipass", raising=False)
        monkeypatch.delenv("PYTHONPATH", raising=False)
        env_result = client._get_backend_env()
        assert "PYTHONPATH" not in env_result
        # 恢复
        monkeypatch.delattr(sys, "frozen", raising=False)
        monkeypatch.delattr(sys, "_MEIPASS", raising=False)


class TestFindFreePort:
    def test_returns_int(self):
        client = PdfBackendClient()
        port = client._find_free_port()
        assert isinstance(port, int)
        assert port > 0


class TestStartLogReader:
    def test_reads_stdout_lines(self):
        client = PdfBackendClient()
        proc = MagicMock()
        proc.stdout = MagicMock()
        proc.stdout.__iter__ = lambda self: iter([b"line1\n", b"line2\n", b""])
        client._start_log_reader(proc)
        time.sleep(0.2)

    def test_read_exception_swallowed(self):
        client = PdfBackendClient()
        proc = MagicMock()
        proc.stdout = MagicMock()
        proc.stdout.__iter__ = MagicMock(side_effect=RuntimeError("read boom"))
        client._start_log_reader(proc)
        time.sleep(0.2)

    def test_empty_line_skipped(self):
        client = PdfBackendClient()
        proc = MagicMock()
        proc.stdout = MagicMock()
        proc.stdout.__iter__ = lambda self: iter([b"", b"real\n", b""])
        client._start_log_reader(proc)
        time.sleep(0.2)


class TestStartAndLifecycle:
    def test_start_idempotent_when_alive(self, monkeypatch):
        client = PdfBackendClient()
        client._started = True
        client._process = MagicMock()
        client._process.poll.return_value = None  # alive
        # 不应重新启动
        with (
            patch.object(client, "_stop_locked") as stop,
            patch.object(client, "_resolve_python_exe") as exe,
        ):
            client.start()
        stop.assert_not_called()
        exe.assert_not_called()

    def test_wait_ready_process_exited_raises(self):
        client = PdfBackendClient()
        client._process = MagicMock()
        client._process.poll.return_value = 1  # exited
        client._process.returncode = 1
        client._process.stdout = MagicMock()
        client._process.stdout.readline.return_value = b""  # EOF
        with pytest.raises(PdfBackendError, match="启动失败"):
            client._wait_ready()

    def test_wait_ready_timeout_raises(self, monkeypatch):
        client = PdfBackendClient()
        client._base_url = "http://127.0.0.1:1"
        client._process = MagicMock()
        client._process.poll.return_value = None  # alive

        # health check 永远失败
        def _fail_get(url, timeout):
            raise ConnectionError("no connection")

        monkeypatch.setattr(
            "vibeocr.backend.services.pdf_backend_client.httpx.get", _fail_get
        )
        # 缩短超时
        import vibeocr.backend.services.pdf_backend_client as mod

        monkeypatch.setattr(mod, "_BACKEND_START_TIMEOUT", 0.5)
        with pytest.raises(PdfBackendError, match="超时"):
            client._wait_ready()

    def test_wait_ready_health_200_returns(self, monkeypatch):
        client = PdfBackendClient()
        client._base_url = "http://127.0.0.1:1"
        client._process = MagicMock()
        client._process.poll.return_value = None

        resp = _mock_resp(status=200)
        monkeypatch.setattr(
            "vibeocr.backend.services.pdf_backend_client.httpx.get",
            lambda *a, **k: resp,
        )
        client._wait_ready()  # 不应抛异常

    def test_drain_stdout_tail_no_process(self):
        client = PdfBackendClient()
        client._process = None
        assert client._drain_stdout_tail() == ""

    def test_drain_stdout_tail_no_stdout(self):
        client = PdfBackendClient()
        client._process = MagicMock()
        client._process.stdout = None
        assert client._drain_stdout_tail() == ""

    def test_drain_stdout_tail_reads_lines(self):
        client = PdfBackendClient()
        client._process = MagicMock()
        client._process.stdout = MagicMock()
        client._process.stdout.readline.side_effect = [b"line1\n", b"line2\n", b""]
        result = client._drain_stdout_tail()
        assert "line1" in result
        assert "line2" in result

    def test_drain_stdout_tail_exception_returns_empty(self):
        client = PdfBackendClient()
        client._process = MagicMock()
        client._process.stdout = MagicMock()
        client._process.stdout.readline.side_effect = RuntimeError("boom")
        assert client._drain_stdout_tail() == ""

    def test_stop_locked_terminates_process(self):
        client = PdfBackendClient()
        mock_proc = MagicMock()
        mock_proc.wait.return_value = 0
        client._process = mock_proc
        mock_guard = MagicMock()
        client._job_guard = mock_guard
        client._http_clients = {1: MagicMock()}
        client._started = True

        client._stop_locked()
        mock_guard.close.assert_called_once()
        mock_proc.terminate.assert_called_once()
        assert client._process is None
        assert client._job_guard is None
        assert client._started is False
        assert len(client._http_clients) == 0

    def test_stop_locked_kill_on_timeout(self):
        import subprocess

        client = PdfBackendClient()
        mock_proc = MagicMock()
        mock_proc.wait.side_effect = subprocess.TimeoutExpired(cmd="x", timeout=5)
        client._process = mock_proc

        client._stop_locked()
        mock_proc.kill.assert_called_once()

    def test_stop_locked_job_guard_close_exception(self):
        client = PdfBackendClient()
        mock_guard = MagicMock()
        mock_guard.close.side_effect = RuntimeError("close boom")
        client._job_guard = mock_guard
        client._process = None

        client._stop_locked()  # 不应抛
        assert client._job_guard is None

    def test_stop_locked_process_terminate_exception(self):
        client = PdfBackendClient()
        mock_proc = MagicMock()
        mock_proc.terminate.side_effect = RuntimeError("term boom")
        mock_proc.kill.side_effect = RuntimeError("kill boom")
        client._process = mock_proc

        client._stop_locked()  # 不应抛
        assert client._process is None

    def test_stop_calls_stop_locked(self):
        client = PdfBackendClient()
        with patch.object(client, "_stop_locked") as stop:
            client.stop()
        stop.assert_called_once()

    def test_is_alive_true(self):
        client = PdfBackendClient()
        client._process = MagicMock()
        client._process.poll.return_value = None
        assert client._is_alive() is True

    def test_is_alive_false_no_process(self):
        client = PdfBackendClient()
        client._process = None
        assert client._is_alive() is False

    def test_is_alive_false_dead(self):
        client = PdfBackendClient()
        client._process = MagicMock()
        client._process.poll.return_value = 1
        assert client._is_alive() is False


class TestEnsureStarted:
    def test_returns_existing_client_when_alive(self):
        client = PdfBackendClient()
        client._started = True
        client._process = MagicMock()
        client._process.poll.return_value = None
        existing = MagicMock(spec=httpx.Client)
        tid = threading.get_ident()
        client._http_clients[tid] = existing
        result = client._ensure_started()
        assert result is existing

    def test_starts_when_not_started(self, monkeypatch):
        client = PdfBackendClient()
        client._started = False
        with patch.object(client, "start") as start_mock:
            client._ensure_started()
        start_mock.assert_called_once()

    def test_restarts_when_dead(self, monkeypatch):
        client = PdfBackendClient()
        client._started = True
        client._process = MagicMock()
        client._process.poll.return_value = 1  # dead
        with patch.object(client, "start") as start_mock:
            client._ensure_started()
        start_mock.assert_called_once()


# ---- _post / _get 错误处理 --------------------------------------------


class TestPostGetErrors:
    def test_post_http_error_raises(self, monkeypatch):
        client = PdfBackendClient()
        mock_client = MagicMock(spec=httpx.Client)
        mock_client.post.side_effect = httpx.ConnectError("conn")
        client._http_clients = {threading.get_ident(): mock_client}
        client._started = True
        client._process = MagicMock()
        client._process.poll.return_value = None
        with pytest.raises(PdfBackendError, match="后端调用失败"):
            client._post("/test")

    def test_post_400_raises_with_detail(self, monkeypatch):
        client = PdfBackendClient()
        mock_client = MagicMock(spec=httpx.Client)
        resp = _mock_resp(
            status=400, json_data={"detail": "bad request"}, text='{"detail":"bad"}'
        )
        mock_client.post.return_value = resp
        client._http_clients = {threading.get_ident(): mock_client}
        client._started = True
        client._process = MagicMock()
        client._process.poll.return_value = None
        with pytest.raises(PdfBackendError, match="bad request"):
            client._post("/test")

    def test_post_400_detail_fallback_to_text(self, monkeypatch):
        client = PdfBackendClient()
        mock_client = MagicMock(spec=httpx.Client)
        resp = _mock_resp(status=500, text="internal error")
        mock_client.post.return_value = resp
        client._http_clients = {threading.get_ident(): mock_client}
        client._started = True
        client._process = MagicMock()
        client._process.poll.return_value = None
        with pytest.raises(PdfBackendError, match="internal error"):
            client._post("/test")

    def test_get_http_error_raises(self, monkeypatch):
        client = PdfBackendClient()
        mock_client = MagicMock(spec=httpx.Client)
        mock_client.get.side_effect = httpx.ConnectError("conn")
        client._http_clients = {threading.get_ident(): mock_client}
        client._started = True
        client._process = MagicMock()
        client._process.poll.return_value = None
        with pytest.raises(PdfBackendError, match="后端调用失败"):
            client._get("/test")

    def test_get_400_raises(self, monkeypatch):
        client = PdfBackendClient()
        mock_client = MagicMock(spec=httpx.Client)
        resp = _mock_resp(status=404, text="not found")
        mock_client.get.return_value = resp
        client._http_clients = {threading.get_ident(): mock_client}
        client._started = True
        client._process = MagicMock()
        client._process.poll.return_value = None
        with pytest.raises(PdfBackendError, match="404"):
            client._get("/test")

    def test_post_no_payload(self, monkeypatch):
        """_post payload=None → 不传 json（line 387 False 分支）。"""
        client = PdfBackendClient()
        mock_client = MagicMock(spec=httpx.Client)
        resp = _mock_resp(status=200)
        mock_client.post.return_value = resp
        client._http_clients = {threading.get_ident(): mock_client}
        client._started = True
        client._process = MagicMock()
        client._process.poll.return_value = None
        result = client._post("/test")  # 无 payload
        assert result is resp
        # 确认 json 未传
        call_kwargs = mock_client.post.call_args.kwargs
        assert "json" not in call_kwargs or call_kwargs.get("json") is None


# ---- _estimate_request_bytes / _estimate_response_bytes ---------------


class TestEstimateBytes:
    def test_estimate_request_none(self):
        assert PdfBackendClient._estimate_request_bytes(None) is None

    def test_estimate_request_bytes(self):
        assert PdfBackendClient._estimate_request_bytes(b"hello") == 5

    def test_estimate_request_bytearray(self):
        assert PdfBackendClient._estimate_request_bytes(bytearray(b"hi")) == 2

    def test_estimate_request_str(self):
        assert PdfBackendClient._estimate_request_bytes("hello") == 5

    def test_estimate_request_other(self):
        assert PdfBackendClient._estimate_request_bytes({"a": 1}) is not None

    def test_estimate_request_exception(self):
        class _Boom:
            def __str__(self):
                raise RuntimeError("boom")

        assert PdfBackendClient._estimate_request_bytes(_Boom()) is None

    def test_estimate_response_normal(self):
        resp = MagicMock(spec=httpx.Response)
        resp.headers = {"content-length": "100"}
        resp.content = b"x" * 100
        result = PdfBackendClient._estimate_response_bytes(resp)
        assert result is not None

    def test_estimate_response_no_headers(self):
        resp = MagicMock(spec=httpx.Response)
        resp.headers = None
        resp.content = b"data"
        result = PdfBackendClient._estimate_response_bytes(resp)
        assert result is not None

    def test_estimate_response_headers_exception(self):
        """headers 转为 dict 时抛异常 → 回退到 content（line 374-376）。"""
        resp = MagicMock(spec=httpx.Response)

        # headers 是一个 dict() 会抛异常的对象
        class _BadHeaders:
            def __iter__(self):
                raise RuntimeError("boom")

        resp.headers = _BadHeaders()
        resp.content = b"data"
        result = PdfBackendClient._estimate_response_bytes(resp)
        # headers 异常被捕获 → 回退到 content 长度
        assert result == len(b"data")

    def test_estimate_response_no_content(self):
        """content 非 bytes/str 且 headers 无 content-length → None。"""
        resp = MagicMock(spec=httpx.Response)
        resp.headers = {"x": "y"}
        resp.content = None  # 非 bytes/str
        result = PdfBackendClient._estimate_response_bytes(resp)
        assert result is None  # 无 content 且无 content-length header


# ---- 流式方法 ----------------------------------------------------------


class TestLoadStream:
    def test_load_stream_yields_events(self, monkeypatch):
        client = PdfBackendClient()
        client._started = True
        client._process = MagicMock()
        client._process.poll.return_value = None
        client._base_url = "http://x"

        from vibeocr.backend.ipc.schemas import ProgressEvent, ProgressPhase

        ev_json = ProgressEvent(
            phase=ProgressPhase.LOAD, current=1, total=1, message="done"
        ).model_dump_json()

        mock_stream_resp = MagicMock()
        mock_stream_resp.status_code = 200
        mock_stream_resp.iter_lines.return_value = iter([ev_json, ""])
        mock_stream_resp.headers = {}
        mock_stream_resp.reason_phrase = "OK"

        mock_client = MagicMock(spec=httpx.Client)
        mock_cm = MagicMock()
        mock_cm.__enter__ = lambda self: mock_stream_resp
        mock_cm.__exit__ = lambda *a: None
        mock_client.stream.return_value = mock_cm
        client._http_clients = {threading.get_ident(): mock_client}

        events = list(client.load_stream("s1"))
        assert len(events) == 1
        assert events[0].message == "done"

    def test_load_stream_400_raises(self, monkeypatch):
        client = PdfBackendClient()
        client._started = True
        client._process = MagicMock()
        client._process.poll.return_value = None
        client._base_url = "http://x"

        mock_stream_resp = MagicMock()
        mock_stream_resp.status_code = 400
        mock_stream_resp.iter_lines.return_value = iter([])
        mock_stream_resp.headers = {}
        mock_stream_resp.reason_phrase = "Bad"

        mock_client = MagicMock(spec=httpx.Client)
        mock_cm = MagicMock()
        mock_cm.__enter__ = lambda self: mock_stream_resp
        mock_cm.__exit__ = lambda *a: None
        mock_client.stream.return_value = mock_cm
        client._http_clients = {threading.get_ident(): mock_client}

        with pytest.raises(PdfBackendError, match="load 失败"):
            list(client.load_stream("s1"))

    def test_load_stream_http_error_raises(self, monkeypatch):
        client = PdfBackendClient()
        client._started = True
        client._process = MagicMock()
        client._process.poll.return_value = None
        client._base_url = "http://x"

        mock_client = MagicMock(spec=httpx.Client)
        mock_client.stream.side_effect = httpx.ConnectError("conn")
        client._http_clients = {threading.get_ident(): mock_client}

        with pytest.raises(PdfBackendError, match="流式调用失败"):
            list(client.load_stream("s1"))


class TestDeleteTextLayersStream:
    def test_stream_yields_events(self):
        client = PdfBackendClient()
        client._started = True
        client._process = MagicMock()
        client._process.poll.return_value = None
        client._base_url = "http://x"

        from vibeocr.backend.ipc.schemas import ProgressEvent, ProgressPhase

        ev_json = ProgressEvent(
            phase=ProgressPhase.DELETE, current=1, total=1, message="done"
        ).model_dump_json()

        mock_stream_resp = MagicMock()
        mock_stream_resp.status_code = 200
        mock_stream_resp.iter_lines.return_value = iter([ev_json, ""])
        mock_stream_resp.headers = {}
        mock_stream_resp.reason_phrase = "OK"

        mock_client = MagicMock(spec=httpx.Client)
        mock_cm = MagicMock()
        mock_cm.__enter__ = lambda self: mock_stream_resp
        mock_cm.__exit__ = lambda *a: None
        mock_client.stream.return_value = mock_cm
        client._http_clients = {threading.get_ident(): mock_client}

        events = list(client.delete_text_layers_stream("s1", [0]))
        assert len(events) == 1

    def test_stream_400_raises(self):
        client = PdfBackendClient()
        client._started = True
        client._process = MagicMock()
        client._process.poll.return_value = None

        mock_stream_resp = MagicMock()
        mock_stream_resp.status_code = 500
        mock_stream_resp.headers = {}
        mock_stream_resp.reason_phrase = "Err"

        mock_client = MagicMock(spec=httpx.Client)
        mock_cm = MagicMock()
        mock_cm.__enter__ = lambda self: mock_stream_resp
        mock_cm.__exit__ = lambda *a: None
        mock_client.stream.return_value = mock_cm
        client._http_clients = {threading.get_ident(): mock_client}

        with pytest.raises(PdfBackendError, match="删除文字层失败"):
            list(client.delete_text_layers_stream("s1", [0]))

    def test_stream_http_error_raises(self):
        client = PdfBackendClient()
        client._started = True
        client._process = MagicMock()
        client._process.poll.return_value = None

        mock_client = MagicMock(spec=httpx.Client)
        mock_client.stream.side_effect = httpx.ConnectError("conn")
        client._http_clients = {threading.get_ident(): mock_client}

        with pytest.raises(PdfBackendError, match="流式调用失败"):
            list(client.delete_text_layers_stream("s1", [0]))


# ---- instance 单例 ----------------------------------------------------


class TestInstance:
    def test_instance_creates_singleton(self):
        PdfBackendClient._instance = None
        c1 = PdfBackendClient.instance()
        c2 = PdfBackendClient.instance()
        assert c1 is c2
        PdfBackendClient._instance = None
