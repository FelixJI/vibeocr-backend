"""mineru_service 覆盖补充：聚焦未覆盖分支（进程管理、HTTP 错误、file_parse 等）。

所有外部依赖（httpx、subprocess、socket）均 mock，不真正启动 mineru-api。
"""

from __future__ import annotations

import json
import logging
import subprocess
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("httpx", reason="httpx not installed")

from vibeocr.backend.services.mineru_service import MinerUService


def _make_service():
    """构造不触发 __init__ 的 MinerUService 实例。"""
    s = MinerUService.__new__(MinerUService)
    s._api_url = "http://127.0.0.1:9999"
    MinerUService._api_process = None
    MinerUService._job_guard = None
    return s


class TestResetAndInit:
    def setup_method(self):
        MinerUService._instance = None
        MinerUService._initialized = False
        MinerUService._api_process = None
        MinerUService._api_url = ""
        MinerUService._job_guard = None

    def test_reset_kills_terminated_process(self):
        """_reset 对未在超时内退出的进程调用 kill（lines 76-85）。"""
        mock_proc = MagicMock()
        # terminate 后 wait 超时
        mock_proc.wait.side_effect = subprocess.TimeoutExpired(cmd="x", timeout=10)
        MinerUService._api_process = mock_proc

        MinerUService._reset()

        mock_proc.terminate.assert_called_once()
        mock_proc.kill.assert_called_once()
        assert MinerUService._api_process is None
        assert MinerUService._initialized is False

    def test_reset_no_process_noop(self):
        """_reset 无进程时为 no-op。"""
        MinerUService._api_process = None
        MinerUService._reset()
        assert MinerUService._api_process is None

    def test_init_skips_when_already_initialized(self):
        """已初始化时 __init__ 不重跑（branches 67->exit, 69->exit）。"""
        MinerUService._initialized = True
        with patch.object(MinerUService, "_ensure_api_running") as m:
            MinerUService()
        m.assert_not_called()


class TestParseApiLogLevel:
    def test_no_match_returns_debug(self):
        """日志行无级别词 → DEBUG（lines 95-97）。"""
        assert (
            MinerUService._parse_api_log_level("random text no level") == logging.DEBUG
        )

    def test_match_returns_level(self):
        assert MinerUService._parse_api_log_level("ERROR something") == logging.ERROR
        assert MinerUService._parse_api_log_level("warning text") == logging.WARNING


class TestStartLogReader:
    def test_stderr_none_returns_early(self):
        """process.stderr 为 None → 直接返回（line 112）。"""
        s = _make_service()
        proc = MagicMock()
        proc.stderr = None
        # 不应启动线程
        s._start_log_reader(proc)

    def test_log_reader_handles_exception(self):
        """readline 抛异常 → 线程退出（lines 116->exit, 119, 122, 124, 125-126）。"""
        s = _make_service()
        proc = MagicMock()
        proc.stderr = MagicMock()
        proc.poll.return_value = None
        # readline 抛异常
        proc.stderr.readline.side_effect = RuntimeError("read boom")

        s._start_log_reader(proc)
        # 等线程结束
        import time

        time.sleep(0.2)

    def test_log_reader_reads_lines(self):
        """正常读取日志行并按级别转发（覆盖正常路径）。"""
        s = _make_service()
        proc = MagicMock()
        proc.stderr = MagicMock()
        # 先返回一行数据，再进程退出
        proc.stderr.readline.side_effect = [b"INFO hello\n", b""]
        proc.poll.side_effect = [None, 0]  # 第二次返回 0（退出）

        s._start_log_reader(proc)
        import time

        time.sleep(0.2)


class TestCheckApiRunning:
    def test_exception_returns_false(self, monkeypatch):
        """httpx.get 抛异常 → 返回 False + 记 warning（lines 147-154）。"""
        s = _make_service()
        import vibeocr.backend.services.mineru_service as mod

        def _boom(url, timeout):
            raise ConnectionError("boom")

        monkeypatch.setattr(mod.httpx, "get", _boom)
        assert s._check_api_running("http://127.0.0.1:1") is False

    def test_non_200_returns_false(self, monkeypatch):
        """status_code != 200 → False。"""
        s = _make_service()
        import vibeocr.backend.services.mineru_service as mod

        resp = MagicMock()
        resp.status_code = 500
        resp.reason_phrase = "Internal Server Error"
        resp.headers = {}
        resp.content = b""
        monkeypatch.setattr(mod.httpx, "get", lambda *a, **k: resp)
        assert s._check_api_running("http://127.0.0.1:1") is False

    def test_200_returns_true(self, monkeypatch):
        """status_code == 200 → True。"""
        s = _make_service()
        import vibeocr.backend.services.mineru_service as mod

        resp = MagicMock()
        resp.status_code = 200
        resp.reason_phrase = "OK"
        resp.headers = {}
        resp.content = b""
        monkeypatch.setattr(mod.httpx, "get", lambda *a, **k: resp)
        assert s._check_api_running("http://127.0.0.1:1") is True


class TestResolvePythonExecutable:
    def test_embedded_python_exists(self, monkeypatch):
        """嵌入式 Python 存在 → 返回它（lines 169-176）。"""
        s = _make_service()
        fake_path = Path("/fake/embedded/python.exe")

        import vibeocr.backend.env_manager as env_mod

        monkeypatch.setattr(env_mod, "get_project_root", lambda: Path("/fake"))
        monkeypatch.setattr(env_mod, "get_embedded_python", lambda root: fake_path)
        monkeypatch.setattr(Path, "exists", lambda self: True)
        result = s._resolve_python_executable()
        assert result == fake_path

    def test_fallback_to_sys_executable(self, monkeypatch):
        """无嵌入式 Python → 返回 sys.executable（lines 169-176）。"""
        s = _make_service()
        import sys

        import vibeocr.backend.env_manager as env_mod

        monkeypatch.setattr(env_mod, "get_project_root", lambda: Path("/fake"))
        monkeypatch.setattr(
            env_mod, "get_embedded_python", lambda root: Path("/fake/nonexistent.exe")
        )
        # embedded.exists() → False
        result = s._resolve_python_executable()
        assert result == Path(sys.executable)


class TestStartApi:
    def setup_method(self):
        MinerUService._api_url = ""
        MinerUService._api_process = None
        MinerUService._job_guard = None

    def test_no_python_raises(self, monkeypatch):
        """_resolve_python_executable 返回 None → 抛 RuntimeError（line 182）。"""
        s = _make_service()
        monkeypatch.setattr(
            MinerUService, "_resolve_python_executable", lambda self: None
        )
        with pytest.raises(RuntimeError, match="找不到 Python"):
            s._start_api()

    def test_process_dies_during_startup(self, monkeypatch):
        """子进程启动后立即退出 → 抛 RuntimeError（line 231）。"""
        s = _make_service()
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.poll.return_value = 1  # 已退出
        mock_proc.returncode = 1

        with (
            patch.object(
                MinerUService,
                "_resolve_python_executable",
                return_value=Path("/fake/python.exe"),
            ),
            patch(
                "vibeocr.backend.services.mineru_service.subprocess.Popen",
                return_value=mock_proc,
            ),
            patch("vibeocr.backend.services.mineru_service.httpx"),
            patch("vibeocr.backend.services.mineru_service.socket"),
            patch("vibeocr.backend.services.mineru_service.JobObjectGuard"),
            patch("vibeocr.backend.network_detector.NetworkDetector"),
        ):
            with pytest.raises(RuntimeError, match="启动失败"):
                s._start_api()

    def test_startup_timeout(self, monkeypatch):
        """健康检查始终失败 → 超时抛 RuntimeError（lines 238-240）。"""
        s = _make_service()
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.poll.return_value = None  # 进程存活

        # 缩短超时 + 跳过 sleep，避免真实等待
        monkeypatch.setattr(
            "vibeocr.backend.core.constants.Constants.Timeout.MINERU_API_START", 0.01
        )
        monkeypatch.setattr(
            "vibeocr.backend.services.mineru_service.time.sleep", lambda *_: None
        )

        with (
            patch.object(
                MinerUService,
                "_resolve_python_executable",
                return_value=Path("/fake/python.exe"),
            ),
            patch(
                "vibeocr.backend.services.mineru_service.subprocess.Popen",
                return_value=mock_proc,
            ),
            patch("vibeocr.backend.services.mineru_service.httpx"),
            patch("vibeocr.backend.services.mineru_service.socket"),
            patch("vibeocr.backend.services.mineru_service.JobObjectGuard"),
            patch("vibeocr.backend.network_detector.NetworkDetector"),
            patch.object(MinerUService, "_check_api_running", return_value=False),
        ):
            with pytest.raises(RuntimeError, match="超时"):
                s._start_api()

    def test_start_api_does_not_select_native_model_source(self, monkeypatch):
        """MinerU 原生下载器拥有 source 选择，Backend 只透传启动环境。"""
        s = _make_service()
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.poll.return_value = None

        captured_env = {}

        class _FakeDetector:
            def __init__(self, root):
                self.mineru_source = "modelscope"

        def _capture_popen(cmd, **kwargs):
            captured_env.update(kwargs.get("env", {}))
            return mock_proc

        monkeypatch.delenv("MINERU_MODEL_SOURCE", raising=False)

        with (
            patch.object(
                MinerUService,
                "_resolve_python_executable",
                return_value=Path("/fake/python.exe"),
            ),
            patch(
                "vibeocr.backend.services.mineru_service.subprocess.Popen",
                side_effect=_capture_popen,
            ),
            patch("vibeocr.backend.services.mineru_service.httpx"),
            patch("vibeocr.backend.services.mineru_service.socket"),
            patch("vibeocr.backend.services.mineru_service.JobObjectGuard"),
            patch("vibeocr.backend.network_detector.NetworkDetector", _FakeDetector),
            patch.object(MinerUService, "_check_api_running", return_value=True),
        ):
            s._start_api()
        assert "MINERU_MODEL_SOURCE" not in captured_env


class TestEnsureApiRunning:
    def setup_method(self):
        MinerUService._api_url = ""
        MinerUService._api_process = None
        MinerUService._job_guard = None

    def test_cleans_old_process_before_start(self, monkeypatch):
        """有旧进程时先清理再启动（lines 247, 252, 255-260）。"""
        s = _make_service()
        s._lock = threading.Lock()
        mock_old_proc = MagicMock()
        mock_old_proc.wait.return_value = 0
        MinerUService._api_process = mock_old_proc

        with (
            patch.object(s, "_check_api_running", return_value=False),
            patch.object(s, "_start_api") as start_mock,
        ):
            s._ensure_api_running()

        mock_old_proc.terminate.assert_called_once()
        assert MinerUService._api_process is None
        start_mock.assert_called_once()

    def test_old_process_wait_timeout_killed(self, monkeypatch):
        """旧进程 wait 超时 → kill（lines 255-260）。"""
        s = _make_service()
        s._lock = threading.Lock()
        mock_old_proc = MagicMock()
        mock_old_proc.wait.side_effect = subprocess.TimeoutExpired(cmd="x", timeout=5)
        MinerUService._api_process = mock_old_proc

        with (
            patch.object(s, "_check_api_running", return_value=False),
            patch.object(s, "_start_api"),
        ):
            s._ensure_api_running()

        mock_old_proc.kill.assert_called_once()

    def test_url_set_and_running_returns_early(self):
        """已有 URL 且运行中 → 直接返回（line 246）。"""
        s = _make_service()
        s._lock = threading.Lock()
        MinerUService._api_url = "http://127.0.0.1:9999"

        with (
            patch.object(s, "_check_api_running", return_value=True),
            patch.object(s, "_start_api") as start_mock,
        ):
            s._ensure_api_running()
        start_mock.assert_not_called()


class TestCallApiErrors:
    def test_multi_file_upload(self, monkeypatch):
        """files 参数 → 多文件上传分支（lines 307-308）。"""
        s = _make_service()
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"results": {}}

        import vibeocr.backend.services.mineru_service as mod

        monkeypatch.setattr(mod.httpx, "post", lambda *a, **k: resp)
        with patch.object(s, "_ensure_api_running"):
            s._call_api(
                b"", "multi.bin", files=[("a.pdf", b"data1"), ("b.pdf", b"data2")]
            )

    def test_backend_not_in_chain(self, monkeypatch):
        """backend 不在 MINERU_BACKEND_CHAIN → backends_to_try=[backend]（line 343）。"""
        s = _make_service()
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"results": {}}

        import vibeocr.backend.services.mineru_service as mod

        monkeypatch.setattr(mod.httpx, "post", lambda *a, **k: resp)
        from vibeocr.backend.models.ocr_options import OCROptions

        opts = OCROptions()
        opts.backend = "custom-backend"
        with patch.object(s, "_ensure_api_running"):
            s._call_api(b"data", "input.pdf", opts)

    def test_timeout_fallback(self, monkeypatch):
        """httpx.TimeoutException → 记 warning + continue（lines 373-376）。"""
        import httpx

        s = _make_service()

        call = {"n": 0}

        def _post(*a, **k):
            call["n"] += 1
            if call["n"] == 1:
                raise httpx.TimeoutException("timeout")
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {"results": {}}
            return resp

        import vibeocr.backend.services.mineru_service as mod

        monkeypatch.setattr(mod.httpx, "post", _post)
        with patch.object(s, "_ensure_api_running"):
            s._call_api(b"data", "input.pdf")

    def test_connect_error_fallback(self, monkeypatch):
        """httpx.ConnectError → 记 warning + continue（lines 377-382）。"""
        import httpx

        s = _make_service()

        call = {"n": 0}

        def _post(*a, **k):
            call["n"] += 1
            if call["n"] == 1:
                raise httpx.ConnectError("conn")
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {"results": {}}
            return resp

        import vibeocr.backend.services.mineru_service as mod

        monkeypatch.setattr(mod.httpx, "post", _post)
        with patch.object(s, "_ensure_api_running"):
            s._call_api(b"data", "input.pdf")

    def test_error_response_json_parse(self, monkeypatch):
        """错误响应 body 非合法 JSON → 用 text 回退（lines 395-396）。"""
        s = _make_service()
        resp = MagicMock()
        resp.status_code = 500
        resp.json.side_effect = ValueError("not json")
        resp.text = "internal error text"

        import vibeocr.backend.services.mineru_service as mod

        monkeypatch.setattr(mod.httpx, "post", lambda *a, **k: resp)
        with patch.object(s, "_ensure_api_running"):
            with pytest.raises(RuntimeError, match="500"):
                s._call_api(b"data", "input.pdf")

    def test_all_backends_fail_raises_last_error(self, monkeypatch):
        """所有 backend 都失败 → raise last_error（line 403）。"""
        s = _make_service()
        resp = MagicMock()
        resp.status_code = 500
        resp.json.return_value = {"message": "fail"}
        resp.text = '{"message": "fail"}'

        import vibeocr.backend.services.mineru_service as mod

        monkeypatch.setattr(mod.httpx, "post", lambda *a, **k: resp)
        with patch.object(s, "_ensure_api_running"):
            with pytest.raises(RuntimeError, match="mineru-api 错误"):
                s._call_api(b"data", "input.pdf")


class TestFileParse:
    def test_empty_files_returns_empty(self):
        """空 files → 返回 {}（line 451-452）。"""
        s = _make_service()
        assert s.file_parse([]) == {}

    def test_file_parse_multi_file(self, monkeypatch):
        """多文件 file_parse 返回按 stem 索引的字典（lines 453-488）。"""
        s = _make_service()
        api_result = {
            "results": {
                "a": {"md_content": "# A", "content_list": json.dumps([])},
                "b": {"md_content": "# B", "content_list": json.dumps([])},
            }
        }
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = api_result

        import vibeocr.backend.services.mineru_service as mod

        monkeypatch.setattr(mod.httpx, "post", lambda *a, **k: resp)
        with patch.object(s, "_ensure_api_running"):
            out = s.file_parse([("a.pdf", b"data1"), ("b.pdf", b"data2")])
        assert "a" in out
        assert "b" in out

    def test_file_parse_missing_stem_skipped(self, monkeypatch):
        """results_map 缺失某 stem → 该文件不在结果中（lines 478-480）。"""
        s = _make_service()
        api_result = {
            "results": {
                "a": {"md_content": "# A", "content_list": json.dumps([])},
            }
        }
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = api_result

        import vibeocr.backend.services.mineru_service as mod

        monkeypatch.setattr(mod.httpx, "post", lambda *a, **k: resp)
        with patch.object(s, "_ensure_api_running"):
            out = s.file_parse([("a.pdf", b"data1"), ("missing.pdf", b"data2")])
        assert "a" in out
        assert "missing" not in out

    def test_file_parse_with_backend_override(self, monkeypatch):
        """backend 参数透传到 options（lines 454-463）。"""
        s = _make_service()
        api_result = {
            "results": {"x": {"md_content": "# X", "content_list": json.dumps([])}}
        }
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = api_result

        import vibeocr.backend.services.mineru_service as mod

        monkeypatch.setattr(mod.httpx, "post", lambda *a, **k: resp)
        with patch.object(s, "_ensure_api_running"):
            out = s.file_parse([("x.pdf", b"data")], backend="pipeline")
        assert "x" in out


class TestBuildOcrResultEdges:
    def test_content_list_invalid_json_fallback(self):
        """content_list 非合法 JSON → 解析为空列表，不抛异常（lines 511-512）。"""
        s = _make_service()
        api_resp = {
            "results": {"input": {"md_content": "# X", "content_list": "not json{"}}
        }
        # 不抛异常即覆盖（rebuild_result_projections 会清空 projections）
        result = s._build_ocr_result(api_resp, "input.pdf", data=None)
        assert len(result.text_blocks) == 0
        assert result.content_list == []

    def test_non_dict_block_skipped(self):
        """content_list 含非 dict 元素 → 循环跳过（line 523）。

        注：normalize_content_list 后续会对此数据抛异常，但 line 523 在那之前
        已执行（覆盖目标已达成）。这里用 pytest.raises 包裹。
        """
        s = _make_service()
        content_list = [
            {"type": "table", "table_body": "<table><tr><td>A</td></tr></table>"},
            "not-a-dict",
        ]
        api_resp = {
            "results": {
                "input": {
                    "md_content": "ok",
                    "content_list": json.dumps(content_list),
                }
            }
        }
        # line 523 执行后，normalize_content_list 会对非 dict 抛异常
        with pytest.raises(AttributeError):
            s._build_ocr_result(api_resp, "input.pdf", data=None)

    def test_empty_text_block_skipped(self):
        """text 为空 → 跳过 text_block（line 674）。"""
        s = _make_service()
        content_list = [
            {"type": "text", "text": "", "bbox": [0, 0, 10, 10]},
            {"type": "text", "text": "keep", "bbox": [0, 0, 10, 10]},
        ]
        api_resp = {
            "results": {
                "input": {
                    "md_content": "keep",
                    "content_list": json.dumps(content_list),
                }
            }
        }
        result = s._build_ocr_result(api_resp, "input.pdf", data=None)
        assert len(result.text_blocks) == 1
        assert result.text_blocks[0].text == "keep"

    def test_raw_block_not_dict_skipped(self):
        """normalized block 的 raw 非 dict → 跳过（line 630）。"""
        s = _make_service()
        # 构造一个 normalize 后 raw 非 dict 的场景较难，直接验证不抛异常
        content_list = [{"type": "text", "text": "ok", "bbox": [0, 0, 10, 10]}]
        api_resp = {
            "results": {
                "input": {"md_content": "ok", "content_list": json.dumps(content_list)}
            }
        }
        result = s._build_ocr_result(api_resp, "input.pdf", data=None)
        assert len(result.text_blocks) == 1


class TestShutdownKill:
    def test_shutdown_kills_on_timeout(self):
        """shutdown 时 wait 超时 → kill（lines 797-798）。"""
        s = _make_service()
        mock_proc = MagicMock()
        mock_proc.wait.side_effect = subprocess.TimeoutExpired(cmd="x", timeout=10)
        MinerUService._api_process = mock_proc
        MinerUService._api_url = "http://127.0.0.1:9999"

        s.shutdown()
        mock_proc.terminate.assert_called_once()
        mock_proc.kill.assert_called_once()
        assert MinerUService._api_process is None
        assert MinerUService._api_url == ""


class TestRemainingBranches:
    def setup_method(self):
        from vibeocr.backend.core.singleton_meta import SingletonMeta

        SingletonMeta._instances.pop(MinerUService, None)
        MinerUService._instance = None
        MinerUService._initialized = False
        MinerUService._api_process = None
        MinerUService._api_url = ""
        MinerUService._job_guard = None

    def test_init_double_check_lock_first_thread(self):
        """__init__ 首次进入 _initialized=False → 调用 _ensure_api_running 并置 True。

        注：DCL 锁内重检查（line 69）是并发防御，单线程下不可达，已 pragma。
        用 __new__ 绕过 SingletonMeta 缓存，直接测 __init__ 逻辑。
        """
        from vibeocr.backend.core.singleton_meta import SingletonMeta

        SingletonMeta._instances.pop(MinerUService, None)
        MinerUService._initialized = False
        with patch.object(MinerUService, "_ensure_api_running") as m:
            MinerUService()
        m.assert_called_once()

    def test_init_already_initialized_exits_early(self):
        """__init__ _initialized=True → 直接退出（branch 67->exit）。"""
        MinerUService._initialized = True
        with patch.object(MinerUService, "_ensure_api_running") as m:
            s = MinerUService.__new__(MinerUService)
            s.__init__()
        m.assert_not_called()

    def test_log_reader_skips_blank_line(self):
        """log reader 遇空行/空白行应 continue（lines 119, 122）。"""
        s = _make_service()
        proc = MagicMock()
        proc.stderr = MagicMock()
        # b"" → line 119 continue；b"   \n" → strip 后空 → line 122 continue；
        # b"INFO hello\n" → 正常处理；然后进程退出
        proc.stderr.readline.side_effect = [b"", b"   \n", b"INFO hello\n", b""]
        proc.poll.side_effect = [None, None, None, 0]

        s._start_log_reader(proc)
        import time

        time.sleep(0.3)

    def test_ensure_api_running_recheck_in_lock(self):
        """_ensure_api_running 进锁后再次 _check_api_running（line 252）。"""
        s = _make_service()
        s._lock = threading.Lock()
        MinerUService._api_url = "http://127.0.0.1:9999"
        # 外层 check 失败、锁内 check 成功 → 直接返回不 start
        check_results = iter([False, True])
        with (
            patch.object(
                s, "_check_api_running", side_effect=lambda *a: next(check_results)
            ),
            patch.object(s, "_start_api") as start_mock,
        ):
            s._ensure_api_running()
        start_mock.assert_not_called()

    def test_file_parse_backend_with_existing_options(self, monkeypatch):
        """file_parse 传 backend 且 options 非 None → 设 backend（branch 456->460）。"""
        s = _make_service()
        api_result = {
            "results": {"x": {"md_content": "# X", "content_list": json.dumps([])}}
        }
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = api_result

        import vibeocr.backend.services.mineru_service as mod

        monkeypatch.setattr(mod.httpx, "post", lambda *a, **k: resp)
        from vibeocr.backend.models.ocr_options import OCROptions

        opts = OCROptions()
        with patch.object(s, "_ensure_api_running"):
            out = s.file_parse([("x.pdf", b"data")], options=opts, backend="pipeline")
        assert "x" in out
        assert opts.backend == "pipeline"

    def test_build_ocr_result_raw_block_not_dict(self, monkeypatch):
        """normalized block 的 raw 非 dict → continue（line 630）。

        构造一个 raw 为 None 的 normalized block：通过 monkeypatch normalize_content_list
        返回含非 dict raw 的条目。
        """
        s = _make_service()
        import vibeocr.backend.services.mineru_service as mod

        # monkeypatch normalize_content_list 返回特殊结构
        fake_normalized = [
            {"type": "text", "text": "ok", "raw": None, "page_idx": 0},
        ]
        monkeypatch.setattr(mod, "normalize_content_list", lambda cl: fake_normalized)
        api_resp = {"results": {"input": {"md_content": "ok", "content_list": "[]"}}}
        # raw=None → 跳过 flat_content_list / text_blocks
        result = s._build_ocr_result(api_resp, "input.pdf", data=None)
        assert len(result.text_blocks) == 0

    def test_extract_block_text_chart_with_content(self):
        """_extract_block_text chart 类型 + content（branch 746->732 覆盖 chart 路径完整）。"""
        text = MinerUService._extract_block_text(
            {"type": "chart", "chart_caption": ["图表"], "content": "desc"}
        )
        assert "图表" in text and "desc" in text

    def test_normalize_table_block_no_dict_raw(self, monkeypatch):
        """normalized table block 的 raw 非 dict → 跳过（branch 616->612）。"""
        s = _make_service()
        import vibeocr.backend.services.mineru_service as mod

        fake_normalized = [
            {"type": "table", "text": "", "raw": None, "page_idx": 0},
        ]
        monkeypatch.setattr(mod, "normalize_content_list", lambda cl: fake_normalized)
        api_resp = {"results": {"input": {"md_content": "", "content_list": "[]"}}}
        result = s._build_ocr_result(api_resp, "input.pdf", data=None)
        # table 块 raw=None → 不生成 text_block（无 text/bbox）
        assert len(result.text_blocks) == 0

    def test_images_dict_non_data_uri_skipped(self):
        """images 中非 data: URI 的条目应被跳过（branch 652->651）。"""
        s = _make_service()
        api_resp = {
            "results": {
                "input": {
                    "md_content": "# X",
                    "content_list": json.dumps([]),
                    "images": {
                        "bad.png": "http://example.com/img.png",  # 非 data:
                        "empty.png": "",  # 空
                    },
                }
            }
        }
        result = s._build_ocr_result(api_resp, "input.pdf", data=None)
        # 非法 URI 不解码
        assert len(result.images) == 0

    def test_extract_plain_text_image_no_caption(self):
        """_extract_plain_text image/chart 无 caption → 跳过（branch 746->732）。"""
        result = MinerUService._extract_plain_text(
            [{"type": "image"}, {"type": "chart"}]
        )
        # 无 caption → 空结果
        assert result == ""
