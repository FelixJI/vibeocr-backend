"""PDF 后端子进程的客户端 + 进程托管。

职责:
- 启动/停止 pdf_backend_process 子进程(端口探测 + JobObjectGuard 孤儿清理)
- 等待 /health 就绪
- 暴露全部 PDF 操作为 httpx 调用,返回 schema 对象
- 崩溃检测 + 自动重启(透明重连)
- 后台线程读子进程 stdout 转发到项目日志

主进程通过 PdfBackendClient 单例访问 PDF 后端,完全不碰 fitz。
"""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import sys
import threading
import time
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from collections.abc import Iterator

from vibeocr.backend.ipc.schemas import (
    AddTextLayerRequest,
    BatchAddTextLayerPage,
    BatchAddTextLayerRequest,
    DeletePagesRequest,
    DetectTextLayersRequest,
    DetectTextLayersResponse,
    HealthResponse,
    InsertBlankRequest,
    InsertFromRequest,
    MovePageRequest,
    MutateResponse,
    OpenRequest,
    OpenResponse,
    PageListRequest,
    PdfDocumentMirror,
    ProgressEvent,
    RenderPreviewRequest,
    RenderThumbnailRequest,
    ReorderRequest,
    RewriteTextLayerRequest,
    RotateRequest,
    SaveRequest,
    SaveResponse,
    UpdateBlockTextRequest,
)
from vibeocr.backend.utils.job_object import JobObjectGuard
from vibeocr.backend.utils.subprocess_log import SubprocessLogForwarder
from vibeocr.runtime_contracts.utils.http_log import (
    guess_response_size,
    log_http_response,
)

logger = logging.getLogger(__name__)

# 启动后等待就绪的超时(秒)
_BACKEND_START_TIMEOUT = 30.0
_HTTP_TIMEOUT = httpx.Timeout(60.0, connect=5.0)
# 长操作(保存/摆正/删除文字层)用更长超时
_HTTP_LONG_TIMEOUT = httpx.Timeout(600.0, connect=5.0)


class PdfBackendError(RuntimeError):
    """PDF 后端调用失败。"""


class PdfBackendClient:
    """PDF 后端单例客户端。延迟启动子进程,主进程首次 open 时拉起。"""

    _instance: PdfBackendClient | None = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._process: subprocess.Popen | None = None
        self._base_url: str = ""
        self._job_guard: JobObjectGuard | None = None
        self._lock = threading.RLock()
        self._started = False
        # httpx 同步 Client 非线程安全:按线程标识各持独立 client。
        # 缩略图并发渲染时多个 worker 线程并发调用,各自取本线程的 client。
        self._http_clients: dict[int, httpx.Client] = {}
        self._log_thread: threading.Thread | None = None

    @classmethod
    def instance(cls) -> PdfBackendClient:
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ---- 进程生命周期 ---------------------------------------------------

    def _resolve_python_exe(self) -> str:
        """选择子进程 Python 解释器(对齐 MinerU 范式)。"""
        from vibeocr.backend.env_manager import get_embedded_python, get_project_root

        project_root = get_project_root()
        embedded = get_embedded_python(project_root)
        if embedded.exists():
            return str(embedded)
        return sys.executable

    def _get_backend_env(self) -> dict[str, str]:
        """构造 PDF 后端子进程环境变量。

        Runtime Installer 已把精确 Backend wheel 安装进内容寻址 Runtime。
        子进程必须从该解释器的 site-packages 导入，不能继承开发工作区路径、
        editable 环境或冻结前端的临时解包目录。
        """
        env = os.environ.copy()
        for name in (
            "PYTHONPATH",
            "PYTHONHOME",
            "VIRTUAL_ENV",
            "VIBEOCR_REPOSITORY_ROOT",
        ):
            env.pop(name, None)
        return env

    def _find_free_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    def _start_log_reader(self, process: subprocess.Popen) -> None:
        """后台线程读子进程 stdout,转发到项目日志。

        用统一的 SubprocessLogForwarder（与 OCR worker、MinerU 共用同一套逻辑）：
        结构化日志行按原始级别转发，uvicorn/库的裸 print 按行数折叠，
        避免泄漏用户文档内容。
        """
        forwarder = SubprocessLogForwarder(
            logger_name="vibeocr.subprocess.pdf_backend",
            source_label="[PDF Backend]",
        )

        def _read() -> None:
            try:
                assert process.stdout is not None
                for raw in process.stdout:
                    text = raw.decode("utf-8", errors="replace")
                    if not text:
                        continue
                    # uvicorn 有时无换行拼接多行，按 datetime 模式切分
                    for line in forwarder.split_mixed_lines(text):
                        forwarder.forward(line)
                # 进程退出后 flush 累积的裸 print 概括
                forwarder.flush()
            except Exception:
                pass

        t = threading.Thread(target=_read, daemon=True, name="PdfBackendLogReader")
        t.start()
        self._log_thread = t

    def start(self) -> None:
        """启动 PDF 后端子进程并等待就绪。线程安全,幂等。"""
        with self._lock:
            if self._started and self._is_alive():
                return
            self._stop_locked()

            python_exe = self._resolve_python_exe()
            port = self._find_free_port()
            self._base_url = f"http://127.0.0.1:{port}"

            cmd = [
                python_exe,
                "-m",
                "vibeocr.backend.services.pdf_backend_process",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--log-level",
                "info",
            ]
            logger.info("[pdf-backend] 启动子进程 @ %s", self._base_url)
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # 合并到 stdout 统一读
                text=False,
                env=self._get_backend_env(),
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            # 绑定 Job Object:主进程崩溃时内核连带终止后端
            self._job_guard = JobObjectGuard(name="vibeocr_pdf_backend")
            self._job_guard.assign_from_popen(self._process)
            self._start_log_reader(self._process)

            # 等待就绪
            self._wait_ready()
            # http client 改为按线程懒建(见 _ensure_started),此处不再预建
            self._started = True

    def _is_alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _wait_ready(self) -> None:
        import time

        deadline = time.monotonic() + _BACKEND_START_TIMEOUT
        last_err: Exception | None = None
        while time.monotonic() < deadline:
            if self._process is not None and self._process.poll() is not None:
                # 子进程已退出：排空 stdout 提取真实错误（traceback），否则只剩退出码无法定位
                tail = self._drain_stdout_tail()
                msg = f"PDF 后端启动失败,退出码 {self._process.returncode}"
                if tail:
                    msg += f"\n子进程输出末尾:\n{tail}"
                raise PdfBackendError(msg)
            started = time.perf_counter()
            try:
                resp = httpx.get(f"{self._base_url}/health", timeout=2.0)
                self._log_http_response(
                    "GET",
                    "/health",
                    resp,
                    elapsed_ms=(time.perf_counter() - started) * 1000,
                )
                if resp.status_code == 200:
                    logger.info("[pdf-backend] 就绪")
                    return
            except Exception as e:
                last_err = e
                logger.warning(
                    "[pdf-backend] health check failed after %.1f ms: %s",
                    (time.perf_counter() - started) * 1000,
                    e,
                )
            time.sleep(0.3)
        raise PdfBackendError(f"PDF 后端 {self._base_url} 启动超时({last_err})")

    def _drain_stdout_tail(self, max_lines: int = 30) -> str:
        """读取子进程 stdout 末尾若干行用于错误诊断。

        日志转发线程已在并行读取 stdout，但进程退出后 pipe 中可能仍有未消费
        缓冲。这里非阻塞地读出末尾内容（traceback），附加到异常消息中。
        """
        if self._process is None or self._process.stdout is None:
            return ""
        try:
            lines: list[str] = []
            # 进程已退出，pipe 写端关闭，readline 会立即返回 EOF。
            while len(lines) < max_lines:
                raw = self._process.stdout.readline()
                if not raw:
                    break
                text = raw.decode("utf-8", errors="replace").rstrip()
                if text:
                    lines.append(text)
            # 只保留末尾 max_lines 行（traceback 通常在末尾）
            return "\n".join(lines[-max_lines:]) if lines else ""
        except Exception:
            return ""

    def _stop_locked(self) -> None:
        if self._job_guard is not None:
            try:
                self._job_guard.close()
            except Exception:
                pass
            self._job_guard = None
        if self._process is not None:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
            self._process = None
        # 关闭所有线程的 http client
        for c in self._http_clients.values():
            try:
                c.close()
            except Exception:
                pass
        self._http_clients.clear()
        self._started = False

    def stop(self) -> None:
        with self._lock:
            self._stop_locked()

    def _ensure_started(self) -> httpx.Client:
        """确保后端已启动,返回当前线程专属的 http client。崩溃则重启。

        httpx 同步 Client 非线程安全,故每个调用线程持独立 client(按 thread ident)。
        首次调用或后端崩溃重启后,懒建新 client。
        """
        tid = threading.get_ident()
        client = self._http_clients.get(tid)
        if client is not None and self._started and self._is_alive():
            return client
        # 需要启动/重启后端:加锁避免多线程并发首次启动
        if not self._started or not self._is_alive():
            self.start()
        client = httpx.Client(base_url=self._base_url, timeout=_HTTP_TIMEOUT)
        self._http_clients[tid] = client
        return client

    def _log_http_response(
        self,
        method: str,
        path: str,
        resp: httpx.Response,
        *,
        request_payload: object | None = None,
        elapsed_ms: float | None = None,
        response_bytes: int | None = None,
        stream: bool = False,
    ) -> None:
        log_http_response(
            logger=logger,
            method=method,
            url=f"{self._base_url}{path}",
            status_code=resp.status_code,
            reason=resp.reason_phrase,
            elapsed_ms=elapsed_ms,
            request_bytes=self._estimate_request_bytes(request_payload),
            response_bytes=(
                response_bytes
                if response_bytes is not None
                else self._estimate_response_bytes(resp)
            ),
            stream=stream,
        )

    @staticmethod
    def _estimate_request_bytes(payload: object | None) -> int | None:
        if payload is None:
            return None
        if isinstance(payload, (bytes, bytearray)):
            return len(payload)
        if isinstance(payload, str):
            return len(payload.encode("utf-8"))
        try:
            return len(str(payload).encode("utf-8"))
        except Exception:
            return None

    @staticmethod
    def _estimate_response_bytes(
        resp: httpx.Response,
        *,
        include_content: bool = True,
    ) -> int | None:
        headers_obj = getattr(resp, "headers", None)
        try:
            headers = dict(headers_obj) if headers_obj is not None else None
        except Exception:
            headers = None
        content_obj = getattr(resp, "content", None) if include_content else None
        content = content_obj if isinstance(content_obj, (bytes, str)) else None
        return guess_response_size(headers, content)

    # ---- HTTP 调用辅助 ---------------------------------------------------

    def _post(
        self, path: str, payload: object | None = None, *, timeout=None
    ) -> httpx.Response:
        client = self._ensure_started()
        started = time.perf_counter()
        try:
            resp = (
                client.post(path, json=payload, timeout=timeout)
                if payload is not None
                else client.post(path, timeout=timeout)
            )
        except httpx.HTTPError as e:
            raise PdfBackendError(f"后端调用失败 {path}: {e}") from e
        self._log_http_response(
            "POST",
            path,
            resp,
            request_payload=payload,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )
        if resp.status_code >= 400:
            detail = resp.text
            try:
                detail = resp.json().get("detail", detail)
            except Exception:
                pass
            raise PdfBackendError(f"后端错误 {path} ({resp.status_code}): {detail}")
        return resp

    def _get(self, path: str, *, timeout=None) -> httpx.Response:
        client = self._ensure_started()
        started = time.perf_counter()
        try:
            resp = client.get(path, timeout=timeout)
        except httpx.HTTPError as e:
            raise PdfBackendError(f"后端调用失败 {path}: {e}") from e
        self._log_http_response(
            "GET",
            path,
            resp,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )
        if resp.status_code >= 400:
            raise PdfBackendError(f"后端错误 {path} ({resp.status_code}): {resp.text}")
        return resp

    @staticmethod
    def _parse(resp: httpx.Response, model_cls):
        return model_cls.model_validate_json(resp.content)

    # ---- 业务 API -------------------------------------------------------

    def health(self) -> HealthResponse:
        return self._parse(self._get("/health"), HealthResponse)

    def open_session(self, path: str) -> OpenResponse:
        return self._parse(
            self._post("/session/open", OpenRequest(path=path).model_dump()),
            OpenResponse,
        )

    def close_session(self, sid: str) -> None:
        self._post(f"/session/{sid}/close")

    def get_model(self, sid: str) -> PdfDocumentMirror:
        return self._parse(self._post(f"/session/{sid}/model"), PdfDocumentMirror)

    def load_stream(self, sid: str) -> Iterator[ProgressEvent]:
        """打开后流式逐页文字层检测:每页 yield 一个 ProgressEvent。

        每个事件的 page_payload 是该页的 PdfPageInfoMirror dict。
        末行 message="done" 表示完成。
        """
        client = self._ensure_started()
        path = f"/session/{sid}/load"
        started = time.perf_counter()
        try:
            with client.stream(
                "POST",
                path,
                timeout=_HTTP_LONG_TIMEOUT,
            ) as resp:
                response_bytes = 0
                try:
                    if resp.status_code >= 400:
                        raise PdfBackendError(f"load 失败 ({resp.status_code})")
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        response_bytes += len(line.encode("utf-8")) + 1
                        yield ProgressEvent.model_validate_json(line)
                finally:
                    self._log_http_response(
                        "POST",
                        path,
                        resp,
                        elapsed_ms=(time.perf_counter() - started) * 1000,
                        response_bytes=(
                            response_bytes
                            or self._estimate_response_bytes(
                                resp,
                                include_content=False,
                            )
                            or 0
                        ),
                        stream=True,
                    )
        except httpx.HTTPError as e:
            logger.warning(
                "[pdf-backend] POST %s stream failed after %.1f ms: %s",
                path,
                (time.perf_counter() - started) * 1000,
                e,
            )
            raise PdfBackendError(f"load 流式调用失败: {e}") from e

    def render_thumbnail(self, sid: str, page: int, size: int = 160) -> bytes:
        """渲染缩略图,返回 PNG 字节。

        使用 _HTTP_TIMEOUT（60s 读 + 5s 连接）而非无限超时：缩略图是快速操作，
        若后端卡死应在 60s 内超时返回异常，而非无限阻塞导致 cancel 后
        ThreadPoolExecutor 无法收尾。
        """
        resp = self._post(
            f"/session/{sid}/render_thumbnail",
            RenderThumbnailRequest(page=page, size=size).model_dump(),
            timeout=_HTTP_TIMEOUT,
        )
        return resp.content

    def render_preview(self, sid: str, page: int, dpi: int = 150) -> bytes:
        """渲染预览页,返回 PNG 字节。"""
        resp = self._post(
            f"/session/{sid}/render_preview",
            RenderPreviewRequest(page=page, dpi=dpi).model_dump(),
            timeout=_HTTP_LONG_TIMEOUT,
        )
        return resp.content

    def detect_text_layers(self, sid: str, page: int) -> DetectTextLayersResponse:
        return self._parse(
            self._post(
                f"/session/{sid}/detect_text_layers",
                DetectTextLayersRequest(page=page).model_dump(),
            ),
            DetectTextLayersResponse,
        )

    def rotate(self, sid: str, pages: list[int], angle: int) -> MutateResponse:
        return self._parse(
            self._post(
                f"/session/{sid}/rotate",
                RotateRequest(pages=pages, angle=angle).model_dump(),
            ),
            MutateResponse,
        )

    def delete_pages(self, sid: str, pages: list[int]) -> MutateResponse:
        return self._parse(
            self._post(
                f"/session/{sid}/delete_pages",
                DeletePagesRequest(pages=pages).model_dump(),
            ),
            MutateResponse,
        )

    def insert_blank(
        self, sid: str, after_index: int, width: float = 612.0, height: float = 792.0
    ) -> MutateResponse:
        return self._parse(
            self._post(
                f"/session/{sid}/insert_blank",
                InsertBlankRequest(
                    after_index=after_index, width=width, height=height
                ).model_dump(),
            ),
            MutateResponse,
        )

    def insert_from(
        self, sid: str, source_path: str, after_index: int
    ) -> MutateResponse:
        return self._parse(
            self._post(
                f"/session/{sid}/insert_from",
                InsertFromRequest(
                    source_path=source_path, after_index=after_index
                ).model_dump(),
            ),
            MutateResponse,
        )

    def move_page(self, sid: str, from_index: int, to_index: int) -> MutateResponse:
        return self._parse(
            self._post(
                f"/session/{sid}/move_page",
                MovePageRequest(from_index=from_index, to_index=to_index).model_dump(),
            ),
            MutateResponse,
        )

    def reorder(self, sid: str, new_order: list[int]) -> MutateResponse:
        return self._parse(
            self._post(
                f"/session/{sid}/reorder",
                ReorderRequest(new_order=new_order).model_dump(),
            ),
            MutateResponse,
        )

    def add_text_layer(
        self,
        sid: str,
        page: int,
        ocr_result: dict,
        pdf_settings: dict | None = None,
        overwrite: bool = False,
    ) -> MutateResponse:
        return self._parse(
            self._post(
                f"/session/{sid}/add_text_layer",
                AddTextLayerRequest(
                    page=page,
                    ocr_result=ocr_result,
                    pdf_settings=pdf_settings,
                    overwrite=overwrite,
                ).model_dump(),
            ),
            MutateResponse,
        )

    def add_text_layer_batch(
        self,
        sid: str,
        pages_data: list[dict],  # [{page, ocr_result}, ...]
        pdf_settings: dict | None = None,
        overwrite: bool = False,
        save: bool = False,
    ) -> MutateResponse:
        """批量写 OCR 文字层，一批页共享单一聚合子集字体。

        供 PdfSessionManager._run_ocr 阶段3 攒一批结果后一次调用，替代逐页
        add_text_layer。后端聚合本批所有页字符解析单一子集字体，避免每页一份。

        save=True 时后端写层后 incremental 落盘，resp.extra["saved"] 标记结果
        （False=回滚，调用方不写 sidecar）。
        """
        pages = [
            BatchAddTextLayerPage(page=p["page"], ocr_result=p["ocr_result"])
            for p in pages_data
        ]
        return self._parse(
            self._post(
                f"/session/{sid}/add_text_layer_batch",
                BatchAddTextLayerRequest(
                    pages=pages,
                    pdf_settings=pdf_settings,
                    overwrite=overwrite,
                    save=save,
                ).model_dump(),
                timeout=_HTTP_LONG_TIMEOUT,
            ),
            MutateResponse,
        )

    def rewrite_text_layer(
        self,
        sid: str,
        page: int,
        text_blocks: list,
        preproc_angle: int = 0,
        pdf_settings: dict | None = None,
    ) -> MutateResponse:
        from vibeocr.backend.ipc.schemas import TextBlockMirror

        blocks = [
            TextBlockMirror(
                text=b.text,
                score=b.score,
                bbox=b.bbox,
                polygon=b.polygon,
                page_idx=b.page_idx,
                is_manually_edited=b.is_manually_edited,
                label=b.label,
                order=b.order,
            )
            for b in text_blocks
        ]
        return self._parse(
            self._post(
                f"/session/{sid}/rewrite_text_layer",
                RewriteTextLayerRequest(
                    page=page,
                    text_blocks=blocks,
                    preproc_angle=preproc_angle,
                    pdf_settings=pdf_settings,
                ).model_dump(),
            ),
            MutateResponse,
        )

    def update_block_text(
        self, sid: str, page: int, block_index: int, new_text: str
    ) -> MutateResponse:
        return self._parse(
            self._post(
                f"/session/{sid}/update_block_text",
                UpdateBlockTextRequest(
                    page=page, block_index=block_index, new_text=new_text
                ).model_dump(),
            ),
            MutateResponse,
        )

    def delete_text_layers_stream(
        self, sid: str, pages: list[int]
    ) -> Iterator[ProgressEvent]:
        """逐页删除文字层,流式返回 ProgressEvent。"""
        client = self._ensure_started()
        path = f"/session/{sid}/delete_text_layers"
        payload = PageListRequest(pages=pages).model_dump()
        started = time.perf_counter()
        try:
            with client.stream(
                "POST",
                path,
                json=payload,
                timeout=_HTTP_LONG_TIMEOUT,
            ) as resp:
                response_bytes = 0
                try:
                    if resp.status_code >= 400:
                        raise PdfBackendError(f"删除文字层失败 ({resp.status_code})")
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        response_bytes += len(line.encode("utf-8")) + 1
                        yield ProgressEvent.model_validate_json(line)
                finally:
                    self._log_http_response(
                        "POST",
                        path,
                        resp,
                        request_payload=payload,
                        elapsed_ms=(time.perf_counter() - started) * 1000,
                        response_bytes=(
                            response_bytes
                            or self._estimate_response_bytes(
                                resp,
                                include_content=False,
                            )
                            or 0
                        ),
                        stream=True,
                    )
        except httpx.HTTPError as e:
            logger.warning(
                "[pdf-backend] POST %s stream failed after %.1f ms: %s",
                path,
                (time.perf_counter() - started) * 1000,
                e,
            )
            raise PdfBackendError(f"删除文字层流式调用失败: {e}") from e

    def save(
        self,
        sid: str,
        path: str | None = None,
        pdf_settings: dict | None = None,
        *,
        rewrite_text_layers: bool = True,
    ) -> SaveResponse:
        return self._parse(
            self._post(
                f"/session/{sid}/save",
                SaveRequest(
                    path=path,
                    pdf_settings=pdf_settings,
                    rewrite_text_layers=rewrite_text_layers,
                ).model_dump(),
                timeout=_HTTP_LONG_TIMEOUT,
            ),
            SaveResponse,
        )

    def cancel(self, sid: str) -> None:
        self._post(f"/session/{sid}/cancel")

    def reset_cancel(self, sid: str) -> None:
        self._post(f"/session/{sid}/reset_cancel")
