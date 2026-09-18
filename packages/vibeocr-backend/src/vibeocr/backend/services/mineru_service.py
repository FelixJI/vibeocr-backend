"""MinerU 文档解析服务

通过 mineru-api FastAPI 服务进行文档解析。
自动管理 mineru-api 进程的生命周期。
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
from vibeocr.backend.core.constants import Constants
from vibeocr.backend.core.singleton_meta import SingletonMeta
from vibeocr.backend.models.ocr_result import (
    DISCARDED_BLOCK_TYPES,
    OCRResult,
    TextBlock,
    normalize_bbox,
    normalize_content_list,
)
from vibeocr.backend.tables.blocks import (
    canonicalize_table_block,
    table_model_from_block,
)
from vibeocr.backend.tables.projections import table_model_to_plain_text
from vibeocr.backend.tables.reducer import rebuild_result_projections
from vibeocr.backend.utils.job_object import JobObjectGuard
from vibeocr.backend.utils.mime_types import mime_to_extension
from vibeocr.runtime_contracts import MineruConfig, MineruOcrMode, MineruTier
from vibeocr.runtime_contracts.contracts.tables import TableProvenanceV1
from vibeocr.runtime_contracts.utils.http_log import (
    guess_response_size,
    log_http_response,
)

from .mineru_api import MineruApiClient, MineruApiError, MineruDocument
from .mineru_config import migrate_legacy_options
from .mineru_result import project_document

if TYPE_CHECKING:
    from vibeocr.backend.models.ocr_options import OCROptions

_logger = logging.getLogger(__name__)


class MinerUService(metaclass=SingletonMeta):
    """MinerU 文档解析服务（单例）

    通过 mineru-api FastAPI 服务进行文档解析。
    自动管理 mineru-api 进程的生命周期。
    """

    _api_process: subprocess.Popen | None = None
    _api_url: str = ""
    _lock = threading.RLock()
    _initialized = False
    _language = "ch"
    _server_tier = "basic"
    _job_guard: JobObjectGuard | None = None

    def __init__(self):
        if not self._initialized:
            with self._lock:
                if (
                    not self._initialized
                ):  # pragma: no cover - DCL inner recheck, only races under concurrency
                    self._ensure_api_running()
                    self._initialized = True

    @classmethod
    def _reset(cls) -> None:
        """重置服务状态（供测试使用）"""
        with cls._lock:
            if cls._api_process is not None:
                cls._api_process.terminate()
                try:
                    cls._api_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    cls._api_process.kill()
                cls._api_process = None
            cls._api_url = ""
            cls._initialized = False

    @staticmethod
    def _parse_api_log_level(text: str) -> int:
        """从 mineru-api 日志行中提取日志级别"""
        import re

        match = re.search(
            r"\b(DEBUG|INFO|WARNING|ERROR|CRITICAL)\b", text, re.IGNORECASE
        )
        if match:
            return getattr(logging, match.group(1).upper(), logging.DEBUG)
        return logging.DEBUG

    def _start_log_reader(self, process: subprocess.Popen) -> None:
        """启动守护线程读取 mineru-api 子进程的 stderr 并转发到项目日志系统。

        注意：mineru-api 是第三方 FastAPI 服务，其日志格式（uvicorn 风格，
        如 ``INFO:     127.0.0.1:... "POST /v1/parse/jobs HTTP/1.1" 200``）不符合
        SubprocessLogForwarder 的结构化正则（YYYY-MM-DD HH:MM:SS [LEVEL] name: msg），
        因此这里不复用 forwarder 的折叠逻辑（会把 uvicorn 的有用 INFO 全折叠掉），
        而是保留原有的"按级别词匹配 + 原文转发"。仅 logger 名统一到
        vibeocr.subprocess.<name> 规范。
        """
        mineru_logger = logging.getLogger("vibeocr.subprocess.mineru_api")
        stderr = process.stderr
        if stderr is None:
            return

        def _read():
            try:
                while process.poll() is None:
                    line = stderr.readline()
                    if not line:
                        continue
                    text = line.decode("utf-8", errors="replace").strip()
                    if not text:
                        continue
                    level = self._parse_api_log_level(text)
                    mineru_logger.log(level, text)
            except Exception:
                pass

        t = threading.Thread(target=_read, daemon=True, name="MinerUApiLogReader")
        t.start()

    def _check_api_running(self, url: str) -> bool:
        """检查 mineru-api 是否运行"""
        request_url = f"{url}/v1/health"
        started = time.perf_counter()
        try:
            resp = httpx.get(request_url, timeout=3, trust_env=False)
            log_http_response(
                logger=_logger,
                method="GET",
                url=request_url,
                status_code=resp.status_code,
                reason=resp.reason_phrase,
                elapsed_ms=(time.perf_counter() - started) * 1000,
                response_bytes=guess_response_size(dict(resp.headers), resp.content),
            )
            return (
                resp.status_code == 200
                and str(resp.json().get("version", "")).split(".")[0] == "4"
            )
        except Exception as exc:
            _logger.warning(
                "[MinerU] GET %s failed after %.1f ms: %s",
                request_url,
                (time.perf_counter() - started) * 1000,
                exc,
            )
            return False

    def _find_free_port(self) -> int:
        """找一个可用端口"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    def _resolve_python_executable(self) -> Path | None:
        """查找可用的 Python 解释器

        查找顺序:
        1. 嵌入式 Python（便携模式）
        2. 当前 Python 解释器（开发模式）
        """
        from vibeocr.backend.env_manager import get_embedded_python, get_project_root

        project_root = get_project_root()
        embedded = get_embedded_python(project_root)
        if embedded.exists():
            return embedded

        return Path(sys.executable)

    def _start_api(self) -> None:
        """启动 mineru-api 进程"""
        python_exe = self._resolve_python_executable()
        if python_exe is None:
            raise RuntimeError("找不到 Python 解释器。请确保已安装 Python 和 MinerU 4")

        port = self._find_free_port()
        url = f"http://127.0.0.1:{port}"

        _logger.info(f"[MinerU] 启动 mineru-api 服务 @ {url}...")

        cmd = [
            str(python_exe),
            "-m",
            "mineru.parser.api_server",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--tier",
            self.__class__._server_tier,
            "--language",
            self.__class__._language,
        ]

        env = os.environ.copy()
        # A separate home leaves MinerU 3 config and model paths untouched.
        from vibeocr.backend.env_manager import get_project_root

        home = Path(
            env.get("MINERU_HOME") or get_project_root() / "data" / "mineru4"
        ).resolve()
        home.mkdir(parents=True, exist_ok=True)
        config_path = Path(env.get("MINERU_CONFIG") or home / "config.yaml").resolve()
        if "MINERU_CONFIG" in env and not config_path.is_file():
            raise ValueError("MINERU_CONFIG must reference an existing config file")
        if not config_path.exists():
            config_path.write_text(
                "model:\n  small_backend: onnx\n  vlm:\n    engine: llama-cpp\n",
                encoding="utf-8",
            )
        env["MINERU_HOME"] = str(home)
        env["MINERU_CONFIG"] = str(config_path)

        self.__class__._api_process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env=env,
            cwd=home,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )

        # 绑定 Windows Job Object：主进程崩溃时内核连带终止 mineru-api
        self.__class__._job_guard = JobObjectGuard(name="vibeocr_mineru_api")
        self.__class__._job_guard.assign_from_popen(self.__class__._api_process)

        # 读取 mineru-api 子进程的 stderr 并转发到项目日志系统
        self._start_log_reader(self.__class__._api_process)

        _logger.debug("[MinerU] 日志输出到项目日志系统")

        # 等待 API 就绪
        for _ in range(int(Constants.Timeout.MINERU_API_START)):
            if self.__class__._api_process.poll() is not None:
                raise RuntimeError(
                    f"mineru-api 启动失败，退出码: {self.__class__._api_process.returncode}"
                )
            if self._check_api_running(url):
                self.__class__._api_url = url
                _logger.info(f"[MinerU] mineru-api 服务已就绪 @ {url}")
                return
            time.sleep(1)

        raise RuntimeError(
            f"mineru-api 启动超时（{int(Constants.Timeout.MINERU_API_START)}秒）"
        )

    def _ensure_api_running(self) -> None:
        """确保 mineru-api 正在运行"""
        if self.__class__._api_url and self._check_api_running(self.__class__._api_url):
            return
        with self._lock:
            if self.__class__._api_url and self._check_api_running(
                self.__class__._api_url
            ):
                return
            # 清理旧进程
            if self.__class__._api_process is not None:
                self.__class__._api_process.terminate()
                try:
                    self.__class__._api_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.__class__._api_process.kill()
                self.__class__._api_process = None
            # 不预探测/预下载模型：mineru-api 不依赖模型即可启动，模型在首次
            # job 解析时由 mineru 自己按需下载（auto_download_and_get_
            # model_root_path）。旧的预探测用 mineru.cli.models_download 当探测
            # 命令、30s 超时，反而会杀掉 mineru 自己正在进行的下载，导致"永远
            # 下不完"。模型下载是 mineru 内部事务，我们只管发请求 + 给足超时。
            self._start_api()

    @staticmethod
    def _config(options: OCROptions | MineruConfig | None) -> MineruConfig:
        if isinstance(options, MineruConfig):
            return options
        values = (
            {}
            if options is None
            else {
                name: getattr(options, name)
                for name in (
                    "backend",
                    "effort",
                    "parse_method",
                    "enable_formula",
                    "enable_table",
                    "lang_list",
                    "start_page_id",
                    "end_page_id",
                )
            }
        )
        return migrate_legacy_options(values)

    def _call_api(
        self,
        data: bytes,
        filename: str,
        options: OCROptions | MineruConfig | None = None,
        *,
        files: list[tuple[str, bytes]] | None = None,
        cancelled: Callable[[], bool] = lambda: False,
    ) -> dict[str, MineruDocument | MineruApiError]:
        config = self._config(options)
        # Language is service-wide. Serialize jobs through shutdown/start and
        # result download; never mutate a live service used by another job.
        with self._lock:
            tier = (
                "standard"
                if config.tier in (MineruTier.STANDARD, MineruTier.ADVANCED)
                else "basic"
            )
            if (
                self.__class__._language != config.language
                or self.__class__._server_tier != tier
            ):
                self.shutdown()
                self.__class__._language = config.language
                self.__class__._server_tier = tier
            self._ensure_api_running()
            return MineruApiClient(
                self.__class__._api_url, timeout=Constants.Timeout.MINERU_HTTP_TOTAL
            ).parse(
                files if files is not None else [(filename, data)],
                config,
                cancelled=cancelled,
            )

    def prepare(self) -> None:
        """Explicit preload verifies actual parsing instead of claiming health is ready."""
        import fitz

        from .mineru_readiness import mark_executed, mark_failed

        with fitz.open() as pdf:
            page = pdf.new_page()
            page.insert_textbox(
                fitz.Rect(72, 150, 520, 500),
                "VibeOCR readiness document. Document parsing preserves readable text.\n"
                "This generated sample verifies the requested parsing tier.",
                fontsize=14,
            )
            data = pdf.tobytes()
        for tier in MineruTier:
            try:
                document = self._call_api(
                    data,
                    "readiness.pdf",
                    MineruConfig(
                        tier=tier,
                        ocr_mode=MineruOcrMode.TXT
                        if tier == MineruTier.FLASH
                        else MineruOcrMode.OCR,
                    ),
                )["readiness.pdf"]
                if isinstance(document, MineruApiError):
                    raise document
                result = project_document(document)
                if not result.raw_text.strip():
                    raise MineruApiError(
                        f"MinerU {tier.value} readiness produced no text"
                    )
            except Exception:
                mark_failed(tier)
                raise
            mark_executed(tier)

    def parse(
        self,
        data: bytes,
        mime_type: str,
        options: OCROptions | MineruConfig | None = None,
    ) -> OCRResult:
        filename = "input" + self._get_extension(mime_type)
        document = self._call_api(data, filename, options)[filename]
        if isinstance(document, MineruApiError):
            raise document
        return project_document(document)

    def file_parse(
        self,
        files: list[tuple[str, bytes]],
        *,
        options: OCROptions | MineruConfig | None = None,
        backend: str | None = None,
        cancelled: Callable[[], bool] = lambda: False,
    ) -> dict[str, dict[str, Any]]:
        if not files:
            return {}
        if backend is not None and backend != "hybrid-engine":
            # Old callers must explicitly reselect an equivalent new tier.
            migrate_legacy_options({"backend": backend})
        documents = self._call_api(
            b"", "multi.bin", options, files=files, cancelled=cancelled
        )
        from vibeocr.backend.models import ocr_result_to_payload

        result = {}
        for filename, document in documents.items():
            if isinstance(document, MineruApiError):
                result[Path(filename).stem] = {"mineru_error": str(document)}
                continue
            result[Path(filename).stem] = ocr_result_to_payload(
                project_document(document)
            )
        return result

    def _get_extension(self, mime_type: str) -> str:
        return mime_to_extension(mime_type) or ".pdf"

    def _build_ocr_result(
        self,
        api_result: dict[str, Any],
        filename: str,
        data: bytes | None = None,
    ) -> OCRResult:
        """从 API 响应构建 OCRResult"""
        stem = Path(filename).stem
        results = api_result.get("results", {})
        file_result = results.get(stem, {})

        md_content = file_result.get("md_content") or ""

        content_list_raw = file_result.get("content_list")
        content_list_parsed: list = []
        if content_list_raw:
            try:
                content_list_parsed = json.loads(content_list_raw)
            except (json.JSONDecodeError, TypeError):
                content_list_parsed = []

        table_sequence = 0
        page_groups = (
            content_list_parsed
            if content_list_parsed and isinstance(content_list_parsed[0], list)
            else [content_list_parsed]
        )
        for inferred_page_idx, page_blocks in enumerate(page_groups):
            for block_index, block in enumerate(page_blocks):
                if not isinstance(block, dict):
                    continue
                block.setdefault(
                    "block_id",
                    (f"mineru-{stem}-page-{inferred_page_idx}-block-{block_index}"),
                )
                if block.get("type") != "table":
                    continue
                content = block.get("content")
                nested_html = ""
                if isinstance(content, dict):
                    nested_html = str(
                        content.get("table_body")
                        or content.get("table_html")
                        or content.get("html")
                        or ""
                    )
                if nested_html and not (block.get("table_body") or block.get("html")):
                    block["table_body"] = nested_html
                if not (
                    isinstance(block.get("table"), dict)
                    or block.get("table_body")
                    or block.get("html")
                ):
                    block["source_type"] = "table"
                    block["type"] = "table_unparsed"
                    table_content = (
                        content.get("table_content")
                        if isinstance(content, dict)
                        else None
                    )
                    block["text"] = " ".join(
                        str(item.get("content") or "")
                        for item in (
                            table_content if isinstance(table_content, list) else []
                        )
                        if isinstance(item, dict) and item.get("content")
                    )
                    block["projection_warnings"] = [
                        (f"{block['block_id']}:structured-table-unsupported")
                    ]
                    continue
                page_idx = block.get("page_idx", inferred_page_idx)
                table_id = str(
                    block.get("block_id")
                    or block.get("table_id")
                    or (f"mineru-{stem}-page-{page_idx}-table-{table_sequence}")
                )
                source_html = str(block.get("table_body") or block.get("html") or "")
                canonical_input = dict(block)
                if source_html and not re.search(
                    r"<table\b", source_html, flags=re.IGNORECASE
                ):
                    canonical_input["table_body"] = f"<table>{source_html}</table>"
                    source = dict(block.get("source") or {})
                    source.setdefault("source_html", source_html)
                    canonical_input["source"] = source
                canonical_block = canonicalize_table_block(
                    canonical_input,
                    table_id=table_id,
                    pipeline="MinerU",
                )
                table_model = table_model_from_block(canonical_block)
                table_model = replace(
                    table_model,
                    provenance=TableProvenanceV1(
                        pipeline="MinerU",
                        provider_schema="mineru-content-list",
                    ),
                )
                canonical_block["table"] = table_model.to_payload()
                page_blocks[block_index] = canonical_block
                table_sequence += 1

        # 通过正常化层统一格式
        normalized = normalize_content_list(content_list_parsed)
        for normalized_block in normalized:
            if normalized_block.get("type") != "table":
                continue
            raw_block = normalized_block.get("raw")
            if isinstance(raw_block, dict) and isinstance(raw_block.get("table"), dict):
                table_model = table_model_from_block(raw_block)
                normalized_block["text"] = table_model_to_plain_text(table_model)

        flat_content_list: list[dict[str, Any]] = []
        is_v2_content = bool(
            content_list_parsed and isinstance(content_list_parsed[0], list)
        )
        for normalized_block in normalized:
            raw_block = normalized_block.get("raw")
            if not isinstance(raw_block, dict):
                continue
            flat_block = dict(raw_block)
            if is_v2_content:
                # V2 使用 page_header/page_footer 等 provider 类型；投影层只识别
                # 归一化后的 header/footer，因此扁平化时携带统一语义，避免页眉泄漏。
                flat_block["type"] = normalized_block.get(
                    "type", flat_block.get("type")
                )
                flat_block["text"] = normalized_block.get("text", "")
            elif flat_block.get("type") == "text" and isinstance(
                flat_block.get("text_level"), int
            ):
                # MinerU legacy 标题以 text + text_level 表示。
                flat_block["type"] = "title"
                flat_block["level"] = flat_block["text_level"]
            flat_block.setdefault("page_idx", normalized_block.get("page_idx"))
            flat_content_list.append(flat_block)

        images: dict[str, bytes] = {}
        images_dict = file_result.get("images", {})
        for img_name, data_uri in images_dict.items():
            if data_uri and data_uri.startswith("data:"):
                b64_part = data_uri.split(",", 1)[-1]
                images[img_name] = base64.b64decode(b64_part)

        # 从 normalized 提取纯文本
        raw_text = (
            self._extract_plain_text_normalized(normalized)
            if normalized
            else md_content
        )

        # 从 normalized 构建 text_blocks
        text_blocks: list[TextBlock] = []
        for i, block in enumerate(normalized):
            block_type = block.get("type", "")
            if block_type in DISCARDED_BLOCK_TYPES:
                continue
            bbox = block.get("bbox")
            if (not bbox or len(bbox) < 4) and block_type != "table":
                continue
            text = block.get("text", "")
            if not text:
                continue

            text_blocks.append(
                TextBlock(
                    text=text,
                    score=1.0,  # MineRU content_list 不提供 confidence
                    bbox=normalize_bbox(bbox[:4]) if bbox else None,
                    page_idx=block.get("page_idx"),
                    content_index=i,
                    content_id=(block.get("raw") or {}).get("block_id"),
                )
            )

        text_with_scores = [(b.text, b.score) for b in text_blocks]
        avg_score = (
            sum(s for _, s in text_with_scores) / len(text_with_scores)
            if text_with_scores
            else 0.0
        )

        result = OCRResult(
            raw_text=raw_text,
            markdown_text=md_content,
            html_text="",
            text_with_scores=text_with_scores,
            avg_score=avg_score,
            low_confidence_items=[],
            pipeline_type="MinerU",
            images=images,
            content_list=flat_content_list,
            text_blocks=text_blocks,
        )
        rebuild_result_projections(result)
        return result

    @staticmethod
    def _strip_html(html: str) -> str:
        """从 HTML 中提取纯文本，合并多余空白"""
        text = re.sub(r"<[^>]+>", " ", html)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _extract_plain_text_normalized(normalized: list[dict]) -> str:
        """从 normalize_content_list 的输出提取纯文本"""
        parts: list[str] = []
        for block in normalized:
            block_type = block.get("type", "")
            if block_type in DISCARDED_BLOCK_TYPES:
                continue
            text = block.get("text", "")
            if text:
                parts.append(text)
        return "\n".join(parts)

    @staticmethod
    def _extract_plain_text(content_list: list[dict]) -> str:
        """从 content_list 提取纯文本"""
        parts: list[str] = []
        for block in content_list:
            block_type = block.get("type", "")
            if block_type in DISCARDED_BLOCK_TYPES:
                continue
            if block_type == "table":
                captions = block.get("table_caption") or []
                if captions:
                    parts.append(" ".join(captions))
                html = block.get("table_body", "")
                parts.append(MinerUService._strip_html(html))
            elif block_type in ("image", "chart"):
                captions = (
                    block.get("image_caption") or block.get("chart_caption") or []
                )
                if captions:
                    parts.append(" ".join(captions))
            elif block_type == "list":
                items = block.get("list_items", [])
                parts.extend(items)
            elif block_type == "code":
                body = block.get("code_body", "")
                parts.append(body)
            else:
                text = block.get("text", "")
                if text:
                    parts.append(text)
        return "\n".join(p for p in parts if p)

    @staticmethod
    def _extract_block_text(block: dict) -> str:
        """从单个 content_list 块提取用于 TextBlock 的文本"""
        block_type = block.get("type", "")
        if block_type == "table":
            captions = block.get("table_caption") or []
            cap_text = " ".join(captions)
            html = block.get("table_body", "")
            body_text = MinerUService._strip_html(html)
            if cap_text and body_text:
                return f"{cap_text}\n{body_text}"
            return cap_text or body_text
        if block_type in ("image", "chart"):
            captions = block.get("image_caption") or block.get("chart_caption") or []
            content = block.get("content", "")
            text = " ".join(captions)
            if content:
                text = f"{text} {content}".strip()
            return text or f"[{block_type}]"
        if block_type == "list":
            items = block.get("list_items", [])
            return "; ".join(items)
        if block_type == "code":
            return block.get("code_body", "")[:200]
        return block.get("text", "")

    def shutdown(self) -> None:
        """停止 mineru-api 进程"""
        # 先关闭 Job 守卫：触发内核 kill mineru-api，使后续 terminate 对已死进程为 no-op
        if self.__class__._job_guard is not None:
            self.__class__._job_guard.close()
            self.__class__._job_guard = None
        if self.__class__._api_process is not None:
            _logger.debug("[MinerU] 停止 mineru-api 服务...")
            self.__class__._api_process.terminate()
            try:
                self.__class__._api_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.__class__._api_process.kill()
            self.__class__._api_process = None
            self.__class__._api_url = ""
            _logger.debug("[MinerU] mineru-api 服务已停止")
