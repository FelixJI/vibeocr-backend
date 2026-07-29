"""应用服务契约：DTO、Protocol 和错误类型。

所有类型都是 frozen dataclass 或 Protocol，不依赖 PySide6。
这是 WorkerHost（Phase 1）和 WinUI 壳（Phase 2+）共享的接口定义。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

# ---------------------------------------------------------------------------
# 取消令牌
# ---------------------------------------------------------------------------


class CancelToken:
    """协作取消令牌（线程安全）。

    用 threading.Event 实现。adapter 在每个昂贵步骤前检查 is_cancelled。
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        """请求取消。"""
        self._event.set()


# ---------------------------------------------------------------------------
# OCR DTOs
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OcrRequest:
    """OCR 识别请求。

    Attributes:
        image_data: 图片二进制数据（PNG/JPEG bytes）。
        pipeline: 管道名称（如 "OCR"、"TABLE_RECOGNITION"、"FORMULA_RECOGNITION"）。
        language: 识别语言（可选，默认由管道决定）。
    """

    image_data: bytes
    pipeline: str = "OCR"
    language: str | None = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OcrResult:
    """OCR 识别结果。

    Attributes:
        text: 识别出的纯文本。
        raw_blocks: 原始文本块（结构化数据，供编辑器/表格渲染）。
        pipeline: 使用的管道名称。
        markdown_text: Markdown 格式内容。
        html_text: HTML 格式内容。
        raw_text: 原始纯文本。
        text_blocks: 序列化的文本块列表（每块含 text/bbox/confidence/order）。
        text_with_scores: [(text, score), ...] 文本块及置信度。
        content_list: 结构化内容列表（MinerU/表格/公式管道）。
        image_width: 原图宽度（像素）。
        image_height: 原图高度（像素）。
        preprocessed_image_ref: 预处理后图像的 shared-payload 描述符（可选）。
    """

    text: str
    raw_blocks: list[Any] = field(default_factory=list)
    pipeline: str = "OCR"
    markdown_text: str = ""
    html_text: str = ""
    raw_text: str = ""
    text_blocks: list[dict[str, Any]] = field(default_factory=list)
    text_with_scores: list[list[Any]] = field(default_factory=list)
    content_list: list[dict[str, Any]] = field(default_factory=list)
    image_width: int = 0
    image_height: int = 0
    preproc_angle: int = 0
    preproc_img_w: int = 0
    preproc_img_h: int = 0
    preprocessed_image_ref: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class OcrExportRequest:
    raw_text: str
    markdown_text: str
    html_text: str
    raw_blocks: list[Any]
    output_path: Path
    format: str
    overwrite: bool = False


@dataclass(frozen=True, slots=True)
class OcrExportResult:
    output_path: Path
    bytes_written: int


# ---------------------------------------------------------------------------
# PDF DTOs
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PdfOpenRequest:
    """PDF 打开请求。

    Attributes:
        file_path: PDF 文件路径。
    """

    file_path: Path


@dataclass(frozen=True, slots=True)
class PdfSessionDto:
    """PDF 会话 DTO（打开结果）。

    Attributes:
        session_id: 会话标识（传给后续 PDF 操作）。
        file_path: 文件路径。
        page_count: 总页数。
    """

    session_id: str
    file_path: Path
    page_count: int


# ---------------------------------------------------------------------------
# Settings DTOs
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SettingsSnapshot:
    """设置快照（只读视图）。

    Attributes:
        backend: 当前后端（"cpu" 或 "gpu"）。
        preload_pipelines: 预加载管道列表。
        pipeline_ttls: 每管道 TTL（秒）字典，0=持久。
    """

    backend: str = "cpu"
    preload_pipelines: tuple[str, ...] = ()
    pipeline_ttls: Mapping[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 错误类型
# ---------------------------------------------------------------------------


class OcrError(Exception):
    """OCR 操作错误（含取消）。"""


class PdfError(Exception):
    """PDF 操作错误（含取消）。"""


class SettingsError(Exception):
    """设置操作错误。"""


# ---------------------------------------------------------------------------
# Application Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class OcrApplication(Protocol):
    """OCR 应用服务接口。"""

    def recognize(self, request: OcrRequest, cancel: CancelToken) -> OcrResult: ...


@runtime_checkable
class PdfApplication(Protocol):
    """PDF 应用服务接口。"""

    def open(self, request: PdfOpenRequest, cancel: CancelToken) -> PdfSessionDto: ...


@runtime_checkable
class SettingsApplication(Protocol):
    """设置应用服务接口。"""

    def get_snapshot(self) -> SettingsSnapshot: ...
