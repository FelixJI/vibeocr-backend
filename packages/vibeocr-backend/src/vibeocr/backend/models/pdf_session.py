"""PDF 会话数据模型 — 单个已打开文件的状态(进程化后,主进程只持 mirror)。"""

from __future__ import annotations

from dataclasses import dataclass, field

from vibeocr.backend.models.pdf_document import PdfDocument


@dataclass
class PdfSession:
    """单个 PDF 文件的会话状态(主进程侧,不持 fitz.Document)。

    进程化重构后,fitz.Document 只存在于 PDF 后端子进程。主进程通过
    session_id 标识后端会话,pdf_document 是从 PdfDocumentMirror 转换来的
    显示镜像(供 UI 读页数/旋转/文字层状态用)。doc_lock 已删除——后端
    单线程,无需锁。

    保留 loaded_pages / ocr_stats 等 UI 状态字段,PdfTab/manager 依赖它们。
    """

    file_path: str
    session_id: str = ""  # 后端 session 标识
    pdf_document: PdfDocument = field(default_factory=PdfDocument)
    loaded_pages: set[int] = field(default_factory=set)
    _ocr_stats: dict[str, int] = field(
        default_factory=lambda: {"written": 0, "skipped": 0}, repr=False
    )

    @property
    def is_modified(self) -> bool:
        return self.pdf_document.is_modified

    @property
    def load_progress(self) -> float:
        total = self.pdf_document.page_count
        if total == 0:
            return 1.0
        return len(self.loaded_pages) / total

    @property
    def ocr_stats(self) -> dict[str, int]:
        return self._ocr_stats

    def reset_ocr_stats(self) -> None:
        self._ocr_stats = {"written": 0, "skipped": 0}

    def add_ocr_stats(self, written: int, skipped: int) -> None:
        self._ocr_stats["written"] += written
        self._ocr_stats["skipped"] += skipped
