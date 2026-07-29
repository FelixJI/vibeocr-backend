"""PDF 文档数据模型"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TextLayerInfo:
    """单个文字层信息"""

    index: int
    text_preview: str
    char_count: int
    bbox: tuple[float, float, float, float]
    color_id: int


@dataclass
class PdfPageInfo:
    """单页状态"""

    page_index: int
    rotation: int = 0
    has_text_layer: bool = False
    text_layers: list[TextLayerInfo] = field(default_factory=list)
    is_scanned: bool = False
    # Frontends may cache their native thumbnail object here.  The shared
    # model deliberately does not name a Qt type so backend/client wheels stay
    # importable without PySide6.
    thumbnail: object | None = None
    # 页面几何（x0, y0, x1, y1，PDF point）。打开/结构变更时由 PdfService 回填，
    # 供主进程预览 highlight 计算用——下沉子进程后主进程不再直接访问 fitz 对象。
    rect: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    # OCR 原始块（归一化 [0,1000] bbox），预览/编辑/重写 PDF 文字层的唯一信源。
    # detect_text_layers 重读会被 PyMuPDF 合并块，不能作为预览信源，故单独缓存。
    ocr_text_blocks: list = field(default_factory=list)
    ocr_preproc_angle: int = 0
    deskewed: bool = False  # 本会话内是否被自动摆正纠正过（仅 UI 标记，不持久化）


@dataclass
class PdfDocument:
    """PDF 文档状态"""

    file_path: str | None = None
    pages: list[PdfPageInfo] = field(default_factory=list)
    is_modified: bool = False
    has_structural_change: bool = False  # 结构性改动（删页/插页/重排），影响保存策略
    render_dpi: int = 300
    thumbnail_dpi: int = 96

    def get_page(self, index: int | None) -> PdfPageInfo | None:
        if index is None:
            return None
        if 0 <= index < len(self.pages):
            return self.pages[index]
        return None

    @property
    def page_count(self) -> int:
        return len(self.pages)
