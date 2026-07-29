# src/vibeocr/models/pdf_ocr_options.py
"""PDF OCR 全局设置数据模型

PDF 处理的专用参数（DPI、内存控制、字号策略等），
与截图识别的 OCROptions 完全独立。
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class PdfGlobalSettings:
    """PDF OCR 全局设置

    控制 PDF 页面渲染和文字层写入行为。
    不包含管道选项（管道选项复用 OCROptions，通过 OCRPreferences "pdf" 数据源管理）。

    Attributes:
        render_dpi: PDF 页面渲染 DPI（传给 render_page_as_array）。
        max_pixels: 单页最大像素数，超过时自动降低 DPI。
        font_size_ratio: 字号占矩形高度的比例。
        text_layer_visible: True → render_mode=0（可见），False → render_mode=3（隐形）。
        font_size_retry_count: 文字溢出时字号缩放重试次数。
        font_size_shrink_factor: 每次重试的字号缩放因子。
        min_font_size: 字号下限（pt）。矮行/窄框导致字号过小时夹紧到此值，
            确保隐形文字层仍可被阅读器提取，避免因“塞不进”整块丢弃。
        compress_on_save: 覆盖保存时的压缩策略。True（默认）= 全量重写
            （garbage=3 + deflate + object streams + 临时文件原子替换），兼顾体积与速度，
            代价是
            大文件保存稍慢；False = 增量追加快路径（incremental，快但大，保留
            旧行为）。OCR 加文字层后建议 True，否则字体/文字流不压缩、旧对象
            不回收，720MB 文档可膨胀上百 MB。
        clean_on_save: 全量压缩时是否『深度清理』内容流（PyMuPDF clean 参数）。
            True = save(garbage=3 + deflate + object streams + clean)，重写并规范化所有内容流，
            适合来源不规范/需要清洗的 PDF。False（默认）= save(garbage=3 +
            deflate + clean=False)，保留原始内容流压缩，不重写。
            重要：很多扫描件驱动用 ObjStm/CrossRefStream 高度压缩，MuPDF（PyMuPDF
            底层）在序列化时无法保留这种压缩结构（会展平成传统 xref 表），
            clean=True 还会解压并重写内容流，两者叠加可使 1.6MB 扫描件膨胀到 3MB+。
            仅加一层隐形文字层是纯增量场景，默认 clean=False 保留原压缩度、体积
            最小；需要清洗/规范化时再手动开 True。
    """

    render_dpi: int = 300
    max_pixels: int = 16_000_000
    font_size_ratio: float = 0.8
    text_layer_visible: bool = False
    font_size_retry_count: int = 5
    font_size_shrink_factor: float = 0.75
    min_font_size: float = 4.0
    compress_on_save: bool = True
    clean_on_save: bool = False

    def to_dict(self) -> dict:
        return {
            "render_dpi": self.render_dpi,
            "max_pixels": self.max_pixels,
            "font_size_ratio": self.font_size_ratio,
            "text_layer_visible": self.text_layer_visible,
            "font_size_retry_count": self.font_size_retry_count,
            "font_size_shrink_factor": self.font_size_shrink_factor,
            "min_font_size": self.min_font_size,
            "compress_on_save": self.compress_on_save,
            "clean_on_save": self.clean_on_save,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PdfGlobalSettings:
        if not data:
            return cls()
        return cls(
            render_dpi=data.get("render_dpi", 300),
            max_pixels=data.get("max_pixels", 16_000_000),
            font_size_ratio=data.get("font_size_ratio", 0.8),
            text_layer_visible=data.get("text_layer_visible", False),
            font_size_retry_count=data.get("font_size_retry_count", 5),
            font_size_shrink_factor=data.get("font_size_shrink_factor", 0.75),
            min_font_size=data.get("min_font_size", 4.0),
            compress_on_save=data.get("compress_on_save", True),
            clean_on_save=data.get("clean_on_save", False),
        )

    def adjust_dpi(self, page_width: float, page_height: float) -> int:
        """根据像素上限自动调整 DPI。

        Args:
            page_width: PDF 页面宽度（points）。
            page_height: PDF 页面高度（points）。

        Returns:
            调整后的 DPI（不超过 render_dpi，不超过 max_pixels 限制）。
        """
        target_dpi = self.render_dpi
        pixel_w = page_width / 72.0 * target_dpi
        pixel_h = page_height / 72.0 * target_dpi
        total_pixels = pixel_w * pixel_h
        if total_pixels <= self.max_pixels:
            return target_dpi
        # 计算满足 max_pixels 的最大 DPI
        scale = math.sqrt(self.max_pixels / total_pixels)
        adjusted = int(target_dpi * scale)
        # 确保至少 72 DPI
        return max(72, adjusted)
