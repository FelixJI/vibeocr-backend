"""OCR 识别结果数据类"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

_log = logging.getLogger(__name__)

# MinerU 丢弃的块类型——应排除在文本提取和渲染之外
DISCARDED_BLOCK_TYPES = frozenset(
    {
        "header",
        "footer",
        "page_number",
        "page_footnote",
        "aside_text",
    }
)


def normalize_bbox(
    bbox_raw: list | tuple,
    img_w: int = 0,
    img_h: int = 0,
) -> tuple[float, float, float, float]:
    """将任意范围的 bbox 统一为 [0, 1000] 归一化坐标。

    支持输入: [0, 1] 归一化、[0, 1000] 归一化、像素坐标。
    像素坐标需传入 img_w/img_h 才能归一化，否则保留原值并警告。
    """
    vals = (
        float(bbox_raw[0]),
        float(bbox_raw[1]),
        float(bbox_raw[2]),
        float(bbox_raw[3]),
    )
    max_val = max(vals)
    if max_val < 1.1:
        # [0, 1] 归一化 → 放大到 [0, 1000]
        result = tuple(v * 1000 for v in vals)
        _log.debug("[bbox] [0,1] → [0,1000]: %s", result)
        return result  # type: ignore[return-value]
    if max_val > 1001 and img_w > 0 and img_h > 0:
        # 像素坐标（max > 1001，超出 [0,1000] 归一化范围）→ 归一化
        result = (
            vals[0] / img_w * 1000,
            vals[1] / img_h * 1000,
            vals[2] / img_w * 1000,
            vals[3] / img_h * 1000,
        )
        _log.debug("[bbox] pixel → [0,1000] (%dx%d): %s", img_w, img_h, result)
        return result
    if max_val >= 1100:
        _log.warning("[bbox] unexpected pixel coords (max=%s): %s", max_val, vals)
    return vals


def normalize_polygon(
    poly_raw: list | tuple,
    img_w: int = 0,
    img_h: int = 0,
) -> tuple[float, ...] | None:
    """将 4 点检测多边形 [x0,y0,x1,y1,...] 统一为 [0, 1000] 归一化坐标。

    与 normalize_bbox 同口径：像素坐标需 img_w/img_h，[0,1] 自动放大。
    多边形用于 PDF 文字层判断文本行方向（横/竖排），是 bbox AABB 之外的
    几何补充——长边方向编码阅读方向，AABB 把它塌缩掉了。
    返回 None 表示无法解析（调用方回退到 bbox 长宽比）。
    """
    flat: list[float] = []
    for p in poly_raw:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            flat.append(float(p[0]))
            flat.append(float(p[1]))
        elif isinstance(p, (int, float)):
            flat.append(float(p))
    if len(flat) < 8 or len(flat) % 2 != 0:
        return None
    max_val = max(flat)
    if max_val < 1.1:
        return tuple(v * 1000 for v in flat)
    if max_val > 1001 and img_w > 0 and img_h > 0:
        out: list[float] = []
        for i, v in enumerate(flat):
            out.append(v / (img_w if i % 2 == 0 else img_h) * 1000)
        return tuple(out)
    return tuple(flat)


# V2 类型 → 内部统一类型名映射
_V2_TYPE_MAP: dict[str, str] = {
    "title": "title",
    "paragraph": "text",
    "equation_interline": "equation",
    "equation_inline": "equation",
    "page_header": "header",
    "page_footer": "footer",
    "page_aside_text": "aside_text",
    "page_footnote": "page_footnote",
    "page_number": "page_number",
    "image": "image",
    "table": "table",
    "chart": "chart",
    "code": "code",
    "algorithm": "code",
    "list": "list",
    "index": "list",
    "seal": "image",
    "phonetic": "text",
    "ref_text": "text",
}


def _extract_v2_text(block: dict) -> str:
    """从 V2 块的 content 字段提取文本"""
    content = block.get("content", {})

    text_parts: list[str] = []
    for key, val in content.items():
        if key.endswith("_content") and isinstance(val, list):
            for item in val:
                if isinstance(item, dict):
                    text_parts.append(item.get("content", ""))
        elif key == "math_content" and isinstance(val, str):
            text_parts.append(val)
        elif key in ("list_items",) and isinstance(val, list):
            text_parts.extend(str(v) for v in val)
        elif key == "code_body" and isinstance(val, str):
            text_parts.append(val)

    if not text_parts:
        text_parts.append(block.get("text", ""))

    return " ".join(p for p in text_parts if p).replace("  ", " ")


def _normalize_v2(raw: list[list[dict]]) -> list[dict]:
    """将 V2 格式（按页分组的嵌套数组）转为扁平的内部格式"""
    result: list[dict] = []
    for page_idx, page_blocks in enumerate(raw):
        for block in page_blocks:
            v2_type = block.get("type", "")
            mapped_type = _V2_TYPE_MAP.get(v2_type, v2_type)
            text = _extract_v2_text(block)
            result.append(
                {
                    "type": mapped_type,
                    "text": text,
                    "bbox": tuple(block["bbox"]) if "bbox" in block else None,
                    "page_idx": page_idx,
                    "raw": block,
                }
            )
    return result


def _extract_legacy_text(block: dict, block_type: str) -> str:
    """从 legacy 格式的单个块提取文本（处理 table/list/image 等特殊类型）"""
    import re

    if block_type == "table":
        captions = block.get("table_caption") or []
        cap_text = " ".join(captions)
        html = block.get("table_body", "")
        # 从 HTML 提取纯文本
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()
        if cap_text and text:
            return f"{cap_text}\n{text}"
        return cap_text or text
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


def _normalize_legacy(raw: list[dict]) -> list[dict]:
    """Legacy 格式直接透传，附加 raw 字段"""
    result: list[dict] = []
    for block in raw:
        block_type = block.get("type", "")
        text = _extract_legacy_text(block, block_type)
        result.append(
            {
                "type": block_type,
                "text": text,
                "bbox": tuple(block["bbox"]) if "bbox" in block else None,
                "page_idx": block.get("page_idx"),
                "raw": block,
            }
        )
    return result


def normalize_content_list(raw: list | None) -> list[dict]:
    """将 legacy 或 V2 content_list 统一为内部格式

    返回扁平的 list[dict]，每项包含:
    - type: str（统一类型名）
    - text: str（提取的文本）
    - bbox: tuple | None
    - page_idx: int | None
    - raw: dict（原始块数据）
    """
    if not raw:
        return []
    if isinstance(raw[0], list):
        return _normalize_v2(raw)
    return _normalize_legacy(raw)


@dataclass
class TextBlock:
    """单个文本块，含文本、置信度和位置信息"""

    text: str
    score: float
    bbox: tuple[float, float, float, float] | None  # 归一化 [0, 1000] 坐标
    polygon: tuple[float, ...] | None = None  # 4 点检测多边形 [x,y,...]，归一化 [0,1000]
    page_idx: int | None = None  # 页码（0 起始），PDF 多页时使用
    is_manually_edited: bool = False  # 是否被手动修改过
    content_index: int | None = None  # 对应 content_list 中的索引（MinerU 管道）
    content_id: str | None = None  # 对应 content_list block_id 的稳定标识
    label: str = "text"  # 元素类型: text/table/formula/image/doc_title...
    order: int = -1  # 阅读顺序


@dataclass
class OCRResult:
    """OCR 识别结果

    Attributes:
        raw_text: 纯文本内容
        markdown_text: Markdown 格式内容（包含表格、公式等）
        html_text: HTML 格式内容（用于富文本显示）
        text_with_scores: 文本块及置信度列表 [(文本, 置信度), ...]
        avg_score: 平均置信度
        low_confidence_items: 低置信度文本块列表 [(文本, 置信度), ...]
        pipeline_type: 管道类型名称
        images: 图像字典（如 markdown_images）
    """

    raw_text: str = ""
    markdown_text: str = ""
    html_text: str = ""
    text_with_scores: list[tuple[str, float]] = field(default_factory=list)
    avg_score: float = 0.0
    low_confidence_items: list[tuple[str, float]] = field(default_factory=list)
    pipeline_type: str = "OCR"
    images: dict[str, Any] = field(default_factory=dict)
    content_list: list[dict[str, Any]] = field(default_factory=list)
    text_blocks: list[TextBlock] = field(default_factory=list)
    image_width: int = 0
    image_height: int = 0
    preproc_angle: int = 0
    preprocessed_image: bytes | None = None
    preproc_img_w: int = 0
    preproc_img_h: int = 0

    @property
    def has_rich_content(self) -> bool:
        """是否有富文本内容（表格、公式等）"""
        return bool(self.html_text and self.html_text != self.raw_text)

    @property
    def has_content_list(self) -> bool:
        """是否包含结构化内容列表（含布局信息）"""
        return bool(self.content_list)

    @property
    def display_text(self) -> str:
        """用于显示的文本（优先使用富文本）"""
        return self.html_text if self.has_rich_content else self.raw_text

    @property
    def copy_text(self) -> str:
        """用于复制的纯文本（Markdown 格式）"""
        return self.markdown_text if self.markdown_text else self.raw_text
