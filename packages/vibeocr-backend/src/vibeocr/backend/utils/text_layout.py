"""文本块后处理器（UI 层输出排版逻辑）。

纯函数、无 Qt 依赖。将 OCR 识别出的文本块按用户选项拼接为最终 raw_text，
是输出排版逻辑（不是 OCR 引擎逻辑）。按 ADR §5.2，UI 负责「展示结果」，
故此模块属于 UI/工具层，不放在后端 ``services`` 中。

处理流水线：去空白块 → 排序 → 按 line_mode 拼接 → 可选中文段落缩进。

Backend-owned text layout helpers used by OCR result projection.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

from vibeocr.backend.models.text_block_options import (
    LINE_MODE_KEEP,
    LINE_MODE_MERGE,
    TextBlockOptions,
)

if TYPE_CHECKING:
    from vibeocr.backend.models.ocr_result import TextBlock

# smart 模式段落边界阈值：垂直 gap > 阈值因子 × 前块行高 → 视为新段落。
# 不暴露给用户。
_PARAGRAPH_GAP_FACTOR = 1.5

# 中文段落缩进：两个全角空格（U+3000 × 2）
_CJK_INDENT = "\u3000\u3000"


class TextBlockProcessor:
    """文本块后处理器（静态方法集合，无状态）。"""

    @staticmethod
    def process(
        text_blocks: list[TextBlock],
        options: TextBlockOptions,
        image_height: int = 0,
    ) -> str:
        """文本块 → 处理后的 raw_text。

        Args:
            text_blocks: OCR 识别出的文本块列表。
            options: 文本块处理选项。
            image_height: 原图高度（像素，当前实现未使用，预留）。

        Returns:
            拼接后的纯文本。空块列表返回空字符串。
        """
        if not text_blocks:
            return ""

        # 1. 去空白块
        blocks = text_blocks
        if options.drop_blank_blocks:
            blocks = [b for b in blocks if b.text and b.text.strip()]

        if not blocks:
            return ""

        # 2. 排序
        blocks = TextBlockProcessor._sort_blocks(blocks)

        # 3. 按 line_mode 拼接
        mode = options.line_mode
        if mode == LINE_MODE_KEEP:
            # keep 模式不缩进，每块一行（现状）
            return "\n".join(b.text for b in blocks)

        if mode == LINE_MODE_MERGE:
            text = TextBlockProcessor._join_segment(
                blocks, options.block_join_space
            )
            if options.chinese_indent:
                text = _CJK_INDENT + text
            return text

        # LINE_MODE_SMART
        segments = TextBlockProcessor._split_into_segments(blocks)
        parts = [
            TextBlockProcessor._join_segment(seg, options.block_join_space)
            for seg in segments
        ]
        if options.chinese_indent:
            parts = [_CJK_INDENT + p for p in parts]
        return "\n\n".join(parts)

    # ── 内部辅助 ──

    @staticmethod
    def _sort_blocks(blocks: list[TextBlock]) -> list[TextBlock]:
        """排序：当存在有效的 order 值（非全为 -1）时按 order 升序；
        否则 fallback 到 bbox 排序（先 y1 后 x1），保证确定性。
        """
        has_order = any(b.order != -1 for b in blocks)
        if has_order:
            return sorted(blocks, key=lambda b: (b.order, _bbox_sort_key(b)))
        return sorted(blocks, key=_bbox_sort_key)

    @staticmethod
    def _split_into_segments(blocks: list[TextBlock]) -> list[list[TextBlock]]:
        """smart 模式：按垂直间距将块切分为段。

        相邻块 gap > 1.5 × 前块行高 → 段落边界。
        bbox 为 None 的块无法判段，视为同段（并入当前段）。
        行高为 0 时（y2==y1）按同段处理，避免退化。
        """
        segments: list[list[TextBlock]] = [[blocks[0]]]
        for prev, cur in itertools.pairwise(blocks):
            if TextBlockProcessor._is_paragraph_break(prev, cur):
                segments.append([cur])
            else:
                segments[-1].append(cur)
        return segments

    # ── 索引保留变体（供显示层排版用，保持原始 text_blocks 下标）──
    # 入参为 (原始下标, 块) 元组列表，输出同构；排序/分段逻辑与上方一致，
    # 仅在比较键上解包出块。这样 drop_blank 过滤或排序重排后，显示层仍能
    # 用原始下标作为 data-block-index，保证双击编辑/悬停联动按 index 命中。

    @staticmethod
    def _sort_indexed(
        indexed: list[tuple[int, TextBlock]],
    ) -> list[tuple[int, TextBlock]]:
        """_sort_blocks 的索引保留变体。"""
        has_order = any(b.order != -1 for _, b in indexed)
        if has_order:
            return sorted(
                indexed, key=lambda ib: (ib[1].order, _bbox_sort_key(ib[1]))
            )
        return sorted(indexed, key=lambda ib: _bbox_sort_key(ib[1]))

    @staticmethod
    def _split_indexed_into_segments(
        indexed: list[tuple[int, TextBlock]],
    ) -> list[list[tuple[int, TextBlock]]]:
        """_split_into_segments 的索引保留变体。"""
        segments: list[list[tuple[int, TextBlock]]] = [[indexed[0]]]
        for (_pi, prev), (_ci, cur) in itertools.pairwise(indexed):
            if TextBlockProcessor._is_paragraph_break(prev, cur):
                segments.append([(_ci, cur)])
            else:
                segments[-1].append((_ci, cur))
        return segments

    @staticmethod
    def _is_paragraph_break(prev: TextBlock, cur: TextBlock) -> bool:
        """判定 cur 是否开启新段落。"""
        if prev.bbox is None or cur.bbox is None:
            return False
        _, py1, _, py2 = prev.bbox
        _cx1, cy1, _, _ = cur.bbox
        prev_height = py2 - py1
        if prev_height <= 0:
            return False
        gap = cy1 - py2
        return gap > _PARAGRAPH_GAP_FACTOR * prev_height

    @staticmethod
    def _join_segment(blocks: list[TextBlock], join_space: bool) -> str:
        """将一段内的块拼接为连续文本。

        join_space=True 时块间插入半角空格；否则直接拼接。
        """
        texts = [b.text for b in blocks if b.text]
        if not texts:
            return ""
        sep = " " if join_space else ""
        return sep.join(texts)


def _bbox_sort_key(b: TextBlock) -> tuple[float, float]:
    """bbox 排序键（先 y1 后 x1）；bbox 为 None 时排到最后且互相稳定。"""
    if b.bbox is None:
        return (float("inf"), float("inf"))
    x1, y1, _, _ = b.bbox
    return (y1, x1)


__all__ = ["TextBlockProcessor"]
