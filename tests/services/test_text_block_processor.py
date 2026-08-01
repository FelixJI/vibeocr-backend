# tests/services/test_text_block_processor.py
"""TextBlockProcessor 单元测试（纯函数，无 Qt 依赖）"""

from vibeocr.backend.models.ocr_result import TextBlock
from vibeocr.backend.models.text_block_options import (
    LINE_MODE_KEEP,
    LINE_MODE_MERGE,
    LINE_MODE_SMART,
    TextBlockOptions,
)
from vibeocr.backend.utils.text_layout import TextBlockProcessor


def _block(text, bbox=None, order=-1):
    """构造测试用 TextBlock。bbox=(x1,y1,x2,y2) 归一化 [0,1000]。"""
    return TextBlock(text=text, score=0.9, bbox=bbox, order=order)


# ── keep 模式 ──


class TestKeepMode:
    def test_keep_each_block_on_own_line(self):
        blocks = [_block("甲"), _block("乙"), _block("丙")]
        opts = TextBlockOptions(line_mode=LINE_MODE_KEEP)
        assert TextBlockProcessor.process(blocks, opts) == "甲\n乙\n丙"

    def test_keep_no_indent_even_if_enabled(self):
        blocks = [_block("甲"), _block("乙")]
        opts = TextBlockOptions(line_mode=LINE_MODE_KEEP, chinese_indent=True)
        assert TextBlockProcessor.process(blocks, opts) == "甲\n乙"


# ── merge 模式 ──


class TestMergeMode:
    def test_merge_no_space(self):
        blocks = [_block("第一"), _block("第二"), _block("第三")]
        opts = TextBlockOptions(line_mode=LINE_MODE_MERGE, block_join_space=False)
        assert TextBlockProcessor.process(blocks, opts) == "第一第二第三"

    def test_merge_with_space(self):
        blocks = [_block("hello"), _block("world")]
        opts = TextBlockOptions(line_mode=LINE_MODE_MERGE, block_join_space=True)
        assert TextBlockProcessor.process(blocks, opts) == "hello world"

    def test_merge_chinese_indent(self):
        blocks = [_block("第一"), _block("第二")]
        opts = TextBlockOptions(line_mode=LINE_MODE_MERGE, chinese_indent=True)
        result = TextBlockProcessor.process(blocks, opts)
        assert result.startswith("\u3000\u3000")
        assert result == "\u3000\u3000第一第二"


# ── smart 模式 ──


class TestSmartMode:
    def _two_paragraph_blocks(self):
        """段内两块（紧邻）、段间大 gap。

        前块行高 = 100（y:100→200），后段首块 gap = 200 > 1.5*100=150 → 断段。
        """
        return [
            _block("段1行1", bbox=(0, 100, 100, 200)),  # 行高100
            _block("段1行2", bbox=(0, 210, 100, 310)),  # gap=10 ≤ 150 同段
            _block("段2行1", bbox=(0, 511, 100, 611)),  # gap=201 > 150 新段
            _block("段2行2", bbox=(0, 621, 100, 721)),  # gap=10 同段
        ]

    def test_smart_segments_split_by_gap(self):
        blocks = self._two_paragraph_blocks()
        opts = TextBlockOptions(line_mode=LINE_MODE_SMART, block_join_space=False)
        result = TextBlockProcessor.process(blocks, opts)
        assert result == "段1行1段1行2\n\n段2行1段2行2"

    def test_smart_with_space(self):
        blocks = self._two_paragraph_blocks()
        opts = TextBlockOptions(line_mode=LINE_MODE_SMART, block_join_space=True)
        result = TextBlockProcessor.process(blocks, opts)
        assert result == "段1行1 段1行2\n\n段2行1 段2行2"

    def test_smart_chinese_indent(self):
        blocks = self._two_paragraph_blocks()
        opts = TextBlockOptions(
            line_mode=LINE_MODE_SMART,
            block_join_space=False,
            chinese_indent=True,
        )
        result = TextBlockProcessor.process(blocks, opts)
        assert result == "\u3000\u3000段1行1段1行2\n\n\u3000\u3000段2行1段2行2"

    def test_smart_no_gap_single_paragraph(self):
        """所有块紧邻 → 视为单段。"""
        blocks = [
            _block("甲", bbox=(0, 0, 100, 100)),
            _block("乙", bbox=(0, 110, 100, 210)),  # gap=10
        ]
        opts = TextBlockOptions(line_mode=LINE_MODE_SMART, block_join_space=True)
        assert TextBlockProcessor.process(blocks, opts) == "甲 乙"

    def test_smart_gap_boundary_exactly_1_5x(self):
        """gap 恰好等于 1.5× 行高 → 不算断段（> 严格大于）。"""
        # 行高 100，gap=150 == 1.5×100，不断段
        blocks = [
            _block("甲", bbox=(0, 0, 100, 100)),
            _block("乙", bbox=(0, 250, 100, 350)),  # gap=150
        ]
        opts = TextBlockOptions(line_mode=LINE_MODE_SMART, block_join_space=True)
        assert TextBlockProcessor.process(blocks, opts) == "甲 乙"

    def test_smart_gap_just_over_boundary(self):
        """gap 刚超过 1.5× 行高 → 断段。"""
        blocks = [
            _block("甲", bbox=(0, 0, 100, 100)),
            _block("乙", bbox=(0, 251, 100, 351)),  # gap=151 > 150
        ]
        opts = TextBlockOptions(line_mode=LINE_MODE_SMART, block_join_space=True)
        assert TextBlockProcessor.process(blocks, opts) == "甲\n\n乙"


# ── 边界情况 ──


class TestEdgeCases:
    def test_empty_blocks_returns_empty(self):
        assert TextBlockProcessor.process([], TextBlockOptions()) == ""

    def test_drop_blank_blocks(self):
        blocks = [_block("甲"), _block("   "), _block(""), _block("乙")]
        opts = TextBlockOptions(line_mode=LINE_MODE_MERGE, drop_blank_blocks=True)
        assert TextBlockProcessor.process(blocks, opts) == "甲乙"

    def test_keep_blank_blocks_when_disabled(self):
        blocks = [_block("甲"), _block("   "), _block("乙")]
        opts = TextBlockOptions(line_mode=LINE_MODE_KEEP, drop_blank_blocks=False)
        # 空白块保留，按原样 join
        assert TextBlockProcessor.process(blocks, opts) == "甲\n   \n乙"

    def test_all_blank_blocks_returns_empty(self):
        blocks = [_block("   "), _block("")]
        opts = TextBlockOptions(line_mode=LINE_MODE_MERGE)
        assert TextBlockProcessor.process(blocks, opts) == ""

    def test_bbox_none_smart_treated_as_same_segment(self):
        """bbox 为 None 的块在 smart 模式下视为同段。"""
        blocks = [
            _block("甲", bbox=(0, 0, 100, 100)),
            _block("乙", bbox=None),  # 无法判段 → 同段
        ]
        opts = TextBlockOptions(line_mode=LINE_MODE_SMART, block_join_space=True)
        assert TextBlockProcessor.process(blocks, opts) == "甲 乙"

    def test_zero_height_block_same_segment(self):
        """行高为 0 的块（y2==y1）按同段处理，避免除零。"""
        blocks = [
            _block("甲", bbox=(0, 100, 100, 100)),  # 行高0
            _block("乙", bbox=(0, 500, 100, 600)),  # gap 很大但前块行高0
        ]
        opts = TextBlockOptions(line_mode=LINE_MODE_SMART, block_join_space=True)
        assert TextBlockProcessor.process(blocks, opts) == "甲 乙"


# ── 排序 ──


class TestSorting:
    def test_order_sort_when_present(self):
        """有有效 order 值时按 order 升序（无视 bbox 顺序）。"""
        blocks = [
            _block("丙", bbox=(0, 900, 100, 1000), order=3),
            _block("甲", bbox=(0, 100, 100, 200), order=1),
            _block("乙", bbox=(0, 500, 100, 600), order=2),
        ]
        opts = TextBlockOptions(line_mode=LINE_MODE_MERGE)
        assert TextBlockProcessor.process(blocks, opts) == "甲乙丙"

    def test_bbox_fallback_when_all_default_order(self):
        """order 全为 -1 时 fallback 到 bbox 排序（先 y 后 x）。
        构造顺序颠倒的块，验证输出按 y 排序。
        """
        blocks = [
            _block("下", bbox=(0, 800, 100, 900)),
            _block("上", bbox=(0, 100, 100, 200)),
            _block("中", bbox=(0, 500, 100, 600)),
        ]
        opts = TextBlockOptions(line_mode=LINE_MODE_MERGE)
        assert TextBlockProcessor.process(blocks, opts) == "上中下"

    def test_bbox_sort_by_x_when_same_y(self):
        """同 y 不同 x 时按 x 排序（从左到右）。"""
        blocks = [
            _block("右", bbox=(500, 100, 600, 200)),
            _block("左", bbox=(0, 100, 100, 200)),
        ]
        opts = TextBlockOptions(line_mode=LINE_MODE_MERGE)
        assert TextBlockProcessor.process(blocks, opts) == "左右"


# ── 索引保留变体（_sort_indexed / _split_indexed_into_segments / _join_segment 空文本）──


class TestIndexedVariants:
    """显示层排版用的索引保留变体。"""

    def test_sort_indexed_by_order(self):
        indexed = [
            (0, _block("A", order=2)),
            (1, _block("B", order=0)),
            (2, _block("C", order=1)),
        ]
        result = TextBlockProcessor._sort_indexed(indexed)
        assert [i for i, _b in result] == [1, 2, 0]

    def test_sort_indexed_by_bbox_when_no_order(self):
        indexed = [
            (0, _block("A", bbox=(10, 100, 20, 110))),
            (1, _block("B", bbox=(10, 50, 20, 60))),
        ]
        result = TextBlockProcessor._sort_indexed(indexed)
        # y1 小的排前：B(50) < A(100)
        assert [i for i, _b in result] == [1, 0]

    def test_split_indexed_into_segments_breaks_on_gap(self):
        # prev 高度 10（y1=100,y2=110），cur y1=200 → gap=90 > 1.5*10=15 → 新段
        indexed = [
            (0, _block("A", bbox=(0, 100, 10, 110))),
            (1, _block("B", bbox=(0, 200, 10, 210))),
        ]
        segments = TextBlockProcessor._split_indexed_into_segments(indexed)
        assert len(segments) == 2
        assert [i for seg in segments for i, _b in seg] == [0, 1]

    def test_split_indexed_into_segments_same_segment_on_small_gap(self):
        # prev 高度 100（y1=100,y2=200），cur y1=210 → gap=10 < 150 → 同段
        indexed = [
            (0, _block("A", bbox=(0, 100, 10, 200))),
            (1, _block("B", bbox=(0, 210, 10, 310))),
        ]
        segments = TextBlockProcessor._split_indexed_into_segments(indexed)
        assert len(segments) == 1
        assert len(segments[0]) == 2

    def test_join_segment_returns_empty_when_all_blank(self):
        blocks = [_block(""), _block("")]
        assert TextBlockProcessor._join_segment(blocks, join_space=True) == ""
        assert TextBlockProcessor._join_segment(blocks, join_space=False) == ""
