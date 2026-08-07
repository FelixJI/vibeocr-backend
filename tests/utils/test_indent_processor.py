"""indent_processor 中文判定与 Markdown 缩进处理的边缘用例测试。"""

from __future__ import annotations

import pytest
from vibeocr.backend.utils.indent_processor import IndentConfig, IndentProcessor


@pytest.fixture
def processor() -> IndentProcessor:
    """默认配置的处理器。"""
    return IndentProcessor()


def test_default_config_matches_markdown_presentation_policy() -> None:
    """默认处理器使用两字符缩进和 5% 中文判定阈值。"""
    config = IndentConfig()
    assert config.chinese_indent == "2em"
    assert config.chinese_threshold == 0.05


class TestIsChineseText:
    """is_chinese_text 边缘用例。"""

    def test_empty_string_false(self, processor: IndentProcessor):
        """空串返回 False。"""
        assert processor.is_chinese_text("") is False

    def test_whitespace_only_false(self, processor: IndentProcessor):
        """纯空白返回 False。"""
        assert processor.is_chinese_text("   \n\t  ") is False

    def test_pure_english_false(self, processor: IndentProcessor):
        """纯英文返回 False。"""
        assert processor.is_chinese_text("This is English text") is False

    def test_pure_chinese_true(self, processor: IndentProcessor):
        """纯中文返回 True。"""
        assert processor.is_chinese_text("这是一段中文") is True

    def test_threshold_boundary(self):
        """阈值边界：刚好等于 chinese_threshold 视为中文。"""
        proc = IndentProcessor(IndentConfig(chinese_threshold=0.5))
        # 2 个中文字符 / 4 总字符 = 0.5，命中阈值
        assert proc.is_chinese_text("中中aa") is True
        # 1 / 4 = 0.25 低于阈值
        assert proc.is_chinese_text("中aaa") is False

    def test_mixed_meets_threshold(self, processor: IndentProcessor):
        """中英混合且中文占比达标视为中文。"""
        assert processor.is_chinese_text("中文内容 mixed words") is True


class TestProcessMarkdown:
    """process_markdown 边缘用例。"""

    def test_empty_returns_empty(self, processor: IndentProcessor):
        """空输入返回空串。"""
        assert processor.process_markdown("") == ""

    def test_chinese_paragraph_wrapped(self, processor: IndentProcessor):
        """中文段落被包进 zh-paragraph div。"""
        result = processor.process_markdown("这是一段中文")
        assert '<div class="zh-paragraph">这是一段中文</div>' in result

    def test_english_paragraph_not_wrapped(self, processor: IndentProcessor):
        """英文段落不加 div。"""
        result = processor.process_markdown("plain english text")
        assert "zh-paragraph" not in result

    def test_table_paragraph_skipped(self, processor: IndentProcessor):
        """表格段跳过不加 div。"""
        md = "| 列1 | 列2 |\n|---|---|\n| 值1 | 值2 |"
        result = processor.process_markdown(md)
        assert "zh-paragraph" not in result

    def test_list_paragraph_skipped(self, processor: IndentProcessor):
        """列表段（无序/有序）跳过不加 div。"""
        md = "- 项目一\n- 项目二"
        assert "zh-paragraph" not in processor.process_markdown(md)
        md2 = "1. 第一\n2. 第二"
        assert "zh-paragraph" not in processor.process_markdown(md2)

    def test_code_block_preserved(self, processor: IndentProcessor):
        """代码块内容原样保留。"""
        md = "```python\n中文注释\nprint(1)\n```"
        result = processor.process_markdown(md)
        assert "```python" in result
        assert "中文注释" in result
        # 代码块内部不应被包成 zh-paragraph
        assert "zh-paragraph" not in result

    def test_html_block_skipped(self, processor: IndentProcessor):
        """含 ≥3 个 HTML 块标签的 VLM 输出整段跳过。"""
        md = "<table><tr><td>中文</td></tr></table>"
        result = processor.process_markdown(md)
        assert "zh-paragraph" not in result
        assert result == md

    def test_few_html_tags_still_processed(self, processor: IndentProcessor):
        """HTML 块标签 <3 个时不触发跳过，照常处理段落。"""
        # 仅 2 个块标签，触发段落处理逻辑
        md = "<div>中文段落内容比较长哦</div>"
        result = processor.process_markdown(md)
        # 段落被 strip 处理，div 标签本身不计入表格/列表，照常判断中文
        assert "中文段落内容比较长哦" in result

    def test_lone_code_fence_paragraph_skipped(self, processor: IndentProcessor):
        """独立成段的单行代码围栏标记（非成对代码块）原样保留，不包 div。

        覆盖 process_markdown 中 ``para.startswith("```")`` 的跳过分支：
        当按双换行分割后某段恰好是单个 ``` 围栏时，避免误判为中文段落。
        """
        md = "这是中文段落。\n\n```\n\n另一段中文。"
        result = processor.process_markdown(md)
        # 独立的 ``` 段落原样保留，不被包进 zh-paragraph
        assert "zh-paragraph" in result  # 中文段落仍被包装
        # 围栏标记本身不应被包进 div
        assert '<div class="zh-paragraph">```' not in result

    def test_mixed_code_blocks_and_chinese(self, processor: IndentProcessor):
        """成对代码块与中文段落混合时各自正确处理。"""
        md = "```python\ncode\n```\n\n这是一段中文内容。"
        result = processor.process_markdown(md)
        assert "```python" in result
        assert '<div class="zh-paragraph">这是一段中文内容。</div>' in result
