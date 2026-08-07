"""markdown_converter 的 LaTeX 预处理、HTML 抽取与渲染兜底边缘用例测试。"""

from __future__ import annotations

import pytest
from vibeocr.backend.utils.markdown_converter import (
    HTML_STYLE,
    _process_latex_formulas,
    extract_plain_text,
    markdown_to_html,
)


def test_html_style_preserves_chinese_and_nested_list_indentation() -> None:
    """正式 HTML 样式包含中文段落和嵌套列表缩进规则。"""
    assert ".zh-paragraph" in HTML_STYLE
    assert "text-indent" in HTML_STYLE
    assert "li p" in HTML_STYLE
    assert "margin-left" in HTML_STYLE


class TestProcessLatexFormulas:
    """_process_latex_formulas 边缘用例。"""

    def test_block_formula_becomes_div(self):
        """块级 $$...$$ 转为 latex-formula div。"""
        result = _process_latex_formulas("$$E = mc^2$$")
        assert '<div class="latex-formula">E = mc^2</div>' in result

    def test_block_formula_multiline(self):
        """多行块级公式整体捕获。"""
        result = _process_latex_formulas("$$\na = b\n$$")
        assert '<div class="latex-formula">' in result
        assert "a = b" in result

    def test_inline_formula_becomes_span(self):
        """行内 $...$ 转为 latex-inline span。"""
        result = _process_latex_formulas("能量 $E$ 很大")
        assert '<span class="latex-inline">E</span>' in result

    def test_html_special_chars_escaped_in_formula(self):
        """公式内 <,>,& 被 HTML 转义。"""
        result = _process_latex_formulas("$$a < b & c > d$$")
        assert '<div class="latex-formula">' in result
        # 公式主体内的特殊字符已转义（不再以裸 <,>,& 出现）
        assert "a &lt; b &amp; c &gt; d" in result
        assert "&lt;" in result
        assert "&gt;" in result
        assert "&amp;" in result

    def test_table_separator_not_injured(self):
        """表格分隔行 |---| 不被行内公式规则误伤。"""
        result = _process_latex_formulas("|---|---|")
        assert "latex-inline" not in result

    def test_plain_text_unchanged(self):
        """无公式的纯文本保持不变。"""
        assert _process_latex_formulas("普通文本") == "普通文本"


class TestExtractPlainText:
    """extract_plain_text 边缘用例。"""

    def test_empty_returns_empty(self):
        """空输入返回空串。"""
        assert extract_plain_text("") == ""

    def test_removes_style_block(self):
        """style 标签整块移除。"""
        html = "<style>body{color:red}</style><p>hi</p>"
        assert "color" not in extract_plain_text(html)
        assert "hi" in extract_plain_text(html)

    def test_block_elements_become_newlines_tabs(self):
        """</p>/<div>/<tr> 换行，</td>/<th> 制表。"""
        html = "<table><tr><td>A</td><td>B</td></tr></table>"
        text = extract_plain_text(html)
        assert "A" in text and "B" in text
        assert "\t" in text

    def test_html_entities_decoded(self):
        """常见 HTML 实体解码回原字符。"""
        html = "a &amp; b &lt; c &gt; d &nbsp; e &quot;f&quot;"
        text = extract_plain_text(html)
        assert "&" in text
        assert "<" in text
        assert ">" in text
        assert '"' in text

    def test_strips_all_tags(self):
        """所有 HTML 标签被移除。"""
        html = "<p><strong>bold</strong> <a href='#x'>link</a></p>"
        text = extract_plain_text(html)
        assert "<" not in text
        assert "bold" in text
        assert "link" in text

    def test_blank_lines_collapsed(self):
        """结果不包含空行。"""
        html = "<p>a</p><p></p><p>b</p>"
        text = extract_plain_text(html)
        assert "" not in text.split("\n")
        assert "a" in text and "b" in text


class TestMarkdownToHtml:
    """markdown_to_html 边缘用例。"""

    def test_empty_returns_empty(self):
        """空文本返回空串。"""
        assert markdown_to_html("") == ""

    def test_includes_style_by_default(self):
        """默认带 CSS 样式块。"""
        html = markdown_to_html("# Title")
        assert HTML_STYLE in html
        assert "<body>" in html

    def test_without_style(self):
        """include_style=False 仅返回 body HTML。"""
        html = markdown_to_html("# Title", include_style=False)
        assert HTML_STYLE not in html
        assert "<h1>" in html

    def test_table_extension_active(self):
        """表格扩展生效。"""
        md = "| a | b |\n|---|---|\n| 1 | 2 |"
        html = markdown_to_html(md, include_style=False)
        assert "<table>" in html

    def test_exception_falls_back_to_escaped_pre(self, monkeypatch: pytest.MonkeyPatch):
        """markdown 库抛错时走兜底 <pre> 转义分支。"""
        import markdown as md_lib

        def raise_always(*args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(md_lib, "markdown", raise_always)
        html = markdown_to_html("a < b & c", include_style=False)
        assert "<pre>" in html
        # 特殊字符已转义
        assert "&lt;" in html
        assert "&amp;" in html

    def test_exception_fallback_with_style(self, monkeypatch: pytest.MonkeyPatch):
        """兜底分支在 include_style=True 时也带样式。"""
        import markdown as md_lib

        monkeypatch.setattr(
            md_lib, "markdown", lambda *a, **k: (_ for _ in ()).throw(RuntimeError())
        )
        html = markdown_to_html("oops", include_style=True)
        assert HTML_STYLE in html
        assert "<pre>" in html
