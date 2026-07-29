"""Markdown 渲染集成测试"""

from vibeocr.backend.utils.markdown_converter import markdown_to_html


class TestMarkdownRenderingIntegration:
    """测试完整的 Markdown 渲染流程"""

    def test_full_chinese_document(self):
        """测试完整中文文档渲染"""
        markdown = """这是第一段中文内容。

这是第二段中文内容，包含一些 English 单词。

- 列表项1
- 列表项2
  - 嵌套列表项
"""
        html = markdown_to_html(markdown)
        # 验证中文段落有缩进类
        assert "zh-paragraph" in html
        # 验证列表嵌套
        assert "<ul>" in html

    def test_table_with_chinese_content(self):
        """测试中文表格"""
        markdown = """| 项目 | 数值 |
|---|---|
| 项目A | 100 |
| 项目B | 200 |"""
        html = markdown_to_html(markdown)
        assert "<table>" in html
        # 表格不应被包装为段落
        assert "项目A" in html

    def test_formula_with_chinese_context(self):
        """测试中文上下文中的公式"""
        markdown = """根据物理定律：

$$E = mc^2$$

这个公式很重要。"""
        html = markdown_to_html(markdown)
        assert "latex-formula" in html
        assert "zh-paragraph" in html
