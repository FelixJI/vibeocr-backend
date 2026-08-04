"""tables.projections 规范化表格投影的边缘用例测试。

覆盖空网格 markdown 投影、merged_ranges markdown 警告、tsv/plain_text 输出。
"""

from __future__ import annotations

from vibeocr.backend.tables.html_adapter import table_model_from_html
from vibeocr.backend.tables.projections import (
    table_model_to_grid,
    table_model_to_markdown,
    table_model_to_plain_text,
    table_model_to_tsv,
)


def _model(html: str = "<table><tr><td>A</td></tr></table>", table_id: str = "t"):
    return table_model_from_html(html, table_id=table_id)


class TestTableModelToMarkdown:
    def test_empty_grid_returns_empty(self):
        """row_count=0 的空表 markdown 投影返回空文本。"""
        model = table_model_from_html("<table></table>", table_id="t")
        proj = table_model_to_markdown(model)
        assert proj.text == ""
        assert proj.warnings == ()

    def test_merged_ranges_emit_warning(self):
        """含合并单元格的表 markdown 投影带 lossy 警告。"""
        html = (
            "<table>"
            "<tr><td rowspan='2'>A</td><td>B</td></tr>"
            "<tr><td>C</td></tr>"
            "</table>"
        )
        model = table_model_from_html(html, table_id="t")
        proj = table_model_to_markdown(model)
        assert "lossy_markdown_source" in proj.warnings

    def test_no_merge_no_warning(self):
        model = _model()
        proj = table_model_to_markdown(model)
        assert proj.warnings == ()

    def test_pipe_escaped(self):
        html = "<table><tr><td>a|b</td></tr></table>"
        model = table_model_from_html(html, table_id="t")
        proj = table_model_to_markdown(model)
        assert "\\|" in proj.text

    def test_separator_row(self):
        model = _model("<table><tr><td>H</td><td>K</td></tr></table>")
        proj = table_model_to_markdown(model)
        lines = proj.text.splitlines()
        # line 0 header, line 1 separator, line 2+ body
        assert set(lines[1]) <= {"|", "-", " "}


class TestTableModelToGrid:
    def test_dense_grid(self):
        model = _model("<table><tr><td>A</td><td>B</td></tr></table>")
        grid = table_model_to_grid(model)
        assert grid == [["A", "B"]]

    def test_span_leaves_covered_empty(self):
        html = "<table><tr><td colspan='2'>A</td></tr></table>"
        model = table_model_from_html(html, table_id="t")
        grid = table_model_to_grid(model)
        assert grid == [["A", ""]]


class TestTableModelToPlainText:
    def test_tab_separated(self):
        model = _model("<table><tr><td>A</td><td>B</td></tr></table>")
        text = table_model_to_plain_text(model)
        assert text == "A\tB"

    def test_multiple_rows(self):
        model = _model("<table><tr><td>A</td></tr><tr><td>B</td></tr></table>")
        text = table_model_to_plain_text(model)
        assert text == "A\nB"


class TestTableModelToTsv:
    def test_full_grid_tsv(self):
        model = _model("<table><tr><td>A</td><td>B</td></tr></table>")
        assert table_model_to_tsv(model) == "A\tB"

    def test_empty_cells_in_tsv(self):
        html = "<table><tr><td colspan='2'>A</td></tr></table>"
        model = table_model_from_html(html, table_id="t")
        assert table_model_to_tsv(model) == "A\t"
