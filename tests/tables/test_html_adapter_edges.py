"""tables.html_adapter 规范化表格 HTML 解析的边缘用例测试。

覆盖 _TableParser 与坐标投影的异常路径：嵌套表/多表/未闭合表、
span 非法值、cell/coverage/dimension 超限、source span parser 与
table_model_to_html 往返。
"""

from __future__ import annotations

import pytest
from vibeocr.backend.tables.html_adapter import (
    MAX_HTML_TABLE_SOURCE_CHARS,
    TableCellSourceSpan,
    TableSourceLayout,
    parse_table_source_layout,
    table_model_from_html,
    table_model_to_html,
)


class TestTableModelFromHtmlErrors:
    def test_source_exceeds_limit(self):
        """源超过 MAX_HTML_TABLE_SOURCE_CHARS 抛 ValueError。"""
        huge = "x" * (MAX_HTML_TABLE_SOURCE_CHARS + 1)
        with pytest.raises(ValueError, match="source exceeds"):
            table_model_from_html(huge, table_id="t")

    def test_no_table_raises(self):
        with pytest.raises(ValueError, match="does not contain a table"):
            table_model_from_html("<div>no table</div>", table_id="t")

    def test_nested_table_raises(self):
        nested = "<table><tr><td><table><tr><td>x</td></tr></table></td></tr></table>"
        with pytest.raises(ValueError, match="nested"):
            table_model_from_html(nested, table_id="t")

    def test_multiple_top_level_tables_raises(self):
        multi = "<table><tr><td>A</td></tr></table><table><tr><td>B</td></tr></table>"
        with pytest.raises(ValueError, match="multiple"):
            table_model_from_html(multi, table_id="t")

    def test_unclosed_table_raises(self):
        with pytest.raises(ValueError, match="not closed"):
            table_model_from_html("<table><tr><td>x</td></tr>", table_id="t")

    def test_invalid_span_value_raises(self):
        with pytest.raises(ValueError, match="invalid HTML table colspan"):
            table_model_from_html(
                '<table><tr><td colspan="abc">x</td></tr></table>', table_id="t"
            )

    def test_span_zero_raises(self):
        with pytest.raises(ValueError, match="invalid HTML table rowspan"):
            table_model_from_html(
                '<table><tr><td rowspan="0">x</td></tr></table>', table_id="t"
            )


class TestSelfClosingTags:
    def test_self_closing_cell(self):
        """自闭合 <td/> 经 startendtag 处理后产生空单元格。"""
        html = "<table><tr><td/></tr></table>"
        model = table_model_from_html(html, table_id="t")
        assert model.row_count == 1
        assert model.column_count == 1
        assert model.cells[0].text == ""

    def test_self_closing_table_tag(self):
        """自闭合 <table/> 在 startendtag 中触发 endtag，得到空表（0 行）。"""
        html = "<table/>"
        # startendtag(table) -> starttag(table) depth 0->1, then handle_endtag
        # closes at depth 1 -> completed_table=True
        model = table_model_from_html(html, table_id="t")
        assert model.row_count == 0
        assert model.column_count == 0


class TestSpansAndCoordinates:
    def test_rowspan_colspan_coordinates(self):
        html = (
            "<table>"
            "<tr><td rowspan='2'>A</td><td colspan='2'>B</td></tr>"
            "<tr><td>C</td></tr>"
            "</table>"
        )
        model = table_model_from_html(html, table_id="t")
        # A 占 r0c0 + r1c0; B 占 r0c1,r0c2; 第二行从 c1 起找空位，C 落到 r1c1
        assert model.row_count == 2
        assert model.column_count == 3
        cell_c = next(c for c in model.cells if c.text == "C")
        assert cell_c.row == 1
        assert cell_c.column == 1

    def test_data_cell_id_preserved(self):
        html = '<table><tr><td data-cell-id="custom">x</td></tr></table>'
        model = table_model_from_html(html, table_id="t")
        assert model.cells[0].cell_id == "custom"

    def test_header_flag(self):
        html = "<table><tr><th>H</th><td>D</td></tr></table>"
        model = table_model_from_html(html, table_id="t")
        assert model.cells[0].is_header is True
        assert model.cells[1].is_header is False

    def test_empty_table_model(self):
        """有 <table></table> 但无行：row_count=0。"""
        html = "<table></table>"
        # </table> 直接连，completed_table=True, rows 空
        model = table_model_from_html(html, table_id="t")
        assert model.row_count == 0
        assert model.column_count == 0
        assert len(model.cells) == 0


class TestParseTableSourceLayout:
    def test_basic_layout(self):
        html = "<table><tr><td>A</td><td>B</td></tr></table>"
        layout = parse_table_source_layout(html, table_id="t")
        assert isinstance(layout, TableSourceLayout)
        assert len(layout.cells) == 2
        assert all(isinstance(c, TableCellSourceSpan) for c in layout.cells)
        # source_text 与 canonical cell text 一致
        assert layout.cells[0].source_text == "A"

    def test_br_in_cell_preserved_in_source_text(self):
        """<br> 在单元格内：canonical text 保留换行，source_text 与之一致。"""
        html = "<table><tr><td>A<br>B</td></tr></table>"
        layout = parse_table_source_layout(html, table_id="t")
        assert len(layout.cells) == 1
        assert layout.cells[0].source_text == "A\nB"

    def test_propagates_parse_errors(self):
        """底层 table_model_from_html 的错误透传。"""
        with pytest.raises(ValueError, match="does not contain"):
            parse_table_source_layout("<div>nope</div>", table_id="t")

    def test_source_exceeds_limit(self):
        with pytest.raises(ValueError, match="source exceeds"):
            parse_table_source_layout(
                "x" * (MAX_HTML_TABLE_SOURCE_CHARS + 1), table_id="t"
            )


class TestTableModelToHtml:
    def test_roundtrip(self):
        html = '<table><tr><td data-cell-id="r0c0">A</td></tr></table>'
        model = table_model_from_html(html, table_id="t1")
        rendered = table_model_to_html(model)
        assert rendered.startswith('<table data-table-id="t1">')
        assert ">A<" in rendered

    def test_escapes_table_id(self):
        html = "<table><tr><td>x</td></tr></table>"
        model = table_model_from_html(html, table_id='t"x')
        rendered = table_model_to_html(model)
        # 双引号转义为 &quot;，不出现在属性值的原始位置
        assert 'data-table-id="t&quot;x"' in rendered

    def test_renders_spans(self):
        html = (
            "<table>"
            "<tr><td rowspan='2'>A</td><td colspan='2'>B</td></tr>"
            "<tr><td>C</td></tr>"
            "</table>"
        )
        model = table_model_from_html(html, table_id="t")
        rendered = table_model_to_html(model)
        assert 'rowspan="2"' in rendered
        assert 'colspan="2"' in rendered

    def test_newline_as_br(self):
        html = "<table><tr><td>line1\nline2</td></tr></table>"
        model = table_model_from_html(html, table_id="t")
        rendered = table_model_to_html(model)
        assert "<br>" in rendered
