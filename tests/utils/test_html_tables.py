"""utils.html_tables 表格 HTML 规整化与提取的边缘用例测试。

覆盖纯函数分支：rowspan/colspan 占位布局、空表/无表格降级、
markdown 换行与管道转义、cell_grid 跨行占位补齐、多表收集去重、
tables_from_result 的 canonical table 优先路径与各来源兜底。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from vibeocr.backend.tables.html_adapter import table_model_from_html
from vibeocr.backend.utils.html_tables import (
    extract_table_html,
    html_table_to_cell_grid,
    html_table_to_markdown,
    html_tables_to_cell_grid,
    normalize_table_html,
    tables_from_result,
)


class TestExtractTableHtml:
    def test_returns_first_table_block(self):
        html = "before<table><tr><td>A</td></tr></table>after"
        assert extract_table_html(html) == "<table><tr><td>A</td></tr></table>"

    def test_returns_input_when_no_table(self):
        """无 <table> 时原样返回输入。"""
        assert extract_table_html("no table here") == "no table here"

    def test_case_insensitive_table_tag(self):
        html = "<TABLE><tr><td>A</td></tr></TABLE>"
        assert extract_table_html(html).lower().startswith("<table")


class TestHtmlTableToMarkdown:
    def test_empty_when_no_rows(self):
        assert html_table_to_markdown("<table></table>") == ""

    def test_pipe_character_escaped(self):
        """单元格内的管道符必须转义，避免破坏 GFM 分隔语义。"""
        html = "<table><tr><td>a|b</td></tr></table>"
        md = html_table_to_markdown(html)
        # 转义后的 \| 出现在单元格位置；分隔 | 仍存在但单元格内的已转义
        assert "a\\|b" in md

    def test_newline_converted_to_br(self):
        """单元格内换行转为 <br>，避免 GFM 吞掉换行。"""
        html = "<table><tr><td>line1<br>line2</td></tr></table>"
        md = html_table_to_markdown(html)
        assert "<br>" in md

    def test_html_entities_unescaped(self):
        html = "<table><tr><td>a&amp;b</td></tr></table>"
        md = html_table_to_markdown(html)
        assert "a&b" in md

    def test_single_column_body_only(self):
        """仅一行数据时 header 与 separator 正确，body 为空。"""
        html = "<table><tr><td>only</td></tr></table>"
        md = html_table_to_markdown(html)
        lines = md.splitlines()
        assert lines[0].startswith("| only |")
        assert set(lines[1]) <= {"|", "-", " "}

    def test_ragged_rows_padded(self):
        """列数不齐的行用空串补齐到最大列数。"""
        html = "<table><tr><td>h1</td><td>h2</td></tr><tr><td>x</td></tr></table>"
        md = html_table_to_markdown(html)
        body_lines = md.splitlines()[2:]
        for line in body_lines:
            # 每行（header/sep/body）都应有 2 个单元格 = 3 个 |
            assert line.count("|") == 3


class TestNormalizeTableHtml:
    def test_strips_inline_style(self):
        """PaddleX 的 inline style 被剥离，仅保留数字 rowspan/colspan。"""
        html = (
            '<table><tr><td style="background:red">A</td>'
            '<td colspan="2" class="x">B</td></tr></table>'
        )
        out = normalize_table_html(html)
        assert "style" not in out
        assert "background" not in out
        assert 'colspan="2"' in out

    def test_empty_rows_returns_empty_table(self):
        assert normalize_table_html("<table><tr></tr></table>") == "<table></table>"

    def test_no_table_tag_returns_empty(self):
        """无 <table> 时把整个输入当表格 HTML 处理；无单元格则返回空表。"""
        assert normalize_table_html("plain text") == "<table></table>"

    def test_rowspan_attribute_preserved(self):
        html = "<table><tr><td rowspan='2'>A</td></tr><tr><td>B</td></tr></table>"
        out = normalize_table_html(html)
        assert 'rowspan="2"' in out

    def test_span_value_clamped_to_1000(self):
        """超出 1000 的 span 值被钳制到 1000。"""
        html = "<table><tr><td colspan='9999'>A</td></tr></table>"
        out = normalize_table_html(html)
        assert 'colspan="1000"' in out

    def test_padding_to_max_cols(self):
        """不足最大列数的行用空 <td></td> 补齐。"""
        html = "<table><tr><td>A</td><td>B</td></tr><tr><td>C</td></tr></table>"
        out = normalize_table_html(html)
        assert out.count("<td></td>") >= 1

    def test_html_escaped_cell_text(self):
        """单元格文本中的特殊字符被 HTML 转义。"""
        html = "<table><tr><td>a<b>x</b></td></tr></table>"
        out = normalize_table_html(html)
        # 标签被 _cell_text 剥离，剩纯文本 "ax"，再 escape
        assert "<b>" not in out


class TestHtmlTableToCellGrid:
    def test_empty_when_no_table(self):
        assert html_table_to_cell_grid("no table") == []

    def test_rowspan_creates_placeholder(self):
        """rowspan 使后续行对应列留空占位。"""
        html = (
            "<table>"
            "<tr><td rowspan='2'>A</td><td>B</td></tr>"
            "<tr><td>C</td></tr>"
            "</table>"
        )
        grid = html_table_to_cell_grid(html)
        assert grid == [["A", "B"], ["", "C"]]

    def test_colspan_fills_covered_columns(self):
        """colspan>1 的单元格占据多列，后续列被填充为空。"""
        html = "<table><tr><td colspan='2'>A</td></tr></table>"
        grid = html_table_to_cell_grid(html)
        assert grid == [["A", ""]]

    def test_strips_html_shell(self):
        """带 <html><body> 外壳的表格能正常解析。"""
        html = "<html><body><table><tr><td>X</td></tr></table></body></html>"
        grid = html_table_to_cell_grid(html)
        assert grid == [["X"]]

    def test_drops_empty_rows(self):
        """无单元格的空 <tr> 被丢弃。"""
        html = "<table><tr><td>A</td></tr><tr></tr></table>"
        grid = html_table_to_cell_grid(html)
        assert len(grid) == 1

    def test_ragged_rows_padded_to_width(self):
        """各行补齐到最大宽度。"""
        html = "<table><tr><td>A</td><td>B</td></tr><tr><td>C</td></tr></table>"
        grid = html_table_to_cell_grid(html)
        assert all(len(row) == 2 for row in grid)


class TestHtmlTablesToCellGrid:
    def test_multiple_tables(self):
        html = "<table><tr><td>A</td></tr></table><table><tr><td>B</td></tr></table>"
        grids = html_tables_to_cell_grid(html)
        assert len(grids) == 2
        assert grids[0] == [["A"]]
        assert grids[1] == [["B"]]

    def test_no_tables_returns_empty(self):
        assert html_tables_to_cell_grid("no tables here") == []

    def test_empty_table_skipped(self):
        """空表格（无单元格）不产生矩阵。"""
        html = "<table><tr></tr></table><table><tr><td>X</td></tr></table>"
        grids = html_tables_to_cell_grid(html)
        assert len(grids) == 1
        assert grids[0] == [["X"]]


def _canonical_table_block(table_id: str = "t1", text: str = "A") -> dict[str, Any]:
    """构造含 canonical table payload 的 block（触发 tables_from_result 优先路径）。"""
    table = table_model_from_html(
        f"<table><tr><td>{text}</td></tr></table>", table_id=table_id
    )
    return {"type": "table", "table": table.to_payload(), "block_id": table_id}


class TestTablesFromResult:
    def test_canonical_content_list_priority(self):
        """content_list 含 canonical table block 时走优先路径，输出 HTML。"""
        block = _canonical_table_block("t-canonical", "DATA")
        result = {"content_list": [block]}
        tables = tables_from_result(result)
        assert len(tables) == 1
        assert "DATA" in tables[0]
        assert tables[0].startswith("<table")

    def test_canonical_dedup(self):
        """多个相同 canonical table 去重。"""
        block = _canonical_table_block("t1", "X")
        result = {"content_list": [block, dict(block)]}
        tables = tables_from_result(result)
        assert len(tables) == 1

    def test_legacy_table_body_fallback(self):
        """无 canonical table 时回退到 table_body/html 字段。"""
        block = {"type": "table", "table_body": "<table><tr><td>L</td></tr></table>"}
        result = {"content_list": [block]}
        tables = tables_from_result(result)
        assert tables == ["<table><tr><td>L</td></tr></table>"]

    def test_legacy_html_field_fallback(self):
        block = {"type": "table", "html": "<table><tr><td>H</td></tr></table>"}
        tables = tables_from_result({"content_list": [block]})
        assert len(tables) == 1

    def test_text_blocks_table_label(self):
        """text_blocks 中 label=table 的块从 text 提取表格。"""
        result = {
            "text_blocks": [
                {"label": "table", "text": "<table><tr><td>T</td></tr></table>"}
            ]
        }
        tables = tables_from_result(result)
        assert tables == ["<table><tr><td>T</td></tr></table>"]

    def test_text_blocks_object_attr(self):
        """text_blocks 是对象（非 dict）时用 getattr 读取。"""
        block = SimpleNamespace(
            label="table", text="<table><tr><td>O</td></tr></table>"
        )
        tables = tables_from_result({"text_blocks": [block]})
        assert len(tables) == 1

    def test_raw_text_table_extraction(self):
        """raw_text 字段中残留的裸 <table> 被提取。"""
        result = {"raw_text": "junk<table><tr><td>R</td></tr></table>junk"}
        tables = tables_from_result(result)
        assert tables == ["<table><tr><td>R</td></tr></table>"]

    def test_dedup_across_sources(self):
        """跨来源去重：同一表格 HTML 只出现一次。"""
        frag = "<table><tr><td>D</td></tr></table>"
        result = {
            "content_list": [{"type": "table", "table_body": frag}],
            "raw_text": frag,
        }
        tables = tables_from_result(result)
        assert len(tables) == 1

    def test_empty_result(self):
        assert tables_from_result({}) == []

    def test_non_table_block_ignored(self):
        """content_list 中非 table 类型的块被跳过。"""
        result = {"content_list": [{"type": "text", "text": "hello"}]}
        assert tables_from_result(result) == []

    def test_object_result_with_getattr(self):
        """result 是对象（非 dict）时用 getattr 读取字段。"""
        result = SimpleNamespace(
            content_list=None,
            text_blocks=None,
            html_text="<table><tr><td>OBJ</td></tr></table>",
        )
        tables = tables_from_result(result)
        assert len(tables) == 1

    def test_text_blocks_non_string_text_ignored(self):
        """text_blocks 的 text 非 str 时跳过。"""
        result = {"text_blocks": [{"label": "table", "text": 123}]}
        assert tables_from_result(result) == []
