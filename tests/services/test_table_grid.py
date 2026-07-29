"""Tests for HTML table normalization utility.

验证 ``normalize_table_html`` 的 inline style 剥离、空单元格补齐、
标签保留、HTML 实体转义等。``normalize_table_html`` 不依赖 paddle/Qt，
可独立运行。
"""

from __future__ import annotations

import re

from vibeocr.backend.services.ocr_service import normalize_table_html


class TestNormalizeTableHtml:
    """normalize_table_html：剥离 inline style、补齐空单元格、保留标签。

    这是解决"复制带底纹"和"空单元格错位"两个问题的核心。
    """

    def test_strips_inline_style(self) -> None:
        """PaddleX 自带 style 属性应被剥离。"""
        html = (
            '<table><tr><td style="background:#eee;border:1px">A</td>'
            '<th style="color:red">B</th></tr></table>'
        )
        out = normalize_table_html(html)
        assert "style" not in out
        assert "<td>A</td>" in out
        assert "<th>B</th>" in out

    def test_fills_missing_cells_to_rectangular(self) -> None:
        """A1 空、A2 有内容场景：第二行只有一列，应补齐为两列空 td。

        回归：修复前若 HTML 某行单元格数不足，Excel 粘贴会把后续单元格
        前移（A2 内容跑到 A1）。规整后每行列数一致，Excel 列对齐正确。
        """
        # 第一行 2 列，第二行只有 1 列
        html = "<table><tr><th>H1</th><th>H2</th></tr><tr><td>only</td></tr></table>"
        out = normalize_table_html(html)
        # 第二行应补一个空 td
        assert "<tr><td>only</td><td></td></tr>" in out

    def test_preserves_empty_cell_explicitly(self) -> None:
        """空单元格应显式保留为 <td></td>，不能被丢弃。"""
        html = (
            "<table><tr><td></td><td>filled</td></tr>"
            "<tr><td>x</td><td></td></tr></table>"
        )
        out = normalize_table_html(html)
        # 空单元格仍在
        assert out.count("<td></td>") == 2

    def test_preserves_td_th_tags(self) -> None:
        """normalize 保留原 td/th，不强制首行 th。"""
        html = "<table><tr><td>not-header</td><td>also-td</td></tr></table>"
        out = normalize_table_html(html)
        assert "<td>not-header</td>" in out
        assert "<th>" not in out

    def test_strips_style_from_wrapper_html(self) -> None:
        """PaddleX pred_html 外层包 <html><body>，内部 table 带 style。"""
        html = (
            "<html><body><table>"
            '<tr><td style="background:#ccc">A</td></tr>'
            "</table></body></html>"
        )
        out = normalize_table_html(html)
        assert out.startswith("<table>")
        assert "style" not in out
        assert "background" not in out

    def test_html_entities_roundtrip(self) -> None:
        """实体应正确解码后重新转义，不产生双重转义。"""
        html = "<table><tr><td>a&amp;b</td><td>c&lt;d</td></tr></table>"
        out = normalize_table_html(html)
        assert "<td>a&amp;b</td>" in out
        assert "<td>c&lt;d</td>" in out

    def test_all_rows_same_column_count(self) -> None:
        """规整后每行的 td+th 数量必须一致（矩形）。"""
        html = (
            "<table>"
            "<tr><th>A</th><th>B</th><th>C</th></tr>"
            "<tr><td>1</td></tr>"
            "<tr><td>x</td><td>y</td></tr>"
            "</table>"
        )
        out = normalize_table_html(html)
        # 数每行的 td+th 开标签，验证每行都是 3 个
        row_cell_counts = [
            len(re.findall(r"<t[dh]>", tr)) for tr in re.findall(r"<tr>.*?</tr>", out)
        ]
        assert row_cell_counts == [3, 3, 3]

    def test_empty_table(self) -> None:
        assert normalize_table_html("<table></table>") == "<table></table>"
        assert normalize_table_html("") == "<table></table>"

    def test_only_structural_span_attributes_remain(self) -> None:
        """样式属性被剥离，但 colspan/rowspan 结构必须保留。"""
        html = (
            '<table><tr><td class="c" colspan="2" style="bg:1">text</td></tr>'
            '<tr><th id="h" style="font:bold">h</th><td>2</td></tr></table>'
        )
        out = normalize_table_html(html)
        assert 'colspan="2"' in out
        assert "style=" not in out
        assert "class=" not in out
        assert "id=" not in out
