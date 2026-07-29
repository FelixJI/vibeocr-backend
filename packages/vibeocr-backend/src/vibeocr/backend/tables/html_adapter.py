"""Structured HTML adapter for the canonical table contract."""

from __future__ import annotations

import html
from dataclasses import dataclass
from html.parser import HTMLParser

from vibeocr.runtime_contracts.contracts.tables import (
    MAX_TABLE_CELLS,
    MAX_TABLE_COVERAGE,
    MAX_TABLE_DIMENSION,
    TableCellV1,
    TableModelV1,
)

MAX_HTML_TABLE_TEXT_CHARS = 10_000_000
MAX_HTML_TABLE_SOURCE_CHARS = 20_000_000


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[tuple[str, dict[str, str], str]]] = []
        self._table_depth = 0
        self.found_table = False
        self.completed_table = False
        self.nested_table = False
        self.multiple_tables = False
        self._row: list[tuple[str, dict[str, str], str]] | None = None
        self._cell_tag: str | None = None
        self._cell_attrs: dict[str, str] = {}
        self._cell_text: list[str] = []
        self._cell_count = 0
        self._coverage = 0
        self._text_chars = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lowered = tag.lower()
        if self.completed_table:
            if lowered == "table":
                self.multiple_tables = True
            return
        if lowered == "table":
            if not self.found_table:
                self.found_table = True
            elif self._table_depth >= 1:
                self.nested_table = True
            self._table_depth += 1
            return
        if self._table_depth != 1:
            return
        if lowered == "tr":
            self._close_cell()
            self._close_row()
            self._row = []
        elif lowered in {"td", "th"} and self._row is not None:
            self._close_cell()
            self._cell_tag = lowered
            self._cell_attrs = {name.lower(): value or "" for name, value in attrs}
            self._cell_text = []
        elif lowered == "br" and self._cell_tag is not None:
            self._append_cell_text("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag.lower() in {"td", "th"}:
            self._close_cell()
        elif tag.lower() == "table":
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if self.completed_table:
            return
        if lowered == "table":
            if self._table_depth == 1:
                self._close_cell()
                self._close_row()
                self.completed_table = True
            self._table_depth = max(0, self._table_depth - 1)
            return
        if self._table_depth != 1:
            return
        if lowered in {"td", "th"} and self._cell_tag is not None:
            self._close_cell()
        elif lowered == "tr" and self._row is not None:
            self._close_cell()
            self._close_row()

    def handle_data(self, data: str) -> None:
        if self._table_depth == 1 and self._cell_tag is not None:
            self._append_cell_text(data)

    def _append_cell_text(self, value: str) -> None:
        self._text_chars += len(value)
        if self._text_chars > MAX_HTML_TABLE_TEXT_CHARS:
            raise ValueError("HTML table text exceeds supported limit")
        self._cell_text.append(value)

    def _close_cell(self) -> None:
        if self._cell_tag is None or self._row is None:
            return
        self._cell_count += 1
        if self._cell_count > MAX_TABLE_CELLS:
            raise ValueError("HTML table cell count exceeds supported limit")
        self._coverage += _span(self._cell_attrs, "rowspan") * _span(
            self._cell_attrs, "colspan"
        )
        if self._coverage > MAX_TABLE_COVERAGE:
            raise ValueError("HTML table cell coverage exceeds supported limit")
        self._row.append(
            (
                self._cell_tag,
                self._cell_attrs,
                "".join(self._cell_text),
            )
        )
        self._cell_tag = None
        self._cell_attrs = {}
        self._cell_text = []

    def _close_row(self) -> None:
        if self._row is None:
            return
        if len(self.rows) >= MAX_TABLE_DIMENSION:
            raise ValueError("HTML table row count exceeds supported limit")
        self.rows.append(self._row)
        self._row = None


def _span(attrs: dict[str, str], name: str) -> int:
    raw = attrs.get(name)
    if raw is None:
        return 1
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"invalid HTML table {name}: {raw!r}") from error
    if value < 1 or value > MAX_TABLE_DIMENSION:
        raise ValueError(f"invalid HTML table {name}: {raw!r}")
    return value


def table_model_from_html(html_text: str, *, table_id: str) -> TableModelV1:
    """Parse the first HTML table into canonical logical coordinates."""

    if len(html_text) > MAX_HTML_TABLE_SOURCE_CHARS:
        raise ValueError("HTML table source exceeds supported limit")
    parser = _TableParser()
    parser.feed(html_text)
    if not parser.found_table:
        raise ValueError("HTML does not contain a table")
    if parser.nested_table:
        raise ValueError("nested HTML tables are not supported")
    if parser.multiple_tables:
        raise ValueError("multiple top-level HTML tables require separate blocks")
    if not parser.completed_table:
        raise ValueError("HTML table is not closed")
    occupied: set[tuple[int, int]] = set()
    cells: list[TableCellV1] = []
    row_count = 0
    column_count = 0
    coverage = 0

    for row_index, raw_row in enumerate(parser.rows):
        column = 0
        row_count = max(row_count, row_index + 1)
        for tag, attrs, text in raw_row:
            while (row_index, column) in occupied:
                column += 1
            rowspan = _span(attrs, "rowspan")
            colspan = _span(attrs, "colspan")
            coverage += rowspan * colspan
            if coverage > MAX_TABLE_COVERAGE:
                raise ValueError("HTML table cell coverage exceeds supported limit")
            cells.append(
                TableCellV1(
                    cell_id=attrs.get("data-cell-id") or f"r{row_index}c{column}",
                    row=row_index,
                    column=column,
                    rowspan=rowspan,
                    colspan=colspan,
                    text=text,
                    is_header=tag == "th",
                )
            )
            for covered_row in range(row_index, row_index + rowspan):
                for covered_column in range(column, column + colspan):
                    occupied.add((covered_row, covered_column))
            row_count = max(row_count, row_index + rowspan)
            column_count = max(column_count, column + colspan)
            column += colspan

    return TableModelV1(
        table_id=table_id,
        row_count=row_count,
        column_count=column_count,
        cells=tuple(cells),
    )


@dataclass(frozen=True, slots=True)
class TableCellSourceSpan:
    """Original HTML offsets for one logical anchor cell's inner content."""

    content_start: int
    content_end: int
    source_text: str


@dataclass(frozen=True, slots=True)
class TableSourceLayout:
    """Canonical layout paired with source spans in canonical cell order."""

    model: TableModelV1
    cells: tuple[TableCellSourceSpan, ...]


class _TableSourceSpanParser(HTMLParser):
    def __init__(self, source: str) -> None:
        super().__init__(convert_charrefs=False)
        self._source = source
        self._offset_line = 1
        self._line_start = 0
        self._table_depth = 0
        self._completed_table = False
        self._cell_start: int | None = None
        self.spans: list[tuple[int, int]] = []

    def _offset(self) -> int:
        line, column = self.getpos()
        while self._offset_line < line:
            newline = self._source.find("\n", self._line_start)
            if newline < 0:
                raise ValueError("HTML parser position exceeds source text")
            self._line_start = newline + 1
            self._offset_line += 1
        return self._line_start + column

    def _close_cell(self, end: int) -> None:
        if self._cell_start is None:
            return
        self.spans.append((self._cell_start, max(self._cell_start, end)))
        self._cell_start = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        lowered = tag.lower()
        if self._completed_table:
            return
        position = self._offset()
        if lowered == "table":
            self._table_depth += 1
            return
        if self._table_depth != 1:
            return
        if lowered in {"tr", "td", "th"}:
            self._close_cell(position)
        if lowered in {"td", "th"}:
            raw_tag = self.get_starttag_text() or ""
            self._cell_start = position + len(raw_tag)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        position = self._offset()
        self.handle_starttag(tag, attrs)
        if tag.lower() in {"td", "th"}:
            raw_tag = self.get_starttag_text() or ""
            self._close_cell(position + len(raw_tag))
        elif tag.lower() == "table":
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if self._completed_table:
            return
        position = self._offset()
        if lowered == "table":
            if self._table_depth == 1:
                self._close_cell(position)
                self._completed_table = True
            self._table_depth = max(0, self._table_depth - 1)
            return
        if self._table_depth == 1 and lowered in {"td", "th", "tr"}:
            self._close_cell(position)


def parse_table_source_layout(html_text: str, *, table_id: str) -> TableSourceLayout:
    """Parse canonical layout and safe inner-content offsets without regex layout."""

    if len(html_text) > MAX_HTML_TABLE_SOURCE_CHARS:
        raise ValueError("HTML table source exceeds supported limit")
    model = table_model_from_html(html_text, table_id=table_id)
    parser = _TableSourceSpanParser(html_text)
    parser.feed(html_text)
    if len(parser.spans) != len(model.cells):
        raise ValueError(
            "HTML table source span count does not match canonical cell count"
        )
    spans = tuple(
        TableCellSourceSpan(
            content_start=start,
            content_end=end,
            source_text=cell.text,
        )
        for (start, end), cell in zip(parser.spans, model.cells, strict=True)
    )
    return TableSourceLayout(model=model, cells=spans)


def table_model_to_html(table: TableModelV1) -> str:
    """Render canonical anchor cells as a compact HTML table."""

    cells_by_row: dict[int, list[TableCellV1]] = {}
    for cell in table.cells:
        cells_by_row.setdefault(cell.row, []).append(cell)

    rendered_rows: list[str] = []
    for row_index in range(table.row_count):
        rendered_cells: list[str] = []
        for cell in sorted(
            cells_by_row.get(row_index, ()), key=lambda item: item.column
        ):
            tag = "th" if cell.is_header else "td"
            attrs = f' data-cell-id="{html.escape(cell.cell_id, quote=True)}"'
            if cell.rowspan > 1:
                attrs += f' rowspan="{cell.rowspan}"'
            if cell.colspan > 1:
                attrs += f' colspan="{cell.colspan}"'
            text = html.escape(cell.text).replace("\n", "<br>")
            rendered_cells.append(f"<{tag}{attrs}>{text}</{tag}>")
        rendered_rows.append(f"<tr>{''.join(rendered_cells)}</tr>")
    table_id = html.escape(table.table_id, quote=True)
    return f'<table data-table-id="{table_id}">{"".join(rendered_rows)}</table>'


__all__ = [
    "TableCellSourceSpan",
    "TableSourceLayout",
    "parse_table_source_layout",
    "table_model_from_html",
    "table_model_to_html",
]
