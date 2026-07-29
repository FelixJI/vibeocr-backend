"""Deterministic projections of canonical table semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vibeocr.runtime_contracts.contracts.tables import TableModelV1


def table_model_to_grid(table: TableModelV1) -> list[list[str]]:
    """Project anchor text onto a dense grid; covered positions remain empty."""

    grid = [["" for _ in range(table.column_count)] for _ in range(table.row_count)]
    for cell in table.cells:
        grid[cell.row][cell.column] = cell.text
    return grid


def table_model_to_plain_text(table: TableModelV1) -> str:
    """Project anchor cells to searchable text without embedding HTML."""

    cells_by_row: dict[int, list[tuple[int, str]]] = {}
    for cell in table.cells:
        cells_by_row.setdefault(cell.row, []).append((cell.column, cell.text))
    return "\n".join(
        "\t".join(
            text
            for _column, text in sorted(
                cells_by_row.get(row_index, ()), key=lambda item: item[0]
            )
        )
        for row_index in range(table.row_count)
    )


def table_model_to_tsv(table: TableModelV1) -> str:
    """Project the full logical grid, including placeholders covered by spans."""

    return "\n".join("\t".join(row) for row in table_model_to_grid(table))


@dataclass(frozen=True, slots=True)
class MarkdownTableProjection:
    text: str
    warnings: tuple[str, ...] = ()


def table_model_to_markdown(table: TableModelV1) -> MarkdownTableProjection:
    """Create a GFM projection and disclose merge-semantics loss."""

    grid = table_model_to_grid(table)
    if not grid:
        return MarkdownTableProjection(text="")

    def escape(value: str) -> str:
        return value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")

    rows = ["| " + " | ".join(escape(value) for value in row) + " |" for row in grid]
    separator = "| " + " | ".join("---" for _ in range(table.column_count)) + " |"
    rows.insert(1, separator)
    warnings = ("lossy_markdown_source",) if table.merged_ranges() else ()
    return MarkdownTableProjection(text="\n".join(rows), warnings=warnings)


__all__ = [
    "MarkdownTableProjection",
    "table_model_to_grid",
    "table_model_to_markdown",
    "table_model_to_plain_text",
    "table_model_to_tsv",
]
