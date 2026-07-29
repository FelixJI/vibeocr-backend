"""Canonical table adapters and projections shared by UI and backend."""

from vibeocr.backend.tables.blocks import (
    canonicalize_table_block,
    table_model_from_block,
    validate_table_blocks,
)
from vibeocr.backend.tables.html_adapter import (
    TableCellSourceSpan,
    TableSourceLayout,
    parse_table_source_layout,
    table_model_from_html,
    table_model_to_html,
)
from vibeocr.backend.tables.projections import (
    MarkdownTableProjection,
    table_model_to_grid,
    table_model_to_markdown,
    table_model_to_plain_text,
    table_model_to_tsv,
)
from vibeocr.backend.tables.reducer import (
    build_result_projections,
    rebuild_result_projections,
    update_result_table_cell,
    update_table_cell,
)

__all__ = [
    "MarkdownTableProjection",
    "TableCellSourceSpan",
    "TableSourceLayout",
    "build_result_projections",
    "canonicalize_table_block",
    "parse_table_source_layout",
    "rebuild_result_projections",
    "table_model_from_block",
    "table_model_from_html",
    "table_model_to_grid",
    "table_model_to_html",
    "table_model_to_markdown",
    "table_model_to_plain_text",
    "table_model_to_tsv",
    "update_result_table_cell",
    "update_table_cell",
    "validate_table_blocks",
]
