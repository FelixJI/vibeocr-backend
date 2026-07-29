"""Canonical table-block upgrade and lookup helpers."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from vibeocr.backend.tables.html_adapter import (
    table_model_from_html,
    table_model_to_html,
)
from vibeocr.runtime_contracts.contracts.tables import (
    TableModelV1,
    TableProvenanceV1,
)


def table_model_from_block(
    block: dict[str, Any],
    *,
    fallback_table_id: str = "table",
    strict_canonical: bool = True,
) -> TableModelV1:
    """Read a canonical table, upgrading legacy HTML blocks when necessary."""

    payload = block.get("table")
    if isinstance(payload, dict):
        try:
            return TableModelV1.from_payload(payload)
        except (KeyError, TypeError, ValueError):
            if strict_canonical:
                raise

    html = (
        block.get("table_body")
        or block.get("html")
        or (block.get("source") or {}).get("source_html")
    )
    if not isinstance(html, str) or not html.strip():
        raise ValueError("table block has neither canonical table nor legacy HTML")
    table_id = str(block.get("table_id") or block.get("block_id") or fallback_table_id)
    return table_model_from_html(html, table_id=table_id)


def canonicalize_table_block(
    block: dict[str, Any],
    *,
    table_id: str,
    pipeline: str,
) -> dict[str, Any]:
    """Return a compatible table block with one canonical semantic source."""

    source_html = str(block.get("table_body") or block.get("html") or "")
    has_canonical = isinstance(block.get("table"), dict)
    table = table_model_from_block(block, fallback_table_id=table_id)
    if table.table_id != table_id:
        table = replace(table, table_id=table_id)
    if table.provenance is None:
        table = replace(
            table,
            provenance=TableProvenanceV1(
                pipeline=pipeline,
                provider_schema="canonical-v1" if has_canonical else "legacy-html",
                warnings=() if has_canonical else ("legacy_html_adapted",),
            ),
        )

    upgraded = dict(block)
    upgraded["type"] = "table"
    upgraded.setdefault("block_id", table_id)
    upgraded["table"] = table.to_payload()
    upgraded["table_body"] = table_model_to_html(table)
    source = dict(block.get("source") or {})
    source.setdefault("pipeline", pipeline)
    if source_html:
        source.setdefault("source_html", source_html)
    upgraded["source"] = source
    return upgraded


def validate_table_blocks(content_list: Any) -> None:
    """Validate every canonical table carried by a content-list boundary."""

    if not isinstance(content_list, (list, tuple)):
        return
    for block in content_list:
        if not isinstance(block, dict) or block.get("type") != "table":
            continue
        table = block.get("table")
        if isinstance(table, dict):
            TableModelV1.from_payload(table)


__all__ = [
    "canonicalize_table_block",
    "table_model_from_block",
    "validate_table_blocks",
]
