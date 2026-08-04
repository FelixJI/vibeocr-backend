"""tables.blocks 规范化表格 block 升级与校验的边缘用例测试。

覆盖 table_model_from_block 的 strict_canonical raise、legacy HTML 兜底、
canonicalize_table_block 的 provenance 补全，以及 validate_table_blocks。
"""

from __future__ import annotations

import pytest
from vibeocr.backend.tables.blocks import (
    canonicalize_table_block,
    table_model_from_block,
    validate_table_blocks,
)
from vibeocr.backend.tables.html_adapter import table_model_from_html


class TestTableModelFromBlock:
    def test_canonical_payload(self):
        table = table_model_from_html(
            "<table><tr><td>X</td></tr></table>", table_id="t"
        )
        block = {"type": "table", "table": table.to_payload()}
        result = table_model_from_block(block)
        assert result.row_count == 1

    def test_canonical_invalid_strict_raises(self):
        """canonical payload 非法且 strict_canonical=True 时抛错。"""
        block = {"type": "table", "table": {"invalid": "payload"}}
        with pytest.raises((KeyError, TypeError, ValueError)):
            table_model_from_block(block, strict_canonical=True)

    def test_canonical_invalid_non_strict_falls_back(self):
        """canonical 非法但 strict_canonical=False 时回退到 legacy HTML。"""
        block = {
            "type": "table",
            "table": {"invalid": "payload"},
            "table_body": "<table><tr><td>F</td></tr></table>",
        }
        result = table_model_from_block(block, strict_canonical=False)
        assert result.row_count == 1

    def test_legacy_html_fallback(self):
        block = {"type": "table", "table_body": "<table><tr><td>L</td></tr></table>"}
        result = table_model_from_block(block)
        assert result.row_count == 1

    def test_legacy_html_field(self):
        block = {"type": "table", "html": "<table><tr><td>H</td></tr></table>"}
        result = table_model_from_block(block)
        assert result.row_count == 1

    def test_legacy_source_html(self):
        block = {
            "type": "table",
            "source": {"source_html": "<table><tr><td>S</td></tr></table>"},
        }
        result = table_model_from_block(block)
        assert result.row_count == 1

    def test_no_html_raises(self):
        """既无 canonical table 也无 legacy HTML 时抛 ValueError。"""
        block = {"type": "table"}
        with pytest.raises(ValueError, match="neither canonical"):
            table_model_from_block(block)

    def test_empty_html_raises(self):
        block = {"type": "table", "table_body": "   "}
        with pytest.raises(ValueError, match="neither canonical"):
            table_model_from_block(block)

    def test_table_id_precedence(self):
        """table_id 取 block.table_id > block.block_id > fallback。"""
        block = {
            "type": "table",
            "table_id": "tid",
            "block_id": "bid",
            "table_body": "<table><tr><td>x</td></tr></table>",
        }
        result = table_model_from_block(block, fallback_table_id="fallback")
        assert result.table_id == "tid"

    def test_block_id_fallback(self):
        block = {
            "type": "table",
            "block_id": "bid",
            "table_body": "<table><tr><td>x</td></tr></table>",
        }
        result = table_model_from_block(block, fallback_table_id="fallback")
        assert result.table_id == "bid"

    def test_default_fallback_id(self):
        block = {"type": "table", "table_body": "<table><tr><td>x</td></tr></table>"}
        result = table_model_from_block(block)
        assert result.table_id == "table"


class TestCanonicalizeTableBlock:
    def test_legacy_html_gets_provenance_and_warning(self):
        block = {"type": "table", "table_body": "<table><tr><td>L</td></tr></table>"}
        out = canonicalize_table_block(block, table_id="t1", pipeline="paddle")
        assert out["type"] == "table"
        assert out["block_id"] == "t1"
        assert isinstance(out["table"], dict)
        assert out["table_body"].startswith("<table")
        # provenance 在 canonical table payload 里
        table = out["table"]
        assert table.get("provenance", {}).get("pipeline") == "paddle"
        assert "legacy_html_adapted" in table.get("provenance", {}).get("warnings", [])

    def test_canonical_input_keeps_provider_schema(self):
        table = table_model_from_html(
            "<table><tr><td>C</td></tr></table>", table_id="t1"
        )
        block = {"type": "table", "table": table.to_payload()}
        out = canonicalize_table_block(block, table_id="t1", pipeline="mineru")
        prov = out["table"].get("provenance", {})
        assert prov.get("provider_schema") == "canonical-v1"
        # 反序列化后 warnings 为空序列（list 或 tuple）
        assert not prov.get("warnings")

    def test_overrides_table_id(self):
        block = {
            "type": "table",
            "table_body": "<table><tr><td>x</td></tr></table>",
        }
        out = canonicalize_table_block(block, table_id="forced", pipeline="p")
        assert out["table"]["table_id"] == "forced"

    def test_preserves_source_html(self):
        block = {
            "type": "table",
            "table_body": "<table><tr><td>x</td></tr></table>",
        }
        out = canonicalize_table_block(block, table_id="t", pipeline="p")
        assert out["source"]["source_html"] == "<table><tr><td>x</td></tr></table>"


class TestValidateTableBlocks:
    def test_skips_non_list(self):
        """非 list/tuple 直接返回，不报错。"""
        validate_table_blocks(None)
        validate_table_blocks("not a list")
        validate_table_blocks({"a": 1})

    def test_skips_non_table_blocks(self):
        validate_table_blocks([{"type": "text"}, {"type": "image"}])

    def test_validates_canonical_tables(self):
        table = table_model_from_html(
            "<table><tr><td>x</td></tr></table>", table_id="t"
        )
        validate_table_blocks([{"type": "table", "table": table.to_payload()}])

    def test_invalid_canonical_raises(self):
        with pytest.raises((KeyError, TypeError, ValueError)):
            validate_table_blocks([{"type": "table", "table": {"bad": 1}}])

    def test_skips_non_dict_blocks(self):
        """非 dict 的块被跳过。"""
        validate_table_blocks(["str", 123, None])
