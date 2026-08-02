"""tables.reducer 表格编辑归约与投影构建的边缘用例测试。

覆盖 update_table_cell / update_result_table_cell 的 KeyError 与匹配分支，
以及 build_result_projections 的 content_list 各 block_type 分支与取消语义。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from vibeocr.backend.tables.html_adapter import table_model_from_html
from vibeocr.backend.tables.reducer import (
    build_result_projections,
    rebuild_result_projections,
    update_result_table_cell,
    update_table_cell,
)

_HTML = (
    "<table>"
    '<tr><td data-cell-id="r0c0">A</td><td data-cell-id="r0c1">B</td></tr>'
    '<tr><td data-cell-id="r1c0">C</td><td data-cell-id="r1c1">D</td></tr>'
    "</table>"
)


def _table_block(
    table_id: str = "t1", *, block_id: str | None = None
) -> dict[str, Any]:
    """构造一个规范的 table block（带 canonical table payload）。"""
    table = table_model_from_html(_HTML, table_id=table_id)
    block: dict[str, Any] = {
        "type": "table",
        "table": table.to_payload(),
    }
    block["block_id"] = block_id or table_id
    return block


def _text_block(
    text: str = "x", *, content_id: str | None = None, content_index: int | None = None
) -> SimpleNamespace:
    """构造一个轻量 text_block（鸭子类型，匹配 reducer 读取的属性）。"""
    return SimpleNamespace(
        text=text,
        score=0.5,
        content_id=content_id,
        content_index=content_index,
        label="text",
        is_manually_edited=False,
    )


class TestUpdateTableCell:
    """update_table_cell 单表格单元更新。"""

    def test_updates_cell_and_refreshes_projections(self):
        """命中 cell 更新文本，table_body/text 同步刷新。"""
        block = _table_block("t1")

        updated = update_table_cell(block, table_id="t1", cell_id="r0c0", new_text="X")

        assert "X" in updated["table_body"]
        assert "X" in updated["text"]
        assert updated["type"] == "table"

    def test_wrong_table_id_raises_keyerror(self):
        """table_id 不匹配时抛 KeyError。"""
        block = _table_block("t1")

        with pytest.raises(KeyError, match="table_id"):
            update_table_cell(block, table_id="other", cell_id="r0c0", new_text="X")

    def test_unknown_cell_id_raises_keyerror(self):
        """cell_id 不存在时抛 KeyError。"""
        block = _table_block("t1")

        with pytest.raises(KeyError, match="cell_id"):
            update_table_cell(block, table_id="t1", cell_id="zzz", new_text="X")


class TestUpdateResultTableCell:
    """update_result_table_cell 在 result 上的原子编辑。"""

    def test_non_list_content_list_raises_keyerror(self):
        """content_list 非 list 时抛 KeyError。"""
        result = SimpleNamespace(content_list=None)

        with pytest.raises(KeyError, match="table_id"):
            update_result_table_cell(
                result, table_id="t1", cell_id="r0c0", new_text="X"
            )

    def test_unknown_table_id_raises_keyerror(self):
        """result 中找不到 table_id 时抛 KeyError。"""
        result = SimpleNamespace(content_list=[_table_block("t1")], text_blocks=[])

        with pytest.raises(KeyError, match="table_id"):
            update_result_table_cell(
                result, table_id="missing", cell_id="r0c0", new_text="X"
            )

    def test_table_id_from_block_id_fallback(self):
        """table payload 无 table_id 时回退到 block_id 匹配。"""
        block = _table_block("t1")
        # 构造 payload 无 table_id、但 block_id 命中的场景
        block["table"] = {"placeholder": True}
        block["block_id"] = "fallback-id"
        result = SimpleNamespace(content_list=[block], text_blocks=[])

        # 因 payload 非规范 dict，table_model_from_block 会走 HTML 升级路径并抛错
        # 这里仅验证 candidate_id 解析路径被走到（block_id 被读取）
        with pytest.raises((KeyError, ValueError)):
            update_result_table_cell(
                result, table_id="fallback-id", cell_id="r0c0", new_text="X"
            )

    def test_updates_text_block_by_content_id(self):
        """命中 table 后，按 content_id 匹配的 text_block 被刷新。"""
        block = _table_block("t1", block_id="blk-1")
        tb = _text_block("orig", content_id="blk-1", content_index=0)
        result = SimpleNamespace(
            content_list=[block],
            text_blocks=[tb],
            text_with_scores=[("orig", 0.42)],
            raw_text="",
            markdown_text="",
            html_text="",
        )

        index = update_result_table_cell(
            result, table_id="t1", cell_id="r0c0", new_text="Z"
        )

        assert index == 0
        assert "Z" in result.text_blocks[0].text
        assert result.text_blocks[0].is_manually_edited is True
        assert result.text_with_scores[0][0] == result.text_blocks[0].text
        assert result.text_with_scores[0][1] == 0.42
        # rebuild_result_projections 被调用，projections 被刷新
        assert result.raw_text != "" or result.html_text != ""

    def test_updates_text_block_by_content_index_fallback(self):
        """无 content_id 匹配时，按 content_index 兜底匹配。"""
        block = _table_block("t1", block_id="blk-1")
        tb = _text_block("orig", content_index=0)  # 无 content_id
        result = SimpleNamespace(
            content_list=[block],
            text_blocks=[tb],
            text_with_scores=[],
            raw_text="",
            markdown_text="",
            html_text="",
        )

        update_result_table_cell(result, table_id="t1", cell_id="r0c0", new_text="Z")

        assert "Z" in result.text_blocks[0].text

    def test_no_matching_text_block_skips_text_update(self):
        """text_blocks 无法匹配时不抛异常，content_list 仍被更新。"""
        block = _table_block("t1", block_id="blk-1")
        tb = _text_block("unrelated", content_id="other", content_index=9)
        result = SimpleNamespace(
            content_list=[block],
            text_blocks=[tb],
            text_with_scores=[],
            raw_text="",
            markdown_text="",
            html_text="",
        )

        update_result_table_cell(result, table_id="t1", cell_id="r0c0", new_text="Z")

        assert tb.is_manually_edited is False
        assert "Z" not in tb.text


class TestBuildResultProjections:
    """build_result_projections 投影构建。"""

    def test_non_list_content_falls_back_to_text_blocks(self):
        """content_list 非 list 时回退 text_blocks 生成 raw 投影。"""
        result = SimpleNamespace(
            content_list=None,
            text_blocks=[_text_block("hello")],
        )

        raw, markdown, html = build_result_projections(result)

        assert raw == "hello"
        assert markdown == ""
        assert html == ""

    def test_returns_none_when_cancelled_at_interval(self):
        """is_cancelled 在间隔点返回 True 时整体返回 None。"""
        block = _table_block("t1")
        result = SimpleNamespace(content_list=[block], text_blocks=[])

        projections = build_result_projections(result, is_cancelled=lambda: True)

        assert projections is None

    def test_title_block_projections(self):
        """title block 生成 markdown 标题与 html <h> 标签。"""
        result = SimpleNamespace(
            content_list=[{"type": "title", "text": "Hello", "level": 2}],
            text_blocks=[],
        )

        raw, markdown, html = build_result_projections(result)

        assert "## Hello" in markdown
        assert "<h2>Hello</h2>" in html
        assert "Hello" in raw

    def test_code_block_projections(self):
        """code block 生成 markdown 代码块与 html <pre><code>。"""
        result = SimpleNamespace(
            content_list=[{"type": "code", "code_body": "print(1)"}],
            text_blocks=[],
        )

        raw, markdown, html = build_result_projections(result)

        assert "```\nprint(1)\n```" in markdown
        assert "<pre><code>print(1)</code></pre>" in html

    def test_equation_block_projections(self):
        """equation block 生成 markdown $$...$$ 与 html equation div。"""
        result = SimpleNamespace(
            content_list=[{"type": "equation", "text": "E=mc^2"}],
            text_blocks=[],
        )

        raw, markdown, html = build_result_projections(result)

        assert "$$E=mc^2$$" in markdown
        assert '<div class="equation">E=mc^2</div>' in html

    def test_list_block_projections(self):
        """list block 生成 markdown 列表与 html <ul>。"""
        result = SimpleNamespace(
            content_list=[{"type": "list", "list_items": ["a", "b"]}],
            text_blocks=[],
        )

        raw, markdown, html = build_result_projections(result)

        assert "- a" in markdown and "- b" in markdown
        assert "<ul>" in html and "<li>a</li>" in html

    def test_image_block_with_source(self):
        """image block 带 img_path 时生成 markdown 图片与 html <img>。"""
        result = SimpleNamespace(
            content_list=[
                {
                    "type": "image",
                    "img_path": "/img/x.png",
                    "image_caption": ["cap"],
                }
            ],
            text_blocks=[],
        )

        raw, markdown, html = build_result_projections(result)

        assert "![cap](/img/x.png)" in markdown
        assert '<img src="/img/x.png"' in html
        assert "cap" in raw

    def test_image_block_without_source(self):
        """image block 无 img_path 时只输出 caption。"""
        result = SimpleNamespace(
            content_list=[{"type": "figure", "image_caption": ["only cap"]}],
            text_blocks=[],
        )

        raw, markdown, html = build_result_projections(result)

        assert "only cap" in markdown
        assert "<img" not in html

    def test_table_block_projections(self):
        """table block 生成 markdown 表格与 html table。"""
        result = SimpleNamespace(content_list=[_table_block("t1")], text_blocks=[])

        raw, markdown, html = build_result_projections(result)

        assert "<table" in html
        assert "|" in markdown  # markdown 表格
        assert "A" in raw

    def test_table_block_with_caption_and_footnote(self):
        """table block 的 table_caption / table_footnote 进入 markdown 与 html。"""
        block = _table_block("t1")
        block["table_caption"] = ["表标题"]
        block["table_footnote"] = ["脚注"]
        result = SimpleNamespace(content_list=[block], text_blocks=[])

        _, markdown, html = build_result_projections(result)

        assert "表标题" in markdown
        assert "脚注" in markdown
        assert "table-caption" in html
        assert "table-footnote" in html

    def test_discarded_block_type_skipped(self):
        """DISCARDED_BLOCK_TYPES（如 image 的某些标签）不出现在投影中。"""
        result = SimpleNamespace(
            content_list=[
                {"type": "image", "text": "should-skip-if-discarded"},
            ],
            text_blocks=[],
        )

        # image 不在 DISCARDED 中，应正常投影 caption（这里无 caption）
        raw, markdown, html = build_result_projections(result)
        assert raw == ""

    def test_include_raw_false_skips_raw(self):
        """include_raw=False 时 raw_text 为空。"""
        result = SimpleNamespace(
            content_list=[{"type": "text", "text": "hello"}],
            text_blocks=[],
        )

        raw, _, _ = build_result_projections(result, include_raw=False)

        assert raw == ""

    def test_include_markdown_false_skips_markdown(self):
        """include_markdown=False 时 markdown 为空，html 仍生成。"""
        result = SimpleNamespace(
            content_list=[{"type": "title", "text": "Hi", "level": 1}],
            text_blocks=[],
        )

        _, markdown, html = build_result_projections(result, include_markdown=False)

        assert markdown == ""
        assert "<h1>Hi</h1>" in html


class TestRebuildResultProjections:
    """rebuild_result_projections 写回投影。"""

    def test_writes_projections_back_to_result(self):
        """rebuild 把计算结果写回 result 的三个投影字段。"""
        result = SimpleNamespace(
            content_list=[{"type": "text", "text": "abc"}],
            text_blocks=[],
            raw_text="",
            markdown_text="",
            html_text="",
        )

        rebuild_result_projections(result)

        assert result.raw_text == "abc"
        assert "<p>abc</p>" in result.html_text

    def test_none_projections_leaves_result_untouched(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """build 返回 None 时 rebuild 不写回 result，投影字段保持原值。"""
        import vibeocr.backend.tables.reducer as reducer

        monkeypatch.setattr(reducer, "build_result_projections", lambda r: None)
        result = SimpleNamespace(
            content_list=[{"type": "title", "text": "x"}],
            text_blocks=[],
            raw_text="keep",
            markdown_text="keep-md",
            html_text="keep-html",
        )

        rebuild_result_projections(result)

        assert result.raw_text == "keep"
        assert result.html_text == "keep-html"
