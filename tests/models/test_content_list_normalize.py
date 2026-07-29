"""content_list 正常化层测试"""

from vibeocr.backend.models.ocr_result import normalize_content_list


class TestNormalizeLegacyFormat:
    def test_legacy_passthrough(self):
        raw = [
            {"type": "text", "text": "Hello", "bbox": [10, 20, 100, 50], "page_idx": 0},
        ]
        result = normalize_content_list(raw)
        assert len(result) == 1
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "Hello"
        assert result[0]["raw"] == raw[0]

    def test_empty_list(self):
        assert normalize_content_list([]) == []

    def test_legacy_table_type(self):
        raw = [
            {
                "type": "table",
                "table_body": "<table></table>",
                "bbox": [0, 0, 500, 200],
                "page_idx": 0,
            },
        ]
        result = normalize_content_list(raw)
        assert result[0]["type"] == "table"


class TestNormalizeV2Format:
    def test_v2_title_becomes_title(self):
        raw = [
            [
                {
                    "type": "title",
                    "content": {
                        "title_content": [{"type": "text", "content": "1 Intro"}],
                        "level": 1,
                    },
                    "bbox": [83, 121, 917, 156],
                },
            ],
        ]
        result = normalize_content_list(raw)
        assert len(result) == 1
        assert result[0]["type"] == "title"
        assert result[0]["text"] == "1 Intro"
        assert result[0]["page_idx"] == 0

    def test_v2_paragraph_becomes_text(self):
        raw = [
            [
                {
                    "type": "paragraph",
                    "content": {
                        "paragraph_content": [{"type": "text", "content": "Body text"}]
                    },
                    "bbox": [83, 200, 917, 300],
                },
            ],
        ]
        result = normalize_content_list(raw)
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "Body text"

    def test_v2_equation_interline(self):
        raw = [
            [
                {
                    "type": "equation_interline",
                    "content": {"math_content": "E=mc^2", "math_type": "latex"},
                    "bbox": [100, 400, 900, 450],
                },
            ],
        ]
        result = normalize_content_list(raw)
        assert result[0]["type"] == "equation"
        assert result[0]["text"] == "E=mc^2"

    def test_v2_page_auxiliary_types(self):
        raw = [
            [
                {
                    "type": "page_header",
                    "content": {
                        "page_header_content": [{"type": "text", "content": "H"}]
                    },
                    "bbox": [0, 0, 100, 30],
                },
                {
                    "type": "page_footer",
                    "content": {
                        "page_footer_content": [{"type": "text", "content": "F"}]
                    },
                    "bbox": [0, 950, 100, 990],
                },
                {
                    "type": "page_aside_text",
                    "content": {
                        "page_aside_text_content": [{"type": "text", "content": "A"}]
                    },
                    "bbox": [0, 100, 90, 200],
                },
                {
                    "type": "page_footnote",
                    "content": {
                        "page_footnote_content": [{"type": "text", "content": "FN"}]
                    },
                    "bbox": [0, 900, 100, 940],
                },
                {
                    "type": "page_number",
                    "content": {
                        "page_number_content": [{"type": "text", "content": "1"}]
                    },
                    "bbox": [450, 980, 550, 999],
                },
            ],
        ]
        result = normalize_content_list(raw)
        types = [r["type"] for r in result]
        assert types == [
            "header",
            "footer",
            "aside_text",
            "page_footnote",
            "page_number",
        ]

    def test_v2_multi_page(self):
        raw = [
            [
                {
                    "type": "title",
                    "content": {
                        "title_content": [{"type": "text", "content": "P0"}],
                        "level": 1,
                    },
                    "bbox": [0, 0, 100, 30],
                }
            ],
            [
                {
                    "type": "title",
                    "content": {
                        "title_content": [{"type": "text", "content": "P1"}],
                        "level": 1,
                    },
                    "bbox": [0, 0, 100, 30],
                }
            ],
        ]
        result = normalize_content_list(raw)
        assert len(result) == 2
        assert result[0]["page_idx"] == 0
        assert result[1]["page_idx"] == 1
        assert result[0]["text"] == "P0"
        assert result[1]["text"] == "P1"

    def test_v2_preserves_raw(self):
        raw = [
            [
                {
                    "type": "title",
                    "content": {
                        "title_content": [{"type": "text", "content": "X"}],
                        "level": 2,
                    },
                    "bbox": [0, 0, 100, 30],
                },
            ],
        ]
        result = normalize_content_list(raw)
        assert result[0]["raw"]["content"]["level"] == 2


class TestNormalizeEdgeCases:
    def test_none_returns_empty(self):
        assert normalize_content_list(None) == []

    def test_mixed_v2_types_text_extraction(self):
        raw = [
            [
                {
                    "type": "title",
                    "content": {
                        "title_content": [
                            {"type": "text", "content": "Chapter "},
                            {"type": "text", "content": "1"},
                        ],
                        "level": 1,
                    },
                    "bbox": [0, 0, 100, 30],
                },
            ],
        ]
        result = normalize_content_list(raw)
        assert result[0]["text"] == "Chapter 1"


class TestExtractV2TextBranches:
    """覆盖 _extract_v2_text 的 math_content / list_items / code_body / text 回退分支。"""

    def test_v2_math_content_string(self):
        raw = [[{"type": "equation_interline", "content": {"math_content": "a^2+b^2"}, "bbox": [0, 0, 10, 10]}]]
        result = normalize_content_list(raw)
        assert result[0]["text"] == "a^2+b^2"

    def test_v2_list_items(self):
        raw = [[{"type": "list", "content": {"list_items": ["one", "two"]}, "bbox": [0, 0, 10, 10]}]]
        result = normalize_content_list(raw)
        assert result[0]["text"] == "one two"

    def test_v2_code_body(self):
        raw = [[{"type": "code", "content": {"code_body": "print(1)"}, "bbox": [0, 0, 10, 10]}]]
        result = normalize_content_list(raw)
        assert result[0]["text"] == "print(1)"

    def test_v2_text_fallback_when_no_content(self):
        """content 无 *_content 字段时回退到 block['text']。"""
        raw = [[{"type": "title", "text": "Fallback Title", "bbox": [0, 0, 10, 10]}]]
        result = normalize_content_list(raw)
        assert result[0]["text"] == "Fallback Title"


class TestExtractLegacyTextBranches:
    """覆盖 _extract_legacy_text 的 image / list / code / table-caption 分支。"""

    def test_legacy_image_with_caption_and_content(self):
        raw = [{"type": "image", "image_caption": ["Fig 1"], "content": "desc"}]
        result = normalize_content_list(raw)
        assert "Fig 1" in result[0]["text"]
        assert "desc" in result[0]["text"]

    def test_legacy_image_without_any_text(self):
        raw = [{"type": "image"}]
        result = normalize_content_list(raw)
        assert result[0]["text"] == "[image]"

    def test_legacy_chart_uses_chart_caption(self):
        raw = [{"type": "chart", "chart_caption": ["Chart A"]}]
        result = normalize_content_list(raw)
        assert "Chart A" in result[0]["text"]

    def test_legacy_list_joins_items(self):
        raw = [{"type": "list", "list_items": ["苹果", "香蕉"]}]
        result = normalize_content_list(raw)
        assert result[0]["text"] == "苹果; 香蕉"

    def test_legacy_code_truncates(self):
        long_body = "x" * 500
        raw = [{"type": "code", "code_body": long_body}]
        result = normalize_content_list(raw)
        assert len(result[0]["text"]) == 200

    def test_legacy_table_caption_and_body(self):
        raw = [{"type": "table", "table_caption": ["T1"], "table_body": "<table><tr><td>A</td></tr></table>"}]
        result = normalize_content_list(raw)
        assert result[0]["text"].startswith("T1")
        assert "A" in result[0]["text"]


def test_v2_content_list_with_non_dict_item_is_skipped():
    """title_content list 含非 dict 元素时跳过该元素（line 124->123 partial）。"""
    raw = [[{
        "type": "title",
        "content": {"title_content": ["str-item", {"type": "text", "content": "Valid"}]},
        "bbox": [0, 0, 10, 10],
    }]]
    result = normalize_content_list(raw)
    assert result[0]["text"] == "Valid"
