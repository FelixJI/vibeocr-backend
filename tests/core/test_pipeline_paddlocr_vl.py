# tests/core/test_pipeline_paddlocr_vl.py
"""PaddleOCR-VL 管道解析回归测试。

回归背景：PaddleX 结果对象是 dict 子类，parsing_res_list/content_list/images
是 dict key 而非实例属性。早期代码用 hasattr(res, ...) + res.xxx 属性访问，
对 dict 子类 hasattr 恒为 False，导致 VL 识别丢块（与表格/公式同一类 bug）。
"""

from vibeocr.backend.core.pipelines.pipeline_paddlocr_vl import (
    PADDLEOCR_VL_SPEC,
    PaddleOCRVLOptions,
    _recognize_paddlocr_vl,
)


class _DictResult(dict):
    """模拟 PaddleX 结果：dict 子类，键需下标访问。"""


class _FakePipeline:
    def __init__(self, result_list):
        self._result_list = result_list

    def predict(self, input, **kwargs):  # noqa: A002 — 模拟 PaddleOCR API（input 关键字参数）
        return list(self._result_list)


class _FakeService:
    def __init__(self, result_list):
        self._pipeline = _FakePipeline(result_list)

    def get_or_create_pipeline(self, name):
        return self._pipeline


def test_vl_options_defaults():
    opts = PaddleOCRVLOptions()
    assert opts.pipeline == "PaddleOCR-VL"


def test_vl_spec():
    assert PADDLEOCR_VL_SPEC.name == "PaddleOCR-VL"
    assert PADDLEOCR_VL_SPEC.display_name == "文档P（PaddleOCR-VL）"


def test_recognize_vl_extracts_blocks_from_dict_result():
    """回归：parsing_res_list 必须用下标访问（dict 子类）。

    修复前 hasattr(res, "parsing_res_list") 对 dict 子类恒为 False，
    VL 识别所有块被丢弃。
    """
    res = _DictResult(
        {
            "parsing_res_list": [
                {
                    "block_bbox": [10, 20, 100, 50],
                    "block_content": "hello world",
                    "block_label": "text",
                    "block_order": 0,
                },
                {
                    "block_bbox": [10, 60, 100, 90],
                    "block_content": "",  # 空内容应被跳过
                    "block_label": "figure",
                    "block_order": 1,
                },
            ]
        }
    )
    service = _FakeService([res])
    result = _recognize_paddlocr_vl(service, image=None, options=PaddleOCRVLOptions())

    assert result.pipeline_type == "PaddleOCR-VL"
    assert len(result.text_blocks) == 1
    assert result.text_blocks[0].text == "hello world"
    assert len(result.content_list) == 1
    assert result.content_list[0]["block_id"] == result.text_blocks[0].content_id


def test_recognize_vl_extracts_content_list_and_images():
    """content_list / images 同样是 dict key，必须下标访问。"""
    res = _DictResult(
        {
            "parsing_res_list": [],
            "content_list": [{"type": "text", "text": "from cl"}],
            "images": {"img1": b"\x89PNG"},
        }
    )
    service = _FakeService([res])
    result = _recognize_paddlocr_vl(service, image=None, options=PaddleOCRVLOptions())

    content = next(c for c in result.content_list if c.get("text") == "from cl")
    text_block = next(b for b in result.text_blocks if b.text == "from cl")
    assert content["block_id"] == text_block.content_id
    assert result.images and "img1" in result.images


class _VLBlock:
    """模拟真实 PaddleOCRVLBlock（普通对象，属性访问，非 dict）。

    真实属性：content/label/bbox/global_block_id。早期代码误用
    block.get("block_content")，对非 dict 对象会 AttributeError。
    """

    def __init__(self, content, label="text", bbox=None, global_block_id=-1):
        self.content = content
        self.label = label
        self.bbox = bbox or [10, 20, 100, 50]
        self.global_block_id = global_block_id


def test_recognize_vl_extracts_object_blocks():
    """回归：PaddleOCRVLBlock 是对象（非 dict），必须属性访问。

    修复前 block.get("block_content") 对对象抛 AttributeError，VL 识别
    整体异常被吞 → 返回空。
    """
    res = _DictResult(
        {
            "parsing_res_list": [
                _VLBlock(content="title text", label="title", global_block_id=0),
                _VLBlock(content="", label="figure", global_block_id=1),
            ]
        }
    )
    service = _FakeService([res])
    result = _recognize_paddlocr_vl(service, image=None, options=PaddleOCRVLOptions())

    assert len(result.text_blocks) == 1
    assert result.text_blocks[0].text == "title text"
    assert result.text_blocks[0].label == "title"


def test_vl_deduplicates_table_across_content_and_parsing_lists():
    table_html = "<table><tr><td>A</td><td>B</td></tr></table>"
    res = _DictResult(
        {
            "content_list": [
                {"type": "table", "table_body": table_html, "bbox": [1, 2, 30, 40]}
            ],
            "parsing_res_list": [
                {
                    "block_bbox": [1, 2, 30, 40],
                    "block_content": table_html,
                    "block_label": "table",
                    "block_order": 0,
                }
            ],
        }
    )

    result = _recognize_paddlocr_vl(
        _FakeService([res]), image=None, options=PaddleOCRVLOptions()
    )

    tables = [block for block in result.content_list if block["type"] == "table"]
    assert len(tables) == 1
    assert tables[0]["table"]["table_id"] == tables[0]["block_id"]
    assert tables[0]["table"]["provenance"]["provider_schema"] == "paddlex-paddlocr-vl"
    assert tables[0]["table_body"]
    assert result.text_blocks[0].content_index == 0
    assert result.text_blocks[0].content_id == tables[0]["block_id"]
    assert result.text_blocks[0].text == "A\tB"
    assert result.text_with_scores[0] == ("A\tB", 0.9)
    assert result.markdown_text == "| A | B |\n| --- | --- |"


def test_vl_deduplicates_table_when_content_list_bbox_is_missing():
    table_html = "<table><tr><td>A</td></tr></table>"
    res = _DictResult(
        {
            "content_list": [{"type": "table", "table_body": table_html}],
            "parsing_res_list": [
                {
                    "block_bbox": [1, 2, 30, 40],
                    "block_content": table_html,
                    "block_label": "table",
                    "block_order": 0,
                }
            ],
        }
    )

    result = _recognize_paddlocr_vl(
        _FakeService([res]), image=None, options=PaddleOCRVLOptions()
    )

    tables = [block for block in result.content_list if block["type"] == "table"]
    assert len(tables) == 1
    assert tables[0]["bbox"] == (1.0, 2.0, 30.0, 40.0)


def test_vl_does_not_deduplicate_same_table_across_results():
    table_html = "<table><tr><td>A</td></tr></table>"
    results = [
        _DictResult(
            {
                "parsing_res_list": [
                    {
                        "block_bbox": [1, 2, 30, 40],
                        "block_content": table_html,
                        "block_label": "table",
                        "block_order": 0,
                    }
                ]
            }
        )
        for _ in range(2)
    ]

    result = _recognize_paddlocr_vl(
        _FakeService(results), image=None, options=PaddleOCRVLOptions()
    )

    tables = [block for block in result.content_list if block["type"] == "table"]
    assert len(tables) == 2
    assert len(result.text_blocks) == 2


def test_vl_prefers_provider_block_id_for_ambiguous_tables():
    table_html = "<table><tr><td>A</td></tr></table>"
    res = _DictResult(
        {
            "content_list": [
                {
                    "type": "table",
                    "table_body": table_html,
                    "global_block_id": 0,
                },
                {
                    "type": "table",
                    "table_body": table_html,
                    "global_block_id": 1,
                },
            ],
            "parsing_res_list": [
                {
                    "block_bbox": [10, 20, 30, 40],
                    "block_content": table_html,
                    "block_label": "table",
                    "block_order": 1,
                }
            ],
        }
    )

    result = _recognize_paddlocr_vl(
        _FakeService([res]), image=None, options=PaddleOCRVLOptions()
    )

    tables = [block for block in result.content_list if block["type"] == "table"]
    assert len(tables) == 2
    assert tables[0]["global_block_id"] == 1
    assert tables[0]["bbox"] == (10.0, 20.0, 30.0, 40.0)


def test_vl_projects_content_list_only_table():
    table_html = (
        "<table><tr><td rowspan='2'>A</td><td>B</td></tr><tr><td>C</td></tr></table>"
    )
    res = _DictResult(
        {
            "content_list": [{"type": "table", "table_body": table_html}],
            "parsing_res_list": [],
        }
    )

    result = _recognize_paddlocr_vl(
        _FakeService([res]), image=None, options=PaddleOCRVLOptions()
    )

    assert result.text_blocks[0].text == "A\tB\nC"
    assert result.raw_text == "A\tB\nC"
    assert "| A | B |" in result.markdown_text
    assert 'rowspan="2"' in result.content_list[0]["table_body"]
    assert 'rowspan="2"' in result.html_text
    warnings = result.content_list[0]["table"]["provenance"]["warnings"]
    assert "lossy_markdown_source" in warnings


# ---- 纯函数覆盖：bbox/score/canonical table 投影 ----


class TestVlPureHelpers:
    def test_extract_bbox_from_rec_boxes_polygon(self):
        from vibeocr.backend.core.pipelines.pipeline_paddlocr_vl import (
            _extract_bbox_from_rec_boxes,
        )

        boxes = [[[0, 0], [10, 0], [10, 5], [0, 5]]]
        assert _extract_bbox_from_rec_boxes(boxes, 0) == (0.0, 0.0, 10.0, 5.0)

    def test_extract_bbox_from_rec_boxes_two_point(self):
        from vibeocr.backend.core.pipelines.pipeline_paddlocr_vl import (
            _extract_bbox_from_rec_boxes,
        )

        boxes = [[[1, 2], [3, 4]]]
        assert _extract_bbox_from_rec_boxes(boxes, 0) == (1.0, 2.0, 3.0, 4.0)

    def test_extract_bbox_from_rec_boxes_invalid(self):
        from vibeocr.backend.core.pipelines.pipeline_paddlocr_vl import (
            _extract_bbox_from_rec_boxes,
        )

        assert _extract_bbox_from_rec_boxes([], 0) is None
        assert _extract_bbox_from_rec_boxes([[]], 0) is None

    def test_extract_block_bbox_4_floats(self):
        from vibeocr.backend.core.pipelines.pipeline_paddlocr_vl import (
            _extract_block_bbox,
        )

        assert _extract_block_bbox([1, 2, 3, 4]) == (1.0, 2.0, 3.0, 4.0)

    def test_extract_block_bbox_polygon_points(self):
        from vibeocr.backend.core.pipelines.pipeline_paddlocr_vl import (
            _extract_block_bbox,
        )

        # 多点 → 外接矩形
        result = _extract_block_bbox([[0, 0], [10, 0], [10, 5], [0, 5]])
        assert result == (0.0, 0.0, 10.0, 5.0)

    def test_extract_block_bbox_empty_returns_none(self):
        from vibeocr.backend.core.pipelines.pipeline_paddlocr_vl import (
            _extract_block_bbox,
        )

        assert _extract_block_bbox(None) is None
        assert _extract_block_bbox([]) is None

    def test_get_block_score_default(self):
        from vibeocr.backend.core.pipelines.pipeline_paddlocr_vl import _get_block_score

        # 无 layout_det_res → 默认 0.9
        assert _get_block_score({}, {"block_order": 0}) == 0.9

    def test_get_block_score_from_layout_det_res(self):
        from vibeocr.backend.core.pipelines.pipeline_paddlocr_vl import _get_block_score

        class _FakeBoxes:
            def __init__(self, boxes):
                self._boxes = boxes

            def __getitem__(self, idx):
                return self._boxes[idx]

            def __len__(self):
                return len(self._boxes)

        class _FakeLayoutDet:
            boxes = _FakeBoxes([{"score": 0.75}])

        class _FakeRes:
            layout_det_res = _FakeLayoutDet()

        assert _get_block_score(_FakeRes(), {"block_order": 0}) == 0.75

    def test_project_canonical_table(self):
        from vibeocr.backend.core.pipelines.pipeline_paddlocr_vl import (
            _project_canonical_table,
        )

        block = {"type": "table", "table_body": "<table><tr><td>x</td></tr></table>"}
        result = _project_canonical_table(block)
        # 返回 (html, markdown) 元组
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(x, str) for x in result)
