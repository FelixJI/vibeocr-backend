"""pipeline_pp_structure.py 数据处理纯函数测试。"""

from __future__ import annotations

import gc
from types import SimpleNamespace

import pytest
from vibeocr.backend.core.pipelines.pipeline_pp_structure import (
    PPStructureV3Options,
    _build_ocr_result,
    _consume_generator_safely,
    _recognize_pp_structure,
)


class TestBuildOcrResult:
    """_build_ocr_result（PP-Structure 版）。"""

    def test_minimal(self):
        result = _build_ocr_result("text")
        assert result.raw_text == "text"
        assert result.pipeline_type == "PP-StructureV3"
        assert result.avg_score == 0.0

    def test_with_scores_and_low_confidence(self):
        result = _build_ocr_result("t", text_with_scores=[("ok", 0.9), ("low", 0.3)])
        assert result.avg_score == pytest.approx(0.6)
        assert len(result.low_confidence_items) == 1


class TestConsumeGeneratorSafely:
    """_consume_generator_safely（PP-Structure 版，逻辑同 pipeline_ocr）。"""

    def test_consumes_generator(self):
        def gen():
            yield 1
            yield 2

        assert _consume_generator_safely(gen()) == [1, 2]

    def test_exception_is_propagated(self):
        gc.enable()

        def bad_gen():
            yield 1
            raise ValueError("err")

        with pytest.raises(ValueError, match="err"):
            _consume_generator_safely(bad_gen())
        assert gc.isenabled()

    def test_gc_reenabled(self):
        gc.enable()
        _consume_generator_safely(iter([1]))
        assert gc.isenabled()


class TestPPStructureV3Options:
    """PPStructureV3Options dataclass 默认值。"""

    def test_defaults(self):
        opts = PPStructureV3Options()
        assert opts.pipeline == "PP-StructureV3"
        assert opts.use_table_recognition is True
        assert opts.use_formula_recognition is True
        assert opts.use_seal_recognition is False
        assert opts.use_chart_recognition is False


def test_pp_structure_emits_canonical_table_and_keeps_content_index():
    block = SimpleNamespace(
        label="table",
        bbox=[10, 20, 100, 80],
        content=(
            "<table><tr><td rowspan='2'>A</td><td>B</td></tr>"
            "<tr><td>C</td></tr></table>"
        ),
        order_index=0,
        image=None,
    )

    class Pipeline:
        def predict(self, **_kwargs):
            return [{"parsing_res_list": [block]}]

    class Service:
        def get_or_create_pipeline(self, _name):
            return Pipeline()

    result = _recognize_pp_structure(
        Service(), image=None, options=PPStructureV3Options()
    )

    table_block = result.content_list[0]
    assert table_block["table"]["table_id"] == table_block["block_id"]
    assert (
        table_block["table"]["provenance"]["provider_schema"]
        == "paddlex-pp-structure-v3"
    )
    assert result.text_blocks[0].content_index == 0
    assert result.text_blocks[0].content_id == table_block["block_id"]
    assert result.text_blocks[0].order == 0
    assert result.text_blocks[0].text == "A\tB\nC"
    assert result.text_with_scores[0] == ("A\tB\nC", 0.9)
    assert result.raw_text == "A\tB\nC"
    assert 'rowspan="2"' in result.html_text


# ---- _recognize_pp_structure 多 block 类型（formula/image/text）----


class TestRecognizePpStructureBlockTypes:
    """覆盖 formula/image/text/markdown 分支（line 173-184, 196-203, 273-323）。"""

    def test_formula_block(self):
        from vibeocr.backend.core.pipelines.pipeline_pp_structure import (
            PPStructureV3Options,
            _recognize_pp_structure,
        )

        class _Block:
            def __init__(self, label, content, bbox=None, order_index=0):
                self.label = label
                self.content = content
                self.bbox = bbox or [1, 2, 3, 4]
                self.order_index = order_index
                self.image = None

        class _DictResult(dict):
            pass

        res = _DictResult({"parsing_res_list": [_Block("formula", "a^2+b^2")]})

        class _Pipeline:
            def predict(self, input, **kwargs):  # noqa: A002
                return [res]

        class _Service:
            def get_or_create_pipeline(self, _name):
                return _Pipeline()

        result = _recognize_pp_structure(
            _Service(), image=None, options=PPStructureV3Options()
        )
        formula_blocks = [b for b in result.content_list if b.get("type") == "formula"]
        assert formula_blocks
        assert "$$" in result.markdown_text

    def test_image_block_with_path(self):
        from vibeocr.backend.core.pipelines.pipeline_pp_structure import (
            PPStructureV3Options,
            _recognize_pp_structure,
        )

        class _Block:
            label = "image"
            content = ""
            bbox = [0, 0, 10, 10]
            order_index = 0
            image = {"path": "imgs/fig0.png"}

        class _DictResult(dict):
            pass

        res = _DictResult({"parsing_res_list": [_Block()]})

        class _Pipeline:
            def predict(self, input, **kwargs):  # noqa: A002
                return [res]

        class _Service:
            def get_or_create_pipeline(self, _name):
                return _Pipeline()

        result = _recognize_pp_structure(
            _Service(), image=None, options=PPStructureV3Options()
        )
        img_blocks = [b for b in result.content_list if b.get("type") == "image"]
        assert img_blocks
        assert img_blocks[0].get("img_path") == "imgs/fig0.png"

    def test_text_block(self):
        from vibeocr.backend.core.pipelines.pipeline_pp_structure import (
            PPStructureV3Options,
            _recognize_pp_structure,
        )

        class _Block:
            label = "text"
            content = "plain text"
            bbox = [0, 0, 10, 10]
            order_index = 2
            image = None

        class _DictResult(dict):
            pass

        res = _DictResult({"parsing_res_list": [_Block()]})

        class _Pipeline:
            def predict(self, input, **kwargs):  # noqa: A002
                return [res]

        class _Service:
            def get_or_create_pipeline(self, _name):
                return _Pipeline()

        result = _recognize_pp_structure(
            _Service(), image=None, options=PPStructureV3Options()
        )
        text_blocks = [b for b in result.content_list if b.get("type") == "text"]
        assert text_blocks
        assert text_blocks[0]["text"] == "plain text"

    def test_markdown_extraction_from_result(self):
        """res.markdown dict 含 markdown_texts/images → 提取（line 195-203）。"""
        from vibeocr.backend.core.pipelines.pipeline_pp_structure import (
            PPStructureV3Options,
            _recognize_pp_structure,
        )

        class _DictResult(dict):
            pass

        res = _DictResult({"parsing_res_list": []})
        # res 需带 .markdown 属性（hasattr 检查），不能仅靠 dict key
        res.markdown = {
            "markdown_texts": "# Title",
            "markdown_images": {"fig1.png": b"img-bytes"},
        }

        class _Pipeline:
            def predict(self, input, **kwargs):  # noqa: A002
                return [res]

        class _Service:
            def get_or_create_pipeline(self, _name):
                return _Pipeline()

        result = _recognize_pp_structure(
            _Service(), image=None, options=PPStructureV3Options()
        )
        # markdown_images 被提取到 result.images（markdown_text 可能被
        # rebuild_result_projections 重算，这里只验证 images 提取）
        assert "fig1.png" in result.images

    def test_preproc_info_extraction(self):
        """doc_preprocessor_res.output_img 被提取（line 169-184）。"""
        import numpy as np
        from vibeocr.backend.core.pipelines.pipeline_pp_structure import (
            PPStructureV3Options,
            _recognize_pp_structure,
        )

        arr = np.zeros((2, 3, 3), dtype=np.uint8)
        res = {
            "parsing_res_list": [],
            "doc_preprocessor_res": {"angle": 90, "output_img": arr},
        }

        class _Pipeline:
            def predict(self, input, **kwargs):  # noqa: A002
                return [res]

        class _Service:
            def get_or_create_pipeline(self, _name):
                return _Pipeline()

        result = _recognize_pp_structure(
            _Service(), image=None, options=PPStructureV3Options()
        )
        assert result.preproc_angle == 90
        assert result.preproc_img_w == 3
        assert result.preprocessed_image is not None
