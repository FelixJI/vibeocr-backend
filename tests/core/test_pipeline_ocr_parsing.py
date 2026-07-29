"""pipeline_ocr.py 数据处理纯函数测试。

pipeline_ocr.py 仅 17% 覆盖——现有 test_pipelines.py 只测元数据，
完全没覆盖 _extract_bbox / _extract_polygon / _parse_single_result /
_extract_preproc_info / _build_ocr_result / _consume_generator_safely
这些 PaddleOCR 结果解析逻辑。本文件补齐这些函数的全分支。
"""

from __future__ import annotations

import gc
from types import SimpleNamespace

import numpy as np
import pytest

from vibeocr.backend.core.pipelines.pipeline_ocr import (
    _build_ocr_result,
    _consume_generator_safely,
    _extract_bbox,
    _extract_polygon,
    _extract_preproc_info,
    _parse_single_result,
    _recognize_ocr,
    _recognize_ocr_batch,
)
from vibeocr.backend.models.ocr_options import OCROptions


class TestExtractBbox:
    """_extract_bbox：从 rec_boxes 提取 bbox，支持 3 种格式。"""

    def test_axis_aligned_4_values(self):
        """(N,4) 格式：[x0,y0,x1,y1]"""
        boxes = [[10, 20, 30, 40], [50, 60, 70, 80]]
        assert _extract_bbox(boxes, 0) == (10.0, 20.0, 30.0, 40.0)
        assert _extract_bbox(boxes, 1) == (50.0, 60.0, 70.0, 80.0)

    def test_polygon_4_points(self):
        """(N,4,2) 四点多边形：取外接矩形"""
        boxes = [[[0, 0], [10, 0], [10, 5], [0, 5]]]
        result = _extract_bbox(boxes, 0)
        assert result == (0.0, 0.0, 10.0, 5.0)

    def test_two_point_rectangle(self):
        """(N,2,2) 两点矩形"""
        boxes = [[[1, 2], [3, 4]]]
        assert _extract_bbox(boxes, 0) == (1.0, 2.0, 3.0, 4.0)

    def test_numpy_array_box(self):
        """box 是 numpy array 时自动 tolist"""
        boxes = [np.array([1.0, 2.0, 3.0, 4.0])]
        assert _extract_bbox(boxes, 0) == (1.0, 2.0, 3.0, 4.0)

    def test_index_out_of_range_returns_none(self):
        """越界返回 None"""
        assert _extract_bbox([[1, 2, 3, 4]], 5) is None

    def test_malformed_returns_none(self):
        """畸形数据返回 None 而非抛异常"""
        assert _extract_bbox(None, 0) is None
        assert _extract_bbox([], 0) is None


class TestExtractPolygon:
    """_extract_polygon：提取 4 点检测多边形。"""

    def test_four_point_polygon(self):
        polys = [[[0, 0], [10, 0], [10, 5], [0, 5]]]
        result = _extract_polygon(polys, 0)
        assert result == (0.0, 0.0, 10.0, 0.0, 10.0, 5.0, 0.0, 5.0)

    def test_numpy_polygon(self):
        polys = [np.array([[1, 2], [3, 4], [5, 6], [7, 8]])]
        result = _extract_polygon(polys, 0)
        assert result == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)

    def test_insufficient_points_returns_none(self):
        """少于 4 点返回 None"""
        polys = [[[0, 0], [1, 1]]]
        assert _extract_polygon(polys, 0) is None

    def test_index_out_of_range_returns_none(self):
        assert _extract_polygon([[[0, 0], [1, 1], [2, 2], [3, 3]]], 5) is None

    def test_malformed_returns_none(self):
        assert _extract_polygon(None, 0) is None


class TestConsumeGeneratorSafely:
    """_consume_generator_safely：禁用 GC 消费 generator。"""

    def test_normal_generator_consumed(self):
        def gen():
            yield 1
            yield 2
            yield 3

        assert _consume_generator_safely(gen()) == [1, 2, 3]

    def test_list_input(self):
        assert _consume_generator_safely([10, 20]) == [10, 20]

    def test_exception_returns_empty_list(self):
        def bad_gen():
            yield 1
            raise RuntimeError("boom")

        assert _consume_generator_safely(bad_gen()) == []

    def test_gc_reenabled_after_consumption(self):
        """消费完成后 GC 应恢复原状态。"""
        gc.enable()
        _consume_generator_safely(iter([1, 2, 3]))
        assert gc.isenabled()

    def test_gc_reenabled_even_on_exception(self):
        gc.enable()

        def bad_gen():
            yield 1
            raise RuntimeError("boom")

        _consume_generator_safely(bad_gen())
        assert gc.isenabled()


class TestBuildOcrResult:
    """_build_ocr_result：构建 OCRResult 对象。"""

    def test_minimal_raw_text_only(self):
        result = _build_ocr_result("hello")
        assert result.raw_text == "hello"
        assert result.markdown_text == "hello"
        assert result.html_text == "hello"
        assert result.avg_score == 0.0
        assert result.text_with_scores == []

    def test_with_scores_computes_avg(self):
        result = _build_ocr_result(
            "a b", text_with_scores=[("a", 0.9), ("b", 0.7)]
        )
        assert result.avg_score == pytest.approx(0.8)
        assert len(result.text_with_scores) == 2

    def test_low_confidence_items_collected(self):
        result = _build_ocr_result(
            "x", text_with_scores=[("good", 0.95), ("bad", 0.5), ("ok", 0.85)]
        )
        assert len(result.low_confidence_items) == 1
        assert result.low_confidence_items[0] == ("bad", 0.5)

    def test_explicit_markdown_html(self):
        result = _build_ocr_result(
            "raw", markdown_text="# md", html_text="<p>html</p>"
        )
        assert result.markdown_text == "# md"
        assert result.html_text == "<p>html</p>"


class TestParseSingleResult:
    """_parse_single_result：从 OCR 结果提取 (text,score) 与 TextBlock。"""

    def test_obj_with_rec_texts_and_scores(self):
        """对象有 rec_texts + rec_scores + rec_boxes"""
        res = SimpleNamespace(
            rec_texts=["hello", "world"],
            rec_scores=[0.9, 0.8],
            rec_boxes=[[0, 0, 10, 10], [20, 20, 30, 30]],
            rec_polys=None,
            dt_polys=None,
        )
        tws, blocks = _parse_single_result(res)
        assert tws == [("hello", 0.9), ("world", 0.8)]
        assert len(blocks) == 2
        assert blocks[0].text == "hello"
        assert blocks[0].bbox == (0.0, 0.0, 10.0, 10.0)

    def test_obj_with_rec_texts_only(self):
        """对象只有 rec_texts（无 scores）→ score 默认 1.0"""
        res = SimpleNamespace(
            rec_texts=["a", "b"],
            rec_scores=None,
            rec_boxes=None,
            rec_polys=None,
            dt_polys=None,
        )
        # 无 rec_scores 属性时不走第一分支；构造无 rec_scores
        res = SimpleNamespace(rec_texts=["a", "b"], rec_boxes=None, rec_polys=None)
        tws, _blocks = _parse_single_result(res)
        assert tws == [("a", 1.0), ("b", 1.0)]

    def test_obj_with_ocr_text(self):
        """对象只有 ocr_text 属性"""
        res = SimpleNamespace(ocr_text="single line")
        tws, blocks = _parse_single_result(res)
        assert tws == [("single line", 1.0)]
        assert len(blocks) == 1

    def test_dict_with_scores(self):
        """dict 形态，带 rec_scores"""
        res = {
            "rec_texts": ["foo", "bar"],
            "rec_scores": [0.95, 0.6],
            "rec_boxes": [[1, 2, 3, 4], None],
        }
        tws, blocks = _parse_single_result(res)
        assert tws == [("foo", 0.95), ("bar", 0.6)]
        assert blocks[0].bbox == (1.0, 2.0, 3.0, 4.0)

    def test_dict_without_scores(self):
        """dict 形态，无 rec_scores → score 默认 1.0"""
        res = {"rec_texts": ["x", "y"]}
        tws, _blocks = _parse_single_result(res)
        assert tws == [("x", 1.0), ("y", 1.0)]

    def test_empty_texts_skipped(self):
        """空文本被跳过"""
        res = SimpleNamespace(
            rec_texts=["", "valid"], rec_scores=[0.5, 0.9], rec_boxes=None
        )
        tws, _blocks = _parse_single_result(res)
        assert tws == [("valid", 0.9)]

    def test_exception_returns_empty(self):
        """处理异常时返回空列表而非崩溃"""
        tws, blocks = _parse_single_result(None)
        assert tws == []
        assert blocks == []


class TestExtractPreprocInfo:
    """_extract_preproc_info：提取预处理信息 (angle, png, w, h)。"""

    def test_no_preprocessor_res(self):
        """无 doc_preprocessor_res → 全默认"""
        res = {"other": "data"}
        angle, png, w, h = _extract_preproc_info(res)
        assert angle == 0
        assert png is None
        assert w == 0 and h == 0

    def test_with_angle_no_image(self):
        """有 angle 但无 output_img"""
        res = {"doc_preprocessor_res": {"angle": 90}}
        angle, png, _w, _h = _extract_preproc_info(res)
        assert angle == 90
        assert png is None

    def test_with_output_img(self):
        """有 output_img → 计算 w/h 并编码 PNG"""
        img = np.zeros((20, 30, 3), dtype=np.uint8)
        res = {
            "doc_preprocessor_res": {
                "angle": 180,
                "output_img": img,
            }
        }
        angle, png, w, h = _extract_preproc_info(res)
        assert angle == 180
        assert png is not None
        assert png[:4] == b"\x89PNG"  # PNG 魔数
        assert w == 30 and h == 20

    def test_include_image_false_skips_png(self):
        """include_image=False → 不编码 PNG 但仍返回 w/h"""
        img = np.zeros((10, 15, 3), dtype=np.uint8)
        res = {"doc_preprocessor_res": {"angle": 0, "output_img": img}}
        _angle, png, w, h = _extract_preproc_info(res, include_image=False)
        assert png is None
        assert w == 15 and h == 10

    def test_object_without_get(self):
        """res 无 get 方法（非 dict）→ 默认值"""
        res = SimpleNamespace(foo="bar")
        _angle, png, _w, _h = _extract_preproc_info(res)
        assert png is None


# ---- _recognize_ocr / _recognize_ocr_batch via fake service/pipeline ----


class _FakeOcrPipeline:
    """模拟 PaddleOCR pipeline.predict 返回合成 dict 输出。"""

    def __init__(self, output_list):
        self._output = output_list

    def predict(self, input, **kwargs):  # noqa: A002
        return list(self._output)


class _FakeOcrService:
    def __init__(self, output_list):
        self._pipeline = _FakeOcrPipeline(output_list)

    def get_or_create_pipeline(self, name):
        return self._pipeline


def _make_ocr_result_dict(texts, scores=None, boxes=None):
    """构造单个 PaddleOCR OCR 结果 dict。"""
    return {
        "rec_texts": list(texts),
        "rec_scores": scores or [0.9] * len(texts),
        "rec_boxes": boxes,
    }


class TestRecognizeOcr:
    def test_recognize_single_image(self):

        output = [_make_ocr_result_dict(["hello", "world"], boxes=[[0, 0, 10, 10], [1, 1, 2, 2]])]
        service = _FakeOcrService(output)
        result = _recognize_ocr(service, image=None, options=OCROptions())
        assert result.raw_text == "hello\nworld"
        assert len(result.text_blocks) == 2

    def test_recognize_empty_output(self):

        service = _FakeOcrService([])
        result = _recognize_ocr(service, image=None, options=OCROptions())
        assert result.raw_text == ""

    def test_recognize_with_preproc_info(self):
        import numpy as np


        arr = np.zeros((2, 3, 3), dtype=np.uint8)
        output = [
            {
                "rec_texts": ["t"],
                "rec_scores": [0.9],
                "doc_preprocessor_res": {"angle": 180, "output_img": arr},
            }
        ]
        service = _FakeOcrService(output)
        result = _recognize_ocr(service, image=None, options=OCROptions())
        assert result.preproc_angle == 180
        assert result.preproc_img_w == 3
        assert result.preprocessed_image is not None

    def test_recognize_predict_exception_propagates(self):

        class _CrashPipeline:
            def predict(self, input, **kwargs):  # noqa: A002
                raise RuntimeError("predict failed")

        class _CrashService:
            def get_or_create_pipeline(self, name):
                return _CrashPipeline()

        with pytest.raises(RuntimeError, match="predict failed"):
            _recognize_ocr(_CrashService(), image=None, options=OCROptions())


class TestRecognizeOcrBatch:
    def test_batch_multiple_images(self):

        output = [
            _make_ocr_result_dict(["a"]),
            _make_ocr_result_dict(["b", "c"]),
        ]
        service = _FakeOcrService(output)
        results = _recognize_ocr_batch(service, images=[None, None], options=OCROptions())
        assert len(results) == 2
        assert results[0].raw_text == "a"
        assert results[1].raw_text == "b\nc"

    def test_batch_pads_missing_results(self):
        """输出项少于输入图时补空结果（line 474-477）。"""

        output = [_make_ocr_result_dict(["only-one"])]  # 只返回 1 个
        service = _FakeOcrService(output)
        results = _recognize_ocr_batch(
            service, images=[None, None, None], options=OCROptions()
        )
        assert len(results) == 3  # 补齐到 3
        assert results[0].raw_text == "only-one"
        assert results[1].raw_text == ""
        assert results[2].raw_text == ""

    def test_batch_error_item_skipped_preproc(self):
        """结果项含 error → 跳过 preproc 提取（line 452）。"""

        output = [{"error": "failed"}]
        service = _FakeOcrService(output)
        results = _recognize_ocr_batch(service, images=[None], options=OCROptions())
        assert len(results) == 1
        assert results[0].raw_text == ""
        assert results[0].preprocessed_image is None

    def test_batch_predict_exception_propagates(self):

        class _CrashPipeline:
            def predict(self, input, **kwargs):  # noqa: A002
                raise RuntimeError("batch failed")

        class _CrashService:
            def get_or_create_pipeline(self, name):
                return _CrashPipeline()

        with pytest.raises(RuntimeError, match="batch failed"):
            _recognize_ocr_batch(
                _CrashService(), images=[None], options=OCROptions()
            )
