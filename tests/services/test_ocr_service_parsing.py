"""ocr_service 纯解析逻辑单元测试。

覆盖 OCRService 的输出解析与归一化静态/实例方法，使用合成 dict/object
fixtures，不依赖真实 PaddleOCR 模型加载：
- _extract_bbox: rec_boxes 多种格式
- _consume_generator_safely: generator 消费
- _process_ocr_output_safe: dict / object / 多格式输出解析
- _build_ocr_result: OCRResult 构建 + 平均置信度/低置信项
- _normalize_result_bbox: 像素坐标归一化到 [0,1000]
- _decide_enable_mkldnn: CPU/GPU + 探测缓存
- _is_known_onednn_pir_failure: 异常链匹配
- OCRPreset.display_name
"""

from __future__ import annotations

import pytest

from vibeocr.backend.models.ocr_result import OCRResult, TextBlock
from vibeocr.backend.services.ocr_service import OCRPreset, OCRService


def _make_service() -> OCRService:
    """构造一个绕过 __init__ 的 OCRService 实例（避免模型加载）。"""
    return OCRService.__new__(OCRService)


# ---------------------------------------------------------------------------
# OCRPreset
# ---------------------------------------------------------------------------


class TestOCRPreset:
    def test_general_display_name(self):
        assert OCRPreset.GENERAL.display_name == "通用"

    def test_scanned_display_name(self):
        assert OCRPreset.SCANNED.display_name == "扫描件"


# ---------------------------------------------------------------------------
# _extract_bbox
# ---------------------------------------------------------------------------


class TestExtractBbox:
    def test_axis_aligned_4_tuple(self):
        """(N,4) 轴对齐矩形格式 [x0,y0,x1,y1]。"""
        boxes = [[10, 20, 30, 40], [50, 60, 70, 80]]
        assert OCRService._extract_bbox(boxes, 0) == (10.0, 20.0, 30.0, 40.0)
        assert OCRService._extract_bbox(boxes, 1) == (50.0, 60.0, 70.0, 80.0)

    def test_polygon_4_points(self):
        """(N,4,2) 四点多边形 → 外接矩形。"""
        boxes = [[[0, 0], [100, 0], [100, 50], [0, 50]]]
        result = OCRService._extract_bbox(boxes, 0)
        assert result == (0.0, 0.0, 100.0, 50.0)

    def test_two_point_rectangle(self):
        """(N,2,2) 两点矩形 [[x0,y0],[x1,y1]]。"""
        boxes = [[[5, 6], [15, 16]]]
        assert OCRService._extract_bbox(boxes, 0) == (5.0, 6.0, 15.0, 16.0)

    def test_index_out_of_range_returns_none(self):
        assert OCRService._extract_bbox([[1, 2, 3, 4]], 5) is None

    def test_invalid_box_returns_none(self):
        assert OCRService._extract_bbox([], 0) is None
        assert OCRService._extract_bbox([[]], 0) is None

    def test_object_with_tolist(self):
        """box 带 tolist() 方法（如 numpy array）→ 转 list 后解析。"""

        class _FakeArray:
            def __init__(self, data):
                self._d = data

            def tolist(self):
                return self._d

        boxes = [_FakeArray([1, 2, 3, 4])]
        assert OCRService._extract_bbox(boxes, 0) == (1.0, 2.0, 3.0, 4.0)


# ---------------------------------------------------------------------------
# _consume_generator_safely
# ---------------------------------------------------------------------------


class TestConsumeGeneratorSafely:
    def test_consumes_generator_to_list(self):
        gen = (x for x in [{"a": 1}, {"b": 2}])
        result = OCRService._consume_generator_safely(gen)
        assert result == [{"a": 1}, {"b": 2}]

    def test_passthrough_list(self):
        data = [{"x": 1}]
        assert OCRService._consume_generator_safely(data) == [{"x": 1}]

    def test_empty_generator(self):
        assert OCRService._consume_generator_safely(iter([])) == []


# ---------------------------------------------------------------------------
# _build_ocr_result
# ---------------------------------------------------------------------------


class TestBuildOcrResult:
    def test_basic_build(self):
        svc = _make_service()
        result = svc._build_ocr_result(
            raw_text="hello\nworld",
            text_with_scores=[("hello", 0.9), ("world", 0.7)],
        )
        assert result.raw_text == "hello\nworld"
        assert result.markdown_text == "hello\nworld"
        assert result.html_text == "hello\nworld"
        assert result.avg_score == pytest.approx(0.8)
        assert ("world", 0.7) in result.low_confidence_items
        assert ("hello", 0.9) not in result.low_confidence_items

    def test_empty_text_with_scores(self):
        svc = _make_service()
        result = svc._build_ocr_result(raw_text="")
        assert result.raw_text == ""
        assert result.avg_score == 0.0
        assert result.low_confidence_items == []

    def test_custom_markdown_html(self):
        svc = _make_service()
        result = svc._build_ocr_result(
            raw_text="plain",
            markdown_text="# plain",
            html_text="<p>plain</p>",
        )
        assert result.markdown_text == "# plain"
        assert result.html_text == "<p>plain</p>"


# ---------------------------------------------------------------------------
# _process_ocr_output_safe
# ---------------------------------------------------------------------------


class TestProcessOcrOutputSafe:
    def test_dict_with_rec_texts_and_scores(self):
        """dict 含 rec_texts/rec_scores/rec_boxes → 解析为 text_blocks。"""
        svc = _make_service()
        output = [
            {
                "rec_texts": ["hello", "world"],
                "rec_scores": [0.9, 0.8],
                "rec_boxes": [[0, 0, 10, 10], [20, 20, 30, 30]],
            }
        ]
        result = svc._process_ocr_output_safe(output)
        assert result.raw_text == "hello\nworld"
        assert len(result.text_blocks) == 2
        assert result.text_blocks[0].text == "hello"
        assert result.text_blocks[0].score == 0.9
        assert result.text_blocks[0].bbox == (0.0, 0.0, 10.0, 10.0)

    def test_dict_without_scores_defaults_to_1(self):
        """dict 无 rec_scores → score 默认 1.0。"""
        svc = _make_service()
        output = [{"rec_texts": ["abc"], "rec_boxes": [[1, 2, 3, 4]]}]
        result = svc._process_ocr_output_safe(output)
        assert result.text_blocks[0].score == 1.0

    def test_dict_no_rec_boxes_gives_none_bbox(self):
        svc = _make_service()
        output = [{"rec_texts": ["x"], "rec_scores": [0.5]}]
        result = svc._process_ocr_output_safe(output)
        assert result.text_blocks[0].bbox is None

    def test_object_with_rec_texts_and_scores_attrs(self):
        """对象带 rec_texts/rec_scores/rec_boxes 属性。"""

        class _FakeRes:
            rec_texts = ["a", "b"]
            rec_scores = [0.8, 0.6]
            rec_boxes = [[0, 0, 5, 5], [1, 1, 6, 6]]

            def get(self, key, default=None):
                return getattr(self, key, default)

        svc = _make_service()
        result = svc._process_ocr_output_safe([_FakeRes()])
        assert result.raw_text == "a\nb"
        assert result.text_blocks[1].score == 0.6

    def test_object_with_rec_texts_only(self):
        """对象只有 rec_texts（无 rec_scores）→ score=1.0。"""

        class _FakeRes:
            rec_texts = ["only-text"]

            def get(self, key, default=None):
                return getattr(self, key, default)

        svc = _make_service()
        result = svc._process_ocr_output_safe([_FakeRes()])
        assert result.text_blocks[0].text == "only-text"
        assert result.text_blocks[0].score == 1.0

    def test_object_with_ocr_text_attr(self):
        """对象带 ocr_text 属性（单文本）。"""

        class _FakeRes:
            ocr_text = "single line"

            def get(self, key, default=None):
                return getattr(self, key, default)

        svc = _make_service()
        result = svc._process_ocr_output_safe([_FakeRes()])
        assert result.raw_text == "single line"

    def test_empty_texts_skipped(self):
        """rec_texts 含空串 → 跳过。"""
        svc = _make_service()
        output = [{"rec_texts": ["", "keep", ""], "rec_scores": [0.1, 0.9, 0.2]}]
        result = svc._process_ocr_output_safe(output)
        assert result.raw_text == "keep"
        assert len(result.text_blocks) == 1

    def test_preproc_image_extracted(self):
        """doc_preprocessor_res.output_img (numpy RGB) 被提取为 PNG。"""
        import numpy as np

        arr = np.zeros((2, 3, 3), dtype=np.uint8)
        output = [
            {
                "rec_texts": ["t"],
                "rec_scores": [0.9],
                "doc_preprocessor_res": {"angle": 90, "output_img": arr},
            }
        ]
        svc = _make_service()
        result = svc._process_ocr_output_safe(output)
        assert result.preproc_angle == 90
        assert result.preproc_img_w == 3
        assert result.preproc_img_h == 2
        assert result.preprocessed_image is not None
        assert result.preprocessed_image.startswith(b"\x89PNG")

    def test_exception_in_item_continues(self):
        """单个结果项处理抛异常时跳过并继续。"""

        class _Bad:
            rec_texts = property(lambda self: (_ for _ in ()).throw(RuntimeError("x")))

            def get(self, key, default=None):
                return getattr(self, key, default)

        good = {"rec_texts": ["ok"], "rec_scores": [1.0]}
        svc = _make_service()
        result = svc._process_ocr_output_safe([_Bad(), good])
        assert result.raw_text == "ok"

    def test_generator_input(self):
        """generator 输入被正确消费。"""
        svc = _make_service()

        def gen():
            yield {"rec_texts": ["g1"], "rec_scores": [0.9]}

        result = svc._process_ocr_output_safe(gen())
        assert result.raw_text == "g1"


# ---------------------------------------------------------------------------
# _normalize_result_bbox
# ---------------------------------------------------------------------------


class TestNormalizeResultBbox:
    def test_preproc_dims_used(self):
        """优先用 preproc_img_w/h 归一化。"""
        result = OCRResult(
            raw_text="x",
            text_blocks=[TextBlock(text="x", score=1.0, bbox=(100, 200, 300, 400))],
        )
        result.preproc_img_w = 1000
        result.preproc_img_h = 2000
        OCRService._normalize_result_bbox(result, image=None)
        assert result.text_blocks[0].bbox == (100.0, 100.0, 300.0, 200.0)

    def test_numpy_image_shape_fallback(self):
        """无 preproc 尺寸时回退到 image.shape。"""

        class _FakeImg:
            shape = (400, 800, 3)  # (h, w)

        result = OCRResult(
            raw_text="x",
            text_blocks=[TextBlock(text="x", score=1.0, bbox=(100, 100, 200, 200))],
        )
        OCRService._normalize_result_bbox(result, image=_FakeImg())
        # img_w=800, img_h=400
        assert result.text_blocks[0].bbox == (125.0, 250.0, 250.0, 500.0)

    def test_pil_image_size_fallback(self):
        """回退到 image.size (w, h)。"""

        class _FakeImg:
            size = (500, 1000)

        result = OCRResult(
            raw_text="x",
            text_blocks=[TextBlock(text="x", score=1.0, bbox=(50, 100, 100, 200))],
        )
        OCRService._normalize_result_bbox(result, image=_FakeImg())
        assert result.text_blocks[0].bbox == (100.0, 100.0, 200.0, 200.0)

    def test_rotation_swaps_dims(self):
        """90°/270° 旋转时宽高互换。"""

        class _FakeImg:
            shape = (100, 200, 3)  # h=100, w=200

        result = OCRResult(
            raw_text="x",
            text_blocks=[TextBlock(text="x", score=1.0, bbox=(100, 50, 200, 100))],
        )
        result.preproc_angle = 90
        OCRService._normalize_result_bbox(result, image=_FakeImg())
        # 旋转后 img_w=100, img_h=200（互换）
        assert result.text_blocks[0].bbox == (1000.0, 250.0, 2000.0, 500.0)

    def test_no_image_returns_unchanged(self):
        """无尺寸信息时不修改。"""
        result = OCRResult(
            raw_text="x",
            text_blocks=[TextBlock(text="x", score=1.0, bbox=(1, 2, 3, 4))],
        )
        OCRService._normalize_result_bbox(result, image=None)
        assert result.text_blocks[0].bbox == (1, 2, 3, 4)

    def test_polygon_normalized_when_large(self):
        """polygon 像素坐标 > 1001 时归一化到 [0,1000]。"""
        result = OCRResult(
            raw_text="x",
            text_blocks=[
                TextBlock(
                    text="x",
                    score=1.0,
                    bbox=(10, 10, 20, 20),
                    # 像素在图像尺寸内但 > 1001（y 用 1500/1800，img_h=2000）
                    polygon=(800, 1500, 900, 1800),
                )
            ],
        )
        result.preproc_img_w = 1000
        result.preproc_img_h = 2000
        OCRService._normalize_result_bbox(result, image=None)
        poly = result.text_blocks[0].polygon
        assert poly is not None
        assert max(poly) <= 1001


# ---------------------------------------------------------------------------
# _is_known_onednn_pir_failure
# ---------------------------------------------------------------------------


class TestIsKnownOnednnPirFailure:
    def test_matching_message_returns_true(self):
        err = RuntimeError(
            "ConvertPirAttribute2RuntimeAttribute failed for onednn_instruction"
        )
        assert OCRService._is_known_onednn_pir_failure(err) is True

    def test_non_matching_returns_false(self):
        assert OCRService._is_known_onednn_pir_failure(ValueError("other")) is False

    def test_chained_exception_matched(self):
        """异常链中某层命中 → True。"""
        root = RuntimeError("ConvertPirAttribute2RuntimeAttribute onednn_instruction")
        mid = ValueError("wrapped")
        mid.__cause__ = root
        assert OCRService._is_known_onednn_pir_failure(mid) is True

    def test_missing_onednn_keyword_returns_false(self):
        err = RuntimeError("ConvertPirAttribute2RuntimeAttribute without keyword")
        assert OCRService._is_known_onednn_pir_failure(err) is False

    def test_circular_chain_terminates(self):
        """循环异常链不无限递归。"""
        a = RuntimeError("x")
        b = RuntimeError("y")
        a.__cause__ = b
        b.__cause__ = a
        assert OCRService._is_known_onednn_pir_failure(a) is False


# ---------------------------------------------------------------------------
# _decide_enable_mkldnn
# ---------------------------------------------------------------------------


class TestDecideEnableMkldnn:
    def test_gpu_device_returns_false(self):
        # 重置缓存
        OCRService._onednn_safe_cache = None
        OCRService._onednn_runtime_disabled = False
        assert OCRService._decide_enable_mkldnn("gpu") is False

    def test_runtime_disabled_returns_false(self):
        OCRService._onednn_safe_cache = True
        OCRService._onednn_runtime_disabled = True
        assert OCRService._decide_enable_mkldnn("cpu") is False
        OCRService._onednn_runtime_disabled = False

    def test_cpu_probes_and_caches(self, monkeypatch):
        OCRService._onednn_safe_cache = None
        OCRService._onednn_runtime_disabled = False

        called = {"n": 0}

        def fake_probe():
            called["n"] += 1
            return (True, "safe")

        monkeypatch.setattr(
            "vibeocr.backend.utils.cpu_info.can_safely_enable_onednn", fake_probe
        )
        assert OCRService._decide_enable_mkldnn("cpu") is True
        # 缓存生效：第二次不重新探测
        OCRService._decide_enable_mkldnn("cpu")
        assert called["n"] == 1

    def test_probe_exception_disables(self, monkeypatch):
        OCRService._onednn_safe_cache = None
        OCRService._onednn_runtime_disabled = False

        def _raise():
            raise RuntimeError("probe failed")

        monkeypatch.setattr(
            "vibeocr.backend.utils.cpu_info.can_safely_enable_onednn", _raise
        )
        assert OCRService._decide_enable_mkldnn("cpu") is False
        assert OCRService._onednn_safe_cache is False
