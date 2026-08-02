"""ocr_service.py staticmethod 纯逻辑测试。

ocr_service.py 仅 33% 覆盖。聚焦可独立测试的 staticmethod：
_to_ndarray / _is_known_onednn_pir_failure / _normalize_result_bbox /
_decide_enable_mkldnn（class 级缓存逻辑）。
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
from vibeocr.backend.models.ocr_result import OCRResult, TextBlock
from vibeocr.backend.services.ocr_service import OCRService


class TestToNdarray:
    """_to_ndarray：统一输入为 ndarray 或路径字符串。"""

    def test_ndarray_passthrough(self):
        arr = np.zeros((10, 10, 3), dtype=np.uint8)
        # ndarray 无 convert 属性 → 原样返回
        result = OCRService._to_ndarray(arr)
        assert result is arr

    def test_str_path_passthrough(self):
        result = OCRService._to_ndarray("/some/path/image.png")
        assert result == "/some/path/image.png"

    def test_bytes_converted_to_ndarray(self):
        import io

        from PIL import Image

        img = Image.new("RGB", (5, 3), color=(255, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        result = OCRService._to_ndarray(buf.getvalue())
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 5, 3)

    def test_pil_image_converted(self):
        from PIL import Image

        img = Image.new("RGBA", (4, 2), color=(0, 255, 0, 128))
        result = OCRService._to_ndarray(img)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 4, 3)  # converted to RGB


class TestIsKnownOnednnPirFailure:
    """_is_known_onednn_pir_failure：判断异常链是否命中 PIR/oneDNN 回归。"""

    def test_direct_match(self):
        exc = RuntimeError(
            "ConvertPirAttribute2RuntimeAttribute onednn_instruction not impl"
        )
        assert OCRService._is_known_onednn_pir_failure(exc) is True

    def test_no_match_plain_error(self):
        assert OCRService._is_known_onednn_pir_failure(ValueError("普通错误")) is False

    def test_match_via_cause_chain(self):
        """异常 __cause__ 链中包含匹配模式"""
        root = RuntimeError(
            "ConvertPirAttribute2RuntimeAttribute failed for onednn_instruction"
        )
        try:
            try:
                raise root
            except RuntimeError as e:
                raise RuntimeError("包装错误") from e
        except RuntimeError as wrapper:
            assert OCRService._is_known_onednn_pir_failure(wrapper) is True

    def test_match_via_context_chain(self):
        """异常 __context__ 链中包含匹配模式"""
        root = RuntimeError(
            "ConvertPirAttribute2RuntimeAttribute onednn_instruction bug"
        )
        try:
            try:
                raise root
            except RuntimeError:
                raise RuntimeError("隐式传播")  # noqa: B904 - 有意测试隐式 __context__ 传播
        except RuntimeError as wrapper:
            assert OCRService._is_known_onednn_pir_failure(wrapper) is True

    def test_missing_onednn_keyword_no_match(self):
        """只有 ConvertPirAttribute2RuntimeAttribute 但无 onednn → 不匹配"""
        exc = RuntimeError("ConvertPirAttribute2RuntimeAttribute something else")
        assert OCRService._is_known_onednn_pir_failure(exc) is False

    def test_circular_reference_does_not_loop(self):
        """异常链出现循环引用时不死循环（id 去重保护）"""
        exc1 = RuntimeError("a")
        exc2 = RuntimeError("b")
        exc1.__cause__ = exc2
        exc2.__cause__ = exc1  # 循环
        # 应正常返回而非死循环
        assert OCRService._is_known_onednn_pir_failure(exc1) is False


class TestNormalizeResultBbox:
    """_normalize_result_bbox：像素坐标归一化到 [0,1000]。"""

    def test_uses_preproc_dimensions(self):
        """优先用 result.preproc_img_w/h"""
        result = OCRResult(
            raw_text="x",
            preproc_img_w=100,
            preproc_img_h=50,
            text_blocks=[TextBlock(text="a", score=1.0, bbox=(10, 10, 50, 25))],
        )
        OCRService._normalize_result_bbox(result, image=None)
        assert result.text_blocks[0].bbox == (100.0, 200.0, 500.0, 500.0)

    def test_falls_back_to_image_shape(self):
        """无 preproc 尺寸时回退到 ndarray.shape"""
        img = np.zeros((50, 100, 3), dtype=np.uint8)  # h=50, w=100
        result = OCRResult(
            raw_text="x",
            preproc_img_w=0,
            preproc_img_h=0,
            text_blocks=[TextBlock(text="a", score=1.0, bbox=(50, 25, 100, 50))],
        )
        OCRService._normalize_result_bbox(result, image=img)
        assert result.text_blocks[0].bbox == (500.0, 500.0, 1000.0, 1000.0)

    def test_falls_back_to_pil_size(self):
        """无 shape 时回退到 PIL .size"""
        from PIL import Image

        img = Image.new("RGB", (200, 100))  # w=200, h=100
        result = OCRResult(
            raw_text="x",
            preproc_img_w=0,
            preproc_img_h=0,
            text_blocks=[TextBlock(text="a", score=1.0, bbox=(100, 50, 200, 100))],
        )
        OCRService._normalize_result_bbox(result, image=img)
        assert result.text_blocks[0].bbox == (500.0, 500.0, 1000.0, 1000.0)

    def test_rotation_swaps_dimensions(self):
        """90°/270° 旋转时宽高互换（仅无 preproc 尺寸时）"""
        img = np.zeros((50, 100, 3), dtype=np.uint8)  # h=50, w=100
        result = OCRResult(
            raw_text="x",
            preproc_img_w=0,
            preproc_img_h=0,
            preproc_angle=90,
            text_blocks=[TextBlock(text="a", score=1.0, bbox=(25, 50, 50, 100))],
        )
        OCRService._normalize_result_bbox(result, image=img)
        # 旋转后 img_w/h 互换：原 shape h=50,w=100 → 旋转后 w=50,h=100
        # bbox (25,50,50,100) / (50,100) * 1000
        assert result.text_blocks[0].bbox == (500.0, 500.0, 1000.0, 1000.0)

    def test_zero_dimensions_no_change(self):
        """尺寸为 0 时不修改（早返回）"""
        result = OCRResult(
            raw_text="x",
            preproc_img_w=0,
            preproc_img_h=0,
            text_blocks=[TextBlock(text="a", score=1.0, bbox=(10, 10, 20, 20))],
        )
        OCRService._normalize_result_bbox(result, image=None)
        assert result.text_blocks[0].bbox == (10, 10, 20, 20)

    def test_polygon_normalized_when_large(self):
        """polygon 像素值 >1001 时归一化"""
        result = OCRResult(
            raw_text="x",
            preproc_img_w=100,
            preproc_img_h=100,
            text_blocks=[
                TextBlock(
                    text="a",
                    score=1.0,
                    bbox=None,
                    polygon=(0, 0, 100, 0, 100, 100, 0, 100),
                )
            ],
        )
        OCRService._normalize_result_bbox(result, image=None)
        poly = result.text_blocks[0].polygon
        assert poly is not None
        assert max(poly) <= 1001

    def test_content_list_bbox_normalized(self):
        """content_list 的 bbox 也被归一化"""
        result = OCRResult(
            raw_text="x",
            preproc_img_w=100,
            preproc_img_h=100,
            text_blocks=[],
            content_list=[{"bbox": [10, 10, 50, 50]}],
        )
        OCRService._normalize_result_bbox(result, image=None)
        assert result.content_list[0]["bbox"] == [100.0, 100.0, 500.0, 500.0]


class TestDecideEnableMkldnn:
    """_decide_enable_mkldnn：决定是否启用 oneDNN（class 级缓存）。"""

    def setup_method(self):
        """每个测试前重置 class 级缓存，避免测试间污染。"""
        OCRService._onednn_safe_cache = None
        OCRService._onednn_runtime_disabled = False

    def test_gpu_device_returns_false(self):
        """GPU 设备不启用 oneDNN"""
        assert OCRService._decide_enable_mkldnn("gpu") is False

    def test_runtime_disabled_returns_false(self):
        """运行时已锁定禁用时返回 False（即使 CPU）"""
        OCRService._onednn_runtime_disabled = True
        assert OCRService._decide_enable_mkldnn("cpu") is False

    def test_cpu_safe_probe_true(self):
        """CPU 设备 + 探测安全 → True，且缓存结果"""
        with patch(
            "vibeocr.backend.utils.cpu_info.can_safely_enable_onednn"
        ) as mock_probe:
            mock_probe.return_value = (True, "supported CPU")
            assert OCRService._decide_enable_mkldnn("cpu") is True
            # 缓存生效：第二次不再探测
            mock_probe.return_value = (False, "should not be reached")
            assert OCRService._decide_enable_mkldnn("cpu") is True

    def test_cpu_unsafe_probe_false(self):
        """CPU 设备 + 探测不安全 → False"""
        with patch(
            "vibeocr.backend.utils.cpu_info.can_safely_enable_onednn"
        ) as mock_probe:
            mock_probe.return_value = (False, "unsupported instruction set")
            assert OCRService._decide_enable_mkldnn("cpu") is False

    def test_probe_exception_defaults_false(self):
        """探测抛异常时保守禁用"""
        with patch(
            "vibeocr.backend.utils.cpu_info.can_safely_enable_onednn"
        ) as mock_probe:
            mock_probe.side_effect = RuntimeError("probe crashed")
            assert OCRService._decide_enable_mkldnn("cpu") is False
