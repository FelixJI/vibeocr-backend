"""Tests for OCRService registry integration.

Verifies that:
- get_or_create_pipeline() caches pipeline instances
- get_or_create_pipeline() uses registry's spec.create_pipeline when available
- get_or_create_pipeline() falls back to old _create_pipeline for unregistered names
- recognize() dispatches to registered pipeline specs via registry
- recognize() falls back to spec.recognize (逐张) for pipelines without recognize_batch
- 表格/公式等无 recognize_batch 的管道走对应 spec.recognize，而非被当 OCR
- Old OCROptions (enum pipeline) still works end-to-end
"""

from unittest.mock import MagicMock, patch

import pytest

from vibeocr.backend.core.pipelines import OCRPipeline
from vibeocr.backend.models.ocr_options import OCROptions
from vibeocr.backend.models.ocr_result import OCRResult
from vibeocr.backend.services.ocr_service import OCRService
from vibeocr.backend.services.pipeline_cache_manager import PipelineCacheManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ocr_result(**overrides) -> OCRResult:
    """Create a minimal OCRResult suitable for tests."""
    defaults = {
        "raw_text": "hello",
        "markdown_text": "hello",
        "html_text": "hello",
        "text_with_scores": [("hello", 0.95)],
        "avg_score": 0.95,
        "low_confidence_items": [],
        "pipeline_type": "OCR",
        "images": {},
        "text_blocks": [],
        "content_list": [],
        "image_width": 100,
        "image_height": 50,
        "preproc_angle": 0,
        "preprocessed_image": None,
        "preproc_img_w": 0,
        "preproc_img_h": 0,
    }
    defaults.update(overrides)
    return OCRResult(**defaults)


def _known_onednn_pir_error() -> NotImplementedError:
    return NotImplementedError(
        "ConvertPirAttribute2RuntimeAttribute not support "
        "[pir::ArrayAttribute<pir::DoubleAttribute>] "
        "(at ..\\instruction\\onednn\\onednn_instruction.cc:118)"
    )


# ---------------------------------------------------------------------------
# get_or_create_pipeline
# ---------------------------------------------------------------------------


class TestGetOrCreatePipeline:
    """Tests for OCRService.get_or_create_pipeline()."""

    def setup_method(self):
        """Reset singleton state before each test."""
        OCRService._reset()

    def teardown_method(self):
        """Reset singleton state after each test."""
        OCRService._reset()

    def test_caches_pipeline_instance(self):
        """Pipeline instances are cached and returned on subsequent calls."""
        service = OCRService()
        mock_pipeline = MagicMock(name="cached_ocr_pipeline")
        service._pipelines = {"OCR": mock_pipeline}

        result = service.get_or_create_pipeline("OCR")
        assert result is mock_pipeline

    def test_creates_pipeline_via_registry(self):
        """Uses spec.create_pipeline from registry when pipeline is registered.

        On CPU, enable_mkldnn must be forwarded to the factory. The probe is
        mocked so the assertion is stable regardless of the host's CPU/paddle
        version (avoids order-dependent leakage of the process-level cache).
        """
        service = OCRService()
        service._pipelines = {}

        mock_pipeline = MagicMock(name="ocr_pipeline_from_registry")
        mock_spec = MagicMock()
        mock_spec.create_pipeline.return_value = mock_pipeline

        mock_registry = MagicMock()
        mock_registry.has.return_value = True
        mock_registry.get.return_value = mock_spec

        with (
            patch("vibeocr.backend.services.ocr_service.OCRService._setup_cuda_dll_path"),
            patch(
                "vibeocr.backend.services.ocr_service.OCRService._get_device",
                return_value="cpu",
            ),
            patch(
                "vibeocr.backend.core.pipelines.get_registry",
                return_value=mock_registry,
            ),
            patch(
                "vibeocr.backend.services.ocr_service.OCRService._create_pipeline"
            ) as mock_old_create,
            # 固定探测结果为 False，断言不依赖宿主环境
            patch(
                "vibeocr.backend.utils.cpu_info.can_safely_enable_onednn",
                return_value=(False, "mocked: paddle 3.3 黑名单"),
            ),
        ):
            result = service.get_or_create_pipeline("OCR")

        # Should use registry, not old _create_pipeline.
        # On CPU the factory must receive enable_mkldnn=False (mocked probe).
        mock_spec.create_pipeline.assert_called_once_with("cpu", enable_mkldnn=False)
        mock_old_create.assert_not_called()
        assert result is mock_pipeline
        assert "OCR" in service._pipelines

    def test_cpu_forwards_enable_mkldnn_true_when_probe_says_safe(self):
        """When the probe says oneDNN is safe, enable_mkldnn=True is forwarded.

        Covers the True branch so the assertion doesn't silently rot until a
        future paddle upgrade makes the probe return True on the host.
        """
        service = OCRService()
        service._pipelines = {}

        mock_pipeline = MagicMock(name="ocr_pipeline_mkldnn")
        mock_spec = MagicMock()
        mock_spec.create_pipeline.return_value = mock_pipeline

        mock_registry = MagicMock()
        mock_registry.has.return_value = True
        mock_registry.get.return_value = mock_spec

        with (
            patch("vibeocr.backend.services.ocr_service.OCRService._setup_cuda_dll_path"),
            patch(
                "vibeocr.backend.services.ocr_service.OCRService._get_device",
                return_value="cpu",
            ),
            patch(
                "vibeocr.backend.core.pipelines.get_registry",
                return_value=mock_registry,
            ),
            patch(
                "vibeocr.backend.utils.cpu_info.can_safely_enable_onednn",
                return_value=(True, "mocked: AVX2 + 新 paddle"),
            ),
        ):
            service.get_or_create_pipeline("OCR")

        mock_spec.create_pipeline.assert_called_once_with("cpu", enable_mkldnn=True)

    def test_gpu_device_omits_enable_mkldnn(self):
        """On GPU the factory is called with device only — no enable_mkldnn kwarg."""
        service = OCRService()
        service._pipelines = {}

        mock_pipeline = MagicMock(name="gpu_pipeline")
        mock_spec = MagicMock()
        mock_spec.create_pipeline.return_value = mock_pipeline

        mock_registry = MagicMock()
        mock_registry.has.return_value = True
        mock_registry.get.return_value = mock_spec

        with (
            patch("vibeocr.backend.services.ocr_service.OCRService._setup_cuda_dll_path"),
            patch(
                "vibeocr.backend.services.ocr_service.OCRService._get_device",
                return_value="gpu",
            ),
            patch(
                "vibeocr.backend.core.pipelines.get_registry",
                return_value=mock_registry,
            ),
        ):
            service.get_or_create_pipeline("OCR")

        mock_spec.create_pipeline.assert_called_once_with("gpu")

    def test_falls_back_to_old_create_pipeline(self):
        """Falls back to _create_pipeline for unregistered pipeline names."""
        service = OCRService()
        service._pipelines = {}

        mock_pipeline = MagicMock(name="fallback_pipeline")
        mock_registry = MagicMock()
        mock_registry.has.return_value = False

        with (
            patch("vibeocr.backend.services.ocr_service.OCRService._setup_cuda_dll_path"),
            patch(
                "vibeocr.backend.core.pipelines.get_registry",
                return_value=mock_registry,
            ),
            patch.object(
                service, "_create_pipeline", return_value=mock_pipeline
            ) as mock_create,
        ):
            result = service.get_or_create_pipeline("OCR")

        mock_create.assert_called_once_with(OCRPipeline.OCR)
        assert result is mock_pipeline

    def test_raises_for_unknown_pipeline_name(self):
        """Raises ValueError for a pipeline name not in registry or OCRPipeline enum."""
        service = OCRService()
        service._pipelines = {}

        mock_registry = MagicMock()
        mock_registry.has.return_value = False

        with (
            patch("vibeocr.backend.services.ocr_service.OCRService._setup_cuda_dll_path"),
            patch(
                "vibeocr.backend.core.pipelines.get_registry",
                return_value=mock_registry,
            ),
            pytest.raises(ValueError, match="不支持的管道类型"),
        ):
            service.get_or_create_pipeline("NONEXISTENT_PIPELINE")

    def test_thread_safety_double_check(self):
        """Double-checked locking: only creates once even under contention."""
        service = OCRService()
        service._pipelines = {}

        mock_pipeline = MagicMock(name="pipeline")
        mock_spec = MagicMock()
        mock_spec.create_pipeline.return_value = mock_pipeline

        mock_registry = MagicMock()
        mock_registry.has.return_value = True
        mock_registry.get.return_value = mock_spec

        with (
            patch("vibeocr.backend.services.ocr_service.OCRService._setup_cuda_dll_path"),
            patch(
                "vibeocr.backend.services.ocr_service.OCRService._get_device",
                return_value="cpu",
            ),
            patch(
                "vibeocr.backend.core.pipelines.get_registry",
                return_value=mock_registry,
            ),
            patch(
                "vibeocr.backend.utils.cpu_info.can_safely_enable_onednn",
                return_value=(False, "mocked for registry locking test"),
            ),
        ):
            # Call twice — second should return cached instance
            result1 = service.get_or_create_pipeline("OCR")
            result2 = service.get_or_create_pipeline("OCR")

        assert result1 is result2
        assert mock_spec.create_pipeline.call_count == 1


# ---------------------------------------------------------------------------
# get_pipeline backward compat
# ---------------------------------------------------------------------------


class TestGetPipelineBackwardCompat:
    """Tests that get_pipeline(OCRPipeline) still works."""

    def setup_method(self):
        OCRService._reset()

    def teardown_method(self):
        OCRService._reset()

    def test_get_pipeline_delegates_to_get_or_create(self):
        """get_pipeline delegates to get_or_create_pipeline."""
        service = OCRService()
        mock_pipeline = MagicMock(name="ocr_pipeline")
        service._pipelines = {"OCR": mock_pipeline}

        result = service.get_pipeline(OCRPipeline.OCR)
        assert result is mock_pipeline

    def test_get_pipeline_pp_structure(self):
        """get_pipeline works for PP-StructureV3."""
        service = OCRService()
        mock_pipeline = MagicMock(name="pp_structure_pipeline")
        service._pipelines = {"PP-StructureV3": mock_pipeline}

        result = service.get_pipeline(OCRPipeline.PP_STRUCTURE_V3)
        assert result is mock_pipeline


# ---------------------------------------------------------------------------
# recognize() registry dispatch
# ---------------------------------------------------------------------------


class TestRecognizeRegistryDispatch:
    """Tests for recognize() dispatching via registry."""

    def setup_method(self):
        OCRService._reset()

    def teardown_method(self):
        OCRService._reset()

    def test_dispatches_to_registered_spec(self):
        """recognize() dispatches to spec.recognize_batch for registered pipelines."""
        service = OCRService()

        mock_result = _make_ocr_result()

        mock_spec = MagicMock()
        mock_spec.recognize_batch.return_value = [mock_result]

        mock_registry = MagicMock()
        mock_registry.has.return_value = True
        mock_registry.get.return_value = mock_spec

        with patch(
            "vibeocr.backend.core.pipelines.get_registry",
            return_value=mock_registry,
        ):
            import numpy as np

            img = np.zeros((50, 100, 3), dtype=np.uint8)
            opts = OCROptions(pipeline=OCRPipeline.OCR)
            result = service.recognize(img, opts)

        mock_spec.recognize_batch.assert_called_once()
        # First arg is service, second is images list, third is options
        call_args = mock_spec.recognize_batch.call_args
        assert call_args[0][0] is service
        assert len(call_args[0][1]) == 1
        assert result.raw_text == "hello"

    def test_dispatches_to_spec_recognize_when_no_batch(self):
        """管道注册了但无 recognize_batch 时，逐张走 spec.recognize（统一分发）。

        这覆盖 OCR / PP-StructureV3 / PaddleOCR-VL / 表格 / 公式 等所有
        未提供 recognize_batch 的管道——它们都应通过 spec.recognize 分发，
        而非硬编码的私有方法（曾导致表格/公式被当 OCR 处理）。
        """
        service = OCRService()

        mock_result = _make_ocr_result()
        mock_spec = MagicMock()
        mock_spec.recognize_batch = None  # 该管道无批量接口
        mock_spec.recognize.return_value = mock_result

        mock_registry = MagicMock()
        mock_registry.has.return_value = True
        mock_registry.get.return_value = mock_spec

        with patch(
            "vibeocr.backend.core.pipelines.get_registry",
            return_value=mock_registry,
        ):
            import numpy as np

            img = np.zeros((50, 100, 3), dtype=np.uint8)
            opts = OCROptions(pipeline=OCRPipeline.OCR)
            result = service.recognize(img, opts)

        mock_spec.recognize.assert_called_once()
        assert result.raw_text == "hello"

    def test_table_pipeline_dispatches_to_table_spec(self):
        """表格管道走 TABLE_RECOGNITION spec.recognize，不被当 OCR 处理。

        回归测试：修复前 dispatch 的 else 兜底分支硬编码 if/elif 漏掉表格，
        导致点表格按钮实际走 _recognize_ocr（纯文字识别）。
        """
        service = OCRService()

        table_result = _make_ocr_result(
            raw_text="<table>", pipeline_type="TABLE_RECOGNITION"
        )
        mock_table_spec = MagicMock()
        mock_table_spec.recognize_batch = None
        mock_table_spec.recognize.return_value = table_result

        mock_registry = MagicMock()
        mock_registry.has.return_value = True
        mock_registry.get.return_value = mock_table_spec

        with patch(
            "vibeocr.backend.core.pipelines.get_registry",
            return_value=mock_registry,
        ):
            import numpy as np

            img = np.zeros((50, 100, 3), dtype=np.uint8)
            opts = OCROptions(pipeline=OCRPipeline.TABLE_RECOGNITION)
            result = service.recognize(img, opts)

        # 必须调用表格 spec 的 recognize，而非 OCR
        mock_table_spec.recognize.assert_called_once()
        call_args = mock_table_spec.recognize.call_args[0]
        assert call_args[0] is service  # service
        assert opts.pipeline == OCRPipeline.TABLE_RECOGNITION
        assert result.pipeline_type == "TABLE_RECOGNITION"

    def test_formula_pipeline_dispatches_to_formula_spec(self):
        """公式管道走 FORMULA_RECOGNITION spec.recognize（同根因回归测试）。"""
        service = OCRService()

        formula_result = _make_ocr_result(
            raw_text="E=mc^2", pipeline_type="FORMULA_RECOGNITION"
        )
        mock_formula_spec = MagicMock()
        mock_formula_spec.recognize_batch = None
        mock_formula_spec.recognize.return_value = formula_result

        mock_registry = MagicMock()
        mock_registry.has.return_value = True
        mock_registry.get.return_value = mock_formula_spec

        with patch(
            "vibeocr.backend.core.pipelines.get_registry",
            return_value=mock_registry,
        ):
            import numpy as np

            img = np.zeros((50, 100, 3), dtype=np.uint8)
            opts = OCROptions(pipeline=OCRPipeline.FORMULA_RECOGNITION)
            result = service.recognize(img, opts)

        mock_formula_spec.recognize.assert_called_once()
        assert result.pipeline_type == "FORMULA_RECOGNITION"

    def test_bytes_input_converted_before_dispatch(self):
        """bytes input is converted to numpy before dispatching to registry."""
        service = OCRService()

        mock_result = _make_ocr_result()
        mock_spec = MagicMock()
        mock_spec.recognize_batch.return_value = [mock_result]

        mock_registry = MagicMock()
        mock_registry.has.return_value = True
        mock_registry.get.return_value = mock_spec

        with patch(
            "vibeocr.backend.core.pipelines.get_registry",
            return_value=mock_registry,
        ):
            # Use a valid PNG bytes input
            import io

            from PIL import Image

            img = Image.new("RGB", (100, 50), color="white")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            png_bytes = buf.getvalue()

            opts = OCROptions(pipeline=OCRPipeline.OCR)
            service.recognize(png_bytes, opts)

        # The dispatched image should be numpy array, not bytes
        call_args = mock_spec.recognize_batch.call_args
        dispatched_images = call_args[0][1]
        assert len(dispatched_images) == 1
        assert hasattr(dispatched_images[0], "shape")  # numpy array

    def test_bbox_normalization_preserved_with_registry(self):
        """Bbox normalization in recognize() still works with registry dispatch."""
        from vibeocr.backend.models.ocr_result import TextBlock

        service = OCRService()

        # Result with pixel-space bbox
        tb = TextBlock(text="hello", score=0.9, bbox=(10.0, 20.0, 90.0, 40.0))
        mock_result = _make_ocr_result(
            text_blocks=[tb],
            preproc_img_w=100,
            preproc_img_h=50,
        )

        mock_spec = MagicMock()
        mock_spec.recognize_batch.return_value = [mock_result]

        mock_registry = MagicMock()
        mock_registry.has.return_value = True
        mock_registry.get.return_value = mock_spec

        with patch(
            "vibeocr.backend.core.pipelines.get_registry",
            return_value=mock_registry,
        ):
            import numpy as np

            img = np.zeros((50, 100, 3), dtype=np.uint8)
            opts = OCROptions(pipeline=OCRPipeline.OCR)
            result = service.recognize(img, opts)

        # Bbox should be normalized to [0-1000]
        assert result.text_blocks[0].bbox is not None
        x0, y0, x1, y1 = result.text_blocks[0].bbox
        assert 0 <= x0 <= 1000
        assert 0 <= y0 <= 1000
        assert 0 <= x1 <= 1000
        assert 0 <= y1 <= 1000


# ---------------------------------------------------------------------------
# recognize_batch()
# ---------------------------------------------------------------------------


class TestRecognizeBatch:
    """Tests for OCRService.recognize_batch() — batched predict(list) path."""

    def setup_method(self):
        OCRService._reset()

    def teardown_method(self):
        OCRService._reset()

    def test_dispatches_to_spec_recognize_batch(self):
        """recognize_batch() dispatches to spec.recognize_batch for OCR pipeline."""
        service = OCRService()

        results = [_make_ocr_result(raw_text=f"img{i}") for i in range(3)]
        mock_spec = MagicMock()
        mock_spec.recognize_batch.return_value = results

        mock_registry = MagicMock()
        mock_registry.has.return_value = True
        mock_registry.get.return_value = mock_spec

        with patch(
            "vibeocr.backend.core.pipelines.get_registry",
            return_value=mock_registry,
        ):
            import numpy as np

            images = [np.zeros((50, 100, 3), dtype=np.uint8) for _ in range(3)]
            opts = OCROptions(pipeline=OCRPipeline.OCR)
            got = service.recognize_batch(images, opts)

        # 单次批量调用，传入整个列表
        mock_spec.recognize_batch.assert_called_once()
        call_args = mock_spec.recognize_batch.call_args[0]
        assert call_args[0] is service
        assert len(call_args[1]) == 3  # images list
        assert [r.raw_text for r in got] == ["img0", "img1", "img2"]

    def test_returns_empty_for_empty_input(self):
        """空输入返回空列表，不调用管道。"""
        service = OCRService()
        mock_spec = MagicMock()
        mock_registry = MagicMock()
        mock_registry.has.return_value = True
        mock_registry.get.return_value = mock_spec

        with patch(
            "vibeocr.backend.core.pipelines.get_registry",
            return_value=mock_registry,
        ):
            got = service.recognize_batch([], OCROptions(pipeline=OCRPipeline.OCR))

        assert got == []
        mock_spec.recognize_batch.assert_not_called()

    def test_falls_back_to_per_image_when_no_batch_spec(self):
        """管道未注册 recognize_batch 时，逐张走 spec.recognize 并保持顺序与数量。"""
        service = OCRService()
        mock_spec = MagicMock()
        mock_spec.recognize_batch = None  # 该管道无批量接口
        mock_spec.recognize.side_effect = [
            _make_ocr_result(raw_text="a"),
            _make_ocr_result(raw_text="b"),
        ]

        mock_registry = MagicMock()
        mock_registry.has.return_value = True
        mock_registry.get.return_value = mock_spec

        with patch(
            "vibeocr.backend.core.pipelines.get_registry",
            return_value=mock_registry,
        ):
            import numpy as np

            images = [np.zeros((50, 100, 3), dtype=np.uint8) for _ in range(2)]
            got = service.recognize_batch(images, OCROptions(pipeline=OCRPipeline.OCR))

        assert [r.raw_text for r in got] == ["a", "b"]
        assert mock_spec.recognize.call_count == 2

    def test_bbox_normalization_per_image_in_batch(self):
        """批量结果中每张图的 bbox 按各自预处理图尺寸独立归一化到 [0,1000]。

        模拟两张图：图 A 预处理尺寸 100x50（bbox 像素 50,25）；图 B 200x100
        （bbox 像素 100,50）。归一化后两者都应落在 [0,1000] 中点附近。
        """
        from vibeocr.backend.models.ocr_result import TextBlock

        service = OCRService()

        tb_a = TextBlock(text="A", score=1.0, bbox=(50.0, 25.0, 50.0, 25.0))
        tb_b = TextBlock(text="B", score=1.0, bbox=(100.0, 50.0, 100.0, 50.0))
        res_a = _make_ocr_result(
            raw_text="A",
            text_blocks=[tb_a],
            preproc_img_w=100,
            preproc_img_h=50,
        )
        res_b = _make_ocr_result(
            raw_text="B",
            text_blocks=[tb_b],
            preproc_img_w=200,
            preproc_img_h=100,
        )

        mock_spec = MagicMock()
        mock_spec.recognize_batch.return_value = [res_a, res_b]
        mock_registry = MagicMock()
        mock_registry.has.return_value = True
        mock_registry.get.return_value = mock_spec

        with patch(
            "vibeocr.backend.core.pipelines.get_registry",
            return_value=mock_registry,
        ):
            import numpy as np

            images = [
                np.zeros((50, 100, 3), dtype=np.uint8),
                np.zeros((100, 200, 3), dtype=np.uint8),
            ]
            got = service.recognize_batch(images, OCROptions(pipeline=OCRPipeline.OCR))

        # 两张图各自归一化：50/100*1000=500, 25/50*1000=500
        bbox_a = got[0].text_blocks[0].bbox
        bbox_b = got[1].text_blocks[0].bbox
        assert bbox_a is not None and bbox_b is not None
        assert abs(bbox_a[0] - 500) < 1 and abs(bbox_a[1] - 500) < 1
        assert abs(bbox_b[0] - 500) < 1 and abs(bbox_b[1] - 500) < 1
        # 归一化后记录的图像尺寸为预处理尺寸
        assert got[0].image_width == 100 and got[0].image_height == 50
        assert got[1].image_width == 200 and got[1].image_height == 100

    def test_single_recognize_routes_through_batch(self):
        """单图 recognize() 内部走 recognize_batch([img])，行为等价。"""
        service = OCRService()
        res = _make_ocr_result(raw_text="single")
        mock_spec = MagicMock()
        mock_spec.recognize_batch.return_value = [res]
        mock_registry = MagicMock()
        mock_registry.has.return_value = True
        mock_registry.get.return_value = mock_spec

        with patch(
            "vibeocr.backend.core.pipelines.get_registry",
            return_value=mock_registry,
        ):
            import numpy as np

            img = np.zeros((50, 100, 3), dtype=np.uint8)
            got = service.recognize(img, OCROptions(pipeline=OCRPipeline.OCR))

        assert got.raw_text == "single"
        mock_spec.recognize_batch.assert_called_once()
        assert len(mock_spec.recognize_batch.call_args[0][1]) == 1

    def test_known_onednn_failure_rebuilds_disabled_and_retries_once(self):
        """已知 PIR/oneDNN 异常会释放管道，以 mkldnn=False 重建并成功重试。"""
        import numpy as np

        service = OCRService()
        service._pipelines = {}
        service._cache_manager = PipelineCacheManager(service, {}, max_heavy=1)
        type(service)._onednn_safe_cache = True

        first_pipeline = MagicMock(name="onednn_pipeline")
        fallback_pipeline = MagicMock(name="plain_cpu_pipeline")
        mock_spec = MagicMock()
        mock_spec.create_pipeline.side_effect = [first_pipeline, fallback_pipeline]

        dispatch_count = 0

        def recognize_batch(svc, _images, _options):
            nonlocal dispatch_count
            dispatch_count += 1
            svc.get_or_create_pipeline("OCR")
            if dispatch_count == 1:
                raise _known_onednn_pir_error()
            return [_make_ocr_result(raw_text="fallback-ok")]

        mock_spec.recognize_batch.side_effect = recognize_batch
        mock_registry = MagicMock()
        mock_registry.has.return_value = True
        mock_registry.get.return_value = mock_spec

        with (
            patch("vibeocr.backend.services.ocr_service.OCRService._setup_cuda_dll_path"),
            patch(
                "vibeocr.backend.services.ocr_service.OCRService._get_device",
                return_value="cpu",
            ),
            patch("vibeocr.backend.core.pipelines.get_registry", return_value=mock_registry),
        ):
            got = service.recognize_batch(
                [np.zeros((50, 100, 3), dtype=np.uint8)],
                OCROptions(pipeline=OCRPipeline.OCR),
            )

        assert got[0].raw_text == "fallback-ok"
        assert dispatch_count == 2
        assert mock_spec.create_pipeline.call_args_list == [
            (("cpu",), {"enable_mkldnn": True}),
            (("cpu",), {"enable_mkldnn": False}),
        ]
        assert service._pipelines["OCR"] is fallback_pipeline
        assert type(service)._onednn_runtime_disabled is True
        assert type(service)._onednn_safe_cache is False

    def test_non_onednn_failure_is_not_retried(self):
        """普通业务异常不触发兼容回退，也不改变 oneDNN 决策。"""
        import numpy as np

        service = OCRService()
        type(service)._onednn_safe_cache = True
        mock_spec = MagicMock()
        mock_spec.recognize_batch.side_effect = ValueError("bad input")
        mock_registry = MagicMock()
        mock_registry.has.return_value = True
        mock_registry.get.return_value = mock_spec

        with (
            patch("vibeocr.backend.core.pipelines.get_registry", return_value=mock_registry),
            pytest.raises(ValueError, match="bad input"),
        ):
            service.recognize_batch(
                [np.zeros((50, 100, 3), dtype=np.uint8)],
                OCROptions(pipeline=OCRPipeline.OCR),
            )

        assert mock_spec.recognize_batch.call_count == 1
        assert type(service)._onednn_runtime_disabled is False
        assert type(service)._onednn_safe_cache is True

    def test_known_onednn_failure_on_retry_is_propagated_without_loop(self):
        """禁用重建后若仍失败，第二次异常原样抛出且不继续循环。"""
        import numpy as np

        service = OCRService()
        service._pipelines = {}
        service._cache_manager = PipelineCacheManager(service, {}, max_heavy=1)
        type(service)._onednn_safe_cache = True
        mock_spec = MagicMock()
        mock_spec.create_pipeline.side_effect = [MagicMock(), MagicMock()]

        def recognize_batch(svc, _images, _options):
            svc.get_or_create_pipeline("OCR")
            raise _known_onednn_pir_error()

        mock_spec.recognize_batch.side_effect = recognize_batch
        mock_registry = MagicMock()
        mock_registry.has.return_value = True
        mock_registry.get.return_value = mock_spec

        with (
            patch("vibeocr.backend.services.ocr_service.OCRService._setup_cuda_dll_path"),
            patch(
                "vibeocr.backend.services.ocr_service.OCRService._get_device",
                return_value="cpu",
            ),
            patch("vibeocr.backend.core.pipelines.get_registry", return_value=mock_registry),
            pytest.raises(NotImplementedError, match="ConvertPirAttribute"),
        ):
            service.recognize_batch(
                [np.zeros((50, 100, 3), dtype=np.uint8)],
                OCROptions(pipeline=OCRPipeline.OCR),
            )

        assert mock_spec.recognize_batch.call_count == 2
        assert mock_spec.create_pipeline.call_count == 2

    def test_gpu_path_does_not_apply_cpu_onednn_fallback(self):
        """未启用 CPU oneDNN 的路径即使出现同文案异常也不执行重建。"""
        import numpy as np

        service = OCRService()
        type(service)._onednn_safe_cache = None
        mock_spec = MagicMock()
        mock_spec.recognize_batch.side_effect = _known_onednn_pir_error()
        mock_registry = MagicMock()
        mock_registry.has.return_value = True
        mock_registry.get.return_value = mock_spec

        with (
            patch("vibeocr.backend.core.pipelines.get_registry", return_value=mock_registry),
            pytest.raises(NotImplementedError, match="ConvertPirAttribute"),
        ):
            service.recognize_batch(
                [np.zeros((50, 100, 3), dtype=np.uint8)],
                OCROptions(pipeline=OCRPipeline.OCR),
            )

        assert mock_spec.recognize_batch.call_count == 1
        assert type(service)._onednn_runtime_disabled is False
