"""OCRServiceBase 测试"""

from typing import Any
from unittest.mock import Mock

import pytest

from vibeocr.backend.services.ocr_service_base import OCRServiceBase


class ConcreteOCRService(OCRServiceBase):
    """用于测试的具体 OCR 服务实现"""

    def _init_gpu(self) -> None:
        """初始化 GPU"""
        self._device = "cpu"

    def recognize(self, image: Any, options: Any = None) -> dict:  # type: ignore[override]
        """执行 OCR

        测试桩返回 dict（与基类 OCRResult 契约的简化）；忽略 options 参数。
        返回类型与基类不一致，此处显式 ignore。
        """
        return {"text": "test result"}

    def is_ready(self) -> bool:
        """检查是否就绪"""
        return self._device is not None


class TestOCRServiceBase:
    """OCRServiceBase 测试"""

    @pytest.fixture
    def service(self):
        """创建测试服务实例"""
        return ConcreteOCRService()

    def test_service_creation(self, service):
        """测试服务创建"""
        assert service is not None
        assert service._device is None
        assert service._pipelines == {}

    def test_device_property(self, service):
        """测试设备属性"""
        assert service.device is None

        service._device = "cuda"
        assert service.device == "cuda"

    def test_pipelines_property(self, service):
        """测试管道属性"""
        assert service.pipelines == {}

        service._pipelines["test"] = Mock()
        assert "test" in service.pipelines

    def test_set_status_callback(self, service):
        """测试设置状态回调"""
        callback = Mock()
        service.set_status_callback(callback)

        assert service._status_callback is callback

    def test_notify_status_calls_callback(self, service):
        """测试状态通知调用回调"""
        callback = Mock()
        service.set_status_callback(callback)

        service._notify_status("test_stage", "test_message")

        callback.assert_called_once_with("test_stage", "test_message")

    def test_notify_status_ignores_callback_error(self, service):
        """测试状态通知忽略回调错误"""

        def error_callback(stage, message):
            raise ValueError("Callback error")

        service.set_status_callback(error_callback)

        # 不应该抛出异常
        service._notify_status("test", "message")

    def test_notify_status_without_callback(self, service):
        """测试无回调时的状态通知"""
        service._status_callback = None

        # 不应该抛出异常
        service._notify_status("test", "message")

    def test_preload_pipelines_empty_list(self, service):
        """测试预加载空管道列表"""
        results = service.preload_pipelines([])

        assert results == {}

    def test_preload_pipelines(self, service):
        """测试预加载管道"""
        # 添加一个管道到缓存
        service._pipelines["existing"] = Mock()

        results = service.preload_pipelines(["existing"])

        assert results["existing"] is True

    def test_get_pipeline(self, service):
        """测试获取管道"""
        pipeline = Mock()
        service._pipelines["test"] = pipeline

        result = service.get_pipeline("test")

        assert result is pipeline

    def test_get_pipeline_not_found(self, service):
        """测试获取不存在的管道"""
        result = service.get_pipeline("nonexistent")

        assert result is None

    def test_clear_pipelines(self, service):
        """测试清除管道"""
        service._pipelines["test1"] = Mock()
        service._pipelines["test2"] = Mock()

        service.clear_pipelines()

        assert service._pipelines == {}

    def test_shutdown(self, service):
        """测试关闭服务"""
        service._device = "cuda"
        service._pipelines["test"] = Mock()

        service.shutdown()

        assert service._device is None
        assert service._pipelines == {}


class TestOCRServiceBaseAbstract:
    """OCRServiceBase 抽象方法测试"""

    def test_cannot_instantiate_base_class(self):
        """测试不能直接实例化基类"""
        with pytest.raises(TypeError):
            OCRServiceBase()  # type: ignore[abstract]

    def test_subclass_must_implement_abstract_methods(self):
        """测试子类必须实现抽象方法"""

        class IncompleteService(OCRServiceBase):
            def _init_gpu(self) -> None:
                pass

            # 缺少 recognize 和 is_ready

        with pytest.raises(TypeError):
            IncompleteService()  # type: ignore[abstract]


class TestOCRServiceBaseDefaults:
    """基类默认实现方法覆盖（recognize_batch/preload/release_idle/set_ttls）。"""

    @pytest.fixture
    def service(self):
        return ConcreteOCRService()

    def test_recognize_batch_delegates_to_recognize(self, service):
        """recognize_batch 默认逐个调用 recognize（line 102）。"""
        results = service.recognize_batch(["img1", "img2"])
        assert len(results) == 2
        assert all(r == {"text": "test result"} for r in results)

    def test_preload_pipelines_swallows_exceptions(self, service):
        """preload_pipelines 默认实现吞掉异常并记 False（line 126-127）。"""
        # _preload_pipeline 默认会尝试 get_or_create_pipeline，可能抛
        results = service.preload_pipelines(["OCR", "Table"])
        assert set(results.keys()) == {"OCR", "Table"}
        assert all(isinstance(v, bool) for v in results.values())

    def test_release_idle_returns_and_clears(self, service):
        """release_pipelines 默认返回当前管道名并清空（line 165-167）。"""
        # 注入一个伪管道
        service._pipelines["OCR"] = object()
        names = service.release_pipelines()
        assert names == ["OCR"]
        assert len(service._pipelines) == 0

    def test_set_pipeline_ttls_default_returns_false(self, service):
        """set_pipeline_ttls 基类默认返回 False（line 177）。"""
        assert service.set_pipeline_ttls({"OCR": 300}) is False
