"""OCR 服务抽象基类

定义所有 OCR 服务实现必须遵循的接口。
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from vibeocr.backend.models.ocr_options import OCROptions
    from vibeocr.backend.models.ocr_result import OCRResult


class OCRServiceBase(ABC):
    """OCR 服务抽象基类

    定义所有 OCR 服务实现必须遵循的接口。
    支持：
    - 单图识别
    - 批量处理
    - 管道预加载
    - GPU/CPU 模式切换

    子类需要实现：
    - _init_gpu(): 初始化 GPU 环境
    - recognize(): 执行 OCR 识别
    - is_ready(): 检查服务是否就绪
    """

    def __init__(self) -> None:
        self._device: str | None = None
        self._pipelines: dict[str, Any] = {}
        self._status_callback: Callable[[str, str], None] | None = None

    @property
    def device(self) -> str | None:
        """获取当前设备"""
        return self._device

    @property
    def pipelines(self) -> dict[str, Any]:
        """获取管道缓存"""
        return self._pipelines

    def set_status_callback(self, callback: Callable[[str, str], None] | None) -> None:
        """设置状态回调函数

        Args:
            callback: 回调函数，接收 (stage, message) 参数
        """
        self._status_callback = callback

    def _notify_status(self, stage: str, message: str) -> None:
        """通知状态变化"""
        if self._status_callback:
            try:
                self._status_callback(stage, message)
            except Exception:
                pass  # 忽略回调错误

    @abstractmethod
    def _init_gpu(self) -> None:
        """初始化 GPU 环境

        子类必须实现此方法来检测和配置 GPU。
        """

    @abstractmethod
    def recognize(
        self,
        image: Any,
        options: Optional["OCROptions"] = None,
    ) -> "OCRResult":
        """执行 OCR 识别

        Args:
            image: 输入图像（PIL Image, numpy 数组, 路径或字节数据）
            options: OCR 识别选项

        Returns:
            OCRResult 对象
        """

    def recognize_batch(
        self,
        images: list,
        options: Optional["OCROptions"] = None,
    ) -> list:
        """批量识别多张图像（默认实现：逐张调用 recognize）。

        子类可重写以利用 PaddleOCR 的 predict(list) 批处理，提升多页吞吐。
        结果顺序与 images 一致。

        Args:
            images: 输入图像列表。
            options: OCR 识别选项。

        Returns:
            OCRResult 列表，顺序与 images 一致。
        """
        return [self.recognize(img, options) for img in images]

    @abstractmethod
    def is_ready(self) -> bool:
        """检查服务是否就绪

        Returns:
            服务是否可以执行 OCR
        """

    def preload_pipelines(self, pipelines: list[str]) -> dict[str, bool]:
        """预加载管道

        Args:
            pipelines: 要预加载的管道名称列表

        Returns:
            {pipeline_name: success} 字典
        """
        results = {}
        for pipeline_name in pipelines:
            try:
                self._preload_pipeline(pipeline_name)
                results[pipeline_name] = True
            except Exception:
                results[pipeline_name] = False
        return results

    def _preload_pipeline(self, pipeline_name: str) -> None:
        """预加载单个管道

        Args:
            pipeline_name: 管道名称
        """
        # 默认实现：如果管道已存在则跳过
        if pipeline_name in self._pipelines:
            return
        # 子类可以重写此方法来实现实际的预加载逻辑

    def get_pipeline(self, pipeline_name: str) -> Any | None:
        """获取管道实例

        Args:
            pipeline_name: 管道名称

        Returns:
            管道实例，如果不存在返回 None
        """
        return self._pipelines.get(pipeline_name)

    def clear_pipelines(self) -> None:
        """清除所有管道缓存"""
        self._pipelines.clear()

    def release_pipelines(self, heavy_only: bool = True) -> list[str]:
        """释放管道缓存。heavy_only=True 只释放重管道。

        子类（OCRService 直连 / OCRServiceSubprocess 经 RPC）应重写此方法。
        基类默认实现仅清空全部缓存。

        Returns:
            被释放的管道名列表。
        """
        names = list(self._pipelines.keys())
        self.clear_pipelines()
        return names

    def set_pipeline_ttls(self, pipeline_ttls: dict[str, int]) -> bool:
        """设置每管道 TTL 闲置回收时间。

        子类应重写此方法。基类默认空实现。

        Returns:
            是否设置成功。
        """
        return False

    def shutdown(self) -> None:
        """关闭服务，释放资源

        子类可以重写此方法来实现清理逻辑。
        """
        self.clear_pipelines()
        self._device = None
