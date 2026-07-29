# src/vibeocr/core/pipelines/registry.py
"""Pipeline 注册表

定义 PipelineSpec 数据类和 PipelineRegistry 注册表，
用于集中管理所有管道的元数据和工厂函数。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class PipelineSpec:
    """管道规格

    描述一个 OCR 管道的完整元数据，包括名称、描述、选项类和工厂函数。

    Attributes:
        name: 唯一标识符 (e.g. "OCR", "TABLE_RECOGNITION")
        display_name: UI 友好名称 (e.g. "通用 OCR", "表格")
        description: 管道描述
        options_class: 该管道对应的 Options dataclass 类
        create_pipeline: 工厂函数，签名 (device: str, **kwargs) -> 管道实例。
            额外 kwargs 透传给底层 PaddleX 管道构造器（例如 enable_mkldnn）。
        recognize: 执行识别的函数，签名 (service, image, options) -> result
        recognize_batch: 批量识别的函数，签名
            (service, images: list, options) -> list[result]。
            None 表示该管道不支持批量识别（调用方需回退逐张）。
    """

    name: str
    display_name: str
    description: str
    options_class: type
    create_pipeline: Callable[..., Any]
    recognize: Callable[..., Any]
    recognize_batch: Callable[..., Any] | None = None


class PipelineRegistry:
    """管道注册表

    基于字典的简单注册表，用于按名称查找 PipelineSpec。
    """

    def __init__(self) -> None:
        self._specs: dict[str, PipelineSpec] = {}

    def register(self, spec: PipelineSpec) -> None:
        """注册一个管道规格

        Args:
            spec: 要注册的 PipelineSpec
        """
        self._specs[spec.name] = spec

    def get(self, name: str) -> PipelineSpec:
        """按名称获取管道规格

        Args:
            name: 管道唯一标识符

        Returns:
            对应的 PipelineSpec

        Raises:
            KeyError: 管道未注册时抛出
        """
        if name not in self._specs:
            raise KeyError(f"Pipeline '{name}' not registered")
        return self._specs[name]

    def list_all(self) -> list[PipelineSpec]:
        """获取所有已注册的管道规格

        Returns:
            PipelineSpec 列表
        """
        return list(self._specs.values())

    def list_display_names(self) -> list[str]:
        """获取所有已注册管道的显示名称

        Returns:
            显示名称字符串列表
        """
        return [s.display_name for s in self._specs.values()]

    def has(self, name: str) -> bool:
        """检查指定名称的管道是否已注册

        Args:
            name: 管道唯一标识符

        Returns:
            是否已注册
        """
        return name in self._specs
