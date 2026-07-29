# src/vibeocr/core/pipelines/base_options.py
"""管道选项基类

所有管道特定选项类的基类，提供 to_dict / from_dict / copy 通用实现。
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any


@dataclass
class BasePipelineOptions:
    """管道选项基类

    所有管道特定的选项类都应继承此类。
    通过 dataclass fields 自动实现序列化/反序列化。
    """

    pipeline: str = ""

    def to_dict(self) -> dict[str, Any]:
        """转换为字典

        Returns:
            包含所有字段的字典
        """
        result = {}
        for f in fields(self):
            val = getattr(self, f.name)
            result[f.name] = val
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BasePipelineOptions:
        """从字典创建实例

        Args:
            data: 包含选项的字典

        Returns:
            新的实例
        """
        kwargs = {}
        for f in fields(cls):
            if f.name in data:
                kwargs[f.name] = data[f.name]
        return cls(**kwargs)

    def copy(self, **updates) -> BasePipelineOptions:
        """创建副本，可选地更新部分字段

        Args:
            **updates: 要更新的字段

        Returns:
            新的实例
        """
        data = self.to_dict()
        data.update(updates)
        return self.__class__.from_dict(data)
