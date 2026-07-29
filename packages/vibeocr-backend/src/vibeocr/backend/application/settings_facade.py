"""设置应用服务 facade。

封装设置 adapter（如 ConfigManager），对外暴露 SettingsApplication 接口。
不发 Qt signal，不接触 widget。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from vibeocr.backend.application.contracts import (
    SettingsError,
    SettingsSnapshot,
)


@runtime_checkable
class SettingsAdapter(Protocol):
    """设置 adapter 协议：facade 委托的实际执行者。"""

    def get_snapshot(self) -> SettingsSnapshot: ...


class SettingsFacade:
    """设置应用服务实现。

    通过注入的 SettingsAdapter 读取设置快照，包装异常为 SettingsError。
    """

    def __init__(self, adapter: SettingsAdapter) -> None:
        self._adapter = adapter

    def get_snapshot(self) -> SettingsSnapshot:
        """获取当前设置快照（只读）。

        Returns:
            SettingsSnapshot。

        Raises:
            SettingsError: adapter 异常。
        """
        try:
            return self._adapter.get_snapshot()
        except SettingsError:
            raise
        except Exception as e:
            raise SettingsError(f"get settings snapshot failed: {e}") from e
