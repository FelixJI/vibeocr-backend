"""MinerUExecutor: drive ``MINERU_PARSE`` jobs through MinerUProcessAdapter.

Shares the backend-agnostic job state machine with
:class:`~vibeocr.backend.supervisor.inference.paddle_executor.AdapterExecutor`; only
the adapter type differs. The supervisor routes ``MINERU_PARSE`` jobs to this
executor via :class:`~vibeocr.backend.supervisor.inference.composite_executor.CompositeExecutor`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from .mineru_adapter import MinerUProcessAdapter

from .budgets import InputItem
from .paddle_executor import AdapterExecutor


class MinerUExecutor(AdapterExecutor):
    """Drives MinerU document-parse jobs through a MinerUProcessAdapter."""

    def __init__(
        self,
        adapter_factory: Callable[[], MinerUProcessAdapter],
        **coordinator_options: Any,
    ) -> None:
        super().__init__(adapter_factory, **coordinator_options)

    def _recognize_many(
        self, record: Any, items: list[InputItem], options: Any
    ) -> list[dict[str, Any]]:
        return self.adapter.recognize_many(
            items,
            options=options,
            cancelled=lambda: record.cancel_requested_at is not None,
        )

    def _commit_payload(
        self, record: Any, item: InputItem, payload_type: str, payload: dict
    ) -> None:
        if "mineru_error" in payload:
            record.commit_item_failure(
                item.item_id,
                error_code="BACKEND_UNAVAILABLE",
                error=payload["mineru_error"],
            )
            return
        super()._commit_payload(record, item, payload_type, payload)

    def configure_settings(self, snapshot: Any):
        """连接配置在共同位置生效，即使惰性 adapter 尚未构建。

        adapter 已构建时由 ``MinerUProcessAdapter.configure_settings`` 原子
        应用（含活动任务门禁）；这里只覆盖 adapter 不存在的窗口，保持
        mode/tier 目录与实际连接一致。非法配置在此 fail closed，
        ``update_settings`` 的 configure→store→rollback 链照常回滚。
        """
        if self._adapter is None:
            from vibeocr.backend.services.mineru_readiness import (
                configure_connection,
                connection_from_extra,
            )

            configure_connection(
                connection_from_extra(getattr(snapshot, "extra", None))
            )
        return super().configure_settings(snapshot)

    @property
    def adapter(self) -> MinerUProcessAdapter:  # type: ignore[override]
        return super().adapter  # type: ignore[return-value]


__all__ = ["MinerUExecutor"]
