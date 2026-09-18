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

    @property
    def adapter(self) -> MinerUProcessAdapter:  # type: ignore[override]
        return super().adapter  # type: ignore[return-value]


__all__ = ["MinerUExecutor"]
