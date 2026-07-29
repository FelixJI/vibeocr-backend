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

from .paddle_executor import AdapterExecutor


class MinerUExecutor(AdapterExecutor):
    """Drives MinerU document-parse jobs through a MinerUProcessAdapter."""

    def __init__(
        self,
        adapter_factory: Callable[[], MinerUProcessAdapter],
        **coordinator_options: Any,
    ) -> None:
        super().__init__(adapter_factory, **coordinator_options)

    @property
    def adapter(self) -> MinerUProcessAdapter:  # type: ignore[override]
        return super().adapter  # type: ignore[return-value]


__all__ = ["MinerUExecutor"]
