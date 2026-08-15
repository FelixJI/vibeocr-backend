"""PaddleExecutor: bridge between SupervisorModule and PaddlePipelineAdapter.

The supervisor's :class:`~vibeocr.backend.supervisor.module.Executor` protocol takes a
``JobRecord`` + staged inputs and drives the job to terminal. The
:class:`PaddlePipelineAdapter` exposes the unified ``recognize_many`` seam but
does not know about jobs/items/state. This executor glues them: it converts
staged inputs into :class:`InputItem` instances, calls the adapter, maps per-
item results/errors back onto the record, and follows the honest cancel state
machine (running → cancel_requested → cancelled).

This is the production executor wired by :func:`build_supervisor` once a real
``OCRService`` is available.

The job-driving state machine is backend-agnostic (it only calls
``adapter.recognize_many`` / ``residency_status`` / ``release_idle``), so it
lives in :class:`AdapterExecutor` and is reused by :class:`MinerUExecutor`.
"""

from __future__ import annotations

import io
import threading
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from .paddle_adapter import PaddlePipelineAdapter

from vibeocr.runtime_contracts import (
    TERMINAL_JOB_STATES,
    CancelMode,
    ErrorCode,
    ItemState,
    JobState,
    ResidencyStatus,
    SettingsSnapshot,
)

from .budgets import AdapterCapability, BudgetPlanner, InputItem
from .ocr_engines import OcrEngineError
from .recovery import FailureClass, RecoveryAction, RecoveryPolicy
from .scheduler import DeviceScheduler


class AdapterExecutor:
    """Backend-agnostic job driver over any ``recognize_many`` adapter.

    Subclasses (or callers) supply an ``adapter_factory`` returning an object
    that exposes ``recognize_many(items)``, ``residency_status()`` and
    ``release_idle(pipeline)``. The factory is called lazily on first use so
    heavy model loads stay deferred until the first job.

    The execute flow is identical for Paddle and MinerU: queued → running →
    per-item mapping → completed/completed_with_errors, with whole-batch and
    mid-flight cancel handling. Only the adapter differs.
    """

    def __init__(
        self,
        adapter_factory: Callable[[], Any],
        *,
        scheduler: DeviceScheduler | None = None,
        budget_planner: BudgetPlanner | None = None,
        recovery_policy_factory: Callable[[], RecoveryPolicy] | None = None,
        device: str = "gpu:0",
        sleeper: Callable[[float], None] = time.sleep,
        clear_cache: Callable[[], None] | None = None,
    ) -> None:
        self._adapter_factory = adapter_factory
        self._adapter: Any | None = None
        self._adapter_lock = threading.Lock()
        self._scheduler = scheduler or DeviceScheduler(devices=[device])
        self._budget_planner = budget_planner or BudgetPlanner()
        self._recovery_policy_factory = recovery_policy_factory or RecoveryPolicy
        self._device = device
        self._sleeper = sleeper
        self._clear_cache = clear_cache or (lambda: None)
        self._settings = SettingsSnapshot()

    @property
    def adapter(self) -> Any:
        if self._adapter is None:
            with self._adapter_lock:
                if self._adapter is None:
                    self._adapter = self._adapter_factory()
                    configure = getattr(self._adapter, "configure_settings", None)
                    if callable(configure):
                        configure(self._settings)
        return self._adapter

    # ------------------------------------------------------------------
    # Executor protocol
    # ------------------------------------------------------------------

    def execute(self, record: Any, staged: Any) -> None:
        """Run the job to terminal.

        ``staged`` is the list of :class:`StagedInput` produced by InputStager.
        We build one :class:`InputItem` per staged file (carrying its raw
        bytes), call ``recognize_many`` once, and map results back in input
        order. Per-item failures are isolated (continue-on-failure); a cancel
        requested mid-flight stops after the current item and transitions the
        job through the cancel state machine.
        """
        if record.state in TERMINAL_JOB_STATES:
            return
        # Transition queued → running.
        if record.state is JobState.QUEUED:
            record.transition(JobState.RUNNING)
            record.append_event("running")
        if record.state is JobState.CANCEL_REQUESTED:
            record.transition(JobState.CANCELLED)
            record.append_event("cancelled")
            return

        items = self._staged_to_items(staged)
        record.append_event("recognize_started", detail={"items": len(items)})
        options = getattr(record, "pipeline", None)
        try:
            capability = self.adapter.capabilities(options)
        except (AttributeError, TypeError):
            capability = AdapterCapability(
                name=getattr(options, "pipeline_id", "unknown"),
                real_batch=False,
                max_compute_batch=1,
            )
        transport_batches = self._budget_planner.plan_transport(items)
        compute_batches = [
            compute
            for transport in transport_batches
            for compute in self._budget_planner.plan_compute(
                transport.items, capability
            )
        ]
        record.append_event(
            "batch_plan",
            detail={
                "transport_batches": len(transport_batches),
                "compute_batches": len(compute_batches),
                "real_batch": capability.real_batch,
            },
        )

        for compute in compute_batches:
            if record.cancel_requested_at is not None:
                break
            lease = self._scheduler.acquire(
                job_id=record.job_id,
                device=self._device,
                priority=record.priority,
                cancelled=lambda: record.cancel_requested_at is not None,
            )
            if lease is None:
                break
            try:
                self._execute_with_recovery(
                    record,
                    list(compute.items),
                    options=options,
                    policy=self._recovery_policy_factory(),
                )
            finally:
                self._scheduler.release(lease)

        # Honor a cancel requested during the run.
        if (
            record.cancel_requested_at is not None
            and record.state not in TERMINAL_JOB_STATES
        ):
            self._cancel_non_terminal_items(record)
            if record.state is not JobState.CANCEL_REQUESTED:
                record.transition(JobState.CANCEL_REQUESTED)
            record.transition(JobState.CANCELLED)
            record.append_event("cancelled")
            return

        if record.state not in TERMINAL_JOB_STATES:
            total_succeeded = sum(
                1 for item in record.items if item.state is ItemState.SUCCEEDED
            )
            total_failed = sum(
                1 for item in record.items if item.state is ItemState.FAILED
            )
            if total_failed == 0:
                terminal = JobState.COMPLETED
            elif total_succeeded == 0:
                terminal = JobState.FAILED
            else:
                terminal = JobState.COMPLETED_WITH_ERRORS
            record.transition(terminal)
            record.append_event(
                "done",
                detail={"succeeded": total_succeeded, "failed": total_failed},
            )

    def _execute_with_recovery(
        self,
        record: Any,
        items: list[InputItem],
        *,
        options: Any,
        policy: RecoveryPolicy,
        attempt: int = 0,
    ) -> None:
        if not items or record.cancel_requested_at is not None:
            return
        try:
            payloads = self.adapter.recognize_many(items, options=options)
        except OcrEngineError as exc:
            # 引擎选择失败是确定性错误：直接按协议错误码标记本批 item，
            # 不进入 bisect/backoff 恢复路径，也不切换引擎。
            record.append_event(
                "ocr_engine_rejected",
                detail={
                    "code": exc.code.value,
                    "engine": exc.engine,
                    "reason_code": exc.reason_code,
                },
            )
            self._fail_items(
                record,
                items,
                error_code=exc.code.value,
                error=str(exc),
            )
            return
        except Exception as exc:
            failure = policy.classify(
                str(exc), cancelled=record.cancel_requested_at is not None
            )
            decision = policy.next_action(
                failure=failure,
                current_batch_size=len(items),
                attempt=attempt,
            )
            if decision.degraded:
                record.mark_degraded()
            record.append_event(
                "recovery_decision",
                detail={
                    "failure": failure.value,
                    "action": decision.action.value,
                    "attempt": decision.attempt,
                    "batch_size": len(items),
                },
            )
            if decision.action is RecoveryAction.BISECT_ISOLATE and len(items) > 1:
                midpoint = max(1, len(items) // 2)
                self._execute_with_recovery(
                    record,
                    items[:midpoint],
                    options=options,
                    policy=policy,
                    attempt=decision.attempt,
                )
                self._execute_with_recovery(
                    record,
                    items[midpoint:],
                    options=options,
                    policy=policy,
                    attempt=decision.attempt,
                )
                return
            if decision.action is RecoveryAction.SHRINK_AND_RETRY:
                self._clear_cache()
                next_size = decision.next_batch_size or 1
                for index in range(0, len(items), next_size):
                    self._execute_with_recovery(
                        record,
                        items[index : index + next_size],
                        options=options,
                        policy=policy,
                        attempt=decision.attempt,
                    )
                return
            if decision.action is RecoveryAction.BACKOFF_RETRY:
                self._sleeper(decision.delay_seconds)
                policy.elapsed_seconds += decision.delay_seconds
                self._execute_with_recovery(
                    record,
                    items,
                    options=options,
                    policy=policy,
                    attempt=decision.attempt,
                )
                return
            self._fail_items(
                record,
                items,
                error_code=self._error_code_for(failure),
                error=str(exc),
            )
            return

        # Positional native output is converted and validated at this boundary.
        if not isinstance(payloads, list) or len(payloads) != len(items):
            actual = len(payloads) if isinstance(payloads, list) else None
            message = (
                f"adapter result count mismatch: expected={len(items)} actual={actual}"
            )
            self._fail_items(
                record,
                items,
                error_code=ErrorCode.ADAPTER_PROTOCOL_VIOLATION.value,
                error=message,
            )
            record.append_event(
                "adapter_protocol_violation",
                detail={"expected": len(items), "actual": actual},
            )
            return

        payload_type = (
            "mineru.v1"
            if getattr(getattr(record, "kind", None), "value", None) == "mineru_parse"
            else "ocr.v1"
        )
        for input_item, payload in zip(items, payloads, strict=True):
            if record.cancel_requested_at is not None:
                return
            if not isinstance(payload, dict) or not payload:
                record.commit_item_failure(
                    input_item.item_id,
                    error_code=ErrorCode.ADAPTER_PROTOCOL_VIOLATION.value,
                    error="adapter returned an empty or non-object payload",
                )
                continue
            record.commit_item_success(
                input_item.item_id,
                payload_type=payload_type,
                payload=payload,
            )

    @staticmethod
    def _fail_items(
        record: Any,
        items: list[InputItem],
        *,
        error_code: str,
        error: str,
    ) -> None:
        for item in items:
            current = next(
                candidate
                for candidate in record.items
                if candidate.item_id == item.item_id
            )
            if current.state in (
                ItemState.SUCCEEDED,
                ItemState.FAILED,
                ItemState.CANCELLED,
            ):
                continue
            record.commit_item_failure(
                item.item_id,
                error_code=error_code,
                error=error,
            )

    @staticmethod
    def _error_code_for(failure: FailureClass) -> str:
        return {
            FailureClass.OOM: "RESOURCE_EXHAUSTED",
            FailureClass.BAD_INPUT: "BAD_INPUT",
            FailureClass.TRANSIENT: "BACKEND_UNAVAILABLE",
            FailureClass.CANCELLED: "CANCELLED",
            FailureClass.CONFIG_ERROR: "VALIDATION_ERROR",
            FailureClass.DETERMINISTIC_MODEL: "MODEL_ERROR",
            FailureClass.UNKNOWN: "BACKEND_UNAVAILABLE",
        }[failure]

    @staticmethod
    def _cancel_non_terminal_items(record: Any) -> None:
        for item in list(record.items):
            if item.state not in (
                ItemState.SUCCEEDED,
                ItemState.FAILED,
                ItemState.CANCELLED,
            ):
                record.commit_item_cancelled(item.item_id)

    def cancel_mode_for(self, record: Any) -> CancelMode:
        if record.state is JobState.QUEUED:
            return CancelMode.QUEUED_ONLY
        return CancelMode.COOPERATIVE

    def residency_status(self) -> ResidencyStatus:
        if self._adapter is None:
            return ResidencyStatus(
                default_ttl_seconds=self._settings.default_ttl_seconds,
                pipelines=self._settings.pipelines,
            )
        return self.adapter.residency_status()

    def release_idle(self, pipeline: str | None = None) -> ResidencyStatus:
        if self._adapter is None:
            return self.residency_status()
        return self.adapter.release_idle(pipeline)

    def preload(self, pipelines: tuple[str, ...]) -> ResidencyStatus:
        return self.adapter.preload(pipelines)

    def configure_settings(self, snapshot: SettingsSnapshot) -> ResidencyStatus:
        self._settings = snapshot
        if self._adapter is not None:
            configure = getattr(self._adapter, "configure_settings", None)
            if callable(configure):
                configure(snapshot)
        return self.residency_status()

    def close(self) -> None:
        if self._adapter is None:
            return
        close = getattr(self._adapter, "close", None)
        if callable(close):
            close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _staged_to_items(staged: Any) -> list[InputItem]:
        """Convert StagedInput list → InputItem list carrying raw bytes."""
        out: list[InputItem] = []
        for entry in staged or []:
            data = entry.path.read_bytes() if hasattr(entry, "path") else b""
            decoded_pixels = 0
            try:
                from PIL import Image

                with Image.open(io.BytesIO(data)) as image:
                    decoded_pixels = int(image.width) * int(image.height)
            except Exception:
                pass
            out.append(
                InputItem(
                    item_id=getattr(entry, "item_id", f"it-{len(out)}"),
                    encoded_bytes=len(data),
                    decoded_pixels=decoded_pixels,
                    estimated_pages=1,
                    display_name=getattr(entry, "display_name", "input"),
                    data=data,
                )
            )
        return out


class PaddleExecutor(AdapterExecutor):
    """Drives recognition jobs through a :class:`PaddlePipelineAdapter`.

    Thin specialization of :class:`AdapterExecutor`; kept as a named class so
    composition and tests can reference the Paddle backend explicitly while
    the generic state machine stays shared with MinerU.
    """

    def __init__(
        self,
        adapter_factory: Callable[[], PaddlePipelineAdapter],
        **coordinator_options: Any,
    ) -> None:
        super().__init__(adapter_factory, **coordinator_options)

    @property
    def adapter(self) -> PaddlePipelineAdapter:  # type: ignore[override]
        return super().adapter  # type: ignore[return-value]


__all__ = ["AdapterExecutor", "PaddleExecutor"]
