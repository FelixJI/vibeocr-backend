"""RecoveryPolicy: classify failures and decide the next bounded action.

Plan §3 Phase 3 / §4.3:

* OOM → clear cache + halve the microbatch, bounded retries.
* Suspected bad input → binary isolation (only the offending item fails).
* Transient backend error → exponential backoff under a total time/attempt budget.
* Cancellation / config error / deterministic model error → no retry.

The policy is pure: given a classified error and current state it returns the
next action without performing it. Adapters execute the action.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class FailureClass(StrEnum):
    OOM = "oom"
    BAD_INPUT = "bad_input"
    TRANSIENT = "transient"
    CANCELLED = "cancelled"
    CONFIG_ERROR = "config_error"
    DETERMINISTIC_MODEL = "deterministic_model"
    UNKNOWN = "unknown"


class RecoveryAction(StrEnum):
    SHRINK_AND_RETRY = "shrink_and_retry"
    BISECT_ISOLATE = "bisect_isolate"
    BACKOFF_RETRY = "backoff_retry"
    FAIL_FAST = "fail_fast"


@dataclass(frozen=True, slots=True)
class RecoveryDecision:
    action: RecoveryAction
    next_batch_size: int | None
    delay_seconds: float
    attempt: int
    degraded: bool
    reason: str


@dataclass
class RecoveryPolicy:
    """Bounded recovery policy. Tracks attempt counts per item set."""

    max_oom_retries: int = 2
    max_transient_retries: int = 3
    transient_base_delay: float = 0.25
    transient_max_delay: float = 5.0
    transient_total_budget_seconds: float = 30.0
    min_batch_size: int = 1
    elapsed_seconds: float = 0.0

    def classify(self, error_message: str, *, cancelled: bool = False) -> FailureClass:
        if cancelled:
            return FailureClass.CANCELLED
        msg = (error_message or "").lower()
        if "out of memory" in msg or "oom" in msg or "cuda oom" in msg:
            return FailureClass.OOM
        if "corrupt" in msg or "decode" in msg or "invalid image" in msg or "truncated" in msg:
            return FailureClass.BAD_INPUT
        if "timeout" in msg or "temporarily" in msg or "unavailable" in msg or "connection" in msg:
            return FailureClass.TRANSIENT
        if "config" in msg or "option" in msg or "unsupported" in msg:
            return FailureClass.CONFIG_ERROR
        if "shape" in msg or "dtype" in msg or "inference" in msg:
            return FailureClass.DETERMINISTIC_MODEL
        return FailureClass.UNKNOWN

    def next_action(
        self,
        *,
        failure: FailureClass,
        current_batch_size: int,
        attempt: int,
    ) -> RecoveryDecision:
        if failure is FailureClass.CANCELLED:
            return RecoveryDecision(
                RecoveryAction.FAIL_FAST, None, 0.0, attempt, False, "cancelled"
            )
        if failure is FailureClass.CONFIG_ERROR:
            return RecoveryDecision(
                RecoveryAction.FAIL_FAST, None, 0.0, attempt, False, "config error"
            )
        if failure is FailureClass.DETERMINISTIC_MODEL:
            return RecoveryDecision(
                RecoveryAction.FAIL_FAST, None, 0.0, attempt, False, "deterministic model error"
            )
        if failure is FailureClass.BAD_INPUT:
            return RecoveryDecision(
                RecoveryAction.BISECT_ISOLATE,
                max(1, current_batch_size // 2),
                0.0,
                attempt,
                True,
                "suspected bad input; isolate",
            )
        if failure is FailureClass.OOM:
            if attempt >= self.max_oom_retries:
                return RecoveryDecision(
                    RecoveryAction.FAIL_FAST, None, 0.0, attempt, True, "oom retries exhausted"
                )
            halved = max(self.min_batch_size, current_batch_size // 2)
            return RecoveryDecision(
                RecoveryAction.SHRINK_AND_RETRY,
                halved,
                0.0,
                attempt + 1,
                True,
                "oom; halved microbatch",
            )
        if failure is FailureClass.TRANSIENT:
            if attempt >= self.max_transient_retries:
                return RecoveryDecision(
                    RecoveryAction.FAIL_FAST, None, 0.0, attempt, True, "transient retries exhausted"
                )
            delay = min(
                self.transient_max_delay,
                self.transient_base_delay * (2**attempt),
            )
            if self.elapsed_seconds + delay > self.transient_total_budget_seconds:
                return RecoveryDecision(
                    RecoveryAction.FAIL_FAST, None, 0.0, attempt, True, "transient time budget exhausted"
                )
            return RecoveryDecision(
                RecoveryAction.BACKOFF_RETRY,
                current_batch_size,
                delay,
                attempt + 1,
                True,
                "transient; exponential backoff",
            )
        # Unknown → treat conservatively as transient-once.
        if attempt >= 1:
            return RecoveryDecision(
                RecoveryAction.FAIL_FAST, None, 0.0, attempt, True, "unknown failure; no retry"
            )
        return RecoveryDecision(
            RecoveryAction.BACKOFF_RETRY,
            current_batch_size,
            self.transient_base_delay,
            attempt + 1,
            True,
            "unknown failure; one retry",
        )


__all__ = [
    "FailureClass",
    "RecoveryAction",
    "RecoveryDecision",
    "RecoveryPolicy",
]
