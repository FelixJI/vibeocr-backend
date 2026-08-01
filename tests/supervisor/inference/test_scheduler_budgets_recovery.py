"""Deterministic tests for DeviceScheduler, BudgetPlanner, RecoveryPolicy.

These use a fake clock and fake inputs — no real GPU, no sleeps.
"""

from __future__ import annotations

from vibeocr.backend.supervisor.inference.budgets import (
    AdapterCapability,
    BudgetPlanner,
    InputItem,
)
from vibeocr.backend.supervisor.inference.recovery import (
    FailureClass,
    RecoveryAction,
    RecoveryPolicy,
)
from vibeocr.backend.supervisor.inference.scheduler import DeviceScheduler
from vibeocr.runtime_contracts import JobPriority

# ---------------------------------------------------------------------------
# DeviceScheduler
# ---------------------------------------------------------------------------


def test_try_acquire_grants_one_lease_per_device() -> None:
    t = [0.0]
    sched = DeviceScheduler(devices=["gpu:0"], clock=lambda: t[0])
    sched.enqueue(job_id="j1", device="gpu:0", priority=JobPriority.INTERACTIVE)
    sched.enqueue(job_id="j2", device="gpu:0", priority=JobPriority.BACKGROUND)
    lease1 = sched.try_acquire("j1", "gpu:0")
    assert lease1 is not None and lease1.job_id == "j1"
    # Second job cannot acquire while device is busy.
    lease2 = sched.try_acquire("j2", "gpu:0")
    assert lease2 is None
    sched.release(lease1)
    lease2 = sched.try_acquire("j2", "gpu:0")
    assert lease2 is not None and lease2.job_id == "j2"


def test_cancel_removes_queued_job() -> None:
    sched = DeviceScheduler(devices=["gpu:0"])
    sched.enqueue(job_id="j1", device="gpu:0", priority=JobPriority.INTERACTIVE)
    assert sched.cancel("j1") is True
    lease = sched.try_acquire("j1", "gpu:0")
    assert lease is None


def test_drain_cancels_all_queued() -> None:
    sched = DeviceScheduler(devices=["gpu:0", "gpu:1"])
    sched.enqueue(job_id="j1", device="gpu:0", priority=JobPriority.INTERACTIVE)
    sched.enqueue(job_id="j2", device="gpu:1", priority=JobPriority.BACKGROUND)
    drained = sched.drain()
    assert set(drained) == {"j1", "j2"}
    assert sched.queue_depth() == 0


def test_unknown_device_rejected() -> None:
    sched = DeviceScheduler(devices=["gpu:0"])
    try:
        sched.enqueue(job_id="j1", device="gpu:9", priority=JobPriority.INTERACTIVE)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_two_gpus_each_run_one_heavy_task() -> None:
    sched = DeviceScheduler(devices=["gpu:0", "gpu:1"])
    sched.enqueue(job_id="j1", device="gpu:0", priority=JobPriority.INTERACTIVE)
    sched.enqueue(job_id="j2", device="gpu:1", priority=JobPriority.INTERACTIVE)
    l1 = sched.try_acquire("j1", "gpu:0")
    l2 = sched.try_acquire("j2", "gpu:1")
    assert l1 is not None and l2 is not None
    assert set(sched.busy_devices) == {"gpu:0", "gpu:1"}


def test_background_aging_reduces_effective_priority() -> None:
    t = [0.0]
    sched = DeviceScheduler(devices=["gpu:0"], clock=lambda: t[0], aging_interval=1.0)
    sched.enqueue(job_id="bg", device="gpu:0", priority=JobPriority.BACKGROUND)
    initial = sched.effective_priority("bg", JobPriority.BACKGROUND)
    t[0] = 10.0
    aged = sched.effective_priority("bg", JobPriority.BACKGROUND)
    assert aged < initial


def test_interactive_always_outranks_background() -> None:
    sched = DeviceScheduler(devices=["gpu:0"])
    sched.enqueue(job_id="bg", device="gpu:0", priority=JobPriority.BACKGROUND)
    sched.enqueue(job_id="fg", device="gpu:0", priority=JobPriority.INTERACTIVE)
    nxt = sched.next_job_for("gpu:0")
    assert nxt == "fg"


def test_non_head_cannot_bypass_priority_queue() -> None:
    sched = DeviceScheduler(devices=["gpu:0"])
    sched.enqueue(job_id="bg", device="gpu:0", priority=JobPriority.BACKGROUND)
    sched.enqueue(job_id="fg", device="gpu:0", priority=JobPriority.INTERACTIVE)
    assert sched.try_acquire("bg", "gpu:0") is None
    lease = sched.try_acquire("fg", "gpu:0")
    assert lease is not None


def test_aging_is_recomputed_when_selecting_queue_head() -> None:
    t = [0.0]
    sched = DeviceScheduler(devices=["gpu:0"], clock=lambda: t[0], aging_interval=1.0)
    sched.enqueue(job_id="old-bg", device="gpu:0", priority=JobPriority.BACKGROUND)
    t[0] = 11.0
    sched.enqueue(job_id="fresh-fg", device="gpu:0", priority=JobPriority.INTERACTIVE)
    assert sched.next_job_for("gpu:0") == "old-bg"


# ---------------------------------------------------------------------------
# BudgetPlanner
# ---------------------------------------------------------------------------


def _item(
    item_id: str, *, encoded: int = 1024, pixels: int = 1_000_000, pages: int = 1
) -> InputItem:
    return InputItem(
        item_id=item_id,
        encoded_bytes=encoded,
        decoded_pixels=pixels,
        estimated_pages=pages,
    )


def test_transport_plan_groups_under_all_caps() -> None:
    planner = BudgetPlanner(
        max_file_count=4,
        max_encoded_bytes=4096,
        max_decoded_pixels=10_000_000,
        max_pages=10,
    )
    items = [_item(f"i{i}", encoded=1000, pixels=2_000_000, pages=2) for i in range(10)]
    batches = planner.plan_transport(items)
    # Each batch must respect every cap.
    for batch in batches:
        assert len(batch.items) <= 4
        assert sum(i.encoded_bytes for i in batch.items) <= 4096
        assert sum(i.decoded_pixels for i in batch.items) <= 10_000_000
        assert sum(i.estimated_pages for i in batch.items) <= 10
    # All items accounted for, in order.
    flat = [i.item_id for b in batches for i in b.items]
    assert flat == [f"i{i}" for i in range(10)]


def test_transport_default_pixel_cap_is_64m() -> None:
    assert BudgetPlanner().max_decoded_pixels == 64_000_000


def test_transport_plan_isolates_oversized_item() -> None:
    planner = BudgetPlanner(max_encoded_bytes=4096)
    items = [
        _item("small", encoded=100),
        _item("huge", encoded=999999),
        _item("small2", encoded=100),
    ]
    batches = planner.plan_transport(items)
    assert len(batches) == 3
    assert batches[1].oversized is True
    assert batches[1].items[0].item_id == "huge"
    assert batches[0].oversized is False
    assert batches[2].oversized is False


def test_budget_limits_must_be_positive() -> None:
    try:
        BudgetPlanner(max_file_count=0)
    except ValueError as exc:
        assert "max_file_count" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_budget_planner_rejects_negative_device_vram() -> None:
    """device_vram_mb < 0 raise ValueError（line 86-87）。"""
    try:
        BudgetPlanner(device_vram_mb=-1)
    except ValueError as exc:
        assert "device_vram_mb" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for negative device_vram_mb")


def test_compute_plan_real_batch_caps_at_capability() -> None:
    planner = BudgetPlanner()
    cap = AdapterCapability(name="OCR", real_batch=True, max_compute_batch=4)
    items = [_item(f"i{i}") for i in range(10)]
    batches = planner.plan_compute(items, cap)
    assert all(len(b.items) <= 4 for b in batches)
    assert sum(len(b.items) for b in batches) == 10


def test_compute_plan_falls_back_to_single_when_not_real_batch() -> None:
    planner = BudgetPlanner()
    cap = AdapterCapability(
        name="PP-StructureV3", real_batch=False, max_compute_batch=8
    )
    items = [_item(f"i{i}") for i in range(3)]
    batches = planner.plan_compute(items, cap)
    assert len(batches) == 3
    assert all(len(b.items) == 1 for b in batches)


def test_compute_plan_respects_vram() -> None:
    planner = BudgetPlanner(device_vram_mb=8000)
    cap = AdapterCapability(
        name="OCR", real_batch=True, max_compute_batch=16, per_item_vram_mb=2000
    )
    items = [_item(f"i{i}") for i in range(10)]
    batches = planner.plan_compute(items, cap)
    # 8000/2000 = 4 max per batch even though capability says 16.
    assert all(len(b.items) <= 4 for b in batches)


# ---------------------------------------------------------------------------
# RecoveryPolicy
# ---------------------------------------------------------------------------


def test_classify_recognises_oom_and_bad_input() -> None:
    policy = RecoveryPolicy()
    assert policy.classify("CUDA out of memory") is FailureClass.OOM
    assert policy.classify("OOM during predict") is FailureClass.OOM
    assert policy.classify("invalid image: truncated") is FailureClass.BAD_INPUT
    assert policy.classify("connection reset") is FailureClass.TRANSIENT
    assert policy.classify("unsupported option") is FailureClass.CONFIG_ERROR
    assert policy.classify("cancelled", cancelled=True) is FailureClass.CANCELLED


def test_oom_halves_microbatch_with_bounded_retries() -> None:
    policy = RecoveryPolicy(max_oom_retries=2)
    d1 = policy.next_action(failure=FailureClass.OOM, current_batch_size=8, attempt=0)
    assert d1.action is RecoveryAction.SHRINK_AND_RETRY
    assert d1.next_batch_size == 4
    assert d1.degraded is True
    d2 = policy.next_action(failure=FailureClass.OOM, current_batch_size=4, attempt=1)
    assert d2.next_batch_size == 2
    d3 = policy.next_action(failure=FailureClass.OOM, current_batch_size=2, attempt=2)
    assert d3.action is RecoveryAction.FAIL_FAST


def test_bad_input_uses_bisect_isolation() -> None:
    policy = RecoveryPolicy()
    d = policy.next_action(
        failure=FailureClass.BAD_INPUT, current_batch_size=8, attempt=0
    )
    assert d.action is RecoveryAction.BISECT_ISOLATE
    assert d.degraded is True


def test_transient_uses_exponential_backoff_under_budget() -> None:
    policy = RecoveryPolicy(
        max_transient_retries=3, transient_total_budget_seconds=10.0
    )
    d1 = policy.next_action(
        failure=FailureClass.TRANSIENT, current_batch_size=4, attempt=0
    )
    assert d1.action is RecoveryAction.BACKOFF_RETRY
    assert d1.delay_seconds == 0.25
    d2 = policy.next_action(
        failure=FailureClass.TRANSIENT, current_batch_size=4, attempt=1
    )
    assert d2.delay_seconds == 0.5
    d3 = policy.next_action(
        failure=FailureClass.TRANSIENT, current_batch_size=4, attempt=2
    )
    assert d3.delay_seconds == 1.0
    d4 = policy.next_action(
        failure=FailureClass.TRANSIENT, current_batch_size=4, attempt=3
    )
    assert d4.action is RecoveryAction.FAIL_FAST


def test_cancelled_never_retries() -> None:
    policy = RecoveryPolicy()
    d = policy.next_action(
        failure=FailureClass.CANCELLED, current_batch_size=4, attempt=0
    )
    assert d.action is RecoveryAction.FAIL_FAST


def test_config_and_deterministic_fail_fast() -> None:
    policy = RecoveryPolicy()
    assert (
        policy.next_action(
            failure=FailureClass.CONFIG_ERROR, current_batch_size=4, attempt=0
        ).action
        is RecoveryAction.FAIL_FAST
    )
    assert (
        policy.next_action(
            failure=FailureClass.DETERMINISTIC_MODEL, current_batch_size=4, attempt=0
        ).action
        is RecoveryAction.FAIL_FAST
    )


def test_transient_budget_exhaustion_fails_fast() -> None:
    # The caller reports accumulated elapsed time; the policy refuses to
    # schedule a delay that would exceed the total budget.
    policy = RecoveryPolicy(
        max_transient_retries=10,
        transient_base_delay=10.0,
        transient_max_delay=20.0,  # do not clamp the first delay below the budget
        transient_total_budget_seconds=4.0,
        elapsed_seconds=0.0,
    )
    d = policy.next_action(
        failure=FailureClass.TRANSIENT, current_batch_size=4, attempt=0
    )
    # First attempt with delay=10 > budget=5 -> fail fast.
    assert d.action is RecoveryAction.FAIL_FAST
    assert "budget" in d.reason
