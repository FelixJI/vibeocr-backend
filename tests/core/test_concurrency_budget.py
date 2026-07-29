"""测试 ConcurrencyBudget 集中配置。

将分散在各处的线程/进程数配置集中管理，避免未来启用多 worker 时
出现 worker数 × OMP线程数 + GUI线程池 过度订阅。
"""

from vibeocr.backend.core.concurrency_budget import ConcurrencyBudget


class TestConcurrencyBudget:
    def test_default_budget_single_worker(self):
        """默认预算：OCR worker 1 个，避免多进程×OMP 超额"""
        budget = ConcurrencyBudget.default()
        assert budget.ocr_workers == 1
        assert budget.omp_threads_per_worker > 0
        assert budget.pdf_render_concurrency > 0
        assert budget.qt_pool_limit > 0
        assert budget.asyncio_executor_limit > 0

    def test_budget_logs_actual_values(self, caplog):
        """启动时记录实际预算"""
        import logging

        budget = ConcurrencyBudget.default()
        with caplog.at_level(logging.INFO):
            budget.log_summary()
        assert any("OCR workers" in r.message for r in caplog.records)

    def test_total_cpu_threads_not_oversubscribed(self):
        """单 worker 时 OMP 线程不超过逻辑核数"""
        import os

        budget = ConcurrencyBudget.default()
        logical = os.cpu_count() or 4
        total = budget.ocr_workers * budget.omp_threads_per_worker
        assert total <= logical, f"CPU 超额: {total} > {logical}"

    def test_budget_is_frozen(self):
        """ConcurrencyBudget 是 frozen dataclass（不可变）"""
        import dataclasses

        assert dataclasses.is_dataclass(ConcurrencyBudget)
        budget = ConcurrencyBudget.default()
        try:
            budget.ocr_workers = 2  # type: ignore[misc]
            raise AssertionError("frozen dataclass 不应允许修改")
        except dataclasses.FrozenInstanceError:
            pass


def test_default_handles_cpu_thread_count_exception(monkeypatch):
    """get_cpu_thread_count 抛异常时 default 回退到 min(logical, 8)（line 47-49）。"""
    from vibeocr.backend.core.concurrency_budget import ConcurrencyBudget

    def _raise():
        raise RuntimeError("probe failed")

    import vibeocr.backend.utils.cpu_info as cpu_info

    monkeypatch.setattr(cpu_info, "get_cpu_thread_count", _raise)
    budget = ConcurrencyBudget.default()
    assert budget.omp_threads_per_worker > 0
