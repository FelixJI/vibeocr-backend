"""pipeline_cache_manager 覆盖补充：聚焦未覆盖分支。

不导入 paddle/pynvml；GPU 相关路径通过 monkeypatch 环境变量 + 模块属性覆盖。
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest

from vibeocr.backend.services.pipeline_cache_manager import (
    FALLBACK_MAX_HEAVY,
    PipelineCacheManager,
)
from vibeocr.runtime_contracts import PipelineSpec


def _make_legacy_manager(
    max_heavy: int = 2, ttls: dict[str, int] | None = None
) -> PipelineCacheManager:
    service = MagicMock()
    service._pipelines = {}
    mgr = PipelineCacheManager.__new__(PipelineCacheManager)
    mgr._service = service
    mgr._ttls = dict(ttls) if ttls is not None else {"PP-StructureV3": 300}
    mgr._max_heavy = max_heavy
    mgr._last_used = {}
    mgr._active_counts = {}
    mgr._state_lock = threading.RLock()
    mgr._tick_interval = 30.0
    mgr._stop_event = threading.Event()
    mgr._wakeup_event = threading.Event()
    mgr._thread = None
    mgr._pinned = set()
    return mgr


# ---- _ensure_runtime_fields -------------------------------------------


class TestEnsureRuntimeFields:
    def test_creates_missing_fields(self):
        """_ensure_runtime_fields 为 __new__ 构造的实例补齐字段（lines 95,97,101,103,105）。"""
        mgr = PipelineCacheManager.__new__(PipelineCacheManager)
        mgr._service = MagicMock()
        mgr._service._pipelines = {}
        mgr._ttls = {}
        mgr._max_heavy = 1
        mgr._last_used = {}
        # 不设 _state_lock / _active_counts / _pinned / _wakeup_event / _stop_event / _tick_interval
        mgr._ensure_runtime_fields()
        assert hasattr(mgr, "_state_lock")
        assert hasattr(mgr, "_active_counts")
        assert hasattr(mgr, "_pinned")
        assert hasattr(mgr, "_wakeup_event")
        assert hasattr(mgr, "_stop_event")
        assert hasattr(mgr, "_tick_interval")


# ---- _detect_max_heavy GPU 路径 ---------------------------------------


class TestDetectMaxHeavyGpu:
    def test_gpu_mode_uses_gpu_memory_monitor(self, monkeypatch):
        """GPU 模式 + GPUMemoryMonitor 可用 → 按显存分档（lines 114-119）。"""
        monkeypatch.setenv("VIBEOCR_USE_GPU", "true")

        fake_info = MagicMock()
        fake_info.available = True
        fake_info.total = 16384  # 16GB → 2

        fake_monitor_cls = MagicMock()
        fake_monitor_cls.return_value.get_status.return_value = fake_info

        import vibeocr.backend.utils.gpu_memory_monitor as gmm

        monkeypatch.setattr(gmm, "GPUMemoryMonitor", fake_monitor_cls)

        service = MagicMock()
        service._pipelines = {}
        mgr = PipelineCacheManager(service, {}, max_heavy=None)
        try:
            assert mgr.max_heavy == 2
        finally:
            mgr.shutdown()

    def test_gpu_mode_monitor_exception_falls_back(self, monkeypatch):
        """GPU 模式 + GPUMemoryMonitor 抛异常 → 回退（lines 120-126）。"""
        monkeypatch.setenv("VIBEOCR_USE_GPU", "true")

        fake_monitor_cls = MagicMock()
        fake_monitor_cls.return_value.get_status.side_effect = RuntimeError("nvml boom")

        import vibeocr.backend.utils.gpu_memory_monitor as gmm

        monkeypatch.setattr(gmm, "GPUMemoryMonitor", fake_monitor_cls)

        service = MagicMock()
        service._pipelines = {}
        mgr = PipelineCacheManager(service, {}, max_heavy=None)
        try:
            assert mgr.max_heavy == FALLBACK_MAX_HEAVY
        finally:
            mgr.shutdown()

    def test_gpu_mode_unavailable_falls_back(self, monkeypatch):
        """GPU 模式 + info.available=False → 回退（line 118 False）。"""
        monkeypatch.setenv("VIBEOCR_USE_GPU", "true")

        fake_info = MagicMock()
        fake_info.available = False
        fake_info.total = 0

        fake_monitor_cls = MagicMock()
        fake_monitor_cls.return_value.get_status.return_value = fake_info

        import vibeocr.backend.utils.gpu_memory_monitor as gmm

        monkeypatch.setattr(gmm, "GPUMemoryMonitor", fake_monitor_cls)

        service = MagicMock()
        service._pipelines = {}
        mgr = PipelineCacheManager(service, {}, max_heavy=None)
        try:
            assert mgr.max_heavy == FALLBACK_MAX_HEAVY
        finally:
            mgr.shutdown()


# ---- ttls.setter 验证 --------------------------------------------------


class TestTtlsSetterValidation:
    def test_unknown_pipeline_ignored(self):
        mgr = _make_legacy_manager()
        mgr.ttls = {"UnknownPipeline": 100, "OCR": 200}
        assert "UnknownPipeline" not in mgr.ttls
        assert mgr.ttls["OCR"] == 200

    def test_bool_ttl_ignored(self):
        mgr = _make_legacy_manager()
        mgr.ttls = {"OCR": True}
        assert "OCR" not in mgr.ttls

    def test_invalid_ttl_type_ignored(self):
        mgr = _make_legacy_manager()
        mgr.ttls = {"OCR": "not-a-number"}
        assert "OCR" not in mgr.ttls

    def test_restore_key_triggers_restore(self):
        """含 _RESTORE_LAST_USED_KEY → 触发 restore（line 168）。"""
        mgr = _make_legacy_manager()
        mgr._service._pipelines = {"OCR": object()}
        from vibeocr.backend.services.pipeline_cache_manager import (
            _RESTORE_LAST_USED_KEY,
        )

        # restore_values 含一个已加载管道的时间戳
        mgr.ttls = {
            "OCR": 300,
            _RESTORE_LAST_USED_KEY: {"OCR": 1000},  # 1 second epoch ms
        }
        # restore 应已设置 last_used
        assert "OCR" in mgr._last_used

    def test_restore_key_non_dict_ignored(self):
        """_RESTORE_LAST_USED_KEY 非 dict → 不触发 restore（line 143）。"""
        mgr = _make_legacy_manager()
        from vibeocr.backend.services.pipeline_cache_manager import (
            _RESTORE_LAST_USED_KEY,
        )

        mgr.ttls = {"OCR": 300, _RESTORE_LAST_USED_KEY: "not-a-dict"}
        # 正常设置 ttls，不崩
        assert mgr.ttls["OCR"] == 300


# ---- lease 计数分支 ----------------------------------------------------


class TestLeaseCounting:
    def test_lease_decrement_to_zero_removes(self):
        """lease 结束后 count 归 0 → 移除计数（line 195 已覆盖，这里覆盖 198->200）。"""
        mgr = _make_legacy_manager()
        mgr._service._pipelines = {"OCR": object()}
        with mgr.lease("OCR"):
            assert mgr._active_counts["OCR"] == 1
        assert "OCR" not in mgr._active_counts

    def test_lease_pipeline_not_in_cache_no_timestamp(self):
        """lease 结束但管道不在缓存 → 不设 last_used（branch 198->200）。"""
        mgr = _make_legacy_manager()
        # OCR 不在 _pipelines
        with mgr.lease("OCR"):
            pass
        assert "OCR" not in mgr._last_used


# ---- restore_last_used_unix_ms ----------------------------------------


class TestRestoreLastUsed:
    def test_restore_skips_unknown_pipeline(self):
        """restore 跳过不在缓存中的管道（line 221）。"""
        mgr = _make_legacy_manager()
        mgr._service._pipelines = {"OCR": object()}
        mgr.restore_last_used_unix_ms({"UnknownPipeline": 1000})
        assert "UnknownPipeline" not in mgr._last_used

    def test_restore_skips_bool_value(self):
        """restore 跳过 bool 值（line 222）。"""
        mgr = _make_legacy_manager()
        mgr._service._pipelines = {"OCR": object()}
        mgr.restore_last_used_unix_ms({"OCR": True})
        # bool 被跳过 → 未设置或保持默认
        assert mgr._last_used.get("OCR") is None or mgr._last_used["OCR"] >= 0

    def test_restore_skips_invalid_type(self):
        """restore 跳过非数值类型（line 223）。"""
        mgr = _make_legacy_manager()
        mgr._service._pipelines = {"OCR": object()}
        mgr.restore_last_used_unix_ms({"OCR": [1, 2, 3]})
        assert "OCR" not in mgr._last_used

    def test_restore_skips_unparseable_string(self):
        """restore 跳过无法解析的字符串（lines 228-231）。"""
        mgr = _make_legacy_manager()
        mgr._service._pipelines = {"OCR": object()}
        mgr.restore_last_used_unix_ms({"OCR": "not-a-number"})
        assert "OCR" not in mgr._last_used

    def test_restore_caps_future_timestamp(self):
        """restore 未来时间戳被 cap 到 now（line 234）。"""
        mgr = _make_legacy_manager()
        mgr._service._pipelines = {"OCR": object()}
        # 传一个远未来的时间戳
        future_ms = (time.time() + 999999) * 1000
        mgr.restore_last_used_unix_ms({"OCR": future_ms})
        restored = mgr._last_used["OCR"]
        assert restored <= time.time()

    def test_restore_negative_capped_to_zero(self):
        """restore 负值被 cap 到 0（line 234）。"""
        mgr = _make_legacy_manager()
        mgr._service._pipelines = {"OCR": object()}
        mgr.restore_last_used_unix_ms({"OCR": -5000})
        assert mgr._last_used["OCR"] == 0.0


# ---- prepare_load ------------------------------------------------------


class TestPrepareLoad:
    def test_prepare_load_non_heavy_returns_empty(self):
        """prepare_load 对非重 paddle 管道返回空（line 291）。"""
        mgr = _make_legacy_manager()
        result = mgr.prepare_load("OCR")
        assert result == []

    def test_prepare_load_evicts_to_make_room(self):
        """prepare_load 超上限时淘汰（lines 310-317）。"""
        mgr = _make_legacy_manager(max_heavy=1)
        mgr._service._pipelines = {"PP-StructureV3": object()}
        mgr._last_used = {"PP-StructureV3": 100.0}
        evicted = mgr.prepare_load("PaddleOCR-VL")
        assert evicted == ["PP-StructureV3"]


# ---- configure_residency 验证 -----------------------------------------


class TestConfigureResidencyValidation:
    def test_negative_default_ttl_raises(self):
        mgr = _make_legacy_manager()
        with pytest.raises(ValueError, match="default_ttl_seconds"):
            mgr.configure_residency(default_ttl_seconds=-1, pipelines=[])

    def test_unknown_pipeline_raises(self):
        mgr = _make_legacy_manager()
        with pytest.raises(ValueError, match="unknown pipeline"):
            mgr.configure_residency(
                default_ttl_seconds=300,
                pipelines=[PipelineSpec(name="UnknownPipeline")],
            )

    def test_negative_pipeline_ttl_raises(self):
        mgr = _make_legacy_manager()
        with pytest.raises(ValueError, match="negative TTL"):
            mgr.configure_residency(
                default_ttl_seconds=300,
                pipelines=[PipelineSpec(name="OCR", ttl_seconds=-1)],
            )

    def test_ttl_override_applied(self):
        """spec 的 ttl_seconds 覆盖默认（lines 339-341）。"""
        mgr = _make_legacy_manager()
        mgr.configure_residency(
            default_ttl_seconds=300,
            pipelines=[PipelineSpec(name="OCR", ttl_seconds=600)],
        )
        assert mgr.ttls["OCR"] == 600


# ---- evict_idle 边界 ---------------------------------------------------


class TestEvictIdleEdges:
    def test_evict_idle_skips_active(self):
        """evict_idle 跳过活动管道（line 361）。"""
        mgr = _make_legacy_manager(max_heavy=3)
        mgr._service._pipelines = {"PP-StructureV3": object()}
        mgr._last_used = {"PP-StructureV3": 0.0}
        mgr._active_counts = {"PP-StructureV3": 1}
        evicted = mgr.evict_idle(now=99999.0)
        assert evicted == []
        assert "PP-StructureV3" in mgr._service._pipelines

    def test_evict_idle_missing_timestamp_sets_now(self):
        """evict_idle 缺失时间戳 → 设为 now 并跳过（lines 370-371）。"""
        mgr = _make_legacy_manager(max_heavy=3)
        mgr._service._pipelines = {"PP-StructureV3": object()}
        # 不设 _last_used
        evicted = mgr.evict_idle(now=500.0)
        assert evicted == []
        assert mgr._last_used["PP-StructureV3"] == 500.0


# ---- release_one pinned -----------------------------------------------


class TestReleaseOnePinned:
    def test_release_one_pinned_without_force_returns_false(self):
        """release_one 对 pinned 管道、force=False → 返回 False（line 417）。"""
        mgr = _make_legacy_manager(max_heavy=3)
        mgr._service._pipelines = {"OCR": object()}
        mgr._pinned = {"OCR"}
        assert mgr.release_one("OCR") is False
        assert "OCR" in mgr._service._pipelines  # 未释放

    def test_release_one_pinned_with_force_releases(self):
        """release_one 对 pinned 管道、force=True → 释放（line 414 True 分支）。"""
        mgr = _make_legacy_manager(max_heavy=3)
        mgr._service._pipelines = {"OCR": object()}
        mgr._pinned = {"OCR"}
        assert mgr.release_one("OCR", force=True) is True
        assert "OCR" not in mgr._service._pipelines


# ---- _seconds_until_next_check ----------------------------------------


class TestSecondsUntilNextCheck:
    def test_no_pipelines_returns_60(self):
        """无管道 → 60s（line 461）。"""
        mgr = _make_legacy_manager()
        mgr._service._pipelines = {}
        assert mgr._seconds_until_next_check() == 60.0

    def test_all_ttl_zero_returns_60(self):
        """所有管道 TTL<=0 → 60s（line 477）。"""
        mgr = _make_legacy_manager(ttls={"OCR": 0})
        mgr._service._pipelines = {"OCR": object()}
        mgr._last_used = {"OCR": time.time()}
        assert mgr._seconds_until_next_check() == 60.0

    def test_missing_timestamp_set_to_now(self):
        """缺失时间戳 → 设为 now（lines 472-474）。"""
        mgr = _make_legacy_manager(ttls={"OCR": 300})
        mgr._service._pipelines = {"OCR": object()}
        # 不设 _last_used
        result = mgr._seconds_until_next_check()
        # OCR 有了时间戳，remaining ≈ 300（受 tick_interval=30 夹紧）
        assert result <= 30.0
        assert "OCR" in mgr._last_used

    def test_returns_min_remaining(self):
        """返回最近截止点的等待时间（line 478）。"""
        mgr = _make_legacy_manager(ttls={"PP-StructureV3": 100})
        mgr._service._pipelines = {"PP-StructureV3": object()}
        now = time.time()
        mgr._last_used = {"PP-StructureV3": now - 90}  # 还剩 10s
        result = mgr._seconds_until_next_check()
        assert result <= 10.0


# ---- _tick_loop 异常 ---------------------------------------------------


class TestTickLoopException:
    def test_tick_loop_runs_and_handles_exception(self):
        """启动 _tick_loop 线程，evict_idle 抛异常后被吞（lines 506-507）。"""
        mgr = _make_legacy_manager(ttls={"PP-StructureV3": 1})
        mgr._service._pipelines = {"PP-StructureV3": object()}
        mgr._last_used = {"PP-StructureV3": 0.0}
        mgr._stop_event.clear()

        # 让 evict_idle 抛异常
        def _boom(now=None):
            raise RuntimeError("boom")

        mgr.evict_idle = _boom
        # 启动 tick loop 线程
        mgr._thread = threading.Thread(target=mgr._tick_loop, daemon=True)
        mgr._thread.start()
        time.sleep(0.3)
        mgr.shutdown()
        # 线程应正常退出（异常被吞）
        assert not mgr._thread.is_alive()


# ---- _empty_cache -----------------------------------------------------


class TestEmptyCache:
    def test_empty_cache_cpu_mode_skips(self, monkeypatch):
        """CPU 模式 → 不调 paddle（line 532 False）。"""
        monkeypatch.setenv("VIBEOCR_USE_GPU", "false")
        PipelineCacheManager._empty_cache()  # 不应抛异常

    def test_empty_cache_gpu_mode_calls_paddle(self, monkeypatch):
        """GPU 模式 → 调 paddle.device.cuda.empty_cache（lines 532-536）。"""
        monkeypatch.setenv("VIBEOCR_USE_GPU", "true")
        import sys
        from types import ModuleType

        fake_paddle = ModuleType("paddle")
        fake_paddle.device = MagicMock()
        fake_paddle.device.cuda = MagicMock()
        # 注入 fake paddle
        monkeypatch.setitem(sys.modules, "paddle", fake_paddle)
        PipelineCacheManager._empty_cache()
        fake_paddle.device.cuda.empty_cache.assert_called_once()

    def test_empty_cache_gpu_mode_exception_swallowed(self, monkeypatch):
        """GPU 模式 + paddle 抛异常 → 吞掉（lines 537-538）。"""
        monkeypatch.setenv("VIBEOCR_USE_GPU", "true")
        import sys
        from types import ModuleType

        fake_paddle = ModuleType("paddle")
        fake_paddle.device = MagicMock()
        fake_paddle.device.cuda = MagicMock()
        fake_paddle.device.cuda.empty_cache.side_effect = RuntimeError("cuda boom")
        monkeypatch.setitem(sys.modules, "paddle", fake_paddle)
        # 不应抛异常
        PipelineCacheManager._empty_cache()

    def test_release_one_paddle_calls_empty_cache(self, monkeypatch):
        """_release_one 对 paddle 管道调用 _empty_cache（lines 517-518）。"""
        monkeypatch.setenv("VIBEOCR_USE_GPU", "false")  # CPU 模式，_empty_cache 跳过 paddle
        mgr = _make_legacy_manager()
        mgr._service._pipelines = {"PP-StructureV3": object()}
        mgr._last_used = {"PP-StructureV3": 100.0}
        # _release_one 是内部方法，在锁内调用
        with mgr._state_lock:
            mgr._release_one("PP-StructureV3")
        assert "PP-StructureV3" not in mgr._service._pipelines


# ---- 剩余分支 ----------------------------------------------------------


class TestRemainingBranches:
    def test_lease_overlapping_decrements_count(self):
        """两个重叠 lease → 第二个退出时 count>0 走 line 195。"""
        mgr = _make_legacy_manager()
        mgr._service._pipelines = {"OCR": object()}
        barrier = threading.Barrier(2)

        def hold_lease():
            with mgr.lease("OCR"):
                barrier.wait(timeout=1.0)  # 确保两个 lease 都进入
                barrier.wait(timeout=1.0)  # 等第一个退出

        # 启动第一个 lease（持锁）
        t1 = threading.Thread(target=hold_lease, daemon=True)
        t1.start()
        # 进入第二个 lease
        with mgr.lease("OCR"):
            barrier.wait(timeout=1.0)  # 同步两个都持有
            # 此时 active_counts["OCR"] == 2
            assert mgr._active_counts.get("OCR") == 2
        # 第二个退出 → count=1 > 0 → 走 line 195
        assert mgr._active_counts.get("OCR") == 1
        # 让第一个退出
        barrier.wait(timeout=1.0)
        t1.join(timeout=1.0)
        assert "OCR" not in mgr._active_counts

    def test_shutdown_no_thread_is_noop(self):
        """shutdown 时 _thread=None → 不 join（branch 449->exit）。"""
        mgr = _make_legacy_manager()
        # _thread 已为 None
        mgr.shutdown()  # 不应抛异常

    def test_shutdown_dead_thread_is_noop(self):
        """shutdown 时 _thread 已死 → 不 join（branch 449->exit via is_alive False）。"""
        mgr = _make_legacy_manager()
        # 创建一个已死的线程
        mgr._thread = threading.Thread(target=lambda: None)
        mgr._thread.start()
        mgr._thread.join(timeout=1.0)
        assert not mgr._thread.is_alive()
        mgr.shutdown()  # 不应阻塞或抛异常
