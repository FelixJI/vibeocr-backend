"""管道缓存生命周期管理（在 worker 子进程内运行）。

接管 OCRService._pipelines 的生命周期：
- 记录每个管道完成使用后的 last_used 时间戳
- FIFO 淘汰（超并存上限时淘汰最久未用的 paddle 重管道；MinerU 不计入）
- TTL 闲置回收（后台线程按最近到期时间唤醒）
- 显式释放（release）
- 按 cache_kind 分流回收：paddle 调 paddle.device.cuda.empty_cache()，mineru 不调
"""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from vibeocr.backend.services.ocr_service import OCRService

logger = logging.getLogger(__name__)

#: 显存分档阈值（MB）。≤8GB=1 并存，>8GB=2 并存。
VRAM_TIER_8GB = 8192
#: pynvml 不可用时的回退并存上限（保守，防 OOM）。
FALLBACK_MAX_HEAVY = 1
#: Parent-side restart recovery embeds original timestamps under this private key.
_RESTORE_LAST_USED_KEY = "__vibeocr_restore_last_used_unix_ms__"


def compute_max_heavy_by_vram(total_vram_mb: int) -> int:
    """按显存计算 paddle 重管道并存上限。

    Args:
        total_vram_mb: GPU 显存总量（MB），0 表示无法读取。

    Returns:
        并存上限：≤8G=1, >8G=2, 未知=1。
    """
    if total_vram_mb <= 0:
        return FALLBACK_MAX_HEAVY
    if total_vram_mb <= VRAM_TIER_8GB:
        return 1
    return 2


class PipelineCacheManager:
    """管道缓存生命周期管理器。

    在 worker 子进程内实例化，由 OCRService 持有。``lease`` 用活动计数保护
    一次模型加载/推理；TTL 线程据此跳过正在使用的管道，而状态锁只保护短暂的
    快照更新。有限 TTL 从任务完成时重新计时，而不是从任务开始时计时。
    """

    def __init__(
        self,
        service: OCRService,
        ttls: dict[str, int],
        max_heavy: int | None = None,
        tick_interval: float = 30.0,
    ) -> None:
        self._service = service
        self._ttls = dict(ttls)
        self._last_used: dict[str, float] = {}
        self._active_counts: dict[str, int] = {}
        self._pinned: set[str] = set()
        self._state_lock = threading.RLock()
        self._max_heavy = (
            max_heavy if max_heavy is not None else self._detect_max_heavy()
        )
        # tick_interval 是最长巡检间隔；真正等待时间会缩短到最近 TTL 截止点。
        self._tick_interval = max(0.01, float(tick_interval))
        self._stop_event = threading.Event()
        self._wakeup_event = threading.Event()
        self._thread = threading.Thread(
            target=self._tick_loop,
            name="PipelineTTLWatcher",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "[CacheManager] PipelineTTLWatcher 启动 (ttls=%s, max_heavy=%d, max_tick=%ss)",
            dict(self._ttls),
            self._max_heavy,
            self._tick_interval,
        )

    def _ensure_runtime_fields(self) -> None:
        """兼容通过 ``__new__`` 构造的历史单元测试替身。"""
        if not hasattr(self, "_state_lock"):
            self._state_lock = threading.RLock()
        if not hasattr(self, "_active_counts"):
            self._active_counts = {}
        if not hasattr(self, "_pinned"):
            self._pinned = set()
        if not hasattr(self, "_wakeup_event"):
            self._wakeup_event = threading.Event()
        if not hasattr(self, "_stop_event"):
            self._stop_event = threading.Event()
        if not hasattr(self, "_tick_interval"):
            self._tick_interval = 30.0

    def _detect_max_heavy(self) -> int:
        """读 GPU 显存总量算并存上限，失败回退。

        CPU 模式（VIBEOCR_USE_GPU != true）固定返回 1（串行更稳）。
        """
        if os.environ.get("VIBEOCR_USE_GPU", "").lower() != "true":
            return 1
        try:
            from vibeocr.backend.utils.gpu_memory_monitor import GPUMemoryMonitor

            info = GPUMemoryMonitor().get_status()
            if info.available and info.total > 0:
                return compute_max_heavy_by_vram(info.total)
        except Exception as e:
            logger.warning(
                "[CacheManager] 检测显存失败，回退上限 %d: %s",
                FALLBACK_MAX_HEAVY,
                e,
            )
        return FALLBACK_MAX_HEAVY

    # ------------------------------------------------------------------
    # 公共属性 / 使用租约
    # ------------------------------------------------------------------
    @property
    def ttls(self) -> dict[str, int]:
        self._ensure_runtime_fields()
        with self._state_lock:
            return dict(self._ttls)

    @ttls.setter
    def ttls(self, value: Mapping[str, object]) -> None:
        self._ensure_runtime_fields()
        from vibeocr.backend.core.pipelines import get_all_pipelines

        restore_raw = value.get(_RESTORE_LAST_USED_KEY)
        restore_values = restore_raw if isinstance(restore_raw, dict) else None
        valid_names = {pipeline.value for pipeline in get_all_pipelines()}
        validated: dict[str, int] = {}
        for name, ttl in value.items():
            if name == _RESTORE_LAST_USED_KEY:
                continue
            if name not in valid_names:
                logger.warning("[CacheManager] 忽略未知管道 TTL: %s", name)
                continue
            if isinstance(ttl, bool):
                logger.warning("[CacheManager] 忽略 bool TTL: %s=%r", name, ttl)
                continue
            try:
                validated[name] = max(0, int(ttl))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                logger.warning("[CacheManager] 忽略无效 TTL: %s=%r", name, ttl)

        now = time.time()
        with self._state_lock:
            self._ttls = validated
            # TTL 可能在模型已加载后才下发。没有时间戳时从“配置生效”开始计时，
            # 而不是用 0 导致有限 TTL 被立刻回收。
            for name in self._service._pipelines:
                self._last_used.setdefault(name, now)
        if restore_values is not None:
            self.restore_last_used_unix_ms(restore_values)
        self._wakeup_event.set()

    @property
    def max_heavy(self) -> int:
        return self._max_heavy

    @contextmanager
    def lease(self, pipeline_name: str) -> Iterator[None]:
        """保护一次模型加载/推理，并在完成时重置闲置 TTL。

        只在更新活动计数和时间戳时持有状态锁；TTL/显式回收根据
        ``active_counts`` 跳过活动管道。这样长耗时加载或推理不会阻塞只读状态
        快照。``last_used`` 在 ``finally`` 中记录完成时间；异常请求也视为一次
        使用，避免错误处理尚未结束时立刻触发回收。
        """
        self._ensure_runtime_fields()
        with self._state_lock:
            self._active_counts[pipeline_name] = (
                self._active_counts.get(pipeline_name, 0) + 1
            )
        try:
            yield
        finally:
            with self._state_lock:
                count = self._active_counts.get(pipeline_name, 1) - 1
                if count > 0:
                    self._active_counts[pipeline_name] = count
                else:
                    self._active_counts.pop(pipeline_name, None)
                if pipeline_name in self._service._pipelines:
                    self._last_used[pipeline_name] = time.time()
            self._wakeup_event.set()

    # ------------------------------------------------------------------
    # 时间戳 / 容量管理
    # ------------------------------------------------------------------
    def touch(self, pipeline_name: str, now: float | None = None) -> None:
        """记录管道使用时间。每次 get_or_create_pipeline 后调用。"""
        self._ensure_runtime_fields()
        with self._state_lock:
            self._last_used[pipeline_name] = now if now is not None else time.time()
        self._wakeup_event.set()

    def restore_last_used_unix_ms(self, values: dict[str, object]) -> None:
        """恢复重启前的时间戳，使有限 TTL 不因 worker 重启被重新计时。"""
        self._ensure_runtime_fields()
        now = time.time()
        with self._state_lock:
            for name, raw_ms in values.items():
                if (
                    name not in self._service._pipelines
                    or isinstance(raw_ms, bool)
                    or not isinstance(raw_ms, (int, float, str))
                ):
                    continue
                try:
                    timestamp = max(0.0, float(raw_ms) / 1000.0)
                except (TypeError, ValueError):
                    logger.warning(
                        "[CacheManager] 忽略无效恢复时间戳: %s=%r", name, raw_ms
                    )
                    continue
                # 防止错误/未来时间让有限 TTL 无限延长。
                self._last_used[name] = min(timestamp, now)
        self._wakeup_event.set()

    def get_last_used(self, pipeline_name: str) -> float | None:
        self._ensure_runtime_fields()
        with self._state_lock:
            return self._last_used.get(pipeline_name)

    def enforce_capacity(
        self, new_pipeline: str, now: float | None = None
    ) -> list[str]:
        """加载新 paddle 重管道前，FIFO 淘汰至不超并存上限。

        TTL 只定义闲置回收；按既有设计，容量压力仍可提前淘汰 TTL=0 或尚未
        到期的重管道。正在使用的管道不会成为容量淘汰候选。
        """
        del now  # 保留兼容签名；排序使用真实 last_used。
        self._ensure_runtime_fields()
        from vibeocr.backend.core.pipelines import (
            get_heavy_pipelines,
            get_paddle_residency_pipelines,
        )

        paddle_names = {pipeline.value for pipeline in get_paddle_residency_pipelines()}
        heavy_paddle_names = paddle_names & {
            pipeline.value for pipeline in get_heavy_pipelines()
        }
        evicted: list[str] = []
        with self._state_lock:
            cached_heavy = [
                name
                for name in self._service._pipelines
                if name in heavy_paddle_names
                and name != new_pipeline
                and self._active_counts.get(name, 0) <= 0
                and name not in self._pinned
            ]
            while len(cached_heavy) >= self._max_heavy:
                cached_heavy.sort(key=lambda name: self._last_used.get(name, 0.0))
                victim = cached_heavy.pop(0)
                self._release_one(victim)
                evicted.append(victim)
        if evicted:
            logger.info(
                "[CacheManager] 容量 FIFO 淘汰（新管道=%s, max=%d）: %s",
                new_pipeline,
                self._max_heavy,
                evicted,
            )
        return evicted

    def prepare_load(self, new_pipeline: str) -> list[str]:
        """Create capacity before loading without evicting active/pinned models."""
        self._ensure_runtime_fields()
        from vibeocr.backend.core.pipelines import (
            get_heavy_pipelines,
            get_paddle_residency_pipelines,
        )

        heavy = {pipeline.value for pipeline in get_heavy_pipelines()}
        paddle = {pipeline.value for pipeline in get_paddle_residency_pipelines()}
        heavy_paddle = heavy & paddle
        if new_pipeline not in heavy_paddle:
            return []
        evicted: list[str] = []
        with self._state_lock:
            loaded = [
                name
                for name in self._service._pipelines
                if name in heavy_paddle and name != new_pipeline
            ]
            while len(loaded) >= self._max_heavy:
                candidates = [
                    name
                    for name in loaded
                    if self._active_counts.get(name, 0) <= 0
                    and name not in self._pinned
                ]
                if not candidates:
                    raise RuntimeError(
                        "PIN_CAPACITY_CONFLICT: no idle unpinned model can be evicted"
                    )
                victim = min(
                    candidates,
                    key=lambda name: self._last_used.get(name, 0.0),
                )
                self._release_one(victim)
                loaded.remove(victim)
                evicted.append(victim)
        return evicted

    def configure_residency(
        self,
        *,
        default_ttl_seconds: int,
        pipelines: list[object],
    ) -> None:
        """Atomically translate supervisor settings into physical cache policy."""
        from vibeocr.backend.core.pipelines import get_all_pipelines

        if default_ttl_seconds < 0:
            raise ValueError("default_ttl_seconds must be >= 0")
        known = {pipeline.value for pipeline in get_all_pipelines()}
        ttls: dict[str, int] = dict.fromkeys(known, default_ttl_seconds)
        pinned: set[str] = set()
        for spec in pipelines:
            name = str(getattr(spec, "name", ""))
            if name not in known:
                raise ValueError(f"unknown pipeline residency policy: {name}")
            ttl = getattr(spec, "ttl_seconds", None)
            if ttl is not None:
                if int(ttl) < 0:
                    raise ValueError(f"negative TTL for pipeline: {name}")
                ttls[name] = int(ttl)
            if bool(getattr(spec, "pinned", False)):
                pinned.add(name)
                ttls[name] = 0
        with self._state_lock:
            self._pinned = pinned
        self.ttls = ttls

    def evict_idle(self, now: float | None = None) -> list[str]:
        """回收闲置时间达到各自 TTL 的管道。

        ``ttl<=0`` 不受闲置回收；有限 TTL 从最近一次加载/推理完成时计算。
        活跃租约始终跳过。
        """
        self._ensure_runtime_fields()
        current = now if now is not None else time.time()
        evicted: list[str] = []
        with self._state_lock:
            for name in list(self._service._pipelines):
                if self._active_counts.get(name, 0) > 0:
                    continue
                if name in self._pinned:
                    continue
                ttl = self._ttls.get(name, 0)
                if ttl <= 0:
                    continue
                last = self._last_used.get(name)
                if last is None:
                    # 缺失时间戳不等于“从 Unix epoch 闲置”；从现在开始计时。
                    self._last_used[name] = current
                    continue
                if last + ttl <= current:
                    self._release_one(name)
                    evicted.append(name)
        if evicted:
            logger.info(
                "[CacheManager] TTL 回收 %d 个闲置管道: %s",
                len(evicted),
                evicted,
            )
        return evicted

    def release(self, heavy_only: bool = True, *, force: bool = False) -> list[str]:
        """显式释放管道。活跃任务会先持有 lease，故本调用等待任务完成。"""
        self._ensure_runtime_fields()
        from vibeocr.backend.core.pipelines import get_heavy_pipelines

        heavy_names = {pipeline.value for pipeline in get_heavy_pipelines()}
        released: list[str] = []
        with self._state_lock:
            for name in list(self._service._pipelines):
                if heavy_only and name not in heavy_names:
                    continue
                if name in self._pinned and not force:
                    continue
                self._release_one(name)
                released.append(name)
        logger.info(
            "[CacheManager] release(heavy_only=%s) 释放 %d 个管道: %s",
            heavy_only,
            len(released),
            released,
        )
        self._wakeup_event.set()
        return released

    def release_one(self, pipeline_name: str, *, force: bool = False) -> bool:
        """显式释放单个管道并清理其使用记录。"""
        self._ensure_runtime_fields()
        with self._state_lock:
            existed = pipeline_name in self._service._pipelines
            if pipeline_name not in self._pinned or force:
                self._release_one(pipeline_name)
            else:
                existed = False
        if existed:
            logger.info("[CacheManager] 释放单个管道: %s", pipeline_name)
        self._wakeup_event.set()
        return existed

    def status(self) -> dict[str, object]:
        """Return an immutable wire-friendly snapshot of the real worker cache."""
        self._ensure_runtime_fields()
        with self._state_lock:
            loaded = sorted(str(name) for name in self._service._pipelines)
            return {
                "pipeline_ttls": dict(self._ttls),
                "max_heavy": self._max_heavy,
                "loaded_pipelines": loaded,
                "active_counts": {
                    name: self._active_counts.get(name, 0) for name in loaded
                },
                "pinned_pipelines": sorted(self._pinned),
                "last_used_unix_ms": {
                    name: int(self._last_used[name] * 1000)
                    for name in loaded
                    if name in self._last_used
                },
            }

    def shutdown(self) -> None:
        """停止后台 watcher，等待最多 2 秒退出。"""
        self._ensure_runtime_fields()
        self._stop_event.set()
        self._wakeup_event.set()
        thread = self._thread if hasattr(self, "_thread") else None
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    # 后台 watcher
    # ------------------------------------------------------------------
    def _seconds_until_next_check(self) -> float:
        """按最近有限 TTL 截止点计算等待时间。"""
        self._ensure_runtime_fields()
        now = time.time()
        with self._state_lock:
            if not self._service._pipelines:
                return 60.0
            remaining: list[float] = []
            for name in self._service._pipelines:
                ttl = self._ttls.get(name, 0)
                if (
                    ttl <= 0
                    or self._active_counts.get(name, 0) > 0
                    or name in self._pinned
                ):
                    continue
                last = self._last_used.get(name)
                if last is None:
                    self._last_used[name] = now
                    last = now
                remaining.append(max(0.0, last + ttl - now))
            if not remaining:
                return 60.0
            return max(0.0, min(self._tick_interval, min(remaining)))

    def _tick_loop(self) -> None:
        """按最近截止点等待；touch/TTL 变更会立即唤醒并重算。"""
        while not self._stop_event.is_set():
            timeout = self._seconds_until_next_check()
            self._wakeup_event.wait(timeout=timeout)
            self._wakeup_event.clear()
            if self._stop_event.is_set():
                break
            try:
                with self._state_lock:
                    loaded = sorted(self._service._pipelines)
                    now = time.time()
                    due = [
                        name
                        for name in loaded
                        if self._active_counts.get(name, 0) <= 0
                        and name not in self._pinned
                        and self._ttls.get(name, 0) > 0
                        and self._last_used.get(name, now) + self._ttls[name] <= now
                    ]
                logger.debug(
                    "[CacheManager] tick: loaded=%s due_for_evict=%s",
                    loaded,
                    due,
                )
                self.evict_idle()
            except Exception as e:
                logger.warning("[CacheManager] tick evict_idle 失败: %s", e)

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    def _release_one(self, pipeline_name: str) -> None:
        """释放单个管道；调用方已持有可重入 state lock。"""
        self._service._pipelines.pop(pipeline_name, None)
        self._last_used.pop(pipeline_name, None)
        self._active_counts.pop(pipeline_name, None)
        if self._is_paddle(pipeline_name):
            self._empty_cache()

    @staticmethod
    def _is_paddle(pipeline_name: str) -> bool:
        from vibeocr.backend.core.pipelines import get_paddle_residency_pipelines

        return pipeline_name in {
            pipeline.value for pipeline in get_paddle_residency_pipelines()
        }

    @staticmethod
    def _empty_cache() -> None:
        """GPU 模式下回收显存碎片。"""
        try:
            if os.environ.get("VIBEOCR_USE_GPU", "").lower() == "true":
                import paddle

                paddle.device.cuda.empty_cache()
                logger.info("[CacheManager] paddle.device.cuda.empty_cache() 已调用")
        except Exception as e:
            logger.debug("[CacheManager] empty_cache 跳过: %s", e)
