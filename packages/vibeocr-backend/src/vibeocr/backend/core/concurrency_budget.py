"""集中并发预算配置。

将分散在各处的线程/进程数配置集中管理，避免未来启用多 worker 时
出现 ``worker数 × OMP线程数 + GUI线程池`` 过度订阅。

当前 OCR worker 固定 1 个（SubprocessManager max_workers=1）。未来启用
多 worker 时，必须按 ``物理/逻辑核预算 ÷ worker数`` 下调每进程 OMP/MKL
线程，否则会过度订阅。
"""

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConcurrencyBudget:
    """并发预算配置。

    Attributes:
        ocr_workers: OCR worker 子进程数（当前固定 1）。
        omp_threads_per_worker: 每个 worker 进程的 OpenMP/MKL 线程数。
        pdf_render_concurrency: PDF 渲染并发线程数（后端 fitz 栅格化串行，
            但 PIL/PNG 编码 + HTTP 往返可并行）。
        qt_pool_limit: Qt 全局 QThreadPool 上限。
        asyncio_executor_limit: asyncio 默认 executor 线程上限。
    """

    ocr_workers: int = 1
    omp_threads_per_worker: int = 4
    pdf_render_concurrency: int = 4
    qt_pool_limit: int = 4
    asyncio_executor_limit: int = 4

    @classmethod
    def default(cls) -> "ConcurrencyBudget":
        """根据 CPU 核数计算默认预算。

        单 worker 时 OMP 可用全部核；未来多 worker 需按 worker 数下调。
        """
        from vibeocr.backend.utils.cpu_info import get_cpu_thread_count

        logical = os.cpu_count() or 4
        try:
            omp = get_cpu_thread_count()
        except Exception:
            omp = min(logical, 8)

        workers = 1
        omp_per = omp if workers == 1 else max(omp // workers, 1)

        return cls(
            ocr_workers=workers,
            omp_threads_per_worker=omp_per,
            pdf_render_concurrency=min(4, logical),
            qt_pool_limit=min(4, logical),
            asyncio_executor_limit=min(4, logical),
        )

    def log_summary(self) -> None:
        """记录实际预算到日志。"""
        logger.info(
            "ConcurrencyBudget: OCR workers=%d, OMP/worker=%d, "
            "PDF render=%d, Qt pool=%d, asyncio executor=%d, "
            "total CPU threads=%d",
            self.ocr_workers,
            self.omp_threads_per_worker,
            self.pdf_render_concurrency,
            self.qt_pool_limit,
            self.asyncio_executor_limit,
            self.ocr_workers * self.omp_threads_per_worker,
        )
