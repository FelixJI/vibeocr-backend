"""GPU 显存监控工具

通过 pynvml 监控 NVIDIA GPU 显存状态。
"""

import contextlib
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GPUMemoryInfo:
    """GPU 显存信息"""

    total: int  # 总显存 (MB)
    free: int  # 空闲显存 (MB)
    used: int  # 已用显存 (MB)
    available: bool  # 是否可用


class GPUMemoryMonitor:
    """GPU 显存监控器

    使用 pynvml 获取显存信息，供 MineRU 服务判断 GPU 加速能力。
    """

    def __init__(self, device_id: int = 0):
        """初始化显存监控器

        Args:
            device_id: GPU 设备 ID
        """
        self.device_id = device_id
        self._pynvml_available = False

        # 尝试初始化 pynvml
        try:
            import pynvml  # type: ignore[import-untyped]

            pynvml.nvmlInit()
            self._pynvml_available = True
            self._pynvml = pynvml
            logger.debug("pynvml 初始化成功，将使用 NVML 监控显存")
        except Exception as e:
            logger.debug(f"pynvml 不可用: {e}")

    def get_status(self) -> GPUMemoryInfo:
        """获取当前 GPU 显存状态

        Returns:
            GPUMemoryInfo: 显存信息
        """
        if self._pynvml_available:
            return self._get_status_pynvml()

        return GPUMemoryInfo(total=0, free=0, used=0, available=False)

    def _get_status_pynvml(self) -> GPUMemoryInfo:
        """使用 pynvml 获取显存状态"""
        try:
            handle = self._pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
            mem_info = self._pynvml.nvmlDeviceGetMemoryInfo(handle)

            total_mb = mem_info.total // (1024 * 1024)
            free_mb = mem_info.free // (1024 * 1024)
            used_mb = mem_info.used // (1024 * 1024)

            return GPUMemoryInfo(
                total=total_mb, free=free_mb, used=used_mb, available=True
            )
        except Exception as e:
            logger.warning(f"pynvml 获取显存失败: {e}")
            return GPUMemoryInfo(total=0, free=0, used=0, available=False)

    def estimate_batch_size(
        self, avg_image_pixels: int, safety_factor: float = 0.7
    ) -> int:
        """根据当前显存估算安全的 batch_size

        Args:
            avg_image_pixels: 平均图片像素数 (width * height)
            safety_factor: 安全系数 (0-1)，预留显存比例

        Returns:
            推荐的 batch_size
        """
        mem_info = self.get_status()

        if not mem_info.available:
            return 4

        pixels_per_million = avg_image_pixels / 1_000_000
        mem_per_image_mb = pixels_per_million * 3

        usable_mem = mem_info.free * safety_factor
        batch_size = int(usable_mem / mem_per_image_mb)

        return max(1, min(batch_size, 16))

    def is_available(self) -> bool:
        """检查显存监控是否可用"""
        return self._pynvml_available

    def close(self):
        """清理资源"""
        if self._pynvml_available:
            with contextlib.suppress(Exception):
                self._pynvml.nvmlShutdown()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# --- 模块级分批计算工具（供 PDF 动态 BATCH_SIZE 复用）---

#: GPU 模式每页峰值放大系数（含 PaddleOCR 内部多份副本）。
GPU_AMP_FACTOR = 5
#: GPU 模式安全系数（只用一半 free 显存）。
GPU_SAFETY_FACTOR = 0.5
#: GPU 模式 batch 上限（避免超时风险）。
GPU_BATCH_CAP = 10
#: 显存探测失败但已知处于 GPU 模式时的保守兜底批量。
#:
#: 背景：NVML（pynvml）在 Windows 上常因驱动/NVML 版本不匹配而初始化失败，
#: 此时 ``free_mb`` 为 0。若直接返回 batch=1，4090 等大显存卡也会被钉死成
#: 逐张识别（GPU 占用仅 20-40%）。这里给出一个保守但合理的默认值，
#: 让显存探测失败时仍能受益于批量识别；真正的非法输入（avg_pixels<=0）
#: 仍返回 1。
GPU_FALLBACK_BATCH_SIZE = 4
#: 每像素字节数（RGB）。
BYTES_PER_PIXEL = 3


def estimate_gpu_batch_size(free_mb: int, avg_pixels: int) -> int:
    """按可用显存和平均像素数估算 GPU 批量大小。

    单页峰值（MB）= avg_pixels * 3 字节 * 5× 放大 / 1MB。
    batch = free * 0.5 / 单页峰值，夹到 [1, 10]。

    - ``free_mb <= 0``：显存探测失败（NVML 不可用），但调用方已确认处于
      GPU 模式，返回 :data:`GPU_FALLBACK_BATCH_SIZE` 而非 1，避免大显存卡
      被迫逐张识别。
    - ``avg_pixels <= 0``：非法输入（拿不到页面尺寸），返回 1。

    Args:
        free_mb: 可用显存（MB）。
        avg_pixels: 单页平均像素数（width * height）。

    Returns:
        批量大小，范围 [1, 10]。
    """
    if avg_pixels <= 0:
        return 1
    if free_mb <= 0:
        return max(1, min(GPU_FALLBACK_BATCH_SIZE, GPU_BATCH_CAP))
    per_page_peak_mb = (avg_pixels * BYTES_PER_PIXEL * GPU_AMP_FACTOR) / (1024 * 1024)
    if per_page_peak_mb <= 0:
        return 1
    usable_mb = free_mb * GPU_SAFETY_FACTOR
    batch = int(usable_mb / per_page_peak_mb)
    return max(1, min(batch, GPU_BATCH_CAP))
