"""跨平台读取可用物理内存，仅使用标准库（无 psutil 依赖）。

用于 CPU 模式下动态计算 PDF 批量大小，避免在小内存设备上 OOM。
"""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)

#: RAM 读取失败时的保守回退值（MB）。
#: 2GB 可用，按 CPU 规则（8× 放大、0.3 安全系数）→ batch ≈ 1-2。
FALLBACK_RAM_MB = 2048


def get_available_ram_mb() -> int:
    """获取当前可用物理内存（MB）。

    Windows 使用 ctypes 调用 GlobalMemoryStatusEx；
    Linux 读取 /proc/meminfo 的 MemAvailable；
    其他平台或读取失败时回退到 FALLBACK_RAM_MB。

    Returns:
        可用内存（MB），至少为正数。
    """
    try:
        mb = _read_available_ram()
        if mb and mb > 0:
            return int(mb)
    except Exception as e:  # 读取系统信息，任何失败都回退到 FALLBACK_RAM_MB
        logger.warning(
            "[system_memory] 读取可用内存失败，回退到 %dMB: %s", FALLBACK_RAM_MB, e
        )
    return FALLBACK_RAM_MB


def _read_available_ram() -> int | None:
    """平台分发：返回可用内存（MB）或 None。"""
    if sys.platform == "win32":
        return _read_windows()
    if sys.platform.startswith("linux"):
        return _read_linux()
    return None


def _read_windows() -> int | None:
    """Windows: ctypes + GlobalMemoryStatusEx。"""
    import ctypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
        return None
    return stat.ullAvailPhys // (1024 * 1024)


def _read_linux() -> int | None:
    """Linux: 读取 /proc/meminfo 的 MemAvailable（kB）。"""
    try:  # pragma: no cover - Linux-only path, not exercisable on Windows CI
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    # 行格式: "MemAvailable:  12345678 kB"
                    parts = line.split()
                    return int(parts[1]) // 1024
    except (OSError, ValueError, IndexError):  # pragma: no cover - Linux-only
        return None
    return None  # pragma: no cover - Linux-only


#: CPU 模式每页峰值放大系数（oneDNN 工作区 + 多线程缓冲，比 GPU 大）。
CPU_AMP_FACTOR = 8
#: CPU 模式安全系数（RAM 与系统/UI 共享，留更多余量）。
CPU_SAFETY_FACTOR = 0.3
#: CPU 模式 batch 上限（低于 GPU，RAM 更紧张）。
CPU_BATCH_CAP = 6


def estimate_cpu_batch_size(free_mb: int, avg_pixels: int) -> int:
    """按可用 RAM 和平均像素数估算 CPU 批量大小。

    单页峰值（MB）= avg_pixels * 3 字节 * 8× 放大 / 1MB。
    batch = free * 0.3 / 单页峰值，夹到 [1, 6]。

    Args:
        free_mb: 可用 RAM（MB）。
        avg_pixels: 单页平均像素数（width * height）。

    Returns:
        批量大小，范围 [1, 6]。
    """
    if free_mb <= 0 or avg_pixels <= 0:
        return 1
    per_page_peak_mb = (avg_pixels * 3 * CPU_AMP_FACTOR) / (1024 * 1024)
    if per_page_peak_mb <= 0:  # pragma: no cover - avg_pixels>0 guarded above
        return 1
    usable_mb = free_mb * CPU_SAFETY_FACTOR
    batch = int(usable_mb / per_page_peak_mb)
    return max(1, min(batch, CPU_BATCH_CAP))
