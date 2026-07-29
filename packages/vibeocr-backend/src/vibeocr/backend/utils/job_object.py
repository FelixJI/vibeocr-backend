"""Windows Job Object 守卫。

主进程退出（含 os._exit / 段错误 / 任务管理器强杀）时，内核连带终止
被绑定的子进程，回收 GPU 显存和共享内存。

非 Windows 平台为 no-op 兼容实现。
"""

import ctypes
import logging
import subprocess
import sys
from ctypes import wintypes

logger = logging.getLogger(__name__)

_IS_WINDOWS = sys.platform == "win32"

# ---- Windows 常量（不导出，仅模块内使用）----
JOB_OBJECT_LIMIT_BREAKAWAY_OK = 0x00000080
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JobObjectExtendedLimitInformation = 9
PROCESS_SET_QUOTA = 0x0100
PROCESS_TERMINATE = 0x0001


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
        ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.POINTER(wintypes.ULONG)),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _JOBOBJECT_BASIC_UI_RESTRICTIONS(ctypes.Structure):
    _fields_ = [("UIRestrictionsClass", wintypes.DWORD)]


class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", _IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


def _get_kernel32():
    """获取 kernel32 句柄（仅 Windows 可调用）。"""
    return ctypes.windll.kernel32  # type: ignore[attr-defined]


class JobObjectGuard:
    """Windows Job Object 守卫。

    主进程所有 Job 句柄关闭时，内核终止 Job 内全部进程
    （JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE）。

    仅 Windows 生效；其他平台为 no-op。
    """

    def __init__(self, name: str | None = None) -> None:
        self._name = name
        self._handle: int | None = None  # Windows: HANDLE；其他平台: None

        if not _IS_WINDOWS:
            return

        self._create_job()

    def _create_job(self) -> None:
        """创建并配置 Job Object（Windows）。"""
        try:
            kernel32 = _get_kernel32()
            # lpName：可选命名；NULL 表示匿名 Job
            name_c = ctypes.c_wchar_p(self._name) if self._name else None

            handle = kernel32.CreateJobObjectW(None, name_c)
            if not handle:
                logger.warning("[JobObject] CreateJobObjectW 失败，子进程孤儿防护降级")
                return

            # 配置 KILL_ON_JOB_CLOSE + BREAKAWAY_OK
            info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            info.BasicLimitInformation.LimitFlags = (
                JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_BREAKAWAY_OK
            )
            ok = kernel32.SetInformationJobObject(
                handle,
                JobObjectExtendedLimitInformation,
                ctypes.byref(info),
                ctypes.sizeof(info),
            )
            if not ok:
                logger.warning(
                    "[JobObject] SetInformationJobObject 失败，子进程孤儿防护降级"
                )
                kernel32.CloseHandle(handle)
                return

            self._handle = handle
            logger.debug(f"[JobObject] 已创建 Job Object (name={self._name})")
        except Exception as e:
            logger.warning(f"[JobObject] 创建 Job Object 异常: {e}，降级")

    def assign_from_popen(self, popen: subprocess.Popen) -> bool:
        """把 popen 启动的子进程加入本 Job。

        Returns:
            是否成功绑定（False 仅表示降级，不抛异常）。
        """
        if not _IS_WINDOWS or self._handle is None:
            return False
        return self._assign_pid(popen.pid)

    def _assign_pid(self, pid: int) -> bool:
        """通过 pid 绑定进程（Windows）。

        OpenProcess → AssignProcessToJobObject → CloseHandle(子进程句柄)。
        赋值后 Job 已持有引用，子进程句柄即可关闭。
        """
        try:
            kernel32 = _get_kernel32()
            # PROCESS_SET_QUOTA | PROCESS_TERMINATE
            proc_handle = kernel32.OpenProcess(
                PROCESS_SET_QUOTA | PROCESS_TERMINATE, False, pid
            )
            if not proc_handle:
                logger.warning(f"[JobObject] OpenProcess(pid={pid}) 失败，子进程未绑定")
                return False

            try:
                ok = kernel32.AssignProcessToJobObject(self._handle, proc_handle)
                if not ok:
                    logger.warning(
                        f"[JobObject] AssignProcessToJobObject(pid={pid}) 失败"
                    )
                    return False
                logger.debug(f"[JobObject] 已绑定进程 pid={pid}")
                return True
            finally:
                # 子进程句柄用完即关；Job 已持有引用
                kernel32.CloseHandle(proc_handle)
        except Exception as e:
            logger.warning(f"[JobObject] 绑定进程异常: {e}，降级")
            return False

    def close(self) -> None:
        """关闭 Job 句柄。最后一个句柄关闭时内核终止 Job 内所有进程。

        幂等：多次调用安全。
        """
        if not _IS_WINDOWS or self._handle is None:
            return
        try:
            kernel32 = _get_kernel32()
            kernel32.CloseHandle(self._handle)
            logger.debug(f"[JobObject] 已关闭 Job 句柄 (name={self._name})")
        except Exception as e:
            logger.warning(f"[JobObject] 关闭 Job 句柄异常: {e}")
        finally:
            self._handle = None

    def __enter__(self) -> "JobObjectGuard":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
