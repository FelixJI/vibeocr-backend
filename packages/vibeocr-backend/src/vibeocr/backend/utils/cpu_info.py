"""CPU 硬件探测：逻辑核数与 oneDNN 安全性判定，仅使用标准库（无 psutil）。

用途：
- 在进程启动早期自适应设置 OpenMP / paddle CPU 线程数，避免默认 1 线程
  浪费多核 CPU（i9-14900KF 这类 32 线程 CPU 默认仅用 1 核）。
- 在构造 PaddleOCR 管道前判定当前 CPU 是否能安全启用 oneDNN（MKL-DNN），
  替代历史代码里硬编码的 ``enable_mkldnn=False``。

关于 oneDNN 的控制路径（重要）：
- PaddleOCR 推理走 ``paddle.inference.Config``（AnalysisConfig），mkldnn 开关
  由构造函数 ``enable_mkldnn`` 参数控制（最终落到
  ``config.enable_mkldnn()`` / ``config.disable_mkldnn()``）。
- 进程级 FLAGS（``FLAGS_use_mkldnn`` / ``FLAGS_enable_onednn_backend``）对这条
  推理路径【不生效】——paddleocr/paddlex 零处读取它们，只对 eager 路径有意义。
  故本模块的 ``can_safely_enable_onednn`` 返回值（经 OCRService 转成
  ``enable_mkldnn`` kwarg）才是 oneDNN 是否启用的唯一真相。
"""

from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger(__name__)

#: CPU 线程数回退值（探测失败时用，保守）。
FALLBACK_CPU_THREADS = 4
#: 单进程 CPU 线程上限（paddle/OpenMP 线程过多反而因 NUMA/调度抖动变慢，
#: OCR 推理以卷积+矩阵乘为主，经验上限 16）。
CPU_THREADS_CAP = 16


def get_cpu_thread_count() -> int:
    """返回推荐的 CPU 推理线程数，范围 [1, CPU_THREADS_CAP]。

    策略：取逻辑核数，夹到上限；探测失败回退 FALLBACK_CPU_THREADS。
    优先读 ``VIBEOCR_CPU_THREADS`` 环境变量（用户显式覆盖）。

    Returns:
        推荐线程数（正整数）。
    """
    # 用户显式覆盖优先
    override = os.environ.get("VIBEOCR_CPU_THREADS", "").strip()
    if override:
        try:
            n = int(override)
            if n > 0:
                return n
        except ValueError:
            logger.warning("[cpu_info] VIBEOCR_CPU_THREADS=%r 非整数，忽略", override)

    try:
        n = os.cpu_count() or FALLBACK_CPU_THREADS
    except Exception:  # 系统调用失败兜底
        n = FALLBACK_CPU_THREADS
    return max(1, min(n, CPU_THREADS_CAP))


# ---------------------------------------------------------------------------
# CPU 指令集探测
# ---------------------------------------------------------------------------


def detect_cpu_features() -> dict[str, bool]:
    """探测当前 CPU 的关键指令集支持情况。

    Returns:
        含 ``avx`` / ``avx2`` / ``avx512`` / ``fma`` / ``amx`` 布尔值的字典。
        探测失败时各项均为 False。
    """
    flags_text = _read_cpu_flags_text()
    if not flags_text:
        return {
            "avx": False,
            "avx2": False,
            "avx512": False,
            "fma": False,
            "amx": False,
        }

    return {
        "avx": "avx" in flags_text,
        "avx2": "avx2" in flags_text,
        "fma": "fma" in flags_text,
        # AVX-512 家族任一子集即视为支持
        "avx512": bool(re.search(r"avx512[a-z_]*", flags_text)),
        "amx": "amx" in flags_text,
    }


def _read_cpu_flags_text() -> str:
    """返回包含 CPU 特性标志的小写字符串，失败返回空串。

    Windows 通过 ``IsProcessorFeaturePresent`` 探测部分指令集；
    Linux 读取 /proc/cpuinfo 的 flags 行。
    """
    if os.name == "nt":
        return _read_windows_features()
    if os.name == "posix":
        return _read_linux_flags()
    return ""


def _read_windows_features() -> str:
    """Windows: 用 kernel32 IsProcessorFeaturePresent 探测指令集。

    PF 常量见 MSDN ``PROCESSOR_FEATURE_ID``。Python ctypes 无法直接读
    CPUID，故只覆盖 paddle/oneDNN 关心的几项；AMX 无 PF 常量，置 False。
    """
    import ctypes

    # PROCESSOR_FEATURE_ID 子集（仅取 paddle/oneDNN 关心的）
    PF_XMMI_INSTRUCTIONS_AVAILABLE = 1  # SSE
    PF_XMMI64_INSTRUCTIONS_AVAILABLE = 10  # SSE2
    PF_SSE3_INSTRUCTIONS_AVAILABLE = 13
    PF_AVX_INSTRUCTIONS_AVAILABLE = 39  # 含 AVX（PF 命名历史遗留）
    PF_AVX2_INSTRUCTIONS_AVAILABLE = 40
    PF_AVX512F_INSTRUCTIONS_AVAILABLE = 43  # AVX-512 Foundation

    try:
        k32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        present = k32.IsProcessorFeaturePresent

        def has(fid: int) -> bool:
            return bool(present(ctypes.c_uint(fid)))

        parts: list[str] = []
        if has(PF_XMMI_INSTRUCTIONS_AVAILABLE):
            parts.append("sse")
        if has(PF_XMMI64_INSTRUCTIONS_AVAILABLE):
            parts.append("sse2")
        if has(PF_SSE3_INSTRUCTIONS_AVAILABLE):
            parts.append("sse3")
        if has(PF_AVX_INSTRUCTIONS_AVAILABLE):
            parts.append("avx")
        if has(PF_AVX2_INSTRUCTIONS_AVAILABLE):
            parts.append("avx2")
        if has(PF_AVX512F_INSTRUCTIONS_AVAILABLE):
            parts.append("avx512f")
        return " ".join(parts)
    except Exception as e:  # 任何调用失败都不阻断推理
        logger.debug("[cpu_info] Windows 指令集探测失败: %s", e)
        return ""


def _read_linux_flags() -> str:
    """Linux: 读取 /proc/cpuinfo 的 flags 行。"""
    try:  # pragma: no cover - Linux-only path
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.lower().startswith("flags"):
                    # 行格式: "flags		: fpu vme ... avx avx2 ..."
                    return line.split(":", 1)[-1].lower()
    except (OSError, IndexError):  # pragma: no cover - Linux-only
        pass
    return ""  # pragma: no cover - Linux-only


# ---------------------------------------------------------------------------
# oneDNN 安全性判定
# ---------------------------------------------------------------------------

#: paddle 版本黑名单：PIR 新执行器与 oneDNN 不兼容的版本区间。
#: 参考 Paddle #77340、PaddleOCR #17539：paddle 3.3.x 的 PIR new executor
#: 与 oneDNN 存在 ``ConvertPirAttribute2RuntimeAttribute`` 未实现的 bug，
#: ``predict`` 抛 NotImplementedError。这里保守地把整个 3.3.x 视为不安全，
#: 直到上游确认修复版本（暂无明确版本号，需运行时实测）。
_ONEDNN_UNSAFE_PADDLE_RANGES: list[tuple[str, str]] = [
    ("3.3.0", "3.3.99"),
]

#: 已通过项目真实 PP-OCR CPU 推理验证、允许默认启用 oneDNN 的版本区间。
#:
#: 该列表有意默认为空：截至 2026-07-20，上游 #77340/#17539 仍未给出
#: 明确修复版本。只有在项目的真实模型冒烟测试通过后才能加入新区间，不能
#: 因版本号“不在黑名单”就推断安全。高级用户仍可用 VIBEOCR_FORCE_ONEDNN=1
#: 做显式的不安全诊断。
_ONEDNN_VALIDATED_SAFE_PADDLE_RANGES: list[tuple[str, str]] = []


def can_safely_enable_onednn() -> tuple[bool, str]:
    """判定当前环境下能否安全启用 oneDNN（MKL-DNN）。

    综合三个维度：
    1. **指令集产品门槛**：项目仅在 AVX2+ CPU 上考虑启用。oneDNN 的 FP32
       实现本身可支持更早的 ISA，因此这是一条保守的性能/测试覆盖门槛，
       不是声称 oneDNN 普遍以 AVX2 为最低要求。
    2. **paddle 版本**：黑名单版本明确拒绝；只有命中真实推理验证过的安全
       版本区间才允许，未知版本、解析失败和未来未验证版本均 fail-closed。
    3. **用户强制覆盖**：``VIBEOCR_FORCE_ONEDNN=1`` 跳过上述门槛强制启用
       （仅供高级用户做不安全诊断），``=0`` 强制禁用。

    Returns:
        ``(safe, reason)``：``safe`` 为是否可启用，``reason`` 为决策依据
        （用于日志/状态展示）。
    """
    # 用户强制覆盖优先
    force = os.environ.get("VIBEOCR_FORCE_ONEDNN", "").strip()
    if force == "1":
        return True, "用户强制启用 oneDNN（VIBEOCR_FORCE_ONEDNN=1）"
    if force == "0":
        return False, "用户强制禁用 oneDNN（VIBEOCR_FORCE_ONEDNN=0）"

    # 1. 指令集门槛
    feats = detect_cpu_features()
    if not feats["avx2"]:
        return False, "CPU 不支持项目要求的 AVX2 oneDNN 验证门槛"

    # 2. paddle 版本必须可读且命中已验证安全范围
    paddle_ver = _get_paddle_version()
    if not paddle_ver:
        return False, "无法读取 paddle 版本，保守禁用 oneDNN"

    for lo, hi in _ONEDNN_UNSAFE_PADDLE_RANGES:
        if _version_in_range(paddle_ver, lo, hi):
            return (
                False,
                f"paddle {paddle_ver} 的 PIR 执行器与 oneDNN 不兼容"
                f"（参考 Paddle #77340）",
            )

    for lo, hi in _ONEDNN_VALIDATED_SAFE_PADDLE_RANGES:
        if _version_in_range(paddle_ver, lo, hi):
            feat_summary = ",".join(k for k, v in feats.items() if v)
            return True, f"CPU 支持 {feat_summary}，paddle {paddle_ver} 已通过验证"

    return False, f"paddle {paddle_ver} 尚未通过项目 oneDNN 真实推理验证"


def _get_paddle_version() -> str | None:
    """返回已安装 paddle 的版本号（短格式如 '3.3.1'），未安装返回 None。

    延迟 import 避免模块加载副作用；失败静默返回 None，调用方必须 fail-closed。
    """
    try:
        import paddle

        return getattr(paddle, "__version__", "") or None
    except Exception:  # 未安装或 import 失败，保守按"未知版本"处理
        return None


def _version_in_range(ver: str, lo: str, hi: str) -> bool:
    """判断 ``ver`` 是否落在 ``[lo, hi]``（按点分段数值比较）。"""
    try:
        v = _ver_tuple(ver)
        return _ver_tuple(lo) <= v <= _ver_tuple(hi)
    except (ValueError, TypeError):
        return False


def _ver_tuple(ver: str) -> tuple[int, ...]:
    """把 '3.3.1' / '3.3.1+cu126' 解析为 (3, 3, 1)。"""
    core = re.split(r"[+~]", ver, maxsplit=1)[0]
    return tuple(int(x) for x in core.split("."))
