"""PaddleX OCR 服务"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import threading
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from vibeocr.backend.core.pipelines import OCRPipeline
from vibeocr.backend.core.singleton_meta import SingletonMeta
from vibeocr.backend.models.ocr_options import OCROptions
from vibeocr.backend.models.ocr_result import OCRResult, TextBlock

# 重新导出以保持向后兼容性
__all__ = [
    "OCROptions",
    "OCRPipeline",
    "OCRPreset",
    "OCRResult",
    "OCRService",
]

# 跳过模型源网络检测，避免推理时的网络超时开销
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

# HTML 表格规整化/转换已迁移到 vibeocr.backend.utils.html_tables（UI 层展示逻辑）。
# 此处重导出以保持向后兼容。
# 注意：所有操作在同一线程中执行（CPU 模式）
# 工作线程设计已确保这一点
from vibeocr.backend.pipeline_status import (
    LOCAL_MARKABLE_PIPELINES,
    is_pipeline_ever_succeeded,
    mark_pipeline_success,
)
from vibeocr.backend.utils.html_tables import (
    _extract_table_html as _extract_table_html,
)
from vibeocr.backend.utils.html_tables import (
    _html_table_to_markdown as _html_table_to_markdown,
)
from vibeocr.backend.utils.html_tables import (
    normalize_table_html as normalize_table_html,
)

_logger = logging.getLogger(__name__)

# 类型检查时导入（不影响运行时）
if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np
    from PIL import Image


class OCRPreset(Enum):
    """OCR 预设模式（用于通用 OCR 管道的预处理配置）"""

    GENERAL = "general"  # 通用模式：适用于屏幕截图
    SCANNED = "scanned"  # 扫描件模式：适用于扫描文档/拍照

    @property
    def display_name(self) -> str:
        """获取显示名称"""
        names = {
            OCRPreset.GENERAL: "通用",
            OCRPreset.SCANNED: "扫描件",
        }
        return names.get(self, "通用")


class OCRService(metaclass=SingletonMeta):
    """OCR 识别服务 (使用 SingletonMeta 实现线程安全单例)

    双重角色（请勿误删或拆分）：
    1. **子进程模式（默认运行路径）**：本类在 worker 子进程内实例化
       （见 ``workers/ocr_worker.py``），持有真正的 PaddleOCR pipeline，
       由主进程经共享内存（RCBG 协议）调用其 recognize/recognize_batch。
    2. **主进程直连（仅调试逃生口）**：仅当 ``VIBEOCR_USE_SUBPROCESS=false``
       且 ``VIBEOCR_OCR_MODE=direct`` 时，本类才在主进程内直接加载模型运行，
       用于排查子进程开销/通信问题时定位。**生产环境不应走此路径**——
       主进程内加载 PaddleOCR 会阻塞 UI、占用 GPU 上下文。

    因此本类的所有 OCR 逻辑（含 recognize_batch 真批量）都是必需的，
    既是子进程的内核，也是调试路径的实现，二者共享同一份代码。
    """

    _pipelines: dict[str, Any] = {}  # 管道缓存：{pipeline_name: pipeline_instance}
    _lock = threading.Lock()
    _initialized = False
    _status_callback: Callable | None = None  # 状态回调函数
    _cache_manager: Any = None  # PipelineCacheManager 实例（懒加载）

    def is_ready(self) -> bool:
        """服务就绪（直连/worker 内：模型已加载即就绪）"""
        return self._initialized

    # 预加载相关状态
    _preload_progress_callback: Callable[[str, int, int], None] | None = (
        None  # (pipeline_name, current, total)
    )
    _preloaded_pipelines: set[str] = set()  # 已预加载的管道名称
    _is_preloading = False  # 是否正在预加载
    _preload_lock = (
        threading.Lock()
    )  # 预加载专用锁，保护 _preloaded_pipelines 和 _is_preloading

    @classmethod
    def set_status_callback(cls, callback: Callable | None) -> None:
        """设置状态回调函数

        Args:
            callback: 回调函数，接收 (stage, message) 参数
                     例如: ("模型下载", "正在下载 OCR 模型...")
        """
        cls._status_callback = callback

    @classmethod
    def _notify_status(cls, stage: str, message: str) -> None:
        """通知状态变化"""
        if cls._status_callback:
            try:
                cls._status_callback(stage, message)
            except Exception:
                pass  # 忽略回调错误

    def __init__(self):
        """初始化 OCR 服务

        使用 SingletonMeta 确保单例，_initialized 标志防止重复初始化。
        """
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self._initialized = True

    @property
    def cache_manager(self) -> Any:
        """PipelineCacheManager 实例（懒加载，避免循环导入）。"""
        if self._cache_manager is None:
            from vibeocr.backend.services.pipeline_cache_manager import (
                PipelineCacheManager,
            )

            # Backend 不读取前端配置。并存上限与 TTL 由 Supervisor RPC 显式
            # 注入；未注入时按硬件自动分档。
            self._cache_manager = PipelineCacheManager(self, ttls={}, max_heavy=None)
        return self._cache_manager

    @classmethod
    def _reset(cls) -> None:
        """重置服务状态

        供 SingletonMeta.reset_instance() 调用，用于测试清理。
        """
        with cls._lock:
            cls._pipelines = {}
            cls._initialized = False
            cls._status_callback = None
            cls._preload_progress_callback = None
            cls._cache_manager = None  # 重置 cache_manager（类属性）
            # 同步清理实例属性（property 懒加载会设实例属性，遮蔽类属性）
            from vibeocr.backend.core.singleton_meta import SingletonMeta

            instance = SingletonMeta._instances.get(cls)
            if instance is not None:
                instance.__dict__.pop("_cache_manager", None)
            cls._preloaded_pipelines = set()
            cls._is_preloading = False
            # oneDNN 判定缓存必须随重置清空，否则测试间会泄漏上一个探测结果，
            # 导致 _decide_enable_mkldnn 在 monkeypatch 后仍返回旧值。
            cls._onednn_safe_cache = None
            cls._onednn_runtime_disabled = False

    @classmethod
    def set_preload_progress_callback(
        cls, callback: Callable[[str, int, int], None] | None
    ) -> None:
        """设置预加载进度回调函数

        Args:
            callback: 回调函数，接收 (pipeline_name, current, total) 参数
                     例如: ("OCR", 1, 3) 表示正在加载第 1 个管道，共 3 个
        """
        cls._preload_progress_callback = callback

    @classmethod
    def is_pipeline_preloaded(cls, pipeline: OCRPipeline) -> bool:
        """检查指定管道是否已预加载

        Args:
            pipeline: 管道类型

        Returns:
            是否已预加载
        """
        with cls._preload_lock:
            return pipeline.value in cls._preloaded_pipelines

    @classmethod
    def get_preloaded_pipelines(cls) -> list[str]:
        """获取已预加载的管道名称列表"""
        with cls._preload_lock:
            return list(cls._preloaded_pipelines)

    @classmethod
    def preload_pipeline(cls, pipeline: OCRPipeline) -> bool:
        """预加载单个管道（同步）

        Args:
            pipeline: 要预加载的管道

        Returns:
            是否成功加载
        """
        pipeline_name = pipeline.value

        # 检查是否已缓存（使用主锁保护 _pipelines）
        with cls._lock:
            if pipeline_name in cls._pipelines:
                with cls._preload_lock:
                    cls._preloaded_pipelines.add(pipeline_name)
                return True

        try:
            _logger.debug(f"[预加载] 开始加载管道: {pipeline.display_name}")
            instance = cls()
            instance.get_pipeline(pipeline)

            # 更新预加载状态（使用预加载锁）
            with cls._preload_lock:
                cls._preloaded_pipelines.add(pipeline_name)

            _logger.debug(f"[预加载] 管道加载完成: {pipeline.display_name}")
            return True
        except Exception as e:
            _logger.error(f"[预加载] 管道加载失败 {pipeline.display_name}: {e}")
            return False

    @classmethod
    def preload_pipelines_sequential(
        cls,
        pipelines: list[OCRPipeline],
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> dict[str, bool]:
        """顺序预加载多个管道

        Args:
            pipelines: 要预加载的管道列表
            progress_callback: 进度回调 (pipeline_name, current, total)

        Returns:
            各管道加载结果 {pipeline_name: success}
        """
        # 检查是否有预加载任务在进行中
        with cls._preload_lock:
            if cls._is_preloading:
                _logger.warning("[预加载] 已有预加载任务在进行中")
                return {}
            cls._is_preloading = True

        results = {}
        total = len(pipelines)

        try:
            for i, pipeline in enumerate(pipelines, 1):
                pipeline_name = pipeline.value
                display_name = pipeline.display_name

                # 通知进度
                if progress_callback:
                    progress_callback(pipeline_name, i, total)
                if cls._preload_progress_callback:
                    cls._preload_progress_callback(pipeline_name, i, total)

                _logger.debug(f"[预加载] ({i}/{total}) 加载 {display_name}...")
                results[pipeline_name] = cls.preload_pipeline(pipeline)

            # 汇总结果
            success_count = sum(1 for v in results.values() if v)
            _logger.info(f"[预加载] 完成: {success_count}/{total} 个管道加载成功")

        finally:
            # 确保重置状态
            with cls._preload_lock:
                cls._is_preloading = False

        return results

    @classmethod
    def warmup_with_test_image(
        cls,
        pipeline: OCRPipeline | None = None,
        progress_callback: Callable[[str, int], None] | None = None,
    ) -> bool:
        """使用测试图片预热 OCR 服务

        通过执行一次虚拟识别来触发模型加载。

        Args:
            pipeline: 要预热的管道，None 表示使用默认 OCR 管道
            progress_callback: 进度回调 (stage, percent)

        Returns:
            预热是否成功
        """
        from vibeocr.backend.utils.warmup_utils import get_warmup_image

        pipeline = pipeline or OCRPipeline.OCR
        pipeline_name = pipeline.value

        try:
            _logger.debug(f"[预热] 开始使用测试图片预热管道: {pipeline_name}")
            if progress_callback:
                progress_callback("准备测试图片", 10)

            # 获取测试图片
            test_image = get_warmup_image()
            if test_image is None:
                _logger.error("[预热] 无法创建预热测试图片，预热中止")
                return False
            _logger.debug(f"[预热] 测试图片大小: {len(test_image)} 字节")

            if progress_callback:
                progress_callback("执行虚拟识别", 50)

            # 创建选项并执行识别
            options = OCROptions(pipeline=pipeline)
            instance = cls()
            instance.recognize(test_image, options)

            if progress_callback:
                progress_callback("预热完成", 100)

            _logger.debug(f"[预热] 管道 {pipeline_name} 预热成功")
            return True

        except Exception as e:
            _logger.error(f"[预热] 管道 {pipeline_name} 预热失败: {e}")
            return False

    @classmethod
    def warmup_pipelines(
        cls,
        pipelines: list[OCRPipeline],
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> dict[str, bool]:
        """使用测试图片预热多个管道

        Args:
            pipelines: 要预热的管道列表
            progress_callback: 进度回调 (pipeline_name, current, total)

        Returns:
            各管道预热结果 {pipeline_name: success}
        """
        results = {}
        total = len(pipelines)

        _logger.debug(f"[预热] 开始批量预热 {total} 个管道")

        for i, pipeline in enumerate(pipelines, 1):
            pipeline_name = pipeline.value

            def make_progress(
                stage: str, percent: int, idx=i, name=pipeline_name, total_count=total
            ):
                if progress_callback:
                    overall_percent = int(((idx - 1) * 100 + percent) / total_count)
                    progress_callback(name, idx, overall_percent)

            _logger.debug(f"[预热] ({i}/{total}) 预热 {pipeline.display_name}...")
            results[pipeline_name] = cls.warmup_with_test_image(pipeline, make_progress)

        success_count = sum(1 for v in results.values() if v)
        _logger.info(f"[预热] 完成: {success_count}/{total} 个管道预热成功")

        return results

    @classmethod
    def preload_in_background(
        cls,
        pipelines: list[OCRPipeline],
        on_complete: Callable[[dict[str, bool]], None] | None = None,
    ) -> threading.Thread:
        """在后台线程中预加载管道（非阻塞）

        Args:
            pipelines: 要预加载的管道列表
            on_complete: 完成回调，接收加载结果字典

        Returns:
            后台线程对象
        """

        def _preload_task():
            try:
                results = cls.preload_pipelines_sequential(pipelines)

                if on_complete:
                    on_complete(results)
            except Exception as e:
                _logger.error(f"[预加载] 后台预加载任务失败: {e}")
                if on_complete:
                    on_complete({})

        thread = threading.Thread(
            target=_preload_task, daemon=True, name="PipelinePreload"
        )
        thread.start()
        return thread

    _cuda_dll_registered = False

    @classmethod
    def _setup_cuda_dll_path(cls) -> None:
        """将 CUDA 运行时 DLL 目录加入 PATH，使 PaddlePaddle 能找到 cuBLAS/cuDNN 等

        扫描两类来源：
        1. ``nvidia/*`` 包（cu13/cudnn 等，DLL 多在 bin/ 或 bin/<arch>/、lib/<arch>/）
        2. ``torch/lib`` —— PyTorch wheel 自带完整的 CUDA 12 + cuDNN 9 运行时，
           是 paddlepaddle-gpu（CUDA 12 构建）所需 ``cublas64_12.dll`` 等的
           可靠来源。在未安装系统级 CUDA Toolkit 的机器上，这是让 paddle GPU
           可用的关键（否则 ``cublas64_12.dll`` 找不到，error 126 回退 CPU）。
        """
        if cls._cuda_dll_registered:
            return
        import sys

        if sys.platform != "win32":
            cls._cuda_dll_registered = True
            return
        site_packages = next(p for p in sys.path if "site-packages" in p)
        existing = os.environ.get("PATH", "")
        candidate_dirs: list[Path] = []

        # 1) nvidia/* 包
        nvidia_base = Path(site_packages) / "nvidia"
        if nvidia_base.is_dir():
            for entry in os.scandir(nvidia_base):
                if not entry.is_dir():
                    continue
                # 扫描 bin/ 和 lib/ 目录（不同 nvidia 包的 DLL 位置不同）
                for subfolder in ("bin", "lib"):
                    sub_dir = Path(entry.path) / subfolder
                    if sub_dir.is_dir():
                        candidate_dirs.append(sub_dir)
                        # 新版 nvidia 包 (cu13 等) 把 DLL 放在 <subfolder>/<arch>/ 子目录
                        for sub in os.scandir(sub_dir):
                            arch_dir = Path(sub_dir) / sub.name
                            if sub.is_dir():
                                candidate_dirs.append(arch_dir)

        # 2) torch/lib（CUDA 12 + cuDNN 9 全套）
        # 仅当 paddle 不自带 CUDA runtime 时才加 torch/lib 作为 fallback。
        # paddle 3.x+ 自带完整 CUDA runtime（cu129 等），此时把 torch/lib（cu126）
        # 加到 PATH 会导致版本冲突：paddle 加载的 cu129 符号与 torch/lib 的 cu126
        # DLL 不匹配，torch 的 shm.dll 报 WinError 127（实测回归）。
        paddle_has_cuda = False
        try:
            import paddle.version  # type: ignore[import-untyped]

            paddle_has_cuda = bool(getattr(paddle.version, "cuda", lambda: None)())
        except Exception:
            pass

        if not paddle_has_cuda:
            torch_lib = Path(site_packages) / "torch" / "lib"
            if torch_lib.is_dir():
                candidate_dirs.append(torch_lib)

        for d in candidate_dirs:
            s = str(d)
            if s not in existing:
                existing = s + ";" + existing
        os.environ["PATH"] = existing
        cls._cuda_dll_registered = True

    @staticmethod
    def _get_device() -> str:
        """根据环境变量和 GPU 可用性检测推理设备"""
        if os.environ.get("VIBEOCR_USE_GPU", "").lower() != "true":
            _logger.info("[推理设备] CPU（未启用 GPU，VIBEOCR_USE_GPU != true）")
            return "cpu"
        try:
            import paddle
            import paddle.device as paddle_device

            if paddle_device.cuda.device_count() > 0:
                # 驱动可检测到 GPU，但 CUDA 运行时库（cuBLAS 等）可能未安装，
                # 执行一次矩阵乘法验证 cuBLAS 可用性。
                paddle.device.set_device("gpu")
                a = paddle.randn([4, 4])
                _ = paddle.matmul(a, a).numpy()
                # paddle 导入完成后注册 DLL 目录，供 predict() 内部的 ctypes 使用
                # （必须在 paddle 导入后调用，否则会触发 paddle/libs/nvidia 路径错误）
                OCRService._register_dll_directories()
                OCRService._log_gpu_summary()
                return "gpu"
        except Exception as e:
            _logger.warning("[GPU] GPU 可用性验证失败: %s，回退到 CPU", e)
            _logger.info("[推理设备] CPU（GPU 验证失败，已回退）")
            return "cpu"
        _logger.warning("[GPU] VIBEOCR_USE_GPU=true 但未检测到可用 GPU，回退到 CPU")
        _logger.info("[推理设备] CPU（未检测到可用 GPU，已回退）")
        return "cpu"

    # oneDNN 安全性判定结果缓存（进程级，避免每次创建管道重复探测指令集）
    _onednn_safe_cache: bool | None = None
    # 真实推理一旦命中已知 PIR/oneDNN 回归，本进程永久禁用，直到服务重置。
    _onednn_runtime_disabled = False

    @classmethod
    def _decide_enable_mkldnn(cls, device: str) -> bool:
        """决定是否向 PaddleOCR 构造函数传入 ``enable_mkldnn=True``。

        **这是 oneDNN 是否启用的唯一真相**：返回值直接成为构造函数 kwarg，
        而 PaddleOCR 推理路径只读这个 kwarg（最终落到
        ``paddle.inference.Config.enable_mkldnn()``）。进程级 FLAGS
        （FLAGS_use_mkldnn / FLAGS_enable_onednn_backend）对推理路径【不生效】
        ——paddleocr/paddlex 零处读取它们，那些 FLAGS 仅影响 eager 路径。

        - GPU 设备：不传（PaddleOCR 默认），返回 False。
        - CPU 设备：调用 ``cpu_info.can_safely_enable_onednn`` 综合判定
          （指令集产品门槛 + 已验证版本范围 + 用户强制覆盖）。结果缓存。

        历史背景：paddle 3.3.x 的 PIR 新执行器与 oneDNN 不兼容
        （ConvertPirAttribute2RuntimeAttribute 未实现，predict 抛
        NotImplementedError），参考 PaddleOCR #17539、Paddle #77340。
        故默认对受影响、未知和未验证版本拒绝；只有真实模型验证过的版本范围
        才默认启用。若运行时仍命中该已知异常，会锁定本进程禁用并重建管道。
        """
        if device != "cpu":
            return False
        if cls._onednn_runtime_disabled:
            return False
        if cls._onednn_safe_cache is None:
            try:
                from vibeocr.backend.utils.cpu_info import can_safely_enable_onednn

                safe, reason = can_safely_enable_onednn()
                cls._onednn_safe_cache = safe
                _logger.info("[oneDNN] %s: %s", "启用" if safe else "禁用", reason)
            except Exception as e:
                # 探测失败保守禁用（与历史行为一致）
                cls._onednn_safe_cache = False
                _logger.warning("[oneDNN] 安全性探测失败，保守禁用: %s", e)
        return bool(cls._onednn_safe_cache)

    @staticmethod
    def _is_known_onednn_pir_failure(exc: BaseException) -> bool:
        """判断异常链是否命中 Paddle 3.3 PIR/oneDNN 已知回归。"""
        seen: set[int] = set()
        current: BaseException | None = exc
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            message = str(current)
            if (
                "ConvertPirAttribute2RuntimeAttribute" in message
                and "onednn_instruction" in message.lower()
            ):
                return True
            current = current.__cause__ or current.__context__
        return False

    def _disable_onednn_and_release_pipeline(self, pipeline_name: str) -> None:
        """锁定进程级禁用 oneDNN，并只释放发生兼容错误的管道。"""
        cls = type(self)
        with cls._lock:
            cls._onednn_runtime_disabled = True
            cls._onednn_safe_cache = False
            released = self.cache_manager.release_one(pipeline_name)
            with cls._preload_lock:
                cls._preloaded_pipelines.discard(pipeline_name)

        _logger.warning(
            "[oneDNN] 管道 %s 命中已知 PIR/oneDNN 兼容错误；"
            "已锁定本进程禁用并%s管道，当前请求仅重试一次",
            pipeline_name,
            "释放" if released else "清理",
        )
        self._notify_status(
            "兼容性回退",
            f"{pipeline_name} 的 oneDNN 路径不兼容，已切换到标准 CPU 推理",
        )

    @staticmethod
    def _log_gpu_summary() -> None:
        """在确定使用 GPU 推理后，输出一条设备摘要 INFO 日志。

        包含 GPU 名称、显存（总/空闲）和 PDF 批量上限，便于排查“日志写 CPU
        实际跑 GPU”这类不一致问题，也直观反映 4090 等大显存卡的批量能力。
        """
        gpu_name = "未知"
        try:
            import paddle.device as paddle_device

            gpu_name = paddle_device.cuda.get_device_name(0)
        except Exception:
            pass

        total_mb = free_mb = 0
        try:
            from vibeocr.backend.utils.gpu_memory_monitor import GPUMemoryMonitor

            info = GPUMemoryMonitor().get_status()
            if info.available:
                total_mb, free_mb = info.total, info.free
        except Exception:
            pass

        from vibeocr.backend.utils.gpu_memory_monitor import GPU_BATCH_CAP

        _logger.info(
            "[推理设备] GPU=%s, 显存=%dMB (空闲 %dMB), PDF 批量上限=%d",
            gpu_name,
            total_mb,
            free_mb,
            GPU_BATCH_CAP,
        )

    @classmethod
    def _register_dll_directories(cls) -> None:
        """通过 os.add_dll_directory() 和 PATH 注册 CUDA DLL 目录

        Python 3.8+ Windows 上 ctypes.CDLL 不再搜索 os.environ["PATH"]，
        必须通过 os.add_dll_directory() 注册。同时更新 PATH 环境变量，
        确保推理引擎（Paddle Inference）也能找到 CUDA DLL。

        覆盖来源与 :meth:`_setup_cuda_dll_path` 一致：``nvidia/*`` 包，并且
        仅在 Paddle 未自带 CUDA runtime 时回退到 ``torch/lib``。

        此方法必须在 PaddlePaddle 导入完成后调用，否则会触发 PaddlePaddle
        内部的路径错误。
        """
        if not hasattr(os, "add_dll_directory"):
            return
        import sys

        site_packages = next((p for p in sys.path if "site-packages" in p), None)
        if not site_packages:
            return

        def _register(d: Path) -> None:
            with contextlib.suppress(OSError):
                os.add_dll_directory(str(d))
            os.environ["PATH"] = str(d) + ";" + os.environ.get("PATH", "")

        # 1) nvidia/* 包（bin/, bin/<arch>/, lib/<arch>/）
        nvidia_base = Path(site_packages) / "nvidia"
        if nvidia_base.is_dir():
            for entry in os.scandir(nvidia_base):
                if not entry.is_dir():
                    continue
                for subfolder in ("bin", "lib"):
                    sub_dir = Path(entry.path) / subfolder
                    if sub_dir.is_dir():
                        _register(sub_dir)
                        for sub in os.scandir(sub_dir):
                            arch_dir = Path(sub_dir) / sub.name
                            if sub.is_dir():
                                _register(arch_dir)

        # 2) torch/lib fallback。Paddle 自带 CUDA 时禁止混入不同版本的
        # Torch runtime，否则后续加载 torch/shm.dll 可能触发 WinError 127。
        paddle_has_cuda = False
        try:
            import paddle.version  # type: ignore[import-untyped]

            paddle_has_cuda = bool(getattr(paddle.version, "cuda", lambda: None)())
        except Exception:
            pass
        if not paddle_has_cuda:
            torch_lib = Path(site_packages) / "torch" / "lib"
            if torch_lib.is_dir():
                _register(torch_lib)

    @staticmethod
    def _get_project_root():
        from vibeocr.backend.env_manager import get_project_root

        return get_project_root()

    def _create_pipeline(self, pipeline: OCRPipeline) -> Any:
        """创建指定管道（使用 PaddleOCR 3.x API）"""
        _logger.debug("[_create_pipeline] %s: 开始...", pipeline.value)
        device = self._get_device()
        display_name = pipeline.display_name

        models_cached = is_pipeline_ever_succeeded(
            pipeline.value, self._get_project_root()
        )
        if not models_cached:
            self._notify_status(
                "模型初始化",
                f"正在初始化 {display_name} 管道（首次使用需要下载模型）...",
            )

        # CPU 设备的 mkldnn 启用与否由 _decide_enable_mkldnn 综合判定
        # （指令集产品门槛 + 已验证版本范围），而非硬编码 False。
        enable_mkldnn = self._decide_enable_mkldnn(device)
        kwargs = {"enable_mkldnn": enable_mkldnn} if device == "cpu" else {}

        if pipeline == OCRPipeline.OCR:
            from paddleocr import PaddleOCR

            instance = PaddleOCR(device=device, **kwargs)
        elif pipeline == OCRPipeline.PP_STRUCTURE_V3:
            from paddleocr import PPStructureV3

            instance = PPStructureV3(device=device, **kwargs)
        elif pipeline == OCRPipeline.PADDLEOCR_VL:
            from paddleocr import PaddleOCRVL

            instance = PaddleOCRVL(device=device, **kwargs)  # type: ignore[call-arg]
        else:
            msg = f"不支持的管道类型: {pipeline}"
            raise ValueError(msg)

        _logger.debug("管道 %s 初始化于设备: %s", pipeline.value, device)

        if not models_cached:
            self._notify_status("模型初始化", f"{display_name} 管道初始化完成")

        return instance

    def get_pipeline(self, pipeline: OCRPipeline) -> Any:
        """延迟加载指定管道 (线程安全)

        向后兼容方法，内部委托给 get_or_create_pipeline。
        """
        return self.get_or_create_pipeline(pipeline.value)

    def get_or_create_pipeline(self, pipeline_name: str) -> Any:
        """根据管道名获取或创建管道实例

        先尝试从注册表获取 PipelineSpec 并使用其 create_pipeline 工厂，
        回退到旧式 _create_pipeline 以保持向后兼容。

        Args:
            pipeline_name: 管道名称字符串 (e.g. "OCR", "PP-StructureV3")

        Returns:
            管道实例
        """
        if pipeline_name not in self._pipelines:
            with self._lock:
                if pipeline_name not in self._pipelines:  # 双重检查
                    self._setup_cuda_dll_path()
                    self.cache_manager.prepare_load(pipeline_name)
                    _logger.debug(
                        "[get_or_create_pipeline] 创建管道 %s，已加载管道: %s",
                        pipeline_name,
                        list(self._pipelines.keys()),
                    )
                    from vibeocr.backend.core.pipelines import get_registry

                    registry = get_registry()
                    if registry.has(pipeline_name):
                        spec = registry.get(pipeline_name)
                        device = self._get_device()
                        # CPU 设备的 mkldnn 启用与否由 _decide_enable_mkldnn
                        # 综合判定（指令集产品门槛 + 已验证版本范围）。
                        enable_mkldnn = self._decide_enable_mkldnn(device)
                        kwargs = (
                            {"enable_mkldnn": enable_mkldnn} if device == "cpu" else {}
                        )
                        self._pipelines[pipeline_name] = spec.create_pipeline(
                            device, **kwargs
                        )
                    else:
                        # 回退到旧式创建（通过 OCRPipeline 枚举）
                        try:
                            pipeline_enum = OCRPipeline(pipeline_name)
                        except ValueError:
                            msg = f"不支持的管道类型: {pipeline_name}"
                            raise ValueError(msg) from None
                        self._pipelines[pipeline_name] = self._create_pipeline(
                            pipeline_enum
                        )
                    _logger.debug(
                        "[get_or_create_pipeline] 管道 %s 创建完成", pipeline_name
                    )
        # 记录使用时间；容量已在加载前由 prepare_load 原子腾出。
        try:
            self.cache_manager.touch(pipeline_name)
        except Exception as e:
            _logger.debug("[get_or_create_pipeline] cache_manager 操作失败: %s", e)
        return self._pipelines[pipeline_name]

    @classmethod
    def release_pipelines(cls, heavy_only: bool = True) -> list[str]:
        """释放管道缓存（直连模式：直接调 cache_manager.release）。

        Args:
            heavy_only: True 只释放重管道，False 释放全部（含 OCR）。

        Returns:
            被释放的管道名列表。
        """
        return cls().cache_manager.release(heavy_only=heavy_only)

    @classmethod
    def set_pipeline_ttls(cls, pipeline_ttls: dict[str, int]) -> bool:
        """设置每管道 TTL 闲置回收时间（直连模式）。

        Args:
            pipeline_ttls: 每管道 TTL 字典，0=持久。

        Returns:
            是否设置成功。
        """
        cls().cache_manager.ttls = pipeline_ttls
        return True

    @property
    def pipeline(self) -> Any:
        """延迟加载默认 OCR 流水线 (向后兼容)"""
        return self.get_pipeline(OCRPipeline.OCR)

    def recognize(
        self,
        image: Image.Image | np.ndarray | str | bytes,
        options: OCROptions | None = None,
    ) -> OCRResult:
        """
        对图像执行 OCR 识别

        Args:
            image: PIL Image, numpy 数组, 图像路径, 或图像字节数据
            options: OCR 识别选项

        Returns:
            OCRResult 对象，包含识别结果和置信度信息
        """
        image = self._to_ndarray(image)
        # _to_ndarray 对 str 路径输入返回 str，但 recognize_batch 只收 ndarray。
        # 实际调用方均传 bytes/PIL/ndarray（str 路径走 paddlex 内部加载），
        # 此处 cast 与 recognize_batch 的 ndarray 契约对齐。
        return self.recognize_batch([cast("np.ndarray", image)], options)[0]

    @staticmethod
    def _to_ndarray(
        image: Image.Image | np.ndarray | str | bytes,
    ) -> np.ndarray | str:
        """统一输入为 PaddleX 可接受的形态（numpy 数组或路径字符串）。

        bytes/PIL 输入转换为 RGB numpy 数组；ndarray/str 原样返回。

        Args:
            image: PIL Image, numpy 数组, 图像路径, 或图像字节数据。

        Returns:
            numpy 数组（bytes/PIL 输入）或原始字符串路径。
        """
        if isinstance(image, bytes):
            import io

            import numpy as np
            from PIL import Image as PILImage

            _logger.debug(
                f"[recognize] 输入是 bytes ({len(image)} 字节)，转换为 numpy.ndarray"
            )
            pil_image = PILImage.open(io.BytesIO(image))
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")  # type: ignore[assignment]
            return np.array(pil_image)
        if hasattr(image, "convert"):
            # PIL Image（非 ndarray）
            import numpy as np

            if image.mode != "RGB":
                image = image.convert("RGB")  # type: ignore[assignment]
            return np.array(image)
        # 到此 bytes/PIL 分支已 return，image 只剩 ndarray|str（hasattr 无法静态收窄）
        return image  # type: ignore[return-value]  # ndarray|str，符合签名

    def recognize_batch(
        self,
        images: list[np.ndarray],
        options: OCROptions | None = None,
    ) -> list[OCRResult]:
        """对一组图像批量执行 OCR 识别（单次 predict 调用，利用 PaddleOCR 批处理）。

        相比逐张调用 recognize()，本方法将所有图像一次性送入 PaddleOCR 的
        predict(list)，由其内部的 ImageBatchSampler 按 batch_size 分批，
        避免每张图重复的管道开销，显著提升 PDF 等多页场景的吞吐。

        输入图像需为 numpy 数组（RGB，与单次识别路径一致）。结果顺序与输入一致。

        Args:
            images: 输入图像列表（numpy 数组）。
            options: OCR 识别选项（所有图像共享同一组选项）。

        Returns:
            OCRResult 列表，顺序与 images 一致。
        """
        actual_options = options if options is not None else OCROptions()

        # 统一获取管道名称（处理枚举和字符串两种类型）
        pipeline_name = actual_options.pipeline.value
        _logger.debug(
            f"[recognize_batch] 开始批量识别 {len(images)} 张，管道: {pipeline_name}"
        )

        if not images:
            return []

        def _dispatch_once() -> list[OCRResult]:
            """执行一次注册表分发；兼容回退由外层严格控制为至多一次。"""
            from vibeocr.backend.core.pipelines import get_registry

            registry = get_registry()
            if registry.has(pipeline_name) and (
                registry.get(pipeline_name).recognize_batch is not None
            ):
                spec = registry.get(pipeline_name)
                return spec.recognize_batch(  # type: ignore[misc,no-any-return]
                    self, images, actual_options
                )

            _logger.debug(
                "[recognize_batch] 管道 %s 未注册批量接口，回退逐张识别",
                pipeline_name,
            )
            spec = registry.get(pipeline_name)
            return [spec.recognize(self, img, actual_options) for img in images]

        # 根据管道类型分发
        try:
            try:
                results = _dispatch_once()
            except Exception as first_error:
                # 只有确实以 oneDNN 创建的 CPU 管道、且异常精确命中已知
                # PIR 转换回归时才回退；其他异常保持原语义，不掩盖业务错误。
                if not (
                    type(self)._onednn_safe_cache is True
                    and self._is_known_onednn_pir_failure(first_error)
                ):
                    raise
                self._disable_onednn_and_release_pipeline(pipeline_name)
                # 不再包一层同类 catch：第二次失败必须原样抛出，禁止循环重试。
                results = _dispatch_once()

            # Normalize each result's bbox from pixel coords to [0-1000]
            for img, result in zip(images, results, strict=False):
                self._normalize_result_bbox(result, img)

            # 标记管道识别成功：覆盖全部本地管道（见 LOCAL_MARKABLE_PIPELINES），
            # 遗漏会导致 is_pipeline_ever_succeeded 永远 False，触发 QWebEngineView
            # 冷启动卡顿（见该常量的文档注释）。
            pipeline_val = pipeline_name
            if pipeline_val in LOCAL_MARKABLE_PIPELINES:
                try:
                    mark_pipeline_success(pipeline_val, self._get_project_root())
                except Exception:
                    pass

            _logger.debug(f"[recognize_batch] 完成，返回 {len(results)} 个结果")
            return results
        except Exception as e:
            _logger.error(f"[recognize_batch] 识别过程中发生异常: {e}", exc_info=True)
            raise

    @staticmethod
    def _normalize_result_bbox(result: OCRResult, image: Any) -> None:
        """将 OCRResult 中像素坐标 bbox 归一化到 [0, 1000]（原地修改）。

        bbox 坐标在预处理后图像空间中。优先使用 result.preproc_img_w/h；
        若预处理未提供尺寸，则回退到输入图像 shape，并在 90°/270° 旋转时
        互换宽高（旋转后图像宽高与原图互换）。

        Args:
            result: OCRResult，其 text_blocks/content_list 的 bbox 会被原地归一化。
            image: 对应的输入图像（用于回退取尺寸）。
        """
        img_w = img_h = 0
        if result.preproc_img_w > 0 and result.preproc_img_h > 0:
            img_w = result.preproc_img_w
            img_h = result.preproc_img_h
        elif hasattr(image, "shape"):
            _shape = image.shape
            if isinstance(_shape, tuple) and len(_shape) >= 2:
                img_h, img_w = _shape[:2]
        elif hasattr(image, "size"):
            _sz = image.size
            if isinstance(_sz, tuple):
                img_w, img_h = _sz
        # 预处理旋转 90°/270° 时宽高互换（仅当无预处理图像时需要）
        if result.preproc_img_w == 0 and result.preproc_angle in (90, 270):
            img_w, img_h = img_h, img_w
        if img_w <= 0 or img_h <= 0:
            return
        for block in result.text_blocks:
            if block.bbox:
                x0, y0, x1, y1 = block.bbox
                block.bbox = (
                    x0 / img_w * 1000,
                    y0 / img_h * 1000,
                    x1 / img_w * 1000,
                    y1 / img_h * 1000,
                )
            # 多边形与 bbox 同源、同像素空间，用同一 img_w/h 归一化（保持一致）。
            # _parse_single_result 直接透传 PaddleOCR 像素坐标，这里统一归一化。
            if block.polygon:
                max_v = max(block.polygon)
                if max_v > 1001:
                    block.polygon = tuple(
                        v / (img_w if i % 2 == 0 else img_h) * 1000
                        for i, v in enumerate(block.polygon)
                    )
        for cl_block in result.content_list:
            bbox = cl_block.get("bbox")
            if bbox and len(bbox) >= 4:
                cl_block["bbox"] = [
                    bbox[0] / img_w * 1000,
                    bbox[1] / img_h * 1000,
                    bbox[2] / img_w * 1000,
                    bbox[3] / img_h * 1000,
                ]
        result.image_width = img_w
        result.image_height = img_h

    def _recognize_ocr(
        self,
        image: Image.Image | np.ndarray | str,
        options: OCROptions,
    ) -> OCRResult:
        """通用 OCR 识别"""
        _logger.debug("[_recognize_ocr] 获取 OCR 管道...")
        pipeline = self.get_pipeline(OCRPipeline.OCR)
        _logger.debug("[_recognize_ocr] 执行 predict...")
        try:
            output = pipeline.predict(
                input=image,
                use_doc_orientation_classify=options.use_doc_orientation_classify,
                use_doc_unwarping=options.use_doc_unwarping,
                use_textline_orientation=options.use_textline_orientation,
            )
            _logger.debug(f"[_recognize_ocr] predict 返回，类型: {type(output)}")
        except Exception as e:
            _logger.error(f"[_recognize_ocr] predict 调用失败: {e}", exc_info=True)
            raise

        _logger.debug("[_recognize_ocr] 开始处理输出...")
        result = self._process_ocr_output_safe(output)
        _logger.debug(f"[_recognize_ocr] 结果处理完成: {len(result.raw_text)} 字符")
        return result

    def _recognize_structure(
        self,
        image: Image.Image | np.ndarray | str,
        options: OCROptions,
    ) -> OCRResult:
        """通过管道注册表委托 PP-StructureV3 文档解析。"""
        from vibeocr.backend.core.pipelines import get_registry

        spec = get_registry().get("PP-StructureV3")
        provider_options = spec.options_class.from_dict(options.to_dict())
        return spec.recognize(self, image, provider_options)

    def _recognize_paddlocr_vl(
        self,
        image: Image.Image | np.ndarray | str,
        options: OCROptions,
    ) -> OCRResult:
        """通过管道注册表委托 PaddleOCR-VL 文档解析。"""
        from vibeocr.backend.core.pipelines import get_registry

        spec = get_registry().get("PaddleOCR-VL")
        provider_options = spec.options_class.from_dict(options.to_dict())
        return spec.recognize(self, image, provider_options)

    @staticmethod
    def _extract_bbox(
        rec_boxes, index: int
    ) -> tuple[float, float, float, float] | None:
        """从 rec_boxes 提取第 index 个文本框的 bbox [x0, y0, x1, y1]

        支持格式:
        - (N, 4): [x0, y0, x1, y1] 轴对齐矩形
        - (N, 4, 2): [[x0,y0], [x1,y1], [x2,y2], [x3,y3]] 四点多边形
        - (N, 2, 2): [[x0,y0], [x1,y1]] 两点矩形
        """
        try:
            box = rec_boxes[index]
            if hasattr(box, "tolist"):
                box = box.tolist()
            if len(box) == 4:
                # (N, 4) 或 (N, 4, 2)
                if isinstance(box[0], (int, float)):
                    return (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
                # (N, 4, 2) 多边形格式: 取外接矩形
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                return (min(xs), min(ys), max(xs), max(ys))
            if len(box) == 2 and len(box[0]) == 2 and len(box[1]) == 2:
                return (
                    float(box[0][0]),
                    float(box[0][1]),
                    float(box[1][0]),
                    float(box[1][1]),
                )
        except (IndexError, TypeError, ValueError):
            pass
        return None

    @staticmethod
    def _consume_generator_safely(output) -> list:
        """安全地消费 generator（禁用 GC 避免 CUDA 内存管理冲突）"""
        import gc

        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            return list(output)
        except Exception as e:
            _logger.error(f"[安全消费] 消费 generator 时出错: {e}", exc_info=True)
            return []
        finally:
            if gc_was_enabled:
                gc.enable()

    def _process_ocr_output_safe(self, output) -> OCRResult:
        """从 OCR 输出中提取结果

        先将 generator 完全消费为 list（禁用 GC），然后提取数据。
        """
        _logger.debug("[_process_ocr_output_safe] 开始提取结果...")
        text_with_scores: list[tuple[str, float]] = []
        text_blocks: list[TextBlock] = []

        output_list = self._consume_generator_safely(output)

        # 提取预处理信息：旋转角度和实际预处理后图像
        # res.img['preprocessed_img'] 是拼接可视化，不用；
        # 实际预处理图在 doc_preprocessor_res['output_img']（numpy RGB，
        # PaddleOCR 3.7 实测即为 RGB）。不可做 [::-1] 翻转，否则 R/B 对调。
        preproc_angle = 0
        preprocessed_png: bytes | None = None
        preproc_w = preproc_h = 0
        if output_list:
            res = output_list[0]
            dp_res = res.get("doc_preprocessor_res")
            if dp_res is not None:
                preproc_angle = dp_res.get("angle", 0)
                out_arr = dp_res.get("output_img")
                if out_arr is not None:
                    from PIL import Image as _PILImage

                    rgb = out_arr.copy()
                    pil_img = _PILImage.fromarray(rgb)
                    preproc_w, preproc_h = pil_img.size
                    buf = io.BytesIO()
                    pil_img.save(buf, format="PNG")
                    preprocessed_png = buf.getvalue()

        result_count = 0
        for res in output_list:
            result_count += 1
            if result_count > 100:  # 防止异常情况
                _logger.warning("[_process_ocr_output_safe] 结果项过多，可能有问题")
                break
            try:
                if hasattr(res, "rec_texts") and hasattr(res, "rec_scores"):
                    rec_boxes = getattr(res, "rec_boxes", None)
                    for i, (text, score) in enumerate(
                        zip(res.rec_texts, res.rec_scores, strict=False)
                    ):
                        if text:
                            fs = float(score)
                            text_with_scores.append((text, fs))
                            bbox = (
                                self._extract_bbox(rec_boxes, i)
                                if rec_boxes is not None
                                else None
                            )
                            text_blocks.append(
                                TextBlock(text=text, score=fs, bbox=bbox)
                            )
                elif hasattr(res, "rec_texts"):
                    rec_boxes = getattr(res, "rec_boxes", None)
                    for i, text in enumerate(res.rec_texts):
                        if text:
                            text_with_scores.append((text, 1.0))
                            bbox = (
                                self._extract_bbox(rec_boxes, i)
                                if rec_boxes is not None
                                else None
                            )
                            text_blocks.append(
                                TextBlock(text=text, score=1.0, bbox=bbox)
                            )
                elif hasattr(res, "ocr_text"):
                    text_with_scores.append((res.ocr_text, 1.0))
                    text_blocks.append(
                        TextBlock(text=res.ocr_text, score=1.0, bbox=None)
                    )
                elif isinstance(res, dict):
                    rec_texts = res.get("rec_texts", [])
                    rec_scores = res.get("rec_scores", [])
                    rec_boxes = res.get("rec_boxes")
                    if rec_scores:
                        for i, (text, score) in enumerate(
                            zip(rec_texts, rec_scores, strict=False)
                        ):
                            if text:
                                fs = float(score)
                                text_with_scores.append((text, fs))
                                bbox = (
                                    self._extract_bbox(rec_boxes, i)
                                    if rec_boxes is not None
                                    else None
                                )
                                text_blocks.append(
                                    TextBlock(text=text, score=fs, bbox=bbox)
                                )
                    else:
                        for i, text in enumerate(rec_texts):
                            if text:
                                text_with_scores.append((text, 1.0))
                                bbox = (
                                    self._extract_bbox(rec_boxes, i)
                                    if rec_boxes is not None
                                    else None
                                )
                                text_blocks.append(
                                    TextBlock(text=text, score=1.0, bbox=bbox)
                                )
            except Exception as e:
                _logger.error(
                    f"[_process_ocr_output_safe] 处理结果项 #{result_count} 时出错: {e}"
                )
                continue

        raw_text = "\n".join(t for t, _ in text_with_scores)
        _logger.debug(
            f"[_process_ocr_output_safe] 处理完成: 共 {result_count} 个结果项, {len(text_with_scores)} 个文本块"
        )

        result = self._build_ocr_result(
            raw_text=raw_text,
            text_with_scores=text_with_scores,
            pipeline_type="OCR",
            text_blocks=text_blocks,
        )
        result.preproc_angle = preproc_angle
        result.preprocessed_image = preprocessed_png
        result.preproc_img_w = preproc_w
        result.preproc_img_h = preproc_h
        return result

    def _build_ocr_result(
        self,
        raw_text: str,
        markdown_text: str = "",
        html_text: str = "",
        text_with_scores: list[tuple[str, float]] | None = None,
        pipeline_type: str = "OCR",
        images: dict[str, Any] | None = None,
        text_blocks: list[TextBlock] | None = None,
        content_list: list[dict[str, Any]] | None = None,
    ) -> OCRResult:
        """构建 OCRResult 对象

        Args:
            raw_text: 纯文本
            markdown_text: Markdown 格式文本
            html_text: HTML 格式文本
            text_with_scores: 文本块及置信度列表
            pipeline_type: 管道类型
            images: 图像字典
            text_blocks: 含坐标的文本块列表
            content_list: 结构化内容列表（含布局信息）

        Returns:
            OCRResult 对象
        """
        if text_with_scores is None:
            text_with_scores = []

        # 计算平均置信度
        avg_score = 0.0
        if text_with_scores:
            avg_score = sum(s for _, s in text_with_scores) / len(text_with_scores)

        # 收集低置信度项（低于 80%）
        low_confidence_items = [
            (text, score) for text, score in text_with_scores if score < 0.80
        ]

        final_html = html_text or raw_text

        return OCRResult(
            raw_text=raw_text,
            markdown_text=markdown_text or raw_text,
            html_text=final_html,
            text_with_scores=text_with_scores,
            avg_score=avg_score,
            low_confidence_items=low_confidence_items,
            pipeline_type=pipeline_type,
            images=images or {},
            text_blocks=text_blocks or [],
            content_list=content_list or [],
        )
