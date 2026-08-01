"""全局常量定义

集中管理应用程序中使用的所有常量。

注意: OCRPipeline 枚举已移至 pipelines.py，此文件不再定义管道类型。
"""

from enum import Enum, auto

try:
    from vibeocr.backend import __version__ as _APP_VERSION
except (
    ImportError
):  # pragma: no cover - defensive fallback when namespace not installed
    _APP_VERSION = "0.1.0"


class FileType(Enum):
    """支持的文件类型"""

    PDF = auto()
    IMAGE = auto()
    DOC = auto()
    DOCX = auto()
    UNKNOWN = auto()


class Constants:
    """应用程序常量"""

    # 应用程序信息
    APP_NAME = "VibeOCR"
    APP_VERSION = _APP_VERSION
    APP_DESCRIPTION = "基于 PaddleOCR 的文档识别工具"

    # 窗口尺寸
    WINDOW_MIN_WIDTH = 1200
    WINDOW_MIN_HEIGHT = 800
    WINDOW_DEFAULT_WIDTH = 1400
    WINDOW_DEFAULT_HEIGHT = 900

    # 支持文件格式
    SUPPORTED_IMAGE_FORMATS = [
        "*.png",
        "*.jpg",
        "*.jpeg",
        "*.bmp",
        "*.tiff",
        "*.tif",
        "*.webp",
    ]
    SUPPORTED_PDF_FORMATS = ["*.pdf"]
    SUPPORTED_DOC_FORMATS = ["*.doc", "*.docx"]

    # 共享内存配置
    # 传输层须 ≥ 计算层：GPU 一次 predict（text_recognition_batch_size=8）×2 ≈
    # 一个页批(16 页)。单张 300dpi A4 PNG 上限 ≈4MB，16 页 ≈64MB；预算
    # 0.7×(128MB−9)≈90MB 能让一条 SHM 消息装下完整页批，避免传输批(~3 页)卡
    # 住计算批(8 页)，GPU 喂不饱（性能2）。系统内存代价 +112MB，可忽略。
    DEFAULT_SHM_SIZE = 128 * 1024 * 1024  # 128MB
    DEFAULT_SHM_LOG_SIZE = 1 * 1024 * 1024  # 1MB

    # 批量处理配置
    # OCR 批量识别（多文件图片 Tab）单次 predict 的 batch_size 上界。
    # 实际值由 BatchQueueManager._calculate_batch_size 按显卡显存动态估算
    # （gpu_memory_monitor.estimate_batch_size：free*0.7 / 单图显存），此常量
    # 只作为动态估算的上限 clamp，必须 ≥ GPU 计算批(text_recognition_batch_size=8)
    # 才不会人为卡死 GPU。与 estimate_batch_size 内部的 16 上限对齐。
    # 低显存卡由动态估算自动降到更小值，无需改此常量。
    OCR_BATCH_GPU_SIZE_CAP = 16
    # 单次 OCR 传输/推理预算。字节预算低于 SHM 可用区约 90MB，给协议元数据、
    # 结果与并发预取留余量；像素预算约等于 16 张 2000×2000 图，约束解码后
    # 内存/显存，而不仅是压缩文件大小。
    OCR_BATCH_MAX_ENCODED_BYTES = 64 * 1024 * 1024
    OCR_BATCH_MAX_PIXELS = 64_000_000

    # 日志配置
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    # 样式常量
    class Style:
        """UI 样式常量"""

        BORDER_RADIUS = 6
        BORDER_RADIUS_LARGE = 8
        PADDING_SMALL = 8
        PADDING_MEDIUM = 12
        PADDING_LARGE = 16
        SPACING_SMALL = 8
        SPACING_MEDIUM = 12
        SPACING_LARGE = 16

    # 超时配置（秒）
    class Timeout:
        """超时配置（单位:秒,毫秒子类除外）

        所有超时收敛到此处的单一来源。同名常量在不同场景的值差异
        通过命名前缀区分（如 RECOGNIZE_* vs PRELOAD_*）,避免跨文件
        "同名不同值"的隐患。
        """

        # —— OCR 识别 ——
        RECOGNIZE_CACHED = 60.0  # 模型已缓存时的单次识别超时
        RECOGNIZE_UNCACHED = 600.0  # 模型未缓存（首次,给下载留时间）
        DOCUMENT_PARSING = 600.0  # MinerU/PaddleOCR-VL/PP-StructureV3 文档解析
        BATCH_PER_PAGE_EXTRA = 30.0  # 批量识别每页额外超时
        BATCH_MAX = 1800.0  # 批量识别子批封顶（30 分钟）

        # —— 预加载 / 预热 ——
        PRELOAD_CACHED = 60.0  # 所有模型已缓存时的预加载基础超时
        PRELOAD_UNCACHED = 300.0  # 有模型未缓存时的预加载基础超时（5 分钟）
        PRELOAD_PER_PIPELINE = 30.0  # 每个额外管道增加的预加载超时
        PIPELINE_PRELOAD_DEFAULT = 180.0  # preload_pipelines 形参默认
        WARMUP_DEFAULT = 60.0  # warmup_pipelines 形参默认

        # —— Worker 进程 ——
        WORKER_TIMEOUT = 300.0  # SHM 通信/Worker 主循环空闲上限（5 分钟）
        WORKER_START = 120.0  # Worker 启动超时（含 CUDA 上下文初始化）
        WORKER_START_BASE = 30.0  # 基础启动超时（轻量场景）
        RESTART = 60.0  # Worker 重启尝试超时
        SHUTDOWN = 5.0  # 优雅关闭超时
        PROCESS_CANCEL_GRACE = 0.4  # kill 后等待 stdout/stderr drain 的短宽限

        # —— 批量队列 ——
        BATCH_QUEUE = 5.0  # 批量队列 get/put 超时
        BATCH_COMMIT_DEFAULT = 300.0  # batch_commit 形参默认（5 分钟,多文件汇总）

        # —— IPC / SHM 协议 ——
        SHM_WRITE = 30.0  # write_message 默认超时
        SHM_READ = 60.0  # read_message 默认超时
        SHM_WAIT_FOR_READ = 5.0  # wait_for_read 默认超时
        # 尽力而为控制 RPC（set_pipeline_ttls）的 _shm_lock 等待上限。
        # TTL 已持久化且 worker 恢复时重新下发，锁被占（恢复预加载/OCR）时
        # 快速失败返回 False，不阻塞 15s 导致后台任务 traceback + SHM 状态污染。
        CONTROL_RPC_LOCK_BEST_EFFORT = 1.0

        # —— MinerU 远程服务 ——
        MINERU_API_START = 120.0  # mineru-api 启动轮询上限
        MINERU_HTTP_TOTAL = 1800.0  # file_parse httpx 总超时（30 分钟）
        MINERU_HTTP_CONNECT = 30.0  # file_parse httpx 连接超时
        MINERU_MODEL_DOWNLOAD = 1800.0  # ensure_mineru_models 下载超时
        MINERU_MODEL_PROBE = 30.0  # 模型可用性探测子进程超时

        # —— 通用 ——
        FILE_OPERATION = 10.0  # 文件操作（读/写/复制）
        HEALTH_CHECK_INTERVAL = 30.0  # Worker 健康检查间隔

        # —— 向后兼容别名 ——
        # 旧代码可能引用这些名称,保留以避免破坏。新增代码应使用上面的语义命名。
        OCR_RECOGNIZE = 60.0  # = RECOGNIZE_CACHED
        PIPELINE_PRELOAD = 120.0  # 历史值,新代码用 PIPELINE_PRELOAD_DEFAULT

        class Ms:
            """毫秒级超时（Qt API 边界,如 QThread.wait / QLocalSocket.waitFor*）

            这些值直接传给 Qt 的毫秒参数 API,不参与秒级换算。
            """

            SINGLE_INSTANCE = 1000  # 单实例 socket 连接/读写
            SUBPROCESS_SHUTDOWN = 3000  # 子进程线程池关闭
            APP_SHUTDOWN_TOTAL = 5000  # 应用从单一 wall-clock 预算扣减
            BATCH_SHUTDOWN = 750  # 批处理线程协作取消与 drain
            BATCH_DRAIN = 650  # 给协调器线程调度/结果回收预留余量
            PDF_SHUTDOWN = 1400  # PDF worker 与缩略图共享的关闭步骤上限
            PDF_DRAIN = 1250  # PDF 内部 wall-clock drain 预算
            SETTINGS_SHUTDOWN = 800  # 设置页后台硬件探测关闭步骤上限
            SETTINGS_DRAIN = 700  # 覆盖 0.2s 探测轮询 + 0.4s 管道 drain
            ASYNC_RUNNER_SHUTDOWN = 500  # qasync 任务请求取消
            BACKEND_SESSION_SHUTDOWN = 2000  # WorkerHost session 关闭
            PDF_WORKER_CANCEL = 5000  # PDF worker 取消等待默认
            PDF_WORKER_CANCEL_SHORT = 3000  # PDF worker 取消等待（加载场景）
            PDF_WORKER_POLL_STEP = 50  # _wait_thread 单步轮询
            PDF_WORKER_TERMINATE_WAIT = 500  # terminate 后兜底等待
            # 缩略图 HTTP connect timeout 为 5s；后端停止后最终 drain 必须覆盖它。
            PDF_THUMBNAIL_DRAIN_WAIT = 6000


# 向后兼容的模块级导出（旧代码可能直接 import WORKER_TIMEOUT 等）
WORKER_TIMEOUT = Constants.Timeout.WORKER_TIMEOUT
WORKER_START_TIMEOUT = Constants.Timeout.WORKER_START_BASE
BATCH_QUEUE_TIMEOUT = Constants.Timeout.BATCH_QUEUE


# 向后兼容的常量导出
DEFAULT_SHM_SIZE = Constants.DEFAULT_SHM_SIZE
SHM_TIMEOUT = Constants.Timeout.WORKER_TIMEOUT  # = WORKER_TIMEOUT,向后兼容别名
SHORT_DELAY_MS = 100
MEDIUM_DELAY_MS = 500
LONG_DELAY_MS = 1000
TOAST_DELAY_MS = 3000
OCR_BATCH_GPU_SIZE_CAP = Constants.OCR_BATCH_GPU_SIZE_CAP
MIN_BATCH_SIZE = 1
DEFAULT_SPACING = Constants.Style.SPACING_MEDIUM
DEFAULT_MARGIN = Constants.Style.PADDING_MEDIUM


del _APP_VERSION
