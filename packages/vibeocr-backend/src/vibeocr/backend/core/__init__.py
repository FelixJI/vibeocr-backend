"""核心抽象层

包含：
- SingletonMeta: 单例元类
- Constants: 全局常量
- OCRPipeline: OCR 管道枚举（来自 pipelines.py）

本包只导出 UI-free Backend 抽象；Qt Worker 基类由
``vibeocr.classic.core.base_worker`` 独立拥有。
"""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

from vibeocr.backend.core.constants import (
    DEFAULT_MARGIN,
    DEFAULT_SHM_SIZE,
    DEFAULT_SPACING,
    LONG_DELAY_MS,
    MEDIUM_DELAY_MS,
    MIN_BATCH_SIZE,
    OCR_BATCH_GPU_SIZE_CAP,
    SHM_TIMEOUT,
    SHORT_DELAY_MS,
    TOAST_DELAY_MS,
    Constants,
    FileType,
)
from vibeocr.backend.core.pipelines import (
    OCRPipeline,
    get_all_pipelines,
    get_pipeline_description,
    get_pipeline_display_name,
    get_pipeline_supported_options,
    is_option_supported,
)
from vibeocr.backend.core.singleton_meta import SingletonMeta

__all__ = [
    "DEFAULT_MARGIN",
    "DEFAULT_SHM_SIZE",
    "DEFAULT_SPACING",
    "LONG_DELAY_MS",
    "MEDIUM_DELAY_MS",
    "MIN_BATCH_SIZE",
    "OCR_BATCH_GPU_SIZE_CAP",
    "SHM_TIMEOUT",
    "SHORT_DELAY_MS",
    "TOAST_DELAY_MS",
    "Constants",
    "FileType",
    "OCRPipeline",
    "SingletonMeta",
    "get_all_pipelines",
    "get_pipeline_description",
    "get_pipeline_display_name",
    "get_pipeline_supported_options",
    "is_option_supported",
]
