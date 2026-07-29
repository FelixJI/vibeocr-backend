"""预热工具模块

提供测试图片创建和 Worker 预热功能。
"""

import io
import logging
import time

_logger = logging.getLogger(__name__)

# 预热用的测试图片数据（200x50 白色背景）
# 这是一个简单的 PNG 图片数据，避免重复创建
_WARMUP_IMAGE_DATA: bytes | None = None


def get_warmup_image() -> bytes | None:
    """获取预热测试图片数据

    返回一个 200x50 像素的白色 PNG 图片，
    用于触发模型加载和 CUDA 上下文初始化。

    Returns:
        PNG 格式的图像数据（bytes）
    """
    global _WARMUP_IMAGE_DATA

    if _WARMUP_IMAGE_DATA is not None:
        return _WARMUP_IMAGE_DATA

    try:
        from PIL import Image

        # 创建 200x50 的白色测试图片
        img = Image.new("RGB", (200, 50), color="white")

        # 添加一些简单的灰度纹理（模拟文字区域）
        from PIL import ImageDraw

        draw = ImageDraw.Draw(img)
        for i in range(0, 200, 20):
            draw.line([(i, 10), (i + 10, 40)], fill="gray", width=2)

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        _WARMUP_IMAGE_DATA = buffer.getvalue()

        _logger.debug(f"创建预热测试图片: {len(_WARMUP_IMAGE_DATA)} 字节")
        return _WARMUP_IMAGE_DATA

    except Exception as e:
        _logger.error(f"创建预热图片失败: {e}")
        return None  # PIL 不可用时返回 None，调用方需处理


def warmup_with_test_image(ocr_service, pipeline: str | None = None) -> bool:
    """使用测试图片预热 OCR 服务

    通过执行一次虚拟识别来触发模型加载和 CUDA 初始化。

    Args:
        ocr_service: OCR 服务实例
        pipeline: 要预热的管道名称，None 表示使用默认 OCR 管道

    Returns:
        预热是否成功
    """
    try:
        from vibeocr.backend.services.ocr_service import OCROptions, OCRPipeline

        _logger.debug("开始预热 OCR 服务...")
        start_time = time.time()

        # 获取测试图片
        test_image = get_warmup_image()
        if test_image is None:
            _logger.error("无法创建预热测试图片，预热中止")
            return False

        # 创建选项
        pipeline_enum = OCRPipeline(pipeline) if pipeline else OCRPipeline.OCR

        options = OCROptions(pipeline=pipeline_enum)

        # 执行虚拟识别
        ocr_service.recognize(test_image, options)

        elapsed = time.time() - start_time
        _logger.debug(f"预热完成，耗时: {elapsed:.2f}秒")
        return True

    except Exception as e:
        _logger.error(f"预热失败: {e}")
        return False


def warmup_worker_process(
    worker_process, pipeline: str = "OCR", timeout: float = 60.0
) -> bool:
    """预热 Worker 进程

    Args:
        worker_process: OCRWorkerProcess 实例
        pipeline: 要预热的管道名称
        timeout: 超时时间（秒）

    Returns:
        预热是否成功
    """
    try:
        _logger.debug(f"开始预热 Worker {worker_process.worker_id}...")
        start_time = time.time()

        # 获取测试图片数据
        test_image = get_warmup_image()
        if test_image is None:
            _logger.error("无法创建预热测试图片，预热中止")
            return False

        # 准备选项
        options_dict = {"pipeline": pipeline}

        # 执行识别（这会触发模型加载）
        worker_process.recognize(test_image, options_dict, timeout=timeout)

        elapsed = time.time() - start_time
        _logger.debug(
            f"Worker {worker_process.worker_id} 预热完成，耗时: {elapsed:.2f}秒"
        )
        return True

    except Exception as e:
        _logger.error(f"Worker {worker_process.worker_id} 预热失败: {e}")
        return False
