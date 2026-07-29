"""
OCR Service 使用便携式 Python

这个模块通过在主进程中修改 sys.path 来导入 PaddleOCR，
而不是使用子进程。支持：
- 开发环境：使用 .venv 虚拟环境
- 生产环境：使用便携式 python/ 目录

与子进程方案的对比：
- 优点：无进程间通信开销，调试方便，代码更简单
- 缺点：失去进程隔离，PaddleOCR 崩溃会影响主程序
"""

import logging
import os
import threading
from typing import Any, Optional

import numpy as np
from PIL import Image

# 跳过模型源网络检测
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

# 导入路径管理器
from vibeocr.backend.python_path_manager import (
    PythonPathMode,
    get_python_path_manager,
)

_logger = logging.getLogger(__name__)


class OCRServicePortable:
    """
    OCR Service 使用便携式 Python

    通过路径管理器在主进程中导入 PaddleOCR，支持开发和生产环境。

    ⚠️ 非默认路径（仅调试逃生口）：仅当 ``VIBEOCR_USE_SUBPROCESS=false`` 且
    ``VIBEOCR_OCR_MODE != direct``（即默认的直连分支）时由工厂选中。
    与 OCRService 一样在主进程内加载模型，会阻塞 UI、占用 GPU 上下文，
    **生产环境应走子进程模式（OCRServiceSubprocess，默认）**。
    仅用于便携式打包环境下的调试/排查。
    """

    _instance: Optional["OCRServicePortable"] = None
    _pipeline: Any = None
    _lock = threading.Lock()

    def is_ready(self) -> bool:
        """服务就绪（便携式模式：pipeline 已加载即就绪）"""
        return self._pipeline is not None

    def __new__(cls) -> "OCRServicePortable":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """初始化 OCR 服务"""
        self.path_manager = get_python_path_manager()
        self.path_manager.setup_sys_path()

        _logger.debug(f"OCR 服务初始化，Python 模式: {self.path_manager.mode}")
        if self.path_manager.ocr_lib_path:
            _logger.debug(f"OCR 库路径: {self.path_manager.ocr_lib_path}")

    def _import_paddleocr(self):
        """导入 PaddleOCR（延迟导入）"""
        try:
            from paddleocr import PaddleOCR

            return PaddleOCR
        except ImportError as e:
            error_msg = f"无法导入 PaddleOCR: {e}\n"

            if self.path_manager.mode == PythonPathMode.DEVELOPMENT:
                error_msg += "\n开发环境解决方案：\n"
                error_msg += "1. 激活虚拟环境: source .venv/bin/activate (Linux/Mac) 或 .venv\\Scripts\\activate (Windows)\n"
                error_msg += "2. 安装 PaddleOCR: pip install paddleocr paddlepaddle\n"

            elif self.path_manager.mode == PythonPathMode.PORTABLE:
                error_msg += "\n便携式环境解决方案：\n"
                error_msg += "1. 检查 python/ 目录是否存在且完整\n"
                error_msg += "2. 运行环境设置: python -m vibeocr.backend.env_manager\n"

            else:
                error_msg += "\n系统环境解决方案：\n"
                error_msg += "1. 安装 PaddleOCR: pip install paddleocr paddlepaddle\n"

            _logger.error(error_msg)
            raise ImportError(error_msg) from e

    def _create_pipeline(self) -> Any:
        """创建 OCR 流水线（CPU 模式）"""
        PaddleOCR = self._import_paddleocr()
        pipeline = PaddleOCR(device="cpu")
        _logger.debug("OCR 流水线创建成功，设备: cpu")
        return pipeline

    @property
    def pipeline(self) -> Any:
        """获取 OCR 流水线（懒加载，线程安全，CPU 模式）"""
        if self._pipeline is None:
            with self._lock:
                if self._pipeline is None:
                    self._pipeline = self._create_pipeline()

        return self._pipeline

    def recognize(
        self,
        image: Image.Image | np.ndarray | str | bytes,
        options: Any = None,
    ) -> str:
        """
        对图像执行 OCR 识别

        Args:
            image: PIL Image、numpy 数组或图像路径

        Returns:
            识别的文本内容
        """
        output = self.pipeline.predict(
            input=image,
            use_doc_orientation_classify=True,
            use_doc_unwarping=True,
            use_textline_orientation=True,
        )

        texts = []
        for res in output:
            if hasattr(res, "rec_texts"):
                texts.extend(res.rec_texts)
            elif hasattr(res, "ocr_text"):
                texts.append(res.ocr_text)
            elif isinstance(res, dict):
                rec_texts = res.get("rec_texts", [])
                texts.extend(rec_texts)

        return "\\n".join(texts) if texts else ""

    def recognize_batch(self, images, options=None):
        """批量识别多张图像（便携式模式：逐张调用 recognize）。

        Args:
            images: 输入图像列表。
            options: OCR 识别选项（便携式模式忽略，固定开启预处理）。

        Returns:
            文本结果列表，顺序与 images 一致。
        """
        return [self.recognize(img, options) for img in images]

    def get_environment_info(self) -> dict:
        """获取环境信息"""
        return self.path_manager.get_environment_info()

    def verify_environment(self) -> tuple[bool, str]:
        """验证环境是否正确配置"""
        return self.path_manager.verify_environment()


# 便捷函数
def get_ocr_service_portable() -> OCRServicePortable:
    """获取便携式 OCR 服务实例"""
    return OCRServicePortable()


# 为了向后兼容，提供一个别名
OCRService = OCRServicePortable


# 测试函数
def test_portable_ocr():
    """测试便携式 OCR 服务"""
    print("\n" + "=" * 60)
    print("便携式 OCR 服务测试")
    print("=" * 60)

    # 获取服务
    service = get_ocr_service_portable()

    # 打印环境信息
    info = service.get_environment_info()
    print("\n环境信息:")
    print(f"  模式: {info['mode']}")
    print(f"  是否打包: {info['is_frozen']}")
    print(f"  Python: {info['python_executable']}")
    print(f"  OCR 库路径: {info['ocr_lib_path']}")
    print(f"  可导入 PaddleOCR: {info['can_import_paddleocr']}")

    # 验证环境
    success, message = service.verify_environment()
    print(f"\n环境验证: {message}")

    if success:
        print("\n✓ 环境配置正确，可以正常使用 OCR 服务")
        return 0
    print(f"\n✗ 环境配置有问题: {message}")
    return 1


if __name__ == "__main__":
    import sys

    sys.exit(test_portable_ocr())
