"""
Python 路径管理模块

负责在开发环境和生产环境中管理 Python 路径，使 PaddleX 等重型依赖
可以在主进程中导入，而不需要子进程。

支持三种模式：
1. 开发环境：使用虚拟环境 (.venv)
2. 生产环境：使用便携式 Python (python/)
3. 直接导入：使用系统 Python
"""

import logging
import os
import sys
from pathlib import Path

_logger = logging.getLogger(__name__)


class PythonPathMode:
    """Python 路径模式"""

    DEVELOPMENT = "development"  # 开发环境：使用 .venv
    PORTABLE = "portable"  # 便携式：使用 python/
    SYSTEM = "system"  # 系统：使用已安装的包


class PythonPathManager:
    """
    Python 路径管理器

    负责检测和管理 Python 环境的路径配置，使应用可以在不同环境下
    正确导入 PaddleX 等依赖。
    """

    def __init__(self, project_root: Path | None = None):
        """
        初始化路径管理器

        Args:
            project_root: 项目根目录，如果为 None 则自动检测
        """
        self.project_root = project_root or self._detect_project_root()
        self._mode: str | None = None
        self._ocr_lib_path: Path | None = None
        self._python_executable: Path | None = None

    def _detect_project_root(self) -> Path:
        """检测项目根目录"""
        # 从当前文件向上查找
        current = Path(__file__).resolve().parent

        # 查找包含 src/vibeocr 的目录
        while current.parent != current:
            if (current / "src" / "vibeocr").exists():
                return current
            current = current.parent

        # 如果找不到，返回当前工作目录
        return Path.cwd()

    @property
    def is_frozen(self) -> bool:
        """检测是否为打包环境"""
        return getattr(sys, "frozen", False)

    @property
    def mode(self) -> str:
        """获取当前模式"""
        if self._mode is None:
            self._mode = self._detect_mode()
        return self._mode

    def _detect_mode(self) -> str:
        """
        检测当前应该使用的模式

        优先级：
        1. 环境变量 VIBEOCR_PYTHON_MODE
        2. 打包环境 → PORTABLE
        3. 开发环境有 .venv → DEVELOPMENT
        4. 有便携式 python/ → PORTABLE
        5. 否则 → SYSTEM
        """
        # 1. 检查环境变量
        env_mode = os.environ.get("VIBEOCR_PYTHON_MODE", "").lower()
        if env_mode in ["development", "dev", ".venv"]:
            return PythonPathMode.DEVELOPMENT
        if env_mode in ["portable", "python"]:
            return PythonPathMode.PORTABLE
        if env_mode in ["system", "global"]:
            return PythonPathMode.SYSTEM

        # 2. 打包环境强制使用便携式
        if self.is_frozen:
            return PythonPathMode.PORTABLE

        # 3. 开发环境优先使用虚拟环境
        venv_libs = self.project_root / ".venv" / "Lib" / "site-packages"
        if venv_libs.exists():
            return PythonPathMode.DEVELOPMENT

        # 4. 检查便携式 Python
        portable_libs = self.project_root / "python" / "Lib" / "site-packages"
        if portable_libs.exists():
            return PythonPathMode.PORTABLE

        # 5. 默认使用系统
        return PythonPathMode.SYSTEM

    @property
    def ocr_lib_path(self) -> Path | None:
        """获取 OCR 库路径（site-packages）"""
        if self._ocr_lib_path is None:
            self._ocr_lib_path = self._find_ocr_lib_path()
        return self._ocr_lib_path

    def _find_ocr_lib_path(self) -> Path | None:
        """查找 OCR 库路径"""
        mode = self.mode

        if mode == PythonPathMode.DEVELOPMENT:
            # 开发环境：使用 .venv
            path = self.project_root / ".venv" / "Lib" / "site-packages"
            if path.exists():
                return path

        elif mode == PythonPathMode.PORTABLE:
            # 便携式：使用 python/
            if self.is_frozen:
                # 打包环境：python/ 在 exe 同级目录
                app_dir = Path(sys.executable).parent
                path = app_dir / "python" / "Lib" / "site-packages"

                # 检查是否存在，如果不存在尝试其他可能的路径
                if not path.exists():
                    # 尝试 Resources 目录（macOS 打包常见）
                    resources_path = (
                        app_dir / "Resources" / "python" / "Lib" / "site-packages"
                    )
                    if resources_path.exists():
                        return resources_path

                    # 记录警告但继续尝试
                    import logging

                    _logger = logging.getLogger(__name__)
                    _logger.warning(f"便携式 OCR 库路径不存在: {path}")

                return path
            # 开发环境：python/ 在项目根目录
            path = self.project_root / "python" / "Lib" / "site-packages"
            if path.exists():
                return path

        # SYSTEM 模式或找不到路径：返回 None，使用系统默认
        return None

    @property
    def python_executable(self) -> Path | None:
        """获取 Python 可执行文件路径"""
        if self._python_executable is None:
            self._python_executable = self._find_python_executable()
        return self._python_executable

    def _find_python_executable(self) -> Path | None:
        """查找 Python 可执行文件"""
        mode = self.mode

        if mode == PythonPathMode.DEVELOPMENT:
            # 开发环境：使用 .venv
            if os.name == "nt":
                path = self.project_root / ".venv" / "Scripts" / "python.exe"
            else:
                path = self.project_root / ".venv" / "bin" / "python"
            return path if path.exists() else None

        if mode == PythonPathMode.PORTABLE:
            # 便携式：使用 python/
            if self.is_frozen:
                app_dir = Path(sys.executable).parent
            else:
                app_dir = self.project_root

            if os.name == "nt":
                path = app_dir / "python" / "python.exe"
            else:
                path = app_dir / "python" / "bin" / "python"
            return path if path.exists() else None

        # SYSTEM 模式：返回当前 Python
        return Path(sys.executable)

    def setup_sys_path(self) -> bool:
        """
        设置 sys.path 以便正确导入 OCR 库

        Returns:
            是否成功设置路径
        """
        ocr_path = self.ocr_lib_path

        if ocr_path is None:
            _logger.debug("使用系统 Python 路径，无需修改 sys.path")
            return True

        # 检查是否已在路径中
        ocr_path_str = str(ocr_path)
        if ocr_path_str in sys.path:
            _logger.debug(f"OCR 库路径已存在: {ocr_path}")
            return True

        # 添加到 sys.path 的最前面
        sys.path.insert(0, ocr_path_str)
        _logger.debug(f"已添加 OCR 库路径到 sys.path: {ocr_path}")
        _logger.debug(f"sys.path 前 3 项: {sys.path[:3]}")

        return True

    def get_environment_info(self) -> dict:
        """
        获取环境信息

        Returns:
            包含环境详细信息的字典
        """
        info = {
            "mode": self.mode,
            "is_frozen": self.is_frozen,
            "project_root": str(self.project_root),
            "python_executable": (
                str(self.python_executable) if self.python_executable else None
            ),
            "ocr_lib_path": str(self.ocr_lib_path) if self.ocr_lib_path else None,
            "sys_executable": sys.executable,
            "sys_path_first_3": sys.path[:3],
        }

        # 检查关键包是否可导入
        info["can_import_paddleocr"] = self._can_import("paddleocr")
        info["can_import_paddle"] = self._can_import("paddle")

        return info

    def _can_import(self, module_name: str) -> bool:
        """检查是否可以导入模块"""
        try:
            __import__(module_name)
            return True
        except ImportError:
            return False

    def verify_environment(self) -> tuple[bool, str]:
        """
        验证环境是否正确配置

        Returns:
            (是否成功, 错误消息)
        """
        # 检查路径是否存在
        if self.mode != PythonPathMode.SYSTEM:
            if self.ocr_lib_path is None or not self.ocr_lib_path.exists():
                return False, f"OCR 库路径不存在: {self.ocr_lib_path}"

            if self.python_executable is None or not self.python_executable.exists():
                return False, f"Python 可执行文件不存在: {self.python_executable}"

        # 检查 PaddleX 是否可导入
        if not self._can_import("paddleocr"):
            return False, "无法导入 PaddleX，请确保已正确安装"

        return True, "环境验证通过"

    def print_info(self):
        """打印环境信息（用于调试）"""
        info = self.get_environment_info()

        print("\n" + "=" * 60)
        print("Python 路径管理器信息")
        print("=" * 60)
        print(f"模式: {info['mode']}")
        print(f"是否打包: {info['is_frozen']}")
        print(f"项目根目录: {info['project_root']}")
        print(f"Python 可执行文件: {info['python_executable']}")
        print(f"OCR 库路径: {info['ocr_lib_path']}")
        print(f"sys.executable: {info['sys_executable']}")
        print(f"可导入 PaddleOCR: {info['can_import_paddleocr']}")
        print(f"可导入 Paddle: {info['can_import_paddle']}")
        print("=" * 60 + "\n")


# 全局单例
_global_manager: PythonPathManager | None = None


def get_python_path_manager(project_root: Path | None = None) -> PythonPathManager:
    """
    获取全局 Python 路径管理器实例

    Args:
        project_root: 项目根目录，只在首次调用时生效

    Returns:
        PythonPathManager 实例
    """
    global _global_manager

    if _global_manager is None:
        _global_manager = PythonPathManager(project_root)

    return _global_manager


def setup_python_path(project_root: Path | None = None) -> bool:
    """
    便捷函数：设置 Python 路径

    Args:
        project_root: 项目根目录

    Returns:
        是否成功设置
    """
    manager = get_python_path_manager(project_root)
    return manager.setup_sys_path()


def get_environment_info(project_root: Path | None = None) -> dict:
    """
    便捷函数：获取环境信息

    Args:
        project_root: 项目根目录

    Returns:
        环境信息字典
    """
    manager = get_python_path_manager(project_root)
    return manager.get_environment_info()


# 自动初始化（当模块被导入时）
def _auto_init():
    """模块导入时自动初始化"""
    manager = get_python_path_manager()
    manager.setup_sys_path()

    # 记录日志
    _logger.debug(f"Python 路径管理器初始化完成，模式: {manager.mode}")
    if manager.ocr_lib_path:
        _logger.debug(f"OCR 库路径: {manager.ocr_lib_path}")


# 延迟初始化，避免在某些情况下（如文档生成）执行
if not hasattr(sys, "_docutils_running"):
    try:
        _auto_init()
    except Exception as e:
        # 初始化失败不应阻止模块导入
        _logger.warning(f"Python 路径管理器自动初始化失败: {e}")
