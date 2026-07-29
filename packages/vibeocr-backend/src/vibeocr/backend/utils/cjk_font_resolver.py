"""系统 CJK 字体探测 + fontTools 子集化。

为 PDF 文字层提供可嵌入的子集字体：按本页实际用到的字符做子集化，
生成临时小字体文件，PyMuPDF 嵌入后自动生成 ToUnicode CMap，
使文字层在所有主流阅读器可搜索/复制（不依赖阅读器自带 Adobe GB1 CMap）。

跨平台探测系统 CJK 字体，无需随包分发字体。探测失败时返回 None，
调用方回退 china-s（当前行为）。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


class CjkFontResolver:
    """系统 CJK 字体探测 + fontTools 子集化。

    进程级单例：通过模块级 `_CJK_RESOLVER` 实例访问，避免重复探测。
    子集字体按字符集 hash 缓存到临时目录，相同字符集复用。
    """

    # 跨平台候选优先级（复用 qrcode_service._load_font 的模式）
    _WIN_CANDIDATES = [
        "C:/Windows/Fonts/msyh.ttc",  # 微软雅黑
        "C:/Windows/Fonts/simhei.ttf",  # 黑体
        "C:/Windows/Fonts/simsun.ttc",  # 宋体
        "C:/Windows/Fonts/Deng.ttf",  # 等线
    ]
    _MAC_CANDIDATES = [
        "/System/Library/Fonts/PingFang.ttc",
        "/Library/Fonts/Songti.ttc",
    ]
    _LINUX_CANDIDATES = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc",
    ]

    def __init__(self) -> None:
        self._system_font: str | None = None  # 探测缓存（None 表示已探测且无）
        self._probed: bool = False  # 是否已探测过
        self._subset_cache: dict[frozenset[str], str] = {}  # 字符集 → 子集路径

    def _get_candidates(self) -> list[str]:
        """按平台返回候选字体路径列表（可被测试 monkeypatch 覆盖）。

        用方法而非 property：property 无 setter 无法被 monkeypatch.setattr
        覆盖，_find_system_font 调用此方法。
        """
        if sys.platform == "win32":
            return self._WIN_CANDIDATES
        if sys.platform == "darwin":
            return self._MAC_CANDIDATES
        return self._LINUX_CANDIDATES

    def _find_system_font(self) -> str | None:
        """探测首个存在的系统 CJK 字体（结果缓存）。"""
        if self._probed:
            return self._system_font
        for path in self._get_candidates():
            if Path(path).is_file():
                self._system_font = path
                break
        self._probed = True
        if self._system_font is None:
            logger.warning(
                "[CjkFontResolver] 未找到系统 CJK 字体，文字层将回退 china-s"
            )
        return self._system_font

    def resolve(self, chars: str) -> str | None:
        """返回覆盖 chars 的子集字体路径；探测失败或空字符返回 None。

        Args:
            chars: 本页文字层需要的所有字符。
        Returns:
            子集字体临时文件路径，或 None（调用方回退 china-s）。
        """
        if not chars:
            return None
        sys_font = self._find_system_font()
        if sys_font is None:
            return None
        key = frozenset(chars)
        if key not in self._subset_cache:
            try:
                self._subset_cache[key] = self._subset(sys_font, chars)
            except Exception as e:
                logger.warning("[CjkFontResolver] 子集化失败，回退内置字体: %s", e)
                return None
        return self._subset_cache[key]

    @staticmethod
    def _subset(orig_path: str, chars: str) -> str:
        """fontTools 子集化到临时文件，返回路径。

        .ttc（字体集合）需 fontNumber=0 取第一个 face；.ttf 直接打开。
        populate(text=...) 自动闭包 notdef 等必需字形。
        """
        import os
        import tempfile

        from fontTools import subset
        from fontTools.ttLib import TTFont

        is_ttc = orig_path.lower().endswith(".ttc")
        font = TTFont(orig_path, fontNumber=0) if is_ttc else TTFont(orig_path)
        sub = subset.Subsetter()
        sub.populate(text=chars)
        sub.subset(font)
        fd, path = tempfile.mkstemp(suffix=".ttf", prefix="vibeocr_subset_")
        os.close(fd)
        try:
            font.save(path)
        except Exception:
            # font.save 失败时删除已创建的临时文件，避免泄漏（缓存不会收录此 key）
            Path(path).unlink(missing_ok=True)
            raise
        return path

    def cleanup(self) -> None:
        """删除所有缓存的子集临时文件（进程退出或 session 关闭时调用）。"""
        for path in self._subset_cache.values():
            try:
                Path(path).unlink(missing_ok=True)
            except OSError:
                pass
        self._subset_cache.clear()


# 模块级单例：pdf_service.py 通过此实例访问，避免重复探测与子集化。
import atexit  # noqa: E402

_CJK_RESOLVER = CjkFontResolver()


def _cleanup_on_exit() -> None:
    """进程退出时清理临时子集字体文件。"""
    _CJK_RESOLVER.cleanup()


atexit.register(_cleanup_on_exit)
