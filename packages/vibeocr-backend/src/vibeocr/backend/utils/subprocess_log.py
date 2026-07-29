"""子进程 stdout 日志转发公共工具。

项目有三个常驻子进程（OCR worker / PDF 后端 / MinerU API），各自把日志打到
stdout/stderr，由主进程读回并转发到项目日志系统。本模块抽出统一的转发逻辑，
确保三个子进程通道行为一致：

- **结构化行精确还原**：匹配子进程标准日志格式
  ``YYYY-MM-DD HH:MM:SS [LEVEL] name: message`` 的行，按原始级别转发。
- **裸 print 折叠防泄漏**：PaddleX/transformers/fitz 等库会直接 ``print`` 内容，
  可能含用户文档片段。无法匹配结构化格式的行不原样转发，只累加计数，在遇到
  结构化行、显式 :meth:`flush` 或达到阈值时以概括形式输出行数。
- **多行拼接分割**：PaddlePaddle 的 ``warnings.warn()`` 有时无换行符，导致多行
  被拼到一行；:meth:`split_mixed_lines` 按日期时间模式重新切分。

历史：本逻辑最早实现于 ``OCRWorkerProcess._parse_and_forward_log``，是三套转发器
里最完善的，2026-07 统一日志通道时抽出供 PDF 后端、MinerU 复用。
"""

from __future__ import annotations

import logging
import re
import threading

__all__ = ["SubprocessLogForwarder"]


# 子进程标准日志格式：2024-01-15 10:30:45 [INFO] module: message
_STRUCTURED_LINE_RE = re.compile(
    # %(asctime)s 默认格式是 "2026-07-22 13:16:50,123"（带逗号毫秒），
    # 而非 "2026-07-22 13:16:50"。逗号毫秒必须可选匹配，否则 worker
    # 子进程的所有结构化日志行都被当成裸 print 折叠丢失。
    r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:,\d{3})?\s+\[(DEBUG|INFO|WARNING|ERROR|CRITICAL)\]\s+(.*)"
)

# 日期时间模式：YYYY-MM-DD HH:MM:SS（用于分割无换行拼接的多行）
_DATETIME_PATTERN = r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}"

# Python traceback 起始行：Traceback (most recent call last):
_TRACEBACK_START_RE = re.compile(r"Traceback \(most recent call last\):")

# Python 异常末行：xxx.Error: message（如 ModuleNotFoundError: No module named 'x'）
# 用于捕获 traceback 最后一行的真实错误原因。
_EXCEPTION_LINE_RE = re.compile(
    r"^(?:[A-Za-z_][\w.]*\.)?[A-Za-z_][\w]*Error\b.*|^[A-Za-z_][\w]*Exception\b.*|^ImportError\b.*|^ModuleNotFoundError\b.*"
)


class SubprocessLogForwarder:
    """子进程 stdout 日志转发器。

    每个子进程主进程侧持有一个实例。线程安全（裸 print 计数加锁）。

    Args:
        logger_name: 转发目标 logger 名，统一用 ``vibeocr.subprocess.<name>``。
        source_label: 转发时附加的来源前缀，如 ``"[Worker 0]"``、``"[PDF Backend]"``、
            ``"[MinerU API]"``，便于在日志中区分来源。
        raw_flush_threshold: 裸 print 累积达到此行数立即 flush，避免长时间不输出。
            默认 50，与历史 OCR 行为一致。
    """

    def __init__(
        self,
        *,
        logger_name: str,
        source_label: str,
        raw_flush_threshold: int = 50,
    ) -> None:
        self._logger = logging.getLogger(logger_name)
        self._source_label = source_label
        self._raw_flush_threshold = raw_flush_threshold
        self._raw_log_count = 0
        self._raw_log_lock = threading.Lock()

    def forward(self, text: str) -> None:
        """处理一行子进程输出文本。

        匹配结构化格式的行按原始级别转发；不匹配的裸 print 只计数折叠，
        达到阈值时以概括形式输出行数。空白行忽略（不计数、不转发）。

        **异常 traceback 不折叠**：子进程崩溃时（如 import 失败退出码 1）输出
        的 ``Traceback (most recent call last): ...`` 及末行异常名必须以 ERROR
        级别原样转发,否则真实错误被折叠掉,只剩"退出码 1"无法定位。
        """
        # 结构化行到来前，先把累积的裸 print 概括输出
        match = _STRUCTURED_LINE_RE.match(text)
        if match:
            self.flush()
            level_name = match.group(1)
            message = match.group(2)
            level = getattr(logging, level_name, logging.DEBUG)
            self._logger.log(level, f"{self._source_label} {message}")
            return

        # 纯空白行忽略，不计数、不转发
        if not text.strip():
            return

        # 异常 traceback：原样以 ERROR 转发（不折叠,不计数）
        # - Traceback 起始行
        # - 末行异常名（ModuleNotFoundError / ImportError / xxxError / xxxException）
        if _TRACEBACK_START_RE.search(text) or _EXCEPTION_LINE_RE.match(text.strip()):
            self.flush()
            self._logger.error(f"{self._source_label} {text.rstrip()}")
            return

        with self._raw_log_lock:
            self._raw_log_count += 1
            # 累积超过阈值则立即 flush，避免长时间不输出
            if self._raw_log_count >= self._raw_flush_threshold:
                count = self._raw_log_count
                self._raw_log_count = 0
            else:
                return

        self._logger.debug(
            f"{self._source_label} 子进程原始输出 {count} 行（已折叠，含库调试信息）"
        )

    def flush(self) -> None:
        """输出并清空已累积的裸 print 概括计数。

        在结构化日志行到来、或读取循环结束时调用，确保折叠的概括被输出。
        """
        with self._raw_log_lock:
            count = self._raw_log_count
            self._raw_log_count = 0
        if count > 0:
            self._logger.debug(
                f"{self._source_label} 子进程原始输出 {count} 行（已折叠，含库调试信息）"
            )

    @staticmethod
    def split_mixed_lines(text: str) -> list[str]:
        """分割混合的日志行。

        PaddlePaddle 的 ``warnings.warn()`` 输出有时没有换行符，
        导致多个日志行被拼接到一行。本方法识别其中的日期时间模式并切分。

        Args:
            text: 可能包含多个日志行的文本

        Returns:
            分割后的日志行列表
        """
        matches = list(re.finditer(_DATETIME_PATTERN, text))

        if len(matches) <= 1:
            # 没有拼接，直接返回
            return [text] if text else []

        # 按日期时间模式分割
        lines = []
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            line = text[start:end].strip()
            if line:
                lines.append(line)

        return lines
