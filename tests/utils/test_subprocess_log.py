"""utils.subprocess_log 子进程日志转发的边缘用例测试。

覆盖 SubprocessLogForwarder 的结构化行转发、traceback 原样输出、
裸 print 折叠与阈值 flush，以及 split_mixed_lines 多行切分。
"""

from __future__ import annotations

import logging

from vibeocr.backend.utils.subprocess_log import SubprocessLogForwarder


def _make_forwarder(threshold: int = 50) -> SubprocessLogForwarder:
    return SubprocessLogForwarder(
        logger_name="test.subprocess",
        source_label="[Test]",
        raw_flush_threshold=threshold,
    )


class TestStructuredLineForwarding:
    def test_forwards_structured_line_with_level(self, caplog):
        forwarder = _make_forwarder()
        with caplog.at_level(logging.DEBUG, logger="test.subprocess"):
            forwarder.forward("2024-01-15 10:30:45 [WARNING] mod: hello")
        assert len(caplog.records) == 1
        assert caplog.records[0].levelno == logging.WARNING
        assert "[Test]" in caplog.records[0].message
        assert "hello" in caplog.records[0].message

    def test_structured_line_with_millis(self, caplog):
        """带逗号毫秒的结构化行被正确匹配（历史回归）。"""
        forwarder = _make_forwarder()
        with caplog.at_level(logging.DEBUG, logger="test.subprocess"):
            forwarder.forward("2026-07-22 13:16:50,123 [INFO] mod: with-ms")
        assert any("with-ms" in r.message for r in caplog.records)

    def test_critical_level_forwarded(self, caplog):
        forwarder = _make_forwarder()
        with caplog.at_level(logging.DEBUG, logger="test.subprocess"):
            forwarder.forward("2024-01-15 10:30:45 [CRITICAL] mod: boom")
        assert caplog.records[0].levelno == logging.CRITICAL

    def test_debug_info_and_error_levels_forwarded(self, caplog):
        """结构化 DEBUG、INFO、ERROR 日志保持原始级别。"""
        forwarder = _make_forwarder()
        with caplog.at_level(logging.DEBUG, logger="test.subprocess"):
            for level in ("DEBUG", "INFO", "ERROR"):
                forwarder.forward(f"2024-01-15 10:30:45 [{level}] mod: {level.lower()}")
        assert [record.levelname for record in caplog.records] == [
            "DEBUG",
            "INFO",
            "ERROR",
        ]


class TestWhitespaceAndRaw:
    def test_blank_line_ignored(self, caplog):
        forwarder = _make_forwarder()
        with caplog.at_level(logging.DEBUG, logger="test.subprocess"):
            forwarder.forward("   ")
            forwarder.forward("")
            forwarder.forward("\t")
        assert caplog.records == []

    def test_raw_print_accumulated_not_forwarded(self, caplog):
        """裸 print 累积但未达阈值不输出。"""
        forwarder = _make_forwarder(threshold=10)
        with caplog.at_level(logging.DEBUG, logger="test.subprocess"):
            for _ in range(5):
                forwarder.forward("random library output")
        assert caplog.records == []

    def test_raw_print_flushed_at_threshold(self, caplog):
        forwarder = _make_forwarder(threshold=3)
        with caplog.at_level(logging.DEBUG, logger="test.subprocess"):
            for _ in range(3):
                forwarder.forward("noisy line")
        assert len(caplog.records) == 1
        assert "3" in caplog.records[0].message
        assert "已折叠" in caplog.records[0].message

    def test_flush_outputs_accumulated(self, caplog):
        forwarder = _make_forwarder(threshold=100)
        with caplog.at_level(logging.DEBUG, logger="test.subprocess"):
            forwarder.forward("noise one")
            forwarder.forward("noise two")
            forwarder.flush()
        assert len(caplog.records) == 1
        assert "2" in caplog.records[0].message
        assert "noise one" not in caplog.records[0].message
        assert "noise two" not in caplog.records[0].message

    def test_flush_with_no_accumulated_is_noop(self, caplog):
        forwarder = _make_forwarder()
        with caplog.at_level(logging.DEBUG, logger="test.subprocess"):
            forwarder.flush()
        assert caplog.records == []

    def test_structured_line_flushes_accumulated_first(self, caplog):
        """结构化行到来前先 flush 累积的裸 print。"""
        forwarder = _make_forwarder(threshold=100)
        with caplog.at_level(logging.DEBUG, logger="test.subprocess"):
            forwarder.forward("noise")
            forwarder.forward("2024-01-15 10:30:45 [INFO] mod: real")
        # 一条折叠概括 + 一条结构化转发
        assert len(caplog.records) == 2
        assert "已折叠" in caplog.records[0].message
        assert "real" in caplog.records[1].message


class TestTracebackForwarding:
    def test_traceback_start_forwarded_as_error(self, caplog):
        forwarder = _make_forwarder()
        with caplog.at_level(logging.DEBUG, logger="test.subprocess"):
            forwarder.forward("Traceback (most recent call last):")
        assert len(caplog.records) == 1
        assert caplog.records[0].levelno == logging.ERROR
        assert "Traceback" in caplog.records[0].message

    def test_exception_line_forwarded_as_error(self, caplog):
        forwarder = _make_forwarder()
        with caplog.at_level(logging.DEBUG, logger="test.subprocess"):
            forwarder.forward("ModuleNotFoundError: No module named 'x'")
        assert caplog.records[0].levelno == logging.ERROR

    def test_import_error_line(self, caplog):
        forwarder = _make_forwarder()
        with caplog.at_level(logging.DEBUG, logger="test.subprocess"):
            forwarder.forward("ImportError: cannot import name 'y'")
        assert caplog.records[0].levelno == logging.ERROR

    def test_generic_error_line(self, caplog):
        forwarder = _make_forwarder()
        with caplog.at_level(logging.DEBUG, logger="test.subprocess"):
            forwarder.forward("ValueError: bad value here")
        assert caplog.records[0].levelno == logging.ERROR


class TestSplitMixedLines:
    def test_single_line_returned_as_is(self):
        out = SubprocessLogForwarder.split_mixed_lines("single line")
        assert out == ["single line"]

    def test_empty_returns_empty(self):
        assert SubprocessLogForwarder.split_mixed_lines("") == []

    def test_splits_concatenated_datetime_lines(self):
        """PaddlePaddle 无换行拼接的多行按日期时间切分。"""
        text = "2024-01-15 10:30:45 [INFO] a: first2024-01-15 10:30:46 [INFO] b: second"
        out = SubprocessLogForwarder.split_mixed_lines(text)
        assert len(out) == 2
        assert "first" in out[0]
        assert "second" in out[1]
