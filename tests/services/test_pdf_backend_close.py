"""测试 PDF backend session close 与 fitz_lock 同步。

根因：SessionRegistry.remove() 从字典移除后直接 doc.close()，没有获取
fitz_lock。close 可与持锁或即将持锁的 render/load/mutate 请求并发，
导致"从另一线程脚下关闭文档"，可能 native crash。
"""

import threading
import time
from unittest.mock import MagicMock


class TestSessionCloseSync:
    """BackendSession close 必须等待 active op 完成并在 fitz_lock 内 close。"""

    def test_remove_waits_for_active_op(self):
        """remove() 等待 active_ops 归零后再 close doc"""
        from vibeocr.backend.services.pdf_backend_process import (
            BackendSession,
            SessionRegistry,
        )

        reg = SessionRegistry()
        mock_doc = MagicMock()
        session = BackendSession(
            session_id="test1",
            file_path="/fake.pdf",
            doc=mock_doc,
            pdf_document=MagicMock(),
        )
        reg._sessions["test1"] = session

        # 模拟有一个活跃操作
        session.active_ops = 1
        released = threading.Event()

        def release_after():
            time.sleep(0.2)
            with session._ops_cond:
                session.active_ops = 0
                session._ops_cond.notify_all()
            released.set()

        threading.Thread(target=release_after, daemon=True).start()

        start = time.monotonic()
        reg.remove("test1")
        elapsed = time.monotonic() - start

        # 应等待至少 0.2s（等 active_ops 归零）
        assert elapsed >= 0.15, f"remove 未等待 active op，耗时 {elapsed:.3f}s"
        assert released.is_set(), "remove 应等到 active op 释放"
        mock_doc.close.assert_called_once()

    def test_remove_acquires_fitz_lock_before_close(self):
        """remove() 在 fitz_lock 内 close doc"""
        from vibeocr.backend.services.pdf_backend_process import (
            BackendSession,
            SessionRegistry,
        )

        reg = SessionRegistry()
        mock_doc = MagicMock()
        session = BackendSession(
            session_id="test2",
            file_path="/fake.pdf",
            doc=mock_doc,
            pdf_document=MagicMock(),
        )
        reg._sessions["test2"] = session

        # 模拟 fitz_lock 被另一线程持有
        session.fitz_lock.acquire()
        close_thread_done = threading.Event()

        def do_remove():
            reg.remove("test2")
            close_thread_done.set()

        t = threading.Thread(target=do_remove)
        t.start()
        time.sleep(0.1)
        # remove 应阻塞（fitz_lock 被持有）
        assert not close_thread_done.is_set(), "remove 应等待 fitz_lock 释放"
        session.fitz_lock.release()
        t.join(timeout=5)
        assert close_thread_done.is_set(), "remove 应在 fitz_lock 释放后完成"
        mock_doc.close.assert_called_once()

    def test_remove_sets_closing_state(self):
        """remove() 设置 CLOSING 状态，拒绝新操作"""
        from vibeocr.backend.services.pdf_backend_process import (
            BackendSession,
            SessionRegistry,
        )

        reg = SessionRegistry()
        mock_doc = MagicMock()
        session = BackendSession(
            session_id="test3",
            file_path="/fake.pdf",
            doc=mock_doc,
            pdf_document=MagicMock(),
        )
        reg._sessions["test3"] = session

        reg.remove("test3")
        assert session.state == "CLOSED"

    def test_get_rejects_closing_session(self):
        """CLOSING 状态的 session 被 get 时抛 409"""
        from unittest.mock import MagicMock as Mock

        from fastapi import HTTPException
        from vibeocr.backend.services.pdf_backend_process import (
            BackendSession,
            SessionRegistry,
        )

        reg = SessionRegistry()
        mock_doc = Mock()
        session = BackendSession(
            session_id="test4",
            file_path="/fake.pdf",
            doc=mock_doc,
            pdf_document=Mock(),
        )
        session.state = "CLOSING"
        reg._sessions["test4"] = session

        try:
            reg.get("test4")
            raise AssertionError("应抛 HTTPException 409")
        except HTTPException as e:
            assert e.status_code == 409
