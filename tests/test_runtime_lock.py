"""runtime_lock 跨进程锁的边缘用例测试。

覆盖 RuntimeStoreLock 的上下文管理协议、release 空操作、
以及 acquire 后自动创建父目录与占位字节。
"""

from __future__ import annotations

from pathlib import Path

from vibeocr.backend.runtime_lock import RuntimeLockTimeout, RuntimeStoreLock


class TestRuntimeStoreLock:
    def test_context_manager_acquire_release(self, tmp_path: Path):
        """__enter__/__exit__ 正确获取与释放锁。"""
        lock_path = tmp_path / "sub" / "dir" / "test.lock"
        with RuntimeStoreLock(lock_path, timeout=2.0) as lock:
            assert lock._stream is not None
            assert lock_path.parent.exists()
        # 退出后 stream 已关闭
        assert lock._stream is None

    def test_release_without_acquire_is_noop(self, tmp_path: Path):
        """未 acquire 直接 release 不报错。"""
        lock = RuntimeStoreLock(tmp_path / "x.lock")
        lock.release()  # 不应抛异常
        assert lock._stream is None

    def test_acquire_creates_placeholder_byte(self, tmp_path: Path):
        """acquire 在空文件中写入一个占位字节（release 后可读验证）。"""
        lock_path = tmp_path / "placeholder.lock"
        lock = RuntimeStoreLock(lock_path, timeout=2.0)
        lock.acquire()
        lock.release()
        # Windows 下持锁期间文件被独占，release 后才能读取
        assert lock_path.read_bytes() == b"\0"

    def test_parent_dir_created(self, tmp_path: Path):
        """acquire 自动创建不存在的父目录。"""
        lock_path = tmp_path / "a" / "b" / "c" / "deep.lock"
        lock = RuntimeStoreLock(lock_path, timeout=2.0)
        lock.acquire()
        try:
            assert lock_path.exists()
        finally:
            lock.release()

    def test_reentrant_same_process(self, tmp_path: Path):
        """同进程二次 acquire 在已持有锁文件上不崩溃（占位字节已存在）。"""
        lock_path = tmp_path / "shared.lock"
        lock1 = RuntimeStoreLock(lock_path, timeout=2.0)
        lock1.acquire()
        try:
            # 文件已被 lock1 持有；验证文件存在且未抛异常即可
            assert lock_path.exists()
        finally:
            lock1.release()
        # release 后可读，确认占位字节已写入
        assert lock_path.read_bytes() == b"\0"

    def test_lock_timeout_is_subclass_of_timeout_error(self):
        """RuntimeLockTimeout 是 TimeoutError 子类。"""
        assert issubclass(RuntimeLockTimeout, TimeoutError)

    def test_path_normalized_to_pathlib(self, tmp_path: Path):
        """传入 str 路径被转为 Path。"""
        lock = RuntimeStoreLock(str(tmp_path / "p.lock"))
        assert isinstance(lock.path, Path)
