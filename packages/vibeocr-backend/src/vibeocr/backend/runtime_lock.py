"""Cross-process locks for runtime installation, verification and GC."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

if TYPE_CHECKING:
    from types import TracebackType


class RuntimeLockTimeout(TimeoutError):
    pass


class RuntimeStoreLock:
    """An OS-backed exclusive lock represented inside the portable store."""

    def __init__(self, path: str | Path, *, timeout: float = 60.0) -> None:
        self.path = Path(path)
        self.timeout = timeout
        self._stream: BinaryIO | None = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stream = self.path.open("a+b")
        if stream.seek(0, os.SEEK_END) == 0:
            stream.write(b"\0")
            stream.flush()
        deadline = time.monotonic() + max(self.timeout, 0.0)
        while True:
            try:
                self._lock(stream)
                self._stream = stream
                return
            except OSError:
                if time.monotonic() >= deadline:
                    stream.close()
                    raise RuntimeLockTimeout(
                        f"timed out acquiring runtime lock: {self.path}"
                    ) from None
                time.sleep(0.05)

    @staticmethod
    def _lock(stream: BinaryIO) -> None:
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:  # pragma: no cover - Windows is the release platform.
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    @staticmethod
    def _unlock(stream: BinaryIO) -> None:
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:  # pragma: no cover
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def release(self) -> None:
        stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            self._unlock(stream)
        finally:
            stream.close()

    def __enter__(self) -> RuntimeStoreLock:
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()


__all__ = ["RuntimeLockTimeout", "RuntimeStoreLock"]
