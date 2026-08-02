"""Retention policy for terminal jobs.

Terminal jobs (and their staging) are kept for a bounded window so clients
can read partial/final results after completion, then purged. The policy is
intentionally simple and deterministic for testing.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vibeocr.runtime_contracts import TERMINAL_JOB_STATES

if TYPE_CHECKING:
    from .registry import JobRecord, JobRegistry

Clock = Callable[[], float]


@dataclass
class RetentionPolicy:
    """Track terminal-since timestamps and purge jobs past the retention TTL."""

    registry: JobRegistry
    retention_seconds: float = 3600.0
    clock: Clock = field(default=time.monotonic)
    _terminal_at: dict[str, float] = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def mark_terminal(self, record: JobRecord) -> None:
        if record.state in TERMINAL_JOB_STATES:
            with self._lock:
                self._terminal_at[record.job_id] = self.clock()

    def purge_expired(self) -> list[str]:
        now = self.clock()
        purged: list[str] = []
        with self._lock:
            expired = [
                jid
                for jid, ts in self._terminal_at.items()
                if now - ts >= self.retention_seconds
            ]
        for jid in expired:
            try:
                self.registry.purge(jid)
                purged.append(jid)
            except Exception:  # pragma: no cover - already purged
                pass
            with self._lock:
                self._terminal_at.pop(jid, None)
        return purged

    def forget(self, job_id: str) -> None:
        with self._lock:
            self._terminal_at.pop(job_id, None)


__all__ = ["RetentionPolicy"]
