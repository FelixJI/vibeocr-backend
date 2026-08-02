"""Safe WinUI cutover update sequence.

Orchestrates the permanent switch from the legacy PySide6 UI to the WinUI
shell. The sequence is fixed and every failure path lands in the bootstrapper
repair mode — it never launches the legacy UI
(``apps/vibeocr-pyside/src/vibeocr/classic/main.py``).

Steps (each delegating to an injected boundary so the sequence is testable):

1. **Verify archive** — hash/size check of the downloaded package.
2. **Stop old processes** — graceful shutdown of the running UI + worker.
3. **Atomic replace** — swap the install directory contents.
4. **Migrate** — run the idempotent config migrator.
5. **Prerequisite check** — runtime/WebView2/Python present.
6. **WinUI health handshake** — the new App publishes a ready event.
7. **Launch** — hand off to the WinUI entry.

Any failure raises :class:`CutoverError` and the caller (bootstrapper) enters
repair mode, preserving data and the diagnostics package.
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Protocol, runtime_checkable


class CutoverError(Exception):
    """A cutover step failed; the caller must enter bootstrapper repair mode."""


@runtime_checkable
class CutoverBoundary(Protocol):
    """Boundary the cutover sequence drives. Implementations wrap the real
    updater/replacer/migrator/health-check."""

    # Protocol method signatures have ``...`` bodies (no executable branch);
    # coverage's ``->exit`` branch partials on these lines are instrumentation
    # noise, not real uncovered code.
    def verify_archive(
        self, archive_path: str, expected_sha256: str
    ) -> None: ...  # pragma: no cover
    def stop_old_processes(self) -> None: ...  # pragma: no cover
    def atomic_replace(self, archive_path: str) -> None: ...  # pragma: no cover
    def migrate_config(self) -> None: ...  # pragma: no cover
    def check_prerequisites(self) -> None: ...  # pragma: no cover
    def winui_health_handshake(
        self, timeout_seconds: float
    ) -> None: ...  # pragma: no cover
    def launch_winui(self) -> None: ...  # pragma: no cover
    def enter_repair_mode(self, reason: str) -> None: ...  # pragma: no cover


@dataclasses.dataclass
class CutoverPlan:
    archive_path: str
    expected_sha256: str
    health_timeout_seconds: float = 30.0


def run_cutover(boundary: CutoverBoundary, plan: CutoverPlan) -> str:
    """Execute the cutover sequence; return 'launched' or raise CutoverError.

    On any failure the sequence calls ``enter_repair_mode`` and re-raises a
    :class:`CutoverError`. The legacy UI is never launched.
    """
    steps = [
        (
            "verify archive",
            lambda: boundary.verify_archive(plan.archive_path, plan.expected_sha256),
        ),
        ("stop old processes", boundary.stop_old_processes),
        ("atomic replace", lambda: boundary.atomic_replace(plan.archive_path)),
        ("migrate config", boundary.migrate_config),
        ("check prerequisites", boundary.check_prerequisites),
        (
            "winui health handshake",
            lambda: boundary.winui_health_handshake(plan.health_timeout_seconds),
        ),
        ("launch winui", boundary.launch_winui),
    ]
    for name, action in steps:
        try:
            action()
        except Exception as error:
            reason = f"{name} failed: {error}"
            try:
                boundary.enter_repair_mode(reason)
            except Exception:
                pass
            raise CutoverError(reason) from error
    return "launched"


def verify_sha256(data: bytes, expected: str) -> None:
    """Standalone helper: raise CutoverError on sha256 mismatch."""
    actual = hashlib.sha256(data).hexdigest()
    if actual.lower() != expected.lower():
        raise CutoverError(f"sha256 mismatch: expected {expected}, got {actual}")


__all__ = [
    "CutoverBoundary",
    "CutoverError",
    "CutoverPlan",
    "run_cutover",
    "verify_sha256",
]
