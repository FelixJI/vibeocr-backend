"""Tests for PdfProcessAdapter: ownership, bounded proxy, transactional save."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest
from vibeocr.backend.ipc.schemas import OpenResponse, PdfDocumentMirror
from vibeocr.backend.supervisor.pdf.adapter import PdfProcessAdapter

if TYPE_CHECKING:
    from pathlib import Path


class _FakePdfChild:
    def __init__(
        self, *, save_payload: bytes = b"saved", fail_save: bool = False
    ) -> None:
        self.opened: list[str] = []
        self.renders: list[tuple[str, int]] = []
        self.saved: list[tuple[str, str]] = []
        self._save_payload = save_payload
        self._fail_save = fail_save

    def open_session(self, path: str, *, password: str | None = None) -> OpenResponse:
        sid = f"session-{len(self.opened)}"
        self.opened.append(path)
        return OpenResponse(
            session_id=sid,
            model=PdfDocumentMirror(file_path=path, pages=[]),
        )

    def render_preview(self, session_id: str, page: int, dpi: int = 150) -> bytes:
        self.renders.append((session_id, page))
        return b"\x89PNG\r\n"

    def save(self, session_id: str, target: str) -> None:
        if self._fail_save:
            raise RuntimeError("simulated save failure")
        with open(target, "wb") as fh:
            fh.write(self._save_payload)
        self.saved.append((session_id, target))


def _adapter(child: _FakePdfChild | None = None) -> PdfProcessAdapter:
    fake = child or _FakePdfChild()
    return PdfProcessAdapter(child_factory=cast("Any", lambda: fake))


# ---------------------------------------------------------------------------
# Ownership
# ---------------------------------------------------------------------------


def test_supervisor_is_sole_owner() -> None:
    adapter = _adapter()
    assert adapter.is_owner is True


def test_open_session_creates_child_once() -> None:
    fake = _FakePdfChild()
    adapter = _adapter(fake)
    sid = adapter.open_session("doc.pdf").session_id
    assert sid.startswith("session-")
    # Second call reuses the same child.
    adapter.render_preview(sid, 0)
    adapter.render_preview(sid, 1)
    assert fake.renders == [(sid, 0), (sid, 1)]


def test_stop_clears_sessions_and_child() -> None:
    adapter = _adapter()
    adapter.open_session("doc.pdf")
    adapter.stop()
    assert adapter._child is None


# ---------------------------------------------------------------------------
# Transactional save
# ---------------------------------------------------------------------------


def test_save_transactional_writes_and_replaces(tmp_path: Path) -> None:
    fake = _FakePdfChild(save_payload=b"final-content")
    adapter = _adapter(fake)
    sid = adapter.open_session("doc.pdf").session_id
    target = tmp_path / "out.pdf"
    result = adapter.save_transactional(sid, str(target))
    assert target.read_bytes() == b"final-content"
    assert result == str(target)
    # No leftover temp files.
    temps = [p for p in tmp_path.iterdir() if p.name.startswith(".out.pdf.")]
    assert temps == []


def test_save_transactional_leaves_original_on_failure(tmp_path: Path) -> None:
    fake = _FakePdfChild(fail_save=True)
    adapter = _adapter(fake)
    sid = adapter.open_session("doc.pdf").session_id
    target = tmp_path / "existing.pdf"
    target.write_bytes(b"original")
    with pytest.raises(RuntimeError, match="simulated save failure"):
        adapter.save_transactional(sid, str(target))
    # Original file untouched.
    assert target.read_bytes() == b"original"
    # No half-finished temp file left behind.
    temps = [p for p in tmp_path.iterdir() if p.name != "existing.pdf"]
    assert temps == []


def test_save_transactional_replaces_existing_atomically(tmp_path: Path) -> None:
    fake = _FakePdfChild(save_payload=b"new")
    adapter = _adapter(fake)
    sid = adapter.open_session("doc.pdf").session_id
    target = tmp_path / "out.pdf"
    target.write_bytes(b"old")
    adapter.save_transactional(sid, str(target))
    assert target.read_bytes() == b"new"
