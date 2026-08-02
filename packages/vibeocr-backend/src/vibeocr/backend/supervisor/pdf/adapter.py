"""PdfProcessAdapter: supervisor-owned model-free PDF child process.

Plan §4 Phase 6 goals addressed by this seam:

* The supervisor is the sole owner of the PyMuPDF child process; the GUI no
  longer instantiates ``PdfBackendClient``.
* Quick PDF session operations (open/render/mutate/save/text-layer/cancel) are
  proxied through the supervisor with bounded behaviour.
* PDF editing remains a bounded session API, while rendered pages enter the
  same Supervisor recognition-job interface as every other OCR request.
* Transactional final save (temp file + fsync + atomic replace) so a wedged
  save never publishes a half-finished file.

The actual PyMuPDF calls reuse ``services/pdf_backend_process.py``; the
adapter is a thin ownership wrapper over the long-lived
:class:`~vibeocr.backend.services.pdf_backend_client.PdfBackendClient` singleton.

Method names and DTOs (``vibeocr.backend.ipc.schemas``) are identical to the legacy
client so the supervisor's v2 routes can delegate verbatim and the GUI-side
swap (in :mod:`vibeocr.classic.pdf_client`) needs no translation.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from vibeocr.backend.ipc.schemas import (
        DetectTextLayersResponse,
        MutateResponse,
        OpenResponse,
        PdfDocumentMirror,
        ProgressEvent,
        SaveResponse,
    )


class _PdfChildLike(Protocol):
    """The full PdfBackendClient surface the adapter proxies.

    Every attribute is typed loosely: the adapter's job is ownership + bounded
    proxy, not contract enforcement (DTO validation happens in the v2 routes).
    """

    def open_session(self, path: str) -> Any: ...

    def close_session(self, session_id: str) -> None: ...

    def get_model(self, session_id: str) -> Any: ...

    def load_stream(self, session_id: str) -> Any: ...

    def render_thumbnail(
        self, session_id: str, page: int, size: int = 160
    ) -> bytes: ...

    def render_preview(self, session_id: str, page: int, dpi: int = 150) -> bytes: ...

    def detect_text_layers(self, session_id: str, page: int) -> Any: ...

    def rotate(self, session_id: str, pages: list[int], angle: int) -> Any: ...

    def delete_pages(self, session_id: str, pages: list[int]) -> Any: ...

    def insert_blank(
        self,
        session_id: str,
        after_index: int,
        width: float = 612.0,
        height: float = 792.0,
    ) -> Any: ...

    def insert_from(
        self, session_id: str, source_path: str, after_index: int
    ) -> Any: ...

    def move_page(self, session_id: str, from_index: int, to_index: int) -> Any: ...

    def reorder(self, session_id: str, new_order: list[int]) -> Any: ...

    def add_text_layer(
        self,
        session_id: str,
        page: int,
        ocr_result: dict[str, Any],
        pdf_settings: dict[str, Any] | None = None,
        overwrite: bool = False,
    ) -> Any: ...

    def add_text_layer_batch(
        self,
        session_id: str,
        pages_data: list[dict[str, Any]],
        pdf_settings: dict[str, Any] | None = None,
        overwrite: bool = False,
        save: bool = False,
    ) -> Any: ...

    def rewrite_text_layer(
        self,
        session_id: str,
        page: int,
        text_blocks: list[Any],
        preproc_angle: int = 0,
        pdf_settings: dict[str, Any] | None = None,
    ) -> Any: ...

    def update_block_text(
        self, session_id: str, page: int, block_index: int, new_text: str
    ) -> Any: ...

    def delete_text_layers_stream(self, session_id: str, pages: list[int]) -> Any: ...

    def save(
        self,
        session_id: str,
        path: str | None = None,
        pdf_settings: dict[str, Any] | None = None,
        *,
        rewrite_text_layers: bool = True,
    ) -> Any: ...

    def cancel(self, session_id: str) -> None: ...

    def reset_cancel(self, session_id: str) -> None: ...

    def start(self) -> None: ...

    def stop(self) -> None: ...


@dataclass
class PdfProcessAdapter:
    """Supervisor-owned PDF child adapter.

    ``child_factory`` returns the long-lived client (the legacy
    ``PdfBackendClient.instance()`` singleton in production; a fake in tests).
    The adapter owns its lifetime: ``ensure_started`` materialises it on first
    use, ``stop`` tears it down at supervisor shutdown.
    """

    child_factory: Callable[[], _PdfChildLike]
    _child: _PdfChildLike | None = None
    _sessions: set[str] = field(default_factory=set)

    # ------------------------------------------------------------------
    # Ownership
    # ------------------------------------------------------------------

    def ensure_started(self) -> _PdfChildLike:
        if self._child is None:
            self._child = self.child_factory()
            # PdfBackendClient.start() is idempotent + thread-safe; calling it
            # here ensures the FastAPI child subprocess is up before any op.
            start = getattr(self._child, "start", None)
            if callable(start):
                start()
        return self._child

    def stop(self) -> None:
        """Terminate the PDF child process."""
        child = self._child
        self._child = None
        self._sessions.clear()
        if child is not None:
            stop = getattr(child, "stop", None)
            if callable(stop):
                try:
                    stop()
                except Exception:
                    # Best-effort shutdown: a wedged child must not block
                    # supervisor exit. The Job Object / process-group kill in
                    # the launcher handles forceful termination.
                    pass

    @property
    def is_owner(self) -> bool:
        """The supervisor is the sole owner of the PDF child."""
        return True

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def open_session(self, path: str) -> OpenResponse:
        child = self.ensure_started()
        result = child.open_session(path)
        # OpenResponse carries session_id + model; track id for shutdown.
        sid = getattr(result, "session_id", None)
        if sid is not None:
            self._sessions.add(sid)
        return result

    def close_session(self, session_id: str) -> None:
        child = self.ensure_started()
        child.close_session(session_id)
        self._sessions.discard(session_id)

    def get_model(self, session_id: str) -> PdfDocumentMirror:
        return self.ensure_started().get_model(session_id)

    def load_stream(self, session_id: str) -> Iterator[ProgressEvent]:
        return self.ensure_started().load_stream(session_id)

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render_thumbnail(self, session_id: str, page: int, size: int = 160) -> bytes:
        return self.ensure_started().render_thumbnail(session_id, page, size=size)

    def render_preview(self, session_id: str, page: int, dpi: int = 150) -> bytes:
        return self.ensure_started().render_preview(session_id, page, dpi=dpi)

    def detect_text_layers(
        self, session_id: str, page: int
    ) -> DetectTextLayersResponse:
        return self.ensure_started().detect_text_layers(session_id, page)

    # ------------------------------------------------------------------
    # Page mutations (rotate/delete/insert/move/reorder)
    # ------------------------------------------------------------------

    def rotate(self, session_id: str, pages: list[int], angle: int) -> MutateResponse:
        return self.ensure_started().rotate(session_id, pages, angle)

    def delete_pages(self, session_id: str, pages: list[int]) -> MutateResponse:
        return self.ensure_started().delete_pages(session_id, pages)

    def insert_blank(
        self,
        session_id: str,
        after_index: int,
        width: float = 612.0,
        height: float = 792.0,
    ) -> MutateResponse:
        return self.ensure_started().insert_blank(
            session_id, after_index, width, height
        )

    def insert_from(
        self, session_id: str, source_path: str, after_index: int
    ) -> MutateResponse:
        return self.ensure_started().insert_from(session_id, source_path, after_index)

    def move_page(
        self, session_id: str, from_index: int, to_index: int
    ) -> MutateResponse:
        return self.ensure_started().move_page(session_id, from_index, to_index)

    def reorder(self, session_id: str, new_order: list[int]) -> MutateResponse:
        return self.ensure_started().reorder(session_id, new_order)

    # ------------------------------------------------------------------
    # Text layer
    # ------------------------------------------------------------------

    def add_text_layer(
        self,
        session_id: str,
        page: int,
        ocr_result: dict[str, Any],
        pdf_settings: dict[str, Any] | None = None,
        overwrite: bool = False,
    ) -> MutateResponse:
        return self.ensure_started().add_text_layer(
            session_id, page, ocr_result, pdf_settings, overwrite
        )

    def add_text_layer_batch(
        self,
        session_id: str,
        pages_data: list[dict[str, Any]],
        pdf_settings: dict[str, Any] | None = None,
        overwrite: bool = False,
        save: bool = False,
    ) -> MutateResponse:
        return self.ensure_started().add_text_layer_batch(
            session_id, pages_data, pdf_settings, overwrite, save
        )

    def rewrite_text_layer(
        self,
        session_id: str,
        page: int,
        text_blocks: list[Any],
        preproc_angle: int = 0,
        pdf_settings: dict[str, Any] | None = None,
    ) -> MutateResponse:
        return self.ensure_started().rewrite_text_layer(
            session_id, page, text_blocks, preproc_angle, pdf_settings
        )

    def update_block_text(
        self,
        session_id: str,
        page: int,
        block_index: int,
        new_text: str,
    ) -> MutateResponse:
        return self.ensure_started().update_block_text(
            session_id, page, block_index, new_text
        )

    def delete_text_layers_stream(
        self, session_id: str, pages: list[int]
    ) -> Iterator[ProgressEvent]:
        return self.ensure_started().delete_text_layers_stream(session_id, pages)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(
        self,
        session_id: str,
        path: str | None = None,
        pdf_settings: dict[str, Any] | None = None,
        *,
        rewrite_text_layers: bool = True,
    ) -> SaveResponse:
        return self.ensure_started().save(
            session_id,
            path,
            pdf_settings,
            rewrite_text_layers=rewrite_text_layers,
        )

    def save_transactional(self, session_id: str, target_path: str) -> str:
        """Save the session to ``target_path`` atomically.

        Writes to a temp file in the same directory, fsyncs, then renames over
        the target. On any error the original file is left untouched and the
        temp file is removed. This guarantees no half-finished file is ever
        published as a successful save.
        """
        child = self.ensure_started()
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
        )
        os.close(fd)
        try:
            child.save(session_id, tmp_name)
            self._fsync_path(tmp_name)
            Path(tmp_name).replace(target)
        except Exception:
            try:
                Path(tmp_name).unlink()
            except OSError:
                pass
            raise
        return str(target)

    # ------------------------------------------------------------------
    # Cancel
    # ------------------------------------------------------------------

    def cancel(self, session_id: str) -> None:
        self.ensure_started().cancel(session_id)

    def reset_cancel(self, session_id: str) -> None:
        self.ensure_started().reset_cancel(session_id)

    @staticmethod
    def _fsync_path(path: str) -> None:
        # Open read/write so the fd is fsync-able on Windows (read-only fds
        # raise EBADF there).
        with open(path, "r+b") as fh:
            os.fsync(fh.fileno())


__all__ = ["PdfProcessAdapter"]
