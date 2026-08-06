"""Tests for the bounded-proxy surface of PdfProcessAdapter.

Existing test_pdf_adapter.py covers ownership + transactional save; this file
targets the proxy methods left behind (close/model/load/render/detect/mutate/
text-layer/save/cancel) and the ownership branches (start idempotency,
stop-with-error, save_transactional unlink failure).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from vibeocr.backend.ipc.schemas import (
    DetectTextLayersResponse,
    ModelDiff,
    MutateResponse,
    OpenResponse,
    PdfDocumentMirror,
    ProgressEvent,
    ProgressPhase,
    SaveResponse,
)
from vibeocr.backend.supervisor.pdf.adapter import PdfProcessAdapter


class _FullPdfChild:
    """Records every call and returns canned DTOs for the full surface."""

    def __init__(self, *, fail_save: bool = False, fail_unlink: bool = False) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []
        self.start_calls = 0
        self.stop_calls = 0
        self._fail_save = fail_save
        self._fail_unlink = fail_unlink  # raise on save to exercise unlink path

    def _record(self, name: str, args: tuple, kwargs: dict) -> None:
        self.calls.append((name, args, kwargs))

    def start(self) -> None:
        self.start_calls += 1

    def stop(self) -> None:
        self.stop_calls += 1

    def open_session(self, path: str) -> OpenResponse:
        self._record("open_session", (path,), {})
        return OpenResponse(
            session_id="sid-1",
            model=PdfDocumentMirror(file_path=path, pages=[]),
        )

    def close_session(self, session_id: str) -> None:
        self._record("close_session", (session_id,), {})

    def get_model(self, session_id: str) -> PdfDocumentMirror:
        self._record("get_model", (session_id,), {})
        return PdfDocumentMirror(file_path="doc.pdf", pages=[])

    def load_stream(self, session_id: str):  # type: ignore[no-untyped-def]
        self._record("load_stream", (session_id,), {})
        yield ProgressEvent(phase=ProgressPhase.LOAD, current=0, total=1, message="ok")

    def render_thumbnail(self, session_id: str, page: int, size: int = 160) -> bytes:
        self._record("render_thumbnail", (session_id, page), {"size": size})
        return b"\x89PNGthumb"

    def render_preview(self, session_id: str, page: int, dpi: int = 150) -> bytes:
        self._record("render_preview", (session_id, page), {"dpi": dpi})
        return b"\x89PNGpreview"

    def detect_text_layers(
        self, session_id: str, page: int
    ) -> DetectTextLayersResponse:
        self._record("detect_text_layers", (session_id, page), {})
        return DetectTextLayersResponse(text_layers=[])

    def rotate(self, session_id: str, pages: list[int], angle: int) -> MutateResponse:
        self._record("rotate", (session_id, pages, angle), {})
        return MutateResponse(diff=ModelDiff())

    def delete_pages(self, session_id: str, pages: list[int]) -> MutateResponse:
        self._record("delete_pages", (session_id, pages), {})
        return MutateResponse(diff=ModelDiff())

    def insert_blank(
        self,
        session_id: str,
        after_index: int,
        width: float = 612.0,
        height: float = 792.0,
    ) -> MutateResponse:
        self._record(
            "insert_blank",
            (session_id, after_index),
            {"width": width, "height": height},
        )
        return MutateResponse(diff=ModelDiff())

    def insert_from(
        self, session_id: str, source_path: str, after_index: int
    ) -> MutateResponse:
        self._record("insert_from", (session_id, source_path, after_index), {})
        return MutateResponse(diff=ModelDiff())

    def move_page(
        self, session_id: str, from_index: int, to_index: int
    ) -> MutateResponse:
        self._record("move_page", (session_id, from_index, to_index), {})
        return MutateResponse(diff=ModelDiff())

    def reorder(self, session_id: str, new_order: list[int]) -> MutateResponse:
        self._record("reorder", (session_id, new_order), {})
        return MutateResponse(diff=ModelDiff())

    def add_text_layer(
        self,
        session_id: str,
        page: int,
        ocr_result: dict,
        pdf_settings=None,
        overwrite=False,
    ) -> MutateResponse:
        self._record(
            "add_text_layer",
            (session_id, page, ocr_result),
            {"pdf_settings": pdf_settings, "overwrite": overwrite},
        )
        return MutateResponse(diff=ModelDiff())

    def add_text_layer_batch(
        self,
        session_id: str,
        pages_data: list,
        pdf_settings=None,
        overwrite=False,
        save=False,
    ) -> MutateResponse:
        self._record(
            "add_text_layer_batch",
            (session_id, pages_data),
            {"pdf_settings": pdf_settings, "overwrite": overwrite, "save": save},
        )
        return MutateResponse(diff=ModelDiff())

    def rewrite_text_layer(
        self,
        session_id: str,
        page: int,
        text_blocks: list,
        preproc_angle: int = 0,
        pdf_settings=None,
    ) -> MutateResponse:
        self._record(
            "rewrite_text_layer",
            (session_id, page, text_blocks),
            {"preproc_angle": preproc_angle, "pdf_settings": pdf_settings},
        )
        return MutateResponse(diff=ModelDiff())

    def update_block_text(
        self, session_id: str, page: int, block_index: int, new_text: str
    ) -> MutateResponse:
        self._record("update_block_text", (session_id, page, block_index, new_text), {})
        return MutateResponse(diff=ModelDiff())

    def delete_text_layers_stream(self, session_id: str, pages: list):  # type: ignore[no-untyped-def]
        self._record("delete_text_layers_stream", (session_id, pages), {})
        yield ProgressEvent(phase=ProgressPhase.DELETE, current=1, total=1)

    def save(
        self, session_id: str, path=None, pdf_settings=None, *, rewrite_text_layers=True
    ):  # type: ignore[no-untyped-def]
        self._record(
            "save",
            (session_id, path, pdf_settings),
            {"rewrite_text_layers": rewrite_text_layers},
        )
        if self._fail_save or self._fail_unlink:
            raise RuntimeError("save failed")
        if path is not None:
            with open(path, "wb") as fh:
                fh.write(b"saved")
        return SaveResponse(path=path or "out.pdf", diff=ModelDiff())

    def cancel(self, session_id: str) -> None:
        self._record("cancel", (session_id,), {})

    def reset_cancel(self, session_id: str) -> None:
        self._record("reset_cancel", (session_id,), {})


def _adapter(child: _FullPdfChild | None = None) -> PdfProcessAdapter:
    fake = child or _FullPdfChild()
    return PdfProcessAdapter(child_factory=cast("Any", lambda: fake))


# ---------------------------------------------------------------------------
# Ownership: ensure_started idempotency + stop-with-error
# ---------------------------------------------------------------------------


def test_ensure_started_invokes_start_once_per_child() -> None:
    fake = _FullPdfChild()
    adapter = _adapter(fake)
    adapter.ensure_started()
    adapter.ensure_started()  # second call is a no-op
    assert fake.start_calls == 1


def test_stop_swallows_child_stop_error() -> None:
    class _BoomStop(_FullPdfChild):
        def stop(self) -> None:
            raise RuntimeError("wedged")

    fake = _BoomStop()
    adapter = _adapter(fake)
    adapter.open_session("doc.pdf")  # materialise the child
    adapter.stop()  # must not raise
    assert adapter._child is None


def test_stop_is_noop_when_child_never_materialised() -> None:
    adapter = _adapter()
    adapter.stop()  # must not raise
    assert adapter._child is None


# ---------------------------------------------------------------------------
# Session lifecycle: close / model / load_stream
# ---------------------------------------------------------------------------


def test_close_session_proxies_and_discards_id() -> None:
    fake = _FullPdfChild()
    adapter = _adapter(fake)
    adapter.open_session("doc.pdf")
    adapter.close_session("sid-1")
    assert fake.calls[-1][0] == "close_session"
    assert "sid-1" not in adapter._sessions


def test_get_model_proxies() -> None:
    fake = _FullPdfChild()
    adapter = _adapter(fake)
    out = adapter.get_model("sid-1")
    assert isinstance(out, PdfDocumentMirror)
    assert fake.calls[-1][0] == "get_model"


def test_load_stream_proxies() -> None:
    fake = _FullPdfChild()
    adapter = _adapter(fake)
    events = list(adapter.load_stream("sid-1"))
    assert len(events) == 1
    assert events[0].phase is ProgressPhase.LOAD


# ---------------------------------------------------------------------------
# Render / detect
# ---------------------------------------------------------------------------


def test_render_thumbnail_proxies_with_size() -> None:
    fake = _FullPdfChild()
    adapter = _adapter(fake)
    out = adapter.render_thumbnail("sid-1", 0, size=99)
    assert out.startswith(b"\x89PNG")
    name, _args, kwargs = fake.calls[-1]
    assert name == "render_thumbnail"
    assert kwargs == {"size": 99}


def test_render_preview_proxies_with_dpi() -> None:
    fake = _FullPdfChild()
    adapter = _adapter(fake)
    out = adapter.render_preview("sid-1", 1, dpi=200)
    assert out.startswith(b"\x89PNG")
    name, _args, kwargs = fake.calls[-1]
    assert name == "render_preview"
    assert kwargs == {"dpi": 200}


def test_detect_text_layers_proxies() -> None:
    fake = _FullPdfChild()
    adapter = _adapter(fake)
    out = adapter.detect_text_layers("sid-1", 0)
    assert isinstance(out, DetectTextLayersResponse)


# ---------------------------------------------------------------------------
# Page mutations
# ---------------------------------------------------------------------------


def test_rotate_proxies() -> None:
    fake = _FullPdfChild()
    adapter = _adapter(fake)
    out = adapter.rotate("sid-1", [0, 1], 90)
    assert isinstance(out, MutateResponse)
    assert fake.calls[-1][0] == "rotate"


def test_delete_pages_proxies() -> None:
    adapter = _adapter()
    assert isinstance(adapter.delete_pages("sid-1", [0]), MutateResponse)


def test_insert_blank_proxies_with_dimensions() -> None:
    fake = _FullPdfChild()
    adapter = _adapter(fake)
    adapter.insert_blank("sid-1", 0, width=100.0, height=200.0)
    name, _args, kwargs = fake.calls[-1]
    assert name == "insert_blank"
    assert kwargs == {"width": 100.0, "height": 200.0}


def test_insert_from_proxies() -> None:
    fake = _FullPdfChild()
    adapter = _adapter(fake)
    adapter.insert_from("sid-1", "src.pdf", 1)
    name, args, _kwargs = fake.calls[-1]
    assert name == "insert_from"
    assert args == ("sid-1", "src.pdf", 1)


def test_move_page_proxies() -> None:
    fake = _FullPdfChild()
    adapter = _adapter(fake)
    adapter.move_page("sid-1", 0, 1)
    assert fake.calls[-1][0] == "move_page"


def test_reorder_proxies() -> None:
    fake = _FullPdfChild()
    adapter = _adapter(fake)
    adapter.reorder("sid-1", [1, 0])
    assert fake.calls[-1][0] == "reorder"


# ---------------------------------------------------------------------------
# Text layer
# ---------------------------------------------------------------------------


def test_add_text_layer_proxies_with_overwrite() -> None:
    fake = _FullPdfChild()
    adapter = _adapter(fake)
    adapter.add_text_layer("sid-1", 0, {"text": "x"}, overwrite=True)
    name, _args, kwargs = fake.calls[-1]
    assert name == "add_text_layer"
    assert kwargs["overwrite"] is True


def test_add_text_layer_batch_proxies_with_save() -> None:
    fake = _FullPdfChild()
    adapter = _adapter(fake)
    adapter.add_text_layer_batch("sid-1", [{"page": 0}], save=True)
    name, _args, kwargs = fake.calls[-1]
    assert name == "add_text_layer_batch"
    assert kwargs["save"] is True


def test_rewrite_text_layer_proxies_with_preproc_angle() -> None:
    fake = _FullPdfChild()
    adapter = _adapter(fake)
    adapter.rewrite_text_layer("sid-1", 0, [], preproc_angle=90)
    name, _args, kwargs = fake.calls[-1]
    assert name == "rewrite_text_layer"
    assert kwargs["preproc_angle"] == 90


def test_update_block_text_proxies() -> None:
    fake = _FullPdfChild()
    adapter = _adapter(fake)
    adapter.update_block_text("sid-1", 0, 1, "edited")
    name, args, _kwargs = fake.calls[-1]
    assert name == "update_block_text"
    assert args == ("sid-1", 0, 1, "edited")


def test_delete_text_layers_stream_proxies() -> None:
    fake = _FullPdfChild()
    adapter = _adapter(fake)
    events = list(adapter.delete_text_layers_stream("sid-1", [0, 1]))
    assert len(events) == 1
    assert events[0].phase is ProgressPhase.DELETE


# ---------------------------------------------------------------------------
# Save / cancel / reset_cancel
# ---------------------------------------------------------------------------


def test_save_proxies_with_rewrite_flag(tmp_path: Path) -> None:
    fake = _FullPdfChild()
    adapter = _adapter(fake)
    output_path = tmp_path / "out.pdf"

    adapter.save("sid-1", str(output_path), rewrite_text_layers=False)

    name, args, kwargs = fake.calls[-1]
    assert name == "save"
    assert args[1] == str(output_path)
    assert kwargs == {"rewrite_text_layers": False}
    assert output_path.read_bytes() == b"saved"


def test_cancel_proxies() -> None:
    fake = _FullPdfChild()
    adapter = _adapter(fake)
    adapter.cancel("sid-1")
    assert fake.calls[-1][0] == "cancel"


def test_reset_cancel_proxies() -> None:
    fake = _FullPdfChild()
    adapter = _adapter(fake)
    adapter.reset_cancel("sid-1")
    assert fake.calls[-1][0] == "reset_cancel"


# ---------------------------------------------------------------------------
# save_transactional: unlink-on-failure path
# ---------------------------------------------------------------------------


def test_save_transactional_unlinks_tempfile_on_failure(tmp_path: Path) -> None:
    """A failing save leaves no temp files behind (lines 174-180)."""
    fake = _FullPdfChild(fail_save=True)
    adapter = _adapter(fake)
    adapter.open_session("doc.pdf")
    target = tmp_path / "out.pdf"
    target.write_bytes(b"original")
    with pytest.raises(RuntimeError, match="save failed"):
        adapter.save_transactional("sid-1", str(target))
    # Original untouched, no leftover temp files.
    assert target.read_bytes() == b"original"
    temps = [p for p in tmp_path.iterdir() if p.name.startswith(".out.pdf.")]
    assert temps == []


def test_save_transactional_tolerates_unlink_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the cleanup unlink itself raises OSError, the original error still
    propagates (lines 365-366)."""
    fake = _FullPdfChild(fail_save=True)
    adapter = _adapter(fake)
    adapter.open_session("doc.pdf")
    target = tmp_path / "out.pdf"

    real_unlink = Path.unlink

    def raising_unlink(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        # Only raise for temp files (those starting with the dot prefix).
        if self.name.startswith(".out.pdf."):
            raise OSError("unlink denied")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", raising_unlink)
    # The save_transactional must still re-raise the original RuntimeError.
    with pytest.raises(RuntimeError, match="save failed"):
        adapter.save_transactional("sid-1", str(target))
