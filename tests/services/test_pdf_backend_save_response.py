from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock


def test_save_returns_status_only_diff(monkeypatch, tmp_path):
    """保存响应不得携带整份 OCR 文档，避免大文档超过控制帧上限。"""
    from vibeocr.backend.ipc.schemas import SaveRequest
    from vibeocr.backend.models.pdf_document import PdfDocument, PdfPageInfo
    from vibeocr.backend.services import pdf_backend_process as backend
    from vibeocr.backend.services.pdf_service import SaveResult

    output = tmp_path / "saved.pdf"
    document = PdfDocument(file_path=str(output))
    document.pages = [PdfPageInfo(page_index=i) for i in range(700)]
    session = SimpleNamespace(
        doc=MagicMock(),
        pdf_document=document,
        fitz_lock=threading.RLock(),
    )
    registry = MagicMock()
    registry.get.return_value = session
    monkeypatch.setattr(backend, "_get_registry", lambda: registry)
    monkeypatch.setattr(
        backend.PdfService,
        "save_with_rewrite",
        lambda *_args, **_kwargs: SaveResult([], str(output)),
    )

    response = backend.save(
        "sid",
        SaveRequest(path=None, pdf_settings={}, rewrite_text_layers=False),
    )

    assert response.path == str(output)
    assert response.diff.full_model is None
    assert response.diff.replaced_pages == []
    assert response.diff.modified_flag is False
    assert response.diff.structural_flag is False
