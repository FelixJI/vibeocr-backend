from __future__ import annotations

from unittest.mock import MagicMock


def test_compress_uses_balanced_fast_profile(monkeypatch, tmp_path):
    """默认优化避免 garbage=4 的大型 stream 两两比较。"""
    from vibeocr.backend.services import pdf_service as module
    from vibeocr.backend.services.pdf_service import PdfService

    source = tmp_path / "source.pdf"
    source.write_bytes(b"original")
    document = MagicMock()
    document.page_count = 682

    def fake_save(filename, **_kwargs):
        from pathlib import Path

        Path(filename).write_bytes(b"optimized")

    document.save.side_effect = fake_save
    reopened = MagicMock()
    monkeypatch.setattr(module.fitz, "open", lambda _path: reopened)

    result = PdfService._compress_in_place(document, str(source), clean=False)

    assert result is reopened
    kwargs = document.save.call_args.kwargs
    assert kwargs["garbage"] == 3
    assert kwargs["deflate"] is True
    assert kwargs["use_objstms"] == 1
    assert kwargs["compression_effort"] == 1
    assert kwargs["clean"] is False
    assert source.read_bytes() == b"optimized"
    assert not (tmp_path / "source.pdf.bak").exists()
