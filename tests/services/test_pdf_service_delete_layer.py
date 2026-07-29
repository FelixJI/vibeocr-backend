"""delete_text_layers 词级 redact + 循环验证至清零。"""

import fitz

from vibeocr.backend.models.pdf_document import PdfDocument, PdfPageInfo
from vibeocr.backend.services.pdf_service import PdfService


def _make_pdf_with_text(path, texts):
    """创建单页含多段文字的 PDF。"""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    for i, text in enumerate(texts):
        page.insert_text((72, 72 + i * 30), text, fontsize=12)
    doc.save(str(path))
    doc.close()
    return path


class TestDeleteTextLayersWordLevel:
    def test_returns_tuple_with_residual_flag(self):
        """返回 (deleted_count, rounds_used, has_residual) 三元组。"""
        import pathlib
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "t.pdf"
            _make_pdf_with_text(path, ["Hello World", "Second Line"])
            doc = fitz.open(str(path))
            pdf_doc = PdfDocument()
            pdf_doc.pages = [PdfPageInfo(page_index=0, has_text_layer=True)]
            result = PdfService.delete_text_layers(doc, pdf_doc, 0)
            assert isinstance(result, tuple)
            assert len(result) == 3
            deleted, rounds, residual = result
            assert deleted > 0
            assert rounds >= 1
            assert residual is False
            doc.close()

    def test_clears_all_text_no_residual(self):
        """删除后该页 get_text() 应为空。"""
        import pathlib
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "t.pdf"
            _make_pdf_with_text(path, ["Line A", "Line B", "Line C"])
            doc = fitz.open(str(path))
            pdf_doc = PdfDocument()
            pdf_doc.pages = [PdfPageInfo(page_index=0, has_text_layer=True)]
            PdfService.delete_text_layers(doc, pdf_doc, 0)
            assert doc[0].get_text().strip() == ""
            doc.close()

    def test_page_without_text_returns_zero(self):
        """无文字页返回 (0, 0, False)，不做 redact。"""
        doc = fitz.open()
        doc.new_page(width=612, height=792)
        pdf_doc = PdfDocument()
        pdf_doc.pages = [PdfPageInfo(page_index=0)]
        result = PdfService.delete_text_layers(doc, pdf_doc, 0)
        assert result == (0, 0, False)
        doc.close()

    def test_clears_page_info_flags(self):
        """删除后 has_text_layer=False, text_layers=[], ocr_text_blocks=[]。"""
        import pathlib
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "t.pdf"
            _make_pdf_with_text(path, ["text"])
            doc = fitz.open(str(path))
            pdf_doc = PdfDocument()
            info = PdfPageInfo(page_index=0, has_text_layer=True)
            pdf_doc.pages = [info]
            PdfService.delete_text_layers(doc, pdf_doc, 0)
            assert info.has_text_layer is False
            assert info.text_layers == []
            assert info.ocr_text_blocks == []
            doc.close()
