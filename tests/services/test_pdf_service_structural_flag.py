"""结构性操作应置 has_structural_change=True，纯文字层/旋转操作不置。"""

import fitz

from vibeocr.backend.models.pdf_document import PdfDocument, PdfPageInfo
from vibeocr.backend.services.pdf_service import PdfService


def _open_doc(num_pages=3):
    doc = fitz.open()
    for _ in range(num_pages):
        doc.new_page(width=612, height=792)
    pdf_doc = PdfDocument()
    pdf_doc.pages = [PdfPageInfo(page_index=i) for i in range(num_pages)]
    return doc, pdf_doc


class TestStructuralFlag:
    def test_delete_pages_sets_flag(self):
        doc, pdf_doc = _open_doc(3)
        PdfService.delete_pages(doc, pdf_doc, [0])
        assert pdf_doc.has_structural_change is True
        doc.close()

    def test_reorder_pages_sets_flag(self):
        doc, pdf_doc = _open_doc(3)
        PdfService.reorder_pages(doc, pdf_doc, [2, 0, 1])
        assert pdf_doc.has_structural_change is True
        doc.close()

    def test_rotate_pages_does_not_set_flag(self):
        """旋转是页属性修改，incremental save 支持，不置结构标志。"""
        doc, pdf_doc = _open_doc(2)
        PdfService.rotate_pages(doc, pdf_doc, [0], 90)
        assert pdf_doc.has_structural_change is False
        doc.close()

    def test_insert_blank_page_sets_flag(self):
        doc, pdf_doc = _open_doc(1)
        PdfService.insert_blank_page(doc, pdf_doc, 0)
        assert pdf_doc.has_structural_change is True
        doc.close()
