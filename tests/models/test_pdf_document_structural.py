"""PdfDocument.has_structural_change 标志测试。"""

from vibeocr.backend.models.pdf_document import PdfDocument


class TestHasStructuralChange:
    def test_default_false(self):
        doc = PdfDocument()
        assert doc.has_structural_change is False

    def test_can_set_true(self):
        doc = PdfDocument()
        doc.has_structural_change = True
        assert doc.has_structural_change is True

    def test_independent_from_is_modified(self):
        """has_structural_change 与 is_modified 正交。"""
        doc = PdfDocument()
        doc.is_modified = True
        assert doc.has_structural_change is False
        doc.has_structural_change = True
        assert doc.is_modified is True
