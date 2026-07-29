"""Tests for PdfSession dataclass."""

import fitz
import pytest

from vibeocr.backend.models.pdf_document import PdfDocument, PdfPageInfo
from vibeocr.backend.models.pdf_session import PdfSession


@pytest.fixture
def single_page_doc():
    doc = fitz.open()
    doc.new_page(width=612, height=792)
    return doc


class TestPdfSession:
    def test_defaults(self, single_page_doc):
        pdf_doc = PdfDocument(file_path="test.pdf", pages=[PdfPageInfo(page_index=0)])
        session = PdfSession(
            file_path="test.pdf", session_id="test-sid", pdf_document=pdf_doc
        )

        assert session.file_path == "test.pdf"
        assert session.is_modified is False
        assert len(session.loaded_pages) == 0
        assert session.load_progress == 0.0
        single_page_doc.close()

    def test_is_modified_delegates_to_pdf_document(self, single_page_doc):
        pdf_doc = PdfDocument(file_path="test.pdf", pages=[PdfPageInfo(page_index=0)])
        session = PdfSession(
            file_path="test.pdf", session_id="test-sid", pdf_document=pdf_doc
        )

        assert session.is_modified is False
        pdf_doc.is_modified = True
        assert session.is_modified is True
        single_page_doc.close()

    def test_load_progress_with_loaded_pages(self, single_page_doc):
        pdf_doc = PdfDocument(
            file_path="test.pdf",
            pages=[PdfPageInfo(page_index=0), PdfPageInfo(page_index=1)],
        )
        session = PdfSession(
            file_path="test.pdf", session_id="test-sid", pdf_document=pdf_doc
        )
        session.loaded_pages.add(0)

        assert session.load_progress == 0.5

    def test_load_progress_empty_doc(self):
        doc = fitz.open()
        pdf_doc = PdfDocument(file_path="empty.pdf", pages=[])
        session = PdfSession(file_path="empty.pdf", session_id="test-sid", pdf_document=pdf_doc)

        assert session.load_progress == 1.0
        doc.close()

    def test_load_progress_all_loaded(self, single_page_doc):
        pdf_doc = PdfDocument(file_path="test.pdf", pages=[PdfPageInfo(page_index=0)])
        session = PdfSession(
            file_path="test.pdf", session_id="test-sid", pdf_document=pdf_doc
        )
        session.loaded_pages.add(0)

        assert session.load_progress == 1.0
        single_page_doc.close()


class TestPdfSessionOcrStats:
    def test_default_ocr_stats_are_zero(self, single_page_doc):
        pdf_doc = PdfDocument(file_path="test.pdf", pages=[PdfPageInfo(page_index=0)])
        session = PdfSession(
            file_path="test.pdf", session_id="test-sid", pdf_document=pdf_doc
        )
        assert session.ocr_stats == {"written": 0, "skipped": 0}
        single_page_doc.close()

    def test_reset_ocr_stats(self, single_page_doc):
        pdf_doc = PdfDocument(file_path="test.pdf", pages=[PdfPageInfo(page_index=0)])
        session = PdfSession(
            file_path="test.pdf", session_id="test-sid", pdf_document=pdf_doc
        )
        session.add_ocr_stats(3, 1)
        assert session.ocr_stats == {"written": 3, "skipped": 1}
        session.reset_ocr_stats()
        assert session.ocr_stats == {"written": 0, "skipped": 0}
        single_page_doc.close()

    def test_add_ocr_stats_accumulates(self, single_page_doc):
        pdf_doc = PdfDocument(file_path="test.pdf", pages=[PdfPageInfo(page_index=0)])
        session = PdfSession(
            file_path="test.pdf", session_id="test-sid", pdf_document=pdf_doc
        )
        session.add_ocr_stats(2, 1)
        session.add_ocr_stats(5, 0)
        assert session.ocr_stats == {"written": 7, "skipped": 1}
        single_page_doc.close()
