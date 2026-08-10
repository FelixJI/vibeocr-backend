"""PDF OCR 文字层与页面显示坐标的精确对齐回归测试。"""

from __future__ import annotations

import fitz
import pytest
from vibeocr.backend.models.ocr_result import OCRResult, TextBlock
from vibeocr.backend.services.pdf_service import PdfService


def _to_display_rect(page: fitz.Page, rect: fitz.Rect) -> fitz.Rect:
    """把 PyMuPDF 提取的未旋转文字 bbox 映射到页面显示空间。"""
    matrix = page.rotation_matrix
    points = [
        fitz.Point(rect.x0, rect.y0) * matrix,
        fitz.Point(rect.x1, rect.y0) * matrix,
        fitz.Point(rect.x0, rect.y1) * matrix,
        fitz.Point(rect.x1, rect.y1) * matrix,
    ]
    return fitz.Rect(
        min(point.x for point in points),
        min(point.y for point in points),
        max(point.x for point in points),
        max(point.y for point in points),
    )


@pytest.mark.parametrize("rotation", [0, 90])
def test_cropped_page_text_layer_uses_visible_page_origin(tmp_path, rotation) -> None:
    """CropBox 非零时，归一化 bbox 仍以可见页面左上角为原点。"""
    source = tmp_path / f"cropped-{rotation}-source.pdf"
    document = fitz.open()
    page = document.new_page(width=600, height=800)
    page.set_cropbox(fitz.Rect(100, 150, 500, 700))
    page.set_rotation(rotation)
    document.save(source)
    document.close()

    document, model = PdfService.open_doc(str(source))
    visible = document[0].rect
    normalized_bbox = (100.0, 100.0, 400.0, 180.0)
    expected = fitz.Rect(
        visible.width * 0.1,
        visible.height * 0.1,
        visible.width * 0.4,
        visible.height * 0.18,
    )
    result = OCRResult(
        raw_text="ALIGN",
        text_blocks=[TextBlock(text="ALIGN", score=0.99, bbox=normalized_bbox)],
        preproc_angle=0,
    )

    written, skipped = PdfService.add_text_layer(document, model, 0, result)

    assert (written, skipped) == (1, 0)
    words = document[0].get_text("words")
    assert len(words) == 1
    actual = _to_display_rect(document[0], fitz.Rect(*words[0][:4]))
    assert actual.x0 == pytest.approx(expected.x0, abs=2.0)
    assert (actual.y0 + actual.y1) / 2 == pytest.approx(
        (expected.y0 + expected.y1) / 2,
        abs=expected.height * 0.2,
    )
    document.close()
