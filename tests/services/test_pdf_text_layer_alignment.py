"""PDF OCR 文字层与页面显示坐标的精确对齐回归测试。"""

from __future__ import annotations

import fitz
import numpy as np
import pytest
from vibeocr.backend.models.ocr_result import OCRResult, TextBlock
from vibeocr.backend.models.pdf_ocr_options import PdfGlobalSettings
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
    assert actual.height >= expected.height * 0.95, (
        f"rotation={rotation}: 文字层高度 {actual.height:.2f}pt "
        f"未覆盖 OCR bbox 高度 {expected.height:.2f}pt"
    )
    document.close()


@pytest.mark.parametrize("text", ["ALIGN", "glyph", "测试文字"])
def test_visible_text_ink_height_covers_ocr_bbox(tmp_path, text) -> None:
    """文字字形的实际渲染墨迹高度应覆盖 OCR 检测框，而不只是 span bbox。"""
    source = tmp_path / "visible-ink.pdf"
    document = fitz.open()
    document.new_page(width=400, height=550)
    document.save(source)
    document.close()

    document, model = PdfService.open_doc(str(source))
    page_rect = document[0].rect
    normalized_bbox = (100.0, 100.0, 500.0, 180.0)
    expected = PdfService._denormalize_and_unrotate_bbox(
        normalized_bbox,
        0,
        page_rect,
    )
    result = OCRResult(
        raw_text=text,
        text_blocks=[TextBlock(text=text, score=0.99, bbox=normalized_bbox)],
        preproc_angle=0,
    )
    settings = PdfGlobalSettings(text_layer_visible=True)

    written, skipped = PdfService.add_text_layer(
        document,
        model,
        0,
        result,
        settings,
    )

    assert (written, skipped) == (1, 0)
    scale = 2
    pixmap = document[0].get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    pixels = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height,
        pixmap.width,
        pixmap.n,
    )
    ink_y, _ink_x = np.where(np.min(pixels[:, :, :3], axis=2) < 128)
    assert ink_y.size > 0
    actual_ink_height = int(ink_y.max() - ink_y.min() + 1)
    expected_ink_height = expected.height * scale
    assert actual_ink_height == pytest.approx(expected_ink_height, abs=4), (
        f"text={text!r}: 字形墨迹高度 {actual_ink_height}px "
        f"未覆盖 OCR bbox 高度 {expected_ink_height:.1f}px"
    )
    actual_top = int(ink_y.min()) / scale
    actual_bottom = int(ink_y.max() + 1) / scale
    assert actual_top == pytest.approx(expected.y0, abs=2)
    assert actual_bottom == pytest.approx(expected.y1, abs=2)
    document.close()
