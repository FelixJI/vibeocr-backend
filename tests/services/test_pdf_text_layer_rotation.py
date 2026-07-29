# tests/services/test_pdf_text_layer_rotation.py
"""回归测试：带 /Rotate 的页面写入文字层时坐标/方向正确。

复现用户报告：在带 /Rotate 90 的 PDF（竖排 mediabox 595×841、横排显示 841×595）
上，OCR 渲染的是横排显示图，bbox 在『显示空间』，但 insert_textbox 写入的是
『mediabox 空间』。旧代码用 page.rect（显示尺寸）直接做归一化映射，宽高互换
+未补偿 /Rotate，导致『本来上面的字写到了右面』。

验证策略：用无原文字的纯图片页（模拟扫描件），OCR 在『显示渲染图』上检测到
位于显示页面『顶部中央』的文字 → 归一化 bbox（显示空间）→ add_text_layer 写入
→ 读回文字层，断言其 mediabox 坐标经 rotation_matrix 转回显示空间后，仍落在
显示页面顶部中央（而非右/下）。
"""

import fitz
import pytest

from vibeocr.backend.models.ocr_result import OCRResult, TextBlock
from vibeocr.backend.services.pdf_service import PdfService


def _make_blank_rotated_pdf(path, src_width, src_height, rotate):
    """创建 mediabox 为 src_width×src_height、带 /Rotate 的纯空白页（无原文字）。"""
    doc = fitz.open()
    page = doc.new_page(width=src_width, height=src_height)
    page.set_rotation(rotate)
    doc.save(str(path))
    doc.close()
    return path


def _derotate_rect(page, rect_displayed):
    """把『显示空间』rect 转成『mediabox 空间』rect（用 derotation_matrix）。

    用于断言：写回的文字层在 mediabox 空间的 bbox，经此映射应≈ 显示空间 bbox。
    """
    dm = page.derotation_matrix
    a, b, c, d, e, f = dm.a, dm.b, dm.c, dm.d, dm.e, dm.f

    def de(x, y):
        return (a * x + c * y + e, b * x + d * y + f)

    corners = [
        de(rect_displayed.x0, rect_displayed.y0),
        de(rect_displayed.x1, rect_displayed.y0),
        de(rect_displayed.x0, rect_displayed.y1),
        de(rect_displayed.x1, rect_displayed.y1),
    ]
    xs = [pt[0] for pt in corners]
    ys = [pt[1] for pt in corners]
    return fitz.Rect(min(xs), min(ys), max(xs), max(ys))


class TestRotatedPageTextLayer:
    """带 /Rotate 页面的文字层坐标正确性（纯空白页，模拟扫描件）。"""

    @pytest.mark.parametrize("rotate", [90, 180, 270])
    def test_top_text_stays_on_top(self, tmp_path, rotate):
        """在『显示页面顶部中央』的文字，写回后应在显示页面顶部。

        模拟真实管线：
          1. 纯空白 rotate 页（无原文字）。
          2. OCR 在『显示渲染图』上检测到顶部中央文字 → 归一化 bbox（显示空间）。
          3. add_text_layer 写入隐形文字层。
          4. 读回文字层 mediabox bbox → 用 rotation_matrix 转回显示空间。
          5. 断言：显示空间 y 中点 < 页面高度 50%（仍在顶部）。
        """
        path = _make_blank_rotated_pdf(
            tmp_path / f"rot_{rotate}.pdf", src_width=595, src_height=841, rotate=rotate
        )
        doc, pdf_doc = PdfService.open_doc(str(path))
        page = doc[0]
        disp_rect = page.rect  # 显示尺寸
        disp_w, disp_h = disp_rect.width, disp_rect.height

        # 显示空间『顶部中央』窄条：x 40%~60%，y 5%~10%
        disp_box = fitz.Rect(disp_w * 0.4, disp_h * 0.05, disp_w * 0.6, disp_h * 0.10)
        nbbox = (
            disp_box.x0 / disp_w * 1000,
            disp_box.y0 / disp_h * 1000,
            disp_box.x1 / disp_w * 1000,
            disp_box.y1 / disp_h * 1000,
        )
        block = TextBlock(text="TOP", score=0.95, bbox=nbbox)
        result = OCRResult(raw_text="TOP", text_blocks=[block], preproc_angle=0)
        written, skipped = PdfService.add_text_layer(doc, pdf_doc, 0, result)
        assert written == 1, f"未写入: written={written} skipped={skipped}"

        out = tmp_path / f"out_{rotate}.pdf"
        doc.save(str(out))
        doc.close()

        # 读回文字层（页面唯一的文字）
        d2 = fitz.open(str(out))
        p2 = d2[0]
        layer_rects = []
        # get_text("dict") 运行时返回 dict（PyMuPDF），但其 stub 标注为 str，
        # 此处显式标注为 dict 以正确索引 blocks/lines/bbox。
        p2_dict: dict = p2.get_text("dict")  # type: ignore[assignment]
        for b in p2_dict["blocks"]:
            if "lines" in b:
                for line in b["lines"]:
                    layer_rects.append(fitz.Rect(*line["bbox"]))
        d2.close()
        assert layer_rects, f"rotate={rotate}: 文字层未找到"
        mb_rect = layer_rects[0]

        # 把写回的 mediabox bbox 转回显示空间，验证仍在顶部
        rm = p2.rotation_matrix if False else None  # 重新打开避免误用
        d3 = fitz.open(str(out))
        p3 = d3[0]
        rm = p3.rotation_matrix
        a, b, c, d, e, f = rm.a, rm.b, rm.c, rm.d, rm.e, rm.f

        def rot(x, y):
            return (a * x + c * y + e, b * x + d * y + f)

        corners = [
            rot(mb_rect.x0, mb_rect.y0),
            rot(mb_rect.x1, mb_rect.y0),
            rot(mb_rect.x0, mb_rect.y1),
            rot(mb_rect.x1, mb_rect.y1),
        ]
        ys = [pt[1] for pt in corners]
        disp_y_mid = (min(ys) + max(ys)) / 2
        d3.close()

        assert disp_y_mid < disp_h * 0.5, (
            f"rotate={rotate}: 文字应在显示页面顶部(y<{disp_h * 0.5:.0f})，"
            f"实际显示空间 y 中点={disp_y_mid:.1f}（『上面的字写到了右面/下面』症状）"
        )

    def test_rotate_0_baseline(self, tmp_path):
        """无旋转基线：顶部文字写回应仍在顶部。"""
        path = _make_blank_rotated_pdf(
            tmp_path / "rot0.pdf", src_width=595, src_height=841, rotate=0
        )
        doc, pdf_doc = PdfService.open_doc(str(path))
        page = doc[0]
        _disp_w, disp_h = page.rect.width, page.rect.height
        nbbox = (400.0, 50.0, 600.0, 100.0)  # 顶部
        block = TextBlock(text="TOP", score=0.95, bbox=nbbox)
        result = OCRResult(raw_text="TOP", text_blocks=[block], preproc_angle=0)
        written, _ = PdfService.add_text_layer(doc, pdf_doc, 0, result)
        assert written == 1
        doc.save(str(tmp_path / "out0.pdf"))
        doc.close()

        d2 = fitz.open(str(tmp_path / "out0.pdf"))
        d2_dict: dict = d2[0].get_text("dict")  # type: ignore[assignment]
        for b in d2_dict["blocks"]:
            if "lines" in b:
                for line in b["lines"]:
                    r = fitz.Rect(*line["bbox"])
                    assert (r.y0 + r.y1) / 2 < disp_h * 0.5
        d2.close()
