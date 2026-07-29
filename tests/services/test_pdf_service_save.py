"""save_with_rewrite: rewrite + 按结构改动分流落盘。"""

from pathlib import Path

import fitz

from vibeocr.backend.models.ocr_result import OCRResult, TextBlock
from vibeocr.backend.models.pdf_document import PdfDocument, PdfPageInfo
from vibeocr.backend.models.pdf_ocr_options import PdfGlobalSettings
from vibeocr.backend.services.pdf_service import PdfService, SaveResult


def _make_scanned_pdf(path):
    import numpy as np
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    img = np.ones((792, 612, 3), dtype=np.uint8) * 240
    cs = fitz.Colorspace(fitz.CS_RGB)
    pixmap = fitz.Pixmap(cs, 612, 792, img.tobytes(), 0)
    page.insert_image(fitz.Rect(0, 0, 612, 792), pixmap=pixmap)
    doc.save(str(path))
    doc.close()
    return path


class TestSaveWithRewrite:
    def test_resets_is_modified_and_structural_flag(self, tmp_path):
        path = tmp_path / "scan.pdf"
        _make_scanned_pdf(path)
        doc = fitz.open(str(path))
        pdf_doc = PdfDocument(file_path=str(path))
        info = PdfPageInfo(page_index=0)
        pdf_doc.pages = [info]

        # 模拟 OCR 注入文字块
        result = OCRResult(
            raw_text="Hello",
            text_blocks=[TextBlock(text="Hello", score=0.9, bbox=(50, 50, 300, 100))],
        )
        PdfService.add_text_layer(doc, pdf_doc, 0, result)

        pdf_doc.is_modified = True
        # 纯文字层编辑，无结构改动
        save_result = PdfService.save_with_rewrite(doc, pdf_doc, path=None)
        assert isinstance(save_result, SaveResult)
        assert pdf_doc.is_modified is False
        assert pdf_doc.has_structural_change is False
        # 全量压缩覆盖：doc 已 close+reopen，原 doc 失效，关新 doc
        if save_result.new_doc is not None:
            save_result.new_doc.close()

    def test_save_as_writes_new_file(self, tmp_path):
        path = tmp_path / "src.pdf"
        _make_scanned_pdf(path)
        doc = fitz.open(str(path))
        pdf_doc = PdfDocument(file_path=str(path))
        pdf_doc.pages = [PdfPageInfo(page_index=0)]

        dest = tmp_path / "out.pdf"
        PdfService.save_with_rewrite(doc, pdf_doc, path=str(dest))
        assert dest.exists()
        doc.close()

    def test_ocr_finalize_skips_rewriting_existing_text_layers(
        self, tmp_path, monkeypatch
    ):
        path = tmp_path / "scan.pdf"
        _make_scanned_pdf(path)
        doc = fitz.open(str(path))
        pdf_doc = PdfDocument(file_path=str(path))
        pdf_doc.pages = [PdfPageInfo(page_index=0)]
        PdfService.add_text_layer(
            doc,
            pdf_doc,
            0,
            OCRResult(
                raw_text="Fast",
                text_blocks=[
                    TextBlock(text="Fast", score=0.9, bbox=(50, 50, 300, 100))
                ],
            ),
        )

        rewrite_calls = []
        monkeypatch.setattr(
            PdfService,
            "rewrite_text_layer",
            lambda *args, **kwargs: rewrite_calls.append((args, kwargs)),
        )
        result = PdfService.save_with_rewrite(
            doc,
            pdf_doc,
            pdf_settings=PdfGlobalSettings(compress_on_save=False),
            rewrite_text_layers=False,
        )

        assert result.rewritten_pages == []
        assert rewrite_calls == []
        doc.close()
        verify = fitz.open(str(path))
        assert "Fast" in verify[0].get_text()
        verify.close()


class TestSaveCompressOnSave:
    """A/D: compress_on_save 分流 + 临时文件原子替换 + clean。"""

    def test_save_in_place_compress_writes_clean_pdf(self, tmp_path):
        """默认 compress_on_save=True：覆盖保存走全量压缩（临时文件替换）。"""
        path = tmp_path / "scan.pdf"
        _make_scanned_pdf(path)
        doc = fitz.open(str(path))
        pdf_doc = PdfDocument(file_path=str(path))
        info = PdfPageInfo(page_index=0)
        pdf_doc.pages = [info]

        result = OCRResult(
            raw_text="Hello",
            text_blocks=[TextBlock(text="Hello", score=0.9, bbox=(50, 50, 300, 100))],
        )
        PdfService.add_text_layer(doc, pdf_doc, 0, result)

        settings = PdfGlobalSettings(compress_on_save=True)
        new_doc = PdfService.save(doc, pdf_doc, pdf_settings=settings)

        # 临时文件与备份均应清理
        assert not Path(str(path) + ".bak").exists()
        # 产物可重新打开，文字层在
        verify = fitz.open(str(path))
        assert "Hello" in verify[0].get_text()
        verify.close()
        # 全量压缩覆盖：doc 已 close+reopen，关新 doc
        if new_doc is not None:
            new_doc.close()

    def test_save_in_place_incremental_when_compress_off(self, tmp_path):
        """compress_on_save=False：走增量快路径（incremental）。"""
        path = tmp_path / "scan.pdf"
        _make_scanned_pdf(path)
        doc = fitz.open(str(path))
        pdf_doc = PdfDocument(file_path=str(path))
        pdf_doc.pages = [PdfPageInfo(page_index=0)]

        result = OCRResult(
            raw_text="World",
            text_blocks=[TextBlock(text="World", score=0.9, bbox=(50, 50, 300, 100))],
        )
        PdfService.add_text_layer(doc, pdf_doc, 0, result)

        settings = PdfGlobalSettings(compress_on_save=False)
        PdfService.save(doc, pdf_doc, pdf_settings=settings)

        verify = fitz.open(str(path))
        assert "World" in verify[0].get_text()
        verify.close()
        doc.close()

    def test_save_as_applies_clean(self, tmp_path):
        """另存为分支也带 clean：产物可重新打开。"""
        path = tmp_path / "src.pdf"
        _make_scanned_pdf(path)
        doc = fitz.open(str(path))
        pdf_doc = PdfDocument(file_path=str(path))
        pdf_doc.pages = [PdfPageInfo(page_index=0)]

        result = OCRResult(
            raw_text="Clean",
            text_blocks=[TextBlock(text="Clean", score=0.9, bbox=(50, 50, 300, 100))],
        )
        PdfService.add_text_layer(doc, pdf_doc, 0, result)

        dest = tmp_path / "out.pdf"
        PdfService.save(doc, pdf_doc, path=str(dest))
        verify = fitz.open(str(dest))
        assert "Clean" in verify[0].get_text()
        verify.close()
        doc.close()


class TestSaveSharesSubsetFont:
    """B: 整文档一次聚合，多页共享单一子集字体。"""

    def test_save_with_rewrite_shares_single_subset_font(self, tmp_path):
        """两页不同字符 OCR，保存后文档内嵌入子集字体对象应 ≤ 1。"""
        path = tmp_path / "scan.pdf"
        # 造两页扫描 PDF
        import numpy as np

        doc = fitz.open()
        for _ in range(2):
            page = doc.new_page(width=612, height=792)
            img = np.ones((792, 612, 3), dtype=np.uint8) * 240
            cs = fitz.Colorspace(fitz.CS_RGB)
            pixmap = fitz.Pixmap(cs, 612, 792, img.tobytes(), 0)
            page.insert_image(fitz.Rect(0, 0, 612, 792), pixmap=pixmap)
        doc.save(str(path))
        doc.close()

        doc = fitz.open(str(path))
        pdf_doc = PdfDocument(file_path=str(path))
        pdf_doc.pages = [PdfPageInfo(page_index=i) for i in range(2)]

        # 两页不同字符（触发各自子集），但保存时应聚合为一个共享子集
        PdfService.add_text_layer(
            doc, pdf_doc, 0,
            OCRResult(raw_text="甲乙", text_blocks=[TextBlock(text="甲乙", score=0.9, bbox=(50, 50, 300, 100))]),
        )
        PdfService.add_text_layer(
            doc, pdf_doc, 1,
            OCRResult(raw_text="丙丁", text_blocks=[TextBlock(text="丙丁", score=0.9, bbox=(50, 50, 300, 100))]),
        )

        save_result = PdfService.save_with_rewrite(doc, pdf_doc, path=None)

        verify = fitz.open(str(path))
        # 收集全文档嵌入的 TrueType 子集字体（排除内置 CID 字体 china-s 等）
        embedded = set()
        for i in range(verify.page_count):
            for f in verify.get_page_fonts(i, full=True):
                # f: (xref, ext, type, basefont, name, encoding, ...)
                ftype = f[2] if len(f) > 2 else ""
                ext = f[1] if len(f) > 1 else ""
                # 仅统计嵌入的字体文件（子集 TrueType），china-s 等内置不计
                if ext in ("ttf", "otf", "n/a") and "Type" in str(ftype) and "CID" not in str(ftype):
                    embedded.add(f[0])
        verify.close()
        # 全量压缩覆盖：doc 已 close+reopen，关新 doc
        if save_result.new_doc is not None:
            save_result.new_doc.close()
        # 共享单一子集：嵌入的子集字体对象应 ≤ 1
        assert len(embedded) <= 1, f"期望 ≤1 个共享子集字体，实际 {len(embedded)}"
