"""pdf_service 覆盖补充：聚焦未覆盖分支（save 异常、incremental、空页/空文字层等）。

纯 fitz/数学路径，无 paddle/CUDA 依赖。
"""

import logging
from pathlib import Path

import fitz
import numpy as np
import pytest
from vibeocr.backend.models.ocr_result import OCRResult, TextBlock
from vibeocr.backend.services.pdf_service import PdfService, SaveResult


def _create_test_pdf(path: Path, num_pages: int = 2) -> Path:
    doc = fitz.open()
    for i in range(num_pages):
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 72), f"Page {i + 1}", fontsize=12)
    doc.save(str(path))
    doc.close()
    return path


def _create_scanned_pdf(path: Path, width: int = 612, height: int = 792) -> Path:
    doc = fitz.open()
    page = doc.new_page(width=width, height=height)
    img = np.ones((height, width, 3), dtype=np.uint8) * 240
    cs = fitz.Colorspace(fitz.CS_RGB)
    pixmap = fitz.Pixmap(cs, width, height, img.tobytes(), 0)
    page.insert_image(fitz.Rect(0, 0, width, height), pixmap=pixmap)
    doc.save(str(path))
    doc.close()
    return path


def _make_result(blocks, angle=0):
    return OCRResult(
        raw_text=" ".join(b.text for b in blocks),
        text_blocks=list(blocks),
        preproc_angle=angle,
    )


# ---- save() 路径 ----------------------------------------------------------


class TestSaveEdgeCases:
    def test_save_in_place_returns_none_when_no_file_path(self, tmp_path):
        """save() 覆盖保存但 file_path=None 时直接返回 None（line 117-118）。"""
        doc, pdf_doc = PdfService.open_doc(str(_create_test_pdf(tmp_path / "t.pdf")))
        try:
            pdf_doc.file_path = None
            assert PdfService.save(doc, pdf_doc) is None
        finally:
            doc.close()

    def test_save_incremental_path_success(self, tmp_path):
        """compress_on_save=False 走增量保存快路径（lines 123-134）。"""
        from vibeocr.backend.models.pdf_ocr_options import PdfGlobalSettings

        path = tmp_path / "incr.pdf"
        _create_test_pdf(path, num_pages=1)
        doc, pdf_doc = PdfService.open_doc(str(path))
        try:
            PdfService.rotate_pages(doc, pdf_doc, [0], 90)
            settings = PdfGlobalSettings(compress_on_save=False)
            assert PdfService.save(doc, pdf_doc, pdf_settings=settings) is None
            assert pdf_doc.is_modified is False
            # .bak 应已清理
            assert not Path(str(path) + ".bak").exists()
        finally:
            doc.close()

    def test_save_incremental_failure_restores_backup(self, tmp_path, monkeypatch):
        """compress_on_save=False 增量保存抛异常时应回滚备份并 re-raise（lines 129-132）。"""
        from vibeocr.backend.models.pdf_ocr_options import PdfGlobalSettings

        path = tmp_path / "incr_fail.pdf"
        _create_test_pdf(path, num_pages=1)
        original_save = fitz.Document.save
        doc, pdf_doc = PdfService.open_doc(str(path))
        try:
            # 强制 incremental save 失败
            def _fail_save(self, *args, **kwargs):
                if kwargs.get("incremental"):
                    raise RuntimeError("incremental boom")
                return original_save(self, *args, **kwargs)

            monkeypatch.setattr(fitz.Document, "save", _fail_save)
            settings = PdfGlobalSettings(compress_on_save=False)
            with pytest.raises(RuntimeError, match="incremental boom"):
                PdfService.save(doc, pdf_doc, pdf_settings=settings)
        finally:
            monkeypatch.setattr(fitz.Document, "save", original_save)
            doc.close()


# ---- _compress_in_place 异常回滚 -----------------------------------------


class TestCompressInPlaceFailure:
    def test_compress_failure_restores_and_reraises(self, tmp_path, monkeypatch):
        """_compress_in_place save 抛异常时回滚备份+清理临时文件并 re-raise（lines 210-229）。"""
        path = tmp_path / "compress_fail.pdf"
        _create_test_pdf(path, num_pages=1)
        doc, _pdf_doc = PdfService.open_doc(str(path))
        try:
            # 让 doc.save 在 _compress_in_place 内抛异常
            original_save = fitz.Document.save

            call_count = {"n": 0}

            def _fail(self, *args, **kwargs):
                call_count["n"] += 1
                # 仅 _compress_in_place 的 save（带 garbage=3）抛异常
                if kwargs.get("garbage") == 3:
                    raise RuntimeError("compress boom")
                return original_save(self, *args, **kwargs)

            monkeypatch.setattr(fitz.Document, "save", _fail)
            with pytest.raises(RuntimeError, match="compress boom"):
                PdfService._compress_in_place(doc, str(path), clean=False)
            # 备份应已清理
            assert not Path(str(path) + ".bak").exists()
            assert not Path(str(path) + ".tmp").exists()
        finally:
            monkeypatch.setattr(fitz.Document, "save", original_save)
            # doc 可能已被 close，忽略
            try:
                doc.close()
            except Exception:
                pass


# ---- save_with_rewrite 边界 ----------------------------------------------


class TestSaveWithRewriteEdgeCases:
    def test_save_with_rewrite_no_file_path_returns_result_none(self, tmp_path):
        """save_with_rewrite 覆盖但 file_path=None → 不落盘，返回 SaveResult（lines 298-303）。"""
        doc, pdf_doc = PdfService.open_doc(str(_create_test_pdf(tmp_path / "t.pdf")))
        try:
            pdf_doc.file_path = None
            result = PdfService.save_with_rewrite(doc, pdf_doc, path=None)
            assert isinstance(result, SaveResult)
            assert result.path is None
            assert result.new_doc is None
            assert pdf_doc.is_modified is False
        finally:
            doc.close()

    def test_save_with_rewrite_incremental_when_no_structural_change(self, tmp_path):
        """无结构改动且 compress_on_save=False → 走增量路径（lines 306-311）。

        用 rotate（不改结构）保持 has_structural_change=False，且 rotate 后
        doc.can_save_incrementally() 仍为 True。
        """
        from vibeocr.backend.models.pdf_ocr_options import PdfGlobalSettings

        path = tmp_path / "rw_incr.pdf"
        _create_test_pdf(path, num_pages=1)
        doc, pdf_doc = PdfService.open_doc(str(path))
        try:
            # rotate 不触发 has_structural_change
            PdfService.rotate_pages(doc, pdf_doc, [0], 90)
            assert pdf_doc.has_structural_change is False
            assert bool(doc.can_save_incrementally())
            settings = PdfGlobalSettings(compress_on_save=False)
            res = PdfService.save_with_rewrite(
                doc,
                pdf_doc,
                path=None,
                pdf_settings=settings,
                rewrite_text_layers=False,
            )
            # 增量成功 → new_doc=None
            assert res.new_doc is None
            assert res.path is None
        finally:
            doc.close()

    def test_save_with_rewrite_incremental_fail_raises(self, tmp_path, monkeypatch):
        """增量保存失败时 save_with_rewrite 应抛 RuntimeError（line 311）。"""
        from vibeocr.backend.models.pdf_ocr_options import PdfGlobalSettings

        path = tmp_path / "rw_fail.pdf"
        _create_test_pdf(path, num_pages=1)
        doc, pdf_doc = PdfService.open_doc(str(path))
        try:
            # 强制 can_save_incrementally 返回 False
            monkeypatch.setattr(
                fitz.Document, "can_save_incrementally", lambda self: False
            )
            settings = PdfGlobalSettings(compress_on_save=False)
            with pytest.raises(RuntimeError, match="incremental save failed"):
                PdfService.save_with_rewrite(
                    doc, pdf_doc, path=None, pdf_settings=settings
                )
        finally:
            doc.close()


# ---- save_incremental 错误路径 -------------------------------------------


class TestSaveIncrementalErrors:
    def test_save_incremental_returns_false_when_not_supported(
        self, tmp_path, monkeypatch
    ):
        """can_save_incrementally=False → 记日志返回 False（lines 345-353）。"""
        path = tmp_path / "ni.pdf"
        _create_test_pdf(path, num_pages=1)
        doc, _ = PdfService.open_doc(str(path))
        try:
            monkeypatch.setattr(
                fitz.Document, "can_save_incrementally", lambda self: False
            )
            assert PdfService.save_incremental(doc, str(path)) is False
        finally:
            doc.close()

    def test_save_incremental_marker_creation_failure(self, tmp_path, monkeypatch):
        """marker 创建异常（如 stat 失败）→ 返回 False（lines 359-361）。"""
        path = tmp_path / "nf.pdf"
        _create_test_pdf(path, num_pages=1)
        doc, _ = PdfService.open_doc(str(path))
        try:
            # can_save_incrementally=True 但 path.stat 抛异常（让 Path.stat 失败）
            original_stat = Path.stat

            def _fail_stat(self, *args, **kwargs):
                if str(self) == str(path):
                    raise OSError("stat boom")
                return original_stat(self, *args, **kwargs)

            monkeypatch.setattr(Path, "stat", _fail_stat)
            assert PdfService.save_incremental(doc, str(path)) is False
        finally:
            doc.close()

    def test_save_incremental_save_failure_truncates(self, tmp_path, monkeypatch):
        """doc.save incremental 抛异常 → 截断回滚并返回 False（lines 366-375）。"""
        path = tmp_path / "sf.pdf"
        _create_test_pdf(path, num_pages=1)
        doc, _ = PdfService.open_doc(str(path))
        original_size = path.stat().st_size
        try:
            original_save = fitz.Document.save

            def _fail(self, *args, **kwargs):
                if kwargs.get("incremental"):
                    raise RuntimeError("save boom")
                return original_save(self, *args, **kwargs)

            monkeypatch.setattr(fitz.Document, "save", _fail)
            assert PdfService.save_incremental(doc, str(path)) is False
            # 文件长度应被截断回 original_size（或 marker 创建前的长度）
            assert path.stat().st_size == original_size
        finally:
            monkeypatch.setattr(fitz.Document, "save", original_save)
            doc.close()

    def test_truncate_incremental_shrunk_raises(self, tmp_path):
        """_truncate_incremental: current < original 时抛 RuntimeError（line 381）。"""
        path = tmp_path / "short.pdf"
        path.write_bytes(b"short")
        marker = tmp_path / "short.pdf.vibeocr-incremental"
        marker.write_text("100")
        with pytest.raises(RuntimeError, match="shrank"):
            PdfService._truncate_incremental(path, marker, 100)

    def test_truncate_incremental_noop_when_same_size(self, tmp_path):
        """current == original_size → 无操作、删 marker（lines 384-389）。"""
        path = tmp_path / "same.pdf"
        path.write_bytes(b"same")
        marker = tmp_path / "same.pdf.vibeocr-incremental"
        marker.write_text("4")
        PdfService._truncate_incremental(path, marker, 4)
        assert not marker.exists()

    def test_recover_interrupted_negative_size_raises(self, tmp_path):
        """marker 含负数 → ValueError → 异常上抛（lines 400-401, 405-407）。"""
        path = tmp_path / "neg.pdf"
        path.write_bytes(b"x")
        marker = tmp_path / "neg.pdf.vibeocr-incremental"
        marker.write_text("-5")
        with pytest.raises(ValueError):
            PdfService._recover_interrupted_incremental(str(path))

    def test_recover_interrupted_no_marker_returns_false(self, tmp_path):
        """无 marker → 返回 False（line 397）。"""
        path = tmp_path / "clean.pdf"
        path.write_bytes(b"x")
        assert PdfService._recover_interrupted_incremental(str(path)) is False

    def test_recover_interrupted_success(self, tmp_path):
        """有 marker 且合法 → 截断恢复并返回 True。"""
        path = tmp_path / "rec.pdf"
        # 写 5 字节原始 + 3 字节增量尾巴
        path.write_bytes(b"origi123")
        marker = tmp_path / "rec.pdf.vibeocr-incremental"
        marker.write_text("5")
        assert PdfService._recover_interrupted_incremental(str(path)) is True
        assert path.read_bytes() == b"origi"
        assert not marker.exists()


# ---- detect_text_layers / is_page_scanned 边界 --------------------------


class TestDetectAndScanEdges:
    def test_detect_text_layers_empty_page(self, tmp_path):
        """无文字的页应返回空列表（line 432 已覆盖，这里验证 image block 跳过 line 443-444）。"""
        path = tmp_path / "img.pdf"
        doc = fitz.open()
        page = doc.new_page(width=200, height=200)
        # 插入一张图片（type != 0 block）
        img = np.ones((50, 50, 3), dtype=np.uint8) * 128
        cs = fitz.Colorspace(fitz.CS_RGB)
        pixmap = fitz.Pixmap(cs, 50, 50, img.tobytes(), 0)
        page.insert_image(fitz.Rect(10, 10, 60, 60), pixmap=pixmap)
        doc.save(str(path))
        doc.close()
        doc, _ = PdfService.open_doc(str(path))
        try:
            # 页面有内容但无文字 → detect 返回空
            assert PdfService.detect_text_layers(doc, 0) == []
        finally:
            doc.close()

    def test_detect_text_layers_skips_empty_line(self, tmp_path):
        """跨多 span 但 line 文本为空应跳过（line 450-451）。"""
        path = tmp_path / "emptyline.pdf"
        doc = fitz.open()
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 72), "real text", fontsize=12)
        doc.save(str(path))
        doc.close()
        doc, _ = PdfService.open_doc(str(path))
        try:
            layers = PdfService.detect_text_layers(doc, 0)
            # 至少有 1 个非空 layer
            assert len(layers) >= 1
        finally:
            doc.close()

    def test_is_page_scanned_no_images(self, tmp_path):
        """无图片页 → is_page_scanned False（line 474-475）。"""
        path = tmp_path / "text.pdf"
        doc = fitz.open()
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 72), "text", fontsize=12)
        doc.save(str(path))
        doc.close()
        doc, _ = PdfService.open_doc(str(path))
        try:
            assert PdfService.is_page_scanned(doc, 0) is False
        finally:
            doc.close()

    def test_is_page_scanned_small_image_not_scanned(self, tmp_path):
        """小图（覆盖率 < 0.5）→ not scanned（lines 476-485）。"""
        path = tmp_path / "small.pdf"
        doc = fitz.open()
        page = doc.new_page(width=612, height=792)
        img = np.ones((20, 20, 3), dtype=np.uint8) * 100
        cs = fitz.Colorspace(fitz.CS_RGB)
        pixmap = fitz.Pixmap(cs, 20, 20, img.tobytes(), 0)
        page.insert_image(fitz.Rect(10, 10, 30, 30), pixmap=pixmap)
        doc.save(str(path))
        doc.close()
        doc, _ = PdfService.open_doc(str(path))
        try:
            assert PdfService.is_page_scanned(doc, 0) is False
        finally:
            doc.close()

    def test_is_page_scanned_full_page_image(self, tmp_path):
        """整页大图（覆盖率 > 0.5）→ scanned=True（line 483-484）。"""
        path = tmp_path / "scan.pdf"
        _create_scanned_pdf(path)
        doc, _ = PdfService.open_doc(str(path))
        try:
            assert PdfService.is_page_scanned(doc, 0) is True
        finally:
            doc.close()

    def test_page_has_text_true_and_false(self, tmp_path):
        """page_has_text 两条分支（lines 510）。"""
        path = tmp_path / "mixed.pdf"
        doc = fitz.open()
        page_with_text = doc.new_page(width=200, height=200)
        page_with_text.insert_text((10, 10), "hi", fontsize=12)
        doc.new_page(width=200, height=200)  # 空白页
        doc.save(str(path))
        doc.close()
        doc, _ = PdfService.open_doc(str(path))
        try:
            assert PdfService.page_has_text(doc, 0) is True
            assert PdfService.page_has_text(doc, 1) is False
        finally:
            doc.close()

    def test_build_page_infos_scanned_page(self, tmp_path):
        """build_page_infos 对扫描件页填 is_scanned=True（lines 515-531）。"""

        path = tmp_path / "bpi.pdf"
        _create_scanned_pdf(path)
        doc, pdf_doc = PdfService.open_doc(str(path))
        try:
            PdfService.build_page_infos(doc, pdf_doc)
            assert pdf_doc.pages[0].is_scanned is True
            assert pdf_doc.pages[0].has_text_layer is False
        finally:
            doc.close()

    def test_update_page_info_out_of_range_noop(self, tmp_path):
        """update_page_info 页码越界 → 静默返回（line 537-538）。"""
        doc, pdf_doc = PdfService.open_doc(str(_create_test_pdf(tmp_path / "t.pdf")))
        try:
            # 不抛异常即覆盖
            PdfService.update_page_info(doc, pdf_doc, 999)
        finally:
            doc.close()

    def test_update_page_info_sets_scanned(self, tmp_path):
        """update_page_info 对扫描件页设 is_scanned=True（lines 539-549）。"""
        path = tmp_path / "upi.pdf"
        _create_scanned_pdf(path)
        doc, pdf_doc = PdfService.open_doc(str(path))
        try:
            PdfService.update_page_info(doc, pdf_doc, 0)
            assert pdf_doc.pages[0].is_scanned is True
            assert pdf_doc.pages[0].thumbnail is None
        finally:
            doc.close()


# ---- move_page / reorder_pages 边界 -------------------------------------


class TestMoveReorderEdges:
    def test_move_page_same_index_noop(self, tmp_path):
        """move_page from==to → 直接返回（line 620-621）。"""
        doc, pdf_doc = PdfService.open_doc(str(_create_test_pdf(tmp_path / "t.pdf")))
        try:
            PdfService.move_page(doc, pdf_doc, 1, 1)
            # 无变化
            assert pdf_doc.pages[1].page_index == 1
        finally:
            doc.close()

    def test_reorder_pages_invalid_count_noop(self, tmp_path):
        """reorder_pages new_order 长度不符 → 返回（line 642-643）。"""
        doc, pdf_doc = PdfService.open_doc(str(_create_test_pdf(tmp_path / "t.pdf")))
        try:
            PdfService.reorder_pages(doc, pdf_doc, [0])  # 长度 1 != 2 页
            # 不抛即覆盖
        finally:
            doc.close()

    def test_reorder_pages_identity_noop(self, tmp_path):
        """reorder_pages new_order == range → 返回（line 644-645）。"""
        doc, pdf_doc = PdfService.open_doc(str(_create_test_pdf(tmp_path / "t.pdf")))
        try:
            PdfService.reorder_pages(doc, pdf_doc, [0, 1])
        finally:
            doc.close()

    def test_reorder_pages_actual_reorder(self, tmp_path):
        """reorder_pages 实际重排（lines 647-650）。"""
        doc, pdf_doc = PdfService.open_doc(str(_create_test_pdf(tmp_path / "t.pdf")))
        try:
            PdfService.reorder_pages(doc, pdf_doc, [1, 0])
            assert pdf_doc.pages[0].page_index == 1
            assert pdf_doc.pages[1].page_index == 0
            assert pdf_doc.has_structural_change is True
        finally:
            doc.close()


# ---- add_text_layer_batch overwrite -------------------------------------


class TestAddTextLayerBatchOverwrite:
    def test_batch_overwrites_existing_layer(self, tmp_path):
        """已有文字层的页 + overwrite=True → 先删后写（lines 774-778, 784）。"""
        path = tmp_path / "batch.pdf"
        _create_scanned_pdf(path)
        doc, pdf_doc = PdfService.open_doc(str(path))
        try:
            # 先加一层
            r1 = OCRResult(
                raw_text="first",
                text_blocks=[
                    TextBlock(
                        text="first", score=0.9, bbox=(50, 50, 200, 100), page_idx=0
                    )
                ],
            )
            PdfService.add_text_layer(doc, pdf_doc, 0, r1)
            assert pdf_doc.pages[0].has_text_layer is True

            # 批量覆盖
            pages_data = [
                {
                    "page": 0,
                    "ocr_result": {
                        "preproc_angle": 0,
                        "text_blocks": [
                            {
                                "text": "second",
                                "score": 0.9,
                                "bbox": [60, 60, 210, 110],
                                "page_idx": 0,
                            }
                        ],
                    },
                }
            ]
            results = PdfService.add_text_layer_batch(
                doc, pdf_doc, pages_data, overwrite=True
            )
            assert 0 in results
            text = doc[0].get_text()
            assert "second" in text
        finally:
            doc.close()

    def test_batch_skips_existing_without_overwrite(self, tmp_path):
        """已有文字层 + overwrite=False → 跳过（lines 766-773）。"""
        path = tmp_path / "batch_skip.pdf"
        _create_scanned_pdf(path)
        doc, pdf_doc = PdfService.open_doc(str(path))
        try:
            r1 = OCRResult(
                raw_text="keep",
                text_blocks=[
                    TextBlock(
                        text="keep", score=0.9, bbox=(50, 50, 200, 100), page_idx=0
                    )
                ],
            )
            PdfService.add_text_layer(doc, pdf_doc, 0, r1)

            pages_data = [
                {
                    "page": 0,
                    "ocr_result": {
                        "preproc_angle": 0,
                        "text_blocks": [
                            {
                                "text": "new",
                                "score": 0.9,
                                "bbox": [60, 60, 210, 110],
                                "page_idx": 0,
                            }
                        ],
                    },
                }
            ]
            results = PdfService.add_text_layer_batch(
                doc, pdf_doc, pages_data, overwrite=False
            )
            assert results[0] == (0, 1)  # skipped
            text = doc[0].get_text()
            assert "keep" in text
            assert "new" not in text
        finally:
            doc.close()

    def test_batch_cancel_check_stops_early(self, tmp_path):
        """cancel_check 返回 True 时停止写后续页（lines 795-797）。"""
        path = tmp_path / "batch_cancel.pdf"
        doc = fitz.open()
        for _ in range(3):
            doc.new_page(width=612, height=792)
        doc.save(str(path))
        doc.close()
        doc, pdf_doc = PdfService.open_doc(str(path))
        try:
            pages_data = [
                {
                    "page": i,
                    "ocr_result": {
                        "preproc_angle": 0,
                        "text_blocks": [
                            {
                                "text": f"p{i}",
                                "score": 0.9,
                                "bbox": [50, 50, 200, 100],
                                "page_idx": i,
                            }
                        ],
                    },
                }
                for i in range(3)
            ]
            # 第一页后就取消
            call_count = {"n": 0}

            def cancel_check():
                call_count["n"] += 1
                return call_count["n"] > 1

            results = PdfService.add_text_layer_batch(
                doc, pdf_doc, pages_data, cancel_check=cancel_check
            )
            # 只写了第一页
            assert 0 in results
            assert 1 not in results
            assert 2 not in results
        finally:
            doc.close()

    def test_batch_empty_to_write_returns_skipped_only(self, tmp_path):
        """所有页都被跳过时直接返回 skipped 结果（lines 783-784）。"""
        path = tmp_path / "batch_all_skip.pdf"
        _create_scanned_pdf(path)
        doc, pdf_doc = PdfService.open_doc(str(path))
        try:
            # 先给 page 0 加层
            r1 = OCRResult(
                raw_text="x",
                text_blocks=[
                    TextBlock(text="x", score=0.9, bbox=(50, 50, 200, 100), page_idx=0)
                ],
            )
            PdfService.add_text_layer(doc, pdf_doc, 0, r1)
            # 再传同一页、overwrite=False
            pages_data = [
                {
                    "page": 0,
                    "ocr_result": {
                        "preproc_angle": 0,
                        "text_blocks": [],
                    },
                }
            ]
            results = PdfService.add_text_layer_batch(
                doc, pdf_doc, pages_data, overwrite=False
            )
            assert results == {0: (0, 1)}
        finally:
            doc.close()


# ---- _write_blocks_to_page 边界 -----------------------------------------


class TestWriteBlocksEdges:
    def test_write_blocks_empty_rect_skipped(self, tmp_path, caplog):
        """退化矩形（width<=0）应跳过并记警告（lines 988-996）。"""
        from vibeocr.backend.models.pdf_ocr_options import PdfGlobalSettings

        path = tmp_path / "empty_rect.pdf"
        _create_scanned_pdf(path)
        doc, _pdf_doc = PdfService.open_doc(str(path))
        try:
            # bbox 使 disp_rect.width <= 0（x1 <= x0）
            block = TextBlock(text="x", score=0.9, bbox=(200, 50, 200, 100), page_idx=0)
            with caplog.at_level(
                logging.WARNING, logger="vibeocr.backend.services.pdf_service"
            ):
                w, s = PdfService._write_blocks_to_page(
                    doc, 0, [block], 0, PdfGlobalSettings()
                )
            assert w == 0
            assert s == 1
            assert any("rect empty" in r.message for r in caplog.records)
        finally:
            doc.close()

    def test_write_blocks_with_polygon_vertical(self, tmp_path):
        """带竖排多边形的块走 insert_textbox 路径（覆盖 _poly_orientation vertical 分支）。"""
        from vibeocr.backend.models.pdf_ocr_options import PdfGlobalSettings

        path = tmp_path / "poly_v.pdf"
        _create_scanned_pdf(path)
        doc, _pdf_doc = PdfService.open_doc(str(path))
        try:
            # 竖排多边形：TL,TR,BR,BL，左边比顶边长（竖排）
            block = TextBlock(
                text="竖排文字",
                score=0.9,
                bbox=(50, 50, 80, 300),  # 窄高
                polygon=(50, 50, 80, 50, 80, 300, 50, 300),
                page_idx=0,
            )
            w, _s = PdfService._write_blocks_to_page(
                doc, 0, [block], 0, PdfGlobalSettings()
            )
            assert w == 1
        finally:
            doc.close()

    def test_write_blocks_natural_width_fallback_heuristic(self, tmp_path, monkeypatch):
        """无子集字体（china-s 回退）→ _natural_width 用启发式（lines 950-962）。"""
        from vibeocr.backend.models.pdf_ocr_options import PdfGlobalSettings

        path = tmp_path / "nofont.pdf"
        _create_scanned_pdf(path)
        doc, _pdf_doc = PdfService.open_doc(str(path))
        try:
            # 强制 _CJK_RESOLVER.resolve 返回 None → china-s 回退
            import vibeocr.backend.services.pdf_service as psm

            monkeypatch.setattr(psm._CJK_RESOLVER, "resolve", lambda chars: None)
            block = TextBlock(
                text="abc123",  # 拉丁/数字 → 走 0.5 启发式
                score=0.9,
                bbox=(50, 50, 300, 100),
                page_idx=0,
            )
            w, _s = PdfService._write_blocks_to_page(
                doc, 0, [block], 0, PdfGlobalSettings()
            )
            assert w == 1
        finally:
            doc.close()

    def test_write_blocks_insert_text_exception_falls_back(self, tmp_path, monkeypatch):
        """insert_text 抛异常 → 回退 insert_textbox（lines 1088-1094）。"""
        from vibeocr.backend.models.pdf_ocr_options import PdfGlobalSettings

        path = tmp_path / "ins_fail.pdf"
        _create_scanned_pdf(path)
        doc, _pdf_doc = PdfService.open_doc(str(path))
        try:
            block = TextBlock(
                text="hello",
                score=0.9,
                bbox=(50, 50, 300, 100),
                page_idx=0,
            )
            original_insert_text = fitz.Page.insert_text

            call = {"n": 0}

            def _fail(self, point, text, *args, **kwargs):
                call["n"] += 1
                # 第一次 insert_text（主路径）抛异常
                if call["n"] == 1:
                    raise RuntimeError("insert boom")
                return original_insert_text(self, point, text, *args, **kwargs)

            monkeypatch.setattr(fitz.Page, "insert_text", _fail)
            w, _s = PdfService._write_blocks_to_page(
                doc, 0, [block], 0, PdfGlobalSettings()
            )
            assert w == 1  # 回退成功
        finally:
            monkeypatch.setattr(fitz.Page, "insert_text", original_insert_text)
            doc.close()


# ---- delete_text_layers / _clear_page_layer_info ------------------------


class TestDeleteAndClearEdges:
    def test_delete_text_layers_no_words(self, tmp_path):
        """无文字页 delete → 不做 redact，直接清状态（lines 1241-1244）。"""
        path = tmp_path / "no_words.pdf"
        _create_scanned_pdf(path)
        doc, pdf_doc = PdfService.open_doc(str(path))
        try:
            init, rounds, residual = PdfService.delete_text_layers(doc, pdf_doc, 0)
            assert init == 0
            assert rounds == 0
            assert residual is False
        finally:
            doc.close()

    def test_delete_text_layers_with_words(self, tmp_path):
        """有文字页 delete → 多轮 redact 至清零（lines 1246-1268）。"""
        path = tmp_path / "has_words.pdf"
        doc = fitz.open()
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 72), "hello world", fontsize=12)
        doc.save(str(path))
        doc.close()
        doc, pdf_doc = PdfService.open_doc(str(path))
        try:
            init, rounds, residual = PdfService.delete_text_layers(doc, pdf_doc, 0)
            assert init >= 2  # "hello world" 至少 2 个词
            assert rounds >= 1
            assert residual is False
            # 文字应被清空
            assert doc[0].get_text().strip() == ""
        finally:
            doc.close()

    def test_clear_page_layer_info_out_of_range_noop(self, tmp_path):
        """_clear_page_layer_info 页码越界 → 静默返回（line 1273-1274）。"""
        doc, pdf_doc = PdfService.open_doc(str(_create_test_pdf(tmp_path / "t.pdf")))
        try:
            PdfService._clear_page_layer_info(pdf_doc, 999)
        finally:
            doc.close()


# ---- _denormalize_and_unrotate_polygon ----------------------------------


class TestPolygonUnrotate:
    def test_polygon_unrotate_all_angles(self, tmp_path):
        """_denormalize_and_unrotate_polygon 四种角度（lines 1358-1365）。"""
        rect = fitz.Rect(0, 0, 612, 792)
        poly = (100, 100, 200, 100, 200, 200, 100, 200)
        for angle in (0, 90, 180, 270):
            pts = PdfService._denormalize_and_unrotate_polygon(poly, angle, rect)
            assert len(pts) == 4

    def test_poly_orientation_single_char(self):
        """单字符 → horizontal（line 1382-1383）。"""
        assert PdfService._poly_orientation(None, "x") == "horizontal"

    def test_poly_orientation_no_polygon_unknown(self):
        """多字符无多边形 → unknown（line 1384-1385）。"""
        assert PdfService._poly_orientation(None, "abc") == "unknown"

    def test_poly_orientation_horizontal_vs_vertical(self):
        """有 4 点多边形时按顶边/左边长度判方向（lines 1388-1392）。"""
        # 横排：TL(0,0), TR(100,0), BR(100,10), BL(0,10) → 顶边 100 > 左边 10
        horiz = [
            fitz.Point(0, 0),
            fitz.Point(100, 0),
            fitz.Point(100, 10),
            fitz.Point(0, 10),
        ]
        assert PdfService._poly_orientation(horiz, "hello") == "horizontal"
        # 竖排：TL(0,0), TR(10,0), BR(10,100), BL(0,100) → 顶边 10 < 左边 100
        vert = [
            fitz.Point(0, 0),
            fitz.Point(10, 0),
            fitz.Point(10, 100),
            fitz.Point(0, 100),
        ]
        assert PdfService._poly_orientation(vert, "hello") == "vertical"


# ---- invalidate_thumbnails ----------------------------------------------


class TestInvalidateThumbnails:
    def test_invalidate_out_of_range_noop(self, tmp_path):
        """invalidate_thumbnails 越界索引静默跳过（branch 1424->1423）。"""
        doc, pdf_doc = PdfService.open_doc(str(_create_test_pdf(tmp_path / "t.pdf")))
        try:
            # 设一个 thumbnail，再 invalidate 越界+合法
            pdf_doc.pages[0].thumbnail = "x"
            PdfService.invalidate_thumbnails(
                pdf_document=pdf_doc, page_indices=[0, 999]
            )
            assert pdf_doc.pages[0].thumbnail is None
        finally:
            doc.close()


# ---- 剩余细粒度分支 ------------------------------------------------------


class TestRemainingBranches:
    def test_compress_failure_doc_close_exception(self, tmp_path, monkeypatch):
        """_compress_in_place 失败时 doc.close 也抛异常应被吞掉（lines 221-222）。"""
        path = tmp_path / "cf_close.pdf"
        _create_test_pdf(path, num_pages=1)
        doc, _ = PdfService.open_doc(str(path))
        # 让 close 抛异常
        monkeypatch.setattr(
            fitz.Document,
            "close",
            lambda self: (_ for _ in ()).throw(RuntimeError("close boom")),
        )
        original_save = fitz.Document.save

        def _fail(self, *args, **kwargs):
            if kwargs.get("garbage") == 3:
                raise RuntimeError("compress boom")
            return original_save(self, *args, **kwargs)

        monkeypatch.setattr(fitz.Document, "save", _fail)
        with pytest.raises(RuntimeError, match="compress boom"):
            PdfService._compress_in_place(doc, str(path), clean=False)

    def test_compress_failure_replace_oserror_fallback(self, tmp_path, monkeypatch):
        """_compress_in_place 失败 + Path.replace 抛 OSError → shutil.copy2 兜底（225-226）。"""
        path = tmp_path / "cf_replace.pdf"
        _create_test_pdf(path, num_pages=1)
        doc, _ = PdfService.open_doc(str(path))
        original_save = fitz.Document.save

        def _fail(self, *args, **kwargs):
            if kwargs.get("garbage") == 3:
                raise RuntimeError("compress boom")
            return original_save(self, *args, **kwargs)

        monkeypatch.setattr(fitz.Document, "save", _fail)
        # Path.replace 抛 OSError
        original_replace = Path.replace

        def _fail_replace(self, target):
            return (_ for _ in ()).throw(OSError("replace boom"))

        monkeypatch.setattr(Path, "replace", _fail_replace)
        with pytest.raises(RuntimeError, match="compress boom"):
            PdfService._compress_in_place(doc, str(path), clean=False)
        # copy2 兜底后原文件内容应保留（备份内容）
        assert path.exists()
        monkeypatch.setattr(Path, "replace", original_replace)

    def test_save_incremental_truncate_failure_logged(self, tmp_path, monkeypatch):
        """save_incremental: save 失败 + truncate 也失败 → 仅记日志返回 False（373-374）。"""
        path = tmp_path / "sf_tf.pdf"
        _create_test_pdf(path, num_pages=1)
        doc, _ = PdfService.open_doc(str(path))
        original_save = fitz.Document.save

        def _fail(self, *args, **kwargs):
            if kwargs.get("incremental"):
                raise RuntimeError("save boom")
            return original_save(self, *args, **kwargs)

        monkeypatch.setattr(fitz.Document, "save", _fail)
        # 让 _truncate_incremental 抛异常
        monkeypatch.setattr(
            PdfService,
            "_truncate_incremental",
            lambda p, m, s: (_ for _ in ()).throw(RuntimeError("truncate boom")),
        )
        assert PdfService.save_incremental(doc, str(path)) is False
        monkeypatch.setattr(fitz.Document, "save", original_save)
        doc.close()

    def test_page_rotation_helper(self, tmp_path):
        """page_rotation 返回页旋转角（line 502）。"""
        path = tmp_path / "rot.pdf"
        _create_test_pdf(path, num_pages=1)
        doc, pdf_doc = PdfService.open_doc(str(path))
        try:
            assert PdfService.page_rotation(doc, 0) == 0
            PdfService.rotate_pages(doc, pdf_doc, [0], 90)
            assert PdfService.page_rotation(doc, 0) == 90
        finally:
            doc.close()

    def test_write_blocks_empty_text_skipped(self, tmp_path):
        """空文本块应被跳过（line 968）。"""
        from vibeocr.backend.models.pdf_ocr_options import PdfGlobalSettings

        path = tmp_path / "empty_text.pdf"
        _create_scanned_pdf(path)
        doc, _pdf_doc = PdfService.open_doc(str(path))
        try:
            block = TextBlock(
                text="   ", score=0.9, bbox=(50, 50, 200, 100), page_idx=0
            )
            w, s = PdfService._write_blocks_to_page(
                doc, 0, [block], 0, PdfGlobalSettings()
            )
            assert w == 0
            assert s == 0  # 空文本不计 skip
        finally:
            doc.close()

    def test_write_blocks_natural_width_cjk_heuristic(self, tmp_path, monkeypatch):
        """china-s 回退 + CJK 字符 → units+=1.0 分支（line 959）。"""
        from vibeocr.backend.models.pdf_ocr_options import PdfGlobalSettings

        path = tmp_path / "cjk_h.pdf"
        _create_scanned_pdf(path)
        doc, _pdf_doc = PdfService.open_doc(str(path))
        try:
            import vibeocr.backend.services.pdf_service as psm

            monkeypatch.setattr(psm._CJK_RESOLVER, "resolve", lambda chars: None)
            block = TextBlock(
                text="中文",  # CJK → units+=1.0
                score=0.9,
                bbox=(50, 50, 300, 100),
                page_idx=0,
            )
            w, _s = PdfService._write_blocks_to_page(
                doc, 0, [block], 0, PdfGlobalSettings()
            )
            assert w == 1
        finally:
            doc.close()

    def test_write_blocks_retry_zero_loops(self, tmp_path, monkeypatch):
        """font_size_retry_count=0 → for range 空，走兜底（branch 1110->1129）。"""
        from vibeocr.backend.models.pdf_ocr_options import PdfGlobalSettings

        path = tmp_path / "zero_retry.pdf"
        _create_scanned_pdf(path)
        doc, _pdf_doc = PdfService.open_doc(str(path))
        try:
            # 竖排多边形 → is_horizontal=False → insert_textbox 路径
            block = TextBlock(
                text="vertical text here",
                score=0.9,
                bbox=(50, 50, 80, 300),
                polygon=(50, 50, 80, 50, 80, 300, 50, 300),
                page_idx=0,
            )
            settings = PdfGlobalSettings()
            settings.font_size_retry_count = 0
            w, _s = PdfService._write_blocks_to_page(doc, 0, [block], 0, settings)
            # retry=0 → 直接走兜底 insert_text
            assert w == 1
        finally:
            doc.close()

    def test_write_blocks_fallback_insert_text_failure(self, tmp_path, monkeypatch):
        """insert_textbox 失败 + 兜底 insert_text 也失败 → skipped（lines 1159-1168）。"""
        from vibeocr.backend.models.pdf_ocr_options import PdfGlobalSettings

        path = tmp_path / "both_fail.pdf"
        _create_scanned_pdf(path)
        doc, _pdf_doc = PdfService.open_doc(str(path))
        try:
            block = TextBlock(
                text="vertical text here",
                score=0.9,
                bbox=(50, 50, 80, 300),
                polygon=(50, 50, 80, 50, 80, 300, 50, 300),
                page_idx=0,
            )
            settings = PdfGlobalSettings()
            # 让 insert_textbox 永远返回负数（装不下）
            monkeypatch.setattr(fitz.Page, "insert_textbox", lambda *a, **k: -1.0)
            # 让 insert_text 也抛异常
            monkeypatch.setattr(
                fitz.Page,
                "insert_text",
                lambda *a, **k: (_ for _ in ()).throw(RuntimeError("ins boom")),
            )
            w, s = PdfService._write_blocks_to_page(doc, 0, [block], 0, settings)
            assert w == 0
            assert s == 1
        finally:
            doc.close()

    def test_delete_text_layers_residual_after_max_rounds(self, tmp_path, monkeypatch):
        """多轮 redact 仍有残留 → has_residual=True（branch 1248->1257, line 1259）。"""
        path = tmp_path / "residual.pdf"
        doc = fitz.open()
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 72), "persistent text", fontsize=12)
        doc.save(str(path))
        doc.close()
        doc, pdf_doc = PdfService.open_doc(str(path))
        try:
            # 让 apply_redactions 不真正删除（模拟残留）
            monkeypatch.setattr(fitz.Page, "apply_redactions", lambda *a, **k: None)
            init, rounds, residual = PdfService.delete_text_layers(doc, pdf_doc, 0)
            assert init >= 2
            # 走完 max rounds
            assert rounds == PdfService._DELETE_LAYER_MAX_ROUNDS
            assert residual is True
        finally:
            doc.close()

    def test_update_page_info_text_page(self, tmp_path):
        """update_page_info 对有文字页（非扫描件）的分支（branch 561->560）。"""
        path = tmp_path / "text_page.pdf"
        doc = fitz.open()
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 72), "hello", fontsize=12)
        doc.save(str(path))
        doc.close()
        doc, pdf_doc = PdfService.open_doc(str(path))
        try:
            PdfService.update_page_info(doc, pdf_doc, 0)
            assert pdf_doc.pages[0].has_text_layer is True
            assert pdf_doc.pages[0].is_scanned is False
        finally:
            doc.close()

    def test_rotate_pages_out_of_range_index_skipped(self, tmp_path):
        """rotate_pages 含越界索引应跳过该页（branch 561->560）。"""
        doc, pdf_doc = PdfService.open_doc(str(_create_test_pdf(tmp_path / "t.pdf")))
        try:
            PdfService.rotate_pages(doc, pdf_doc, [0, 999], 90)
            # page 0 仍被旋转
            assert pdf_doc.pages[0].rotation == 90
        finally:
            doc.close()

    def test_delete_pages_out_of_range_index_skipped(self, tmp_path):
        """delete_pages 含越界索引应跳过该页（branch 578->577）。"""
        doc, pdf_doc = PdfService.open_doc(str(_create_test_pdf(tmp_path / "t.pdf")))
        try:
            PdfService.delete_pages(doc, pdf_doc, [0, 999])
            assert pdf_doc.page_count == 1
        finally:
            doc.close()

    def test_detect_text_layers_skips_whitespace_only_line(self, tmp_path):
        """detect_text_layers 跳过仅含空白的 line（line 451）。

        需要页面有非空文本（通过 line 431 的 get_text 预检）+ 一条仅空白的 line。
        """
        path = tmp_path / "ws.pdf"
        doc = fitz.open()
        page = doc.new_page(width=612, height=792)
        # 一条正常文本（让 get_text("text") 非空，通过预检）
        page.insert_text((72, 72), "real content", fontsize=12)
        # 一条仅空白的文本（line 级 strip 后为空 → 跳过）
        page.insert_text((72, 150), "   ", fontsize=12)
        doc.save(str(path))
        doc.close()
        doc, _ = PdfService.open_doc(str(path))
        try:
            layers = PdfService.detect_text_layers(doc, 0)
            # 至少有 1 个 layer（来自 real content），空白行被跳过
            assert len(layers) >= 1
            # 无 layer 的 text_preview 是空白
            assert all(layer.text_preview.strip() for layer in layers)
        finally:
            doc.close()
