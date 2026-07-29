from pathlib import Path

import fitz

from vibeocr.backend.services.pdf_service import PdfService


def test_save_incremental_persists_and_keeps_doc_usable(tmp_path):
    pdf = tmp_path / "a.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 72), "hello")
    doc.save(str(pdf))
    doc.close()
    doc = fitz.open(str(pdf))
    # 再加一层文字
    doc[0].insert_text((50, 100), "world")

    ok = PdfService.save_incremental(doc, str(pdf))
    assert ok is True
    # doc 仍可用（不重开，不 close）
    assert doc.page_count == 1
    # 重开验证内容落盘
    doc.close()
    doc2 = fitz.open(str(pdf))
    text = doc2[0].get_text()
    assert "hello" in text and "world" in text
    doc2.close()


def test_save_incremental_returns_false_and_keeps_doc_usable_on_failure(
    tmp_path, monkeypatch
):
    """失败时 doc 保持可用（不 close），文件从备份回滚。调用方据此不写 sidecar。"""
    pdf = tmp_path / "a.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(str(pdf))
    doc.close()
    doc = fitz.open(str(pdf))
    doc[0].insert_text((50, 100), "world")  # 内存改动

    original_bytes = pdf.read_bytes()

    # 模拟 incremental 已追加部分字节后抛异常。
    def boom(self, path, *a, **kw):
        with Path(path).open("ab") as stream:
            stream.write(b"partial incremental bytes")
        raise RuntimeError("disk full")
    monkeypatch.setattr(fitz.Document, "save", boom)

    ok = PdfService.save_incremental(doc, str(pdf))
    assert ok is False
    assert pdf.read_bytes() == original_bytes
    assert not Path(str(pdf) + PdfService._INCREMENTAL_MARKER_SUFFIX).exists()
    # 关键：doc 仍可用（未 close），内存文字层保留，可继续后续操作
    assert doc.page_count == 1
    assert "world" in doc[0].get_text()  # 内存改动还在
    doc.close()
    # 文件从备份回滚（只剩最初的 new_page，无 world）
    monkeypatch.undo()
    doc2 = fitz.open(str(pdf))
    assert doc2.page_count == 1
    assert "world" not in doc2[0].get_text()
    doc2.close()


def test_save_incremental_does_not_copy_whole_pdf(tmp_path, monkeypatch):
    pdf = tmp_path / "fast.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(str(pdf))
    doc.close()
    doc = fitz.open(str(pdf))
    doc[0].insert_text((50, 100), "fast")

    def reject_copy(*args, **kwargs):
        raise AssertionError("incremental checkpoint must not copy the whole PDF")

    monkeypatch.setattr("vibeocr.backend.services.pdf_service.shutil.copy2", reject_copy)
    assert PdfService.save_incremental(doc, str(pdf)) is True
    doc.close()


def test_open_doc_recovers_interrupted_incremental_append(tmp_path):
    pdf = tmp_path / "recover.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(str(pdf))
    doc.close()
    original = pdf.read_bytes()
    marker = Path(str(pdf) + PdfService._INCREMENTAL_MARKER_SUFFIX)
    marker.write_text(str(len(original)), encoding="ascii")
    with pdf.open("ab") as stream:
        stream.write(b"interrupted append")

    recovered_doc, _ = PdfService.open_doc(str(pdf))
    recovered_doc.close()

    assert pdf.read_bytes() == original
    assert not marker.exists()


def test_compress_in_place_after_incremental_unavailable_does_not_crash(
    tmp_path, monkeypatch
):
    """回归 0xC0000409：增量保存恒 False 时，跨批累积的子集字体经
    _compress_in_place 全量重写不应触发 PyMuPDF 原生崩溃。

    场景复现：
    1. doc.can_save_incrementally() 恒返回 False（加密/表单/某些扫描件触发）
    2. 多批 add_text_layer_batch 每批写层后 save_incremental 都失败、不落盘，
       子集字体跨批累积在内存 doc 里
    3. 末尾 _compress_in_place 做全量 garbage=4 重写——本测试验证此步不崩、
       产出的文件可重新打开、字体收敛。

    本测试是 pdf_backend_process.add_text_layer_batch 回退到 _compress_in_place
    的回归保护。
    """
    # 强制增量保存恒不可用，模拟生产环境中的复现条件
    monkeypatch.setattr(
        fitz.Document, "can_save_incrementally", lambda self: False
    )

    pdf = tmp_path / "multi_batch.pdf"
    doc = fitz.open()
    for _ in range(3):
        doc.new_page(width=612, height=792)
    doc.save(str(pdf))
    doc.close()
    doc = fitz.open(str(pdf))

    # 模拟两批 OCR 各自插入文字层（跨批累积字体对象的触发条件）
    # 直接用 insert_text 写 CJK（与生产 _write_blocks_to_page 走子集字体路径一致）
    doc[0].insert_text((50, 100), "第一批中文", fontname="china-s")
    doc[1].insert_text((50, 100), "第二批文字", fontname="china-s")
    # 此时 save_incremental 会因 monkeypatch 返回 False（不落盘、不 close）
    assert PdfService.save_incremental(doc, str(pdf)) is False
    # 文件未增长（增量失败、未回退到全量）
    assert doc.page_count == 3  # doc 仍可用

    # 关键断言：_compress_in_place 全量重写不抛原生异常、产出可重开的文件
    new_doc = PdfService._compress_in_place(doc, str(pdf), clean=False)
    try:
        assert new_doc.page_count == 3
        text0 = new_doc[0].get_text("text")
        text1 = new_doc[1].get_text("text")
        assert "第一批" in text0
        assert "第二批" in text1
    finally:
        new_doc.close()

    # 文件能被独立打开（落盘完整）
    verify = fitz.open(str(pdf))
    try:
        assert verify.page_count == 3
    finally:
        verify.close()


def test_save_incremental_logs_diagnostics_when_unavailable(
    tmp_path, monkeypatch, caplog
):
    """增量保存不可用时，应记录诊断字段（is_encrypted/is_form_pdf/is_dirty），
    便于定位为何生产环境恒 False。"""
    import logging

    from vibeocr.backend.services import pdf_service as ps

    pdf = tmp_path / "diag.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(str(pdf))
    doc.close()
    doc = fitz.open(str(pdf))

    monkeypatch.setattr(
        fitz.Document, "can_save_incrementally", lambda self: False
    )

    with caplog.at_level(logging.ERROR, logger=ps.logger.name):
        ok = PdfService.save_incremental(doc, str(pdf))

    assert ok is False
    # 诊断日志应包含关键字段
    diag_msg = next(
        (r.message for r in caplog.records if "不支持增量保存" in r.message),
        None,
    )
    assert diag_msg is not None, "应记录'不支持增量保存'诊断"
    assert "is_encrypted=" in diag_msg
    assert "is_form_pdf=" in diag_msg
    assert "is_dirty=" in diag_msg
    doc.close()


def test_compress_in_place_closes_input_doc_no_double_close_needed(tmp_path):
    """回归 0xC0000409：_compress_in_place 内部已 close 输入 doc（释放 Windows
    文件锁的必要步骤），调用方不能再 close 一次——对已关闭的 fitz doc 调
    close 会触发原生 use-after-free。

    本测试锁定的契约：_compress_in_place 调用后，传入的 doc 已关闭（is_closed
    为 True），调用方应直接用返回的 new_doc，不再触碰旧 doc。
    """
    pdf = tmp_path / "contract.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(str(pdf))
    doc.close()
    doc = fitz.open(str(pdf))
    doc[0].insert_text((50, 72), "test")

    new_doc = PdfService._compress_in_place(doc, str(pdf), clean=False)
    try:
        # 契约：输入 doc 已被 _compress_in_place 关闭
        assert doc.is_closed, (
            "_compress_in_place 应关闭输入 doc（释放 Windows 文件锁）；"
            "若此断言失败，说明契约改变，需同步审查所有调用方是否 double-close"
        )
        assert new_doc.page_count == 1
    finally:
        # 正确的清理方式：只 close 返回的 new_doc，绝不 close 已关闭的旧 doc
        new_doc.close()

    # 文件落盘完整，能被独立打开
    verify = fitz.open(str(pdf))
    try:
        assert verify.page_count == 1
        assert "test" in verify[0].get_text("text")
    finally:
        verify.close()
