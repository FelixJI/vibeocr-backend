"""ipc.model_bridge 主进程侧模型桥接的边缘用例测试。

覆盖 mirror→doc 转换的字段映射与 apply_diff 的全量替换/单页替换/越界/标志更新分支。
"""

from __future__ import annotations

import logging

import pytest
from vibeocr.backend.ipc.model_bridge import (
    _text_layer_mirror_to_info,
    apply_diff,
    mirror_to_doc,
    page_mirror_to_info,
    text_layer_mirror_to_info,
)
from vibeocr.backend.ipc.schemas import (
    ModelDiff,
    PdfDocumentMirror,
    PdfPageInfoMirror,
    TextBlockMirror,
    TextLayerInfoMirror,
)


def _layer(idx: int = 0) -> TextLayerInfoMirror:
    return TextLayerInfoMirror(
        index=idx,
        text_preview=f"layer-{idx}",
        char_count=idx * 10,
        bbox=(0.0, 0.0, 1.0, 1.0),
        color_id=idx,
    )


def _block(text: str = "hi", page_idx: int = 0) -> TextBlockMirror:
    return TextBlockMirror(
        text=text, score=0.9, bbox=(1.0, 2.0, 3.0, 4.0), page_idx=page_idx
    )


def _page(idx: int = 0, *, rotation: int = 0) -> PdfPageInfoMirror:
    return PdfPageInfoMirror(
        page_index=idx,
        rotation=rotation,
        has_text_layer=True,
        text_layers=[_layer(0), _layer(1)],
        is_scanned=False,
        rect=(0.0, 0.0, 612.0, 792.0),
        ocr_text_blocks=[_block("a", idx), _block("b", idx)],
        ocr_preproc_angle=0,
        deskewed=False,
    )


class TestMirrorToInfoConversion:
    """mirror→info 单层转换。"""

    def test_text_layer_mirror_to_info_maps_fields(self):
        """TextLayerInfoMirror 字段一一映射到 TextLayerInfo。"""
        info = text_layer_mirror_to_info(_layer(3))

        assert info.index == 3
        assert info.text_preview == "layer-3"
        assert info.char_count == 30
        assert info.bbox == (0.0, 0.0, 1.0, 1.0)
        assert info.color_id == 3

    def test_text_layer_alias_is_same_function(self):
        """_text_layer_mirror_to_info 是向后兼容别名，指向同一函数。"""
        assert _text_layer_mirror_to_info is text_layer_mirror_to_info

    def test_page_mirror_to_info_maps_fields(self):
        """PdfPageInfoMirror 完整映射到 PdfPageInfo，含嵌套列表转换。"""
        info = page_mirror_to_info(_page(2, rotation=90))

        assert info.page_index == 2
        assert info.rotation == 90
        assert info.has_text_layer is True
        assert len(info.text_layers) == 2
        assert info.text_layers[1].char_count == 10
        assert len(info.ocr_text_blocks) == 2
        assert info.ocr_text_blocks[0].text == "a"
        assert info.rect == (0.0, 0.0, 612.0, 792.0)

    def test_page_mirror_to_info_empty_lists(self):
        """空 text_layers / ocr_text_blocks 列表正确转换。"""
        page = PdfPageInfoMirror(page_index=0)
        info = page_mirror_to_info(page)

        assert info.text_layers == []
        assert info.ocr_text_blocks == []

    def test_mirror_to_doc_builds_full_document(self):
        """PdfDocumentMirror 重建完整 PdfDocument。"""
        mirror = PdfDocumentMirror(
            file_path="/tmp/a.pdf",
            pages=[_page(0), _page(1)],
            is_modified=True,
            has_structural_change=True,
            render_dpi=200,
            thumbnail_dpi=72,
        )

        doc = mirror_to_doc(mirror)

        assert doc.file_path == "/tmp/a.pdf"
        assert len(doc.pages) == 2
        assert doc.is_modified is True
        assert doc.has_structural_change is True
        assert doc.render_dpi == 200
        assert doc.thumbnail_dpi == 72
        assert doc.pages[0].page_index == 0

    def test_mirror_to_doc_default_pages(self):
        """空 mirror 生成空 pages 的 doc。"""
        doc = mirror_to_doc(PdfDocumentMirror())

        assert doc.pages == []
        assert doc.is_modified is False


class TestApplyDiffFullModel:
    """apply_diff 的 full_model 整体替换分支。"""

    def test_full_model_replaces_pages(self):
        """full_model 非空时整体替换 pages，并同步标志位。"""
        doc = mirror_to_doc(PdfDocumentMirror(pages=[_page(0)]))

        diff = ModelDiff(
            full_model=PdfDocumentMirror(
                pages=[_page(0), _page(1)],
                is_modified=True,
                has_structural_change=True,
            )
        )

        invalidated = apply_diff(doc, diff)

        assert len(doc.pages) == 2
        assert doc.is_modified is True
        assert doc.has_structural_change is True
        # 无显式 invalidated 时，结构变更默认失效所有页
        assert invalidated == [0, 1]

    def test_full_model_with_explicit_invalidated(self):
        """full_model 路径但 diff 已带 invalidated_thumbnails 时透传，不覆盖。"""
        doc = mirror_to_doc(PdfDocumentMirror(pages=[_page(0), _page(1)]))

        diff = ModelDiff(
            full_model=PdfDocumentMirror(pages=[]), invalidated_thumbnails=[5]
        )

        invalidated = apply_diff(doc, diff)

        assert invalidated == [5]

    def test_modified_and_structural_flags_applied(self):
        """modified_flag / structural_flag 各自更新对应标志位。"""
        doc = mirror_to_doc(PdfDocumentMirror(pages=[_page(0)]))

        apply_diff(
            doc,
            ModelDiff(
                replaced_pages=[_page(0)],
                modified_flag=False,
                structural_flag=True,
            ),
        )

        assert doc.is_modified is False
        assert doc.has_structural_change is True


class TestApplyDiffReplacedPages:
    """apply_diff 的 replaced_pages 单页替换分支。"""

    def test_replaced_page_in_range(self):
        """命中的 replaced_page 替换对应页。"""
        doc = mirror_to_doc(PdfDocumentMirror(pages=[_page(0), _page(1)]))

        new_page = _page(1, rotation=180)
        invalidated = apply_diff(doc, ModelDiff(replaced_pages=[new_page]))

        assert doc.pages[1].rotation == 180
        assert doc.pages[0].rotation == 0
        assert invalidated == []

    def test_replaced_page_out_of_range_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ):
        """越界的 replaced_page 索引记录 warning 且不抛异常。"""
        doc = mirror_to_doc(PdfDocumentMirror(pages=[_page(0)]))
        out_of_range = _page(99)

        with caplog.at_level(
            logging.WARNING, logger="vibeocr.backend.ipc.model_bridge"
        ):
            invalidated = apply_diff(doc, ModelDiff(replaced_pages=[out_of_range]))

        assert invalidated == []
        assert any("越界" in rec.message for rec in caplog.records)
        assert doc.pages[0].page_index == 0  # 原页未被破坏

    def test_replaced_page_negative_index_ignored(
        self, caplog: pytest.LogCaptureFixture
    ):
        """负索引视为越界（0 <= idx 不成立），记录 warning。"""
        doc = mirror_to_doc(PdfDocumentMirror(pages=[_page(0)]))
        negative = PdfPageInfoMirror(page_index=-1)

        with caplog.at_level(
            logging.WARNING, logger="vibeocr.backend.ipc.model_bridge"
        ):
            apply_diff(doc, ModelDiff(replaced_pages=[negative]))

        assert any("越界" in rec.message for rec in caplog.records)


class TestApplyDiffInvalidatedPassthrough:
    """apply_diff 的 invalidated_thumbnails 透传。"""

    def test_invalidated_thumbnails_returned_verbatim(self):
        """无 full_model 时 invalidated_thumbnails 原样返回。"""
        doc = mirror_to_doc(PdfDocumentMirror(pages=[_page(0)]))

        invalidated = apply_diff(doc, ModelDiff(invalidated_thumbnails=[2, 4, 6]))

        assert invalidated == [2, 4, 6]
