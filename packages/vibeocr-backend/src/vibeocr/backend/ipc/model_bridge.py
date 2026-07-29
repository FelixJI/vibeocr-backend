"""PdfDocumentMirror ↔ PdfDocument 模型桥接(主进程侧)。

后端子进程持有规范 PdfDocument(fitz 上下文),主进程只持 PdfDocumentMirror
(纯数据,可序列化)。但 PdfTab/ThumbnailModel 等 UI 代码大量直接读
PdfPageInfo 字段(rotation/has_text_layer/text_layers/ocr_text_blocks/rect...),
重写所有读取点成本过高。本桥接提供:

- mirror_to_doc: PdfDocumentMirror → PdfDocument(供 UI 现有代码读)
- apply_diff: 把 ModelDiff apply 到现有 PdfDocument(增量更新)
- page_mirror_to_info: 单页转换

UI 侧的 PdfDocument 现在是"从 mirror 重建的只读视图",字段语义不变。
所有 mutation 走 IPC,回包后通过 apply_diff 刷新视图。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from vibeocr.backend.models.ocr_result import TextBlock
from vibeocr.backend.models.pdf_document import (
    PdfDocument,
    PdfPageInfo,
    TextLayerInfo,
)

if TYPE_CHECKING:
    # schemas 仅用于类型注解（from __future__ annotations 使注解惰性求值），
    # 运行时不构造实例，放 TYPE_CHECKING 块避免无谓导入。
    from vibeocr.backend.ipc.schemas import (
        ModelDiff,
        PdfDocumentMirror,
        PdfPageInfoMirror,
        TextBlockMirror,
        TextLayerInfoMirror,
    )

logger = logging.getLogger(__name__)


def text_layer_mirror_to_info(m: TextLayerInfoMirror) -> TextLayerInfo:
    return TextLayerInfo(
        index=m.index,
        text_preview=m.text_preview,
        char_count=m.char_count,
        bbox=m.bbox,
        color_id=m.color_id,
    )


# 向后兼容别名(原私有名)
_text_layer_mirror_to_info = text_layer_mirror_to_info


def _text_block_mirror_to_info(b: TextBlockMirror) -> TextBlock:
    return TextBlock(
        text=b.text,
        score=b.score,
        bbox=b.bbox,
        polygon=b.polygon,
        page_idx=b.page_idx,
        is_manually_edited=b.is_manually_edited,
        label=b.label,
        order=b.order,
    )


def page_mirror_to_info(m: PdfPageInfoMirror) -> PdfPageInfo:
    """PdfPageInfoMirror → PdfPageInfo(主进程视图)。"""
    return PdfPageInfo(
        page_index=m.page_index,
        rotation=m.rotation,
        has_text_layer=m.has_text_layer,
        text_layers=[text_layer_mirror_to_info(t) for t in m.text_layers],
        is_scanned=m.is_scanned,
        rect=m.rect,
        ocr_text_blocks=[_text_block_mirror_to_info(b) for b in m.ocr_text_blocks],
        ocr_preproc_angle=m.ocr_preproc_angle,
        deskewed=m.deskewed,
    )


def mirror_to_doc(mirror: PdfDocumentMirror) -> PdfDocument:
    """PdfDocumentMirror → PdfDocument(完整重建)。"""
    return PdfDocument(
        file_path=mirror.file_path,
        pages=[page_mirror_to_info(p) for p in mirror.pages],
        is_modified=mirror.is_modified,
        has_structural_change=mirror.has_structural_change,
        render_dpi=mirror.render_dpi,
        thumbnail_dpi=mirror.thumbnail_dpi,
    )


def apply_diff(doc: PdfDocument, diff: ModelDiff) -> list[int]:
    """把 ModelDiff apply 到现有 PdfDocument,返回失效的缩略图页索引列表。

    - full_model 非空:整体替换 pages(结构变更/全量刷新)
    - replaced_pages:逐页替换
    - modified_flag / structural_flag:更新标志
    - invalidated_thumbnails:返回给调用方清缓存
    """
    invalidated = list(diff.invalidated_thumbnails)

    if diff.full_model is not None:
        new_doc = mirror_to_doc(diff.full_model)
        doc.pages = new_doc.pages
        doc.is_modified = new_doc.is_modified
        doc.has_structural_change = new_doc.has_structural_change
        # 结构变更默认失效所有缩略图
        if not invalidated:
            invalidated = list(range(len(doc.pages)))
    else:
        for page_mirror in diff.replaced_pages:
            idx = page_mirror.page_index
            if 0 <= idx < len(doc.pages):
                doc.pages[idx] = page_mirror_to_info(page_mirror)
            else:
                logger.warning("[model_bridge] replaced_pages 索引越界: %d", idx)

    if diff.modified_flag is not None:
        doc.is_modified = diff.modified_flag
    if diff.structural_flag is not None:
        doc.has_structural_change = diff.structural_flag

    return invalidated
